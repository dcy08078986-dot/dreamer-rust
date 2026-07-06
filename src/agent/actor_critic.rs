#![allow(dead_code)]

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor, Distribution::Normal};
use burn::tensor::activation::relu;
use crate::networks::rssm::RSSMState;

// ─── Actor（策略网络）───
#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    pub hidden: Linear<B>,
    pub mean: Linear<B>,
    pub log_std: Linear<B>,
}

impl<B: Backend> Actor<B> {
    pub fn init(
        device: &B::Device,
        deter_size: usize,
        stoch_size: usize,
        action_dim: usize,
    ) -> Self {
        let hidden_size = 256;
        Self {
            hidden: LinearConfig::new(deter_size + stoch_size, hidden_size).init(device),
            mean: LinearConfig::new(hidden_size, action_dim).init(device),
            log_std: LinearConfig::new(hidden_size, action_dim).init(device),
        }
    }

    pub fn forward(&self, state: &RSSMState<B>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = Tensor::cat(vec![state.deter.clone(), state.stoch.clone()], 1);
        let h = relu(self.hidden.forward(x));
        let mean = self.mean.forward(h.clone());
        let log_std = self.log_std.forward(h).clamp(-2.0, 2.0);
        (mean, log_std)
    }

    /// Sample a tanh-squashed Gaussian action.
    /// Returns (squashed action, RAW pre-tanh action, log_prob of the squashed action).
    pub fn sample_with_raw(
        &self,
        state: &RSSMState<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1>) {
        let (mean, log_std) = self.forward(state);
        let std = log_std.clone().exp().add_scalar(1e-4);
        let eps = Tensor::random(mean.shape(), Normal(0.0, 1.0), &mean.device());
        let raw = mean.clone() + eps * std;
        let action = raw.clone().tanh();
        let log_prob = tanh_gaussian_log_prob(mean, log_std, raw.clone());
        (action, raw, log_prob)
    }

    pub fn sample(&self, state: &RSSMState<B>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let (action, _raw, log_prob) = self.sample_with_raw(state);
        (action, log_prob)
    }
}

/// Log-density of a tanh-squashed Gaussian, evaluated at the RAW (pre-tanh) action:
///   log N(raw; mean, std) - sum log(1 - tanh(raw)^2)
/// The Gaussian density must be evaluated at the raw sample — evaluating it at the
/// squashed action (as the previous implementation did) is not a valid log-prob.
pub fn tanh_gaussian_log_prob<B: Backend>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
    raw_action: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let std = log_std.clone().exp().add_scalar(1e-4);
    let var = std.clone().mul(std);
    let diff = raw_action.clone() - mean;
    // -0.5 * [ (x-mu)^2/var + 2 log std + log(2 pi) ]
    let quad = diff.clone().mul(diff).div(var.add_scalar(1e-8));
    let sum = (quad + log_std.mul_scalar(2.0)).add_scalar(1.837877f64); // ln(2*pi)
    let gauss: Tensor<B, 1> = sum.sum_dim(1).mul_scalar(-0.5).squeeze(1);

    let squashed = raw_action.tanh();
    let jac = squashed.clone().mul(squashed).neg().add_scalar(1.0); // 1 - tanh^2
    gauss - jac.clamp_min(1e-6).log().sum_dim(1).squeeze(1)
}

// ─── Critic（价值网络）───
#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    pub hidden: Linear<B>,
    pub value: Linear<B>,
}

impl<B: Backend> Critic<B> {
    pub fn init(device: &B::Device, deter_size: usize, stoch_size: usize) -> Self {
        let hidden_size = 256;
        Self {
            hidden: LinearConfig::new(deter_size + stoch_size, hidden_size).init(device),
            value: LinearConfig::new(hidden_size, 1).init(device),
        }
    }

    pub fn forward(&self, state: &RSSMState<B>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![state.deter.clone(), state.stoch.clone()], 1);
        let h = relu(self.hidden.forward(x));
        self.value.forward(h).squeeze(1)
    }
}

// ─── Lambda 回报（list 形式，梯度路径不经过 stack/select）───
///
/// TD(λ) returns via backward recursion over [B]-shaped tensors.
/// rewards: H entries, values: H+1 entries (bootstrap at the end).
/// All inputs should be detached; the output is a detached target.
pub fn lambda_returns_list<B: Backend>(
    rewards: &[Tensor<B, 1>],
    values: &[Tensor<B, 1>],
    gamma: f64,
    lambda: f64,
) -> Vec<Tensor<B, 1>> {
    let horizon = rewards.len();
    assert_eq!(values.len(), horizon + 1, "values must include the bootstrap");

    let mut ret_next = values[horizon].clone();
    let mut returns: Vec<Tensor<B, 1>> = Vec::with_capacity(horizon);
    for t in (0..horizon).rev() {
        let v_next = values[t + 1].clone();
        // ret[t] = r_t + gamma * ((1-lambda) * v_next + lambda * ret[t+1])
        let mix = v_next.mul_scalar(1.0 - lambda) + ret_next.clone().mul_scalar(lambda);
        let ret_t = rewards[t].clone() + mix.mul_scalar(gamma);
        returns.push(ret_t.clone());
        ret_next = ret_t;
    }
    returns.reverse();
    returns
}

/// Actor and critic losses over an imagined trajectory (REINFORCE + TD(λ)).
///
/// imag_states: H+1 states, detached from the world-model graph.
/// imag_raw_actions: H RAW (pre-tanh) actions stored during the rollout.
/// rewards: H predicted rewards in symlog space, detached, [B] each.
///
/// The critic regresses detached λ-returns; the actor maximizes
/// log π(a|s) · advantage + entropy_coef · H[π].
pub fn imagination_losses<B: Backend>(
    actor: &Actor<B>,
    critic: &Critic<B>,
    imag_states: &[RSSMState<B>],
    imag_raw_actions: &[Tensor<B, 2>],
    rewards: &[Tensor<B, 1>],
    gamma: f64,
    lambda: f64,
    entropy_coef: f64,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let horizon = imag_raw_actions.len();
    let device = rewards[0].device();

    // Critic values (in graph) and their detached copies for target computation.
    let values: Vec<Tensor<B, 1>> = imag_states.iter().map(|s| critic.forward(s)).collect();
    let values_detached: Vec<Tensor<B, 1>> = values.iter().map(|v| v.clone().detach()).collect();
    let returns = lambda_returns_list(rewards, &values_detached, gamma, lambda);

    let mut actor_loss = Tensor::zeros([1], &device);
    let mut critic_loss = Tensor::zeros([1], &device);
    for t in 0..horizon {
        let (mean, log_std) = actor.forward(&imag_states[t]);
        let log_prob = tanh_gaussian_log_prob(mean, log_std.clone(), imag_raw_actions[t].clone());

        let adv = (returns[t].clone() - values_detached[t].clone()).detach();
        // Gaussian entropy up to constants: sum(log_std); enough for a bonus gradient.
        let entropy: Tensor<B, 1> = log_std.sum_dim(1).squeeze(1);
        actor_loss = actor_loss
            + log_prob.neg().mul(adv).mean().reshape([1])
            - entropy.mean().mul_scalar(entropy_coef).reshape([1]);

        let vd = returns[t].clone().detach() - values[t].clone();
        critic_loss = critic_loss + vd.clone().mul(vd).mean().reshape([1]);
    }
    (
        actor_loss.div_scalar(horizon as f64),
        critic_loss.div_scalar(horizon as f64),
    )
}
