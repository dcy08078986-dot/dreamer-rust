#![allow(dead_code)]

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor, Distribution::Normal, Int};
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

    pub fn sample(
        &self,
        state: &RSSMState<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let (mean, log_std) = self.forward(state);
        let std = log_std.exp().add_scalar(1e-4);

        let eps = Tensor::random(mean.shape(), Normal(0.0, 1.0), &mean.device());
        let raw_action = mean.clone() + eps * std.clone();

        let var = std.clone().powf_scalar(2.0);
        let diff = raw_action.clone() - mean;

        let term1 = diff.powf_scalar(2.0) / var.add_scalar(1e-8);
        let term2 = std.mul_scalar(2.0).log();
        let sum = (term1 + term2).add_scalar(1.837877f64);

        let mut log_prob = sum.sum_dim(1).mul_scalar(-0.5).squeeze(1);

        let action = raw_action.tanh();
        let tanh_grad = action.clone().powf_scalar(2.0).neg().add_scalar(1.0);
        log_prob = log_prob - tanh_grad.clamp_min(1e-8).log().sum_dim(1).squeeze(1);

        (action, log_prob)
    }
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

// ─── Lambda 回报计算 ───
///
/// Computes TD(λ) returns using the standard backward recursion.
/// Avoids `select_assign` due to burn 0.18 backend limitations.
pub fn compute_lambda_returns<B: Backend>(
    rewards: Tensor<B, 2>,
    values: Tensor<B, 2>,
    gamma: f64,
    lambda: f64,
) -> Tensor<B, 2> {
    let horizon = rewards.dims()[1];
    let device = &rewards.device();

    // rewards: [B, horizon], values: [B, horizon+1] (with bootstrapping)
    // Build returns backward, collecting [B] slices
    let mut ret_list: Vec<Tensor<B, 1>> = Vec::with_capacity(horizon + 1);

    // Bootstrap: ret[H] = values[H]
    let last_idx = Tensor::<B, 1, Int>::from_ints([(horizon as i64)], device);
    let ret_next = values.clone().select(1, last_idx).squeeze(1); // [B]
    ret_list.push(ret_next);

    // Backward recursion
    for t in (0..horizon).rev() {
        let idx_t = Tensor::<B, 1, Int>::from_ints([(t as i64)], device);
        let idx_next = Tensor::<B, 1, Int>::from_ints([((t + 1) as i64)], device);

        let r_t = rewards.clone().select(1, idx_t).squeeze(1); // [B]
        let v_next = values.clone().select(1, idx_next).squeeze(1); // [B]
        let ret_next = ret_list[0].clone(); // [B]

        // ret[t] = r_t + gamma * ((1-lambda) * v_next + lambda * ret_next)
        let one_minus_lambda = 1.0 - lambda;
        let term = v_next.mul_scalar(one_minus_lambda) + ret_next.mul_scalar(lambda);
        let ret_t = r_t + term.mul_scalar(gamma);
        ret_list.insert(0, ret_t);
    }

    // ret_list has horizon+1 elements (t=0..horizon), each [B]
    // Stack to [B, horizon+1], then narrow to [B, horizon]
    let full = Tensor::stack(ret_list, 1); // [B, horizon+1]
    full.narrow(1, 0, horizon) // [B, horizon]
}

// ─── 在想象轨迹上计算 Actor-Critic 损失 ───
pub fn actor_critic_loss<B: Backend>(
    actor: &Actor<B>,
    critic: &Critic<B>,
    imag_states: &[RSSMState<B>],
    imag_actions: &[Tensor<B, 2>],
    _imag_log_probs: &[Tensor<B, 1>],
    rewards: Tensor<B, 2>,
    gamma: f64,
    lambda: f64,
    _entropy_coef: f64,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let horizon = rewards.dims()[1];
    let device = &rewards.device();

    // Compute values via critic (keeps critic params in autodiff graph)
    let mut values = Vec::with_capacity(horizon + 1);
    for state in imag_states.iter() {
        values.push(critic.forward(state));
    }
    let v_boot = values.last().unwrap().clone();
    let values_tensor = Tensor::stack(values[..horizon].to_vec(), 1);
    let v_boot_col = v_boot.unsqueeze_dim::<2>(1);
    let values_full = Tensor::cat(vec![values_tensor.clone(), v_boot_col], 1);

    let returns = compute_lambda_returns(rewards, values_full, gamma, lambda);
    let advantages = (returns.clone() - values_tensor.clone()).detach();

    // Actor loss: recompute action distribution to keep actor params in graph
    let mut actor_loss = Tensor::zeros([1], device);
    for t in 0..horizon {
        let idx_t = Tensor::<B, 1, Int>::from_ints([(t as i64)], device);

        // Recompute action distribution from actor
        let (mean, log_std) = actor.forward(&imag_states[t]);
        let action = imag_actions[t].clone(); // stored action from rollout
        let std = log_std.exp().add_scalar(1e-4);

        // Compute log_prob of the stored action under current policy
        let var = std.clone().powf_scalar(2.0);
        let diff = action.clone() - mean;
        let term1 = diff.powf_scalar(2.0) / var.add_scalar(1e-8);
        let term2 = std.mul_scalar(2.0).log();
        let sum = (term1 + term2).add_scalar(1.837877f64);
        let mut log_prob: Tensor<B, 1> = sum.sum_dim(1).mul_scalar(-0.5).squeeze(1); // [B]

        // tanh correction for bounded actions
        let action_tanh = action.tanh();
        let tanh_grad = action_tanh.powf_scalar(2.0).neg().add_scalar(1.0);
        log_prob = log_prob - tanh_grad.clamp_min(1e-8).log().sum_dim(1).squeeze(1); // [B]

        let logp = log_prob.unsqueeze_dim::<2>(1); // [B, 1]
        let adv = advantages.clone().select(1, idx_t); // [B, 1]

        actor_loss = actor_loss + logp.neg().mul(adv).mean().reshape([1]);
    }
    actor_loss = actor_loss.div_scalar(horizon as f64);

    // Critic loss: MSE between returns and values
    let value_loss = (returns - values_tensor).powf_scalar(2.0).mean().reshape([1]);

    (actor_loss, value_loss)
}
