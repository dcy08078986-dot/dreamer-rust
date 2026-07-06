use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};
use crate::networks::rssm::{RSSM, RSSMState};
use crate::networks::encoder::Encoder;
use crate::networks::decoder::Decoder;
use crate::config::Config;
use crate::tools::{symlog, motion_saliency, motion_weights, weighted_mse, motion_region_mse};

/// Diagnostics returned by a world-model training step.
pub struct WMDiagnostics<B: Backend> {
    pub total: Tensor<B, 1>,
    pub obs_loss: Tensor<B, 1>,
    pub reward_loss: Tensor<B, 1>,
    pub kl_loss: Tensor<B, 1>,
    /// MSE restricted to moving pixels (the ball/trail) — the metric that matters.
    pub motion_mse: Tensor<B, 1>,
}

#[derive(Module, Debug)]
pub struct WorldModel<B: Backend> {
    pub rssm: RSSM<B>,
    pub encoder: Encoder<B>,
    pub decoder: Decoder<B>,
    pub reward: Linear<B>,
    image_channels: usize,
    image_size: usize,
}

impl<B: Backend> WorldModel<B> {
    pub fn init(
        device: &B::Device,
        deter: usize,
        stoch: usize,
        action_dim: usize,
        embed_dim: usize,
        image_channels: usize,
        image_size: usize,
    ) -> Self {
        Self {
            rssm: RSSM::init(device, deter, stoch, action_dim, embed_dim),
            encoder: Encoder::init(device, embed_dim, image_channels, image_size),
            decoder: Decoder::init(device, deter + stoch, image_channels, image_size),
            reward: LinearConfig::new(deter + stoch, 1).init(device),
            image_channels,
            image_size,
        }
    }

    pub fn init_state(&self, batch: usize, device: &B::Device) -> RSSMState<B> {
        self.rssm.init_state(batch, device)
    }

    /// encode: [B, C*H*W] → reshape → CNN → [B, embed_dim]
    pub fn encode(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, _] = obs.dims();
        let obs_4d = obs.reshape([b, self.image_channels, self.image_size, self.image_size]);
        self.encoder.forward(obs_4d)
    }

    pub fn obs_step(&self, state: &RSSMState<B>, obs: Tensor<B, 2>, action: Tensor<B, 2>) -> RSSMState<B> {
        let emb = self.encode(obs);
        self.rssm.obs_step(state, emb, action)
    }

    pub fn img_step(&self, state: &RSSMState<B>, action: Tensor<B, 2>) -> RSSMState<B> {
        self.rssm.img_step(state, action)
    }

    /// reconstruct: state → decoder CNN → flatten to [B, C*H*W]
    pub fn reconstruct(&self, state: &RSSMState<B>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![state.deter.clone(), state.stoch.clone()], 1);
        let recon_4d = self.decoder.forward(x);
        let [b, c, h, w] = recon_4d.dims();
        recon_4d.reshape([b, c * h * w])
    }

    /// Predicts reward in symlog space (the head is trained on symlog targets).
    pub fn predict_reward(&self, state: &RSSMState<B>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![state.deter.clone(), state.stoch.clone()], 1);
        self.reward.forward(x).squeeze(1)
    }

    /// World-model training step over a sequence window.
    ///
    /// Data alignment: obs[t] is the observation AFTER action[t-1] was taken, and
    /// reward[t-1] is the reward received on arriving at obs[t]. The state at obs[0]
    /// is initialized with a zero previous action (mid-episode approximation).
    pub fn train_step(
        &self,
        batch_obs: Tensor<B, 3>,
        batch_action: Tensor<B, 3>,
        batch_reward: Tensor<B, 2>,
        config: &Config,
    ) -> WMDiagnostics<B> {
        let [_batch, seq_len] = [batch_obs.dims()[0], batch_obs.dims()[1]];
        let device = &batch_obs.device();

        // Burn-in: posterior at t=0 with zero previous action.
        let obs0 = batch_obs.clone().narrow(1, 0, 1).squeeze(1);
        let act_dim = batch_action.dims()[2];
        let zero_act = Tensor::zeros([batch_obs.dims()[0], act_dim], device);
        let mut state = self.obs_step(&self.init_state(batch_obs.dims()[0], device), obs0, zero_act);

        let mut obs_loss = Tensor::zeros([1], device);
        let mut reward_loss = Tensor::zeros([1], device);
        let mut kl_loss = Tensor::zeros([1], device);
        let mut motion_mse = Tensor::zeros([1], device);

        for t in 1..seq_len {
            let obs_t = batch_obs.clone().narrow(1, t, 1).squeeze(1);
            let obs_prev = batch_obs.clone().narrow(1, t - 1, 1).squeeze(1);
            let act_prev = batch_action.clone().narrow(1, t - 1, 1).squeeze(1);
            let rew_prev = batch_reward.clone().narrow(1, t - 1, 1).squeeze(1);

            // Prior and posterior share the SAME h_t.
            let deter = self.rssm.get_deter(&state, act_prev);
            let prior_state = self.rssm.prior_state(deter.clone());
            let obs_emb = self.encode(obs_t.clone());
            let post_state = self.rssm.post_state(deter, obs_emb);

            let recon = self.reconstruct(&post_state);
            let saliency = motion_saliency(obs_t.clone(), obs_prev, self.image_channels);
            let mse = if config.use_motion_loss {
                let w = motion_weights(saliency.clone(), config.motion_lambda, config.motion_wmax);
                weighted_mse(recon.clone(), obs_t.clone(), w)
            } else {
                let d = recon.clone().sub(obs_t.clone());
                d.clone().mul(d).mean().reshape([1])
            };
            obs_loss = obs_loss.add(mse);
            motion_mse = motion_mse.add(motion_region_mse(recon, obs_t, saliency, 0.05));

            // Reward head trained on symlog targets (env rewards reach ~10/step).
            let pred_rew = self.predict_reward(&post_state);
            let rew_target = symlog(rew_prev);
            let rew_diff = pred_rew.sub(rew_target);
            reward_loss = reward_loss.add(rew_diff.clone().mul(rew_diff).mean().reshape([1]));

            // KL balancing (DreamerV3): dynamics term pulls the prior toward the
            // (frozen) posterior, representation term regularizes the posterior.
            kl_loss = kl_loss.add(kl_balanced_loss(
                &post_state, &prior_state,
                config.kl_balance_alpha, config.free_nats,
            ));

            state = post_state;
        }

        let steps = (seq_len - 1).max(1) as f64;
        let total_obs_loss = obs_loss.div_scalar(steps);
        let total_reward_loss = reward_loss.div_scalar(steps);
        let total_kl_loss = kl_loss.div_scalar(steps);
        let total_motion_mse = motion_mse.div_scalar(steps);

        let total = total_obs_loss.clone()
            .add(total_reward_loss.clone().mul_scalar(config.reward_scale as f64))
            .add(total_kl_loss.clone().mul_scalar(config.kl_scale as f64));

        WMDiagnostics {
            total,
            obs_loss: total_obs_loss,
            reward_loss: total_reward_loss,
            kl_loss: total_kl_loss,
            motion_mse: total_motion_mse,
        }
    }
}

/// Element-wise (per-dimension) Gaussian KL: returns [B, D], NOT summed over dims.
pub fn analytic_kl_per_dim<B: Backend>(
    mean1: Tensor<B, 2>,
    std1: Tensor<B, 2>,
    mean2: Tensor<B, 2>,
    std2: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let var1 = std1.clone().mul(std1.clone());
    let var2 = std2.clone().mul(std2.clone());

    let log_var1 = std1.log().mul_scalar(2.0);
    let log_var2 = std2.log().mul_scalar(2.0);

    let diff = mean1.sub(mean2);
    let diff_sq = diff.clone().mul(diff);

    let term1 = log_var2.sub(log_var1);
    let denominator = var2.add_scalar(1e-8);
    let term2 = var1.add(diff_sq).div(denominator);

    term1.add(term2).sub_scalar(1.0).mul_scalar(0.5)
}

/// Summed Gaussian KL per sample: [B, 1].
pub fn analytic_kl<B: Backend>(
    mean1: Tensor<B, 2>,
    std1: Tensor<B, 2>,
    mean2: Tensor<B, 2>,
    std2: Tensor<B, 2>,
) -> Tensor<B, 2> {
    analytic_kl_per_dim(mean1, std1, mean2, std2).sum_dim(1)
}

/// DreamerV3-style balanced KL with per-dimension free bits.
///
/// L = alpha * KL(sg(post) || prior) + (1-alpha) * KL(post || sg(prior))
///
/// Each term gets per-dim free bits (free_nats spread evenly over dimensions) so the
/// gradient vanishes once a dimension's KL is below its budget — this replaces the
/// old adaptive-KL rule, which INCREASED KL pressure when KL was already collapsed.
pub fn kl_balanced_loss<B: Backend>(
    post: &RSSMState<B>,
    prior: &RSSMState<B>,
    alpha: f32,
    free_nats: f32,
) -> Tensor<B, 1> {
    let dims = post.mean.dims()[1] as f32;
    let free_per_dim = free_nats / dims.max(1.0);

    // Dynamics loss: train the prior to match the (frozen) posterior.
    let kl_dyn = analytic_kl_per_dim(
        post.mean.clone().detach(), post.std.clone().detach(),
        prior.mean.clone(), prior.std.clone(),
    );
    // Representation loss: keep the posterior close to the (frozen) prior.
    let kl_rep = analytic_kl_per_dim(
        post.mean.clone(), post.std.clone(),
        prior.mean.clone().detach(), prior.std.clone().detach(),
    );

    let dyn_term = kl_dyn.clamp_min(free_per_dim).sum_dim(1).mean().reshape([1]);
    let rep_term = kl_rep.clamp_min(free_per_dim).sum_dim(1).mean().reshape([1]);

    dyn_term.mul_scalar(alpha as f64).add(rep_term.mul_scalar((1.0 - alpha) as f64))
}
