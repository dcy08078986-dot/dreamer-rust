use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};
use crate::networks::rssm::{RSSM, RSSMState};
use crate::networks::encoder::Encoder;
use crate::networks::decoder::Decoder;

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

    pub fn predict_reward(&self, state: &RSSMState<B>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![state.deter.clone(), state.stoch.clone()], 1);
        self.reward.forward(x).squeeze(1)
    }

    pub fn train_step(
        &self,
        batch_obs: Tensor<B, 3>,
        batch_action: Tensor<B, 3>,
        batch_reward: Tensor<B, 2>,
        kl_low_threshold: f32,
        kl_high_threshold: f32,
        kl_weight_low: f32,
        kl_weight_high: f32,
        kl_scale_default: f32,
    ) -> Tensor<B, 1> {
        let [batch, seq_len] = [batch_obs.dims()[0], batch_obs.dims()[1]];
        let device = &batch_obs.device();

        let mut state = self.init_state(batch, device);

        let seq_len_tensor = Tensor::full([1], seq_len as f32, device);
        let _one_tensor: Tensor<B, 1> = Tensor::full([1], 1.0_f32, device);
        let half_tensor: Tensor<B, 1> = Tensor::full([1], 0.5_f32, device);
        let free_nats = Tensor::full([1, 1], 1.0_f32, device);
        let _zero_tensor: Tensor<B, 1> = Tensor::full([1], 0.0_f32, device);

        let mut obs_loss = Tensor::zeros([1], device);
        let mut reward_loss = Tensor::zeros([1], device);
        let mut kl_loss = Tensor::zeros([1], device);

        for t in 0..seq_len {
            let obs_t = batch_obs.clone().narrow(1, t, 1).squeeze(1);
            let act_t = batch_action.clone().narrow(1, t, 1).squeeze(1);
            let rew_t = batch_reward.clone().narrow(1, t, 1).squeeze(1);

            let prior_state = self.rssm.img_step(&state, act_t.clone());

            let obs_emb = self.encode(obs_t);
            let post_state = self.rssm.obs_step(&prior_state, obs_emb, act_t);

            let recon = self.reconstruct(&post_state);
            let target = batch_obs.clone().narrow(1, t, 1).squeeze(1);
            let diff = recon.sub(target);
            let squared = diff.clone().mul(diff);
            let mse = squared.sum_dim(1).mean();
            obs_loss = obs_loss.add(mse);

            let pred_rew = self.predict_reward(&post_state);
            let rew_diff = pred_rew.sub(rew_t);
            let rew_squared = rew_diff.clone().mul(rew_diff);
            let rew_loss = rew_squared.mean();
            reward_loss = reward_loss.add(rew_loss);

            let kl = analytic_kl(
                post_state.mean.clone(),
                post_state.std.clone(),
                prior_state.mean.clone(),
                prior_state.std.clone(),
            );

            let kl_adj = kl.sub(free_nats.clone()).clamp_min(0.0_f32);
            let kl_free = kl_adj.mean();
            kl_loss = kl_loss.add(kl_free);

            state = post_state;
        }

        let total_obs_loss = obs_loss.div(seq_len_tensor.clone());
        let total_reward_loss = reward_loss.div(seq_len_tensor.clone());
        let total_kl_loss = kl_loss.div(seq_len_tensor);

        // 自适应KL权重调整
        let kl_value = total_kl_loss.clone().mean().to_data();
        let kl_val = kl_value.as_slice::<f32>().unwrap()[0];
        let kl_weight = if kl_val < kl_low_threshold {
            kl_weight_low  // KL太小，增加权重鼓励探索
        } else if kl_val > kl_high_threshold {
            kl_weight_high  // KL太大，降低权重防止后验崩塌
        } else {
            kl_scale_default  // 正常范围使用默认权重
        };

        let kl_weight_tensor = Tensor::full([1], kl_weight, &total_kl_loss.device());

        total_obs_loss
            .add(total_reward_loss.mul(half_tensor))
            .add(total_kl_loss.mul(kl_weight_tensor))
    }
}

fn analytic_kl<B: Backend>(
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

    let kl = term1.add(term2).sub_scalar(1.0);
    kl.sum_dim(1).mul_scalar(0.5)
}
