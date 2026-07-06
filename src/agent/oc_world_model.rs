//! Object-Centric World Model with Slot Attention

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::activation::{relu, sigmoid};
use crate::networks::slot_attention::{SlotAttention, SlotAttentionConfig};
use crate::networks::slot_rssm::{SlotRSSM, SlotStates};
use crate::networks::encoder::Encoder;
use crate::networks::broadcast_decoder::BroadcastDecoder;
use crate::networks::gaussian_decoder::GaussianDecoder;
use crate::networks::hybrid_decoder::HybridDecoder;
use crate::agent::world_model::kl_balanced_loss;
use crate::networks::rssm::RSSMState;
use crate::config::Config;
use crate::tools::{symlog, motion_saliency, motion_weights, weighted_mse, motion_region_mse};

const SLOT_FEATURE_DIM: usize = 64;
const DECODER_LATENT: usize = 64 + 2;
const DECODER_PROJ: usize = 64;

#[derive(Module, Debug)]
pub enum DecoderType<B: Backend> {
    Broadcast(BroadcastDecoder<B>),
    Gaussian(GaussianDecoder<B>),
    Hybrid(HybridDecoder<B>),
}

fn scalar<B: Backend>(t: &Tensor<B, 1>) -> f32 { t.clone().to_data().as_slice::<f32>().unwrap()[0] }

pub struct OCLosses<B: Backend> {
    pub total: Tensor<B, 1>, pub obs_loss: f32, pub reward_loss: f32,
    pub kl_loss: f32, pub pred_loss: f32, pub bt_loss: f32, pub ens_loss: f32,
    pub motion_mse: f32, pub post_std: f32, pub per_slot_kl: Vec<f32>,
}

#[derive(Module, Debug)]
pub struct OCWorldModel<B: Backend> {
    pub slot_attn: SlotAttention<B>, pub slot_rssm: SlotRSSM<B>, pub encoder: Encoder<B>,
    pub decoder: DecoderType<B>, pub slot_proj: Linear<B>,
    pub reward_hidden: Linear<B>, pub reward_out: Linear<B>,
    pub pred1: Linear<B>, pub pred2: Linear<B>,
    image_channels: usize, image_size: usize, num_slots: usize,
    slot_dim: usize, deter_size: usize, stoch_size: usize,
}

impl<B: Backend> OCWorldModel<B> {
    pub fn init(device: &B::Device, config: &Config, act_dim: usize) -> Self {
        let sa_cfg = SlotAttentionConfig { num_slots: config.num_slots, slot_dim: config.slot_dim, num_iterations: config.slot_iterations, feature_dim: SLOT_FEATURE_DIM };
        let state_dim = config.deter_size + config.stoch_size;
        let ctx = if config.use_slot_interaction { config.slot_ctx_dim } else { 0 };
        let ens = if config.use_ensemble_exploration { config.ensemble_size } else { 0 };
        let decoder = match config.decoder_type.as_str() {
            "gaussian" => DecoderType::Gaussian(GaussianDecoder::init(device, DECODER_LATENT, config.image_channels, config.image_size)),
            "hybrid" => DecoderType::Hybrid(HybridDecoder::init(device, DECODER_LATENT, config.image_channels, config.image_size)),
            _ => DecoderType::Broadcast(BroadcastDecoder::init(device, DECODER_LATENT, config.image_channels, config.image_size)),
        };
        Self { slot_attn: SlotAttention::init(device, &sa_cfg), slot_rssm: SlotRSSM::init(device, config.deter_size, config.stoch_size, act_dim, config.slot_dim, config.num_slots, ctx, ens), encoder: Encoder::init(device, config.embed_dim, config.image_channels, config.image_size), decoder, slot_proj: LinearConfig::new(state_dim, DECODER_PROJ).init(device), reward_hidden: LinearConfig::new(state_dim * config.num_slots, 128).init(device), reward_out: LinearConfig::new(128, 1).init(device), pred1: LinearConfig::new(state_dim, config.slot_dim).init(device), pred2: LinearConfig::new(config.slot_dim, config.slot_dim).init(device), image_channels: config.image_channels, image_size: config.image_size, num_slots: config.num_slots, slot_dim: config.slot_dim, deter_size: config.deter_size, stoch_size: config.stoch_size }
    }
    pub fn init_state(&self, batch: usize, device: &B::Device) -> SlotStates<B> { self.slot_rssm.init_state(batch, device) }
    pub fn encode_features(&self, obs_flat: Tensor<B, 2>) -> Tensor<B, 4> {
        let [b, _] = obs_flat.dims();
        self.encoder.forward_features_16x16(obs_flat.reshape([b, self.image_channels, self.image_size, self.image_size]))
    }
    pub fn encode_slots(&self, obs_flat: Tensor<B, 2>) -> Tensor<B, 3> { self.slot_attn.forward(self.encode_features(obs_flat)) }
    pub fn obs_step(&self, states: &SlotStates<B>, obs_flat: Tensor<B, 2>, action: Tensor<B, 2>) -> SlotStates<B> {
        self.slot_rssm.obs_step_all(states, self.encode_slots(obs_flat), action)
    }
    pub fn img_step(&self, states: &SlotStates<B>, action: Tensor<B, 2>) -> SlotStates<B> { self.slot_rssm.img_step_all(states, action) }

    pub fn decode_slots(&self, states: &SlotStates<B>) -> (Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>) {
        let b = states.batch; let k = self.num_slots; let c = self.image_channels; let hw = self.image_size * self.image_size;
        let lat = Tensor::cat(vec![states.deter.clone(), states.stoch.clone()], 1);
        let dec_in = Tensor::cat(vec![relu(self.slot_proj.forward(lat)), states.pos.clone()], 1);
        let out = match &self.decoder {
            DecoderType::Broadcast(dec) => dec.forward(dec_in),
            DecoderType::Gaussian(dec) => dec.forward(dec_in),
            DecoderType::Hybrid(dec) => dec.forward(dec_in),
        };
        let is_raw = matches!(&self.decoder, DecoderType::Broadcast(_));
        let rgb = if is_raw { sigmoid(out.clone().narrow(1, 0, c)) } else { out.clone().narrow(1, 0, c) };
        let mask_logits = out.narrow(1, c, 1);
        let rgb_4d = rgb.reshape([b, k, c, hw]);
        let logits_3d = mask_logits.reshape([b, k, hw]);
        let lmax = logits_3d.clone().max_dim(1).detach();
        let e = logits_3d.sub(lmax).exp();
        let masks = e.clone().div(e.sum_dim(1).add_scalar(1e-8));
        let comp = rgb_4d.clone().mul(masks.clone().reshape([b, k, 1, hw])).sum_dim(1);
        (comp.reshape([b, c * hw]), rgb_4d.reshape([b, k, c * hw]), masks)
    }

    pub fn predict_reward(&self, states: &SlotStates<B>) -> Tensor<B, 1> {
        let b = states.batch; let k = self.num_slots; let dd = self.deter_size; let ds = self.stoch_size;
        let deter = states.deter.clone().reshape([b, k * dd]); let stoch = states.stoch.clone().reshape([b, k * ds]);
        self.reward_out.forward(relu(self.reward_hidden.forward(Tensor::cat(vec![deter, stoch], 1)))).squeeze(1)
    }

    fn predict_slot_embed(&self, prior: &SlotStates<B>) -> Tensor<B, 2> {
        self.pred2.forward(relu(self.pred1.forward(Tensor::cat(vec![prior.deter.clone(), prior.stoch.clone()], 1))))
    }

    fn barlow_twins_loss(&self, embeds: Tensor<B, 2>, beta: f32) -> Tensor<B, 1> {
        let [n, d] = embeds.dims(); let device = embeds.device();
        if n < 4 { return Tensor::zeros([1], &device); }
        let mean = embeds.clone().mean_dim(0);
        let centered = embeds.sub(mean);
        let var = centered.clone().mul(centered.clone()).mean_dim(0);
        let zn = centered.div(var.sqrt().add_scalar(1e-5));
        let c = zn.clone().transpose().matmul(zn).div_scalar(n as f64);
        let mut eye_data = vec![0.0f32; d * d];
        for i in 0..d { eye_data[i * d + i] = 1.0; }
        let eye = Tensor::<B, 1>::from_floats(eye_data.as_slice(), &device).reshape([d, d]);
        let diff = c.sub(eye.clone());
        let on_diag = diff.clone().mul(eye.clone());
        let off_diag = diff.mul(eye.neg().add_scalar(1.0));
        let l_on = on_diag.clone().mul(on_diag).sum();
        let l_off = off_diag.clone().mul(off_diag).sum();
        l_on.add(l_off.mul_scalar(beta as f64)).div_scalar(d as f64)
    }

    pub fn train_step(&self, batch_obs: Tensor<B, 3>, batch_action: Tensor<B, 3>, batch_reward: Tensor<B, 2>, config: &Config, train_probe: bool, effective_kl_scale: f32) -> OCLosses<B> {
        let [batch, seq_len] = [batch_obs.dims()[0], batch_obs.dims()[1]]; let device = &batch_obs.device(); let k = self.num_slots;
        let obs0 = batch_obs.clone().narrow(1, 0, 1).squeeze(1);
        let act_dim = batch_action.dims()[2]; let zero_act = Tensor::zeros([batch, act_dim], device);
        let mut states = self.obs_step(&self.init_state(batch, device), obs0, zero_act);
        let mut obs_loss = Tensor::zeros([1], device); let mut reward_loss = Tensor::zeros([1], device);
        let mut kl_loss = Tensor::zeros([1], device); let mut pred_loss = Tensor::zeros([1], device);
        let mut ens_loss = Tensor::zeros([1], device); let mut motion_mse = Tensor::zeros([1], device);
        let mut per_slot_kl = vec![0.0f32; k]; let mut online_embeds: Vec<Tensor<B, 2>> = Vec::new();
        let run_probe = !config.use_latent_objective || train_probe;
        for t in 1..seq_len {
            let obs_t = batch_obs.clone().narrow(1, t, 1).squeeze(1);
            let obs_prev = batch_obs.clone().narrow(1, t - 1, 1).squeeze(1);
            let act_prev = batch_action.clone().narrow(1, t - 1, 1).squeeze(1);
            let rew_prev = batch_reward.clone().narrow(1, t - 1, 1).squeeze(1);
            let slot_embeds = self.encode_slots(obs_t.clone());
            let (prior_states, post_states) = self.slot_rssm.prior_posterior_step(&states, slot_embeds.clone(), act_prev);
            let embeds_bk = slot_embeds.reshape([batch * k, self.slot_dim]);
            if config.use_latent_objective {
                let z_hat = self.predict_slot_embed(&prior_states);
                let z_tgt = embeds_bk.clone().detach();
                let d = z_hat.sub(z_tgt);
                pred_loss = pred_loss.add(d.clone().mul(d).mean().reshape([1]));
                online_embeds.push(embeds_bk);
                if run_probe {
                    let (recon, _, _) = self.decode_slots(&post_states.detach());
                    let dd = recon.clone().sub(obs_t.clone());
                    obs_loss = obs_loss.add(dd.clone().mul(dd).mean().reshape([1]));
                    let sal = motion_saliency(obs_t.clone(), obs_prev, self.image_channels);
                    motion_mse = motion_mse.add(motion_region_mse(recon, obs_t.clone(), sal, 0.05));
                }
            } else {
                let (recon, _, _) = self.decode_slots(&post_states);
                let sal = motion_saliency(obs_t.clone(), obs_prev, self.image_channels);
                let mse = if config.use_motion_loss {
                    let w = motion_weights(sal.clone(), config.motion_lambda, config.motion_wmax);
                    weighted_mse(recon.clone(), obs_t.clone(), w)
                } else { let d = recon.clone().sub(obs_t.clone()); d.clone().mul(d).mean().reshape([1]) };
                obs_loss = obs_loss.add(mse);
                motion_mse = motion_mse.add(motion_region_mse(recon, obs_t.clone(), sal, 0.05));
            }
            let pred_rew = self.predict_reward(&post_states);
            let rd = pred_rew.sub(symlog(rew_prev));
            reward_loss = reward_loss.add(rd.clone().mul(rd).mean().reshape([1]));
            let post_flat = RSSMState { deter: post_states.deter.clone(), stoch: post_states.stoch.clone(), mean: post_states.mean.clone(), std: post_states.std.clone() };
            let prior_flat = RSSMState { deter: prior_states.deter.clone(), stoch: prior_states.stoch.clone(), mean: prior_states.mean.clone(), std: prior_states.std.clone() };
            kl_loss = kl_loss.add(kl_balanced_loss(&post_flat, &prior_flat, config.kl_balance_alpha, config.free_nats));
            for i in 0..k { let p = post_states.get_slot(i); let q = prior_states.get_slot(i); per_slot_kl[i] += scalar(&crate::agent::world_model::analytic_kl(p.mean, p.std, q.mean, q.std).mean().reshape([1])); }
            if config.use_ensemble_exploration { ens_loss = ens_loss.add(self.slot_rssm.ensemble_loss(post_states.deter.clone(), post_states.mean.clone())); }
            states = post_states;
        }
        let steps = (seq_len - 1).max(1) as f64;
        let obs_loss_f = obs_loss.div_scalar(steps); let reward_loss_f = reward_loss.div_scalar(steps);
        let kl_loss_f = kl_loss.div_scalar(steps); let pred_loss_f = pred_loss.div_scalar(steps);
        let ens_loss_f = ens_loss.div_scalar(steps); let motion_mse_f = if run_probe { motion_mse.div_scalar(steps) } else { motion_mse };
        for v in per_slot_kl.iter_mut() { *v /= steps as f32; }
        let bt_loss = if config.use_latent_objective && !online_embeds.is_empty() { self.barlow_twins_loss(Tensor::cat(online_embeds, 0), config.bt_beta) } else { Tensor::zeros([1], device) };
        let mut total = reward_loss_f.clone().mul_scalar(config.reward_scale as f64).add(kl_loss_f.clone().mul_scalar(effective_kl_scale as f64));
        if config.use_latent_objective { total = total.add(pred_loss_f.clone().mul_scalar(config.latent_pred_scale as f64)).add(bt_loss.clone().mul_scalar(config.lambda_bt as f64)); if run_probe { total = total.add(obs_loss_f.clone()); } }
        else { total = total.add(obs_loss_f.clone()); }
        if config.use_ensemble_exploration { total = total.add(ens_loss_f.clone()); }
        OCLosses { total, obs_loss: scalar(&obs_loss_f), reward_loss: scalar(&reward_loss_f), kl_loss: scalar(&kl_loss_f), pred_loss: scalar(&pred_loss_f), bt_loss: scalar(&bt_loss), ens_loss: scalar(&ens_loss_f), motion_mse: scalar(&motion_mse_f), post_std: scalar(&states.std.mean().reshape([1])), per_slot_kl }
    }
}
