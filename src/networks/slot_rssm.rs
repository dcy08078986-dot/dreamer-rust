//! Per-Slot RSSM: K RSSM instances with shared parameters, plus an optional
//! cross-slot interaction module.
//!
//! Each object slot has its own latent state (deter, stoch). Before each
//! transition, a single-head attention over the K slot states produces an
//! interaction context per slot, which enters the GRU alongside the action —
//! so contacts/bounces can be modeled as slot-slot interaction (SlotSSM,
//! NeurIPS 2024; Interactive World Models, NeurIPS 2025). RSSM parameters are
//! shared across slots for parameter efficiency and consistent dynamics.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softplus};
use burn::tensor::{backend::Backend, Tensor};
use crate::networks::rssm::{RSSM, RSSMState};

/// K individual RSSM states reshaped into a single batched struct.
/// Flattening is BATCH-MAJOR: row b*K + k holds slot k of batch element b.
#[derive(Clone, Debug)]
pub struct SlotStates<B: Backend> {
    /// All slot states flattened: [B*K, ...] for each field
    pub deter: Tensor<B, 2>, // [B*K, deter_dim]
    pub stoch: Tensor<B, 2>, // [B*K, stoch_dim]
    pub mean:  Tensor<B, 2>,
    pub std:   Tensor<B, 2>,
    /// Slot position [B*K, 2] in [0,1]: the attention centroid on obs steps, the
    /// pos_head prediction on imagination steps. Bypasses the stoch bottleneck
    /// straight to the decoder.
    pub pos:   Tensor<B, 2>,
    pub batch: usize,
    pub num_slots: usize,
}

impl<B: Backend> SlotStates<B> {
    pub fn init(batch: usize, num_slots: usize, rssm: &RSSM<B>, device: &B::Device) -> Self {
        let bk = batch * num_slots;
        let s = rssm.init_state(bk, device);
        Self {
            deter: s.deter,
            stoch: s.stoch,
            mean: s.mean,
            std: s.std,
            pos: Tensor::zeros([bk, 2], device).add_scalar(0.5),
            batch,
            num_slots,
        }
    }

    /// Get state for a single slot
    pub fn get_slot(&self, slot_idx: usize) -> RSSMState<B> {
        let b = self.batch;
        let k = self.num_slots;
        let dd = self.deter.dims()[1];
        let ds = self.stoch.dims()[1];

        let deter = self.deter.clone().reshape([b, k, dd])
            .narrow(1, slot_idx, 1).squeeze(1);
        let stoch = self.stoch.clone().reshape([b, k, ds])
            .narrow(1, slot_idx, 1).squeeze(1);
        let mean = self.mean.clone().reshape([b, k, ds])
            .narrow(1, slot_idx, 1).squeeze(1);
        let std = self.std.clone().reshape([b, k, ds])
            .narrow(1, slot_idx, 1).squeeze(1);

        RSSMState { deter, stoch, mean, std }
    }

    /// Detached copy (for imagination rollouts / probe decoding).
    pub fn detach(&self) -> Self {
        Self {
            deter: self.deter.clone().detach(),
            stoch: self.stoch.clone().detach(),
            mean: self.mean.clone().detach(),
            std: self.std.clone().detach(),
            pos: self.pos.clone().detach(),
            batch: self.batch,
            num_slots: self.num_slots,
        }
    }
}

/// Broadcast a per-batch action [B, A] to all K slots, BATCH-MAJOR: [B*K, A].
/// (The previous cat-along-dim-0 version produced slot-major ordering k*B+b,
/// which paired slot states with the WRONG batch element's action for B > 1.)
pub fn broadcast_action<B: Backend>(action: Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
    let [b, a] = action.dims();
    let a3: Tensor<B, 3> = action.unsqueeze_dim(1); // [B, 1, A]
    let copies: Vec<Tensor<B, 3>> = (0..k).map(|_| a3.clone()).collect();
    Tensor::cat(copies, 1).reshape([b * k, a])
}

/// Per-slot RSSM dynamics wrapper
#[derive(Module, Debug)]
pub struct SlotRSSM<B: Backend> {
    pub rssm: RSSM<B>,

    /// Cross-slot interaction attention (single head). Active iff ctx_dim > 0.
    pub inter_q: Linear<B>,
    pub inter_k: Linear<B>,
    pub inter_v: Linear<B>,

    /// Ensemble of extra prior-mean/std heads for epistemic disagreement
    /// (Proposal D). Trained on detached inputs; empty when disabled.
    pub ensemble: Vec<Linear<B>>,

    /// Position dynamics head: (deter, prev pos) → next pos delta. Carries the
    /// centroid bypass through imagination steps, where no attention centroid
    /// exists. Predicting the DELTA keeps the head near-identity at init.
    pub pos_head1: Linear<B>,
    pub pos_head2: Linear<B>,

    pub num_slots: usize,
    ctx_dim: usize,
    stoch_size: usize,
}

impl<B: Backend> SlotRSSM<B> {
    pub fn init(
        device: &B::Device,
        deter: usize,
        stoch: usize,
        action_dim: usize,
        embed_dim: usize,
        num_slots: usize,
        ctx_dim: usize,
        ensemble_size: usize,
    ) -> Self {
        let state_dim = deter + stoch;
        let qkv_in = if ctx_dim > 0 { state_dim } else { 1 };
        let qkv_out = ctx_dim.max(1);
        Self {
            rssm: RSSM::init_with_ctx(device, deter, stoch, action_dim, embed_dim, ctx_dim),
            inter_q: LinearConfig::new(qkv_in, qkv_out).init(device),
            inter_k: LinearConfig::new(qkv_in, qkv_out).init(device),
            inter_v: LinearConfig::new(qkv_in, qkv_out).init(device),
            ensemble: (0..ensemble_size)
                .map(|_| LinearConfig::new(deter, stoch * 2).init(device))
                .collect(),
            pos_head1: LinearConfig::new(deter + 2, 64).init(device),
            pos_head2: LinearConfig::new(64, 2).init(device),
            num_slots,
            ctx_dim,
            stoch_size: stoch,
        }
    }

    pub fn init_state(&self, batch: usize, device: &B::Device) -> SlotStates<B> {
        SlotStates::init(batch, self.num_slots, &self.rssm, device)
    }

    fn state_flat(states: &SlotStates<B>) -> RSSMState<B> {
        RSSMState {
            deter: states.deter.clone(),
            stoch: states.stoch.clone(),
            mean: states.mean.clone(),
            std: states.std.clone(),
        }
    }

    /// Cross-slot attention context: [B*K, ctx_dim]. Zero tensor when disabled.
    fn interaction_context(&self, states: &SlotStates<B>) -> Option<Tensor<B, 2>> {
        if self.ctx_dim == 0 {
            return None;
        }
        let b = states.batch;
        let k = self.num_slots;
        let s = Tensor::cat(vec![states.deter.clone(), states.stoch.clone()], 1); // [B*K, S]
        let sdim = s.dims()[1];
        let s3 = s.reshape([b, k, sdim]);

        let q = self.inter_q.forward(s3.clone()); // [B, K, C]
        let key = self.inter_k.forward(s3.clone());
        let v = self.inter_v.forward(s3);

        let scale = (self.ctx_dim as f32).sqrt();
        let logits = q.matmul(key.swap_dims(1, 2)).div_scalar(scale); // [B, K, K]
        let logits_max = logits.clone().max_dim(2).detach();
        let e = logits.sub(logits_max).exp();
        let attn = e.clone().div(e.sum_dim(2).add_scalar(1e-8)); // softmax over source slots
        let ctx = attn.matmul(v); // [B, K, C]
        Some(ctx.reshape([b * k, self.ctx_dim]))
    }

    fn deter_step(&self, states: &SlotStates<B>, action_bk: Tensor<B, 2>) -> Tensor<B, 2> {
        let flat = Self::state_flat(states);
        match self.interaction_context(states) {
            Some(ctx) => self.rssm.get_deter_ctx(&flat, action_bk, ctx),
            None => self.rssm.get_deter(&flat, action_bk),
        }
    }

    /// Predict next slot position from deter + previous position (delta form).
    fn predict_pos(&self, deter: Tensor<B, 2>, prev_pos: Tensor<B, 2>) -> Tensor<B, 2> {
        let inp = Tensor::cat(vec![deter, prev_pos.clone()], 1);
        let delta = self.pos_head2.forward(relu(self.pos_head1.forward(inp)));
        prev_pos.add(delta).clamp(0.0, 1.0)
    }

    /// Observation step for all slots: each slot conditions on its own slot_embed
    pub fn obs_step_all(
        &self,
        states: &SlotStates<B>,
        slot_embeds: Tensor<B, 3>, // [B, K, embed_dim]; last 2 dims = attention centroid
        action: Tensor<B, 2>,      // [B, act_dim]
    ) -> SlotStates<B> {
        let b = states.batch;
        let k = self.num_slots;
        let action_bk = broadcast_action(action, k);
        let edim = slot_embeds.dims()[2];
        let embeds_bk = slot_embeds.reshape([b * k, edim]);
        // Centroid bypass: observed position comes straight from slot attention.
        let pos = embeds_bk.clone().narrow(1, edim - 2, 2);

        let deter = self.deter_step(states, action_bk);
        let post = self.rssm.post_state(deter, embeds_bk);

        SlotStates {
            deter: post.deter,
            stoch: post.stoch,
            mean: post.mean,
            std: post.std,
            pos,
            batch: b,
            num_slots: k,
        }
    }

    /// Imagination step for all slots: each slot transitions with the prior
    pub fn img_step_all(
        &self,
        states: &SlotStates<B>,
        action: Tensor<B, 2>,
    ) -> SlotStates<B> {
        let b = states.batch;
        let k = self.num_slots;
        let action_bk = broadcast_action(action, k);

        let deter = self.deter_step(states, action_bk);
        let prior = self.rssm.prior_state(deter.clone());
        // Imagination: no attention centroid available — roll position forward
        // with the learned dynamics head.
        let pos = self.predict_pos(deter, states.pos.clone());

        SlotStates {
            deter: prior.deter,
            stoch: prior.stoch,
            mean: prior.mean,
            std: prior.std,
            pos,
            batch: b,
            num_slots: k,
        }
    }

    /// Get per-slot prior/posterior for KL computation — both from the SAME h_t.
    pub fn prior_posterior_step(
        &self,
        states: &SlotStates<B>,
        slot_embeds: Tensor<B, 3>,
        action: Tensor<B, 2>,
    ) -> (SlotStates<B>, SlotStates<B>) {
        let b = states.batch;
        let k = self.num_slots;
        let action_bk = broadcast_action(action, k);

        let deter = self.deter_step(states, action_bk);
        let prior = self.rssm.prior_state(deter.clone());

        let edim = slot_embeds.dims()[2];
        let embeds_bk = slot_embeds.reshape([b * k, edim]);
        let post = self.rssm.post_state(deter.clone(), embeds_bk.clone());

        // Posterior position: observed centroid. Prior position: dynamics-head
        // prediction — trained toward the observed centroid via pos_loss.
        let obs_pos = embeds_bk.narrow(1, edim - 2, 2);
        let prior_pos = self.predict_pos(deter, states.pos.clone());

        let prior_states = SlotStates {
            deter: prior.deter.clone(),
            stoch: prior.stoch.clone(),
            mean: prior.mean.clone(),
            std: prior.std.clone(),
            pos: prior_pos,
            batch: b, num_slots: k,
        };
        let post_states = SlotStates {
            deter: post.deter,
            stoch: post.stoch,
            mean: post.mean,
            std: post.std,
            pos: obs_pos,
            batch: b, num_slots: k,
        };

        (prior_states, post_states)
    }

    /// Per-head prior means from the ensemble: Vec of [N, stoch].
    fn ensemble_means(&self, deter: Tensor<B, 2>) -> Vec<Tensor<B, 2>> {
        self.ensemble.iter().map(|head| {
            let stats = head.forward(deter.clone());
            let chunks = stats.chunk(2, 1);
            chunks[0].clone()
        }).collect()
    }

    /// Ensemble training loss: each head predicts the posterior mean from a
    /// DETACHED h_t (the ensemble must not shape the dynamics). Also uses a
    /// softplus std head so heads stay proper Gaussians. Returns scalar [1].
    pub fn ensemble_loss(
        &self,
        deter: Tensor<B, 2>,
        post_mean: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let device = deter.device();
        if self.ensemble.is_empty() {
            return Tensor::zeros([1], &device);
        }
        let d = deter.detach();
        let target = post_mean.detach();
        let mut loss = Tensor::zeros([1], &device);
        for head in self.ensemble.iter() {
            let stats = head.forward(d.clone());
            let chunks = stats.chunk(2, 1);
            let mean = chunks[0].clone();
            let _std = softplus(chunks[1].clone(), 20.0) + 1e-4;
            let diff = mean.sub(target.clone());
            loss = loss.add(diff.clone().mul(diff).mean().reshape([1]));
        }
        loss.div_scalar(self.ensemble.len() as f64)
    }

    /// Epistemic disagreement: variance of ensemble prior means across heads,
    /// averaged over slots and dimensions. Returns a scalar f32 (no grad).
    pub fn disagreement(&self, deter: Tensor<B, 2>) -> f32 {
        if self.ensemble.len() < 2 {
            return 0.0;
        }
        let d = deter.detach();
        let means = self.ensemble_means(d);
        let e = means.len() as f64;
        let mut avg = means[0].clone();
        for m in means.iter().skip(1) {
            avg = avg.add(m.clone());
        }
        let avg = avg.div_scalar(e);
        let mut var = Tensor::zeros_like(&avg);
        for m in means.into_iter() {
            let diff = m.sub(avg.clone());
            var = var.add(diff.clone().mul(diff));
        }
        let var_mean = var.div_scalar(e).mean();
        var_mean.to_data().as_slice::<f32>().unwrap()[0]
    }

    pub fn stoch_size(&self) -> usize { self.stoch_size }
}
