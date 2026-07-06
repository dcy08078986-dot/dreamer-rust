//! Slot Attention: Object-centric decomposition from CNN feature maps.
//!
//! Based on "Object-Centric Learning with Slot Attention" (Locatello et al., NeurIPS 2020).
//! Takes a CNN feature map [B, C, H, W] and produces K slot vectors [B, K, D].
//! Slots compete via dot-product attention (softmax over SLOTS) to bind to distinct
//! image regions; updates are the attention-renormalized weighted MEAN over positions.
//!
//! Deviations from the paper, on purpose:
//! - Slots are initialized from learned per-slot vectors rather than a shared Gaussian.
//!   This keeps autodiff simple in Burn AND index-aligns slots across forward passes,
//!   which the reconstruction-free objective (train-time target matching) relies on.

use burn::module::Module;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};


/// Configuration for slot attention
#[derive(Debug, Clone)]
pub struct SlotAttentionConfig {
    pub num_slots: usize,      // K
    pub slot_dim: usize,       // D
    pub num_iterations: usize, // refinement iterations (3 is typical)
    pub feature_dim: usize,    // C (input feature channels)
}

impl Default for SlotAttentionConfig {
    fn default() -> Self {
        Self {
            num_slots: 4,          // 1 ball + 1 background + 2 spare
            slot_dim: 64,
            num_iterations: 3,
            feature_dim: 64,       // matches CNN encoder output channels
        }
    }
}

/// Slot Attention module
#[derive(Module, Debug)]
pub struct SlotAttention<B: Backend> {
    /// Projects CNN features to attention key/value space
    pub proj_k: Linear<B>,
    pub proj_v: Linear<B>,

    /// Projects slots to attention query space
    pub proj_q: Linear<B>,

    /// GRU for iterative slot refinement
    pub gru: Gru<B>,

    /// MLP after GRU (residual)
    pub mlp: Linear<B>,     // slot_dim -> slot_dim
    pub mlp2: Linear<B>,    // slot_dim -> slot_dim

    /// Learned initial slot vectors (one per slot)
    pub slot_init: Linear<B>, // 1 -> slot_dim (used to generate initial slots)

    /// Soft position embedding: projects (x, y, 1-x, 1-y) grid to feature space
    pub pos_proj: Linear<B>,

    /// LayerNorms (per the paper: inputs, slots pre-attention, pre-MLP)
    pub norm_input: LayerNorm<B>,
    pub norm_slots: LayerNorm<B>,
    pub norm_mlp: LayerNorm<B>,

    num_slots: usize,
    slot_dim: usize,
    num_iterations: usize,
    feature_dim: usize,
}

impl<B: Backend> SlotAttention<B> {
    pub fn init(device: &B::Device, config: &SlotAttentionConfig) -> Self {
        let d = config.slot_dim;
        let f = config.feature_dim;
        Self {
            proj_k: LinearConfig::new(f, d).init(device),
            proj_v: LinearConfig::new(f, d).init(device),
            proj_q: LinearConfig::new(d, d).init(device),
            gru: GruConfig::new(d, d, false).init(device),
            mlp: LinearConfig::new(d, d).init(device),
            mlp2: LinearConfig::new(d, d).init(device),
            slot_init: LinearConfig::new(1, d * config.num_slots).init(device),
            pos_proj: LinearConfig::new(4, f).init(device),
            norm_input: LayerNormConfig::new(f).init(device),
            norm_slots: LayerNormConfig::new(d).init(device),
            norm_mlp: LayerNormConfig::new(d).init(device),
            num_slots: config.num_slots,
            slot_dim: d,
            num_iterations: config.num_iterations,
            feature_dim: f,
        }
    }

    /// Constant (x, y, 1-x, 1-y) position grid: [1, N, 4]
    fn position_grid(&self, h: usize, w: usize, device: &B::Device) -> Tensor<B, 3> {
        let mut coords = Vec::with_capacity(h * w * 4);
        for y in 0..h {
            for x in 0..w {
                let fx = if w > 1 { x as f32 / (w - 1) as f32 } else { 0.5 };
                let fy = if h > 1 { y as f32 / (h - 1) as f32 } else { 0.5 };
                coords.extend_from_slice(&[fx, fy, 1.0 - fx, 1.0 - fy]);
            }
        }
        Tensor::<B, 1>::from_floats(coords.as_slice(), device).reshape([1, h * w, 4])
    }

    /// Forward pass: features [B, C, H, W] → slots [B, K, D]
    pub fn forward(&self, features: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, c, h, w] = features.dims();
        let device = features.device();
        let k = self.num_slots;
        let d = self.slot_dim;
        let n = h * w; // number of spatial positions

        // Flatten spatial dims: [B, C, H, W] → [B, N, C]
        let feats_flat = features.reshape([b, c, n])
            .swap_dims(1, 2); // [B, N, C]

        // Soft position embedding — without it, slots cannot bind to locations.
        let pos = self.pos_proj.forward(self.position_grid(h, w, &device)); // [1, N, C]
        let feats_pos = feats_flat.add(pos); // broadcast over batch
        let feats_norm = self.norm_input.forward(feats_pos);

        // Project features to keys and values: [B, N, D]
        let keys = self.proj_k.forward(feats_norm.clone());
        let values = self.proj_v.forward(feats_norm);

        // Scale for dot-product attention
        let scale = (d as f32).sqrt();

        // Initialize slots from learned parameters (not random — needed for autodiff)
        let ones: Tensor<B, 2> = Tensor::ones([b, 1], &device);
        let raw = self.slot_init.forward(ones); // [B, K*D]
        let mut slots: Tensor<B, 3> = raw.reshape([b, k, d]); // [B, K, D]

        // Iterative refinement; keep the final position-normalized attention for
        // the centroid readout below.
        let mut last_attn_norm: Option<Tensor<B, 3>> = None;
        for _iter in 0..self.num_iterations {
            // --- Step 1: Attention ---
            let slots_norm = self.norm_slots.forward(slots.clone());
            let queries = self.proj_q.forward(slots_norm);

            // Attention logits: Q @ K^T / sqrt(d) → [B, K, N]
            let keys_t = keys.clone().swap_dims(1, 2); // [B, D, N]
            let attn_logits = queries
                .matmul(keys_t)
                .div_scalar(scale); // [B, K, N]

            // Softmax over SLOTS (dim 1) — slots compete for each position.
            // Manual exp/sum (Burn 0.18 autodiff lacks softmax backward).
            // Subtract the per-position max for numerical stability (constant wrt softmax).
            let logits_max = attn_logits.clone().max_dim(1).detach(); // [B, 1, N]
            let attn_exp = attn_logits.sub(logits_max).exp();
            let attn = attn_exp.clone().div(attn_exp.sum_dim(1).add_scalar(1e-8)); // [B, K, N]

            // --- Step 2: Weighted MEAN over positions (paper eq. 2) ---
            // Renormalize attention over positions so each slot's update is a mean,
            // not a magnitude-N sum.
            let attn_norm = attn.clone().div(attn.sum_dim(2).add_scalar(1e-8)); // [B, K, N]
            let updates = attn_norm.clone().matmul(values.clone()); // [B, K, D]
            last_attn_norm = Some(attn_norm);

            // --- Step 3: GRU update ---
            let slots_2d: Tensor<B, 3> = slots.clone().reshape([b * k, d]).unsqueeze_dim(1);
            let updates_2d: Tensor<B, 3> = updates.reshape([b * k, d]).unsqueeze_dim(1);
            let gru_out = self.gru.forward(updates_2d, Some(slots_2d.squeeze(1)));
            let squeezed: Tensor<B, 2> = gru_out.squeeze(1);
            let slots_new: Tensor<B, 3> = squeezed.reshape([b, k, d]);

            // --- Step 4: Residual MLP (pre-norm) ---
            let mlp_out = self.mlp.forward(self.norm_mlp.forward(slots_new.clone()));
            let mlp_out = burn::tensor::activation::relu(mlp_out);
            let mlp_out = self.mlp2.forward(mlp_out);
            slots = slots_new.add(mlp_out);
        }

        // ── Explicit position readout (SPAIR-style) ──
        // The attention centroid over the coordinate grid is a continuous,
        // differentiable, sub-cell-precise estimate of each slot's location.
        // Write it into the slots' LAST 2 dims so downstream consumers (RSSM
        // posterior → decoder blobs) get position directly instead of having to
        // recover it from the mixed feature average. A weighted feature mean
        // alone quantizes position to the grid pitch — this is what capped the
        // slot-AE ball MSE at ~0.06.
        if let Some(attn_norm) = last_attn_norm {
            let n_pos = h * w;
            let mut grid_xy = Vec::with_capacity(n_pos * 2);
            for y in 0..h {
                for x in 0..w {
                    grid_xy.push(if w > 1 { x as f32 / (w - 1) as f32 } else { 0.5 });
                    grid_xy.push(if h > 1 { y as f32 / (h - 1) as f32 } else { 0.5 });
                }
            }
            let grid = Tensor::<B, 1>::from_floats(grid_xy.as_slice(), &device)
                .reshape([1, n_pos, 2]);
            let grid_b: Tensor<B, 3> = Tensor::zeros([b, n_pos, 2], &device).add(grid);
            let centroid = attn_norm.matmul(grid_b); // [B, K, 2] in [0, 1]
            slots = Tensor::cat(vec![slots.narrow(2, 0, d - 2), centroid], 2);
        }

        slots // [B, K, D]
    }

    pub fn num_slots(&self) -> usize { self.num_slots }
    pub fn slot_dim(&self) -> usize { self.slot_dim }
}
