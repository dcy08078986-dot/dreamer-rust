//! Gaussian Blob Decoder — analytic rendering replaces learned upsampling.
//!
//! Each slot latent predicts K blobs × 9 params (cx, cy, sx, sy, theta, alpha, r, g, b).
//! Blobs are rendered as ROTATED SUPER-GAUSSIANS: kernel = exp(-0.5·q²) where q is the
//! rotated Mahalanobis form. The squared exponent gives a flat top and a sharp rim —
//! the ball is a solid disc with a hard edge, and a plain Gaussian's soft falloff loses
//! exactly the rim pixels that dominate the ball-region metric. theta lets an elongated
//! blob (sx ≫ sy) render the oriented velocity arrow.
//!
//! OUTPUT CONVENTION (differs from BroadcastDecoder): [N, D] → [N, C+1, H, W] where the
//! C RGB channels are ALREADY in [0, 1] — callers must NOT apply sigmoid again — and the
//! last channel is a raw mask logit (log of accumulated blob alpha).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{backend::Backend, Tensor};

const PARAMS_PER_BLOB: usize = 9; // cx, cy, sx, sy, theta, alpha, r, g, b

#[derive(Module, Debug)]
pub struct GaussianDecoder<B: Backend> {
    pub blob_hidden: Linear<B>,
    pub blob_mlp: Linear<B>,
    n_blobs: usize,
    image_size: usize,
    out_channels: usize,
}

impl<B: Backend> GaussianDecoder<B> {
    pub fn init(
        device: &B::Device,
        latent_dim: usize,
        image_channels: usize,
        image_size: usize,
    ) -> Self {
        let n_blobs: usize = 12;
        let hidden = 128;
        // +2 for optional (cx_hint, cy_hint) concatenated before MLP
        Self {
            blob_hidden: LinearConfig::new(latent_dim + 2, hidden).init(device),
            blob_mlp: LinearConfig::new(hidden, n_blobs * PARAMS_PER_BLOB).init(device),
            n_blobs,
            image_size,
            out_channels: image_channels + 1,
        }
    }

    fn make_grid(&self, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let s = self.image_size;
        let n = s * s;
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        let denom = (s - 1).max(1) as f32;
        for py in 0..s {
            for px in 0..s {
                xv.push(px as f32 / denom);
                yv.push(py as f32 / denom);
            }
        }
        let x = Tensor::<B, 1>::from_floats(xv.as_slice(), device).reshape([1, 1, s, s]);
        let y = Tensor::<B, 1>::from_floats(yv.as_slice(), device).reshape([1, 1, s, s]);
        (x, y)
    }

    /// Rotated super-Gaussian kernel: alpha · exp(-0.5·q²),
    /// q = (du/sx)² + (dv/sy)² with (du, dv) the (dx, dy) offset rotated by theta.
    #[allow(clippy::too_many_arguments)]
    fn render_blob(
        &self,
        cx: Tensor<B, 2>,
        cy: Tensor<B, 2>,
        sx: Tensor<B, 2>,
        sy: Tensor<B, 2>,
        theta: Tensor<B, 2>,
        alpha: Tensor<B, 2>,
        x_grid: Tensor<B, 4>,
        y_grid: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let n = cx.dims()[0];
        let dx = x_grid.clone() - cx.reshape([n, 1, 1, 1]);
        let dy = y_grid.clone() - cy.reshape([n, 1, 1, 1]);

        let ct = theta.clone().cos().reshape([n, 1, 1, 1]);
        let st = theta.sin().reshape([n, 1, 1, 1]);
        let du = dx.clone().mul(ct.clone()).add(dy.clone().mul(st.clone()));
        let dv = dy.mul(ct).sub(dx.mul(st));

        let var_x = sx.clone().mul(sx).reshape([n, 1, 1, 1]);
        let var_y = sy.clone().mul(sy).reshape([n, 1, 1, 1]);
        let q = du.clone().mul(du).div(var_x.add_scalar(1e-6))
            + dv.clone().mul(dv).div(var_y.add_scalar(1e-6));
        // Super-Gaussian (order 8): flat top, sharp rim. q ≥ 0 so exp(-0.5·q⁴) is
        // bounded and its gradient vanishes far from the blob — no overflow risk.
        let q2 = q.clone().mul(q);
        let falloff = q2.clone().mul(q2).mul_scalar(-0.5).exp();
        alpha.reshape([n, 1, 1, 1]).mul(falloff)
    }

    /// Paint blobs over `base`. `pos_hint`: optional [N, 2] center hint in [0,1].
    /// When Some, concatenated to latent before MLP (dropout=50% in training).
    pub fn composite_over(
        &self,
        latent: Tensor<B, 2>,
        base: Tensor<B, 4>,
        pos_hint: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [n, _d] = latent.dims();
        let device = latent.device();
        let h = self.image_size;
        let w = self.image_size;
        let k = self.n_blobs;

        // Mix in position hint if provided (50% dropout during training)
        let lat_aug = match pos_hint {
            Some(hint) => Tensor::cat(vec![latent, hint], 1),
            None => {
                let z = Tensor::zeros([n, 2], &device);
                Tensor::cat(vec![latent, z], 1)
            }
        };

        let raw = self.blob_mlp.forward(relu(self.blob_hidden.forward(lat_aug)));
        let p = raw.reshape([n, k, PARAMS_PER_BLOB]);

        let col = |j: usize| -> Tensor<B, 2> { p.clone().narrow(2, j, 1).squeeze(2) };

        let cx = sigmoid(col(0));
        let cy = sigmoid(col(1));
        // Bounded scale: sigma ∈ (0.005, 0.355) of image width. The previous
        // softplus(·, beta=20) overflowed f32 (exp(20x) → inf for x > 4.4) as soon
        // as a blob tried to grow large, poisoning training with NaN.
        let sx = sigmoid(col(2)).mul_scalar(0.35).add_scalar(0.005);
        let sy = sigmoid(col(3)).mul_scalar(0.35).add_scalar(0.005);
        let theta = col(4); // radians, unbounded (periodic)
        let alpha = sigmoid(col(5));
        let rr = sigmoid(col(6));
        let gg = sigmoid(col(7));
        let bb = sigmoid(col(8));

        let (xg, yg) = self.make_grid(&device);

        let mut rgb = base;
        let mut alpha_sum = Tensor::zeros([n, 1, h, w], &device);

        for i in 0..k {
            let idx = |t: &Tensor<B, 2>| t.clone().narrow(1, i, 1);
            let kernel = self.render_blob(
                idx(&cx), idx(&cy), idx(&sx), idx(&sy), idx(&theta), idx(&alpha),
                xg.clone(), yg.clone(),
            ); // [N, 1, H, W] in [0, 1] (alpha is sigmoided)

            let color = Tensor::cat(vec![idx(&rr), idx(&gg), idx(&bb)], 1)
                .reshape([n, 3, 1, 1]); // [N, C, 1, 1]

            let one_minus_k = kernel.clone().neg().add_scalar(1.0);
            rgb = kernel.clone().mul(color).add(one_minus_k.mul(rgb));
            alpha_sum = alpha_sum + kernel;
        }

        (rgb, alpha_sum)
    }

    /// [N, D] → [N, C+1, H, W]: RGB already in [0,1] + raw mask logit.
    /// Position hint is already concatenated into latent by the caller (decode_slots
    /// injects SlotStates.pos). No separate hint parameter needed.
    pub fn forward(&self, latent: Tensor<B, 2>) -> Tensor<B, 4> {
        let [n, _d] = latent.dims();
        let device = latent.device();
        let c = self.out_channels - 1;
        let black = Tensor::zeros([n, c, self.image_size, self.image_size], &device);
        let (rgb, alpha_sum) = self.composite_over(latent, black, None);
        let mask = alpha_sum.add_scalar(1e-8).log();
        Tensor::cat(vec![rgb, mask], 1)
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}
