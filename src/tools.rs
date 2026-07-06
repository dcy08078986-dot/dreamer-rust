#![allow(dead_code)]

use burn::tensor::{backend::Backend, Tensor};

pub fn symlog<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sign = x.clone().sign();
    let abs = x.abs();
    sign * (abs + 1.0).log()
}

pub fn symexp<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sign = x.clone().sign();
    let abs = x.abs();
    sign * (abs.exp() - 1.0)
}

/// Per-pixel motion saliency from a frame difference (Proposal A).
///
/// obs_t / obs_prev: [B, C*H*W] flattened CHW images in [0,1].
/// Returns detached saliency in [0, 1] per pixel, broadcast across channels: [B, C*H*W].
/// Saliency is the max channel-wise absolute difference, so a red ball moving over
/// a blue sky lights up regardless of which channel changed.
pub fn motion_saliency<B: Backend>(
    obs_t: Tensor<B, 2>,
    obs_prev: Tensor<B, 2>,
    channels: usize,
) -> Tensor<B, 2> {
    let [b, chw] = obs_t.dims();
    let hw = chw / channels;
    let diff = obs_t.sub(obs_prev).abs().detach(); // [B, CHW]
    let diff_3d = diff.reshape([b, channels, hw]);
    let sal = diff_3d.max_dim(1); // [B, 1, HW]
    // Broadcast back across channels via zero-add (repeat-free broadcasting).
    let sal_c: Tensor<B, 3> = Tensor::zeros([b, channels, hw], &sal.device()).add(sal);
    sal_c.reshape([b, chw])
}

/// Motion-weighted per-pixel weights: w = min(1 + lambda_m * sal / (mean(sal) + eps), w_max).
/// Weights are detached; normalization is per-sample so every frame contributes.
/// The cap matters: saliency is SPARSE (ball ≈ 1.6% of pixels), so sal/mean(sal) ≈ 60
/// at ball pixels — uncapped, ~90% of the loss lands on 200 pixels and the optimum
/// becomes "paint the whole image ball-colored". w_max = 50 puts roughly 45% of the
/// gradient on moving pixels and 55% on the background.
pub fn motion_weights<B: Backend>(
    saliency: Tensor<B, 2>,
    lambda_m: f32,
    w_max: f32,
) -> Tensor<B, 2> {
    let sal_mean = saliency.clone().mean_dim(1).add_scalar(1e-6); // [B, 1]
    saliency.div(sal_mean).mul_scalar(lambda_m).add_scalar(1.0).clamp_max(w_max)
}

/// Weighted MSE: sum(w * (a-b)^2) / sum(w). Returns a scalar [1] tensor.
pub fn weighted_mse<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    weights: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let d = pred.sub(target);
    let sq = d.clone().mul(d);
    let num = sq.mul(weights.clone()).sum();
    let den = weights.sum().add_scalar(1e-8);
    num.div(den)
}

/// Diagnostic: MSE restricted to moving pixels (saliency above threshold).
/// This is the "ball-region MSE" health metric — full-image MSE is dominated by the
/// static background and can look healthy while the ball is ignored entirely.
pub fn motion_region_mse<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    saliency: Tensor<B, 2>,
    threshold: f32,
) -> Tensor<B, 1> {
    let mask = saliency.greater_elem(threshold).float().detach();
    let d = pred.sub(target);
    let sq = d.clone().mul(d);
    let num = sq.mul(mask.clone()).sum();
    let den = mask.sum().add_scalar(1e-8);
    num.div(den)
}

/// Per-batch weighted centroid of motion saliency. Returns [B, 2] with (cx, cy) in [0, 1].
/// Used as a positional hint for the Gaussian decoder.
pub fn saliency_center<B: Backend>(
    obs_t: Tensor<B, 2>,
    obs_prev: Tensor<B, 2>,
    channels: usize,
) -> Tensor<B, 2> {
    let [b, chw] = obs_t.dims();
    let hw = chw / channels;
    let h = (hw as f32).sqrt() as usize;
    let w = h;

    // Build coordinate grid [H*W]
    let mut xv = Vec::with_capacity(hw);
    let mut yv = Vec::with_capacity(hw);
    for y in 0..h {
        for x in 0..w {
            xv.push(x as f32 / (w - 1).max(1) as f32);
            yv.push(y as f32 / (h - 1).max(1) as f32);
        }
    }
    let x_t = Tensor::<B, 1>::from_floats(xv.as_slice(), &obs_t.device()).unsqueeze();
    let y_t = Tensor::<B, 1>::from_floats(yv.as_slice(), &obs_t.device()).unsqueeze();

    // Saliency: [B, CHW], take max over channels, keep [B, 1, HW]
    let diff = obs_t.sub(obs_prev).abs().detach();
    let diff_3d = diff.reshape([b, channels, hw]);
    let sal = diff_3d.max_dim(1); // [B, 1, HW]
    let sal_sum = sal.clone().sum_dim(2).add_scalar(1e-8); // [B, 1, 1]

    let cx = sal.clone().mul(x_t.clone()).sum_dim(2).div(sal_sum.clone()); // [B, 1]
    let cy = sal.mul(y_t).sum_dim(2).div(sal_sum);
    Tensor::cat(vec![cx, cy], 1).squeeze(2).detach() // [B, 2]
}
