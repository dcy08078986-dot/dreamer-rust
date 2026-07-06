//! Hybrid Decoder — broadcast background + analytic Gaussian-blob foreground.
//!
//! The 8×8 BroadcastDecoder is structurally unable to place a 16-px ball with
//! sub-pixel precision (its coordinate channels live on an 8×8 grid), but it is
//! good at smooth content (sky gradient, clouds). The GaussianDecoder places
//! rotated super-Gaussian blobs at continuous coordinates — sub-pixel foreground
//! localization for free — but cannot paint gradients. This module composites the
//! two per slot:
//!
//!   w      = 1 − exp(−alpha_sum)          (smooth blob coverage in [0, 1))
//!   rgb    = w · rgb_blob + (1 − w) · rgb_background
//!   mask   = log(exp(clamp(bg_logit)) + alpha_sum)   (slot's claim on each pixel)
//!
//! OUTPUT CONVENTION matches GaussianDecoder: [N, D] → [N, C+1, H, W] with RGB
//! ALREADY in [0, 1] (no extra sigmoid) and a raw mask logit in the last channel.

use burn::module::Module;
use burn::tensor::activation::sigmoid;
use burn::tensor::{backend::Backend, Tensor};

use super::broadcast_decoder::BroadcastDecoder;
use super::gaussian_decoder::GaussianDecoder;

#[derive(Module, Debug)]
pub struct HybridDecoder<B: Backend> {
    pub bg: BroadcastDecoder<B>,
    pub fg: GaussianDecoder<B>,
    out_channels: usize,
}

impl<B: Backend> HybridDecoder<B> {
    pub fn init(
        device: &B::Device,
        latent_dim: usize,
        image_channels: usize,
        image_size: usize,
    ) -> Self {
        Self {
            bg: BroadcastDecoder::init(device, latent_dim, image_channels, image_size),
            fg: GaussianDecoder::init(device, latent_dim, image_channels, image_size),
            out_channels: image_channels + 1,
        }
    }

    /// [N, D] → [N, C+1, H, W]: RGB in [0,1] + raw mask logit (see module doc).
    pub fn forward(&self, latent: Tensor<B, 2>) -> Tensor<B, 4> {
        let c = self.out_channels - 1;

        let bg_out = self.bg.forward(latent.clone());
        let bg_rgb = sigmoid(bg_out.clone().narrow(1, 0, c)); // [N, C, H, W]
        let bg_logit = bg_out.narrow(1, c, 1);                // [N, 1, H, W]

        // Sequential alpha-over: blobs painted directly onto the background head's
        // canvas — division-free and bounded in [0, 1].
        let (rgb, alpha_sum) = self.fg.composite_over(latent, bg_rgb, None);

        // Slot mask logit: logsumexp-style merge of the background head's claim and
        // the blob coverage. Clamp keeps exp() finite.
        let mask = bg_logit
            .clamp(-10.0, 10.0)
            .exp()
            .add(alpha_sum)
            .add_scalar(1e-8)
            .log();

        Tensor::cat(vec![rgb, mask], 1)
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}
