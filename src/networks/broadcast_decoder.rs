//! Spatial Broadcast Decoder for per-slot decoding (Watters et al., 2019; used by
//! slot-based world models like SOLD, ICML 2025).
//!
//! Each slot latent is tiled over a small spatial grid, augmented with coordinate
//! channels, and decoded to RGB + a per-pixel mask logit. The coordinate channels
//! bias slots toward spatially local objects, and the mask logits enable per-pixel
//! alpha compositing (softmax over slots) — the ingredient that makes object
//! decomposition possible at all. Cheaper than a full deconv stack per slot.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::relu;
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct BroadcastDecoder<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub up1: ConvTranspose2d<B>,
    pub up2: ConvTranspose2d<B>,
    pub up3: ConvTranspose2d<B>,
    latent_dim: usize,
    base_size: usize,
    out_channels: usize,
}

impl<B: Backend> BroadcastDecoder<B> {
    /// latent_dim: per-slot latent size (deter + stoch after projection).
    /// image_size must be base_size * 8 (three 2x upsamples); 64 → base 8.
    pub fn init(
        device: &B::Device,
        latent_dim: usize,
        image_channels: usize,
        image_size: usize,
    ) -> Self {
        let base_size = image_size / 8;
        let pad1 = PaddingConfig2d::Explicit(1, 1);
        Self {
            conv1: Conv2dConfig::new([latent_dim + 2, 64], [3, 3])
                .with_padding(pad1.clone())
                .init(device),
            conv2: Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(pad1)
                .init(device),
            up1: ConvTranspose2dConfig::new([64, 32], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            up2: ConvTranspose2dConfig::new([32, 16], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            up3: ConvTranspose2dConfig::new([16, image_channels + 1], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            latent_dim,
            base_size,
            out_channels: image_channels + 1,
        }
    }

    /// Constant coordinate grid [1, 2, S, S] in [-1, 1].
    fn coord_grid(&self, device: &B::Device) -> Tensor<B, 4> {
        let s = self.base_size;
        let mut data = Vec::with_capacity(2 * s * s);
        // channel 0: x, channel 1: y (CHW layout)
        for y in 0..s {
            for x in 0..s {
                let _ = y;
                data.push(if s > 1 { 2.0 * x as f32 / (s - 1) as f32 - 1.0 } else { 0.0 });
            }
        }
        for y in 0..s {
            for x in 0..s {
                let _ = x;
                data.push(if s > 1 { 2.0 * y as f32 / (s - 1) as f32 - 1.0 } else { 0.0 });
            }
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 2, s, s])
    }

    /// input: [N, latent_dim] (one row per slot instance)
    /// output: RAW [N, image_channels + 1, H, W] — caller applies sigmoid to the RGB
    /// channels and uses the last channel as the mask logit.
    pub fn forward(&self, latent: Tensor<B, 2>) -> Tensor<B, 4> {
        let [n, dl] = latent.dims();
        let s = self.base_size;
        let device = latent.device();

        // Spatial broadcast: tile latent over the grid (broadcast-add, repeat-free)
        let lat_4d = latent.reshape([n, dl, 1, 1]);
        let tiled: Tensor<B, 4> = Tensor::zeros([n, dl, s, s], &device).add(lat_4d);
        let coords: Tensor<B, 4> =
            Tensor::zeros([n, 2, s, s], &device).add(self.coord_grid(&device));
        let x = Tensor::cat(vec![tiled, coords], 1); // [N, dl+2, S, S]

        let x = relu(self.conv1.forward(x));
        let x = relu(self.conv2.forward(x));
        let x = relu(self.up1.forward(x));
        let x = relu(self.up2.forward(x));
        self.up3.forward(x) // raw logits, [N, C+1, H, W]
    }

    pub fn out_channels(&self) -> usize { self.out_channels }
}
