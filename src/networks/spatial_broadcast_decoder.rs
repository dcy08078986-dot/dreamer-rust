//! Full-Resolution Spatial Broadcast Decoder
//!
//! Key improvement over the original BroadcastDecoder:
//! - Broadcasts slot latents to FULL RESOLUTION (64×64) instead of 8×8
//! - Avoids multiple upsampling layers that lose spatial precision
//! - Uses lightweight CNN to decode from [latent_dim + 2] → 4 channels
//!
//! Architecture:
//!   1. MLP: [B*K, latent_dim] → [B*K, hidden_dim]
//!   2. Broadcast to full resolution: [B*K, hidden_dim, 1, 1] → [B*K, hidden_dim, H, W]
//!   3. Concatenate coordinate grid (x, y): → [B*K, hidden_dim+2, H, W]
//!   4. Lightweight CNN: Conv(hidden_dim+2 → 128 → 64 → 4)
//!   5. Output: RGB (3 channels) + mask logit (1 channel)
//!
//! This design preserves spatial information at full resolution, crucial for
//! small objects like the bouncing ball (4×4 pixels).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::relu;
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct SpatialBroadcastDecoder<B: Backend> {
    /// MLP to expand latent before broadcasting
    pub latent_proj: Linear<B>,

    /// Lightweight CNN decoder (no upsampling needed)
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,

    latent_dim: usize,
    hidden_dim: usize,
    image_size: usize,
    image_channels: usize,
}

impl<B: Backend> SpatialBroadcastDecoder<B> {
    /// Create a new full-resolution spatial broadcast decoder
    ///
    /// # Arguments
    /// * `latent_dim` - Per-slot latent size (e.g., 64 for deter+stoch after projection)
    /// * `hidden_dim` - Hidden dimension after MLP expansion (e.g., 256)
    /// * `image_channels` - Output RGB channels (typically 3)
    /// * `image_size` - Full resolution (e.g., 64)
    pub fn init(
        device: &B::Device,
        latent_dim: usize,
        hidden_dim: usize,
        image_channels: usize,
        image_size: usize,
    ) -> Self {
        let pad = PaddingConfig2d::Explicit(1, 1);

        Self {
            // MLP: latent_dim → hidden_dim
            latent_proj: LinearConfig::new(latent_dim, hidden_dim).init(device),

            // Lightweight CNN (no upsampling)
            // Input: hidden_dim + 2 (coords), Output: 4 (RGB + mask)
            conv1: Conv2dConfig::new([hidden_dim + 2, 128], [3, 3])
                .with_padding(pad.clone())
                .init(device),
            conv2: Conv2dConfig::new([128, 64], [3, 3])
                .with_padding(pad.clone())
                .init(device),
            conv3: Conv2dConfig::new([64, image_channels + 1], [3, 3])
                .with_padding(pad)
                .init(device),

            latent_dim,
            hidden_dim,
            image_size,
            image_channels,
        }
    }

    /// Generate coordinate grid [1, 2, H, W] with x, y in [-1, 1]
    fn coord_grid(&self, device: &B::Device) -> Tensor<B, 4> {
        let h = self.image_size;
        let w = self.image_size;
        let mut data = Vec::with_capacity(2 * h * w);

        // Channel 0: x coordinates (column-wise, CHW layout)
        for _y in 0..h {
            for x in 0..w {
                let x_norm = if w > 1 { 2.0 * x as f32 / (w - 1) as f32 - 1.0 } else { 0.0 };
                data.push(x_norm);
            }
        }

        // Channel 1: y coordinates (row-wise)
        for y in 0..h {
            for _x in 0..w {
                let y_norm = if h > 1 { 2.0 * y as f32 / (h - 1) as f32 - 1.0 } else { 0.0 };
                data.push(y_norm);
            }
        }

        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 2, h, w])
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `latent` - [N, latent_dim] where N = batch_size * num_slots
    ///
    /// # Returns
    /// Raw [N, image_channels + 1, H, W] tensor:
    /// - First `image_channels` are RGB logits (caller applies sigmoid)
    /// - Last channel is mask logit (used for per-pixel softmax over slots)
    pub fn forward(&self, latent: Tensor<B, 2>) -> Tensor<B, 4> {
        let [n, _] = latent.dims();
        let h = self.image_size;
        let w = self.image_size;
        let device = latent.device();

        // 1. MLP expansion: [N, latent_dim] → [N, hidden_dim]
        let expanded = relu(self.latent_proj.forward(latent));

        // 2. Broadcast to full resolution: [N, hidden_dim] → [N, hidden_dim, H, W]
        let expanded_4d = expanded.reshape([n, self.hidden_dim, 1, 1]);
        let broadcasted: Tensor<B, 4> = Tensor::zeros([n, self.hidden_dim, h, w], &device)
            .add(expanded_4d);

        // 3. Concatenate coordinate grid: → [N, hidden_dim+2, H, W]
        let coords: Tensor<B, 4> = Tensor::zeros([n, 2, h, w], &device)
            .add(self.coord_grid(&device));
        let x = Tensor::cat(vec![broadcasted, coords], 1);

        // 4. Lightweight CNN (no upsampling needed)
        let x = relu(self.conv1.forward(x));  // [N, 128, H, W]
        let x = relu(self.conv2.forward(x));  // [N, 64, H, W]
        self.conv3.forward(x)                 // [N, C+1, H, W] (raw logits)
    }

    pub fn out_channels(&self) -> usize {
        self.image_channels + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::activation::sigmoid;

    type TestBackend = NdArray;

    #[test]
    fn test_coord_grid() {
        let device = Default::default();
        let decoder = SpatialBroadcastDecoder::<TestBackend>::init(
            &device,
            64,   // latent_dim
            256,  // hidden_dim
            3,    // image_channels
            4,    // image_size (small for testing)
        );

        let coords = decoder.coord_grid(&device);
        let dims = coords.dims();

        assert_eq!(dims, [1, 2, 4, 4]);

        // Check coordinate range [-1, 1]
        let data = coords.to_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        for &v in &values {
            assert!(v >= -1.0 && v <= 1.0, "Coordinate {} out of range [-1, 1]", v);
        }
    }

    #[test]
    fn test_forward_shape() {
        let device = Default::default();
        let batch_size = 4;
        let num_slots = 3;
        let latent_dim = 64;
        let image_size = 64;
        let image_channels = 3;

        let decoder = SpatialBroadcastDecoder::<TestBackend>::init(
            &device,
            latent_dim,
            256,  // hidden_dim
            image_channels,
            image_size,
        );

        // Input: [B*K, latent_dim]
        let n = batch_size * num_slots;
        let latent = Tensor::<TestBackend, 2>::zeros([n, latent_dim], &device);

        // Forward
        let output = decoder.forward(latent);
        let dims = output.dims();

        // Should output [B*K, C+1, H, W]
        assert_eq!(dims, [n, image_channels + 1, image_size, image_size]);
    }

    /// Test that the decoder can reconstruct a simple ball when given
    /// ground-truth ball parameters as input
    #[test]
    fn test_ball_reconstruction() {
        let device = Default::default();
        let latent_dim = 64;
        let image_size = 64;
        let image_channels = 3;

        let decoder = SpatialBroadcastDecoder::<TestBackend>::init(
            &device,
            latent_dim,
            256,  // hidden_dim
            image_channels,
            image_size,
        );

        // Create a dummy latent (in real test, this would encode ball position)
        let latent = Tensor::<TestBackend, 2>::zeros([1, latent_dim], &device);

        // Forward pass
        let output = decoder.forward(latent);

        // Apply sigmoid to RGB channels
        let rgb = sigmoid(output.clone().narrow(1, 0, image_channels));
        let mask = output.narrow(1, image_channels, 1);

        // Check shapes
        assert_eq!(rgb.dims(), [1, image_channels, image_size, image_size]);
        assert_eq!(mask.dims(), [1, 1, image_size, image_size]);

        // Note: Without trained weights, we can't verify actual reconstruction quality
        // This test just verifies the architecture works end-to-end
    }
}
