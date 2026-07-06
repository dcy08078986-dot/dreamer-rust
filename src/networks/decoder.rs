use burn::module::Module;
use burn::nn::conv::{ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    pub proj: Linear<B>,
    pub deconv1: ConvTranspose2d<B>,
    pub deconv2: ConvTranspose2d<B>,
    pub deconv3: ConvTranspose2d<B>,
    pub deconv4: ConvTranspose2d<B>,
    flat_dim: usize,
    feature_size: usize,
    rgb_channels: usize,  // number of channels to sigmoid (rest kept raw)
}

impl<B: Backend> Decoder<B> {
    pub fn init(
        device: &B::Device,
        latent_dim: usize,
        image_channels: usize,
        image_size: usize,
    ) -> Self {
        Self::init_with_out(device, latent_dim, image_channels, image_size, image_channels)
    }

    /// init_with_out: output `out_c` channels. First `rgb_c` channels get sigmoid,
    /// rest stay raw (for mask logits in slot compositing).
    pub fn init_with_out(
        device: &B::Device,
        latent_dim: usize,
        rgb_c: usize,
        image_size: usize,
        out_c: usize,
    ) -> Self {
        let fs = image_size / 16;
        let flat_dim = 256 * fs * fs;
        Self {
            proj: LinearConfig::new(latent_dim, flat_dim).init(device),
            deconv1: ConvTranspose2dConfig::new([256, 128], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            deconv2: ConvTranspose2dConfig::new([128, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            deconv3: ConvTranspose2dConfig::new([64, 32], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            deconv4: ConvTranspose2dConfig::new([32, out_c], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            flat_dim,
            feature_size: fs,
            rgb_channels: rgb_c,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 4> {
        let [b, _] = input.dims();
        let fs = self.feature_size;
        let x = relu(self.proj.forward(input));
        let x = x.reshape([b, 256, fs, fs]);
        let x = relu(self.deconv1.forward(x));
        let x = relu(self.deconv2.forward(x));
        let x = relu(self.deconv3.forward(x));
        let raw = self.deconv4.forward(x);
        if self.rgb_channels == 0 {
            return raw; // all raw, caller handles activation
        }
        let out_c = raw.dims()[1];
        let rgb = sigmoid(raw.clone().narrow(1, 0, self.rgb_channels));
        if self.rgb_channels < out_c {
            let extra = raw.narrow(1, self.rgb_channels, out_c - self.rgb_channels);
            Tensor::cat(vec![rgb, extra], 1)
        } else {
            rgb
        }
    }
}
