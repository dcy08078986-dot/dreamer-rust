use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
    pub conv4: Conv2d<B>,
    pub proj: Linear<B>,
    flat_dim: usize,
}

impl<B: Backend> Encoder<B> {
    pub fn init(
        device: &B::Device,
        embed_dim: usize,
        image_channels: usize,
        image_size: usize,
    ) -> Self {
        let fs = image_size / 16; // 4 stride-2 convs: size → size/2^4
        let flat_dim = 256 * fs * fs;
        let pad = PaddingConfig2d::Explicit(1, 1);
        Self {
            conv1: Conv2dConfig::new([image_channels, 32], [4, 4])
                .with_stride([2, 2])
                .with_padding(pad.clone())
                .init(device),
            conv2: Conv2dConfig::new([32, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding(pad.clone())
                .init(device),
            conv3: Conv2dConfig::new([64, 128], [4, 4])
                .with_stride([2, 2])
                .with_padding(pad.clone())
                .init(device),
            conv4: Conv2dConfig::new([128, 256], [4, 4])
                .with_stride([2, 2])
                .with_padding(pad)
                .init(device),
            proj: LinearConfig::new(flat_dim, embed_dim).init(device),
            flat_dim,
        }
    }

    /// input: [B, C, H, W]  —  image
    /// output: [B, embed_dim]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.conv1.forward(input));
        let x = relu(self.conv2.forward(x));
        let x = relu(self.conv3.forward(x));
        let x = relu(self.conv4.forward(x));
        let [b, _c, _h, _w] = x.dims();
        let x = x.reshape([b, self.flat_dim]);
        self.proj.forward(x)
    }
}
