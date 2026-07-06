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
        let x = self.forward_features(input);
        let [b, _c, _h, _w] = x.dims();
        let x = x.reshape([b, self.flat_dim]);
        self.proj.forward(x)
    }

    /// input: [B, C, H, W] → output: [B, 256, H/16, W/16]  (spatial feature map)
    pub fn forward_features(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.conv1.forward(input));
        let x = relu(self.conv2.forward(x));
        let x = relu(self.conv3.forward(x));
        relu(self.conv4.forward(x))
    }

    /// input: [B, C, H, W] → output: [B, 128, H/8, W/8]
    /// Higher-resolution feature map for slot attention: at H/16 (4×4 for 64px
    /// images) an 8-px ball covers 1-2 cells and cannot be segmented; at H/8
    /// (8×8) it covers a workable region.
    pub fn forward_features_8x8(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.conv1.forward(input));
        let x = relu(self.conv2.forward(x));
        relu(self.conv3.forward(x))
    }

    /// input: [B, C, H, W] → output: [B, 64, H/4, W/4]
    /// Finest feature map for slot attention (16×16 for 64px images): the ball
    /// covers ~4×4 cells, enough for the attention mask to localize it with
    /// sub-cell precision via its centroid.
    pub fn forward_features_16x16(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.conv1.forward(input));
        relu(self.conv2.forward(x))
    }
}
