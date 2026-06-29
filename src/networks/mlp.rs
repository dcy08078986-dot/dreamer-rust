#![allow(dead_code)]

use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::tensor::backend::Backend;

pub struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Relu,
}

#[derive(Debug, Clone)]
pub struct MlpConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub num_layers: usize,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let mut layers = Vec::new();

        let mut in_dim = self.input_dim;

        // hidden layers
        for _ in 0..self.num_layers {
            let layer = LinearConfig::new(in_dim, self.hidden_dim)
                .init(device);
            layers.push(layer);
            in_dim = self.hidden_dim;
        }

        // output layer
        let out_layer = LinearConfig::new(in_dim, self.output_dim)
            .init(device);

        layers.push(out_layer);

        Mlp {
            layers,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Mlp<B> {
    pub fn forward<const D: usize>(
        &self,
        mut x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let last = self.layers.len() - 1;

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);

            // 最后一层不激活
            if i != last {
                x = self.activation.forward(x);
            }
        }

        x
    }
}