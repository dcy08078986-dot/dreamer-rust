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