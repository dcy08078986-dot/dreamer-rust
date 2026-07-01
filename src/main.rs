#![recursion_limit = "256"]

mod config;
mod tools;
mod replay;
mod train;
mod networks;
mod agent;
mod envs;
mod video;

use burn::backend::{Autodiff, NdArray};
use envs::bouncing_ball::BouncingBall;
use envs::paddle_hitting::PaddleHitting;

fn main() {
    type Backend = Autodiff<NdArray>;
    let device = Default::default();
    let config = config::Config::default();

    let num_envs = config.num_envs;

    match config.env_type.as_str() {
        "bouncing_ball" => {
            println!("Creating {} parallel BouncingBall environments...", num_envs);
            let mut envs: Vec<BouncingBall> = (0..num_envs)
                .map(|i| BouncingBall::new(
                    config.env_max_steps,
                    config.action_dim,
                    config.image_channels,
                    config.image_size,
                    config.seed + i as u64
                ))
                .collect();
            train::run::<Backend, BouncingBall>(device, config, &mut envs);
        }
        "paddle_hitting" => {
            println!("Creating {} parallel PaddleHitting environments...", num_envs);
            let mut envs: Vec<PaddleHitting> = (0..num_envs)
                .map(|i| PaddleHitting::new(
                    config.env_max_steps,
                    config.action_dim,
                    config.image_channels,
                    config.image_size,
                    config.seed + i as u64
                ))
                .collect();
            train::run::<Backend, PaddleHitting>(device, config, &mut envs);
        }
        _ => {
            panic!("Unknown environment type: {}. Use 'bouncing_ball' or 'paddle_hitting'", config.env_type);
        }
    }
}
