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
use envs::maze::MazeEnv;
use envs::pendulum::PendulumEnv;
use envs::visual_navigation::VisualNavigation;
use envs::bouncing_ball::BouncingBall;

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
        "pendulum" => {
            println!("Creating {} parallel Pendulum environments...", num_envs);
            let mut envs: Vec<PendulumEnv> = (0..num_envs)
                .map(|i| PendulumEnv::new(
                    config.env_max_steps,
                    config.action_dim,
                    config.image_channels,
                    config.image_size,
                    config.seed + i as u64
                ))
                .collect();
            train::run::<Backend, PendulumEnv>(device, config, &mut envs);
        }
        "visual_navigation" => {
            println!("Creating {} parallel VisualNavigation environments...", num_envs);
            let mut envs: Vec<VisualNavigation> = (0..num_envs)
                .map(|i| VisualNavigation::new(
                    config.env_max_steps,
                    config.action_dim,
                    config.image_channels,
                    config.image_size,
                    config.num_obstacles,
                    config.seed + i as u64
                ))
                .collect();
            train::run::<Backend, VisualNavigation>(device, config, &mut envs);
        }
        "maze" => {
            println!("Creating {} parallel Maze environments...", num_envs);
            let mut envs: Vec<MazeEnv> = (0..num_envs)
                .map(|i| MazeEnv::new(15, 15, config.env_max_steps, config.image_size, config.seed + i as u64))
                .collect();
            train::run::<Backend, MazeEnv>(device, config, &mut envs);
        }
        _ => {
            panic!("Unknown environment type: {}. Use 'bouncing_ball', 'pendulum', 'visual_navigation', or 'maze'", config.env_type);
        }
    }
}
