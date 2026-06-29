mod config;
mod tools;
mod replay;
mod train;
mod networks;
mod agent;
mod envs;
mod video;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use envs::visual_navigation::VisualNavigation;

// ── Backend ──
// GPU: fast, needs enough VRAM (2-4GB). If OOM, reduce batch or image size.
// CPU (slower): `Autodiff<NdArray>` with `NdArrayDevice::default()`
//   requires Cargo.toml: `features = ["ndarray", "train"]`

fn main() {
    type Backend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    let config = config::Config::default();

    let mut env = VisualNavigation::new(
        config.env_max_steps,
        config.action_dim,
        config.image_channels,
        config.image_size,
        config.num_obstacles,
        config.seed,
    );

    train::run::<Backend, VisualNavigation>(device, config, &mut env);
}
