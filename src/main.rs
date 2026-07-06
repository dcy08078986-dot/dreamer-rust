#![recursion_limit = "256"]

mod config;
mod tools;
mod replay;
mod train;
mod train_oc;
mod networks;
mod agent;
mod envs;
mod video;
mod diagnostics;

use burn::backend::{Autodiff, NdArray};
use envs::bouncing_ball::BouncingBall;
use envs::paddle_hitting::PaddleHitting;

/// Env-var config overrides so ablations run without code edits, e.g.:
///   DREAMER_EPISODES=200 DREAMER_LATENT=1 cargo run --release
fn apply_env_overrides(config: &mut config::Config) {
    fn get<T: std::str::FromStr>(key: &str) -> Option<T> {
        std::env::var(key).ok().and_then(|v| v.parse().ok())
    }
    fn get_bool(key: &str) -> Option<bool> {
        get::<i32>(key).map(|v| v != 0)
    }
    if let Some(v) = get("DREAMER_EPISODES") { config.max_episodes = v; }
    if let Some(v) = get("DREAMER_SEED") { config.seed = v; }
    if let Some(v) = get("DREAMER_SLOTS") { config.num_slots = v; }
    if let Some(v) = get("DREAMER_VIDEO_INTERVAL") { config.video_interval = v; }
    if let Some(v) = get::<String>("DREAMER_ENV") { config.env_type = v; }
    if let Some(v) = get_bool("DREAMER_MONOLITHIC") { config.use_slot_attention = !v; }
    if let Some(v) = get_bool("DREAMER_MOTION") { config.use_motion_loss = v; }
    if let Some(v) = get_bool("DREAMER_OVERFIT") { config.overfit_batch = v; }
    if let Some(v) = get("DREAMER_MOTION_WMAX") { config.motion_wmax = v; }
    if let Some(v) = get("DREAMER_KL_SCALE") { config.kl_scale = v; }
    if let Some(v) = get("DREAMER_FREE_NATS") { config.free_nats = v; }
    if let Some(v) = get("DREAMER_STOCH") { config.stoch_size = v; }
    if let Some(v) = get_bool("DREAMER_DECODE_MEAN") { config.decode_from_mean = v; }
    if let Some(v) = get("DREAMER_LR") { config.model_lr = v; }
    if let Some(v) = get_bool("DREAMER_INTERACTION") { config.use_slot_interaction = v; }
    if let Some(v) = get_bool("DREAMER_LATENT") { config.use_latent_objective = v; }
    if let Some(v) = get_bool("DREAMER_ENSEMBLE") { config.use_ensemble_exploration = v; }
    if let Some(v) = get::<String>("DREAMER_DECODER") { config.decoder_type = v; }
    if let Some(v) = get_bool("DREAMER_SPATIAL_BROADCAST") { if v { config.decoder_type = "sbd".to_string(); } }
}

fn main() {
    type Backend = Autodiff<NdArray>;
    let device = Default::default();
    let mut config = config::Config::default();
    apply_env_overrides(&mut config);

    if let Ok(val) = std::env::var("DREAMER_DECODER_TEST") {
        match val.as_str() {
            "gaussian" | "gaussian_weighted" => diagnostics::gaussian_decoder_fit_test::<Backend>(device),
            "hybrid" | "hybrid_weighted" => diagnostics::hybrid_decoder_fit_test::<Backend>(device),
            "slot_ae" | "slot_ae_weighted" => diagnostics::slot_ae_fit_test::<Backend>(device),
            "deconv" => diagnostics::deconv_decoder_fit_test::<Backend>(device),
            _ => diagnostics::decoder_fit_test::<Backend>(device),
        }
        return;
    }

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
            if config.use_slot_attention {
                println!("Using Object-Centric World Model ({} slots)", config.num_slots);
                train_oc::run_oc::<Backend, BouncingBall>(device, config, &mut envs);
            } else {
                train::run::<Backend, BouncingBall>(device, config, &mut envs);
            }
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
            if config.use_slot_attention {
                println!("Using Object-Centric World Model ({} slots)", config.num_slots);
                train_oc::run_oc::<Backend, PaddleHitting>(device, config, &mut envs);
            } else {
                train::run::<Backend, PaddleHitting>(device, config, &mut envs);
            }
        }
        _ => {
            panic!("Unknown environment type: {}. Use 'bouncing_ball' or 'paddle_hitting'", config.env_type);
        }
    }
}
