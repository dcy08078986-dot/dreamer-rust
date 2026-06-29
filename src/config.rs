#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Config {
    pub seed: u64,

    pub obs_dim: usize,
    pub action_dim: usize,

    pub batch_size: usize,
    pub batch_length: usize,

    pub deter_size: usize,
    pub stoch_size: usize,
    pub hidden_size: usize,

    pub mlp_layers: usize,
    pub units: usize,

    pub discount: f32,
    pub lambda_: f32,
    pub imag_horizon: usize,

    pub model_lr: f64,
    pub actor_lr: f64,
    pub critic_lr: f64,

    // ── visual / env ──
    pub image_size: usize,
    pub image_channels: usize,
    pub embed_dim: usize,
    pub render_size: usize,
    pub num_obstacles: usize,

    // ── training control ──
    pub replay_capacity: usize,
    pub max_episodes: usize,
    pub env_max_steps: usize,
    pub video_interval: usize,

    // ── loss weights ──
    pub free_nats: f32,
    pub kl_scale: f32,
    pub reward_scale: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,

            obs_dim: 8,
            action_dim: 2,

            batch_size: 2,
            batch_length: 4,

            deter_size: 128,
            stoch_size: 32,
            hidden_size: 128,

            mlp_layers: 2,
            units: 128,

            discount: 0.99,
            lambda_: 0.95,
            imag_horizon: 15,

            model_lr: 1e-4,
            actor_lr: 3e-5,
            critic_lr: 3e-5,

            image_size: 64,
            image_channels: 3,
            embed_dim: 256,
            render_size: 64,
            num_obstacles: 3,

            replay_capacity: 100,
            max_episodes: 500,
            env_max_steps: 200,
            video_interval: 50,

            free_nats: 1.0,
            kl_scale: 0.8,
            reward_scale: 0.5,
        }
    }
}