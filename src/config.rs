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

    // ── discrete RSSM ──
    pub num_classes: usize,

    // ── attention encoder ──
    pub d_model: usize,
    pub num_heads: usize,
    pub attn_layers: usize,

    // ── exploration ──
    pub exploration_scale: f32,

    // ── training control ──
    pub replay_capacity: usize,
    pub max_episodes: usize,
    pub env_max_steps: usize,
    pub video_interval: usize,
    pub num_envs: usize,

    // ── checkpoint ──
    pub checkpoint_dir: String,
    pub save_interval: usize,
    pub load_checkpoint: Option<String>,

    // ── environment selection ──
    pub env_type: String,

    // ── loss weights ──
    pub free_nats: f32,
    pub kl_scale: f32,
    pub reward_scale: f32,
    pub kl_low_threshold: f32,
    pub kl_high_threshold: f32,
    pub kl_weight_low: f32,
    pub kl_weight_high: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,

            obs_dim: 8,
            action_dim: 1,   // bouncing ball: 水平力控制

            batch_size: 2,
            batch_length: 8,

            deter_size: 64,
            stoch_size: 16,
            hidden_size: 128,

            mlp_layers: 2,
            units: 128,

            discount: 0.99,
            lambda_: 0.95,
            imag_horizon: 15,

            model_lr: 3e-4,
            actor_lr: 8e-4,
            critic_lr: 8e-4,

            image_size: 48,
            image_channels: 3,
            embed_dim: 128,
            render_size: 64,
            num_obstacles: 0,

            num_classes: 32,

            d_model: 64,
            num_heads: 2,
            attn_layers: 1,

            exploration_scale: 0.02,

            replay_capacity: 200,
            max_episodes: 500,
            env_max_steps: 200,
            video_interval: 50,
            num_envs: 4,

            checkpoint_dir: "checkpoints".to_string(),
            save_interval: 50,
            load_checkpoint: None,

            env_type: "bouncing_ball".to_string(),

            free_nats: 1.0,
            kl_scale: 0.8,
            reward_scale: 0.5,
            kl_low_threshold: 0.5,
            kl_high_threshold: 5.0,
            kl_weight_low: 1.2,
            kl_weight_high: 0.5,
        }
    }
}
