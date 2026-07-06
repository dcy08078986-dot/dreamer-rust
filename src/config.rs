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
    pub imag_batch: usize,

    pub model_lr: f64,
    pub actor_lr: f64,
    pub critic_lr: f64,

    // ── visual / env ──
    pub image_size: usize,
    pub image_channels: usize,
    pub embed_dim: usize,
    pub render_size: usize,

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
    pub kl_balance_alpha: f32, // weight on KL(sg(post)||prior) (dynamics loss); 1-alpha on KL(post||sg(prior))
    pub reward_scale: f32,
    pub entropy_coef: f64,

    // ── slot attention / object-centric ──
    pub use_slot_attention: bool,   // enable slot attention encoder
    pub num_slots: usize,            // K slots (ball, background, ...)
    pub slot_dim: usize,             // slot feature dimension
    pub slot_iterations: usize,      // attention refinement iterations
    pub motion_lambda: f32,          // saliency amplification for motion-weighted recon
    pub motion_wmax: f32,            // cap on per-pixel motion weight (see tools::motion_weights)
    pub use_motion_loss: bool,       // enable motion-weighted recon loss
    pub decode_from_mean: bool,      // decode posterior mean (not sample) for recon
    pub overfit_batch: bool,         // diagnostic: train on ONE fixed batch (capacity check)
    pub slot_mlp_layers: usize,      // MLP layers in slot refinement
    pub use_slot_interaction: bool,  // cross-slot attention in the transition
    pub slot_ctx_dim: usize,         // dimension of cross-slot interaction context

    // ── reconstruction-free latent objective (Proposal C) ──
    pub use_latent_objective: bool,  // replace recon gradient with latent self-prediction
    pub latent_pred_scale: f32,      // weight on next-slot-embedding prediction loss
    pub lambda_bt: f32,              // weight on Barlow-Twins redundancy reduction
    pub bt_beta: f32,                // off-diagonal weight inside the BT loss
    pub probe_interval: usize,       // train the detached probe decoder every N wm updates

    // ── slot-ensemble exploration (Proposal D) ──
    pub use_ensemble_exploration: bool,
    pub ensemble_size: usize,

    // ── decoder architecture ──
    pub decoder_type: String,         // "hybrid" | "broadcast" | "gaussian" | "sbd"
    pub decoder_hidden_dim: usize,    // hidden dimension for spatial broadcast MLP
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
            imag_batch: 16,

            model_lr: 3e-4,
            actor_lr: 8e-4,
            critic_lr: 8e-4,

            image_size: 64,  // Fixed: must match actual render size for spatial consistency
            image_channels: 3,
            embed_dim: 128,
            render_size: 64,

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

            free_nats: 0.5,
            kl_scale: 0.8,
            kl_balance_alpha: 0.8,
            reward_scale: 0.5,
            entropy_coef: 1e-3,

            // slot attention defaults
            use_slot_attention: true,
            num_slots: 3,
            slot_dim: 64,
            slot_iterations: 3,
            motion_lambda: 8.0,
            motion_wmax: 50.0,
            use_motion_loss: true,
            decode_from_mean: false,
            overfit_batch: false,
            slot_mlp_layers: 1,
            use_slot_interaction: false,
            slot_ctx_dim: 32,

            use_latent_objective: false,
            latent_pred_scale: 1.0,
            lambda_bt: 0.1,
            bt_beta: 5e-3,
            probe_interval: 5,

            use_ensemble_exploration: true,
            ensemble_size: 3,

            // Hybrid is the validated default (decoder fit test: ball MSE < 0.02);
            // override via DREAMER_DECODER=broadcast|gaussian|sbd for ablations.
            decoder_type: "hybrid".to_string(),
            decoder_hidden_dim: 256,
        }
    }
}
