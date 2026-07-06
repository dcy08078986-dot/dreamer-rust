use burn::module::Module;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{relu, softplus};
use burn::tensor::{backend::Backend, Tensor, Distribution};

/// =========================
/// RSSM State (FULL)
/// =========================
#[derive(Clone, Debug)]
pub struct RSSMState<B: Backend> {
    pub deter: Tensor<B, 2>, // h_t
    pub stoch: Tensor<B, 2>, // z_t
    pub mean: Tensor<B, 2>,  // q or p mean
    pub std: Tensor<B, 2>,   // q or p std
}

impl<B: Backend> RSSMState<B> {
    /// Detached copy — cuts the autodiff graph (rollouts, collection).
    pub fn detach(&self) -> Self {
        Self {
            deter: self.deter.clone().detach(),
            stoch: self.stoch.clone().detach(),
            mean: self.mean.clone().detach(),
            std: self.std.clone().detach(),
        }
    }
}

/// =========================
/// RSSM Model
/// =========================
#[derive(Module, Debug)]
pub struct RSSM<B: Backend> {
    pub gru: Gru<B>,

    pub prior_h: Linear<B>, // h -> hidden
    pub prior: Linear<B>,   // hidden -> (mean, std)
    pub post_h: Linear<B>,  // (h, obs) -> hidden
    pub post: Linear<B>,    // hidden -> (mean, std)

    /// Normalizes the observation embedding entering the posterior. Without it,
    /// unbounded embeddings produce large posterior means the prior cannot match,
    /// inflating the KL by orders of magnitude early in training.
    pub norm_obs: LayerNorm<B>,

    pub deter_size: usize,
    pub stoch_size: usize,
    pub ctx_size: usize,   // extra context input to the GRU (0 = disabled)
}

const RSSM_HIDDEN: usize = 128;

impl<B: Backend> RSSM<B> {
    /// init (no interaction context)
    pub fn init(
        device: &B::Device,
        deter: usize,
        stoch: usize,
        action: usize,
        obs: usize,
    ) -> Self {
        Self::init_with_ctx(device, deter, stoch, action, obs, 0)
    }

    /// init with an extra context input to the transition GRU. Used by the
    /// slot RSSM to feed cross-slot interaction features into each slot's dynamics.
    pub fn init_with_ctx(
        device: &B::Device,
        deter: usize,
        stoch: usize,
        action: usize,
        obs: usize,
        ctx: usize,
    ) -> Self {
        Self {
            gru: GruConfig::new(
                stoch + action + ctx,
                deter,
                true,
            )
            .init(device),

            prior_h: LinearConfig::new(deter, RSSM_HIDDEN).init(device),
            prior: LinearConfig::new(RSSM_HIDDEN, stoch * 2).init(device),
            post_h: LinearConfig::new(deter + obs, RSSM_HIDDEN).init(device),
            post: LinearConfig::new(RSSM_HIDDEN, stoch * 2).init(device),
            norm_obs: LayerNormConfig::new(obs).init(device),

            deter_size: deter,
            stoch_size: stoch,
            ctx_size: ctx,
        }
    }

    /// init state (all zeros)
    pub fn init_state(
        &self,
        batch: usize,
        device: &B::Device,
    ) -> RSSMState<B> {
        RSSMState {
            deter: Tensor::zeros([batch, self.deter_size], device),
            stoch: Tensor::zeros([batch, self.stoch_size], device),
            mean: Tensor::zeros([batch, self.stoch_size], device),
            std: Tensor::zeros([batch, self.stoch_size], device),
        }
    }

    /// =========================
    /// GRU step (return h_t)
    /// =========================
    fn gru_step(
        &self,
        state: &RSSMState<B>,
        action: Tensor<B, 2>,
        ctx: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        // concatenate stoch, action (and optional interaction context),
        // then add sequence dimension
        let mut parts = vec![state.stoch.clone(), action];
        if self.ctx_size > 0 {
            let c = ctx.unwrap_or_else(|| {
                Tensor::zeros([state.stoch.dims()[0], self.ctx_size], &state.stoch.device())
            });
            parts.push(c);
        }
        let x = Tensor::cat(parts, 1).unsqueeze_dim(1); // [B, 1, F]

        // GRU forward: returns output tensor of shape [B, T, H]
        // Burn newer API: output only, no hidden tuple
        let out = self.gru.forward(
            x,
            Some(state.deter.clone()),
        );

        // squeeze time dimension to get [B, H]
        out.squeeze(1)
    }

    /// =========================
    /// PRIOR: p(z_t | h_t)
    /// =========================
    fn prior(
        &self,
        deter: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = relu(self.prior_h.forward(deter));
        let stats = self.prior.forward(h);
        let chunks = stats.chunk(2, 1);

        let mean = chunks[0].clone();
        // std = softplus + epsilon (more stable than exp)
        let std = softplus(chunks[1].clone(), 20.0) + 1e-4;

        (mean, std)
    }

    /// =========================
    /// POSTERIOR: q(z_t | h_t, o_t)
    /// =========================
    fn post(
        &self,
        deter: Tensor<B, 2>,
        obs: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let obs_n = self.norm_obs.forward(obs);
        let h = relu(self.post_h.forward(Tensor::cat(vec![deter, obs_n], 1)));
        let stats = self.post.forward(h);

        let chunks = stats.chunk(2, 1);

        let mean = chunks[0].clone();
        let std = softplus(chunks[1].clone(), 20.0)+1e-4;

        (mean, std)
    }

    /// =========================
    /// Reparameterized sampling
    /// =========================
    fn sample(
        &self,
        mean: Tensor<B, 2>,
        std: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let eps = Tensor::random(
            mean.shape(),
            Distribution::Normal(0.0, 1.0),
            &mean.device(),
        );

        mean + eps * std
    }

    /// =========================
    /// GRU step: compute h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    /// This is the shared deterministic transition — prior and posterior
    /// MUST use the same h_t for a given timestep.
    /// =========================
    pub fn get_deter(
        &self,
        state: &RSSMState<B>,
        action: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.gru_step(state, action, None)
    }

    /// Like get_deter, but with an explicit interaction context (slot RSSM).
    pub fn get_deter_ctx(
        &self,
        state: &RSSMState<B>,
        action: Tensor<B, 2>,
        ctx: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.gru_step(state, action, Some(ctx))
    }

    /// =========================
    /// Build a state from prior: p(z_t | h_t)
    /// =========================
    pub fn prior_state(
        &self,
        deter: Tensor<B, 2>,
    ) -> RSSMState<B> {
        let (mean, std) = self.prior(deter.clone());
        let stoch = self.sample(mean.clone(), std.clone());
        RSSMState { deter, stoch, mean, std }
    }

    /// =========================
    /// Build a state from posterior: q(z_t | h_t, o_t)
    /// =========================
    pub fn post_state(
        &self,
        deter: Tensor<B, 2>,
        obs: Tensor<B, 2>,
    ) -> RSSMState<B> {
        let (mean, std) = self.post(deter.clone(), obs);
        let stoch = self.sample(mean.clone(), std.clone());
        RSSMState { deter, stoch, mean, std }
    }

    /// =========================
    /// Imagination step (model rollout) — convenience wrapper
    /// =========================
    pub fn img_step(
        &self,
        state: &RSSMState<B>,
        action: Tensor<B, 2>,
    ) -> RSSMState<B> {
        let deter = self.get_deter(state, action);
        self.prior_state(deter)
    }

    /// =========================
    /// Observation step (inference) — convenience wrapper
    /// =========================
    pub fn obs_step(
        &self,
        state: &RSSMState<B>,
        obs: Tensor<B, 2>,
        action: Tensor<B, 2>,
    ) -> RSSMState<B> {
        let deter = self.get_deter(state, action);
        self.post_state(deter, obs)
    }
}