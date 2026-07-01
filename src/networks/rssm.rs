use burn::module::Module;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softplus;
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

/// =========================
/// RSSM Model
/// =========================
#[derive(Module, Debug)]
pub struct RSSM<B: Backend> {
    pub gru: Gru<B>,

    pub prior: Linear<B>,  // h -> (mean, std)
    pub post: Linear<B>,   // (h, obs) -> (mean, std)

    pub deter_size: usize,
    pub stoch_size: usize,
}

impl<B: Backend> RSSM<B> {
    /// init
    pub fn init(
        device: &B::Device,
        deter: usize,
        stoch: usize,
        action: usize,
        obs: usize,
    ) -> Self {
        Self {
            gru: GruConfig::new(
                stoch + action,
                deter,
                true,
            )
            .init(device),

            prior: LinearConfig::new(deter, stoch * 2)
                .init(device),

            post: LinearConfig::new(deter + obs, stoch * 2)
                .init(device),

            deter_size: deter,
            stoch_size: stoch,
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
    ) -> Tensor<B, 2> {
        // concatenate stoch and action, then add sequence dimension
        let x = Tensor::cat(
            vec![state.stoch.clone(), action],
            1,
        ).unsqueeze_dim(1); // [B, 1, F]

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
        let stats = self.prior.forward(deter);
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
        let stats = self.post.forward(
            Tensor::cat(vec![deter, obs], 1),
        );

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
        self.gru_step(state, action)
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