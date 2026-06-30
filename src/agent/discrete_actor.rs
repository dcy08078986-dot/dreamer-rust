use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::{backend::Backend, Tensor, Int, Distribution};
use crate::networks::rssm::RSSMState;

#[derive(Module, Debug)]
pub struct DiscreteActor<B: Backend> {
    pub hidden: Linear<B>,
    pub logits: Linear<B>,
    num_actions: usize,
}

impl<B: Backend> DiscreteActor<B> {
    pub fn init(
        device: &B::Device,
        deter_size: usize,
        stoch_size: usize,
        num_actions: usize,
    ) -> Self {
        Self {
            hidden: LinearConfig::new(deter_size + stoch_size, 256).init(device),
            logits: LinearConfig::new(256, num_actions).init(device),
            num_actions,
        }
    }

    pub fn forward(&self, state: &RSSMState<B>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = Tensor::cat(vec![state.deter.clone(), state.stoch.clone()], 1);
        let h = relu(self.hidden.forward(x));
        let logits = self.logits.forward(h);
        let probs = softmax(logits.clone(), 1);
        (logits, probs)
    }

    /// Sample action: returns `(action_index_tensor [B], log_prob [B])`.
    pub fn sample(&self, state: &RSSMState<B>) -> (Tensor<B, 1, Int>, Tensor<B, 1>) {
        let (_logits, probs) = self.forward(state);
        let [b, k] = probs.dims();

        let noise: Tensor<B, 2> = Tensor::random([b, k], Distribution::Normal(0.0, 1.0), &probs.device());
        let noisy = probs.clone() + noise.mul_scalar(0.1);
        let indices: Tensor<B, 1, Int> = noisy.argmax(1).squeeze(1);

        let idx_data = indices.clone().to_data();
        let raw_val = *idx_data.as_slice::<i32>().unwrap().first().unwrap_or(&0);
        let clamped_val = raw_val.clamp(0, (self.num_actions - 1) as i32) as usize;
        let lp = probs.clone().clamp_min(1e-8).log()
            .narrow(0, 0, 1)
            .narrow(1, clamped_val, 1)
            .squeeze(1);

        (indices, lp)
    }
}
