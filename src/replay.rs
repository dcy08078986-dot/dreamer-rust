#![allow(dead_code)]

use burn::tensor::{backend::Backend, Tensor};

/// 单条 episode 数据
#[derive(Clone, Debug)]
pub struct Episode<B: Backend> {
    pub obs: Vec<Tensor<B, 2>>,
    pub action: Vec<Tensor<B, 2>>,
    pub reward: Vec<Tensor<B, 2>>,
    pub done: Vec<Tensor<B, 2>>,
}

/// batch 输出结构
#[derive(Debug)]
pub struct Batch<B: Backend> {
    pub obs: Tensor<B, 3>,     // [B, T, obs]
    pub action: Tensor<B, 3>,  // [B, T, act]
    pub reward: Tensor<B, 3>,  // [B, T, 1]
    pub done: Tensor<B, 3>,    // [B, T, 1]
}

/// Replay Buffer (sequence version for Dreamer)
pub struct ReplayBuffer<B: Backend> {
    episodes: Vec<Episode<B>>,
    capacity: usize,
}

impl<B: Backend> ReplayBuffer<B> {
    /// create buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            episodes: Vec::new(),
            capacity,
        }
    }

    /// store episode
    pub fn push(&mut self, episode: Episode<B>) {
        if self.episodes.len() >= self.capacity {
            self.episodes.remove(0);
        }
        self.episodes.push(episode);
    }

    /// number of stored episodes
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// sample sequence batch [B, T, ...]
    pub fn sample(
        &self,
        batch_size: usize,
        seq_len: usize,
        _device: &B::Device,
    ) -> Batch<B> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // batch collectors (IMPORTANT: explicit type)
        let mut obs_batch: Vec<Tensor<B, 3>> = Vec::new();
        let mut act_batch: Vec<Tensor<B, 3>> = Vec::new();
        let mut rew_batch: Vec<Tensor<B, 3>> = Vec::new();
        let mut done_batch: Vec<Tensor<B, 3>> = Vec::new();

        // Filter to episodes long enough
        let valid_eps: Vec<_> = self.episodes.iter()
            .filter(|ep| ep.obs.len() >= seq_len + 1)
            .collect();
        if valid_eps.is_empty() { return Batch { obs: Tensor::zeros([1, 1, 1], _device), action: Tensor::zeros([1, 1, 1], _device), reward: Tensor::zeros([1, 1, 1], _device), done: Tensor::zeros([1, 1, 1], _device) }; }

        for _ in 0..batch_size {
            let ep = valid_eps[rng.gen_range(0..valid_eps.len())];
            let start = rng.gen_range(0..(ep.obs.len() - seq_len));

            let obs_slice = &ep.obs[start..start + seq_len];
            let act_slice = &ep.action[start..start + seq_len];
            let rew_slice = &ep.reward[start..start + seq_len];
            let done_slice = &ep.done[start..start + seq_len];

            // [T, ...]
            let obs_t = Tensor::stack(obs_slice.to_vec(), 1);
            let act_t = Tensor::stack(act_slice.to_vec(), 1);
            let rew_t = Tensor::stack(rew_slice.to_vec(), 1);
            let done_t = Tensor::stack(done_slice.to_vec(), 1);

            obs_batch.push(obs_t);
            act_batch.push(act_t);
            rew_batch.push(rew_t);
            done_batch.push(done_t);
        }

        // [B, T, ...]
        Batch {
            obs: Tensor::cat(obs_batch, 0),
            action: Tensor::cat(act_batch, 0),
            reward: Tensor::cat(rew_batch, 0),
            done: Tensor::cat(done_batch, 0),
        }
    }
}