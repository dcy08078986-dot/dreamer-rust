use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use crate::config::Config;
use crate::envs::Environment;
use crate::replay::{Episode, ReplayBuffer};
use crate::agent::world_model::WorldModel;
use crate::agent::actor_critic::{Actor, Critic};
use crate::video::{save_frame, frames_to_mp4, make_comparison_frame};
use crate::tools::symlog;

pub fn run<B: AutodiffBackend, E: Environment>(
    device: B::Device,
    config: Config,
    envs: &mut [E],
) {
    let num_envs = envs.len();
    let [c, h, w] = envs[0].obs_shape();
    let flat_dim = c * h * w;
    let total_eps = config.max_episodes;
    let rounds = total_eps / num_envs;
    let act_dim = envs[0].action_dim();  // use env's action dim, not config

    println!("=== Dreamer Discrete + CarRacing ({} envs, {} rounds) ===", num_envs, rounds);
    println!("Obs: {}x{}x{} | Action: {} | Embed: {} | Classes: {}",
        c, h, w, envs[0].action_dim(), config.embed_dim, config.num_classes);

    let mut world_model = WorldModel::<B>::init(
        &device, config.deter_size, config.stoch_size, act_dim,
        config.embed_dim, config.image_channels, config.image_size,
    );
    let mut actor = Actor::<B>::init(
        &device, config.deter_size, config.stoch_size, act_dim,
    );
    let mut critic = Critic::<B>::init(
        &device, config.deter_size, config.stoch_size,
    );
    let mut model_optim = AdamConfig::new().init::<B, WorldModel<B>>();
    let mut actor_optim = AdamConfig::new().init::<B, Actor<B>>();
    let mut critic_optim = AdamConfig::new().init::<B, Critic<B>>();

    if let Some(ref load_ep) = config.load_checkpoint {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        let prefix = |name: &str| format!("{}/{}_{}", config.checkpoint_dir, name, load_ep);
        world_model = world_model.load_file(prefix("world_model"), &recorder, &device).unwrap();
        actor = actor.load_file(prefix("actor"), &recorder, &device).unwrap();
        critic = critic.load_file(prefix("critic"), &recorder, &device).unwrap();
        println!("Loaded checkpoint from episode {}", load_ep);
    }

    let mut replay = ReplayBuffer::<B>::new(config.replay_capacity);
    let mut reference_ep: Option<Episode<B>> = None;
    let mut ep_counter: usize = 0;
    let effective_batch = (config.batch_size * num_envs).min(config.batch_size.max(1));

    for round in 0..rounds {
        let mut round_reward: f32 = 0.0;
        let mut round_steps: usize = 0;

        // ── 1. Collect episodes ──
        for env in envs.iter_mut() {
            let obs = env.reset::<B>(&device);
            let obs_flat = obs.reshape([1, flat_dim]);
            let init_s = world_model.init_state(1, &device);
            let zero_action = Tensor::zeros([1, act_dim], &device);
            let mut cur_state = world_model.obs_step(&init_s, obs_flat.clone(), zero_action);

            let mut ep_obs: Vec<Tensor<B, 2>> = vec![obs_flat];
            let mut ep_act: Vec<Tensor<B, 2>> = Vec::new();
            let mut ep_rew: Vec<Tensor<B, 2>> = Vec::new();
            let mut ep_done: Vec<Tensor<B, 2>> = Vec::new();
            let mut total_reward: f32 = 0.0;
            let mut steps = 0;

            for _t in 0..env.max_steps() {
                let (action, _logp) = actor.sample(&cur_state);
                let action_data = action.to_data();
                let act_vals = action_data.as_slice::<f32>().unwrap().to_vec();

                let (next_obs, reward, done) = env.step::<B>(&act_vals, &device);
                let next_obs_flat = next_obs.reshape([1, flat_dim]);

                let _mean_rew = world_model.predict_reward(&cur_state);
                let uncertainty_val = 0.0_f32;

                ep_act.push(action.clone());
                ep_rew.push(Tensor::full([1, 1], reward, &device));
                ep_done.push(Tensor::full([1, 1], if done { 1.0f32 } else { 0.0 }, &device));
                ep_obs.push(next_obs_flat.clone());
                total_reward += reward + config.exploration_scale * uncertainty_val;
                steps += 1;
                if done { break; }
                cur_state = world_model.obs_step(&cur_state, next_obs_flat, action);
            }

            let ep_obs_trimmed = ep_obs[..steps].to_vec();
            replay.push(Episode {
                obs: ep_obs_trimmed.clone(), action: ep_act.clone(),
                reward: ep_rew.clone(), done: ep_done.clone(),
            });
            if reference_ep.is_none() {
                reference_ep = Some(Episode {
                    obs: ep_obs_trimmed.clone(), action: ep_act.clone(),
                    reward: ep_rew.clone(), done: ep_done.clone(),
                });
            }
            round_reward += total_reward;
            round_steps = round_steps.max(steps);
            if ep_counter % 10 == 0 {
                println!("ep {:4} | steps {:3} | reward {:+.4}", ep_counter, steps, total_reward);
            }
            ep_counter += 1;
        }

        // ── 2. Train world model ──
        let mut model_loss_val: f32 = 0.0;
        if replay.len() >= 5 {
            for _ in 0..(5 * num_envs) {
                let batch = replay.sample(effective_batch.min(replay.len()),
                    config.batch_length.min(round_steps), &device);
                let reward_2d = batch.reward.clone().squeeze(2);
                let loss = world_model.train_step(
                    batch.obs,
                    batch.action,
                    reward_2d,
                    config.kl_low_threshold,
                    config.kl_high_threshold,
                    config.kl_weight_low,
                    config.kl_weight_high,
                    config.kl_scale,
                );
                let loss_data = loss.clone().to_data();
                model_loss_val = loss_data.as_slice::<f32>().unwrap()[0];
                let grads = loss.backward();
                let grads_wm = GradientsParams::from_grads(grads, &world_model);
                world_model = model_optim.step(config.model_lr, world_model, grads_wm);
            }
        }

        // ── 3. Train actor-critic ──
        let mut actor_loss_val: f32 = 0.0;
        let mut critic_loss_val: f32 = 0.0;
        if replay.len() >= 5 {
            let imag_horizon = config.imag_horizon.min(15);
            let mut imag_states: Vec<crate::networks::rssm::RSSMState<B>> = Vec::new();
            let mut imag_actions: Vec<Tensor<B, 2>> = Vec::new();
            let mut imag_rewards: Vec<Tensor<B, 1>> = Vec::new();

            // 从replay buffer采样真实状态作为起点
            // 注意：需要detach并重新通过模型以启用梯度追踪
            let start_batch = replay.sample(1, 1, &device);
            let start_obs = start_batch.obs.narrow(1, 0, 1).squeeze(1).detach();
            let start_act = start_batch.action.narrow(1, 0, 1).squeeze(1).detach();

            // 通过world model重新编码以启用梯度
            let start_obs_emb = world_model.encode(start_obs);
            let init_state = world_model.init_state(1, &device);
            let mut img_state = world_model.rssm.obs_step(&init_state, start_obs_emb, start_act);

            for _t in 0..imag_horizon {
                let (action, _log_prob) = actor.sample(&img_state);
                let reward = world_model.predict_reward(&img_state);
                let next_state = world_model.img_step(&img_state, action.clone());
                imag_states.push(img_state);
                imag_actions.push(action.clone());
                imag_rewards.push(reward.clone());
                img_state = next_state;
            }
            imag_states.push(img_state);

            if !imag_actions.is_empty() {
                let rewards_tensor: Tensor<B, 2> = Tensor::stack(imag_rewards.clone(), 1);
                let rewards_detached = rewards_tensor.detach();
                let gamma = config.discount as f64;
                let mut critic_values: Vec<Tensor<B, 1>> = Vec::new();
                for state in imag_states.iter() {
                    let s = crate::networks::rssm::RSSMState {
                        deter: state.deter.clone().detach(),
                        stoch: state.stoch.clone().detach(),
                        mean: state.mean.clone().detach(),
                        std: state.std.clone().detach(),
                    };
                    critic_values.push(critic.forward(&s));
                }
                let horizon = imag_actions.len();
                let values_full: Tensor<B, 2> = Tensor::stack(critic_values.clone(), 1);

                let mut actor_loss = Tensor::zeros([1], &device);
                for t in 0..horizon {
                    let (mean, log_std) = actor.forward(&imag_states[t]);
                    let action = imag_actions[t].clone();
                    let std = log_std.exp().add_scalar(1e-4);
                    let var = std.clone().mul(std.clone());
                    let diff = action.clone() - mean;
                    let term1 = diff.clone().mul(diff) / var.add_scalar(1e-8);
                    let term2 = std.mul_scalar(2.0).log();
                    let sum = (term1 + term2).add_scalar(1.837877f64);
                    let mut log_prob: Tensor<B, 1> = sum.sum_dim(1).mul_scalar(-0.5).squeeze(1);
                    let action_tanh = action.tanh();
                    let tanh_grad = action_tanh.clone().mul(action_tanh.clone()).neg().add_scalar(1.0);
                    log_prob = log_prob - tanh_grad.clamp_min(1e-8).log().sum_dim(1).squeeze(1);

                    // 使用symlog变换处理奖励和值函数
                    let r_t = rewards_detached.clone().narrow(1, t, 1).squeeze(1);
                    let r_t_symlog = symlog(r_t.clone());
                    let v_t = critic_values[t].clone();
                    let v_next = critic_values[t + 1].clone();

                    // 在symlog空间计算优势
                    let adv = (r_t_symlog + v_next.mul_scalar(gamma) - v_t).detach();
                    actor_loss = actor_loss + log_prob.neg().mul(adv).mean().reshape([1]);
                }
                actor_loss = actor_loss.div_scalar(horizon as f64);

                // Critic在symlog空间训练
                let val_now = values_full.clone().narrow(1, 0, horizon);
                let val_next = values_full.narrow(1, 1, horizon);
                let rewards_symlog = symlog(rewards_detached.clone());
                let target = rewards_symlog + val_next.mul_scalar(gamma);
                let diff = target - val_now;
                let critic_loss = diff.clone().mul(diff).mean().reshape([1]);

                let c_data = critic_loss.clone().to_data();
                critic_loss_val = c_data.as_slice::<f32>().unwrap()[0];
                let c_grads = critic_loss.backward();
                let grads_c = GradientsParams::from_grads(c_grads, &critic);
                critic = critic_optim.step(config.critic_lr, critic, grads_c);

                let a_data = actor_loss.clone().to_data();
                actor_loss_val = a_data.as_slice::<f32>().unwrap()[0];
                let a_grads = actor_loss.backward();
                let grads_a = GradientsParams::from_grads(a_grads, &actor);
                actor = actor_optim.step(config.actor_lr, actor, grads_a);
            }
        }

        // ── 4. Save checkpoint ──
        let last_ep = ep_counter.saturating_sub(1);
        if ep_counter > 0 && (ep_counter % config.save_interval == 0 || ep_counter >= total_eps) {
            save_checkpoint(&world_model, &actor, &critic, last_ep, &config.checkpoint_dir);
        }

        // ── 5. Video ──
        if config.video_interval > 0
            && (round == 0 || ep_counter % config.video_interval == 0 || ep_counter >= total_eps)
        {
            generate_video::<B>(&world_model, reference_ep.as_ref(), ep_counter - 1, c, h, w);
        }

        // ── 6. Summary ──
        let avg_reward = round_reward / num_envs as f32;
        println!("round {:4} | ep {:4} | avg_reward {:+.4} | model {:.4} | actor {:.4} | critic {:.4}",
            round, last_ep, avg_reward, model_loss_val, actor_loss_val, critic_loss_val);
    }
    println!("=== Done ===");
}

fn generate_video<B: Backend>(wm: &WorldModel<B>, ref_ep: Option<&Episode<B>>, ep_num: usize, c: usize, h: usize, w: usize) {
    let Some(ep) = ref_ep else { return };
    let seq_len = ep.obs.len().min(100);
    if seq_len == 0 { return; }
    let frame_dir = format!("frames/ep_{:04}", ep_num);
    let output_path = format!("output/comparison_ep_{:04}.mp4", ep_num);
    std::fs::create_dir_all(&frame_dir).ok();
    std::fs::create_dir_all("output").ok();
    let device = ep.obs[0].device();
    let mut rs = wm.init_state(1, &device);
    let first = ep.obs[0].clone();
    let zero_action = Tensor::zeros([1, ep.action[0].dims()[1]], &device);
    rs = wm.obs_step(&rs, first, zero_action);

    // 生成左右对比帧：左边真实，右边世界模型重建
    for t in 0..seq_len {
        if t > 0 { rs = wm.img_step(&rs, ep.action[t-1].clone()); }
        let recon = wm.reconstruct(&rs).reshape([c, h, w]);
        let real = ep.obs[t].clone().reshape([c, h, w]);

        // 创建对比帧 [C, H, 2W]
        let comparison = make_comparison_frame::<B>(&real, &recon);
        save_frame::<B>(&comparison, &format!("{}/frame_{:04}.png", frame_dir, t));
    }
    frames_to_mp4(&frame_dir, &output_path, 20);
}

fn save_checkpoint<B: AutodiffBackend>(wm: &WorldModel<B>, actor: &Actor<B>, critic: &Critic<B>, ep_num: usize, dir: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    wm.clone().save_file(format!("{}/world_model_ep_{}", dir, ep_num), &rec).unwrap();
    actor.clone().save_file(format!("{}/actor_ep_{}", dir, ep_num), &rec).unwrap();
    critic.clone().save_file(format!("{}/critic_ep_{}", dir, ep_num), &rec).unwrap();
    println!("Checkpoint saved at episode {} to {}/", ep_num, dir);
}
