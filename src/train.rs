use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use crate::config::Config;
use crate::envs::Environment;
use crate::replay::{Episode, ReplayBuffer};
use crate::agent::world_model::WorldModel;
use crate::agent::actor_critic::{Actor, Critic};
use crate::video::{make_comparison_frame, save_frame, frames_to_mp4};

/// Run the full Dreamer training pipeline with autodiff gradient descent.
pub fn run<B: AutodiffBackend, E: Environment>(
    device: B::Device,
    config: Config,
    env: &mut E,
) {
    let [c, h, w] = env.obs_shape();
    let flat_dim = c * h * w;

    println!("=== Dreamer Visual Navigation (autodiff) ===");
    println!("Obs: {}x{}x{} | Action: {} | Embed: {}",
        c, h, w, env.action_dim(), config.embed_dim);
    println!("RSSM: deter={} stoch={} | LR: model={}, actor={}",
        config.deter_size, config.stoch_size, config.model_lr, config.actor_lr);

    let mut world_model = WorldModel::<B>::init(
        &device,
        config.deter_size,
        config.stoch_size,
        config.action_dim,
        config.embed_dim,
        config.image_channels,
        config.image_size,
    );

    let mut actor = Actor::<B>::init(
        &device,
        config.deter_size,
        config.stoch_size,
        config.action_dim,
    );

    let mut critic = Critic::<B>::init(
        &device,
        config.deter_size,
        config.stoch_size,
    );

    let mut model_optim = AdamConfig::new().init::<B, WorldModel<B>>();
    let mut actor_optim = AdamConfig::new().init::<B, Actor<B>>();
    let mut critic_optim = AdamConfig::new().init::<B, Critic<B>>();

    let mut replay = ReplayBuffer::<B>::new(config.replay_capacity);
    let mut reference_ep: Option<Episode<B>> = None;

    for ep_num in 0..config.max_episodes {
        // ── 1. Collect an episode ──
        let obs = env.reset::<B>(&device);
        let obs_flat = obs.reshape([1, flat_dim]);
        let init_s = world_model.init_state(1, &device);
        let mut cur_state = world_model.obs_step(&init_s, obs_flat.clone());

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

            ep_act.push(action);
            ep_rew.push(Tensor::full([1, 1], reward, &device));
            ep_done.push(Tensor::full([1, 1], if done { 1.0f32 } else { 0.0 }, &device));
            ep_obs.push(next_obs_flat.clone());

            total_reward += reward;
            steps += 1;

            if done { break; }

            cur_state = world_model.obs_step(&cur_state, next_obs_flat);
        }

        let ep_obs_trimmed = ep_obs[..steps].to_vec();

        replay.push(Episode {
            obs: ep_obs_trimmed.clone(),
            action: ep_act.clone(),
            reward: ep_rew.clone(),
            done: ep_done.clone(),
        });

        if reference_ep.is_none() {
            reference_ep = Some(Episode {
                obs: ep_obs_trimmed.clone(),
                action: ep_act.clone(),
                reward: ep_rew.clone(),
                done: ep_done.clone(),
            });
        }

        // ── 2. Train world model ──
        let mut model_loss_val: f32 = 0.0;
        if replay.len() >= 5 {
            for _ in 0..3 {
                let batch = replay.sample(
                    config.batch_size.min(replay.len()),
                    config.batch_length.min(steps),
                    &device,
                );
                let reward_2d = batch.reward.clone().squeeze(2);
                let loss = world_model.train_step(
                    batch.obs,
                    batch.action,
                    reward_2d,
                );
                let loss_data = loss.clone().to_data();
                model_loss_val = loss_data.as_slice::<f32>().unwrap()[0];

                // Backward + update world model
                let grads = loss.backward();
                let grads_wm = GradientsParams::from_grads(grads, &world_model);
                world_model = model_optim.step(config.model_lr, world_model, grads_wm);
            }
        }

        // ── 3. Train actor-critic on imagined trajectories ──
        let mut actor_loss_val: f32 = 0.0;
        let mut critic_loss_val: f32 = 0.0;
        if replay.len() >= 5 {
            let imag_horizon = config.imag_horizon.min(15);
            let mut imag_states: Vec<crate::networks::rssm::RSSMState<B>> = Vec::new();
            let mut imag_actions: Vec<Tensor<B, 2>> = Vec::new();
            let mut imag_rewards: Vec<Tensor<B, 1>> = Vec::new();

            let mut img_state = world_model.init_state(1, &device);

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
                let rewards_tensor: Tensor<B, 2> =
                    Tensor::stack(imag_rewards.clone(), 1);
                let rewards_detached = rewards_tensor.detach();
                let gamma = config.discount as f64;

                // ── Compute critic values (detach world-model states first) ──
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

                // Stack values: [1, H+1]
                let values_full: Tensor<B, 2> = Tensor::stack(critic_values.clone(), 1);

                // ── Actor loss ──
                let mut actor_loss = Tensor::zeros([1], &device);
                for t in 0..horizon {
                    let (mean, log_std) = actor.forward(&imag_states[t]);
                    let action = imag_actions[t].clone();
                    let std = log_std.exp().add_scalar(1e-4);
                    let var = std.clone().powf_scalar(2.0);
                    let diff = action.clone() - mean;
                    let term1 = diff.powf_scalar(2.0) / var.add_scalar(1e-8);
                    let term2 = std.mul_scalar(2.0).log();
                    let sum = (term1 + term2).add_scalar(1.837877f64);
                    let mut log_prob: Tensor<B, 1> = sum.sum_dim(1).mul_scalar(-0.5).squeeze(1);
                    let action_tanh = action.tanh();
                    let tanh_grad = action_tanh.powf_scalar(2.0).neg().add_scalar(1.0);
                    log_prob = log_prob - tanh_grad.clamp_min(1e-8).log().sum_dim(1).squeeze(1);
                    let r_t = rewards_detached.clone().narrow(1, t, 1).squeeze(1);
                    let v_t = critic_values[t].clone();
                    let v_next = critic_values[t+1].clone();
                    let adv = (r_t + v_next.mul_scalar(gamma) - v_t).detach();
                    actor_loss = actor_loss + log_prob.neg().mul(adv).mean().reshape([1]);
                }
                actor_loss = actor_loss.div_scalar(horizon as f64);

                // ── Critic loss: TD(1) via batch tensor ops ──
                let val_now = values_full.clone().narrow(1, 0, horizon); // [1, H]
                let val_next = values_full.narrow(1, 1, horizon); // [1, H]
                let target = rewards_detached + val_next.mul_scalar(gamma); // [1, H]
                let critic_loss = (target - val_now).powf_scalar(2.0).mean().reshape([1]);

                // Update critic first
                let c_data = critic_loss.clone().to_data();
                critic_loss_val = c_data.as_slice::<f32>().unwrap()[0];
                let c_grads = critic_loss.backward();
                let grads_c = GradientsParams::from_grads(c_grads, &critic);
                critic = critic_optim.step(config.critic_lr, critic, grads_c);

                // Update actor
                let a_data = actor_loss.clone().to_data();
                actor_loss_val = a_data.as_slice::<f32>().unwrap()[0];
                let a_grads = actor_loss.backward();
                let grads_a = GradientsParams::from_grads(a_grads, &actor);
                actor = actor_optim.step(config.actor_lr, actor, grads_a);
            }
        }

        // ── 4. Generate comparison video ──
        if ep_num % config.video_interval == 0 || ep_num == config.max_episodes - 1 {
            generate_video::<B>(
                &world_model, reference_ep.as_ref(),
                ep_num, c, h, w,
            );
        }

        // ── 5. Log ──
        if ep_num % 10 == 0 {
            println!(
                "ep {:4} | steps {:3} | reward {:+.4} | model {:.4} | actor {:.4} | critic {:.4}",
                ep_num, steps, total_reward, model_loss_val, actor_loss_val, critic_loss_val,
            );
        }
    }

    println!("=== Done ===");
    println!("Videos: output/comparison_ep_*.mp4");
}

fn generate_video<B: Backend>(
    world_model: &WorldModel<B>,
    ref_ep: Option<&Episode<B>>,
    ep_num: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    let Some(ep) = ref_ep else { return };
    let seq_len = ep.obs.len().min(40);
    if seq_len == 0 { return; }

    let frame_dir = format!("frames/ep_{:04}", ep_num);
    let output_path = format!("output/comparison_ep_{:04}.mp4", ep_num);

    std::fs::create_dir_all(&frame_dir).ok();
    std::fs::create_dir_all("output").ok();

    let device = ep.obs[0].device();
    let mut recon_state = world_model.init_state(1, &device);
    let first_obs = ep.obs[0].clone();
    recon_state = world_model.obs_step(&recon_state, first_obs);

    for t in 0..seq_len {
        let obs_flat = ep.obs[t].clone();
        let real = obs_flat.clone().reshape([c, h, w]);

        if t > 0 {
            let action = ep.action[t - 1].clone();
            recon_state = world_model.img_step(&recon_state, action);
        }

        let recon_flat = world_model.reconstruct(&recon_state);
        let recon = recon_flat.reshape([c, h, w]);

        let comparison = make_comparison_frame::<B>(&real, &recon);
        let frame_path = format!("{}/frame_{:04}.png", frame_dir, t);
        save_frame::<B>(&comparison, &frame_path);
    }

    frames_to_mp4(&frame_dir, &output_path, 10);
}
