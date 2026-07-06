use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use crate::config::Config;
use crate::envs::Environment;
use crate::replay::{Episode, ReplayBuffer};
use crate::agent::world_model::WorldModel;
use crate::agent::actor_critic::{Actor, Critic, imagination_losses};
use crate::networks::rssm::RSSMState;
use crate::video::{save_frame, frames_to_mp4, make_comparison_frame};

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

    println!("=== Dreamer (monolithic RSSM, {} envs, {} rounds) ===", num_envs, rounds);
    println!("Obs: {}x{}x{} | Action: {} | Embed: {} | motion_loss: {}",
        c, h, w, act_dim, config.embed_dim, config.use_motion_loss);

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
    let effective_batch = config.batch_size.max(1);

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
                let action = action.detach();
                let action_data = action.to_data();
                let act_vals = action_data.as_slice::<f32>().unwrap().to_vec();

                let (next_obs, reward, done) = env.step::<B>(&act_vals, &device);
                let next_obs_flat = next_obs.reshape([1, flat_dim]);

                ep_act.push(action.clone());
                ep_rew.push(Tensor::full([1, 1], reward, &device));
                ep_done.push(Tensor::full([1, 1], if done { 1.0f32 } else { 0.0 }, &device));
                ep_obs.push(next_obs_flat.clone());
                total_reward += reward;
                steps += 1;
                if done { break; }
                cur_state = world_model.obs_step(&cur_state, next_obs_flat, action).detach();
            }

            // Keep ALL observations (steps+1 of them): the final transition is
            // trainable and short episodes stay usable down to seq_len steps.
            replay.push(Episode {
                obs: ep_obs.clone(), action: ep_act.clone(),
                reward: ep_rew.clone(), done: ep_done.clone(),
            });
            if reference_ep.is_none() {
                reference_ep = Some(Episode {
                    obs: ep_obs, action: ep_act,
                    reward: ep_rew, done: ep_done,
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
        let mut obs_loss_val: f32 = 0.0;
        let mut rew_loss_val: f32 = 0.0;
        let mut kl_loss_val: f32 = 0.0;
        let mut motion_mse_val: f32 = 0.0;
        let wm_iters = if ep_counter <= 50 {
            20
        } else if ep_counter <= 150 {
            10
        } else {
            5
        };
        if replay.len() >= 5 {
            for _ in 0..wm_iters {
                let batch = replay.sample(effective_batch.min(replay.len()),
                    config.batch_length.min(round_steps.max(2)), &device);
                let reward_2d = batch.reward.clone().squeeze(2);
                let diag = world_model.train_step(batch.obs, batch.action, reward_2d, &config);
                model_loss_val = tensor_scalar(&diag.total);
                obs_loss_val = tensor_scalar(&diag.obs_loss);
                rew_loss_val = tensor_scalar(&diag.reward_loss);
                kl_loss_val = tensor_scalar(&diag.kl_loss);
                motion_mse_val = tensor_scalar(&diag.motion_mse);
                let grads = diag.total.backward();
                let grads_wm = GradientsParams::from_grads(grads, &world_model);
                world_model = model_optim.step(config.model_lr, world_model, grads_wm);
            }
        }

        // ── 3. Train actor-critic on imagined rollouts (skip during warm-up) ──
        let mut actor_loss_val: f32 = 0.0;
        let mut critic_loss_val: f32 = 0.0;
        let warmup_eps = 100; // first 100 episodes: only train world model
        if replay.len() >= 5 && ep_counter > warmup_eps {
            let imag_horizon = config.imag_horizon.min(15);
            let imag_batch = config.imag_batch.min(replay.len()).max(1);

            // Batch of start states re-encoded from replay observations.
            let start = replay.sample(imag_batch, 1, &device);
            let start_obs = start.obs.narrow(1, 0, 1).squeeze(1).detach();
            let b = start_obs.dims()[0];
            let zero_action = Tensor::zeros([b, act_dim], &device);
            let init_state = world_model.init_state(b, &device);
            let mut state = detach_state(&world_model.obs_step(&init_state, start_obs, zero_action));

            // Rollout with detached states: REINFORCE needs no dynamics gradients,
            // and detaching keeps the graph (and CPU memory) per-step.
            let mut imag_states: Vec<RSSMState<B>> = vec![state.clone()];
            let mut imag_raw_actions: Vec<Tensor<B, 2>> = Vec::new();
            let mut imag_rewards: Vec<Tensor<B, 1>> = Vec::new();
            for _t in 0..imag_horizon {
                let (action, raw, _logp) = actor.sample_with_raw(&state);
                let next = detach_state(&world_model.img_step(&state, action.detach()));
                // Reward on ARRIVING at the next state (symlog space; head is symlog-trained).
                imag_rewards.push(world_model.predict_reward(&next).detach());
                imag_raw_actions.push(raw.detach());
                imag_states.push(next.clone());
                state = next;
            }

            let (actor_loss, critic_loss) = imagination_losses(
                &actor, &critic, &imag_states, &imag_raw_actions, &imag_rewards,
                config.discount as f64, config.lambda_ as f64, config.entropy_coef,
            );

            critic_loss_val = tensor_scalar(&critic_loss);
            let c_grads = critic_loss.backward();
            let grads_c = GradientsParams::from_grads(c_grads, &critic);
            critic = critic_optim.step(config.critic_lr, critic, grads_c);

            actor_loss_val = tensor_scalar(&actor_loss);
            let a_grads = actor_loss.backward();
            let grads_a = GradientsParams::from_grads(a_grads, &actor);
            actor = actor_optim.step(config.actor_lr, actor, grads_a);
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
        let warmup_tag = if ep_counter <= 100 { " [WARMUP]" } else { "" };
        println!("round {:4} | ep {:4} | avg_reward {:+.4} | model {:.4} (obs:{:.4} ball:{:.4} rew:{:.4} kl:{:.4}) | actor {:.4} | critic {:.4}{}",
            round, last_ep, avg_reward, model_loss_val, obs_loss_val, motion_mse_val, rew_loss_val, kl_loss_val, actor_loss_val, critic_loss_val, warmup_tag);
    }
    println!("=== Done ===");
}

fn tensor_scalar<B: Backend>(t: &Tensor<B, 1>) -> f32 {
    t.clone().to_data().as_slice::<f32>().unwrap()[0]
}

fn detach_state<B: Backend>(s: &RSSMState<B>) -> RSSMState<B> {
    s.detach()
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
