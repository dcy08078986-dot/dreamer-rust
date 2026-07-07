use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use crate::config::Config;
use crate::envs::Environment;
use crate::replay::{Episode, ReplayBuffer};
use crate::agent::oc_world_model::OCWorldModel;
use crate::agent::actor_critic::{Actor, Critic, imagination_losses};
use crate::networks::rssm::RSSMState;
use crate::networks::slot_rssm::SlotStates;
use crate::video::{save_frame, frames_to_mp4, make_comparison_frame};

pub fn run_oc<B: AutodiffBackend, E: Environment>(device: B::Device, config: Config, envs: &mut [E]) {
    let num_envs = envs.len();
    let [c, h, w] = envs[0].obs_shape();
    let flat_dim = c * h * w;
    let total_eps = config.max_episodes;
    let rounds = total_eps / num_envs;
    let act_dim = envs[0].action_dim();

    println!("=== OC {} slots, {} envs, {} rounds ===", config.num_slots, num_envs, rounds);

    let mut wm = OCWorldModel::<B>::init(&device, &config, act_dim);
    let mut actor = Actor::<B>::init(&device, config.deter_size * config.num_slots, config.stoch_size * config.num_slots, act_dim);
    let mut critic = Critic::<B>::init(&device, config.deter_size * config.num_slots, config.stoch_size * config.num_slots);
    let mut model_optim = AdamConfig::new().init::<B, OCWorldModel<B>>();
    let mut actor_optim = AdamConfig::new().init::<B, Actor<B>>();
    let mut critic_optim = AdamConfig::new().init::<B, Critic<B>>();

    let mut replay = ReplayBuffer::<B>::new(config.replay_capacity);
    let mut reference_ep: Option<Episode<B>> = None;
    let mut ep_counter: usize = 0;
    let effective_batch = config.batch_size.max(1);

    for round in 0..rounds {
        let mut round_reward: f32 = 0.0;
        let mut round_steps: usize = 0;

        for env in envs.iter_mut() {
            let obs = env.reset::<B>(&device);
            let obs_flat = obs.reshape([1, flat_dim]);
            let zero_action = Tensor::zeros([1, act_dim], &device);
            let mut cur_states = wm.obs_step(&wm.init_state(1, &device), obs_flat.clone(), zero_action).detach();
            let mut ep_obs: Vec<Tensor<B, 2>> = vec![obs_flat];
            let mut ep_act: Vec<Tensor<B, 2>> = Vec::new();
            let mut ep_rew: Vec<Tensor<B, 2>> = Vec::new();
            let mut ep_done: Vec<Tensor<B, 2>> = Vec::new();
            let mut total_reward: f32 = 0.0;
            let mut steps = 0;
            let anneal = 1.0 - (ep_counter as f32 / total_eps.max(1) as f32);

            for _t in 0..env.max_steps() {
                let actor_state = slot_states_to_rssm(&cur_states);
                let (action, _logp) = actor.sample(&actor_state);
                let action = action.detach();
                let action_data = action.to_data();
                let act_vals = action_data.as_slice::<f32>().unwrap().to_vec();
                let (next_obs, reward, done) = env.step::<B>(&act_vals, &device);
                let next_obs_flat = next_obs.reshape([1, flat_dim]);
                let stored_reward = if config.use_ensemble_exploration {
                    let bonus = wm.slot_rssm.disagreement(cur_states.deter.clone());
                    reward + config.exploration_scale * anneal * bonus
                } else { reward };

                ep_act.push(action.clone());
                ep_rew.push(Tensor::full([1, 1], stored_reward, &device));
                ep_done.push(Tensor::full([1, 1], if done { 1.0f32 } else { 0.0 }, &device));
                ep_obs.push(next_obs_flat.clone());
                total_reward += reward;
                steps += 1;
                if done { break; }
                cur_states = wm.obs_step(&cur_states, next_obs_flat, action).detach();
            }

            replay.push(Episode { obs: ep_obs.clone(), action: ep_act.clone(), reward: ep_rew.clone(), done: ep_done.clone() });
            if reference_ep.is_none() { reference_ep = Some(Episode { obs: ep_obs, action: ep_act, reward: ep_rew, done: ep_done }); }
            round_reward += total_reward;
            round_steps = round_steps.max(steps);
            if ep_counter % 10 == 0 { println!("ep {:4} | steps {:3} | reward {:+.4}", ep_counter, steps, total_reward); }
            ep_counter += 1;
        }

        let mut model_loss_val: f32 = 0.0; let mut obs_loss_val: f32 = 0.0; let mut rew_loss_val: f32 = 0.0; let mut kl_loss_val: f32 = 0.0;
        let mut motion_mse_val: f32 = 0.0; let mut slot_kl_vals: Vec<f32> = vec![0.0; config.num_slots];
        let mut post_std_val: f32 = 0.0;
        let wm_iters = if ep_counter <= 100 { 3 * num_envs } else { 2 * num_envs };
        if replay.len() >= 5 {
            for _ in 0..wm_iters {
                let batch = replay.sample(effective_batch.min(replay.len()), config.batch_length.min(round_steps.max(2)), &device);
                let reward_2d = batch.reward.clone().squeeze(2);
                let losses = wm.train_step(batch.obs, batch.action, reward_2d, &config, false, config.kl_scale);
                model_loss_val = losses.total.clone().to_data().as_slice::<f32>().unwrap()[0];
                obs_loss_val = losses.obs_loss; rew_loss_val = losses.reward_loss; kl_loss_val = losses.kl_loss;
                motion_mse_val = losses.motion_mse;
                post_std_val = losses.post_std;
                slot_kl_vals = losses.per_slot_kl.clone();
                let grads = losses.total.backward();
                let grads_wm = GradientsParams::from_grads(grads, &wm);
                wm = model_optim.step(config.model_lr, wm, grads_wm);
            }
        }

        let mut actor_loss_val: f32 = 0.0; let mut critic_loss_val: f32 = 0.0;
        if replay.len() >= 5 && ep_counter > 100 {
            let imag_horizon = config.imag_horizon.min(15);
            let imag_batch = config.imag_batch.min(replay.len()).max(1);
            let start = replay.sample(imag_batch, 1, &device);
            let start_obs = start.obs.narrow(1, 0, 1).squeeze(1).detach();
            let b = start_obs.dims()[0];
            let zero_action = Tensor::zeros([b, act_dim], &device);
            let mut slot_states = wm.obs_step(&wm.init_state(b, &device), start_obs, zero_action).detach();
            let mut imag_states: Vec<RSSMState<B>> = vec![slot_states_to_rssm(&slot_states)];
            let mut imag_raw_actions: Vec<Tensor<B, 2>> = Vec::new();
            let mut imag_rewards: Vec<Tensor<B, 1>> = Vec::new();
            for _t in 0..imag_horizon {
                let actor_state = slot_states_to_rssm(&slot_states);
                let (action, raw, _logp) = actor.sample_with_raw(&actor_state);
                slot_states = wm.img_step(&slot_states, action.detach()).detach();
                imag_rewards.push(wm.predict_reward(&slot_states).detach());
                imag_raw_actions.push(raw.detach());
                imag_states.push(slot_states_to_rssm(&slot_states));
            }
            let (a_loss, c_loss) = imagination_losses(&actor, &critic, &imag_states, &imag_raw_actions, &imag_rewards, config.discount as f64, config.lambda_ as f64, config.entropy_coef);
            critic_loss_val = c_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
            let c_grads = c_loss.backward();
            let grads_c = GradientsParams::from_grads(c_grads, &critic);
            critic = critic_optim.step(config.critic_lr, critic, grads_c);
            actor_loss_val = a_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
            let a_grads = a_loss.backward();
            let grads_a = GradientsParams::from_grads(a_grads, &actor);
            actor = actor_optim.step(config.actor_lr, actor, grads_a);
        }

        let last_ep = ep_counter.saturating_sub(1);
        let avg_reward = round_reward / num_envs as f32;
        let warmup_tag = if ep_counter <= 100 { " [WARMUP]" } else { "" };
        println!("round {:4} | ep {:4} | avg_reward {:+.4} | model {:.4} (obs:{:.4} ball:{:.4} rew:{:.4} kl:{:.4} std:{:.3}) | actor {:.4} | critic {:.4} | slot_kl:[{:.2}, {:.2}, {:.2}]{}",
            round, last_ep, avg_reward, model_loss_val, obs_loss_val, motion_mse_val, rew_loss_val, kl_loss_val, post_std_val, actor_loss_val, critic_loss_val,
            slot_kl_vals.get(0).unwrap_or(&0.0), slot_kl_vals.get(1).unwrap_or(&0.0), slot_kl_vals.get(2).unwrap_or(&0.0), warmup_tag);

        // ── Checkpoint ──
        if ep_counter > 100 && last_ep % 50 == 0 {
            let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
            let _ = std::fs::create_dir_all(&config.checkpoint_dir);
            wm.clone().save_file(format!("{}/oc_wm_ep_{}", config.checkpoint_dir, last_ep), &rec).ok();
            actor.clone().save_file(format!("{}/oc_actor_ep_{}", config.checkpoint_dir, last_ep), &rec).ok();
            critic.clone().save_file(format!("{}/oc_critic_ep_{}", config.checkpoint_dir, last_ep), &rec).ok();
            println!("Checkpoint saved at episode {}", last_ep);
        }

        // ── Video ──
        if config.video_interval > 0 && (round == 0 || last_ep % config.video_interval == 0 || ep_counter >= total_eps) {
            generate_oc_video::<B>(&wm, reference_ep.as_ref(), last_ep, c, h, w);
        }
    }
    println!("=== Done ===");
}

fn slot_states_to_rssm<B: Backend>(states: &SlotStates<B>) -> RSSMState<B> {
    let b = states.batch; let k = states.num_slots;
    let dd = states.deter.dims()[1]; let ds = states.stoch.dims()[1];
    let deter = states.deter.clone().reshape([b, k * dd]);
    let stoch = states.stoch.clone().reshape([b, k * ds]);
    let zeros = Tensor::zeros_like(&deter);
    RSSMState { deter, stoch, mean: zeros.clone(), std: zeros }
}

fn generate_oc_video<B: Backend>(wm: &OCWorldModel<B>, ref_ep: Option<&Episode<B>>, ep_num: usize, c: usize, h: usize, w: usize) {
    let Some(ep) = ref_ep else { return };
    let seq_len = ep.obs.len().min(100);
    if seq_len == 0 { return; }
    let frame_dir = format!("frames/oc_ep_{:04}", ep_num);
    let output_path = format!("output/oc_comparison_ep_{:04}.mp4", ep_num);
    std::fs::create_dir_all(&frame_dir).ok(); std::fs::create_dir_all("output").ok();
    let device = ep.obs[0].device();
    let zero_action = Tensor::zeros([1, ep.action[0].dims()[1]], &device);
    let mut states = wm.obs_step(&wm.init_state(1, &device), ep.obs[0].clone(), zero_action).detach();
    for t in 0..seq_len {
        if t > 0 { states = wm.img_step(&states, ep.action[t-1].clone()).detach(); }
        let (recon, _, _) = wm.decode_slots(&states);
        let recon_img = recon.reshape([c, h, w]);
        let real = ep.obs[t].clone().reshape([c, h, w]);
        save_frame::<B>(&make_comparison_frame::<B>(&real, &recon_img), &format!("{}/frame_{:04}.png", frame_dir, t));
    }
    frames_to_mp4(&frame_dir, &output_path, 20);
}
