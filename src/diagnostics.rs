//! Standalone diagnostics that isolate pipeline stages.
//!
//! Decoder fit tests: can decoder X learn (x, y, vx, vy) → frame, given the TRUE
//! state as input? Splits "decoder can't localize" from "the latent never contains
//! position". Run with DREAMER_DECODER_TEST=<variant>:
//!
//!   1 | weighted           BroadcastDecoder (8×8 grid + upsampling)
//!   gaussian[_weighted]    GaussianDecoder (analytic rotated super-Gaussian blobs)
//!   hybrid[_weighted]      HybridDecoder (broadcast background + blob foreground)
//!   deconv                 monolithic deconv Decoder
//!
//! `*weighted` variants train with 50× loss weight in the true ball region
//! (mirrors the capped motion weighting of training). EVERY variant reports the
//! unweighted ball-region MSE (r=10 disc around the true center) — the gate is
//! ball MSE < 0.02 before a decoder graduates to full training.
//!
//! NOTE on activations: BroadcastDecoder / Decoder emit raw logits (sigmoid applied
//! here); GaussianDecoder / HybridDecoder emit RGB already in [0, 1] — applying a
//! second sigmoid would squash the range to [0.5, 0.73] and make fitting impossible
//! (this exact bug previously produced a false-negative verdict on the Gaussian head).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Tensor};
use crate::envs::Environment;
use crate::envs::bouncing_ball::BouncingBall;
use crate::networks::broadcast_decoder::BroadcastDecoder;
use crate::networks::gaussian_decoder::GaussianDecoder;
use crate::networks::hybrid_decoder::HybridDecoder;
use crate::networks::decoder::Decoder;
use crate::networks::encoder::Encoder;
use crate::networks::slot_attention::{SlotAttention, SlotAttentionConfig};
use crate::video::save_frame;

const SIZE: usize = 64;
const CHANNELS: usize = 3;
const N_FRAMES: usize = 32;
const STEPS: usize = 3000;
const BALL_GATE: f32 = 0.02;

#[derive(Module, Debug)]
struct DecoderProbe<B: Backend> {
    proj: Linear<B>,
    dec: BroadcastDecoder<B>,
}

#[derive(Module, Debug)]
struct GProbe<B: Backend> {
    proj: Linear<B>,
    dec: GaussianDecoder<B>,
}

#[derive(Module, Debug)]
struct HProbe<B: Backend> {
    proj: Linear<B>,
    proj2: Linear<B>,
    dec: HybridDecoder<B>,
}

#[derive(Module, Debug)]
struct DProbe<B: Backend> {
    proj: Linear<B>,
    dec: Decoder<B>,
}

/// Autoencoder stage test: encoder → slot attention → proj → hybrid decoder,
/// with per-pixel softmax compositing — the FULL perception path minus the RSSM
/// and KL. Isolates "encoder/slots lose the ball position" from "RSSM/KL blur it".
#[derive(Module, Debug)]
struct AEProbe<B: Backend> {
    enc: Encoder<B>,
    slots: SlotAttention<B>,
    proj: Linear<B>,
    dec: HybridDecoder<B>,
}

impl<B: Backend> AEProbe<B> {
    /// obs [N, C*H*W] → composited recon [N, C*H*W] (RGB already in [0,1]).
    fn forward(&self, obs: Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
        let [n, _] = obs.dims();
        let hw = SIZE * SIZE;
        let obs_4d = obs.reshape([n, CHANNELS, SIZE, SIZE]);
        let feats = self.enc.forward_features_16x16(obs_4d);     // [N, 64, 16, 16]
        let slots = self.slots.forward(feats);                    // [N, K, D]
        let d = slots.dims()[2];
        let dec_in = relu(self.proj.forward(slots.reshape([n * k, d])));
        let out = self.dec.forward(dec_in);                       // [N*K, C+1, H, W]

        let rgb = out.clone().narrow(1, 0, CHANNELS);             // already [0,1]
        let logits = out.narrow(1, CHANNELS, 1);
        let rgb_4d = rgb.reshape([n, k, CHANNELS, hw]);
        let logits_3d = logits.reshape([n, k, hw]);
        let lmax = logits_3d.clone().max_dim(1).detach();
        let e = logits_3d.sub(lmax).exp();
        let masks = e.clone().div(e.sum_dim(1).add_scalar(1e-8)); // [N, K, HW]
        let comp = rgb_4d.mul(masks.reshape([n, k, 1, hw])).sum_dim(1);
        comp.reshape([n, CHANNELS * hw])
    }
}

/// Frames + true states + loss weights + ball-region mask, shared by all fit tests.
struct FitData<B: Backend> {
    targets: Tensor<B, 2>,   // [N, C*H*W] ground-truth frames
    inputs: Tensor<B, 2>,    // [N, 4] true (x, y, vx, vy)
    weights: Tensor<B, 2>,   // [N, C*H*W]: 50 within r=10 of the ball if weighted, else 1
    ball_mask: Tensor<B, 2>, // [N, C*H*W]: 1 within r=10 of the true center, else 0
}

impl<B: Backend> FitData<B> {
    /// For autoencoder-style tests the input IS the frame itself.
    fn inputs_obs(&self) -> Tensor<B, 2> {
        self.targets.clone()
    }
}

fn weighted_flag() -> bool {
    std::env::var("DREAMER_DECODER_TEST")
        .map(|v| v.ends_with("weighted"))
        .unwrap_or(false)
}

fn collect_fit_data<B: Backend>(device: &B::Device, weighted: bool) -> FitData<B> {
    let flat = CHANNELS * SIZE * SIZE;

    // Collect frames + true states from the env under scripted actions.
    let mut env = BouncingBall::new(200, 1, CHANNELS, SIZE, 7);
    let mut frames: Vec<Tensor<B, 2>> = Vec::new();
    let mut states: Vec<[f32; 4]> = Vec::new();
    let obs = env.reset::<B>(device);
    frames.push(obs.reshape([1, flat]));
    states.push(env.state());
    while frames.len() < N_FRAMES {
        let a = ((frames.len() % 5) as f32 - 2.0) / 2.0;
        let (obs, _r, done) = env.step::<B>(&[a], device);
        frames.push(obs.reshape([1, flat]));
        states.push(env.state());
        if done {
            let obs = env.reset::<B>(device);
            frames.push(obs.reshape([1, flat]));
            states.push(env.state());
        }
    }
    let targets: Tensor<B, 2> = Tensor::cat(frames, 0);
    let state_data: Vec<f32> = states.iter().flatten().copied().collect();
    let inputs: Tensor<B, 2> =
        Tensor::<B, 1>::from_floats(state_data.as_slice(), device).reshape([N_FRAMES, 4]);

    // Ball-region mask (r=10 around the true center) and, if weighted, 50× loss
    // weights on that region — mirrors the capped motion weighting of training.
    let n_px = SIZE * SIZE;
    let mut mdata = vec![0.0f32; N_FRAMES * flat];
    for (f, s) in states.iter().enumerate() {
        let bx = (s[0] * SIZE as f32) as i32;
        let by = ((1.0 - s[1]) * SIZE as f32) as i32;
        for y in 0..SIZE as i32 {
            for x in 0..SIZE as i32 {
                let dx = x - bx;
                let dy = y - by;
                if dx * dx + dy * dy <= 100 {
                    let p = (y as usize) * SIZE + x as usize;
                    for ch in 0..CHANNELS {
                        mdata[f * flat + ch * n_px + p] = 1.0;
                    }
                }
            }
        }
    }
    let wdata: Vec<f32> = mdata
        .iter()
        .map(|&m| if weighted && m > 0.0 { 50.0 } else { 1.0 })
        .collect();
    let ball_mask =
        Tensor::<B, 1>::from_floats(mdata.as_slice(), device).reshape([N_FRAMES, flat]);
    let weights =
        Tensor::<B, 1>::from_floats(wdata.as_slice(), device).reshape([N_FRAMES, flat]);

    FitData { targets, inputs, weights, ball_mask }
}

/// Weighted MSE training loss over the whole frame.
fn fit_loss<B: Backend>(rgb: Tensor<B, 2>, data: &FitData<B>) -> Tensor<B, 1> {
    let d = rgb.sub(data.targets.clone());
    let num = d.clone().mul(d).mul(data.weights.clone()).sum();
    num.div(data.weights.clone().sum())
}

/// Unweighted MSE restricted to the true ball region — the gate metric.
fn ball_mse<B: Backend>(rgb: Tensor<B, 2>, data: &FitData<B>) -> f32 {
    let d = rgb.sub(data.targets.clone());
    let num = d.clone().mul(d).mul(data.ball_mask.clone()).sum();
    let v = num.div(data.ball_mask.clone().sum().add_scalar(1e-8));
    v.to_data().as_slice::<f32>().unwrap()[0]
}

fn scalar<B: Backend>(t: &Tensor<B, 1>) -> f32 {
    t.clone().to_data().as_slice::<f32>().unwrap()[0]
}

/// Save real|recon comparison PNGs and print the final verdict.
fn finish<B: Backend>(prefix: &str, rgb: Tensor<B, 2>, data: &FitData<B>) {
    std::fs::create_dir_all("output").ok();
    for idx in [0usize, N_FRAMES / 2] {
        let recon = rgb.clone().narrow(0, idx, 1).reshape([CHANNELS, SIZE, SIZE]);
        let real = data.targets.clone().narrow(0, idx, 1).reshape([CHANNELS, SIZE, SIZE]);
        let cmp = crate::video::make_comparison_frame::<B>(&real, &recon);
        save_frame::<B>(&cmp, &format!("output/{}_{}.png", prefix, idx));
    }
    let bm = ball_mse(rgb, data);
    let verdict = if bm < BALL_GATE { "PASS" } else { "FAIL" };
    println!(
        "FINAL ball-region MSE = {:.5} (gate < {}) → {}",
        bm, BALL_GATE, verdict
    );
    println!("Saved output/{}_*.png (left real, right recon)", prefix);
}

pub fn decoder_fit_test<B: AutodiffBackend>(device: B::Device) {
    let latent = 64usize;
    let flat = CHANNELS * SIZE * SIZE;
    let weighted = weighted_flag();
    let data = collect_fit_data::<B>(&device, weighted);

    let mut probe = DecoderProbe::<B> {
        proj: LinearConfig::new(4, latent).init(&device),
        dec: BroadcastDecoder::init(&device, latent, CHANNELS, SIZE),
    };
    let mut optim = AdamConfig::new().init::<B, DecoderProbe<B>>();

    println!(
        "=== BroadcastDecoder fit test: (x,y,vx,vy) -> frame, {} frames, weighted={} ===",
        N_FRAMES, weighted
    );
    for step in 0..STEPS {
        let z = relu(probe.proj.forward(data.inputs.clone()));
        let out = probe.dec.forward(z); // [N, C+1, H, W] raw logits
        let rgb = sigmoid(out.narrow(1, 0, CHANNELS)).reshape([N_FRAMES, flat]);
        let loss = fit_loss(rgb.clone(), &data);
        if step % 100 == 0 {
            println!(
                "step {:4} | weighted mse {:.5} | ball mse {:.5}",
                step, scalar(&loss), ball_mse(rgb, &data)
            );
        }
        let grads = loss.backward();
        let g = GradientsParams::from_grads(grads, &probe);
        probe = optim.step(1e-3, probe, g);
    }

    let z = relu(probe.proj.forward(data.inputs.clone()));
    let rgb = sigmoid(probe.dec.forward(z).narrow(1, 0, CHANNELS)).reshape([N_FRAMES, flat]);
    finish("decoder_test", rgb, &data);
}

pub fn gaussian_decoder_fit_test<B: AutodiffBackend>(device: B::Device) {
    let latent = 64usize;
    let flat = CHANNELS * SIZE * SIZE;
    let weighted = weighted_flag();
    let data = collect_fit_data::<B>(&device, weighted);

    let mut probe = GProbe::<B> {
        proj: LinearConfig::new(4, latent).init(&device),
        dec: GaussianDecoder::init(&device, latent, CHANNELS, SIZE),
    };
    let mut optim = AdamConfig::new().init::<B, GProbe<B>>();

    println!(
        "=== GaussianDecoder fit test: {} frames, weighted={} ===",
        N_FRAMES, weighted
    );
    for step in 0..STEPS {
        let z = relu(probe.proj.forward(data.inputs.clone()));
        let out = probe.dec.forward(z);
        // RGB is already in [0,1] — no sigmoid (see module doc).
        let rgb = out.narrow(1, 0, CHANNELS).reshape([N_FRAMES, flat]);
        let loss = fit_loss(rgb.clone(), &data);
        if step % 100 == 0 {
            println!(
                "step {:4} | weighted mse {:.5} | ball mse {:.5}",
                step, scalar(&loss), ball_mse(rgb, &data)
            );
        }
        let grads = loss.backward();
        let g = GradientsParams::from_grads(grads, &probe);
        probe = optim.step(1e-3, probe, g);
    }

    let z = relu(probe.proj.forward(data.inputs.clone()));
    let rgb = probe.dec.forward(z).narrow(1, 0, CHANNELS).reshape([N_FRAMES, flat]);
    finish("gaussian_test", rgb, &data);
}

pub fn hybrid_decoder_fit_test<B: AutodiffBackend>(device: B::Device) {
    let latent = 64usize;
    let flat = CHANNELS * SIZE * SIZE;
    let weighted = weighted_flag();
    let data = collect_fit_data::<B>(&device, weighted);

    let mut probe = HProbe::<B> {
        proj: LinearConfig::new(4, latent).init(&device),
        proj2: LinearConfig::new(latent, latent).init(&device),
        dec: HybridDecoder::init(&device, latent, CHANNELS, SIZE),
    };
    let mut optim = AdamConfig::new().init::<B, HProbe<B>>();

    println!(
        "=== HybridDecoder fit test: {} frames, weighted={} ===",
        N_FRAMES, weighted
    );
    for step in 0..STEPS {
        // Two-layer probe: a single Linear(4→latent) under-parameterizes the
        // state→latent map and caps blob-parameter precision.
        let z = relu(probe.proj2.forward(relu(probe.proj.forward(data.inputs.clone()))));
        let out = probe.dec.forward(z);
        // RGB is already in [0,1] — no sigmoid (see module doc).
        let rgb = out.narrow(1, 0, CHANNELS).reshape([N_FRAMES, flat]);
        let loss = fit_loss(rgb.clone(), &data);
        if step % 100 == 0 {
            println!(
                "step {:4} | weighted mse {:.5} | ball mse {:.5}",
                step, scalar(&loss), ball_mse(rgb, &data)
            );
        }
        let grads = loss.backward();
        let g = GradientsParams::from_grads(grads, &probe);
        // Three-stage LR decay: blob positions converge early; the rim, highlight,
        // and arrow detail need progressively smaller steps.
        let lr = if step < STEPS / 2 { 1e-3 } else if step < 5 * STEPS / 6 { 2e-4 } else { 5e-5 };
        probe = optim.step(lr, probe, g);
    }

    let z = relu(probe.proj2.forward(relu(probe.proj.forward(data.inputs.clone()))));
    let rgb = probe.dec.forward(z).narrow(1, 0, CHANNELS).reshape([N_FRAMES, flat]);
    finish("hybrid_test", rgb, &data);
}

/// Stage test: obs → encoder → slot attention → hybrid decoder (softmax
/// compositing), trained as a plain autoencoder with ball-region weighting.
/// No RSSM, no KL. If this passes but full training stalls at ball ≈ 0.1, the
/// information loss is in the RSSM/KL stage, not perception.
pub fn slot_ae_fit_test<B: AutodiffBackend>(device: B::Device) {
    let flat = CHANNELS * SIZE * SIZE;
    let weighted = weighted_flag();
    let data = collect_fit_data::<B>(&device, weighted);
    let k = 3usize; // num_slots (config default)
    let slot_dim = 64usize;
    let dec_latent = 64usize;

    let sa_cfg = SlotAttentionConfig {
        num_slots: k,
        slot_dim,
        num_iterations: 3,
        feature_dim: 64,
    };
    let mut probe = AEProbe::<B> {
        enc: Encoder::init(&device, 128, CHANNELS, SIZE),
        slots: SlotAttention::init(&device, &sa_cfg),
        proj: LinearConfig::new(slot_dim, dec_latent).init(&device),
        dec: HybridDecoder::init(&device, dec_latent, CHANNELS, SIZE),
    };
    let mut optim = AdamConfig::new().init::<B, AEProbe<B>>();

    println!(
        "=== Slot-AE fit test (encoder+slots+hybrid, no RSSM): {} frames, weighted={} ===",
        N_FRAMES, weighted
    );
    for step in 0..STEPS {
        let rgb = probe.forward(data.inputs_obs(), k);
        let loss = fit_loss(rgb.clone(), &data);
        if step % 100 == 0 {
            println!(
                "step {:4} | weighted mse {:.5} | ball mse {:.5}",
                step, scalar(&loss), ball_mse(rgb, &data)
            );
        }
        let grads = loss.backward();
        let g = GradientsParams::from_grads(grads, &probe);
        let lr = if step < STEPS / 2 { 1e-3 } else if step < 5 * STEPS / 6 { 2e-4 } else { 5e-5 };
        probe = optim.step(lr, probe, g);
    }

    let rgb = probe.forward(data.inputs_obs(), k);
    finish("slot_ae_test", rgb, &data);
}

pub fn deconv_decoder_fit_test<B: AutodiffBackend>(device: B::Device) {
    let latent = 80usize; // deter+stoch
    let flat = CHANNELS * SIZE * SIZE;
    let data = collect_fit_data::<B>(&device, false);

    let mut probe = DProbe::<B> {
        proj: LinearConfig::new(4, latent).init(&device),
        dec: Decoder::init_with_out(&device, latent, CHANNELS, SIZE, CHANNELS),
    };
    let mut optim = AdamConfig::new().init::<B, DProbe<B>>();

    println!("=== DeconvDecoder fit test: {} frames ===", N_FRAMES);
    for step in 0..STEPS {
        let z = relu(probe.proj.forward(data.inputs.clone()));
        let out = probe.dec.forward(z);
        let rgb = sigmoid(out.narrow(1, 0, CHANNELS)).reshape([N_FRAMES, flat]);
        let loss = fit_loss(rgb.clone(), &data);
        if step % 100 == 0 {
            println!(
                "step {:4} | mse {:.5} | ball mse {:.5}",
                step, scalar(&loss), ball_mse(rgb, &data)
            );
        }
        let grads = loss.backward();
        let g = GradientsParams::from_grads(grads, &probe);
        probe = optim.step(1e-3, probe, g);
    }

    let z = relu(probe.proj.forward(data.inputs.clone()));
    let rgb = sigmoid(probe.dec.forward(z).narrow(1, 0, CHANNELS)).reshape([N_FRAMES, flat]);
    finish("deconv_test", rgb, &data);
}
