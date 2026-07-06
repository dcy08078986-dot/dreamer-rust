# Dreamer-Rust Training Failure: Comprehensive Diagnostic Plan

## Executive Summary
**Problem**: Training crashes with memory errors (wgpu "Not enough memory left") and the model loss remains persistently high (~200) when it runs, indicating the model fails to learn.

**Root Causes Identified**:
1. **CRITICAL**: Memory leak from excessive gradient graph accumulation
2. **CRITICAL**: Image size mismatch (config says 48x48 but env renders 64x64)
3. **MAJOR**: Reconstruction loss scale is enormous (raw pixel MSE on 48×48×3=6912 dimensions)
4. **MAJOR**: Missing gradient detachment in training loop causes graph retention
5. **MODERATE**: Extremely aggressive training schedule (10× world model updates during warmup)
6. **MODERATE**: Actor-critic uses concatenated slot states (doubling latent dim)

---

## Part 1: Critical Bugs (Fix Immediately)

### Bug #1: Image Size Mismatch ⚠️ CRITICAL
**File**: `src/config.rs` vs `src/envs/bouncing_ball.rs`

**Issue**:
```rust
// config.rs:104
image_size: 48,

// bouncing_ball.rs:49 (constructor receives image_size=48)
// but main.rs:32 uses image_size from config
// Yet encoder/decoder expect image_size / 16 to work cleanly
```

**Evidence**: 
- Config: `image_size: 48` 
- Environment rendered at 64×64 in the demo video
- Encoder expects `size / 16` for conv stride calculations
- 48/16 = 3, but 64/16 = 4 (cleaner)

**Fix Priority**: **P0 - Fix First**
```rust
// In config.rs, change:
image_size: 64,  // was 48 — must match actual render size
```

**Why This Matters**: Dimension mismatch causes spatial feature map size errors, leading to reconstruction failures and numerical instability.

---

### Bug #2: Missing Detachment in Training Loop ⚠️ CRITICAL
**File**: `src/train.rs:165-167`

**Issue**:
```rust
// Line 165-167: WRONG — reuses tensors from replay buffer that still have gradient tracking
let start_batch = replay.sample(1, 1, &device);
let start_obs = start_batch.obs.narrow(1, 0, 1).squeeze(1).detach();  // ✓ detached
let start_act = start_batch.action.narrow(1, 0, 1).squeeze(1).detach(); // ✓ detached

// Line 169-172: RE-ENCODES through world model (GOOD for gradients)
let start_obs_emb = world_model.encode(start_obs);
let init_state = world_model.init_state(1, &device);
let mut img_state = world_model.rssm.obs_step(&init_state, start_obs_emb, start_act);
```

**BUT**: The real problem is earlier — **world model training accumulates graphs**:

```rust
// Line 126-150: World model training loop
for _ in 0..wm_iters {  // wm_iters = 10 * num_envs = 40 iterations!
    let batch = replay.sample(...);
    let (loss, obs_loss, rew_loss, kl_loss, kl_w) = world_model.train_step(...);
    
    // CRITICAL BUG: These retain computation graphs from train_step!
    let loss_data = loss.clone().to_data();  
    model_loss_val = loss_data.as_slice::<f32>().unwrap()[0];
    // ... same for obs_loss, rew_loss, kl_loss
    
    let grads = loss.backward();  // ← Backward AFTER extracting scalar
    // ... optimizer step
}
// RESULT: 40 backward passes accumulate in memory without cleanup!
```

**Fix Priority**: **P0 - Fix First**
```rust
// In train.rs line 126-150, change to:
for _ in 0..wm_iters {
    let batch = replay.sample(...);
    let (loss, obs_loss, rew_loss, kl_loss, kl_w) = world_model.train_step(...);
    
    // Extract scalars BEFORE backward to prevent graph retention
    model_loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    obs_loss_val = obs_loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    rew_loss_val = rew_loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    kl_loss_val = kl_loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    kl_weight_val = kl_w.clone().into_data().as_slice::<f32>().unwrap()[0];
    
    // Now backward and step
    let grads = loss.backward();
    let grads_wm = GradientsParams::from_grads(grads, &world_model);
    world_model = model_optim.step(config.model_lr, world_model, grads_wm);
}
```

---

### Bug #3: Reconstruction Loss Scale ⚠️ CRITICAL
**File**: `src/agent/world_model.rs:115-118`

**Issue**:
```rust
// Line 115-118: Raw MSE over 6912 dimensions (3×48×48 or 3×64×64)
let recon = self.reconstruct(&post_state);
let target = batch_obs.clone().narrow(1, t, 1).squeeze(1);
let diff = recon.sub(target);
let squared = diff.clone().mul(diff);
let mse = squared.sum_dim(1).mean();  // Sum over ALL pixels!
obs_loss = obs_loss.add(mse);
```

**Math**: 
- Each pixel MSE contributes 1 term
- Total dimensions: 3 × 64 × 64 = 12,288
- Typical pixel error: ~0.1-0.5 per dimension
- **Result**: obs_loss ≈ 0.25² × 12288 ≈ **768 per timestep**!
- Over 8 timesteps: **6,144** — explains the "loss ~200" observation

**Fix Priority**: **P0 - Fix First**
```rust
// Option A: Normalize by number of pixels (RECOMMENDED)
let mse = squared.mean();  // Mean over ALL dimensions [B, C*H*W]

// Option B: Use loss weight
let mse = squared.sum_dim(1).mean();
obs_loss = obs_loss.add(mse.mul_scalar(1.0 / (c * h * w) as f64));
```

---

### Bug #4: Training Configuration Too Aggressive
**File**: `src/train.rs:124`

**Issue**:
```rust
// Line 124: EXTREMELY aggressive training
let wm_iters = if ep_counter <= 100 { 10 * num_envs } else { 5 * num_envs };
// With num_envs=4: 40 updates during warmup, 20 afterwards PER ROUND
```

**Problem**: 
- 40 gradient updates per round × 125 rounds = **5,000 world model updates** in first 100 episodes
- With memory leak (Bug #2), this crashes within minutes
- Even without leak, this overfits heavily on tiny replay buffer (5-20 episodes)

**Fix Priority**: **P1 - Fix After P0**
```rust
// More conservative schedule:
let wm_iters = if ep_counter <= 50 { 
    20  // Fixed 20 updates during early warmup
} else if ep_counter <= 150 {
    10  // Reduce to 10 during mid-training
} else { 
    5   // Standard 5 updates after warmup
};
```

---

## Part 2: Architectural Issues (Fix After Critical Bugs)

### Issue #5: Actor-Critic Latent Dimension Explosion
**File**: `src/train_oc.rs:36-37`

**Current**:
```rust
let mut actor = Actor::<B>::init(&device, 
    config.deter_size * config.num_slots,   // 64 * 2 = 128
    config.stoch_size * config.num_slots,   // 16 * 2 = 32
    act_dim);
```

**Problem**: This concatenates ALL slot states, doubling the latent dimension. For 2 slots, this is manageable, but conceptually wrong — the actor should operate on aggregated slot features, not raw concatenation.

**Fix Priority**: **P2 - Improve After Basics Work**
```rust
// Option A: Use pooled slot representation
fn aggregate_slots(slot_states: &[RSSMState<B>]) -> RSSMState<B> {
    // Mean pooling over slots
    let deter = Tensor::stack(slot_states.iter().map(|s| s.deter.clone()).collect(), 0).mean_dim(0);
    let stoch = Tensor::stack(slot_states.iter().map(|s| s.stoch.clone()).collect(), 0).mean_dim(0);
    // ... similar for mean and std
}

// Option B: Use attention-based aggregation (better but more complex)
```

---

### Issue #6: KL Divergence Free Nats Implementation
**File**: `src/agent/world_model.rs:134`

**Current**:
```rust
let kl_adj = kl.sub(free_nats_tensor.clone()).clamp_min(0.0_f32);
let kl_free = kl_adj.mean();
```

**Issue**: The free nats (0.5) is applied AFTER mean, but should be per-dimension or per-batch-element. Current implementation:
- Computes KL per example: [B] tensor
- Subtracts scalar 0.5 from each batch element
- This is too aggressive — allows very low KL per example

**Fix Priority**: **P2**
```rust
// Apply free nats per stochastic dimension
let kl_per_dim = kl.div_scalar(stoch_size as f64);  // Normalize by dimensions
let kl_adj = kl_per_dim.sub_scalar(free_nats).clamp_min(0.0);
let kl_free = kl_adj.mul_scalar(stoch_size as f64).mean();
```

---

## Part 3: Diagnostic Experiments

### Experiment 1: Verify World Model Reconstruction (1-2 hours)
**Goal**: Isolate whether world model can learn basic reconstruction

**Setup**:
```rust
// Disable actor-critic training completely
// Focus ONLY on world model for 100 episodes
// Monitor: obs_loss, kl_loss, reward_prediction_loss
```

**Success Criteria**:
- obs_loss drops from ~200 to <10 within 50 episodes
- KL loss stabilizes between 1.0-3.0
- Generated comparison videos show reasonable reconstruction

**What This Tells Us**:
- If SUCCESS: Problem is in actor-critic
- If FAILURE: Encoder/decoder architecture issue OR data pipeline bug

---

### Experiment 2: Simplified Environment Test (30 min)
**Goal**: Verify data pipeline with trivial task

**Setup**:
```rust
// Modify BouncingBall to render static ball (no physics)
// Ball position fixed at (0.5, 0.5)
// Background always same
```

**Expected**:
- obs_loss → 0 within 10 episodes
- KL loss → free_nats (model learns deterministic encoding)

**What This Tells Us**: If this fails, encoder/decoder has fundamental bug (wrong activation, dimension mismatch)

---

### Experiment 3: Gradient Norm Monitoring (Add to training loop)
**Goal**: Detect gradient explosion/vanishing

**Code**:
```rust
// Add after line 148 in train.rs
let grad_norm = grads_wm.grads.iter()
    .map(|(_, g)| g.clone().powf_scalar(2.0).sum().sqrt())
    .fold(Tensor::zeros([1], device), |acc, x| acc + x);
println!("grad_norm: {:.4}", grad_norm.into_data().as_slice::<f32>().unwrap()[0]);
```

**Success Criteria**: Grad norms in range [0.1, 10.0]

---

## Part 4: Implementation Priority

### Phase 1: Fix Critical Bugs (Day 1 — 2-4 hours)
1. ✅ Fix image size mismatch (config.rs)
2. ✅ Fix reconstruction loss scale (world_model.rs)  
3. ✅ Fix memory leak from gradient retention (train.rs)
4. ✅ Reduce training iteration count (train.rs)

**Expected Outcome**: Training runs without crash, loss starts to decrease

---

### Phase 2: Verify Learning (Day 1-2 — 4-8 hours)
5. Run Experiment 1 (world model only)
6. Add gradient monitoring
7. Tune hyperparameters:
   - Learning rates (try 1e-4 for model, 8e-5 for actor/critic)
   - Batch size (increase to 4)
   - Sequence length (reduce to 6 during warmup)

**Expected Outcome**: obs_loss < 10, reasonable video reconstructions

---

### Phase 3: Enable Actor-Critic (Day 2-3 — 4-8 hours)
8. Fix actor-critic latent aggregation
9. Tune discount/lambda for value estimation
10. Add entropy regularization to prevent policy collapse

**Expected Outcome**: Agent learns basic control (ball stays airborne longer)

---

### Phase 4: Advanced Improvements (Day 3+ — optional)
11. Implement proper free nats
12. Add model ensemble for uncertainty estimation
13. Implement curiosity-driven exploration
14. Switch to object-centric world model if monolithic version works

---

## Part 5: Quick Validation Checklist

Before running training, verify:
- [ ] `config.image_size == env.image_size` (both 64)
- [ ] Encoder output dim matches `embed_dim` (128)
- [ ] Decoder input dim matches `deter_size + stoch_size` (80)
- [ ] Flat observation dim is `c * h * w` (12,288)
- [ ] Replay buffer has min 5 episodes before training
- [ ] Batch sampling works for `batch_size=2, seq_len=8`
- [ ] All loss scalars extracted via `.into_data()` not `.to_data()`
- [ ] Reconstruction loss uses `.mean()` not `.sum_dim(1).mean()`

---

## Part 6: Expected Metrics After Fixes

### Healthy Training Signatures:
```
round    0 | ep    3 | avg_reward +120 | model 180.00 (obs:178 rew:1.2 kl:0.8 kl_w:3.0) [WARMUP]
round    5 | ep   23 | avg_reward +135 | model  45.00 (obs:43 rew:1.1 kl:0.9 kl_w:3.0) [WARMUP]
round   10 | ep   43 | avg_reward +142 | model  12.00 (obs:10 rew:1.2 kl:0.8 kl_w:0.8)
round   20 | ep   83 | avg_reward +155 | model   4.50 (obs:3.2 rew:0.6 kl:0.7 kl_w:0.8)
round   30 | ep  123 | avg_reward +168 | model   2.80 (obs:1.8 rew:0.4 kl:0.6 kl_w:0.8) | actor 0.8 | critic 12.3
```

### Key Indicators:
- **obs_loss**: Should drop 180 → 10 → 3 over first 50 episodes
- **kl_loss**: Stable around 0.5-1.5 (not collapsed to 0.0)
- **reward_loss**: Decreases as reward predictor improves
- **actor_loss**: Meaningful values (not NaN, not 0)
- **avg_reward**: Gradually increases from ~130 to ~180+

---

## Part 7: Risk Assessment

### High Risk (Will Definitely Cause Failure)
- ✅ Image size mismatch → **Spatial dimension errors**
- ✅ Reconstruction loss scale → **Dominates gradient, prevents learning**
- ✅ Memory leak → **Training crashes within minutes**

### Medium Risk (May Prevent Convergence)
- Training schedule too aggressive → Overfitting on small replay
- Missing proper latent aggregation → Actor operates on wrong representation
- No exploration bonus → Policy may collapse to local minimum

### Low Risk (Affects Final Performance)
- Free nats implementation → May allow posterior collapse
- Symlog transformation → Good idea but not critical for simple env
- Reward scale (0.5) → Conservative, may slow learning

---

## Contact / Next Steps

**Immediate Action**: 
Implement Phase 1 fixes (4 changes in 3 files) and run training for 50 episodes.

**Decision Point** (after 20 min of training):
- If obs_loss still >100 → Check Experiment 2 (static environment)
- If obs_loss decreasing → Continue to 100 episodes
- If crash → Check memory usage and gradient norms

**Timeline**:
- Fixes: 2-4 hours
- Validation: 4-8 hours
- Full convergence: 1-2 days

Would you like me to proceed with implementing these fixes?
