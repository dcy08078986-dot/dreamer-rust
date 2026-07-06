# RESULTS — Step 1 (hygiene) + Step 2 (motion-weighted reconstruction)

Date: 2026-07-03. Runs: CPU `Autodiff<NdArray>`, BouncingBall 64×64×3, default config
unless noted. Logs in `logs/`. Ablation flags: see `apply_env_overrides` (src/main.rs).

## Step 1 — hygiene fixes (all landed, gate PASSED)

Changes: normalized OC recon loss (was un-normalized sum over 12,288 px — the source of
"loss ≈ 200"); DreamerV3 KL balancing (α=0.8) + per-dim free bits, replacing the inverted
adaptive-KL rule; symlog reward targets; λ-returns + entropy bonus + imagination batch 16
(replacing 1-step TD at batch 1); correct tanh-Gaussian log-probs (density at the raw
action); fixed slot action-broadcast ordering (was slot-major vs batch-major mismatch —
wrong batch element's action per slot when B>1); kept final obs of each episode; added
"ball MSE" (motion-region MSE) metric.

Critical extra fix: Gaussian-RSSM KL was O(10²–10⁴) and swamped everything. Adding
**LayerNorm on the posterior's obs embedding + a hidden layer in prior/post heads**
(src/networks/rssm.rs) brought KL from 2310→1.4 nats within 60 episodes
(`logs/step2_oc_60ep.log` vs `logs/step2_oc_60ep_v2.log`). Training is now stable:
no NaN, no divergence, slot KLs balanced.

## Step 2 — motion-saliency-weighted reconstruction

**Implementation**: per-pixel weights `w = min(1 + λ·sal/mean(sal), w_max)` with
λ=8, w_max=50, saliency = channel-max frame difference (detached). `tools.rs::motion_weights`,
used in both world models. The cap is essential: saliency is sparse (ball ≈ 1.6% of
pixels), so uncapped `sal/mean ≈ 60` → ball weight ≈ 490× → ~89% of the loss on 200 px,
and the trained decoder painted the ENTIRE image ball-colored (`frames/oc_ep_0099`).

**A/B evidence that the loss does what it claims** (200 episodes each):
| run | full-img MSE (end) | ball-region MSE (trajectory) |
|---|---|---|
| motion OFF (`logs/step2_oc_200ep_motion_off.log`) | 0.05 → 0.14 | 0.10 → **0.23–0.36 (rising)** |
| motion ON, uncapped (`logs/step3_oc_200ep.log`) | ~0.12 | **0.10–0.15 (held)** |

Motion-OFF reproduces the predicted failure mode exactly: full-image MSE looks healthy
while the model progressively discards the ball. Motion weighting prevents that decay.

**But the gate (ball MSE ↓ 5×) is NOT passed.** Ball-region MSE plateaus ≈ 0.10 in all
configurations. Root cause isolated with two diagnostics:

1. **One-batch overfit** (`DREAMER_OVERFIT=1`, `logs/step2_overfit.log`): even 400
   updates on a single fixed batch leave ball MSE at 0.099; with `DREAMER_KL_SCALE=0`
   the same — so the KL bottleneck is ruled out.
2. **Decoder-only fit test** (`DREAMER_DECODER_TEST=1|weighted`, src/diagnostics.rs):
   the BroadcastDecoder given the TRUE (x,y,vx,vy) as input, 1500 steps @ lr 1e-3,
   learns the sky gradient but never paints the ball — with plain OR ball-weighted MSE
   (`output/decoder_test_*.png`). 

**Conclusion**: the remaining bottleneck is the spatial-broadcast decoder's slow
localization (a known property of SBD — literature trains 100k+ steps), not the loss,
not the latent KL, and not the data. Motion weighting is implemented, calibrated
(capped), and validated as necessary-but-not-sufficient.

**Next step when resuming** (not done, per scope cut): make localization learnable
faster — e.g. broadcast at full 64×64 resolution with stride-1 convs (paper-faithful
SBD), a Gaussian-blob decoder head (predict (x,y,σ) per slot and render analytically),
or simply a much longer training budget.

## Where things stand

- Steps 3 (per-pixel masks + broadcast decoder + slot interaction), 4 (reconstruction-
  free latent objective, flag `DREAMER_LATENT=1`), 5 (ensemble exploration,
  `DREAMER_ENSEMBLE=1`) are implemented, compile, and smoke-tested, but NOT validated —
  scope was narrowed to Step 2.
- Slot masks are still uniform (no decomposition) — expected until the decoder can
  localize (see above).
- Build: `cargo build --release` clean (0 errors).
