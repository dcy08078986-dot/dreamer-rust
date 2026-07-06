# 🎯 Ball MSE < 0.02 优化计划

**当前状态**：`ball = 0.111`（最佳配置：ensemble）  
**目标**：`ball < 0.02`（需降低 **5.5×**）  
**根本瓶颈**：实验 3c 证明 BroadcastDecoder 给定真实坐标也无法重建球体

---

## 📋 执行路线图

### Phase 0：解码器根本性重构（P0，预计改善 3-5×）
**诊断**：实验 3c 显示 BroadcastDecoder 即使给定真实 `(x,y,vx,vy)` 也无法画出球（MSE 0.166 vs 目标区域），这是所有基于重建的目标函数的根本限制。

#### 方案 0A：全分辨率 Spatial Broadcast Decoder（首选）
**原理**：不再将 slot 向量广播到低分辨率 grid，而是：
1. 将 slot 向量通过 MLP 扩展为 `[slot_dim → 256]`
2. 广播到 **全分辨率** `64×64` grid：`[B, 3, 256] → [B, 3, 256, 64, 64]`
3. 拼接空间坐标后用轻量 CNN 解码：`Conv(258→128→64→4)` → mask + RGB

**优势**：
- 避免上采样丢失位置信息
- 每个 slot 在全分辨率操作，球的边界更清晰
- 参数量增加不大（主要是广播，无需学习）

**实现**：
```rust
// src/networks/spatial_broadcast_decoder.rs（新建）
pub struct SpatialBroadcastConfig {
    slot_dim: usize,        // 64
    hidden_dim: usize,      // 256
    output_channels: usize, // 4 (RGB + mask)
    resolution: usize,      // 64
}

// Forward:
// 1. MLP: [B,3,64] → [B,3,256]
// 2. Broadcast: → [B,3,256,64,64]
// 3. 拼接坐标 grid (x,y): → [B,3,258,64,64]
// 4. CNN: Conv2d(258→128, k=3) → Conv2d(128→64, k=3) → Conv2d(64→4, k=1)
// 5. Split: mask=[B,3,1,64,64], rgb=[B,3,3,64,64]
// 6. Compose: weighted_sum(rgb * softmax(mask))
```

**验证步骤**：
1. 单元测试：给定真实球坐标，验证能否重建（ball MSE < 0.05）
2. 集成到 OC world model，保持其他超参不变
3. 对比 motion-on 配置下的 ball MSE

**预期改善**：`ball: 0.111 → 0.03-0.05`

---

#### 方案 0B：解析式高斯斑点渲染器（备选）
**原理**：将球体建模为 2D 高斯分布，解码器输出解析参数：
```rust
struct GaussianParams {
    mu_x: f32,    // 中心 x
    mu_y: f32,    // 中心 y
    sigma: f32,   // 半径
    color: [f32; 3], // RGB
    alpha: f32,   // 不透明度
}
```

逐像素计算强度：`I(x,y) = alpha * color * exp(-((x-mu_x)^2 + (y-mu_y)^2) / (2*sigma^2))`

**优势**：
- 完美匹配球体先验
- 参数量极小（每个 slot 7 个参数）
- 可微分渲染，梯度直达球心/半径

**劣势**：
- 不适用于复杂形状（但当前环境只有球+背景）
- 需要手写 backward pass 或用自动微分

**实现优先级**：如果 0A 仍无法达标（< 0.05），再尝试此方案

---

### Phase 1：应用已验证的架构优化（P1，预计改善 1.1×）

#### 1A：默认开启 Ensemble 探索（✅ 实验 B 验证）
**修改**：
```rust
// src/config.rs
pub const ENSEMBLE_HEADS: usize = 2; // 默认开启
```

**效果**：
- `kl: 1.06 → 0.55`（降低 48%）
- `actor: -0.29 → +3.71`（唯一保持正值的配置）
- `slot_kl` 更均衡：`[1.09,0.97,0.79] → [0.30,0.32,0.31]`

**原理**：额外 prior 头的 ensemble loss 充当正则化器，防止 posterior 过拟合背景，迫使 RSSM 学习更紧凑的表示。

---

#### 1B：关闭跨 Slot 交互（✅ 实验 C 验证）
**修改**：
```rust
// src/config.rs
pub const SLOT_INTERACTION: bool = false; // 默认关闭
```

**效果**：
- `actor: -0.29 → +3.09`（改善 3.38）
- `ball: 0.111 → 0.118`（略差 6%，可接受）

**原因**：当前的交叉注意力机制可能引入噪声或过度耦合。在解码器修复前，交互的收益不明显。

---

#### 1C：简化 Actor 输入（🔶 实验 D 启发）
**问题**：OC 版本当前将 3 个 slot 的 state 直接拼接 `[64×3=192]` 输入 actor，但单体 baseline（64 维）的 actor loss 更好（`+2.97` vs `-0.29`）。

**修改**：
```rust
// src/agent/actor_critic.rs
// 原：actor_input = concat([s0, s1, s2])  // [B, 192]
// 新：actor_input = mean_pool([s0, s1, s2])  // [B, 64]
//     或使用注意力加权：weighted_sum based on critic value
```

**验证**：训练 200 轮，对比 actor loss 和 avg_reward

**预期**：actor loss 从 -0.29 改善到 +1.0 左右

---

### Phase 2：高级损失函数优化（P2，预计改善 1.2-1.5×）

#### 2A：感知损失（Perceptual Loss）
**原理**：用预训练 CNN（如 VGG）提取特征，匹配高层语义而非像素值：
```
L_percep = MSE(VGG(pred), VGG(target))  // 在 conv3_3 层
```

**优势**：
- 对球的形状/颜色更敏感
- 减少对背景纹理的过度关注

**实现**：
- 使用 `burn-vision` 加载预训练 VGG
- 冻结权重，只提取特征
- 与 pixel-wise MSE 混合：`0.5 * L_mse + 0.5 * L_percep`

**障碍**：Burn 生态可能没有现成的预训练 VGG，需要：
1. 从 PyTorch 导出权重
2. 手动实现 VGG 前 3 层
3. 或使用更简单的 proxy（如 Sobel edge detector）

---

#### 2B：对抗性训练（GAN Loss）
**原理**：训练判别器 `D(x)` 判断图像真假，decoder 优化 `L_adv = -log(D(pred))`

**优势**：
- 迫使生成的球更真实
- 边界更锐利（GAN 擅长细节）

**风险**：
- 训练不稳定（需要仔细调 D/G 学习率）
- 可能让 RSSM 学会"作弊"（只优化视觉真实度而非语义）

**建议**：仅在 Phase 0 和 Phase 1 完成后，ball MSE 仍 > 0.03 时尝试

---

#### 2C：多尺度损失
**原理**：在多个分辨率计算 MSE：
```
L_multi = MSE(pred_64, target_64) 
        + 0.5 * MSE(downsample(pred_64, 32), downsample(target_64, 32))
        + 0.25 * MSE(downsample(pred_64, 16), downsample(target_64, 16))
```

**优势**：
- 低分辨率约束全局结构（球的位置）
- 高分辨率约束细节（球的边界）

**实现简单**：用 `avg_pool2d` 下采样即可

---

### Phase 3：数据与训练策略（P3，预计改善 1.1-1.2×）

#### 3A：球区自适应采样
**原理**：每个 batch 额外采样 2× 的"有球"帧（通过检测像素和 > 某阈值）：
```rust
// 训练时：70% 正常采样，30% 强制采样有球帧
let has_ball = obs.sum() > threshold;
if has_ball { sample_prob *= 3.0; }
```

**效果**：让 decoder 看到更多球的样本，打破"背景多球少"的数据偏差

---

#### 3B：Curriculum Learning（课程学习）
**原理**：
1. 前 50 轮：球移动慢（`v_max = 2`），decoder 容易学
2. 50-100 轮：正常速度（`v_max = 4`）
3. 100+ 轮：高速（`v_max = 6`）

**实现**：通过环境变量 `BALL_SPEED_SCHEDULE`

---

#### 3C：增加球的对比度
**环境修改**：
```rust
// src/envs/bouncing_ball.rs
let ball_color = [255, 100, 100];  // 原：浅红
// 改为：
let ball_color = [255, 0, 0];  // 纯红，与背景对比度更高
```

**效果**：降低 decoder 学习难度（可能改善 5-10%）

---

## 🗓️ 实施时间表

| Week | 任务 | 验证指标 | 里程碑 |
|------|------|----------|--------|
| **W1** | 实现并测试 Spatial Broadcast Decoder（0A） | 单元测试 ball < 0.05 | 解码器瓶颈突破 |
| **W2** | 集成到 OC，训练 200 轮 | `ball < 0.04` | Phase 0 完成 |
| **W3** | 应用 1A（ensemble）+ 1B（no interaction）+ 1C（mean-pool actor） | `ball < 0.03` | Phase 1 完成 |
| **W4** | 实现多尺度损失（2C）+ 球区采样（3A） | `ball < 0.025` | 接近目标 |
| **W5** | 如需要：感知损失（2A）或高斯渲染器（0B） | `ball < 0.02` | 🎯 达成目标 |

---

## 📊 验证协议

每个 phase 完成后运行标准测试：
```bash
# 200 轮训练，记录完整日志
cargo run --release 2>&1 | tee logs/phase_X.log

# 提取关键指标（ep 199）
grep "round 199" logs/phase_X.log

# 生成对比视频
cargo run --release --example compare_videos
```

**通过标准**：
- `ball < 0.02`（主目标）
- `kl < 2.0`（训练稳定）
- `actor > 0`（策略有效）
- `avg_reward > 200`（环境掌握）

---

## 🚨 风险与备选方案

### 风险 1：Spatial Broadcast Decoder 仍不够（ball 停在 0.04）
**备选方案**：切换到解析式高斯渲染器（0B），用域知识直接建模球体

### 风险 2：感知损失在 Burn 中难以实现
**备选方案**：用手工 Sobel 边缘检测器代替 VGG：
```rust
let edge_pred = sobel(pred);
let edge_target = sobel(target);
L_edge = MSE(edge_pred, edge_target);
```

### 风险 3：即使 ball < 0.02，agent 策略仍然很差
**诊断**：
- 如果 `actor < 0`：actor 输入表示有问题（应用 1C）
- 如果 `avg_reward < 150`：想象质量差（检查 `kl` 和 `pred`）
- 如果 slot_kl 不均衡：某些 slot 崩溃（调整 `free_nats` 或 slot loss 权重）

---

## 💡 关键洞察（基于实验结果）

1. **解码器是瓶颈**：实验 3c 证明 BroadcastDecoder 架构性无法重建球体，这是优先级最高的修复点
2. **运动加权是必要的**：实验 2 显示关闭后 ball 恶化 2.2×，必须保留
3. **Ensemble 是最佳正则化器**：实验 B 显示它同时改善 KL、actor 和 slot 均衡性
4. **跨 slot 交互当前是噪声**：实验 C 显示关闭后 actor 改善，当前实现可能有问题
5. **单体 baseline 的 actor 更好**：实验 D 提示 OC 的 192 维拼接输入可能冗余

---

## 📝 开发检查清单

### Phase 0（解码器）
- [ ] 实现 `SpatialBroadcastDecoder`（`src/networks/spatial_broadcast_decoder.rs`）
- [ ] 单元测试：给定真实球坐标，验证 ball MSE < 0.05
- [ ] 集成到 `OCWorldModel`，替换 `BroadcastDecoder`
- [ ] 训练 200 轮，记录 `logs/phase0.log`
- [ ] 如失败：实现 `GaussianBlobDecoder`（0B）

### Phase 1（架构优化）
- [ ] `ENSEMBLE_HEADS = 2`（默认开启）
- [ ] `SLOT_INTERACTION = false`（默认关闭）
- [ ] Actor 改用 `mean_pool(slots)` 而非 `concat(slots)`
- [ ] 训练 200 轮，记录 `logs/phase1.log`

### Phase 2（高级损失）
- [ ] 实现多尺度损失（`src/tools.rs::multi_scale_mse`）
- [ ] 如需要：实现 Sobel 边缘损失
- [ ] 如需要：实现对抗训练（谨慎）

### Phase 3（数据策略）
- [ ] 球区自适应采样
- [ ] 提高球的颜色对比度（环境修改）

---

## 🎯 成功标准

**最终目标**：
```
ep 199:
  ball: < 0.02      ✅
  kl: < 2.0         ✅
  actor: > 0        ✅
  avg_reward: > 200 ✅
```

当所有指标同时满足时，表示 OC-Dreamer 在 BouncingBall 环境中达到预期性能。
