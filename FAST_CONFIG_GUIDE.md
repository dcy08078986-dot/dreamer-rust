# 🚀 快速实验配置

## 问题：原配置太慢

200 轮实验需要 1-2 小时，主要耗时在：
- 200 个 episodes
- 每个 episode 200 步
- batch_length=8（长序列训练）
- imag_horizon=15（想象轨迹长）

---

## ⚡ 快速配置（加速 4-6×）

### 方式 1：使用快速脚本（推荐）

```bash
chmod +x run_phase0_fast.sh
./run_phase0_fast.sh
```

**加速措施**：
- Episodes: 200 → **30**（6.7× 更少）
- Env steps: 200 → **100**（2× 更短）
- Batch length: 8 → **4**（序列更短）
- Imag horizon: 15 → **8**（想象更短）

**预计时间**：15-20 分钟（两个实验）

---

### 方式 2：极速单次测试（5 分钟）

只测试新解码器，跳过 baseline：

```bash
DREAMER_SPATIAL_BROADCAST=1 \
DREAMER_ENSEMBLE=1 \
DREAMER_INTERACTION=0 \
DREAMER_EPISODES=20 \
cargo run --release 2>&1 | tee logs/quick_test.log

# 查看结果
grep "round 19" logs/quick_test.log
```

---

### 方式 3：更激进的加速（3 分钟，仅验证运行）

```bash
DREAMER_SPATIAL_BROADCAST=1 \
DREAMER_ENSEMBLE=1 \
DREAMER_INTERACTION=0 \
DREAMER_EPISODES=10 \
DREAMER_VIDEO_INTERVAL=999 \
cargo run --release
```

**注意**：10 轮不足以评估性能，仅用于验证代码运行正常

---

## 📊 如何解读快速实验结果

### 30 轮结果的可靠性

| 指标 | 30 轮 | 200 轮（完整） | 可靠性 |
|------|-------|----------------|--------|
| **ball** | 有参考价值 | 完整收敛 | ⚠️ 可能未收敛 |
| **kl** | 趋势明确 | 稳定值 | ✅ 可靠 |
| **actor** | 趋势明确 | 稳定值 | ✅ 可靠 |
| **相对改善** | ✅ 可靠 | ✅ 可靠 | ✅ **最重要** |

**关键**：看 **相对改善**（新解码器 vs 旧解码器的比值），而非绝对值

### 判断标准（30 轮）

**成功信号**：
```
ball_spatial / ball_baseline < 0.6  （改善 ≥40%）
```

**如果 30 轮显示明显改善** → 值得跑完整 200 轮  
**如果 30 轮无改善** → 直接诊断问题，节省时间

---

## 🔧 进一步加速技巧

### 1. 减少环境数量（默认 4 → 2）

```bash
# 在 src/config.rs 修改默认值
num_envs: 2,  // 原本 4
```

或通过环境变量（需要添加支持）：
```bash
DREAMER_NUM_ENVS=2 cargo run --release
```

### 2. 减少 slot 数量（3 → 2）

```bash
DREAMER_SLOTS=2 \
DREAMER_SPATIAL_BROADCAST=1 \
DREAMER_EPISODES=30 \
cargo run --release
```

**权衡**：可能影响表征质量，但加速约 30%

### 3. 使用更小的批次

在 `src/config.rs` 修改：
```rust
batch_size: 2,      // 默认已经是 2
batch_length: 4,    // 从 8 改为 4（通过环境变量）
```

---

## 📈 推荐的实验流程

### 第 1 步：快速验证（5 分钟）
```bash
# 确认新解码器能正常运行
DREAMER_SPATIAL_BROADCAST=1 DREAMER_EPISODES=10 cargo run --release
```

### 第 2 步：趋势检查（15-20 分钟）
```bash
# 运行快速脚本，看相对改善
./run_phase0_fast.sh
```

### 第 3 步：完整评估（如果第 2 步有改善，1-2 小时）
```bash
# 运行完整 200 轮
./run_phase0.sh
```

**总时间节约**：
- 如果快速测试失败，省下 1.5 小时
- 如果快速测试成功，只多花 20 分钟

---

## 💡 哪些参数影响训练速度

| 参数 | 默认值 | 快速值 | 加速 | 副作用 |
|------|--------|--------|------|--------|
| `episodes` | 200 | 30 | 6.7× | 可能未收敛 |
| `env_max_steps` | 200 | 100 | 2× | 轨迹更短 |
| `batch_length` | 8 | 4 | 1.3× | 梯度噪声更大 |
| `imag_horizon` | 15 | 8 | 1.2× | 想象更短 |
| `num_envs` | 4 | 2 | 1.5× | 样本多样性降低 |
| `num_slots` | 3 | 2 | 1.3× | 表征能力降低 |

**综合加速**：6.7 × 2 × 1.3 = **17×** 理论加速（实际约 6-8×，因为有固定开销）

---

## 🎯 快速配置的局限性

**30 轮快速测试不能替代完整 200 轮**，原因：
1. KL 可能未充分下降
2. Slot attention 可能未学会正确分配
3. Actor 策略可能不稳定

**但快速测试足以回答**：
- ✅ 新解码器是否比旧解码器更好？
- ✅ 改善幅度是否值得继续？
- ✅ 代码是否有 bug？

---

## 🚀 立即开始

**最简单**：
```bash
./run_phase0_fast.sh
```

**最快**（单个实验）：
```bash
DREAMER_SPATIAL_BROADCAST=1 DREAMER_ENSEMBLE=1 DREAMER_INTERACTION=0 \
DREAMER_EPISODES=20 cargo run --release
```

查看结果：
```bash
# 最后一轮
grep "round 19" logs/quick_test.log

# 或查看趋势
grep -E "round (4|9|14|19)" logs/quick_test.log
```
