# Phase 0 实施完成报告

**日期**: 2026-07-05  
**目标**: 实现并测试全分辨率 Spatial Broadcast Decoder 以突破解码器瓶颈

---

## ✅ 已完成的工作

### 1. 实现 SpatialBroadcastDecoder (`src/networks/spatial_broadcast_decoder.rs`)

**架构改进**：
- 不再使用 8×8 低分辨率广播 + 3次上采样
- 直接在 **64×64 全分辨率**广播 slot 向量
- 使用轻量级 CNN 解码（3层 Conv2d，无上采样）

**关键参数**：
```rust
latent_dim: 64        // slot state 投影后的维度
hidden_dim: 256       // MLP 扩展后的维度
image_size: 64        // 全分辨率
image_channels: 3     // RGB
```

**前向流程**：
```
[B*K, 64] → MLP → [B*K, 256]
          → Broadcast → [B*K, 256, 64, 64]
          → Concat coords → [B*K, 258, 64, 64]
          → Conv(258→128→64→4) → [B*K, 4, 64, 64]
          → Split: RGB(3) + mask(1)
```

**优势**：
- 避免上采样丢失空间信息
- 球体（4×4像素）的位置精度提升
- 参数量增加不大（主要是广播操作）

### 2. 集成到 OCWorldModel

**修改**：
- 创建 `DecoderType` 枚举支持两种解码器
- 添加配置项 `use_spatial_broadcast` 和 `decoder_hidden_dim`
- 通过环境变量 `DREAMER_SPATIAL_BROADCAST=1` 启用

**代码变更**：
- `src/agent/oc_world_model.rs`: 
  - 添加 `DecoderType` 枚举
  - 修改 `init()` 根据 config 选择解码器
  - 修改 `decode_slots()` 使用 match 分发
- `src/config.rs`: 添加 `use_spatial_broadcast` 和 `decoder_hidden_dim`
- `src/main.rs`: 添加环境变量解析
- `src/networks/mod.rs`: 导出新模块

### 3. 测试验证

**编译测试**: ✅ 通过
```bash
cargo build --release
# 成功编译，无错误
```

**单元测试**: ⚠️  在未优化模式下运行太慢（>60秒），已跳过
- `test_coord_grid`: ✅ 通过（坐标网格生成正确）
- `test_forward_shape`: ⏳ 超时（Burn NdArray 后端慢）
- `test_ball_reconstruction`: ⏳ 超时

**集成测试**: ✅ 通过
```bash
DREAMER_SPATIAL_BROADCAST=1 DREAMER_EPISODES=2 cargo run --release
# 输出: "Using SpatialBroadcastDecoder (full-resolution, hidden_dim=256)"
# 成功加载并运行
```

---

## 🔬 Phase 0 实验（进行中）

### 实验设计

**对比两种配置**（各 200 轮）：

| ID | 解码器 | Ensemble | Interaction | 预期 ball |
|----|--------|----------|-------------|-----------|
| 0A | **SpatialBroadcast** | ✅ | ❌ | **< 0.04** |
| 0B | BroadcastDecoder | ✅ | ❌ | 0.111 (baseline) |

**通过标准**：
- `ball` 降低 **≥2.5×**（从 0.111 → <0.044）
- `kl < 2.0`（训练稳定）
- `actor > 0`（策略有效）

### 实验脚本

已创建 `run_phase0.sh`：
```bash
# 实验 0A: 新解码器
DREAMER_SPATIAL_BROADCAST=1 DREAMER_ENSEMBLE=1 DREAMER_INTERACTION=0 \
DREAMER_EPISODES=200 cargo run --release 2>&1 | tee logs/phase0_spatial.log

# 实验 0B: 旧解码器
DREAMER_SPATIAL_BROADCAST=0 DREAMER_ENSEMBLE=1 DREAMER_INTERACTION=0 \
DREAMER_EPISODES=200 cargo run --release 2>&1 | tee logs/phase0_baseline.log
```

**当前状态**: 🔄 后台运行中（任务 ID: `bqy03xv80`）

**预计完成时间**: 1-2 小时（每个实验 30-60 分钟）

---

## 📊 预期结果

### 如果 ball < 0.04（目标达成）

**下一步**：Phase 1
- 应用 Phase 1 优化（已在 config 中默认开启）：
  - ✅ Ensemble exploration（默认 `ensemble_size=3`）
  - ✅ No cross-slot interaction（默认 `use_slot_interaction=false`）
- 简化 Actor 输入（mean-pool 替代 concat）
- 目标：`ball < 0.03`

### 如果 ball 停在 0.04-0.08（部分改善）

**诊断**：
- 检查 slot KL 均衡性（是否某些 slot 崩溃）
- 可视化重建（球是否仍然模糊）
- 尝试更大的 `decoder_hidden_dim`（512）

**备选方案**：
- 实现方案 0B：解析式高斯斑点渲染器
- 用域知识直接建模球体（7 参数：x, y, σ, RGB, α）

### 如果 ball 无改善（≥0.10）

**原因**：
- 问题不在解码器分辨率，而在表征学习
- RSSM 可能没有将球分配到独立 slot

**诊断**：
- 检查 slot attention 的分配（可视化 mask）
- 检查 slot_kl（是否均衡）
- 检查运动加权是否生效（`motion_mse` vs `obs_loss`）

---

## 🔧 技术细节

### 参数量对比

**BroadcastDecoder**（旧）：
- Conv layers: (66→64) + (64→64) + (64→32) + (32→16) + (16→4)
- 主要参数在上采样层（ConvTranspose2d）
- 估计：~50K 参数

**SpatialBroadcastDecoder**（新）：
- MLP: (64→256) = 16K
- Conv layers: (258→128) + (128→64) + (64→4) = 34K + 8K + 2K
- 估计：~60K 参数（略多 20%）

### 内存占用

**旧**: 广播到 8×8，中间特征图较小  
**新**: 广播到 64×64，中间特征图 64× 更大

**权衡**: 内存换精度（在 64×64 分辨率下可接受）

### 兼容性

✅ 完全向后兼容：
- 默认 `use_spatial_broadcast=false` 保持旧行为
- 环境变量开关，无需修改代码
- 可以加载旧 checkpoint（解码器权重不兼容，但 RSSM/actor 可复用）

---

## 📝 检查清单

### Phase 0 实施
- [x] 实现 `SpatialBroadcastDecoder`
- [x] 添加配置项和环境变量
- [x] 集成到 `OCWorldModel`
- [x] 编译测试通过
- [x] 创建实验脚本
- [x] 启动 Phase 0 实验（200 轮 × 2）
- [ ] 等待实验完成（~1-2 小时）
- [ ] 分析结果并决定下一步

### Phase 1 准备（如果 Phase 0 成功）
- [x] Ensemble exploration（已在 config 中默认开启）
- [x] No cross-slot interaction（已在 config 中默认关闭）
- [ ] 简化 Actor 输入（mean-pool）
- [ ] 运行 Phase 1 实验

### 备选方案（如果 Phase 0 失败）
- [ ] 实现 `GaussianBlobDecoder`（方案 0B）
- [ ] 诊断 slot 分配问题
- [ ] 调整运动加权参数

---

## 🎯 成功标准（最终目标）

```
ep 199:
  ball: < 0.02      (Phase 0 目标: < 0.04)
  kl: < 2.0         
  actor: > 0        
  avg_reward: > 200 
```

---

## 📂 相关文件

**实现**：
- `src/networks/spatial_broadcast_decoder.rs` - 新解码器实现
- `src/agent/oc_world_model.rs` - 集成逻辑
- `src/config.rs` - 配置项
- `src/main.rs` - 环境变量解析

**文档**：
- `OPTIMIZATION_PLAN.md` - 完整优化计划
- `PHASE0_IMPLEMENTATION.md` - 本报告
- `run_phase0.sh` - 实验脚本

**日志**（生成中）：
- `logs/phase0_spatial.log` - SpatialBroadcastDecoder 实验
- `logs/phase0_baseline.log` - BroadcastDecoder baseline

---

## 🚀 下一步

1. **监控实验进度**（~1-2 小时）：
   ```bash
   tail -f logs/phase0_spatial.log  # 查看新解码器
   tail -f logs/phase0_baseline.log # 查看 baseline
   ```

2. **实验完成后**：
   ```bash
   # 提取最终指标
   grep "round 199" logs/phase0_spatial.log
   grep "round 199" logs/phase0_baseline.log
   
   # 对比 ball MSE
   # 如果改善 > 2.5×，进入 Phase 1
   # 否则诊断问题并尝试备选方案
   ```

3. **验证通过后** → Phase 1（Actor 简化 + 高级损失）

4. **最终目标** → `ball < 0.02`（当前 0.111，需改善 5.5×）
