#!/bin/bash
# 快速调试配置：Phase 0 实验（30 轮，加速版）

echo "=========================================="
echo "Phase 0 快速测试（30轮，约15-20分钟）"
echo "=========================================="

# 快速配置参数（通过减少训练量加速）
# - 减少 episodes: 200 → 30
# - 减少 env_max_steps: 200 → 100（每个 episode 更短）
# - 减少 batch_length: 8 → 4（序列更短）
# - 减少 imag_horizon: 15 → 8（想象更短）

export DREAMER_ENV_MAX_STEPS=100
export DREAMER_BATCH_LENGTH=4
export DREAMER_IMAG_HORIZON=8

# 实验 0A: 新解码器 (Spatial Broadcast)
echo ""
echo "实验 0A: SpatialBroadcast + 快速配置"
DREAMER_SPATIAL_BROADCAST=1 \
DREAMER_ENSEMBLE=1 \
DREAMER_INTERACTION=0 \
DREAMER_EPISODES=30 \
cargo run --release 2>&1 | tee logs/phase0_spatial_fast.log

echo ""
echo "实验 0A 完成！查看结果："
echo "最后一轮 (ep 29):"
grep "round 29" logs/phase0_spatial_fast.log | tail -1

echo ""
echo "ball 指标趋势（每10轮）:"
grep -E "round (9|19|29)" logs/phase0_spatial_fast.log | grep -oE "ball=[0-9.]+|round [0-9]+"

# 实验 0B: 旧解码器 baseline
echo ""
echo "=========================================="
echo "实验 0B: BroadcastDecoder + 快速配置"
DREAMER_SPATIAL_BROADCAST=0 \
DREAMER_ENSEMBLE=1 \
DREAMER_INTERACTION=0 \
DREAMER_EPISODES=30 \
cargo run --release 2>&1 | tee logs/phase0_baseline_fast.log

echo ""
echo "实验 0B 完成！查看结果："
echo "最后一轮 (ep 29):"
grep "round 29" logs/phase0_baseline_fast.log | tail -1

echo ""
echo "=========================================="
echo "Phase 0 快速测试完成！对比结果："
echo "=========================================="
echo ""
echo "Baseline (BroadcastDecoder) - ep 29:"
grep "round 29" logs/phase0_baseline_fast.log | tail -1 | grep -oE "ball=[0-9.]+"
echo ""
echo "New (SpatialBroadcastDecoder) - ep 29:"
grep "round 29" logs/phase0_spatial_fast.log | tail -1 | grep -oE "ball=[0-9.]+"
echo ""
echo "注意：30轮的结果仅供趋势参考，完整评估需要 200 轮"
