#!/bin/bash
# Phase 0 实验：对比新旧解码器

echo "=========================================="
echo "Phase 0: Spatial Broadcast Decoder 测试"
echo "=========================================="

# 实验 0A: 新解码器 (Spatial Broadcast) + Ensemble + No Interaction
echo ""
echo "实验 0A: SpatialBroadcast + Ensemble + NoInteraction"
echo "预期: ball < 0.04"
DREAMER_SPATIAL_BROADCAST=1 \
DREAMER_ENSEMBLE=1 \
DREAMER_INTERACTION=0 \
DREAMER_EPISODES=200 \
cargo run --release 2>&1 | tee logs/phase0_spatial.log

# 实验 0B: 旧解码器 baseline (当前最佳配置)
echo ""
echo "实验 0B: BroadcastDecoder + Ensemble + NoInteraction (baseline)"
echo "预期: ball = 0.111 (已知)"
DREAMER_SPATIAL_BROADCAST=0 \
DREAMER_ENSEMBLE=1 \
DREAMER_INTERACTION=0 \
DREAMER_EPISODES=200 \
cargo run --release 2>&1 | tee logs/phase0_baseline.log

echo ""
echo "=========================================="
echo "Phase 0 完成！对比结果："
echo "=========================================="
echo ""
echo "Baseline (BroadcastDecoder):"
grep "round 199" logs/phase0_baseline.log | tail -1
echo ""
echo "New (SpatialBroadcastDecoder):"
grep "round 199" logs/phase0_spatial.log | tail -1
echo ""
echo "关键指标对比 (ep 199):"
echo "Metric       | Baseline | Spatial  | 改善"
echo "-------------|----------|----------|------"
printf "ball         | "
grep "round 199" logs/phase0_baseline.log | tail -1 | grep -oE "ball=[0-9.]+" | cut -d= -f2 | tr '\n' ' '
printf "| "
grep "round 199" logs/phase0_spatial.log | tail -1 | grep -oE "ball=[0-9.]+" | cut -d= -f2 | tr '\n' ' '
printf "|\n"
