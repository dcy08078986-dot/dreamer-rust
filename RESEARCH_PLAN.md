# 研究论文计划：World Model的因果推理与反事实学习

## 标题
**Causal World Models: Learning Interventional Dynamics for Counterfactual Reasoning in Model-Based Reinforcement Learning**

因果世界模型：模型基强化学习中的干预动力学学习与反事实推理

---

## 核心创新点

### 1. 因果RSSM（Causal-RSSM）
**问题**：传统RSSM只学习观察相关性，无法区分因果关系和虚假相关
- 现有DreamerV2/V3：`p(s_t | s_{t-1}, a_{t-1})`仅建模条件分布
- **创新**：引入结构因果模型（SCM）到潜在状态空间

**方法**：
```
传统RSSM: z_t ~ p(z_t | h_t, o_t)
因果RSSM: z_t ~ p(z_t | do(a_{t-1}), h_t, o_t)
          其中 do(·) 表示Pearl的因果干预算子
```

实现：
- **外生变量分离**：`z = [z_causal, z_spurious]`
  - z_causal: 受动作因果影响的状态
  - z_spurious: 相关但非因果的混淆因素（如背景、光照）
- **因果注意力机制**：动态调整哪些状态维度受动作影响
- **对比损失**：最大化同一轨迹不同动作下的状态差异

### 2. 反事实想象（Counterfactual Imagination）
**问题**：现有想象rollout只能"如果我这样做会怎样"，无法回答"如果我当时那样做会怎样"

**创新**：引入反事实推理到策略学习
```python
# 传统想象
s'_t = imagine_forward(s_t, a_new)  # 从当前状态向前

# 反事实想象（创新）
s_cf = counterfactual_imagine(
    s_actual_t,      # 实际到达的状态
    a_actual,        # 实际采取的动作
    a_counterfactual # 反事实动作（如果当时这样做）
)
```

**实现机制**：
1. **Abduction**：从观察反推外生变量 `u ~ p(u | s_t, a_t, s_{t+1})`
2. **Action**：替换为反事实动作 `a' ≠ a`
3. **Prediction**：重新前向推理 `s'_{t+1} = f(s_t, a', u)`

**应用**：
- 信用分配：哪个过去动作导致了当前失败？
- 探索策略：如果之前选择不同，现在会在哪里？
- 后见之明经验回放：将失败轨迹转化为成功轨迹

### 3. 因果图学习（Causal Graph Discovery）
**问题**：手工指定因果结构，不适应不同环境

**创新**：从数据自动发现状态-动作因果图
- 使用可微因果发现（如NOTEARS、GraN-DAG）
- 学习邻接矩阵 `A[i,j] = 1` 表示 `s_j` 因果影响 `s_i`
- 稀疏化正则：`||A||_1` 鼓励简洁因果图

**动态因果图**：
- 不同环境/任务 → 不同因果结构
- 元学习快速适应新因果图
- 零样本泛化到新动态

### 4. 对比反事实正则化（Contrastive Counterfactual Regularization）
**问题**：如何让模型真正学到因果而非相关？

**创新方法**：
```python
# 数据增强生成反事实对
(s_t, a, s_{t+1})         # 实际轨迹
(s_t, a', s'_{t+1})       # 反事实轨迹（同状态不同动作）

# 对比损失
L_contrast = -log[ 
    exp(sim(Δs, Δa) / τ) / 
    Σ exp(sim(Δs, Δa_neg) / τ)
]
其中 Δs = s_{t+1} - s_t, Δa = embed(a)
```

**效果**：
- 强制模型关注动作导致的状态变化
- 消除虚假相关（如"按左键时天总是蓝色"）
- 提升样本效率：一条轨迹生成多个反事实样本

---

## 理论贡献

### 1. 因果世界模型的表示定理
**定理1（因果充分性）**：如果潜在因果图G包含真实环境的所有因果边，则Causal-RSSM能够完美预测任意干预下的状态转移。

**定理2（反事实可识别性）**：在马尔可夫性和外生独立性假设下，反事实状态 `s_cf` 可从观察数据唯一识别。

### 2. 泛化界限
**定理3（因果泛化界）**：设 `d_causal` 为因果图的最大入度，则OOD泛化误差：
```
ε_OOD ≤ ε_train + O(d_causal · √(n_train))
```
优于非因果模型的 `O(d_total · √(n_train))`

### 3. 样本复杂度分析
**定理4**：通过反事实增强，有效样本数从 `N` 增至 `N × |A|^k`，其中 `k` 为反事实深度。

---

## 实验设计

### 基准环境（验证有效性）

#### 1. 因果混淆环境（Causal Confounding Tasks）
**设计**：
- 环境A：球颜色与重力相关（红球=低重力，蓝球=高重力）
- 任务：学会跳跃到高台
- **混淆因素**：颜色与重力同时出现，非因果模型会过拟合颜色

**评估**：
- 训练：红球低重力
- 测试：红球高重力（颜色-重力解耦）
- 指标：非因果模型失败，因果模型成功

#### 2. 反事实信用分配（Counterfactual Credit Assignment）
**环境**：改进的BouncingBall
- 必须先收集"能量球"，然后才能跳高
- 稀疏奖励：只在最后给分

**对比方法**：
- 基线：标准Dreamer（时间信用分配困难）
- **我们的**：反事实推理"如果没收集能量球会怎样"→快速定位关键决策

#### 3. 零样本动态迁移（Zero-Shot Dynamics Transfer）
**设置**：
- 训练环境：正常重力（g=0.002）
- 测试环境：月球重力（g=0.0005），火星重力（g=0.001）
- **不允许**在新重力下训练

**评估**：
- 非因果模型：完全失败（记住了特定轨迹）
- 因果模型：学到了 `Δy = v_y - g` 的因果规律→迁移

### 标准基准（SOTA对比）

#### DMControl Suite
- 选择：Cartpole, Reacher, Walker, Quadruped
- 指标：样本效率曲线（100K/500K/1M步）
- 对比：DreamerV3, TD-MPC2, FOWM

#### Atari 100K
- 26款游戏
- 指标：人类归一化得分
- 重点游戏：Montezuma's Revenge（需要长期信用分配）

#### RoboDesk（真实机器人迁移）
- 训练：模拟器
- 测试：真实Franka Panda机器人
- 任务：开抽屉、按按钮
- **因果优势**：模拟器↔真实 物理参数不同，因果结构相同

---

## 实现细节

### 网络架构

#### Causal-RSSM
```python
class CausalRSSM(nn.Module):
    def __init__(self, deter=256, stoch=32, action=6):
        self.gru = GRU(stoch + action, deter)
        
        # 因果分解
        self.causal_prior = MLP(deter, stoch // 2 * 2)  # μ, σ
        self.spurious_prior = MLP(deter, stoch // 2 * 2)
        
        # 因果注意力
        self.causal_attn = CausalAttention(deter, action)
        
        # 因果图（可学习邻接矩阵）
        self.adj_matrix = nn.Parameter(torch.randn(stoch, stoch))
        
    def forward(self, s_prev, a, obs):
        h = self.gru(concat(s_prev.stoch, a))
        
        # 动作影响mask
        causal_mask = self.causal_attn(h, a)  # [batch, stoch//2]
        
        # 分离采样
        z_causal = sample_gaussian(self.causal_prior(h)) * causal_mask
        z_spurious = sample_gaussian(self.spurious_prior(h))
        z = concat(z_causal, z_spurious)
        
        # 因果图约束
        z = z @ soft_threshold(self.adj_matrix)  # 稀疏化
        
        return RSSMState(h, z, ...)
```

#### 反事实模块
```python
class CounterfactualImagination(nn.Module):
    def abduction(self, s_t, a_t, s_next):
        """反推外生噪声"""
        u = self.encoder_u(s_t, a_t, s_next)
        return u
    
    def counterfactual(self, s_t, a_actual, a_cf, u):
        """生成反事实状态"""
        # 替换动作
        s_cf = self.rssm.img_step(s_t, a_cf, exogenous=u)
        return s_cf
```

### 训练流程

```python
# 标准世界模型训练
for batch in replay_buffer:
    # 1. 前向预测
    prior = rssm.img_step(s[t], a[t])
    post = rssm.obs_step(prior, obs[t+1], a[t])  # 我们已修复的版本
    
    # 2. 标准损失
    L_recon = ||decoder(post) - obs[t+1]||²
    L_reward = ||predict_reward(post) - r[t]||²
    L_kl = KL(post || prior)
    
    # 3. 因果损失（新增）
    # 3a. 对比因果
    a_neg = sample_different_action()
    s_neg = imagine(s[t], a_neg)
    L_causal = contrastive_loss(
        positive=(s[t+1] - s[t], a[t]),
        negative=(s_neg - s[t], a_neg)
    )
    
    # 3b. 反事实一致性
    u = abduction(s[t], a[t], s[t+1])
    s_cf = counterfactual(s[t], a[t], a_cf, u)
    s_cf_recon = forward(s[t], a_cf)  # 直接前向
    L_cf = ||s_cf - s_cf_recon||²  # 两种路径应一致
    
    # 3c. 因果图稀疏性
    L_sparse = ||adj_matrix||_1
    
    # 总损失
    L_total = L_recon + 0.5*L_reward + 0.8*L_kl + 
              0.3*L_causal + 0.2*L_cf + 0.01*L_sparse
```

---

## 预期结果

### 定量指标
1. **样本效率**：达到相同性能所需样本数
   - 目标：比DreamerV3减少30-50%
   
2. **零样本泛化**：OOD环境上的成功率
   - 目标：>80%（vs 基线<50%）

3. **因果发现准确率**：恢复真实因果图的F1
   - 目标：>0.85（在已知因果图的合成环境）

### 定性分析
1. **可解释性可视化**：
   - 显示学到的因果图
   - 反事实想象轨迹对比
   - 注意力热图（哪些状态受动作影响）

2. **消融研究**：
   - 移除因果分解 → 性能下降
   - 移除反事实 → 信用分配变差
   - 移除对比损失 → 学到虚假相关

---

## 相关工作对比

| 方法 | 因果建模 | 反事实推理 | 结构学习 | OOD泛化 |
|------|---------|-----------|---------|---------|
| DreamerV3 | ✗ | ✗ | ✗ | 弱 |
| TD-MPC2 | ✗ | ✗ | ✗ | 弱 |
| CIRL (offline RL) | ✓ | ✗ | ✗ | 中 |
| CausalMBRL (prior work) | ✓ | ✗ | ✗ | 中 |
| **Ours** | ✓ | ✓ | ✓ | 强 |

**关键区别**：
- 现有因果RL主要用于offline设置
- 我们首次将反事实推理引入world model的在线学习
- 结合因果发现+反事实想象+世界模型 = 三重创新

---

## 实施时间线

### Phase 1: 核心算法（2个月）
- Week 1-2: 实现Causal-RSSM架构
- Week 3-4: 反事实想象模块
- Week 5-6: 因果图学习
- Week 7-8: 集成训练流程

### Phase 2: 验证实验（1.5个月）
- Week 9-10: 因果混淆环境实验
- Week 11-12: 反事实信用分配测试
- Week 13-14: 零样本迁移实验

### Phase 3: 标准基准（2个月）
- Week 15-17: DMControl Suite（500K步×8任务）
- Week 18-20: Atari 100K（100K步×26游戏）
- Week 21-22: 真实机器人迁移（如有条件）

### Phase 4: 论文撰写（1个月）
- Week 23-24: 消融研究+可视化
- Week 25-26: 初稿完成+内部审阅

**总计：6.5个月**

---

## 预期发表目标

### 顶会
- **NeurIPS 2026**（Deadline: 5月）：ML+RL双轨
- **ICLR 2027**（Deadline: 10月）：偏重因果推理
- **ICML 2027**（Deadline: 1月）：备选

### 理由
- **原创性**：首次系统性地将因果推理引入世界模型
- **理论贡献**：因果泛化界、反事实可识别性定理
- **实证强度**：合成+标准+真实机器人 三层验证
- **影响力**：推动model-based RL向可解释、可泛化方向发展

---

## 技术风险与缓解

### 风险1：因果图学习不稳定
- **缓解**：先在小规模状态空间验证，再扩展；使用预训练+微调

### 风险2：反事实计算昂贵
- **缓解**：
  - 异步计算反事实（不阻塞主训练）
  - 只在关键时刻采样（如失败轨迹）
  - 缓存反事实结果

### 风险3：标准基准提升不明显
- **缓解**：
  - 重点展示零样本迁移等优势
  - 强调可解释性和样本效率
  - 即使性能相当，因果理解仍是贡献

---

## 代码开源计划

**仓库结构**：
```
causal-world-models/
├── src/
│   ├── models/
│   │   ├── causal_rssm.py
│   │   ├── counterfactual.py
│   │   └── causal_graph.py
│   ├── envs/
│   │   ├── causal_confounding.py
│   │   └── bouncing_ball_causal.py  # 扩展你的环境
│   └── train/
│       └── causal_dreamer.py
├── experiments/
│   ├── dmcontrol/
│   ├── atari/
│   └── causality_tests/
├── notebooks/
│   └── causal_visualization.ipynb
└── README.md
```

**开源时机**：论文接收后立即开源
**目标**：成为因果RL的标准库

---

## 总结

这个方向结合了：
1. **因果推理**（Pearl, Schölkopf）- 当前AI热点
2. **世界模型**（Hafner et al.）- MBRL前沿
3. **反事实学习**（Buesing et al.）- 理论深度

**为什么是研究级创新**：
- ✅ 理论贡献：新定理+泛化界
- ✅ 方法创新：3个核心模块全新设计
- ✅ 实证全面：合成+标准+真实机器人
- ✅ 实用价值：可解释+零样本泛化

**与你的项目结合**：
- 可以直接在当前Dreamer-Rust代码上扩展
- BouncingBall环境是理想的因果测试床
- 已修复的RSSM为因果版本打好基础

要开始实施吗？我可以立即帮你实现Causal-RSSM的第一版原型！
