# 银行板块 Panel 特征验证

- 训练样本：9199（约 460 行/股）
- 测试样本：2319
- 股票数：20

## 基线模型

- logistic_regression: accuracy=46.96%, auc=0.5112
- random_forest: accuracy=51.57%, auc=0.5024

## 特征相关性

- event_super_positive_count: -0.0034
- event_super_negative_count: 0.0026
- event_strength_mean: -0.0029
- announcement_score: 0.0001
- pre_5d_return: 0.0022
- volume_spike: -0.0048

## 判读

- AUC 稳定 > 0.55：存在初步 edge，可进入 PPO panel 训练。
- AUC ≈ 0.5：特征仍接近随机，应继续优化公告规则或延长样本。