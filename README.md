# Deep Learning Based Stock Trading Agent System

基于 [ytq0198/RL-Stock](https://github.com/ytq0198/RL-Stock) 思路演进的深度强化学习股票交易 Agent 系统。使用 baostock 下载 A 股日线数据，自定义 `gymnasium` 交易环境，以 `stable-baselines3` PPO 为下层策略，并扩展多层 Agent（Planner、世界模型、风控、执行层）及新闻/公告/事件特征融合。

**官方仓库**：[ytq0198/Deep-Learning-Based-Stock-Trading-Agent-System](https://github.com/ytq0198/Deep-Learning-Based-Stock-Trading-Agent-System)  
**设计文档**：[项目设计.md](项目设计.md)

> 免责声明：本项目只用于学习和模拟实验，不构成投资建议。强化学习回测收益不代表真实交易收益。

## 环境安装

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 快速验证

没有真实数据时，可以先生成一份演示数据并跑通训练、评估、画图流程：

```bash
python main.py --make-demo-data --timesteps 1000
```

运行后会生成：

- `stockdata/train/` 和 `stockdata/test/`：演示 CSV 数据
- `models/ppo_stock.zip`：训练好的 PPO 模型
- `img/sh.600036.png`：测试集每日收益曲线

## 下载真实股票数据

下载招商银行 `sh.600036` 的训练集和测试集：

```bash
python get_stock_data.py --code sh.600036
```

下载全部股票会比较慢：

```bash
python get_stock_data.py
```

默认数据切分与原仓库介绍一致：

- 训练集：`1990-01-01` 到 `2019-11-29`
- 测试集：`2019-12-01` 到 `2019-12-31`

## 训练与测试

```bash
python main.py --stock-code sh.600036 --timesteps 10000
```

可以切换奖励策略：

```bash
python main.py --stock-code sh.600036 --reward-strategy profit_delta
```

奖励策略说明：

- `profit_sign`：贴近原仓库介绍，盈利给正奖励，亏损给较大惩罚。
- `profit_delta`：使用净值变化率作为奖励，更适合继续调参和扩展实验。

## 新闻 + 行情融合训练

项目支持把新闻、公告或舆情文本转成每日情绪特征，再合并到 K 线数据里训练。

先用演示新闻生成融合数据：

```bash
python news_pipeline.py --stock-code sh.600036 --company-keyword 招商银行 --make-demo-news
```

输出目录：

- `newsdata/raw_news.csv`：原始新闻
- `newsdata/scored_news.csv`：打分后的新闻
- `newsdata/daily_sentiment_features.csv`：每日情绪特征
- `newsdata/sentiment_quality.json`：新闻覆盖率质量报告
- `stockdata_sentiment/train/` 和 `stockdata_sentiment/test/`：已融合新闻特征的训练/测试数据

然后用融合数据重新训练：

```bash
python main.py --stock-code sh.600036 --train-dir stockdata_sentiment/train --test-dir stockdata_sentiment/test --timesteps 10000 --model-path models/ppo_stock_sentiment
```

如果你已经有自己的新闻 CSV，可以使用同样的列名：`date,code,title,source,url,published_at,content`，然后运行：

```bash
python news_pipeline.py --stock-code sh.600036 --company-keyword 招商银行 --news-csv newsdata/raw_news.csv
```

抓取东方财富官方公告（推荐用于强事件/财报/分红等高质量信号）：

```bash
python news_pipeline.py --stock-code sh.600036 --announcements --announcement-start 2019-01-01 --announcement-end 2019-12-31 --news-csv newsdata/announcements_2019.csv --features-csv newsdata/announcements_2019_features.csv --output-root stockdata_announcements_2019
```

如需同时保留股吧讨论并合并去重：

```bash
python news_pipeline.py --stock-code sh.600036 --announcements --announcement-start 2019-01-01 --announcement-end 2019-12-31 --guba --guba-pages 5 --news-csv newsdata/announcements_guba.csv --features-csv newsdata/announcements_guba_features.csv --output-root stockdata_announcements_guba
```

如需抓取公告正文（更慢，但有利于情绪/事件抽取）：

```bash
python news_pipeline.py --stock-code sh.600036 --announcements --announcement-fetch-content --announcement-start 2019-12-01 --announcement-end 2019-12-31
```

本地人工公告 CSV 仍可通过 `--announcement-local` 接入；与已有 `--news-csv` 合并时使用 `--merge-news-csv`。

新闻特征默认后移一天，避免训练时偷看未来。

用同一个随机种子进行无新闻和新闻融合对比：

```bash
python compare_experiment.py --stock-code sh.600036 --timesteps 50000 --seed 42
```

在继续训练新闻增强模型前，建议先分析新闻特征是否和未来收益有关：

```bash
python sentiment_correlation.py --data-dir stockdata_recent_balanced_guba_sentiment --stock-code sh.600036
```

报告会输出到 `reports/sentiment_correlation/`。

也可以用传统机器学习先做冷启动验证，判断特征能否预测次日涨跌：

```bash
python feature_baseline.py --data-dir stockdata_recent_balanced_guba_sentiment --stock-code sh.600036 --feature-set full
```

自动比较多组特征组合：

```bash
python feature_selection.py --data-dir stockdata_recent_regime --stock-code sh.600036
```

报告会输出到 `reports/feature_selection/`，用于判断哪些特征组合最值得进入 PPO。

支持的特征集：

- `price_only`：只使用价格/成交量上下文。
- `sentiment_light`：价格特征 + `sentiment_mean`、`social_heat`。
- `event`：价格特征 + 事件结构化特征。
- `full`：价格 + 情绪 + 事件全部特征。

更完整的实验说明见 `docs/sentiment_experiment.md`。

## 动态训练与市场状态

为了应对金融时间序列非平稳性，项目支持先用长期/较长窗口识别市场状态，再用近期滚动窗口动态训练。

生成市场状态特征：

```bash
python market_regime.py --source-train-dir stockdata_recent_balanced_guba_event/train --source-test-dir stockdata_recent_balanced_guba_event/test --output-root stockdata_recent_regime
```

运行 walk-forward 动态训练回测：

```bash
python walk_forward.py --data-dir stockdata_recent_regime --stock-code sh.600036 --train-window 30 --test-window 1 --timesteps 2048 --max-windows 10
```

启用自动高风险 regime 过滤：

```bash
python walk_forward.py --data-dir stockdata_recent_regime --stock-code sh.600036 --train-window 30 --test-window 1 --timesteps 2048 --max-windows 10 --auto-block-risk-regimes
```

启用 `minimal_signal` 风险评分过滤：

```bash
python walk_forward.py --data-dir stockdata_recent_regime --stock-code sh.600036 --train-window 30 --test-window 1 --timesteps 2048 --max-windows 10 --use-risk-signal
```

使用精简特征集和 Planner 强事件放大：

```bash
python walk_forward.py --data-dir stockdata_recent_regime --stock-code sh.600036 --feature-set minimal_signal --planner-amplify --use-risk-signal --auto-block-risk-regimes
```

验证人工强事件的 Planner 行为：

```bash
python planner_event_test.py
```

将人工强事件注入新闻数据：

```bash
python inject_strong_events.py --base-news newsdata/guba_news.csv --strong-events newsdata/sample_strong_events.csv --output newsdata/guba_with_strong_events.csv
```

计算事件是否已被市场提前消化：

```bash
python priced_in_analyzer.py --feature-csv newsdata/guba_strong_event_features.csv --stock-csv stockdata_recent_stage10_event/train/sh.600036.sh.600036.csv
```

过滤逻辑包括：

- 模型动作比例低于 `--action-threshold` 时转为空仓。
- 动作类型靠近买/卖/持有边界时转为空仓。
- 高风险 `regime_code` 中阻止买入。
- `RiskSignalScorer` 使用 `pre_5d_return`、`volume_spike`、`social_heat`、`event_risk_count`、`regime_code` 计算风险分，风险过高时阻止买入。
- Planner 在出现超级正面事件时可放大买入比例，在超级负面事件时强制空仓。
- `--opportunity-cost-penalty` 可启用踏空惩罚，避免模型把永远空仓当作局部最优。
- `priced_in_score` 用于判断利好是否已提前兑现，已兑现的正面事件不会触发 Planner 放大。

报告输出到 `reports/walk_forward/`。

## 项目结构

```text
.
├── get_stock_data.py          # baostock 数据下载脚本
├── main.py                    # PPO 训练、测试和画图入口
├── agent_backtest.py          # 多层 Agent 历史回测入口
├── compare_experiment.py      # 无新闻 vs 新闻融合一键对比实验
├── feature_baseline.py        # 传统模型冷启动验证
├── feature_selection.py       # 多组特征组合自动筛选
├── market_regime.py           # 市场状态识别与 regime 特征生成
├── news_pipeline.py           # 新闻情绪特征构建和数据融合入口
├── paper_trading.py           # 单步模拟盘决策入口
├── sentiment_correlation.py   # 新闻特征与未来收益相关性分析
├── walk_forward.py            # 滚动窗口动态训练回测
├── requirements.txt           # Python 依赖
├── 项目设计.md                 # 项目设计思路
├── agent/
│   ├── __init__.py
│   ├── interfaces.py          # 多层交易 Agent 核心接口
│   ├── policy.py              # PPO 策略包装
│   ├── world_model.py         # 轻量世界模型
│   ├── planner.py             # 上层规则 Planner
│   ├── risk.py                # 风险管理器
│   ├── executor.py            # 回测/模拟盘执行器
│   └── trading_agent.py       # Agent 编排器
├── docs/
│   ├── agent_interfaces.md    # Agent 模块接口说明
│   ├── backtest_upgrade.md    # 回测增强方案
│   ├── paper_trading.md       # 模拟盘设计
│   └── sentiment_experiment.md # 新闻融合实验说明
├── news/
│   ├── crawler.py             # RSS/CSV 新闻采集
│   ├── sentiment.py           # 规则情绪打分
│   ├── feature_builder.py     # 每日新闻特征聚合
│   └── merge_features.py      # 新闻特征与 K 线合并
└── rlenv/
    ├── __init__.py
    └── StockTradingEnv0.py    # 自定义股票交易环境
```

## 多层 Agent 设计

当前项目已补充多层交易 Agent 的设计资料。推荐先阅读 `项目设计.md`，再阅读 `docs/agent_interfaces.md`、`docs/backtest_upgrade.md` 和 `docs/paper_trading.md`。

新的 Agent 架构不会让 LLM 直接下单，而是让 PPO 负责快速动作，上层 Planner/LLM 负责解释和推演，`RiskManager` 负责最终风控，执行层只做历史回测或模拟盘。

训练好 PPO 模型后，可以运行多层 Agent 回测：

```bash
python agent_backtest.py --stock-code sh.600036 --model-path models/ppo_stock.zip
```

回测报告会输出到 `reports/agent_backtest/`，包括 `backtest_report.md`、`backtest_summary.json`、`trades.csv` 和 `equity_curve.csv`。

也可以运行一次模拟盘决策。它只更新本地模拟账户日志，不连接真实交易账户：

```bash
python paper_trading.py --stock-code sh.600036 --model-path models/ppo_stock.zip
```
