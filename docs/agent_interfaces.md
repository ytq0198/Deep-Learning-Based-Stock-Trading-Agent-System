# Agent 接口设计

本文档说明 `agent/interfaces.py` 中各模块的职责和数据流。接口先保持抽象，便于后续分别实现历史回测、世界模型、LLM 上层规划和模拟盘执行。

## 核心数据结构

- `MarketState`：单日或实时行情快照，包括价格、成交量和扩展指标。
- `AccountState`：当前账户状态，包括现金、持股、净值、历史最高净值和持仓成本。
- `PolicyAction`：下层模型输出的动作，包括买入、卖出、持有和交易比例。
- `Scenario`：世界模型生成的未来情景，包括预期收益、最差收益和最大回撤。
- `AgentDecision`：上层规划器输出的候选决策和解释。
- `RiskDecision`：风险管理层的最终审核结果。
- `ExecutionResult`：回测或模拟盘执行后的成交结果和新账户状态。

## 模块接口

### DataProvider

负责提供行情数据。第一版可以读取本地 CSV，后续可以接 baostock、实时行情源或模拟盘行情。

### WorldModel

负责根据当前市场状态、账户状态和候选动作模拟未来情景。它不直接下单，只向上层规划器提供推演结果。

### PolicyModel

负责快速输出交易动作。当前 PPO 模型属于这一层，适合根据状态给出买入、卖出或持有。

### Planner

负责组合 PPO 动作和世界模型情景，输出带理由的候选决策。未来可以由规则系统或 LLM 实现。

### RiskManager

负责最终风控审核。即使 Planner 建议买入，RiskManager 也可以因为仓位过高、回撤过大、数据异常等原因拒绝交易。

### Executor

负责执行通过风控的动作。执行器可以有两种实现：历史回测执行器和 paper trading 执行器。

## 推荐调用顺序

```text
DataProvider -> PolicyModel -> WorldModel -> Planner -> RiskManager -> Executor -> Log
```

这个顺序保证了所有交易动作都先经过模型建议、情景推演、上层解释和风控审核，最后才进入回测或模拟盘执行。
