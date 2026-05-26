# LLM 公告事件抽取

## 作用

将公告标题/正文结构化为事件 JSON（含 `is_priced_in`、`event_strength`），供 Planner 与 PPO 特征使用。未配置 API 时自动回退规则抽取。

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` 或 `LLM_API_KEY` | API 密钥 |
| `LLM_BASE_URL` | 可选，OpenAI 兼容端点（本地 Qwen、DeepSeek 等） |
| `LLM_MODEL` | 可选，默认 `gpt-4o-mini` |

### 本地 Qwen（OpenAI 兼容）示例

```bash
set LLM_API_KEY=your-key
set LLM_BASE_URL=http://127.0.0.1:11434/v1
set LLM_MODEL=qwen2.5:7b
```

## 命令

```bash
python batch_announcement_pipeline.py --use-llm --start-date 2024-01-01 --end-date 2026-05-25
python run_bank_alpha_pipeline.py --use-llm
```

## 注意

- LLM 只负责文本结构化，不直接下单。
- 批量 20 股、数千条公告会产生 API 费用，建议先对单股或小样本试跑。
