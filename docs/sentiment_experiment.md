# 新闻融合实验流程

本文档说明如何准备真实新闻数据，并用同一随机种子对比“纯行情模型”和“新闻 + 行情模型”。

## 新闻数据格式

推荐先把新闻整理为 CSV：

```csv
date,code,title,source,url,published_at,content
2019-12-03,sh.600036,招商银行公告称净利润增长,example,,2019-12-03,招商银行发布业绩公告
```

字段说明：

- `date`：新闻发布时间对应日期，格式 `YYYY-MM-DD`。
- `code`：股票代码，例如 `sh.600036`。
- `title`：新闻标题。
- `source`：来源网站。
- `url`：新闻链接，可为空。
- `published_at`：原始发布时间，可为空。
- `content`：正文或摘要。

## 构建新闻融合数据

使用本地 CSV：

```bash
python news_pipeline.py --stock-code sh.600036 --company-keyword 招商银行 --news-csv newsdata/raw_news.csv
```

使用 RSS 源：

```bash
python news_pipeline.py --stock-code sh.600036 --company-keyword 招商银行 --rss-url <RSS_URL>
```

使用东方财富股吧公开帖子列表作为近期舆情源：

```bash
python news_pipeline.py --stock-code sh.600036 --company-keyword 招商银行 --guba --guba-pages 10 --news-csv newsdata/guba_news.csv --features-csv newsdata/guba_daily_sentiment_features.csv --output-root stockdata_guba_sentiment
```

注意：股吧列表主要适合近期舆情。如果你的 K 线训练区间是 2018-2019，而抓到的帖子是 2026 年，则新闻覆盖率仍然是 0。新闻日期必须和行情训练/测试日期重叠，模型才可能学到新闻影响。

使用 GDELT 历史新闻索引：

```bash
python news_pipeline.py --stock-code sh.600036 --company-keyword 招商银行 --gdelt --gdelt-start 20180101000000 --gdelt-end 20191231235959
```

GDELT 是公开接口，可能出现限流或中文查询结果较少。遇到限流时建议稍后再试，或优先使用本地 CSV。

使用演示新闻：

```bash
python news_pipeline.py --stock-code sh.600036 --company-keyword 招商银行 --make-demo-news
```

输出：

- `newsdata/raw_news.csv`
- `newsdata/scored_news.csv`
- `newsdata/daily_sentiment_features.csv`
- `newsdata/sentiment_quality.json`
- `stockdata_sentiment/train/`
- `stockdata_sentiment/test/`

## 检查新闻覆盖率

查看 `newsdata/sentiment_quality.json`：

```json
{
  "trading_days": 7345,
  "covered_days": 3,
  "coverage_ratio": 0.0004,
  "warning": "news coverage is very low; sentiment features may not affect training"
}
```

如果 `coverage_ratio` 很低，说明新闻特征大部分时间都是 0，模型很难学到新闻的影响。做严肃实验时，建议覆盖率至少达到 `10%` 以上。

## 一键对比实验

用同一个随机种子对比无新闻和新闻融合模型：

```bash
python compare_experiment.py --stock-code sh.600036 --timesteps 50000 --seed 42
```

输出：

- `reports/compare_experiment/comparison_summary.json`
- `reports/compare_experiment/comparison_report.md`
- `reports/compare_experiment/price_only_sh.600036.png`
- `reports/compare_experiment/sentiment_sh.600036.png`

## 解释结果

如果新闻版收益更高，只能说明当前数据和当前随机种子下新闻特征有正向表现。要证明新闻特征稳定有效，需要：

1. 使用真实新闻，而不是演示新闻。
2. 扩大测试区间，不能只看一个月。
3. 多只股票重复实验。
4. 多个随机种子重复训练。
5. 对比最大回撤、夏普比率和交易次数，而不是只看最终收益。
