from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bank_universe import BANK_STOCKS
from news import RuleBasedEventExtractor, RuleBasedSentimentScorer, build_daily_sentiment_features, save_news_csv
from news.akshare_announcement import fetch_announcement_items
from news.event_extractor import _is_announcement_item
from news.merge_features import merge_sentiment_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch announcement crawl + sparse features for bank pool.")
    parser.add_argument("--stock-root", default="stockdata_bank")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-05-25")
    parser.add_argument("--output-root", default="stockdata_bank_enriched")
    parser.add_argument("--news-root", default="newsdata_bank")
    parser.add_argument("--codes", nargs="*", help="Optional subset of baostock codes.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-pages", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stock_root = Path(args.stock_root)
    output_root = Path(args.output_root)
    news_root = Path(args.news_root)
    news_root.mkdir(parents=True, exist_ok=True)

    selected = [stock for stock in BANK_STOCKS if not args.codes or stock.code in args.codes]
    event_extractor = RuleBasedEventExtractor()
    scorer = RuleBasedSentimentScorer()
    combined_features: list[pd.DataFrame] = []
    manifest = []

    for stock in selected:
        enriched_file = output_root / "per_stock" / f"{stock.code}.csv"
        if args.skip_existing and enriched_file.exists():
            combined_features.append(pd.read_csv(enriched_file))
            manifest.append({"code": stock.code, "skipped": True})
            continue

        items = fetch_announcement_items(
            code=stock.code,
            start_date=args.start_date,
            end_date=args.end_date,
            max_pages=args.max_pages,
        )
        items = [item for item in items if _is_announcement_item(item)]
        news_file = news_root / f"{stock.code}_announcements.csv"
        save_news_csv(items, news_file)

        sentiment_results = scorer.score_many(items)
        event_results = event_extractor.extract_many(items)
        features = build_daily_sentiment_features(
            sentiment_results,
            shift_to_next_day=True,
            event_results=event_results,
            sparse_strong_events=True,
        )
        features_path = news_root / f"{stock.code}_features.csv"
        features.to_csv(features_path, index=False, encoding="utf-8-sig")

        stock_file = _find_stock_file(stock_root, stock.code)
        if stock_file is None:
            manifest.append({"code": stock.code, "error": "missing stock csv"})
            continue
        stock_df = pd.read_csv(stock_file)
        enriched = merge_sentiment_features(stock_df, features)
        enriched_file.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(enriched_file, index=False, encoding="utf-8-sig")
        combined_features.append(enriched)
        manifest.append(
            {
                "code": stock.code,
                "announcements": len(items),
                "strong_events": int((features.get("has_strong_event", 0) > 0).sum())
                if "has_strong_event" in features.columns
                else int(
                    (
                        features["event_super_positive_count"]
                        + features["event_super_negative_count"]
                    )
                    > 0
                ).sum(),
                "file": str(enriched_file),
            }
        )
        print(f"{stock.code}: announcements={len(items)}")

    if combined_features:
        panel_features = pd.concat(combined_features, ignore_index=True)
        panel_features.to_csv(news_root / "panel_features_all.csv", index=False, encoding="utf-8-sig")

    (news_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved bank announcement manifest to {news_root / 'manifest.json'}")


def _find_stock_file(stock_root: Path, code: str) -> Path | None:
    for sub in ["raw", "train", ""]:
        folder = stock_root / sub if sub else stock_root
        if not folder.exists():
            continue
        matches = sorted(folder.glob(f"{code}*.csv"))
        if matches:
            return matches[0]
    return None


if __name__ == "__main__":
    main()
