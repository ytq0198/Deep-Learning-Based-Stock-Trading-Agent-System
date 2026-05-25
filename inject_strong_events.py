from __future__ import annotations

import argparse
from pathlib import Path

from news.crawler import load_news_csv, save_news_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge manual strong events into a news CSV.")
    parser.add_argument("--base-news", default="newsdata/guba_news.csv")
    parser.add_argument("--strong-events", default="newsdata/sample_strong_events.csv")
    parser.add_argument("--output", default="newsdata/guba_with_strong_events.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_items = load_news_csv(args.base_news)
    strong_items = load_news_csv(args.strong_events)
    merged = deduplicate(base_items + strong_items)
    save_news_csv(merged, args.output)
    print(f"Base news: {len(base_items)}")
    print(f"Strong events: {len(strong_items)}")
    print(f"Merged news: {len(merged)}")
    print(f"Saved to {Path(args.output)}")


def deduplicate(items):
    seen = set()
    unique = []
    for item in items:
        key = (item.date, item.code, item.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


if __name__ == "__main__":
    main()
