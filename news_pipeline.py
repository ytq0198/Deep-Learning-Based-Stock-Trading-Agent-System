from __future__ import annotations

import argparse
import csv
from pathlib import Path

from news import (
    EastmoneyAnnouncementCrawler,
    EastmoneyGubaCrawler,
    GDELTNewsCrawler,
    LocalAnnouncementCSV,
    NewsItem,
    RSSNewsCrawler,
    RuleBasedEventExtractor,
    RuleBasedSentimentScorer,
    build_default_gdelt_query,
    build_daily_sentiment_features,
    build_sentiment_quality_report,
    load_news_csv,
    merge_news_items,
    save_quality_report,
    save_news_csv,
)
from news.event_extractor import _is_announcement_item
from news.merge_features import merge_directory
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build news sentiment features for stock data.")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--company-keyword", default="招商银行")
    parser.add_argument("--news-csv", default="newsdata/raw_news.csv")
    parser.add_argument("--features-csv", default="newsdata/daily_sentiment_features.csv")
    parser.add_argument("--train-dir", default="stockdata/train")
    parser.add_argument("--test-dir", default="stockdata/test")
    parser.add_argument("--output-root", default="stockdata_sentiment")
    parser.add_argument("--rss-url", action="append", default=[])
    parser.add_argument("--gdelt", action="store_true", help="Fetch historical articles from GDELT.")
    parser.add_argument("--gdelt-query", help="Custom GDELT query. Defaults to company/code aliases.")
    parser.add_argument("--gdelt-start", default="20180101000000")
    parser.add_argument("--gdelt-end", default="20191231235959")
    parser.add_argument("--gdelt-max-records", type=int, default=250)
    parser.add_argument("--guba", action="store_true", help="Fetch public Eastmoney Guba discussion titles.")
    parser.add_argument("--guba-pages", type=int, default=5)
    parser.add_argument("--guba-start-page", type=int, default=1)
    parser.add_argument(
        "--announcements",
        action="store_true",
        help="Fetch official company announcements from Eastmoney notice center.",
    )
    parser.add_argument("--announcement-start", default="", help="Announcement start date, e.g. 2019-01-01.")
    parser.add_argument("--announcement-end", default="", help="Announcement end date, e.g. 2019-12-31.")
    parser.add_argument("--announcement-page-size", type=int, default=50)
    parser.add_argument("--announcement-max-pages", type=int, default=20)
    parser.add_argument(
        "--announcement-fetch-content",
        action="store_true",
        help="Fetch full announcement text for each item. Slower but improves sentiment/event extraction.",
    )
    parser.add_argument(
        "--announcement-local",
        help="Load manually curated announcements from a local CSV file.",
    )
    parser.add_argument(
        "--merge-news-csv",
        action="store_true",
        help="Merge fetched news with an existing --news-csv file instead of replacing it.",
    )
    parser.add_argument("--make-demo-news", action="store_true")
    parser.add_argument("--no-shift", action="store_true", help="Do not shift news features to next day.")
    parser.add_argument(
        "--announcement-only",
        action="store_true",
        help="Keep only official announcement items (drop guba/social noise).",
    )
    parser.add_argument(
        "--sparse-strong-events",
        action="store_true",
        help="Zero sentiment/social features on days without a strong event.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_news_path = Path(args.news_csv)
    items = collect_news_items(args)
    if args.announcement_only:
        items = [item for item in items if _is_announcement_item(item)]
    save_news_csv(items, raw_news_path)

    scorer = RuleBasedSentimentScorer()
    event_extractor = RuleBasedEventExtractor()
    sentiment_results = scorer.score_many(items)
    event_results = event_extractor.extract_many(items)
    features = build_daily_sentiment_features(
        sentiment_results,
        shift_to_next_day=not args.no_shift,
        event_results=event_results,
        sparse_strong_events=args.sparse_strong_events,
    )

    features_path = Path(args.features_csv)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    _save_scored_news(sentiment_results, features_path.parent / "scored_news.csv")
    _save_event_news(event_results, features_path.parent / "event_news.csv")

    train_sample = _load_first_stock_csv(Path(args.train_dir), args.stock_code)
    test_sample = _load_first_stock_csv(Path(args.test_dir), args.stock_code)
    quality_report = {}
    if train_sample is not None:
        quality_report["train"] = build_sentiment_quality_report(train_sample, features)
        save_quality_report(quality_report["train"], features_path.parent / "sentiment_quality_train.json")
    if test_sample is not None:
        quality_report["test"] = build_sentiment_quality_report(test_sample, features)
        save_quality_report(quality_report["test"], features_path.parent / "sentiment_quality_test.json")
    if quality_report:
        save_quality_report(quality_report, features_path.parent / "sentiment_quality.json")

    output_root = Path(args.output_root)
    merge_directory(args.train_dir, features, output_root / "train")
    merge_directory(args.test_dir, features, output_root / "test")

    print(f"Loaded news: {len(items)}")
    print(f"Saved daily features to {features_path}")
    print(f"Saved quality report to {features_path.parent / 'sentiment_quality.json'}")
    print(f"Saved merged stock data to {output_root}")


def collect_news_items(args: argparse.Namespace) -> list[NewsItem]:
    groups: list[list[NewsItem]] = []
    used_fetch = False

    if args.make_demo_news:
        groups.append(make_demo_news(args.stock_code, args.company_keyword))
        used_fetch = True

    if args.announcements:
        crawler = EastmoneyAnnouncementCrawler(
            code=args.stock_code,
            start_date=args.announcement_start,
            end_date=args.announcement_end,
            page_size=args.announcement_page_size,
            max_pages=args.announcement_max_pages,
            fetch_content=args.announcement_fetch_content,
        )
        groups.append(crawler.crawl())
        used_fetch = True

    if args.announcement_local:
        groups.append(LocalAnnouncementCSV(args.announcement_local).crawl())
        used_fetch = True

    if args.gdelt:
        query = args.gdelt_query or build_default_gdelt_query(
            args.company_keyword,
            args.stock_code,
        )
        crawler = GDELTNewsCrawler(
            code=args.stock_code,
            query=query,
            start_datetime=args.gdelt_start,
            end_datetime=args.gdelt_end,
            max_records=args.gdelt_max_records,
        )
        groups.append(crawler.crawl())
        used_fetch = True

    if args.guba:
        crawler = EastmoneyGubaCrawler(
            code=args.stock_code,
            pages=args.guba_pages,
            start_page=args.guba_start_page,
        )
        groups.append(crawler.crawl())
        used_fetch = True

    if args.rss_url:
        crawler = RSSNewsCrawler(
            urls=args.rss_url,
            code=args.stock_code,
            keywords=[args.stock_code, args.company_keyword],
        )
        groups.append(crawler.crawl())
        used_fetch = True

    if args.merge_news_csv or not used_fetch:
        existing = load_news_csv(Path(args.news_csv))
        if existing:
            groups.append(existing)

    if not groups:
        return []

    return merge_news_items(*groups)


def make_demo_news(code: str, company_keyword: str) -> list[NewsItem]:
    return [
        NewsItem(
            date="2019-12-03",
            code=code,
            title=f"{company_keyword}公告称净利润增长并计划分红",
            source="demo",
            content="业绩增长、分红和稳健经营被市场视为利好。",
        ),
        NewsItem(
            date="2019-12-10",
            code=code,
            title=f"{company_keyword}遭遇行业风险扰动，市场担忧短期下滑",
            source="demo",
            content="行业风险和利润下滑预期带来负面情绪。",
        ),
        NewsItem(
            date="2019-12-18",
            code=code,
            title=f"{company_keyword}发布回购公告并获机构上调评级",
            source="demo",
            content="回购、上调评级和增持预期形成积极催化。",
        ),
    ]


def _save_scored_news(results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "date",
                "code",
                "title",
                "score",
                "impact",
                "source",
                "matched_positive",
                "matched_negative",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "date": result.date,
                    "code": result.code,
                    "title": result.title,
                    "score": result.score,
                    "impact": result.impact,
                    "source": result.source,
                    "matched_positive": "|".join(result.matched_positive),
                    "matched_negative": "|".join(result.matched_negative),
                }
            )


def _save_event_news(results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "date",
                "code",
                "title",
                "event_type",
                "impact_direction",
                "impact_duration_days",
                "confidence_score",
                "event_strength",
                "is_super_event",
                "is_priced_in",
                "source",
                "matched_keywords",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "date": result.date,
                    "code": result.code,
                    "title": result.title,
                    "event_type": result.event_type,
                    "impact_direction": result.impact_direction,
                    "impact_duration_days": result.impact_duration_days,
                    "confidence_score": result.confidence_score,
                    "event_strength": result.event_strength,
                    "is_super_event": result.is_super_event,
                    "is_priced_in": result.is_priced_in,
                    "source": result.source,
                    "matched_keywords": "|".join(result.matched_keywords),
                }
            )


def _load_first_stock_csv(stock_dir: Path, code: str) -> pd.DataFrame | None:
    candidates = sorted(path for path in stock_dir.glob("*.csv") if code in path.name)
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("demo" in path.name.lower(), len(path.name)))
    return pd.read_csv(candidates[0])


if __name__ == "__main__":
    main()
