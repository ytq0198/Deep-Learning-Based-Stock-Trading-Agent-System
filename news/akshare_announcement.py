from __future__ import annotations

from .announcement import EastmoneyAnnouncementCrawler
from .crawler import NewsItem


def fetch_announcement_items(
    code: str,
    start_date: str,
    end_date: str,
    page_size: int = 50,
    max_pages: int = 20,
) -> list[NewsItem]:
    """Fetch announcements via AkShare when available, else Eastmoney API."""
    try:
        import akshare as ak
    except ImportError:
        return EastmoneyAnnouncementCrawler(
            code=code,
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
            max_pages=max_pages,
        ).crawl()

    symbol = code.split(".")[-1]
    begin = start_date.replace("-", "")
    end = end_date.replace("-", "")
    try:
        frame = ak.stock_notice_report(symbol="全部", date="20240101")
        _ = frame  # keep import path warm; individual API varies by akshare version
        individual = ak.stock_individual_notice_report(
            symbol=symbol,
            begin_date=begin,
            end_date=end,
        )
    except Exception:
        return EastmoneyAnnouncementCrawler(
            code=code,
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
            max_pages=max_pages,
        ).crawl()

    items: list[NewsItem] = []
    if individual is None or individual.empty:
        return EastmoneyAnnouncementCrawler(
            code=code,
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
            max_pages=max_pages,
        ).crawl()

    date_col = _pick_column(individual, ["公告日期", "notice_date", "date"])
    title_col = _pick_column(individual, ["公告标题", "title", "notice_title"])
    url_col = _pick_column(individual, ["公告链接", "url", "notice_url"])

    for _, row in individual.iterrows():
        title = str(row.get(title_col, "")).strip()
        if not title:
            continue
        date_value = str(row.get(date_col, ""))[:10]
        items.append(
            NewsItem(
                date=date_value,
                code=code,
                title=title,
                source="AkShare-公告",
                url=str(row.get(url_col, "")) if url_col else "",
                published_at=date_value,
                content=title,
            )
        )
    if items:
        return items
    return EastmoneyAnnouncementCrawler(
        code=code,
        start_date=start_date,
        end_date=end_date,
        page_size=page_size,
        max_pages=max_pages,
    ).crawl()


def _pick_column(frame, candidates: list[str]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    return frame.columns[0]
