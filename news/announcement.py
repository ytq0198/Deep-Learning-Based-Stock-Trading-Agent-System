from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .crawler import NewsItem, _clean_html, _fetch_text, load_news_csv


class AnnouncementCrawler(Protocol):
    """Interface for high-quality company announcements and filings."""

    def crawl(self) -> list[NewsItem]:
        """Return announcement-like news items."""


class LocalAnnouncementCSV:
    """Loads manually curated announcements from a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def crawl(self) -> list[NewsItem]:
        return load_news_csv(self.path)


class EastmoneyAnnouncementCrawler:
    """Fetch official company announcements from Eastmoney notice center.

    Data source: https://data.eastmoney.com/notices/stock/{code}.html
    """

    LIST_API = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    DETAIL_API = "https://np-cnotice-stock.eastmoney.com/api/content/ann"

    def __init__(
        self,
        code: str,
        start_date: str = "",
        end_date: str = "",
        page_size: int = 50,
        max_pages: int = 20,
        fetch_content: bool = False,
        timeout: float = 15.0,
    ) -> None:
        self.code = code
        self.stock_digits = _code_to_digits(code)
        self.start_date = start_date
        self.end_date = end_date
        self.page_size = page_size
        self.max_pages = max_pages
        self.fetch_content = fetch_content
        self.timeout = timeout

    def crawl(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        total_hits = None

        for page_index in range(1, self.max_pages + 1):
            payload = self._fetch_list_page(page_index)
            page_items = payload.get("list") or []
            if total_hits is None:
                total_hits = int(payload.get("total_hits") or 0)

            if not page_items:
                break

            for row in page_items:
                items.append(self._to_news_item(row))

            if len(items) >= total_hits:
                break
            if len(page_items) < self.page_size:
                break

        if self.fetch_content:
            items = [self._attach_content(item) for item in items]

        return items

    def _fetch_list_page(self, page_index: int) -> dict:
        params = {
            "sr": "-1",
            "page_size": str(self.page_size),
            "page_index": str(page_index),
            "ann_type": "A",
            "client_source": "web",
            "stock_list": self.stock_digits,
            "f_node": "0",
            "s_node": "0",
        }
        if self.start_date:
            params["begin_time"] = self.start_date
        if self.end_date:
            params["end_time"] = self.end_date

        url = f"{self.LIST_API}?{urllib.parse.urlencode(params)}"
        response = _fetch_json(url, self.timeout)
        if not response.get("success"):
            raise RuntimeError(f"Eastmoney announcement API failed: {response}")
        return response.get("data") or {}

    def _to_news_item(self, row: dict) -> NewsItem:
        art_code = str(row.get("art_code") or "")
        title = str(row.get("title_ch") or row.get("title") or "").strip()
        notice_date = _parse_notice_datetime(row.get("notice_date") or row.get("display_time"))
        return NewsItem(
            date=notice_date.date().isoformat(),
            code=self.code,
            title=title,
            source="东方财富-公告",
            url=_announcement_page_url(art_code, self.stock_digits),
            published_at=notice_date.isoformat(sep=" ", timespec="minutes"),
            content=title,
        )

    def _attach_content(self, item: NewsItem) -> NewsItem:
        art_code = _art_code_from_url(item.url)
        if not art_code:
            return item

        detail = self._fetch_detail(art_code)
        content = _clean_notice_content(detail.get("notice_content") or "")
        if not content:
            return item

        return NewsItem(
            date=item.date,
            code=item.code,
            title=item.title,
            source=item.source,
            url=item.url,
            published_at=item.published_at,
            content=content,
        )

    def _fetch_detail(self, art_code: str) -> dict:
        params = {
            "art_code": art_code,
            "client_source": "web",
        }
        url = f"{self.DETAIL_API}?{urllib.parse.urlencode(params)}"
        response = _fetch_json(url, self.timeout)
        if not response.get("success"):
            return {}
        data = response.get("data")
        return data if isinstance(data, dict) else {}


def _code_to_digits(code: str) -> str:
    return code.split(".")[-1]


def _parse_notice_datetime(value: object) -> datetime:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.now()


def _announcement_page_url(art_code: str, stock_digits: str) -> str:
    if not art_code:
        return ""
    return f"https://data.eastmoney.com/notices/detail/{art_code}/{stock_digits}.html"


def _art_code_from_url(url: str) -> str:
    match = re.search(r"/notices/detail/(AN\d+)/", url)
    return match.group(1) if match else ""


def _clean_notice_content(value: str) -> str:
    value = _clean_html(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _fetch_json(url: str, timeout: float) -> dict:
    text = _fetch_text(url, timeout)
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}
