from __future__ import annotations

import csv
import email.utils
import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class NewsItem:
    date: str
    code: str
    title: str
    source: str = ""
    url: str = ""
    published_at: str = ""
    content: str = ""


class RSSNewsCrawler:
    """Small dependency-free RSS crawler for finance news feeds."""

    def __init__(
        self,
        urls: list[str],
        code: str,
        keywords: list[str],
        timeout: float = 15.0,
    ) -> None:
        self.urls = urls
        self.code = code
        self.keywords = [keyword.lower() for keyword in keywords]
        self.timeout = timeout

    def crawl(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        for url in self.urls:
            xml_text = _fetch_text(url, self.timeout)
            root = ET.fromstring(xml_text)
            source = root.findtext("./channel/title", default=url)
            for item in root.findall(".//item"):
                title = item.findtext("title", default="").strip()
                description = item.findtext("description", default="").strip()
                link = item.findtext("link", default="").strip()
                published_at = item.findtext("pubDate", default="").strip()
                haystack = f"{title} {description}".lower()
                if self.keywords and not any(keyword in haystack for keyword in self.keywords):
                    continue
                items.append(
                    NewsItem(
                        date=_date_from_rss(published_at),
                        code=self.code,
                        title=title,
                        source=source,
                        url=link,
                        published_at=published_at,
                        content=description,
                    )
                )
        return items


class EastmoneyGubaCrawler:
    """Crawls public Eastmoney Guba list pages for stock discussion titles."""

    def __init__(
        self,
        code: str,
        pages: int = 5,
        start_page: int = 1,
        timeout: float = 15.0,
    ) -> None:
        self.code = code
        self.code_digits = code.split(".")[-1]
        self.pages = pages
        self.start_page = start_page
        self.timeout = timeout

    def crawl(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        for page in range(self.start_page, self.start_page + self.pages):
            url = f"https://guba.eastmoney.com/list,{self.code_digits}_{page}.html"
            html_text = _fetch_text(url, self.timeout)
            items.extend(self._parse_list_page(html_text))
        return _deduplicate(items)

    def _parse_list_page(self, html_text: str) -> list[NewsItem]:
        pattern = re.compile(
            r'<tr class="listitem">.*?'
            r'<div class="title"><a[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a></div>.*?'
            r'<div class="author"><a[^>]*>(?P<author>.*?)</a></div>.*?'
            r'<div class="update">(?P<update>.*?)</div>',
            re.S,
        )
        items: list[NewsItem] = []
        for match in pattern.finditer(html_text):
            title = _clean_html(match.group("title"))
            update = _clean_html(match.group("update"))
            href = html.unescape(match.group("href"))
            url = urllib.parse.urljoin("https://guba.eastmoney.com", href)
            items.append(
                NewsItem(
                    date=_date_from_guba_update(update),
                    code=self.code,
                    title=title,
                    source="东方财富股吧",
                    url=url,
                    published_at=update,
                    content=_clean_html(match.group("author")),
                )
            )
        return items


def load_news_csv(path: str | Path) -> list[NewsItem]:
    path = Path(path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [
            NewsItem(
                date=row.get("date", ""),
                code=row.get("code", ""),
                title=row.get("title", ""),
                source=row.get("source", ""),
                url=row.get("url", ""),
                published_at=row.get("published_at", ""),
                content=row.get("content", ""),
            )
            for row in reader
        ]


def save_news_csv(items: list[NewsItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "code", "title", "source", "url", "published_at", "content"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "date": item.date,
                    "code": item.code,
                    "title": item.title,
                    "source": item.source,
                    "url": item.url,
                    "published_at": item.published_at,
                    "content": item.content,
                }
            )


def merge_news_items(*item_groups: list[NewsItem]) -> list[NewsItem]:
    """Merge multiple news sources and remove duplicates."""
    merged: list[NewsItem] = []
    for group in item_groups:
        merged.extend(group)
    return _deduplicate(merged)


def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[NewsItem] = []
    for item in items:
        key = (item.date, item.title, item.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; StockResearchBot/1.0)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8", errors="ignore")


def _date_from_rss(value: str) -> str:
    if not value:
        return date.today().isoformat()
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        return parsed.date().isoformat()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return date.today().isoformat()


def _date_from_guba_update(value: str) -> str:
    value = value.strip()
    current_year = date.today().year
    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d", "%m-%d %H:%M", "%m-%d"]:
        try:
            if fmt.startswith("%m"):
                parsed = datetime.strptime(f"{current_year}-{value}", f"%Y-{fmt}")
            else:
                parsed = datetime.strptime(value, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return date.today().isoformat()


def _clean_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()
