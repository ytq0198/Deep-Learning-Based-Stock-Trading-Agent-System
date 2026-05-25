from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime

from .crawler import NewsItem


class GDELTNewsCrawler:
    """Fetches historical articles from the public GDELT DOC API."""

    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        code: str,
        query: str,
        start_datetime: str,
        end_datetime: str,
        max_records: int = 250,
        timeout: float = 30.0,
    ) -> None:
        self.code = code
        self.query = query
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.max_records = min(max_records, 250)
        self.timeout = timeout

    def crawl(self) -> list[NewsItem]:
        params = {
            "query": self.query,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": self.start_datetime,
            "enddatetime": self.end_datetime,
            "maxrecords": str(self.max_records),
            "sort": "datedesc",
        }
        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StockResearchBot/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))

        articles = payload.get("articles", [])
        items: list[NewsItem] = []
        for article in articles:
            title = str(article.get("title", "")).strip()
            if not title:
                continue
            seen_date = str(article.get("seendate", ""))
            items.append(
                NewsItem(
                    date=_parse_gdelt_date(seen_date),
                    code=self.code,
                    title=title,
                    source=str(article.get("domain", "gdelt")),
                    url=str(article.get("url", "")),
                    published_at=seen_date,
                    content=str(article.get("sourcecountry", "")),
                )
            )
        return items


def build_default_gdelt_query(company_keyword: str, stock_code: str) -> str:
    code_digits = stock_code.split(".")[-1]
    aliases = [
        company_keyword,
        "China Merchants Bank",
        code_digits,
    ]
    return " OR ".join(f'"{alias}"' for alias in aliases if alias)


def _parse_gdelt_date(value: str) -> str:
    value = value.strip()
    for fmt in ["%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y%m%d"]:
        try:
            return datetime.strptime(value[: len(datetime.now().strftime(fmt))], fmt).date().isoformat()
        except ValueError:
            continue
    if len(value) >= 8 and value[:8].isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return datetime.today().date().isoformat()
