from __future__ import annotations

from dataclasses import dataclass, field

from .crawler import NewsItem


@dataclass(frozen=True)
class SentimentResult:
    date: str
    code: str
    title: str
    score: float
    impact: str
    source: str = ""
    matched_positive: list[str] = field(default_factory=list)
    matched_negative: list[str] = field(default_factory=list)


class RuleBasedSentimentScorer:
    """Chinese finance lexicon scorer used before an LLM scorer is configured."""

    positive_keywords = {
        "增长": 0.25,
        "大增": 0.45,
        "暴增": 0.75,
        "盈利": 0.25,
        "净利润": 0.15,
        "回购": 0.45,
        "分红": 0.25,
        "增持": 0.35,
        "中标": 0.3,
        "突破": 0.2,
        "上调": 0.25,
        "利好": 0.5,
        "创新高": 0.35,
        "合作": 0.2,
    }
    negative_keywords = {
        "亏损": -0.55,
        "下滑": -0.35,
        "暴跌": -0.75,
        "减持": -0.4,
        "立案": -0.75,
        "调查": -0.45,
        "处罚": -0.55,
        "违规": -0.6,
        "风险": -0.25,
        "下调": -0.3,
        "利空": -0.55,
        "退市": -0.95,
        "诉讼": -0.45,
        "债务": -0.35,
    }
    announcement_keywords = {"公告", "年报", "季报", "业绩", "分红", "回购", "重组"}

    def score(self, item: NewsItem) -> SentimentResult:
        text = f"{item.title} {item.content}"
        matched_positive = [
            keyword for keyword in self.positive_keywords if keyword in text
        ]
        matched_negative = [
            keyword for keyword in self.negative_keywords if keyword in text
        ]
        score = sum(self.positive_keywords[keyword] for keyword in matched_positive)
        score += sum(self.negative_keywords[keyword] for keyword in matched_negative)

        if any(keyword in text for keyword in self.announcement_keywords):
            score *= 1.2

        score = max(-1.0, min(1.0, score))
        if score > 0.2:
            impact = "positive"
        elif score < -0.2:
            impact = "negative"
        else:
            impact = "neutral"

        return SentimentResult(
            date=item.date,
            code=item.code,
            title=item.title,
            score=score,
            impact=impact,
            source=item.source,
            matched_positive=matched_positive,
            matched_negative=matched_negative,
        )

    def score_many(self, items: list[NewsItem]) -> list[SentimentResult]:
        return [self.score(item) for item in items]
