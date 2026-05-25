from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .crawler import NewsItem

ImpactDirection = Literal["positive", "negative", "neutral"]


@dataclass(frozen=True)
class EventResult:
    date: str
    code: str
    title: str
    event_type: str
    impact_direction: ImpactDirection
    impact_duration_days: int
    confidence_score: float
    event_strength: float = 0.0
    is_super_event: bool = False
    is_priced_in: bool = False
    source: str = ""
    matched_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EventRule:
    event_type: str
    direction: ImpactDirection
    duration_days: int
    confidence: float
    keywords: tuple[str, ...]
    strength: float | None = None
    super_event: bool = False


class EventExtractor(Protocol):
    def extract(self, item: NewsItem) -> EventResult:
        """Extract one structured event from a news item."""

    def extract_many(self, items: list[NewsItem]) -> list[EventResult]:
        """Extract structured events from multiple news items."""


class RuleBasedEventExtractor:
    """Maps finance text to coarse event features before an LLM extractor is added."""

    rules = [
        EventRule("performance_growth", "positive", 20, 0.8, ("净利润增长", "业绩增长", "大增", "暴增", "预增"), super_event=True),
        EventRule("performance_decline", "negative", 20, 0.8, ("净利润下滑", "业绩下滑", "亏损", "预亏", "暴跌"), super_event=True),
        EventRule("buyback", "positive", 15, 0.75, ("回购", "股份回购"), super_event=True),
        EventRule("dividend", "positive", 10, 0.65, ("分红", "派息", "现金分红", "股息率", "高股息")),
        EventRule("shareholder_increase", "positive", 10, 0.65, ("增持", "买入评级", "上调评级")),
        EventRule("shareholder_reduce", "negative", 10, 0.7, ("减持", "套现", "卖出评级", "下调评级")),
        EventRule("investigation_penalty", "negative", 30, 0.9, ("立案", "调查", "处罚", "违规", "监管函"), super_event=True),
        EventRule("asset_quality_positive", "positive", 20, 0.75, ("不良率下降", "拨备覆盖率提升", "资产质量改善", "风险抵补能力增强")),
        EventRule("asset_quality_negative", "negative", 20, 0.8, ("不良率上升", "拨备压力", "资产质量承压", "坏账", "逾期贷款", "地产风险敞口"), super_event=True),
        EventRule("net_interest_margin_positive", "positive", 15, 0.65, ("净息差企稳", "息差改善", "息差回升", "贷款收益率上升")),
        EventRule("net_interest_margin_negative", "negative", 15, 0.7, ("净息差收窄", "息差下行", "让利实体", "贷款利率下调")),
        EventRule("wealth_management_positive", "positive", 15, 0.6, ("财富管理增长", "中收增长", "手续费净收入增长", "零售业务增长")),
        EventRule("wealth_management_negative", "negative", 15, 0.65, ("中收下滑", "手续费收入下滑", "理财赎回", "财富管理承压")),
        EventRule("capital_adequacy_positive", "positive", 20, 0.65, ("资本充足率提升", "核心一级资本充足率提升", "补充资本")),
        EventRule("capital_adequacy_negative", "negative", 20, 0.7, ("资本充足率下降", "资本压力", "核心一级资本承压")),
        EventRule("policy_support", "positive", 20, 0.65, ("降准", "流动性投放", "稳增长", "银行板块利好", "金融支持实体")),
        EventRule("policy_pressure", "negative", 20, 0.7, ("降息", "存量房贷利率下调", "监管趋严", "让利", "息差压力")),
        EventRule("valuation_repair", "positive", 10, 0.55, ("估值修复", "破净修复", "低估值", "红利资产", "高股息防守")),
        EventRule("valuation_overheat", "negative", 7, 0.55, ("估值过高", "拥挤交易", "获利盘", "高位放量", "冲高回落")),
        EventRule("industry_risk", "negative", 15, 0.65, ("行业风险", "政策限制", "风险扰动", "债务", "坏账", "地产链风险")),
        EventRule("market_panic", "negative", 5, 0.55, ("扛不住", "恐慌", "阴跌", "要绿", "投诉")),
        EventRule("market_heat", "neutral", 3, 0.45, ("热议", "满仓", "散户", "集中营", "讨论", "人气", "热度")),
    ]

    def extract(self, item: NewsItem) -> EventResult:
        text = f"{item.title} {item.content}"
        matches: list[tuple[EventRule, list[str]]] = []
        for rule in self.rules:
            matched = [keyword for keyword in rule.keywords if keyword in text]
            if matched:
                matches.append((rule, matched))

        if not matches:
            return EventResult(
                date=item.date,
                code=item.code,
                title=item.title,
                event_type="other",
                impact_direction="neutral",
                impact_duration_days=1,
                confidence_score=0.1,
                event_strength=0.0,
                is_super_event=False,
                is_priced_in=_looks_priced_in(text),
                source=item.source,
            )

        # Prefer high-confidence concrete events over generic heat/panic wording.
        rule, matched_keywords = sorted(
            matches,
            key=lambda pair: (pair[0].confidence, len(pair[1])),
            reverse=True,
        )[0]
        return EventResult(
            date=item.date,
            code=item.code,
            title=item.title,
            event_type=rule.event_type,
            impact_direction=rule.direction,
            impact_duration_days=rule.duration_days,
            confidence_score=rule.confidence,
            event_strength=_event_strength(rule),
            is_super_event=rule.super_event or rule.confidence >= 0.75,
            is_priced_in=_looks_priced_in(text),
            source=item.source,
            matched_keywords=matched_keywords,
        )

    def extract_many(self, items: list[NewsItem]) -> list[EventResult]:
        return [self.extract(item) for item in items]


class LLMEventExtractor:
    """Adapter for future LLM-based event extraction.

    The class deliberately receives a client callable instead of depending on a
    specific model provider. The callable should accept a prompt and return JSON.
    """

    def __init__(self, client=None, fallback: EventExtractor | None = None) -> None:
        self.client = client
        self.fallback = fallback or RuleBasedEventExtractor()

    def extract(self, item: NewsItem) -> EventResult:
        if self.client is None:
            return self.fallback.extract(item)
        try:
            payload = json.loads(self.client(build_event_prompt(item)))
            return EventResult(
                date=item.date,
                code=item.code,
                title=item.title,
                event_type=str(payload.get("event_type", "other")),
                impact_direction=_direction(payload.get("impact_direction", "neutral")),
                impact_duration_days=int(payload.get("impact_duration_days", 1)),
                confidence_score=float(payload.get("confidence_score", 0.1)),
                event_strength=float(payload.get("event_strength", 0.0)),
                is_super_event=bool(payload.get("is_super_event", False)),
                is_priced_in=bool(payload.get("is_priced_in", False)),
                source=item.source,
                matched_keywords=[],
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return self.fallback.extract(item)

    def extract_many(self, items: list[NewsItem]) -> list[EventResult]:
        return [self.extract(item) for item in items]


def build_event_prompt(item: NewsItem) -> str:
    return f"""请把以下银行股相关新闻结构化为严格 JSON，不要输出解释。

字段:
- event_type: 事件类型，例如 asset_quality_negative, dividend, policy_pressure, market_panic, other
- impact_direction: positive / negative / neutral
- impact_duration_days: 预计影响天数，整数
- confidence_score: 0 到 1
- event_strength: -1 到 1，表示事件强度
- is_super_event: true / false，是否为重大事件
- is_priced_in: true / false，是否可能已经被市场提前消化

股票代码: {item.code}
标题: {item.title}
正文: {item.content}
"""


def _direction(value: object) -> ImpactDirection:
    text = str(value).lower()
    if text in {"positive", "negative", "neutral"}:
        return text  # type: ignore[return-value]
    return "neutral"


def _event_strength(rule: EventRule) -> float:
    if rule.strength is not None:
        return rule.strength
    if rule.direction == "positive":
        return rule.confidence
    if rule.direction == "negative":
        return -rule.confidence
    return 0.0


def _looks_priced_in(text: str) -> bool:
    return any(keyword in text for keyword in ["连续上涨", "高位", "涨幅已", "创新高", "放量大涨", "冲高回落"])
