from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BankStock:
    code: str
    name: str


# A-share bank pool used for cross-sectional training (baostock codes).
BANK_STOCKS: tuple[BankStock, ...] = (
    BankStock("sh.600036", "招商银行"),
    BankStock("sh.601398", "工商银行"),
    BankStock("sh.601939", "建设银行"),
    BankStock("sh.601288", "农业银行"),
    BankStock("sh.601328", "交通银行"),
    BankStock("sh.600000", "浦发银行"),
    BankStock("sh.601166", "兴业银行"),
    BankStock("sh.600016", "民生银行"),
    BankStock("sh.601818", "光大银行"),
    BankStock("sh.601998", "中信银行"),
    BankStock("sz.000001", "平安银行"),
    BankStock("sh.601009", "南京银行"),
    BankStock("sz.002142", "宁波银行"),
    BankStock("sh.601229", "上海银行"),
    BankStock("sh.600919", "江苏银行"),
    BankStock("sh.601658", "邮储银行"),
    BankStock("sh.601916", "浙商银行"),
    BankStock("sh.601838", "成都银行"),
    BankStock("sh.600926", "杭州银行"),
    BankStock("sh.601577", "长沙银行"),
)


def bank_codes() -> list[str]:
    return [stock.code for stock in BANK_STOCKS]


def bank_name_map() -> dict[str, str]:
    return {stock.code: stock.name for stock in BANK_STOCKS}
