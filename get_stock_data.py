from __future__ import annotations

import argparse
import socket
from pathlib import Path

import baostock as bs
import pandas as pd

FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)


class Downloader:
    def __init__(
        self,
        output_dir: Path,
        date_start: str,
        date_end: str,
        stock_code: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.output_dir = output_dir
        self.date_start = date_start
        self.date_end = date_end
        self.stock_code = stock_code
        self.timeout = timeout

    def _query_codes(self) -> pd.DataFrame:
        if self.stock_code:
            return pd.DataFrame([{"code": self.stock_code, "code_name": self.stock_code}])

        stock_rs = bs.query_all_stock(self.date_end)
        if stock_rs.error_code != "0":
            raise RuntimeError(f"query_all_stock failed: {stock_rs.error_msg}")
        return stock_rs.get_data()

    def _download_one(self, code: str, code_name: str) -> None:
        print(f"processing {code} {code_name}")
        rs = bs.query_history_k_data_plus(
            code,
            FIELDS,
            start_date=self.date_start,
            end_date=self.date_end,
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"{code} download failed: {rs.error_msg}")

        df_code = rs.get_data()
        output_file = self.output_dir / f"{code}.{code_name}.csv"
        df_code.to_csv(output_file, index=False, encoding="utf-8-sig")

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        socket.setdefaulttimeout(self.timeout)
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"baostock login failed: {login.error_msg}")

        try:
            stock_df = self._query_codes()
            for _, row in stock_df.iterrows():
                self._download_one(row["code"], row.get("code_name", row["code"]))
        finally:
            try:
                bs.logout()
            except OSError as err:
                print(f"baostock logout failed: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download baostock daily stock data.")
    parser.add_argument("--code", help="Download one stock, for example sh.600036.")
    parser.add_argument("--train-start", default="1990-01-01")
    parser.add_argument("--train-end", default="2019-11-29")
    parser.add_argument("--test-start", default="2019-12-01")
    parser.add_argument("--test-end", default="2019-12-31")
    parser.add_argument("--output", default="stockdata")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)

    Downloader(
        output_root / "train",
        args.train_start,
        args.train_end,
        args.code,
        args.timeout,
    ).run()
    Downloader(
        output_root / "test",
        args.test_start,
        args.test_end,
        args.code,
        args.timeout,
    ).run()


if __name__ == "__main__":
    main()
