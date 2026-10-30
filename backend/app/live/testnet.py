"""Binance USDT-M Futures TESTNET client (signed REST, stdlib only).

Safety:
  * Base URL must be a testnet/demo host from TESTNET_HOSTS; mainnet is refused
    at construction time, regardless of config or env.
  * Keys are read from the local environment only
    (BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET) and never persisted
    or logged.
  * Resilient network handling: automatic timeout retry with exponential backoff
    for transient connection failures, socket timeouts, rate limits (429/5xx),
    and automatic server time synchronization on clock drift (Binance error -1021).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

TESTNET_HOSTS = {"testnet.binancefuture.com", "demo-fapi.binance.com"}
DEFAULT_BASE = "https://testnet.binancefuture.com"
DEFAULT_RECV_WINDOW = 10000


class NotTestnetError(RuntimeError):
    pass


class ExchangeError(RuntimeError):
    def __init__(self, status: int, body: str, ban_until_ms: int | None = None, retry_after_s: float | None = None):
        super().__init__(f"[{status}] {body[:400]}")
        self.status, self.body = status, body
        self.ban_until_ms = ban_until_ms
        self.retry_after_s = retry_after_s


class RateLimitError(ExchangeError):
    """Raised on HTTP 429 (Too Many Requests) or HTTP 418 / code -1003 (IP Banned)."""
    pass


def parse_ban_time(body: str) -> int | None:
    """Parse timestamp from Binance message: IP(...) banned until <timestamp_ms>."""
    m = re.search(r"banned until (\d+)", body)
    return int(m.group(1)) if m else None


def assert_testnet(base: str) -> str:
    u = urllib.parse.urlparse(base)
    if u.scheme != "https" or u.hostname not in TESTNET_HOSTS:
        raise NotTestnetError(f"refusing non-testnet endpoint: {base!r}")
    return f"https://{u.hostname}"


def sign(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class SymbolRules:
    step: float
    min_qty: float
    tick: float
    min_notional: float

    def floor_qty(self, q: float) -> float:
        n = math.floor(q / self.step + 1e-9)
        v = round(n * self.step, _dp(self.step))
        return v if v >= self.min_qty else 0.0

    def round_px(self, p: float) -> float:
        return round(round(p / self.tick) * self.tick, _dp(self.tick))


def _dp(x: float) -> int:
    s = f"{x:.12f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


class TestnetClient:
    def __init__(self, base: str = DEFAULT_BASE, key: str | None = None, secret: str | None = None,
                 timeout: float = 15.0, max_retries: int = 3, retry_backoff_s: float = 1.0,
                 recv_window: int = DEFAULT_RECV_WINDOW, auto_sync_time: bool = True,
                 transport=None):
        self.base = assert_testnet(base)
        self.key = key if key is not None else os.environ.get("BINANCE_TESTNET_API_KEY", "")
        self.secret = secret if secret is not None else os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.recv_window = max(1000, min(recv_window, 60000))
        self.auto_sync_time = auto_sync_time
        self.time_offset_ms: int = 0
        self._last_sync_time: float = 0.0
        self._transport = transport or self._http

    def _http(self, method: str, url: str, headers: dict) -> tuple[int, str]:
        req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionResetError) as e:
            # Represent transient network disconnect / timeout as status 599 for retry handling
            return 599, f'{{"error": "NetworkError", "detail": "{str(e)}"}}'

    def sync_time(self) -> int:
        """Fetch server time from /fapi/v1/time and compute offset against local system time."""
        t_before = int(time.time() * 1000)
        try:
            status, body = self._transport("GET", f"{self.base}/fapi/v1/time", {"User-Agent": "cgtrs/2"})
            t_after = int(time.time() * 1000)
            if 200 <= status < 300 and body:
                data = json.loads(body)
                server_time = int(data.get("serverTime", 0))
                if server_time > 0:
                    local_est = (t_before + t_after) // 2
                    self.time_offset_ms = server_time - local_est
                    self._last_sync_time = time.time()
                    return self.time_offset_ms
        except Exception:
            pass
        return self.time_offset_ms

    def request(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {"User-Agent": "cgtrs/2"}

        retries = 0
        last_status = 0
        last_body = ""

        # Auto-sync time before first signed request or if synced over 30 minutes ago
        if signed and self.auto_sync_time and (self._last_sync_time == 0.0 or (time.time() - self._last_sync_time > 1800)):
            self.sync_time()

        while retries <= self.max_retries:
            curr_params = dict(params)
            if signed:
                if not (self.key and self.secret):
                    raise RuntimeError("BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET not set")
                ts = int(time.time() * 1000) + self.time_offset_ms
                curr_params |= {"timestamp": ts, "recvWindow": self.recv_window}
                q = urllib.parse.urlencode(curr_params)
                q += "&signature=" + sign(self.secret, q)
                headers["X-MBX-APIKEY"] = self.key
            else:
                q = urllib.parse.urlencode(curr_params)

            try:
                status, body = self._transport(method, f"{self.base}{path}?{q}", headers)
            except Exception as e:
                status, body = 599, f'{{"error": "TransportError", "detail": "{str(e)}"}}'

            # Success
            if 200 <= status < 300:
                return json.loads(body) if body else {}

            last_status, last_body = status, body

            # Clock skew / recvWindow error (Binance code -1021) -> re-sync time offset and retry
            is_time_error = (signed and status == 400 and ("-1021" in body or "outside of the recvWindow" in body))
            if is_time_error:
                retries += 1
                if retries <= self.max_retries:
                    self.sync_time()
                    backoff = self.retry_backoff_s * (2 ** (retries - 1))
                    time.sleep(backoff)
                    continue

            # If client error (4xx except 429), don't retry (e.g. invalid signature, bad parameters)
            if 400 <= status < 500 and status != 429 and not is_time_error:
                raise ExchangeError(status, body)

            # Retry on rate limit 429 or server errors / network disconnects (5xx, 599)
            retries += 1
            if retries <= self.max_retries:
                backoff = self.retry_backoff_s * (2 ** (retries - 1))
                time.sleep(backoff)

        raise ExchangeError(last_status, last_body)

    # --- public
    def klines(self, symbol: str, interval: str, limit: int = 500):
        return self.request("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    def rules(self, symbol: str) -> SymbolRules:
        info = self.request("GET", "/fapi/v1/exchangeInfo")
        s = next(x for x in info["symbols"] if x["symbol"] == symbol)
        f = {x["filterType"]: x for x in s["filters"]}
        return SymbolRules(step=float(f["MARKET_LOT_SIZE"]["stepSize"] if "MARKET_LOT_SIZE" in f else f["LOT_SIZE"]["stepSize"]),
                           min_qty=float(f["LOT_SIZE"]["minQty"]), tick=float(f["PRICE_FILTER"]["tickSize"]),
                           min_notional=float(f.get("MIN_NOTIONAL", {}).get("notional", 5.0)))

    def book(self, symbol: str) -> tuple[float, float]:
        b = self.request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        return float(b["bidPrice"]), float(b["askPrice"])

    # --- signed
    def equity(self) -> float:
        a = self.request("GET", "/fapi/v2/account", signed=True)
        return float(a["totalMarginBalance"])

    def position(self, symbol: str) -> dict:
        ps = self.request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
        p = next((x for x in ps if x["symbol"] == symbol), None) or {"positionAmt": "0", "entryPrice": "0"}
        return {"qty": float(p["positionAmt"]), "entry": float(p["entryPrice"]),
                "upnl": float(p.get("unRealizedProfit", 0)), "liq": float(p.get("liquidationPrice", 0) or 0)}

    def open_orders(self, symbol: str) -> list[dict]:
        return self.request("GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def set_leverage(self, symbol: str, lev: int):
        return self.request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": lev}, signed=True)

    def cancel_all(self, symbol: str):
        return self.request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)

    def order(self, **p):
        return self.request("POST", "/fapi/v1/order", p, signed=True)
