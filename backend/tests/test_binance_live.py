import json
import time

import pytest

from app.data import binance
from app.live.testnet import NotTestnetError, SymbolRules, TestnetClient, assert_testnet, sign


def test_mainnet_endpoint_refused():
    for bad in ["https://fapi.binance.com", "https://api.binance.com", "http://testnet.binancefuture.com",
                "https://testnet.binancefuture.com.evil.io"]:
        with pytest.raises(NotTestnetError):
            assert_testnet(bad)
    assert assert_testnet("https://testnet.binancefuture.com/x") == "https://testnet.binancefuture.com"


def test_hmac_signature_matches_binance_docs_example():
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    q = ("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
         "&recvWindow=5000&timestamp=1499827319559")
    assert sign(secret, q) == "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"


def test_signed_request_needs_keys_and_never_sends_secret():
    sent = {}

    def tr(method, url, headers):
        sent.update(url=url, headers=headers)
        return 200, "{}"

    with pytest.raises(RuntimeError):
        TestnetClient(key="", secret="", transport=tr).equity()
    TestnetClient(key="K", secret="S3CRET", transport=tr).request("GET", "/fapi/v2/account", signed=True)
    assert "signature=" in sent["url"] and "S3CRET" not in sent["url"] and sent["headers"]["X-MBX-APIKEY"] == "K"


def test_qty_floors_to_step_never_up():
    r = SymbolRules(step=0.001, min_qty=0.001, tick=0.1, min_notional=100)
    assert r.floor_qty(0.0129) == 0.012
    assert r.floor_qty(0.0009) == 0.0
    assert r.round_px(65000.06) == 65000.1


def test_open_bar_dropped_and_funding_point_in_time():
    now = int(time.time() * 1000)
    h = 3600_000
    rows = [[now - 3 * h, "1", "2", "0.5", "1.5", "10", now - 2 * h - 1],
            [now - 2 * h, "1", "2", "0.5", "1.5", "10", now - h - 1],
            [now - h + 1, "1", "2", "0.5", "1.5", "10", now + 1]]  # still open
    kl = binance.parse_klines(rows, 60, now)
    assert len(kl) == 2
    funding = [((now - 3 * h) // 1000 - 10, 0.0001), ((now - 2 * h) // 1000 + 60, 0.0009)]  # 2nd settles after bar 2 opens
    bars = binance.attach_funding(kl, funding)
    assert [b.funding_rate for b in bars] == [0.0001, 0.0001]


def test_sync_time_updates_offset():
    server_time = 1793000000000
    def tr(method, url, headers):
        if "/fapi/v1/time" in url:
            return 200, json.dumps({"serverTime": server_time})
        return 200, "{}"

    client = TestnetClient(transport=tr)
    offset = client.sync_time()
    assert client.time_offset_ms != 0
    now_ms = int(time.time() * 1000)
    assert abs(client.time_offset_ms - (server_time - now_ms)) < 500


def test_signed_request_recovers_from_code_1021_recv_window_error():
    calls = []
    def tr(method, url, headers):
        calls.append((method, url))
        if "/fapi/v1/time" in url:
            return 200, json.dumps({"serverTime": int(time.time() * 1000) + 100})
        if "/fapi/v2/account" in url:
            # First attempt fails with -1021
            if len([c for c in calls if "/fapi/v2/account" in c[1]]) == 1:
                return 400, '{"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}'
            # Second attempt succeeds
            return 200, '{"totalMarginBalance": "12345.67"}'
        return 200, "{}"

    client = TestnetClient(key="TESTKEY", secret="TESTSECRET", transport=tr, retry_backoff_s=0.01)
    res = client.equity()
    assert res == 12345.67
    account_calls = [c for c in calls if "/fapi/v2/account" in c[1]]
    assert len(account_calls) == 2  # Failed once, recovered and succeeded on retry

def test_signed_request_raises_other_4xx_without_retry():
    calls = []
    def tr(method, url, headers):
        calls.append((method, url))
        if "/fapi/v1/time" in url:
            return 200, json.dumps({"serverTime": int(time.time() * 1000)})
        return 400, '{"code":-1100,"msg":"Illegal characters found in parameter."}'

    from app.live.testnet import ExchangeError
    client = TestnetClient(key="TESTKEY", secret="TESTSECRET", transport=tr, retry_backoff_s=0.01)
    with pytest.raises(ExchangeError) as exc:
        client.equity()
    assert "-1100" in exc.value.body
    # Only 1 attempt on /fapi/v2/account (no retries)
    account_calls = [c for c in calls if "/fapi/v2/account" in c[1]]
    assert len(account_calls) == 1
