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
