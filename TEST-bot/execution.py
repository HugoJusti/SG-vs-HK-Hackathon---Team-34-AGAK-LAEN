"""
Execution module: Roostoo API interaction, order management, cooldown, rate limiting.
"""
import time
import hmac
import hashlib
import json
import logging
import requests
from datetime import UTC, datetime, timedelta
from config import (
    BASE_URL, API_KEY, SECRET_KEY,
    USE_LIMIT_ORDERS, LIMIT_ORDER_OFFSET_PCT, LIMIT_ORDER_TIMEOUT_SEC,
    LIMIT_ORDER_FALLBACK_MARKET,
    COOLDOWN_MIN_MINUTES, COOLDOWN_STABILITY_CHECKS, COOLDOWN_STABILITY_THRESHOLD,
    RETRY_BASE_DELAY_SEC, RETRY_MAX_ATTEMPTS,
    MAX_API_CALLS_PER_MIN, LOG_TRADES_FILE, LOG_ERRORS_FILE
)

logger = logging.getLogger(__name__)


def _json_default(value):
    """Normalize datetime and scalar wrapper values for JSON logging."""
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


# ═══════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """Track API calls and enforce 30/minute limit."""

    def __init__(self, max_calls: int = MAX_API_CALLS_PER_MIN):
        self.max_calls = max_calls
        self.call_timestamps = []

    def wait_if_needed(self):
        """Block until we're under the rate limit."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self.call_timestamps = [t for t in self.call_timestamps if now - t < 60]

        if len(self.call_timestamps) >= self.max_calls:
            sleep_time = 60 - (now - self.call_timestamps[0]) + 0.1
            if sleep_time > 0:
                logger.warning(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        self.call_timestamps.append(time.time())

    @property
    def remaining_calls(self) -> int:
        now = time.time()
        self.call_timestamps = [t for t in self.call_timestamps if now - t < 60]
        return self.max_calls - len(self.call_timestamps)


rate_limiter = RateLimiter()


# ═══════════════════════════════════════════════════════════════
# ROOSTOO API CLIENT
# ═══════════════════════════════════════════════════════════════

class RoostooClient:
    """Wrapper for Roostoo mock exchange API with retry + rate limiting."""

    def __init__(self):
        self.base_url = BASE_URL
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

    def _timestamp(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, payload: dict) -> tuple:
        """Generate HMAC SHA256 signature. Returns (headers, signed_payload_str)."""
        payload["timestamp"] = self._timestamp()
        sorted_keys = sorted(payload.keys())
        param_str = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        return headers, payload, param_str

    def _request_with_retry(self, method: str, endpoint: str,
                            payload: dict = None, signed: bool = False) -> dict:
        """Make API request with rate limiting and exponential backoff retry."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(RETRY_MAX_ATTEMPTS):
            rate_limiter.wait_if_needed()

            try:
                if signed:
                    headers, payload, param_str = self._sign(payload or {})

                    if method == "GET":
                        resp = requests.get(url, headers=headers, params=payload, timeout=10)
                    else:
                        headers["Content-Type"] = "application/x-www-form-urlencoded"
                        resp = requests.post(url, headers=headers, data=param_str, timeout=10)
                else:
                    if method == "GET":
                        params = payload or {}
                        params["timestamp"] = self._timestamp()
                        resp = requests.get(url, params=params, timeout=10)
                    else:
                        resp = requests.post(url, data=payload, timeout=10)

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.RequestException as e:
                delay = RETRY_BASE_DELAY_SEC * (2 ** attempt)
                logger.warning(
                    f"API error on {endpoint} (attempt {attempt+1}/{RETRY_MAX_ATTEMPTS}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

        logger.error(f"API request failed after {RETRY_MAX_ATTEMPTS} attempts: {endpoint}")
        return {"Success": False, "ErrMsg": "Max retries exceeded"}

    # ── Public endpoints ──

    def get_server_time(self) -> dict:
        return self._request_with_retry("GET", "/v3/serverTime")

    def get_exchange_info(self) -> dict:
        return self._request_with_retry("GET", "/v3/exchangeInfo")

    def get_ticker(self, pair: str = None) -> dict:
        """Get live ticker. Returns price + spread data."""
        payload = {}
        if pair:
            payload["pair"] = pair
        return self._request_with_retry("GET", "/v3/ticker", payload)

    # ── Signed endpoints ──

    def get_balance(self) -> dict:
        return self._request_with_retry("GET", "/v3/balance", {}, signed=True)

    def get_pending_count(self) -> dict:
        return self._request_with_retry("GET", "/v3/pending_count", {}, signed=True)

    def place_order(self, pair: str, side: str, quantity: float,
                    price: float = None, order_type: str = None) -> dict:
        """Place a LIMIT or MARKET order."""
        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"

        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if order_type.upper() == "LIMIT" and price is not None:
            payload["price"] = str(price)

        result = self._request_with_retry("POST", "/v3/place_order", payload, signed=True)
        self._log_trade(pair, side, quantity, price, order_type, result)
        return result

    def query_order(self, order_id: str = None, pair: str = None,
                    pending_only: bool = None) -> dict:
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
        return self._request_with_retry("POST", "/v3/query_order", payload, signed=True)

    def cancel_order(self, order_id: str = None, pair: str = None) -> dict:
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        return self._request_with_retry("POST", "/v3/cancel_order", payload, signed=True)

    # ── Helpers ──

    def get_portfolio_value(self) -> float:
        """Estimate total portfolio value in USD."""
        balance = self.get_balance()
        if not balance.get("Success") and "Wallet" not in balance:
            return 0.0

        wallet = balance.get("SpotWallet", balance.get("Wallet", {}))
        total = 0.0

        # USD balance
        usd = wallet.get("USD", {})
        total += usd.get("Free", 0) + usd.get("Lock", 0)

        # Crypto holdings — price them via ticker
        for coin, amounts in wallet.items():
            if coin == "USD":
                continue
            holding = amounts.get("Free", 0) + amounts.get("Lock", 0)
            if holding > 0:
                ticker = self.get_ticker(f"{coin}/USD")
                if ticker.get("Success"):
                    price = ticker["Data"][f"{coin}/USD"]["LastPrice"]
                    total += holding * price

        return total

    def get_ticker_spread(self, pair: str) -> dict:
        """Get bid, ask, spread info for a pair."""
        ticker = self.get_ticker(pair)
        if not ticker.get("Success"):
            return {"bid": 0, "ask": 0, "last": 0, "spread_pct": 999}

        data = ticker["Data"][pair]
        bid = data["MaxBid"]
        ask = data["MinAsk"]
        last = data["LastPrice"]
        spread_pct = ((ask - bid) / last) * 100 if last > 0 else 999

        return {
            "bid": bid,
            "ask": ask,
            "last": last,
            "spread_pct": spread_pct,
            "change_24h": data.get("Change", 0),
            "volume": data.get("CoinTradeValue", 0),
        }

    def _log_trade(self, pair, side, quantity, price, order_type, result):
        """Append trade to JSONL log file."""
        import os
        os.makedirs(os.path.dirname(LOG_TRADES_FILE), exist_ok=True)
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "pair": pair,
            "side": side,
            "quantity": quantity,
            "price": price,
            "type": order_type,
            "success": result.get("Success", False),
            "order_id": result.get("OrderDetail", {}).get("OrderID"),
            "status": result.get("OrderDetail", {}).get("Status"),
            "filled_price": result.get("OrderDetail", {}).get("FilledAverPrice"),
            "commission": result.get("OrderDetail", {}).get("CommissionChargeValue"),
            "api_response": result,
        }
        with open(LOG_TRADES_FILE, "a") as f:
            f.write(json.dumps(entry, default=_json_default) + "\n")


# ═══════════════════════════════════════════════════════════════
# SMART ORDER EXECUTOR (limit order with fallback)
# ═══════════════════════════════════════════════════════════════

class OrderExecutor:
    """
    Executes orders using limit-first strategy:
    1. Place limit order slightly inside the spread
    2. Wait up to LIMIT_ORDER_TIMEOUT_SEC for fill
    3. If not filled, cancel and optionally fall back to market order
    """

    def __init__(self, client: RoostooClient):
        self.client = client

    def execute_buy(self, pair: str, quantity: float, ticker_data: dict) -> dict:
        """Execute a buy order with limit-first strategy."""
        if USE_LIMIT_ORDERS and ticker_data.get("ask"):
            # Place limit slightly below ask (inside the spread)
            limit_price = round(ticker_data["ask"] * (1 - LIMIT_ORDER_OFFSET_PCT), 2)

            logger.info(f"Placing LIMIT BUY {pair}: qty={quantity}, price={limit_price}")
            result = self.client.place_order(pair, "BUY", quantity,
                                             price=limit_price, order_type="LIMIT")

            if result.get("Success") and result.get("OrderDetail", {}).get("Status") == "PENDING":
                # Wait for fill
                order_id = result["OrderDetail"]["OrderID"]
                filled = self._wait_for_fill(order_id)

                if filled:
                    return filled

                # Not filled — cancel and fall back
                logger.info(f"Limit order {order_id} not filled, cancelling...")
                self.client.cancel_order(order_id=str(order_id))

                if LIMIT_ORDER_FALLBACK_MARKET:
                    logger.info(f"Falling back to MARKET BUY {pair}: qty={quantity}")
                    return self.client.place_order(pair, "BUY", quantity, order_type="MARKET")
                return result

            # Already filled as taker (crossed the spread) or error
            return result

        # Market order fallback
        logger.info(f"Placing MARKET BUY {pair}: qty={quantity}")
        return self.client.place_order(pair, "BUY", quantity, order_type="MARKET")

    def execute_sell(self, pair: str, quantity: float, ticker_data: dict) -> dict:
        """Execute a sell order with limit-first strategy."""
        if USE_LIMIT_ORDERS and ticker_data.get("bid"):
            # Place limit slightly above bid (inside the spread)
            limit_price = round(ticker_data["bid"] * (1 + LIMIT_ORDER_OFFSET_PCT), 2)

            logger.info(f"Placing LIMIT SELL {pair}: qty={quantity}, price={limit_price}")
            result = self.client.place_order(pair, "SELL", quantity,
                                             price=limit_price, order_type="LIMIT")

            if result.get("Success") and result.get("OrderDetail", {}).get("Status") == "PENDING":
                order_id = result["OrderDetail"]["OrderID"]
                filled = self._wait_for_fill(order_id)

                if filled:
                    return filled

                logger.info(f"Limit order {order_id} not filled, cancelling...")
                self.client.cancel_order(order_id=str(order_id))

                if LIMIT_ORDER_FALLBACK_MARKET:
                    logger.info(f"Falling back to MARKET SELL {pair}: qty={quantity}")
                    return self.client.place_order(pair, "SELL", quantity, order_type="MARKET")
                return result

            return result

        logger.info(f"Placing MARKET SELL {pair}: qty={quantity}")
        return self.client.place_order(pair, "SELL", quantity, order_type="MARKET")

    def _wait_for_fill(self, order_id: int) -> dict:
        """Poll order status until filled or timeout."""
        start = time.time()
        while time.time() - start < LIMIT_ORDER_TIMEOUT_SEC:
            time.sleep(5)  # Check every 5 seconds
            result = self.client.query_order(order_id=str(order_id))
            if result.get("Success"):
                orders = result.get("OrderMatched", [])
                if orders and orders[0].get("Status") == "FILLED":
                    logger.info(f"Limit order {order_id} filled!")
                    return result
        return None


# ═══════════════════════════════════════════════════════════════
# COOLDOWN MANAGER
# ═══════════════════════════════════════════════════════════════

class CooldownManager:
    """
    Manages post-exit cooldown per trading pair.
    
    Two conditions must be met to exit cooldown:
    1. Minimum time has elapsed (COOLDOWN_MIN_MINUTES)
    2. HMM regime has stabilised (consistent confidence for N checks)
    """

    def __init__(self):
        # {pair: {"exit_time": datetime, "stability_count": int}}
        self.cooldowns = {}

    def start_cooldown(self, pair: str):
        """Call this after exiting a position."""
        self.cooldowns[pair] = {
            "exit_time": datetime.now(UTC),
            "stability_count": 0,
        }
        logger.info(f"Cooldown started for {pair}")

    def is_on_cooldown(self, pair: str, regime_probs: dict) -> bool:
        """
        Check if pair is still on cooldown.
        Updates stability counter based on regime confidence.
        """
        if pair not in self.cooldowns:
            return False

        cd = self.cooldowns[pair]
        elapsed = (datetime.now(UTC) - cd["exit_time"]).total_seconds() / 60

        # Condition 1: minimum time
        if elapsed < COOLDOWN_MIN_MINUTES:
            logger.debug(
                f"{pair} cooldown: {elapsed:.0f}/{COOLDOWN_MIN_MINUTES} min elapsed"
            )
            return True

        # Condition 2: regime stability
        max_prob = max(regime_probs.values())
        if max_prob >= COOLDOWN_STABILITY_THRESHOLD:
            cd["stability_count"] += 1
        else:
            cd["stability_count"] = 0  # Reset if unstable

        if cd["stability_count"] < COOLDOWN_STABILITY_CHECKS:
            logger.debug(
                f"{pair} cooldown: stable checks {cd['stability_count']}/{COOLDOWN_STABILITY_CHECKS}"
            )
            return True

        # Both conditions met — exit cooldown
        logger.info(f"{pair} cooldown complete after {elapsed:.0f} minutes")
        del self.cooldowns[pair]
        return False
