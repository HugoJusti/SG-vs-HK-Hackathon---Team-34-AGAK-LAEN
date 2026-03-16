import requests
import time
import hmac
import hashlib

# =========================
# 1) PUT YOUR KEYS HERE
# =========================
BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "PASTE_YOUR_API_KEY_HERE"
SECRET_KEY = "PASTE_YOUR_SECRET_KEY_HERE"

# =========================
# 2) BASIC HELPER FUNCTIONS
# =========================
def get_timestamp():
    """
    Returns a 13-digit timestamp in milliseconds.
    Example: 1710000000123
    """
    return str(int(time.time() * 1000))


def build_param_string(params: dict) -> str:
    """
    Roostoo signs the query string / body after sorting keys.
    Example:
    {'pair':'BTC/USD', 'side':'BUY', 'timestamp':'123'}
    becomes:
    pair=BTC/USD&side=BUY&timestamp=123
    """
    sorted_keys = sorted(params.keys())
    return "&".join(f"{key}={params[key]}" for key in sorted_keys)


def make_signature(param_string: str) -> str:
    """
    Create HMAC SHA256 signature using your SECRET_KEY.
    """
    return hmac.new(
        SECRET_KEY.encode("utf-8"),
        param_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def make_signed_headers(param_string: str) -> dict:
    """
    Headers needed for signed endpoints.
    """
    signature = make_signature(param_string)
    return {
        "RST-API-KEY": API_KEY,
        "MSG-SIGNATURE": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }


def safe_json(response):
    """
    Safely print response as JSON if possible.
    """
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


# =========================
# 3) PUBLIC / SIMPLE ENDPOINTS
# =========================
def get_server_time():
    """
    GET /v3/serverTime
    No API key needed
    """
    url = f"{BASE_URL}/v3/serverTime"
    response = requests.get(url, timeout=10)
    return safe_json(response)


def get_exchange_info():
    """
    GET /v3/exchangeInfo
    No API key needed
    """
    url = f"{BASE_URL}/v3/exchangeInfo"
    response = requests.get(url, timeout=10)
    return safe_json(response)


def get_ticker(pair=None):
    """
    GET /v3/ticker
    Needs timestamp
    pair is optional
    """
    url = f"{BASE_URL}/v3/ticker"

    params = {
        "timestamp": get_timestamp()
    }

    if pair is not None:
        params["pair"] = pair

    response = requests.get(url, params=params, timeout=10)
    return safe_json(response)


# =========================
# 4) SIGNED ENDPOINTS
# =========================
def get_balance():
    """
    GET /v3/balance
    Signed endpoint
    """
    url = f"{BASE_URL}/v3/balance"

    params = {
        "timestamp": get_timestamp()
    }

    param_string = build_param_string(params)
    headers = make_signed_headers(param_string)

    response = requests.get(url, params=params, headers=headers, timeout=10)
    return safe_json(response)


def place_market_order(pair, side, quantity):
    """
    POST /v3/place_order
    Example market order
    """
    url = f"{BASE_URL}/v3/place_order"

    data = {
        "pair": pair,
        "side": side,          # BUY or SELL
        "type": "MARKET",
        "quantity": str(quantity),
        "timestamp": get_timestamp()
    }

    param_string = build_param_string(data)
    headers = make_signed_headers(param_string)

    response = requests.post(url, data=data, headers=headers, timeout=10)
    return safe_json(response)


def place_limit_order(pair, side, quantity, price):
    """
    POST /v3/place_order
    Example limit order
    """
    url = f"{BASE_URL}/v3/place_order"

    data = {
        "pair": pair,
        "side": side,          # BUY or SELL
        "type": "LIMIT",
        "quantity": str(quantity),
        "price": str(price),
        "timestamp": get_timestamp()
    }

    param_string = build_param_string(data)
    headers = make_signed_headers(param_string)

    response = requests.post(url, data=data, headers=headers, timeout=10)
    return safe_json(response)


def query_order(order_id=None, pair=None, offset=None, limit=None, pending_only=None):
    """
    POST /v3/query_order

    You can:
    - query one specific order_id
    - or query by pair
    - or query everything
    """
    url = f"{BASE_URL}/v3/query_order"

    data = {
        "timestamp": get_timestamp()
    }

    if order_id is not None:
        data["order_id"] = str(order_id)
    else:
        if pair is not None:
            data["pair"] = pair
        if offset is not None:
            data["offset"] = str(offset)
        if limit is not None:
            data["limit"] = str(limit)
        if pending_only is not None:
            data["pending_only"] = "TRUE" if pending_only else "FALSE"

    param_string = build_param_string(data)
    headers = make_signed_headers(param_string)

    response = requests.post(url, data=data, headers=headers, timeout=10)
    return safe_json(response)


def cancel_order(order_id=None, pair=None):
    """
    POST /v3/cancel_order

    According to docs:
    - send order_id to cancel one order
    - OR send pair to cancel pending orders for that pair
    - OR send neither to cancel all pending orders
    """
    url = f"{BASE_URL}/v3/cancel_order"

    data = {
        "timestamp": get_timestamp()
    }

    if order_id is not None and pair is not None:
        raise ValueError("Use either order_id or pair, not both.")

    if order_id is not None:
        data["order_id"] = str(order_id)

    if pair is not None:
        data["pair"] = pair

    param_string = build_param_string(data)
    headers = make_signed_headers(param_string)

    response = requests.post(url, data=data, headers=headers, timeout=10)
    return safe_json(response)


# =========================
# 5) EXAMPLE MAIN PROGRAM
# =========================
if __name__ == "__main__":
    print("1. SERVER TIME")
    print(get_server_time())
    print()

    print("2. EXCHANGE INFO")
    exchange_info = get_exchange_info()
    print(exchange_info)
    print()

    print("3. TICKER FOR BTC/USD")
    print(get_ticker("BTC/USD"))
    print()

    # These next ones need real API_KEY + SECRET_KEY
    print("4. BALANCE")
    print(get_balance())
    print()

    # BE CAREFUL: the next calls can create/cancel real mock-exchange orders
    # Uncomment only when you are ready.

    # print("5. PLACE MARKET ORDER")
    # market_result = place_market_order("BTC/USD", "BUY", 0.001)
    # print(market_result)
    # print()

    # print("6. PLACE LIMIT ORDER")
    # limit_result = place_limit_order("BTC/USD", "BUY", 0.001, 20000)
    # print(limit_result)
    # print()

    # print("7. QUERY ALL ORDERS")
    # print(query_order(limit=10))
    # print()

    # Example: query one order if you know the order id
    # print("8. QUERY ONE ORDER")
    # print(query_order(order_id=81))
    # print()

    # Example: cancel one order if you know the order id
    # print("9. CANCEL ORDER")
    # print(cancel_order(order_id=83))
    # print()