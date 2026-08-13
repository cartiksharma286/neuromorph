import asyncio
import math
import os
import threading
import time

import numpy as np


class IBKRUnavailableError(RuntimeError):
    pass


def calculate_volatility_signal(closes):
    prices = np.asarray(closes, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 21:
        raise ValueError("at least 21 positive closing prices are required")

    returns = np.diff(np.log(prices))
    lookback_returns = returns[-20:]
    realized_volatility = float(np.std(lookback_returns, ddof=1) * math.sqrt(252))
    sma_20 = float(np.mean(prices[-20:]))
    sma_50 = float(np.mean(prices[-min(50, prices.size):]))
    momentum_20 = float(prices[-1] / prices[-21] - 1.0)

    trend_score = np.clip((sma_20 / max(sma_50, 1e-12) - 1.0) * 20.0, -1.0, 1.0)
    momentum_score = np.clip(momentum_20 * 5.0, -1.0, 1.0)
    volatility_penalty = np.clip((realized_volatility - 0.35) * 1.5, 0.0, 1.0)
    score = float(np.clip(50.0 + 25.0 * trend_score + 25.0 * momentum_score - 15.0 * volatility_penalty, 0.0, 100.0))

    if score >= 70.0:
        signal = "BUY"
    elif score <= 30.0:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "signal_score": round(score, 2),
        "realized_volatility": round(realized_volatility, 6),
        "realized_volatility_pct": round(realized_volatility * 100.0, 2),
        "sma_20": round(sma_20, 4),
        "sma_50": round(sma_50, 4),
        "momentum_20_pct": round(momentum_20 * 100.0, 2),
        "method": "20-day log-return volatility with 20/50-day trend and momentum",
    }


class IBKRBridge:
    def __init__(self):
        self.host = os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(os.environ.get("IBKR_PORT", "7497"))
        self.client_id = int(os.environ.get("IBKR_CLIENT_ID", "17"))
        self.account = os.environ.get("IBKR_ACCOUNT", "")
        self.trading_mode = os.environ.get("IBKR_TRADING_MODE", "paper").lower()
        self.allow_live_trading = os.environ.get("IBKR_ALLOW_LIVE_TRADING", "false").lower() == "true"
        self.market_data_type = int(os.environ.get("IBKR_MARKET_DATA_TYPE", "3"))
        self.max_order_quantity = int(os.environ.get("IBKR_MAX_ORDER_QUANTITY", "100"))
        self._ib = None
        self._lock = threading.RLock()
        self._feed_cache = {}

    @staticmethod
    def _ensure_event_loop():
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    @staticmethod
    def _imports():
        try:
            from ib_insync import IB, MarketOrder, Stock
        except ImportError as exc:
            raise IBKRUnavailableError(
                "ib_insync is not installed; install requirements.txt in the app environment"
            ) from exc
        return IB, MarketOrder, Stock

    @property
    def connected(self):
        return bool(self._ib and self._ib.isConnected())

    def connect(self):
        with self._lock:
            if self.connected:
                return self.status()
            self._ensure_event_loop()
            IB, _, _ = self._imports()
            candidate_ports = tuple(dict.fromkeys((self.port, 7497, 4002)))
            errors = []
            for candidate_port in candidate_ports:
                self._ib = IB()
                try:
                    self._ib.connect(
                        self.host,
                        candidate_port,
                        clientId=self.client_id,
                        timeout=3,
                        readonly=False,
                        account=self.account,
                    )
                    self.port = candidate_port
                    self._ib.reqMarketDataType(self.market_data_type)
                    return self.status()
                except Exception as exc:
                    errors.append(f"{candidate_port}: {exc or type(exc).__name__}")
                    self._ib.disconnect()
                    self._ib = None
            raise IBKRUnavailableError(
                "Unable to connect to IBKR paper services ("
                + "; ".join(errors)
                + "). Start TWS on 7497 or IB Gateway on 4002 and enable socket clients."
            )

    def disconnect(self):
        with self._lock:
            if self._ib:
                self._ib.disconnect()
            self._ib = None

    def status(self):
        accounts = self._ib.managedAccounts() if self.connected else []
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "trading_mode": self.trading_mode,
            "market_data_type": self.market_data_type,
            "accounts": accounts,
        }

    def _require_connection(self):
        if not self.connected:
            raise IBKRUnavailableError("IBKR is not connected; start TWS or IB Gateway and connect first")

    def stock_feed(self, symbols):
        self._require_connection()
        _, _, Stock = self._imports()
        normalized = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip()))
        if not normalized or len(normalized) > 20:
            raise ValueError("provide between 1 and 20 stock symbols")

        cache_key = (normalized, self.market_data_type)
        cached = self._feed_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 20.0:
            return cached[1]

        with self._lock:
            contracts = [Stock(symbol, "SMART", "USD") for symbol in normalized]
            qualified = self._ib.qualifyContracts(*contracts)
            tickers = self._ib.reqTickers(*qualified)
            feed = []
            for contract, ticker in zip(qualified, tickers):
                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="3 M",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False,
                )
                closes = [bar.close for bar in bars]
                signal = calculate_volatility_signal(closes)
                market_price = float(ticker.marketPrice())
                if not math.isfinite(market_price) and closes:
                    market_price = float(closes[-1])
                previous_close = float(ticker.close) if math.isfinite(float(ticker.close)) else float(closes[-2])
                change_pct = (market_price / previous_close - 1.0) * 100.0 if previous_close else 0.0
                feed.append({
                    "ticker": contract.symbol,
                    "price": round(market_price, 4),
                    "bid": round(float(ticker.bid), 4) if math.isfinite(float(ticker.bid)) else None,
                    "ask": round(float(ticker.ask), 4) if math.isfinite(float(ticker.ask)) else None,
                    "change_pct": round(change_pct, 2),
                    "currency": contract.currency,
                    "exchange": contract.primaryExchange or contract.exchange,
                    **signal,
                })
            self._feed_cache[cache_key] = (time.monotonic(), feed)
            return feed

    def place_market_order(self, symbol, action, quantity, confirmed):
        self._require_connection()
        _, MarketOrder, Stock = self._imports()
        normalized_action = str(action).upper()
        normalized_symbol = str(symbol).strip().upper()
        quantity = int(quantity)
        if normalized_action not in {"BUY", "SELL"}:
            raise ValueError("action must be BUY or SELL")
        if not normalized_symbol.isalnum():
            raise ValueError("symbol must be alphanumeric")
        if quantity < 1 or quantity > self.max_order_quantity:
            raise ValueError(f"quantity must be between 1 and {self.max_order_quantity}")
        if confirmed is not True:
            raise ValueError("confirmed=true is required to transmit an order")
        if self.trading_mode != "paper" and not self.allow_live_trading:
            raise PermissionError("live trading is disabled; use paper mode or explicitly enable live trading")

        with self._lock:
            contract = Stock(normalized_symbol, "SMART", "USD")
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                raise ValueError(f"unable to qualify stock contract {normalized_symbol}")
            trade = self._ib.placeOrder(qualified[0], MarketOrder(normalized_action, quantity))
            self._ib.sleep(0.5)
            status = trade.orderStatus
            return {
                "order_id": status.orderId,
                "perm_id": status.permId,
                "status": status.status,
                "ticker": normalized_symbol,
                "action": normalized_action,
                "quantity": quantity,
                "filled": float(status.filled),
                "remaining": float(status.remaining),
                "average_fill_price": float(status.avgFillPrice),
                "trading_mode": self.trading_mode,
                "message": "Order transmitted to IBKR; status is broker-reported and may still be pending.",
            }