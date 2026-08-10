#!/usr/bin/env python3
"""
AI Autonomous Hedge Fund - Background Engine
============================================
Runs continuously in the background. Polls MT5 for market data, runs
3 independent analysis agents (Groq, Hermes, Free Auditor), combines
their signals via consensus voting, and writes the result to signals.json
for the Streamlit dashboard (app.py) to display.

Run:  python ai_agent.py
"""

import os
import json
import time
import datetime
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AI-Engine")

# ---------- MT5 ----------
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    log.warning("MetaTrader5 not installed. Install: pip install MetaTrader5")

# ---------- AI clients ----------
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =========================================================
# Configuration
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FILE = os.path.join(BASE_DIR, "signals.json")
ENV_FILE = os.path.join(BASE_DIR, ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Try to load from .env if env vars not set
if (not GROQ_API_KEY or not OPENAI_API_KEY) and os.path.exists(ENV_FILE):
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    except ImportError:
        pass

SYMBOL = os.environ.get("AI_SYMBOL", "XAUUSD")
TIMEFRAME = os.environ.get("AI_TIMEFRAME", "M15")
POLL_INTERVAL = int(os.environ.get("AI_POLL_SECONDS", "60"))  # seconds between cycles
BAR_COUNT = 300

GROQ_MODEL = "llama-3.3-70b-versatile"
HERMES_MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"  # Hermes via Groq

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1 if mt5 else 1,
    "M5": mt5.TIMEFRAME_M5 if mt5 else 5,
    "M15": mt5.TIMEFRAME_M15 if mt5 else 15,
    "M30": mt5.TIMEFRAME_M30 if mt5 else 30,
    "H1": mt5.TIMEFRAME_H1 if mt5 else 60,
    "H4": mt5.TIMEFRAME_H4 if mt5 else 240,
    "D1": mt5.TIMEFRAME_D1 if mt5 else 1440,
}

# =========================================================
# Technical Indicators
# =========================================================

def get_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def get_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = get_rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = get_macd(df["close"])
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # ATR
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Bollinger
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std

    df["pct_change"] = close.pct_change() * 100
    return df


# =========================================================
# MT5 Data
# =========================================================

def fetch_rates(symbol: str, timeframe: int, count: int = BAR_COUNT) -> Optional[pd.DataFrame]:
    if mt5 is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )
    return df


def get_account_info() -> Optional[dict]:
    if mt5 is None:
        return None
    info = mt5.account_info()
    if info is None:
        return None
    return {
        "login": info.login,
        "server": info.server,
        "name": info.name,
        "balance": info.balance,
        "equity": info.equity,
        "margin_free": info.margin_free,
        "leverage": info.leverage,
        "currency": info.currency,
        "trade_allowed": bool(info.trade_allowed),
    }


def get_latest_tick(symbol: str) -> Optional[dict]:
    if mt5 is None:
        return None
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return {"bid": tick.bid, "ask": tick.ask, "spread": tick.ask - tick.bid}


def get_open_positions() -> List[dict]:
    if mt5 is None:
        return []
    positions = mt5.positions_get()
    if not positions:
        return []
    out = []
    for p in positions:
        out.append({
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == 0 else "SELL",
            "volume": p.volume,
            "price_open": p.price_open,
            "profit": p.profit,
        })
    return out


# =========================================================
# Market context builder
# =========================================================

def build_market_context(df: pd.DataFrame, symbol: str, tf_label: str) -> str:
    last = df.iloc[-1]
    trend = "UP" if last["ema_20"] > last["ema_50"] else "DOWN"
    macd_state = "BULLISH" if last["macd"] > last["macd_signal"] else "BEARISH"
    rsi = last["rsi"]
    rsi_state = "OVERBOUGHT" if rsi >= 70 else ("OVERSOLD" if rsi <= 30 else "NEUTRAL")
    bb_state = "UPPER_BAND" if last["close"] > last["bb_upper"] else \
               ("LOWER_BAND" if last["close"] < last["bb_lower"] else "MIDDLE")
    recent = df.tail(5)[["time", "open", "high", "low", "close"]].to_string(index=False)

    return f"""
SYMBOL: {symbol}
TIMEFRAME: {tf_label}

LATEST PRICE DATA:
- Close: {last['close']:.5f}
- Open: {last['open']:.5f}
- High: {last['high']:.5f}
- Low: {last['low']:.5f}
- Pct Change: {last['pct_change']:.3f}%
- ATR: {last['atr']:.5f}

TECHNICAL SUMMARY:
- Trend (EMA20 vs EMA50): {trend}
- RSI({last['rsi']:.1f}): {rsi_state}
- MACD: {macd_state}
- MACD Histogram: {last['macd_hist']:.5f}
- Bollinger Position: {bb_state}

LAST 5 CANDLES:
{recent}

RULES:
- You are a professional FOREX/GOLD analyst.
- Analyze ONLY the data provided.
- Return EXACTLY this JSON format (no markdown, no extra text):
{{"bias":"BUY or SELL or NEUTRAL","entry":<number>,"stop_loss":<number>,"take_profit":<number>,"confidence":<0-100>,"reasoning":"<2-3 sentences>"}}
"""


AGENT_SYSTEM_PROMPT = (
    "You are an expert algorithmic trading analyst. "
    "Always respond with valid JSON only, using the exact schema requested."
)


def _call_groq_llm(model: str, context: str) -> str:
    if Groq is None:
        raise RuntimeError("groq package not installed")
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content


# =========================================================
# The 3 Agents
# =========================================================

def agent_groq(context: str) -> dict:
    """Groq LLM agent."""
    raw = _call_groq_llm(GROQ_MODEL, context)
    return parse_signal(raw, "groq")


def agent_hermes(context: str) -> dict:
    """Hermes LLM agent (via Groq)."""
    raw = _call_groq_llm(HERMES_MODEL, context)
    return parse_signal(raw, "hermes")


def agent_free_auditor(df: pd.DataFrame, symbol: str) -> dict:
    """Rule-based free auditor - no API cost. Independent technical opinion."""
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    score = 0.0
    reasons = []

    # Trend
    if last["ema_20"] > last["ema_50"]:
        score += 1
        reasons.append("EMA20 above EMA50 (uptrend)")
    else:
        score -= 1
        reasons.append("EMA20 below EMA50 (downtrend)")

    # RSI
    if last["rsi"] < 30:
        score += 1
        reasons.append(f"RSI oversold ({last['rsi']:.1f})")
    elif last["rsi"] > 70:
        score -= 1
        reasons.append(f"RSI overbought ({last['rsi']:.1f})")

    # MACD
    if last["macd"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD bullish cross")
    else:
        score -= 1
        reasons.append("MACD bearish cross")

    # Momentum
    if last["close"] > prev["close"]:
        score += 0.5
    else:
        score -= 0.5

    # Bollinger
    if last["close"] > last["bb_upper"]:
        score -= 0.5
        reasons.append("Price above upper band (overextension)")
    elif last["close"] < last["bb_lower"]:
        score += 0.5
        reasons.append("Price below lower band (oversold extension)")

    if score >= 1.5:
        bias = "BUY"
    elif score <= -1.5:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    atr = last["atr"]
    price = last["close"]
    if bias == "BUY":
        entry = price
        stop_loss = price - 1.5 * atr
        take_profit = price + 2.0 * atr
    elif bias == "SELL":
        entry = price
        stop_loss = price + 1.5 * atr
        take_profit = price - 2.0 * atr
    else:
        entry = price
        stop_loss = price - 1.5 * atr
        take_profit = price + 1.5 * atr

    confidence = min(95, int(50 + abs(score) * 15))

    return {
        "agent": "free_auditor",
        "bias": bias,
        "entry": round(entry, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit": round(take_profit, 5),
        "confidence": confidence,
        "reasoning": "; ".join(reasons) if reasons else "No strong setup detected.",
    }


def parse_signal(raw: str, agent_name: str) -> dict:
    """Parse LLM output into a structured signal dict."""
    default = {
        "agent": agent_name,
        "bias": "NEUTRAL",
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "confidence": 0,
        "reasoning": "Failed to parse LLM output.",
    }
    # Strip markdown fences
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
    # Locate JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        default["reasoning"] = raw[:200]
        return default
    try:
        data = json.loads(raw[start:end + 1])
    except Exception:
        default["reasoning"] = raw[start:end + 1][:200]
        return default

    bias = str(data.get("bias", "NEUTRAL")).upper()
    if bias not in ("BUY", "SELL", "NEUTRAL"):
        bias = "NEUTRAL"

    def _num(v):
        try:
            return round(float(v), 5) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "agent": agent_name,
        "bias": bias,
        "entry": _num(data.get("entry")),
        "stop_loss": _num(data.get("stop_loss")),
        "take_profit": _num(data.get("take_profit")),
        "confidence": int(data.get("confidence", 0) or 0),
        "reasoning": str(data.get("reasoning", ""))[:300],
    }


def compute_consensus(agents: List[dict]) -> dict:
    """Vote on the 3 agent signals to produce a final decision."""
    votes_buy = sum(1 for a in agents if a["bias"] == "BUY")
    votes_sell = sum(1 for a in agents if a["bias"] == "SELL")
    votes_neutral = sum(1 for a in agents if a["bias"] == "NEUTRAL")

    if votes_buy > votes_sell and votes_buy >= 2:
        final_bias = "BUY"
    elif votes_sell > votes_buy and votes_sell >= 2:
        final_bias = "SELL"
    else:
        final_bias = "NEUTRAL"

    confidence = 0
    for a in agents:
        if a["bias"] == final_bias:
            confidence += a["confidence"]
    confidence = int(confidence / max(1, sum(1 for a in agents if a["bias"] == final_bias)))

    # Price levels from the most confident agreeing agent
    entry = sl = tp = None
    for a in agents:
        if a["bias"] == final_bias and a["entry"] is not None:
            if entry is None or a["confidence"] > confidence:
                entry, sl, tp = a["entry"], a["stop_loss"], a["take_profit"]

    return {
        "final_bias": final_bias,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "confidence": confidence,
        "votes": {"BUY": votes_buy, "SELL": votes_sell, "NEUTRAL": votes_neutral},
    }


# =========================================================
# Main loop
# =========================================================

def run_cycle() -> dict:
    """One full cycle: fetch data, run 3 agents, compute consensus."""
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "connected": False,
        "account": None,
        "tick": None,
        "positions": [],
        "agents": [],
        "consensus": None,
        "error": None,
    }

    # Connect MT5
    if mt5 is not None:
        if not mt5.initialize():
            result["error"] = f"MT5 init failed: {mt5.last_error()}"
            log.warning(result["error"])
            return result
        result["connected"] = True
        result["account"] = get_account_info()
        result["tick"] = get_latest_tick(SYMBOL)
        result["positions"] = get_open_positions()
    else:
        result["error"] = "MetaTrader5 package not installed"
        return result

    # Fetch data
    df = fetch_rates(SYMBOL, TIMEFRAMES[TIMEFRAME], BAR_COUNT)
    if df is None or len(df) < 50:
        result["error"] = f"Not enough data for {SYMBOL}"
        log.warning(result["error"])
        return result

    df = compute_indicators(df)
    context = build_market_context(df, SYMBOL, TIMEFRAME)

    # Ensure API keys present
    if not GROQ_API_KEY:
        result["error"] = "GROQ_API_KEY not set. Create .env or set environment variable."
        log.warning(result["error"])
        return result

    # Run 3 agents
    agents = []
    try:
        agents.append(agent_groq(context))
        log.info("Groq agent signal: %s (conf %s)", agents[-1]["bias"], agents[-1]["confidence"])
    except Exception as e:
        log.error("Groq agent failed: %s", e)
        agents.append({"agent": "groq", "bias": "NEUTRAL", "entry": None,
                       "stop_loss": None, "take_profit": None, "confidence": 0,
                       "reasoning": f"Error: {e}"})

    try:
        agents.append(agent_hermes(context))
        log.info("Hermes agent signal: %s (conf %s)", agents[-1]["bias"], agents[-1]["confidence"])
    except Exception as e:
        log.error("Hermes agent failed: %s", e)
        agents.append({"agent": "hermes", "bias": "NEUTRAL", "entry": None,
                       "stop_loss": None, "take_profit": None, "confidence": 0,
                       "reasoning": f"Error: {e}"})

    try:
        agents.append(agent_free_auditor(df, SYMBOL))
        log.info("Free auditor signal: %s (conf %s)", agents[-1]["bias"], agents[-1]["confidence"])
    except Exception as e:
        log.error("Free auditor failed: %s", e)
        agents.append({"agent": "free_auditor", "bias": "NEUTRAL", "entry": None,
                       "stop_loss": None, "take_profit": None, "confidence": 0,
                       "reasoning": f"Error: {e}"})

    result["agents"] = agents
    result["consensus"] = compute_consensus(agents)

    # Save market snapshot for dashboard
    result["market"] = {
        "close": float(df.iloc[-1]["close"]),
        "rsi": float(df.iloc[-1]["rsi"]),
        "ema20": float(df.iloc[-1]["ema_20"]),
        "ema50": float(df.iloc[-1]["ema_50"]),
        "atr": float(df.iloc[-1]["atr"]),
        "macd_hist": float(df.iloc[-1]["macd_hist"]),
    }

    return result


def save_result(result: dict):
    tmp = SIGNALS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp, SIGNALS_FILE)


def main():
    log.info("=" * 50)
    log.info("AI Autonomous Hedge Fund Engine starting")
    log.info("Symbol=%s TF=%s Poll=%ss", SYMBOL, TIMEFRAME, POLL_INTERVAL)
    log.info("Signal file: %s", SIGNALS_FILE)
    if GROQ_API_KEY:
        log.info("Groq API key: set")
    else:
        log.warning("GROQ_API_KEY not set - LLM agents will fail. Create .env file.")
    log.info("=" * 50)

    while True:
        try:
            log.info("Running analysis cycle...")
            result = run_cycle()
            save_result(result)
            if result["consensus"]:
                c = result["consensus"]
                log.info("Consensus: %s (conf %s%%)", c["final_bias"], c["confidence"])
            log.info("Cycle complete. Waiting %s seconds...", POLL_INTERVAL)
        except KeyboardInterrupt:
            log.info("Shutting down.")
            if mt5 is not None:
                mt5.shutdown()
            break
        except Exception as e:
            log.error("Unexpected error in cycle: %s", e)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
