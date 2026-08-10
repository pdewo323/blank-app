#!/usr/bin/env python3
"""
AI Autonomous Hedge Fund - Streamlit Dashboard
==============================================
Reads signals.json produced by the background engine (ai_agent.py) and
displays account info, live prices, charts, the 3 agent signals, and the
consensus decision. Also supports manual trade execution via MT5.

Run:  streamlit run app.py
"""

import os
import json
import time
import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ---------- MT5 (lazy so the dashboard can still boot without MT5) ----------
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FILE = os.path.join(BASE_DIR, "signals.json")

st.set_page_config(page_title="AI Hedge Fund Dashboard", layout="wide")

# =========================================================
# Helpers
# =========================================================

@st.cache_data(ttl=5)
def load_signals():
    """Load the latest signals.json from the background engine."""
    if not os.path.exists(SIGNALS_FILE):
        return None
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_account_info():
    if mt5 is None:
        return None
    if not mt5.initialize():
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


def get_live_tick(symbol):
    if mt5 is None:
        return None
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return {"bid": tick.bid, "ask": tick.ask, "spread": tick.ask - tick.bid}


def get_positions():
    if mt5 is None:
        return []
    positions = mt5.positions_get()
    if not positions:
        return []
    out = []
    for p in positions:
        out.append({
            "Ticket": p.ticket,
            "Symbol": p.symbol,
            "Type": "BUY" if p.type == 0 else "SELL",
            "Volume": p.volume,
            "Open": p.price_open,
            "Profit": round(p.profit, 2),
        })
    return out


def execute_trade(symbol, order_type, volume, sl, tp, magic):
    if mt5 is None:
        return {"success": False, "error": "MetaTrader5 not installed"}
    if not mt5.initialize():
        return {"success": False, "error": f"MT5 init failed: {mt5.last_error()}"}

    info = mt5.symbol_info(symbol)
    if info is None:
        return {"success": False, "error": f"Symbol {symbol} not found"}
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": "Could not get price"}

    if order_type == "BUY":
        order_enum = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_enum = mt5.ORDER_TYPE_SELL
        price = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_enum,
        "price": price,
        "sl": float(sl) if sl else 0.0,
        "tp": float(tp) if tp else 0.0,
        "deviation": 20,
        "magic": int(magic),
        "comment": "AI Hedge Fund",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "error": "Order send returned no result"}
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": f"Retcode={result.retcode}: {result.comment}"}
    return {"success": True, "ticket": result.order, "price": result.price}


# =========================================================
# UI
# =========================================================

st.title("🏦 AI Autonomous Hedge Fund Terminal")
st.caption("Background engine: `ai_agent.py`  |  Dashboard: `app.py`")

# ---- Sidebar ----
st.sidebar.header("Status")
data = load_signals()
if data is None:
    st.sidebar.error("No signals.json found. Start the background engine first:\n\n`python ai_agent.py`")
else:
    ts = data.get("timestamp")
    try:
        ts_dt = datetime.datetime.fromisoformat(ts)
        age = (datetime.datetime.now() - ts_dt).total_seconds()
        st.sidebar.metric("Last update", ts_dt.strftime("%H:%M:%S"))
        if age > 300:
            st.sidebar.warning(f"Signal is {int(age)}s old — engine may be offline.")
        else:
            st.sidebar.success("Engine online")
    except Exception:
        st.sidebar.metric("Last update", "unknown")

    if data.get("connected"):
        st.sidebar.success("MT5 Connected")
    else:
        st.sidebar.warning("MT5 Not Connected")

# ---- Auto refresh ----
st.sidebar.markdown("---")
auto = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if auto:
    time.sleep(0.5)
    st.rerun()

# =========================================================
# If no data yet, show a static overview
# =========================================================
if data is None:
    st.info("Waiting for the AI engine to produce signals. Run `python ai_agent.py` in a separate terminal.")
    st.stop()

# =========================================================
# Account + Price row
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Account")
    acc = data.get("account")
    if acc:
        st.write(f"**{acc['name']}**")
        st.write(f"Login: `{acc['login']}`  |  Server: `{acc['server']}`")
        st.metric("Balance", f"{acc['balance']:,.2f} {acc['currency']}")
        st.metric("Equity", f"{acc['equity']:,.2f}")
        st.metric("Free Margin", f"{acc['margin_free']:,.2f}")
        st.write(f"Leverage: 1:{acc['leverage']}")
        st.write("Trading:", "✅ Enabled" if acc["trade_allowed"] else "❌ Disabled")
    else:
        st.info("No account data")

with col2:
    st.subheader("💹 Live Price")
    tick = data.get("tick")
    if tick:
        st.metric("Bid", f"{tick['bid']:.5f}")
        st.metric("Ask", f"{tick['ask']:.5f}")
        st.metric("Spread", f"{tick['spread']:.5f}")
    else:
        st.info("No tick data")
    st.subheader("📊 Market Snapshot")
    mk = data.get("market")
    if mk:
        st.metric("Close", f"{mk['close']:.5f}")
        st.metric("RSI", f"{mk['rsi']:.1f}")
        st.metric("ATR", f"{mk['atr']:.5f}")
        st.metric("MACD Hist", f"{mk['macd_hist']:.5f}")

with col3:
    st.subheader("📋 Open Positions")
    positions = data.get("positions") or []
    if positions:
        dfp = pd.DataFrame(positions)
        st.dataframe(dfp, use_container_width=True)
        total = sum(p["profit"] for p in positions)
        st.metric("Total P/L", f"{total:+.2f}")
    else:
        st.info("No open positions")

# =========================================================
# Error banner
# =========================================================
if data.get("error"):
    st.warning(f"⚠️ Engine note: {data['error']}")

# =========================================================
# Consensus header
# =========================================================
st.markdown("---")
consensus = data.get("consensus")
if consensus:
    bias = consensus["final_bias"]
    color = {"BUY": ":green", "SELL": ":red", "NEUTRAL": ":orange"}
    st.markdown(f"## 🧠 Consensus Signal: {color[bias]}**{bias}**")
    c = consensus
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Final Bias", bias)
    mc2.metric("Confidence", f"{c['confidence']}%")
    mc3.metric("Entry", f"{c['entry']:.5f}" if c["entry"] else "—")
    mc4.metric("Take Profit", f"{c['take_profit']:.5f}" if c["take_profit"] else "—")

    votes = c.get("votes", {})
    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("BUY votes", votes.get("BUY", 0))
    vc2.metric("SELL votes", votes.get("SELL", 0))
    vc3.metric("NEUTRAL votes", votes.get("NEUTRAL", 0))

# =========================================================
# Agent signals
# =========================================================
st.markdown("---")
st.subheader("🤖 Agent Signals")

agents = data.get("agents") or []
if agents:
    agent_icons = {"groq": "🚀", "hermes": "🔮", "free_auditor": "🛡️"}
    for a in agents:
        name = a["agent"]
        abias = a["bias"]
        acolor = {"BUY": "green", "SELL": "red", "NEUTRAL": "orange"}.get(abias, "grey")
        with st.expander(f"{agent_icons.get(name, '🤖')} {name.upper()} — {abias} (conf {a['confidence']}%)"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Bias", abias)
            c2.metric("Confidence", f"{a['confidence']}%")
            c3.metric("Entry", f"{a['entry']:.5f}" if a["entry"] else "—")
            c1b, c2b = st.columns(2)
            c1b.metric("Stop Loss", f"{a['stop_loss']:.5f}" if a["stop_loss"] else "—")
            c2b.metric("Take Profit", f"{a['take_profit']:.5f}" if a["take_profit"] else "—")
            st.write("**Reasoning:**", a.get("reasoning", ""))
else:
    st.info("No agent signals available yet.")

# =========================================================
# Manual trade execution
# =========================================================
st.markdown("---")
st.subheader("💼 Manual Trade Execution")

with st.expander("Open a trade", expanded=False):
    live_acc = get_account_info()
    if live_acc is None:
        st.info("MT5 not available in this terminal. Ensure MT5 is running and restart the dashboard.")
    else:
        st.write(f"Live account: **{live_acc['login']}** @ {live_acc['server']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            t_symbol = st.text_input("Symbol", data.get("symbol", "XAUUSD"))
        with c2:
            t_type = st.selectbox("Type", ["BUY", "SELL"])
        with c3:
            t_volume = st.number_input("Volume", min_value=0.01, value=0.01, step=0.01)

        # Prefill from consensus
        prefill_entry = consensus["entry"] if consensus and consensus.get("entry") else None
        prefill_sl = consensus["stop_loss"] if consensus and consensus.get("stop_loss") else None
        prefill_tp = consensus["take_profit"] if consensus and consensus.get("take_profit") else None

        c1, c2, c3 = st.columns(3)
        with c1:
            t_sl = st.text_input("Stop Loss (optional)", value=f"{prefill_sl:.5f}" if prefill_sl else "")
        with c2:
            t_tp = st.text_input("Take Profit (optional)", value=f"{prefill_tp:.5f}" if prefill_tp else "")
        with c3:
            t_magic = st.number_input("Magic", value=20240101, step=1)

        if st.button("🚀 Place Order", type="primary", use_container_width=True):
            slv = float(t_sl) if t_sl.strip() else None
            tpv = float(t_tp) if t_tp.strip() else None
            res = execute_trade(t_symbol, t_type, t_volume, slv, tpv, t_magic)
            if res["success"]:
                st.success(f"✅ Order placed! Ticket: {res['ticket']} @ {res['price']:.5f}")
            else:
                st.error(f"❌ Trade failed: {res['error']}")

# =========================================================
# Raw data
# =========================================================
with st.expander("View raw signals.json"):
    st.json(data)

st.caption(f"Dashboard refreshed at {datetime.datetime.now().strftime('%H:%M:%S')}")
