#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# telegram_finance_bot.py
# -------------------------------------------------------------
# ONE FILE: Stocks search + charts + AI-ish insights + Investment planner with pie chart
# + Company actions: Balance Sheet, Ratios, Peer Comparison, Pros & Cons
# -------------------------------------------------------------

import os
import math
import requests
import telebot
from telebot import types
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import difflib
import re

# ---------------- CONFIG ----------------
TOKEN = os.getenv("BOT_TOKEN", "7957451812:AAG1_yo8Jwmc2m01uuEjwxhuUsAty_pskpU")  # Prefer env var
bot = telebot.TeleBot(TOKEN, parse_mode=None)  # set parse_mode per-message

NIFTY50_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NIFTY50_FILE = "nifty50.csv"
NIFTY500_FILE = "nifty500.csv"

# chat_state: per-chat, we track modes and stages
chat_state = {}

# ---------------- NAME → TICKER MAP (popular) ----------------
NAME_TICKER = {
    # US mega caps
    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "micron": "MU", "broadcom": "AVGO", "salesforce": "CRM", "adobe": "ADBE",
    "paypal": "PYPL", "visa": "V", "mastercard": "MA", "coca cola": "KO",
    "cocacola": "KO", "pepsi": "PEP", "jpmorgan": "JPM", "berkshire": "BRK-B",
    "boeing": "BA", "disney": "DIS", "uber": "UBER",
    # India large caps (NSE suffix)
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS", "icici bank": "ICICIBANK.NS", "itc": "ITC.NS",
    "sbi": "SBIN.NS", "larsen & toubro": "LT.NS", "lnt": "LT.NS",
    "bharti airtel": "BHARTIARTL.NS", "hcl": "HCLTECH.NS", "ongc": "ONGC.NS",
    "axis bank": "AXISBANK.NS", "kotak bank": "KOTAKBANK.NS", "maruti": "MARUTI.NS",
    "titan": "TITAN.NS", "ultratech": "ULTRACEMCO.NS", "jsw steel": "JSWSTEEL.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    # Other exchanges (handful for convenience)
    "taiwan semiconductor": "TSM", "asml": "ASML",
    "sap": "SAP", "siemens": "SIE.DE", "lvmh": "MC.PA", "hsbc": "0005.HK",
    "shopify": "SHOP.TO", "rio tinto": "RIO.L", "sony": "6758.T",
    "dbs": "D05.SI", "nestle": "NESN.SW", "toyota": "7203.T",
}

# Predefined universes used to fetch peers quickly (keeps it fast & reliable)
INDIA_LARGE_UNIVERSE = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","ITC.NS","SBIN.NS",
    "LT.NS","BHARTIARTL.NS","HCLTECH.NS","AXISBANK.NS","KOTAKBANK.NS","MARUTI.NS",
    "ULTRACEMCO.NS","JSWSTEEL.NS","HINDUNILVR.NS","TITAN.NS"
]
US_LARGE_UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","V","MA","KO","PEP","INTC",
    "AMD","ADBE","CRM","NFLX","AVGO","BA","DIS","PYPL","UBER"
]

# ---------------- UTILITIES ----------------
def try_download(url, filename):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {filename}")
        return True
    except Exception as e:
        print(f"Download failed for {url}: {e}")
        return False

def load_index_csv(local_filename):
    if not os.path.exists(local_filename):
        return pd.DataFrame(columns=["SYMBOL", "NAME"])
    try:
        df = pd.read_csv(local_filename)
        symbol_col = None
        name_col = None
        for c in df.columns:
            low = c.lower()
            if "symbol" in low and symbol_col is None:
                symbol_col = c
            if ("company" in low or "name" in low) and name_col is None:
                name_col = c
        if symbol_col is None:
            symbol_col = df.columns[0]
        if name_col is None:
            name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        df2 = df[[symbol_col, name_col]].copy()
        df2.columns = ["SYMBOL","NAME"]
        df2["SYMBOL"] = df2["SYMBOL"].astype(str).str.strip().str.upper()
        df2["NAME"] = df2["NAME"].astype(str).str.strip().str.lower()
        return df2
    except Exception as e:
        print("Error parsing CSV:", e)
        return pd.DataFrame(columns=["SYMBOL", "NAME"])

def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=(period-1), adjust=False).mean()
    ema_down = down.ewm(com=(period-1), adjust=False).mean()
    rs = ema_up / (ema_down.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def human_volume(v):
    try:
        v = int(v)
        if v >= 10**12:  # trillion
            return f"{v/10**12:.2f}T"
        if v >= 10**9:
            return f"{v/10**9:.2f}B"
        if v >= 10**7:
            return f"{v/10**7:.2f}Cr"
        if v >= 10**5:
            return f"{v/10**5:.2f}L"
        if v >= 1000:
            return f"{v/1000:.2f}K"
        return str(v)
    except:
        return str(v)

def human_number(n):
    try:
        n = float(n)
        absn = abs(n)
        if absn >= 1e12: return f"{n/1e12:.2f}T"
        if absn >= 1e9:  return f"{n/1e9:.2f}B"
        if absn >= 1e7:  return f"{n/1e7:.2f}Cr"
        if absn >= 1e5:  return f"{n/1e5:.2f}L"
        if absn >= 1e3:  return f"{n/1e3:.2f}K"
        return f"{n:.0f}"
    except:
        return str(n)

def pct(n):
    try:
        return f"{float(n)*100:.1f}%"
    except:
        return "—"

def create_chart_bytes(df, title):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df.index, df["Close"], marker="o", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.grid(True)
    fig.autofmt_xdate()
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf

def currency_symbol_for(used_symbol: str) -> str:
    if not used_symbol:
        return ""
    s = used_symbol.upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return "₹"
    if s.endswith(".L"):
        return "£"
    if s.endswith(".PA") or s.endswith(".DE") or s.endswith(".AS") or s.endswith(".SW"):
        return "€"
    if s.endswith(".HK"):
        return "HK$"
    if s.endswith(".TO"):
        return "C$"
    if s.endswith(".AX"):
        return "A$"
    if s.endswith(".T"):
        return "¥"
    if s.endswith(".SI"):
        return "S$"
    if s.endswith(".SS") or s.endswith(".SZ"):
        return "¥"
    return "$"  # default

# ---------------- PREPARE DATASETS ----------------
if not try_download(NIFTY50_URL, NIFTY50_FILE):
    nifty50_df = pd.DataFrame([
        {"SYMBOL":"RELIANCE","NAME":"reliance industries limited"},
        {"SYMBOL":"TCS","NAME":"tata consultancy services limited"},
        {"SYMBOL":"INFY","NAME":"infosys limited"},
        {"SYMBOL":"HDFCBANK","NAME":"hdfc bank limited"},
        {"SYMBOL":"ITC","NAME":"itc limited"}
    ])
else:
    nifty50_df = load_index_csv(NIFTY50_FILE)

if not try_download(NIFTY500_URL, NIFTY500_FILE):
    nifty500_df = pd.DataFrame([
        {"SYMBOL":"RELIANCE","NAME":"reliance industries limited"},
        {"SYMBOL":"TCS","NAME":"tata consultancy services limited"},
        {"SYMBOL":"ZOMATO","NAME":"zomato limited"},
        {"SYMBOL":"KNRCON","NAME":"knr constructions limited"},
        {"SYMBOL":"IIL","NAME":"iil limited"}
    ])
else:
    nifty500_df = load_index_csv(NIFTY500_FILE)

all_df = pd.concat([nifty50_df, nifty500_df], ignore_index=True)\
           .drop_duplicates(subset="SYMBOL").reset_index(drop=True)

# ---------------- RESOLUTION: name → ticker with fuzzy suggestions ----------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def lookup_name_map(query: str):
    key = normalize_text(query)
    return NAME_TICKER.get(key)

def fuzzy_suggestions(query: str, k=7, cutoff=0.65):
    q = normalize_text(query)
    map_names = list(NAME_TICKER.keys())
    map_matches = difflib.get_close_matches(q, map_names, n=k, cutoff=cutoff)

    suggestions = []
    for name in map_matches:
        suggestions.append((name.title(), NAME_TICKER[name], "map"))

    nse_names = all_df["NAME"].dropna().unique().tolist()
    nse_matches = difflib.get_close_matches(q, nse_names, n=k, cutoff=cutoff)
    for nm in nse_matches:
        row = all_df.loc[all_df["NAME"] == nm].head(1)
        if not row.empty:
            sym = row.iloc[0]["SYMBOL"]
            suggestions.append((nm.title(), sym + ".NS", "nse"))

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for disp, tick, src in suggestions:
        key = (disp.lower(), tick.upper())
        if key not in seen:
            seen.add(key)
            deduped.append((disp, tick, src))
    return deduped[:k]

# ---------------- SEARCH FUNCTION (kept for NSE lists) ----------------
def search_companies(query, max_results=10):
    q = str(query).strip()
    if not q:
        return []
    q_low = q.lower()
    exact = all_df[all_df["SYMBOL"] == q.upper()]
    results = []
    if not exact.empty:
        for _, r in exact.iterrows():
            results.append((r["SYMBOL"], r["NAME"]))
        return results[:max_results]
    mask = all_df["NAME"].str.contains(q_low, na=False)
    filtered = all_df[mask].head(max_results)
    for _, r in filtered.iterrows():
        results.append((r["SYMBOL"], r["NAME"]))
    return results

# ---------------- TICKER PROBE (multi-exchange) ----------------
COMMON_SUFFIXES = [
    ".NS", ".BO",  # India
    ".L", ".DE", ".PA", ".AS", ".SW",  # UK/DE/FR/NL/CH
    ".HK", ".TO", ".AX", ".SI", ".T",  # HK/Canada/Australia/Singapore/Japan
    ".SS", ".SZ"  # China
]
INDEX_MAP = {"NIFTY":"^NSEI","NIFTY50":"^NSEI","BANKNIFTY":"^NSEBANK","NIFTYBANK":"^NSEBANK"}

def probe_yf_symbol(symbol):
    symbol = symbol.strip().upper()

    # Indices
    if symbol in INDEX_MAP:
        try:
            t = yf.Ticker(INDEX_MAP[symbol])
            if not t.history(period="1d").empty:
                return t, INDEX_MAP[symbol]
        except:
            pass

    # Try raw first (for US/global)
    try:
        t = yf.Ticker(symbol)
        if not t.history(period="3d").empty:
            return t, symbol
    except:
        pass

    # NSE/BSE
    for cand in (symbol + ".NS", symbol + ".BO"):
        try:
            t = yf.Ticker(cand)
            if not t.history(period="3d").empty:
                return t, cand
        except:
            pass

    # Other common suffixes
    for suf in COMMON_SUFFIXES:
        cand = symbol + suf
        try:
            t = yf.Ticker(cand)
            if not t.history(period="3d").empty:
                return t, cand
        except:
            pass

    return None, None

# ---------------- MENUS ----------------
def main_menu_keyboard():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("📈 Stocks", "💰 Invest")
    kb.add("🔍 Search Again")
    return kb

# ---------------- TELEGRAM HANDLERS ----------------
@bot.message_handler(commands=["start","help"])
def cmd_start(msg):
    bot.send_message(
        msg.chat.id,
        "Welcome! Choose:\n• 📈 Stocks: Search tickers, charts, insights + BS/Ratios/Peers/Pros&Cons\n• 💰 Invest: Plan allocation with pie chart",
        reply_markup=main_menu_keyboard()
    )
    chat_state[msg.chat.id] = {"stage":"await_query"}  # default to Stocks search mode

@bot.message_handler(func=lambda m: m.text == "📈 Stocks")
def cmd_stocks(msg):
    chat_state[msg.chat.id] = {"stage":"await_query"}  # switch to stocks mode
    bot.send_message(msg.chat.id, "Type company name or ticker (e.g., apple, tesla, reliance, tcs).")

@bot.message_handler(func=lambda m: m.text == "💰 Invest")
def cmd_invest(msg):
    chat_state[msg.chat.id] = {"mode":"invest", "invest_stage":"await_amount"}
    bot.send_message(msg.chat.id, "Enter the amount you want to invest (in ₹):")

# ✅ FIX: make reply-keyboard 'Search Again' robust (emoji or no emoji)
@bot.message_handler(func=lambda m: m.text and m.text.strip().lower().replace("🔍","").strip() == "search again")
def cmd_search_again(msg):
    state = chat_state.get(msg.chat.id, {})
    if state.get("mode") == "invest":
        chat_state[msg.chat.id] = {"mode":"invest", "invest_stage":"await_amount"}
        bot.send_message(msg.chat.id, "Okay — enter the amount again (₹):")
    else:
        chat_state[msg.chat.id] = {"stage":"await_query"}
        bot.send_message(msg.chat.id, "Okay — type company name or symbol to search:")

def build_suggestion_markup(suggestions):
    markup = types.InlineKeyboardMarkup()
    for disp, tick, src in suggestions:
        show = f"{disp} ({tick})"
        markup.add(types.InlineKeyboardButton(show, callback_data=f"choose|{tick}"))
    markup.add(types.InlineKeyboardButton("🔍 Search Again", callback_data="search_again"))
    return markup

# ---------------- INVESTMENT FLOW: amount message handler ----------------
@bot.message_handler(func=lambda m: chat_state.get(m.chat.id,{}).get("mode")=="invest" and chat_state.get(m.chat.id,{}).get("invest_stage")=="await_amount")
def handle_invest_amount(msg):
    chat_id = msg.chat.id
    txt = (msg.text or "").strip().replace(",","")
    try:
        amount = float(txt)
        if amount <= 0:
            raise ValueError("non-positive")
        chat_state[chat_id]["amount"] = amount
        chat_state[chat_id]["invest_stage"] = "await_risk"
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("Low", callback_data="risk_low"),
                   types.InlineKeyboardButton("Medium", callback_data="risk_medium"),
                   types.InlineKeyboardButton("High", callback_data="risk_high"))
        bot.send_message(chat_id, "Select your risk preference:", reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, "❌ Please enter a valid positive number (e.g., 15000).")

# ---------------- STOCKS FLOW: free text search handler ----------------
@bot.message_handler(func=lambda m: chat_state.get(m.chat.id,{}).get("stage")=="await_query")
def handle_query(msg):
    chat_id = msg.chat.id
    q = msg.text.strip()
    bot.send_chat_action(chat_id, "typing")

    mapped = lookup_name_map(q)
    if mapped:
        matches = [(mapped.replace(".NS","").replace(".BO",""), mapped)]
    else:
        matches_raw = search_companies(q, max_results=10)
        matches = []
        for sym, name in matches_raw:
            matches.append((sym, f"{name}"))
        if not matches:
            t, used = probe_yf_symbol(q)
            if t:
                matches = [(used.replace(".NS","").replace(".BO",""), used)]

    if not matches:
        suggestions = fuzzy_suggestions(q, k=7, cutoff=0.6)
        if suggestions:
            bot.send_message(
                chat_id,
                "❓ I couldn't find an exact match. Did you mean one of these?",
                reply_markup=build_suggestion_markup(suggestions)
            )
        else:
            bot.send_message(
                chat_id,
                "❌ No matches found. Try another name/symbol (e.g., *Apple*, *Tesla*, *Reliance*, *TCS*).",
                parse_mode="Markdown"
            )
        chat_state[chat_id] = {"stage":"await_query"}
        return

    chat_state[chat_id] = {"stage":"choose", "matches":matches}
    markup = types.InlineKeyboardMarkup()
    for sym, name in matches:
        display = f"{name} ({sym})"
        markup.add(types.InlineKeyboardButton(display, callback_data=f"choose|{sym}"))
    markup.add(types.InlineKeyboardButton("🔍 Search Again", callback_data="search_again"))
    bot.send_message(chat_id, "Select the correct company:", reply_markup=markup)

# ---------------- CALLBACK ROUTER (Stocks & Invest inline buttons) ----------------
@bot.callback_query_handler(func=lambda call: True)
def callback_router(call):
    data = call.data
    chat_id = call.message.chat.id
    bot.answer_callback_query(call.id)

    # Common inline 'Search Again'
    if data == "search_again":
        state = chat_state.get(chat_id, {})
        if state.get("mode") == "invest":
            chat_state[chat_id] = {"mode":"invest", "invest_stage":"await_amount"}
            bot.send_message(chat_id, "Search again — enter amount (₹):", reply_markup=main_menu_keyboard())
        else:
            chat_state[chat_id] = {"stage":"await_query"}
            bot.send_message(chat_id, "Search again — type company name or symbol.", reply_markup=main_menu_keyboard())
        return

    # ---------------- INVEST FLOW BUTTONS ----------------
    if data.startswith("risk_"):
        risk = data.split("_",1)[1]  # low/medium/high
        st = chat_state.get(chat_id, {})
        if st.get("mode") == "invest":
            st["risk"] = risk
            st["invest_stage"] = "await_choice"
            markup = types.InlineKeyboardMarkup()
            markup.add(types.InlineKeyboardButton("📂 Diversify", callback_data="invest_diversify"),
                       types.InlineKeyboardButton("🎯 Single", callback_data="invest_single"))
            bot.send_message(chat_id, "Diversify portfolio or invest in a single option?", reply_markup=markup)
            return

    if data in ("invest_diversify", "invest_single"):
        st = chat_state.get(chat_id, {})
        if st.get("mode") == "invest":
            amount = float(st.get("amount", 0))
            risk = st.get("risk", "medium")
            if amount <= 0:
                st["invest_stage"] = "await_amount"
                bot.send_message(chat_id, "❌ Amount missing. Enter amount (₹):")
                return

            if data == "invest_diversify":
                plan = generate_allocation_plan(risk)
                text = "📊 Diversified Portfolio:\n"
                for k, v in plan.items():
                    text += f"- {int(v*100)}% (₹{int(round(amount*v))}) → {k}\n"
                bot.send_message(chat_id, text)
                pie_buf = create_pie_chart_bytes(plan, title="Investment Allocation")
                bot.send_photo(chat_id, pie_buf)
            else:
                option = single_invest_option(risk)
                text = f"🎯 Single Investment Choice:\nInvest ₹{int(round(amount))} fully into {option}"
                bot.send_message(chat_id, text)

            markup = types.InlineKeyboardMarkup()
            markup.add(types.InlineKeyboardButton("🔄 Recalculate", callback_data="invest_recalc"))
            markup.add(types.InlineKeyboardButton("🏠 Main Menu", callback_data="go_home"))
            bot.send_message(chat_id, "Do you want to try again?", reply_markup=markup)
            return

    if data == "invest_recalc":
        chat_state[chat_id] = {"mode":"invest", "invest_stage":"await_amount"}
        bot.send_message(chat_id, "Enter the amount you want to invest (₹):")
        return

    if data == "go_home":
        chat_state[chat_id] = {"stage":"await_query"}
        bot.send_message(chat_id, "Back to main menu. Choose an option:", reply_markup=main_menu_keyboard())
        return

    # ---------------- STOCKS FLOW BUTTONS ----------------
    if data.startswith("choose|"):
        sym = data.split("|",1)[1]
        chat_state[chat_id] = {"stage":"selected", "symbol":sym}
        markup = types.InlineKeyboardMarkup()
        # Existing
        markup.add(types.InlineKeyboardButton("📜 7-Day History + Chart", callback_data=f"action|history7|{sym}"))
        markup.add(types.InlineKeyboardButton("📊 30-Day Chart", callback_data=f"action|chart30|{sym}"))
        markup.add(types.InlineKeyboardButton("🤖 Insights", callback_data=f"action|insights|{sym}"))
        # New
        markup.add(types.InlineKeyboardButton("📄 Balance Sheet", callback_data=f"action|balance|{sym}"))
        markup.add(types.InlineKeyboardButton("📊 Ratios", callback_data=f"action|ratios|{sym}"))
        markup.add(types.InlineKeyboardButton("🏢 Peer Comparison", callback_data=f"action|peers|{sym}"))
        markup.add(types.InlineKeyboardButton("✅ Pros & Cons", callback_data=f"action|proscons|{sym}"))
        markup.add(types.InlineKeyboardButton("🔍 Search Again", callback_data="search_again"))
        bot.send_message(chat_id, f"Selected: {sym}\nChoose action:", reply_markup=markup)
        return

    if data.startswith("action|"):
        _, action, sym = data.split("|",2)
        base_sym = sym
        if action == "history7":
            send_history_and_chart(chat_id, base_sym, days=7)
        elif action == "chart30":
            send_history_and_chart(chat_id, base_sym, days=30, only_chart=True)
        elif action == "insights":
            send_ai_insight(chat_id, base_sym)
        elif action == "balance":
            send_balance_sheet(chat_id, base_sym)
        elif action == "ratios":
            send_ratios(chat_id, base_sym)
        elif action == "peers":
            send_peer_comparison(chat_id, base_sym)
        elif action == "proscons":
            send_pros_cons(chat_id, base_sym)
        return

# ---------------- INVESTMENT HELPERS ----------------
def generate_allocation_plan(risk: str):
    risk = (risk or "medium").lower()
    if risk == "low":
        return {"FD & Bonds": 0.6, "Index Fund": 0.3, "Gold ETF": 0.1}
    elif risk == "high":
        return {"Equity (Mid & Small Cap)": 0.5, "Crypto": 0.3, "Tech ETF": 0.2}
    else:
        return {"Nifty 50 Index Fund": 0.5, "Gold ETF": 0.3, "Large-cap stocks": 0.2}

def single_invest_option(risk: str):
    risk = (risk or "medium").lower()
    if risk == "low": return "Index Fund"
    if risk == "high": return "Crypto"
    return "Gold ETF"

def create_pie_chart_bytes(plan: dict, title="Investment Allocation"):
    labels = list(plan.keys())
    sizes = list(plan.values())
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)
    ax.axis("equal")
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------- DATA RESPONSES (STOCKS) ----------------
def fetch_ticker(symbol):
    t, used = probe_yf_symbol(symbol)
    return t, used

def send_history_and_chart(chat_id, symbol, days=7, only_chart=False):
    t, used = fetch_ticker(symbol)
    if t is None:
        bot.send_message(chat_id, f"❌ Market data not found for {symbol}.")
        return search_again_prompt(chat_id)

    hist = t.history(period=f"{days}d")
    if hist.empty:
        bot.send_message(chat_id, f"❌ No historical data for {symbol} ({used}).")
        return search_again_prompt(chat_id)

    if days == 7 and not only_chart:
        ccy = currency_symbol_for(used)
        header = f"📜 *{days}-Day Price History for {symbol}* (symbol: {used})\n"
        fmt = "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10}"
        lines = [fmt.format("Date","Open","High","Low","Close","Volume")]
        for idx,row in hist.iterrows():
            d = idx.strftime("%Y-%m-%d")
            o = f"{ccy}{row['Open']:.2f}"
            h = f"{ccy}{row['High']:.2f}"
            l = f"{ccy}{row['Low']:.2f}"
            c = f"{ccy}{row['Close']:.2f}"
            v = human_volume(row['Volume']) if 'Volume' in row else str(row.get('Volume',''))
            lines.append(fmt.format(d,o,h,l,c,v))
        msg = header + "```\n" + "\n".join(lines) + "\n```"
        bot.send_message(chat_id, msg, parse_mode="Markdown")

    buf = create_chart_bytes(hist.tail(30), f"{symbol} - Last {days} Days")
    bot.send_photo(chat_id, buf)
    return search_again_prompt(chat_id)

def bulletize_news_item(item):
    title = item.get("title", "").strip()
    publisher = item.get("publisher") or item.get("source") or ""
    ts = item.get("providerPublishTime")
    when = ""
    try:
        if ts:
            when = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except Exception:
        pass

    summary = item.get("summary", "") or item.get("content", "")
    bullets = []
    if summary:
        parts = re.split(r"(?<=[.!?])\s+", summary)
        for s in parts[:2]:
            s = s.strip()
            if s:
                bullets.append(f"- {s}")
    else:
        if title:
            bullets.append(f"- {title}")

    context = []
    if publisher: context.append(publisher)
    if when: context.append(when)
    ctx = " · ".join(context)
    return bullets, ctx

def send_ai_insight(chat_id, symbol):
    t, used = fetch_ticker(symbol)
    if t is None:
        bot.send_message(chat_id, f"❌ Market data not found for {symbol}.")
        return search_again_prompt(chat_id)

    hist = t.history(period="1mo")
    if hist.empty or len(hist) < 5:
        bot.send_message(chat_id, f"❌ Not enough data for insights on {symbol}.")
        return search_again_prompt(chat_id)

    close = hist["Close"]
    ma20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else close.mean()
    ma50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else close.mean()
    rsi_series = compute_rsi(close)
    rsi_latest = rsi_series.iloc[-1] if not rsi_series.empty else float('nan')
    last_price = close.iloc[-1]
    first_price = close.iloc[0]
    pct_1m = (last_price - first_price) / first_price * 100

    ccy = currency_symbol_for(used)

    suggestion = "HOLD"
    reasons = []
    if ma20 and ma50 and ma20 > ma50 and (not math.isnan(rsi_latest) and rsi_latest < 70):
        suggestion = "BUY"; reasons.append("MA20 > MA50")
    if ma20 and ma50 and ma20 < ma50:
        if suggestion != "BUY":
            suggestion = "SELL"; reasons.append("MA20 < MA50")
    if not math.isnan(rsi_latest):
        if rsi_latest > 70:
            suggestion = "SELL"; reasons.append("RSI > 70 (overbought)")
        elif rsi_latest < 30:
            suggestion = "BUY"; reasons.append("RSI < 30 (oversold)")

    lines = [
        f"🤖 *AI Insights for {symbol}*",
        f"• Last price: {ccy}{last_price:.2f}",
        f"• 1-month change: {pct_1m:.2f}%",
        f"• MA20: {ccy}{ma20:.2f}  |  MA50: {ccy}{ma50:.2f}",
        (f"• RSI: {rsi_latest:.2f}" if not math.isnan(rsi_latest) else "• RSI: N/A"),
        "",
        f"*Suggestion: {suggestion}*"
    ]
    if reasons:
        lines.append("• Reasons: " + "; ".join(reasons))

    news_text = ""
    try:
        news_items = t.news
        if news_items:
            news_text = "\n\n📰 *Latest news (highlights):*\n"
            for n in news_items[:3]:
                title = n.get("title","news")
                link = n.get("link")
                bullets, ctx = bulletize_news_item(n)
                if link:
                    news_text += f"• [{title}]({link})"
                else:
                    news_text += f"• {title}"
                if ctx:
                    news_text += f" _( {ctx} )_"
                news_text += "\n"
                for b in bullets[:2]:
                    news_text += f"   {b}\n"
    except Exception:
        news_text = "\n\n(No news available)"

    final_msg = "\n".join(lines) + (news_text if news_text else "")
    bot.send_message(chat_id, final_msg, parse_mode="Markdown", disable_web_page_preview=True)
    return search_again_prompt(chat_id)

# ---------------- NEW: Balance Sheet / Ratios / Peers / Pros & Cons ----------------
def send_balance_sheet(chat_id, symbol):
    t, used = fetch_ticker(symbol)
    if t is None:
        bot.send_message(chat_id, f"❌ Could not load data for {symbol}.")
        return search_again_prompt(chat_id)
    try:
        bs = t.balance_sheet
        if bs is None or bs.empty:
            bot.send_message(chat_id, f"📄 Balance Sheet not available for {used}.")
            return search_again_prompt(chat_id)
        # Show the latest column, top rows
        latest_col = bs.columns[0]
        rows_to_show = min(12, len(bs.index))
        subset = bs.iloc[:rows_to_show, [0]].copy()
        subset.columns = [latest_col.strftime("%Y-%m-%d") if hasattr(latest_col, "strftime") else str(latest_col)]
        # Format
        lines = [f"📄 *Balance Sheet (latest)* — {symbol} ({used})"]
        for idx, val in subset.iloc[:,0].items():
            lines.append(f"- {idx}: {human_number(val)}")
        bot.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
    except Exception:
        bot.send_message(chat_id, f"📄 Balance Sheet not available for {used}.")
    return search_again_prompt(chat_id)

def collect_ratios(t):
    """Pull a bunch of ratios with safe defaults."""
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    def g(key, default=None):
        v = info.get(key, default)
        return v

    ratios = {
        "Price": g("currentPrice"),
        "P/E (trailing)": g("trailingPE"),
        "P/E (forward)": g("forwardPE"),
        "P/B": g("priceToBook"),
        "ROE": g("returnOnEquity"),
        "ROA": g("returnOnAssets"),
        "Profit Margin": g("profitMargins"),
        "Operating Margin": g("operatingMargins"),
        "Debt/Equity": g("debtToEquity"),
        "Market Cap": g("marketCap"),
        "Beta": g("beta"),
        "Revenue Growth": g("revenueGrowth"),
        "EBITDA Margins": g("ebitdaMargins"),
        "Industry": g("industry"),
        "Sector": g("sector"),
        "Currency": g("currency")
    }
    return ratios

def send_ratios(chat_id, symbol):
    t, used = fetch_ticker(symbol)
    if t is None:
        bot.send_message(chat_id, f"❌ Could not load data for {symbol}.")
        return search_again_prompt(chat_id)
    ratios = collect_ratios(t)
    lines = [f"📊 *Key Ratios* — {symbol} ({used})"]
    def fmt_ratio(label, val, is_pct=False):
        if val is None or val == 0:
            return f"- {label}: —"
        if is_pct:
            return f"- {label}: {pct(val)}"
        if label in ("Market Cap","Price"):
            return f"- {label}: {human_number(val)}"
        return f"- {label}: {val:.2f}" if isinstance(val, (int, float)) else f"- {label}: {val}"

    lines.append(fmt_ratio("Price", ratios["Price"]))
    lines.append(fmt_ratio("P/E (trailing)", ratios["P/E (trailing)"]))
    lines.append(fmt_ratio("P/E (forward)", ratios["P/E (forward)"]))
    lines.append(fmt_ratio("P/B", ratios["P/B"]))
    lines.append(fmt_ratio("ROE", ratios["ROE"], is_pct=True))
    lines.append(fmt_ratio("ROA", ratios["ROA"], is_pct=True))
    lines.append(fmt_ratio("Profit Margin", ratios["Profit Margin"], is_pct=True))
    lines.append(fmt_ratio("Operating Margin", ratios["Operating Margin"], is_pct=True))
    lines.append(fmt_ratio("Revenue Growth", ratios["Revenue Growth"], is_pct=True))
    lines.append(fmt_ratio("EBITDA Margins", ratios["EBITDA Margins"], is_pct=True))
    lines.append(fmt_ratio("Debt/Equity", ratios["Debt/Equity"]))
    lines.append(fmt_ratio("Market Cap", ratios["Market Cap"]))
    if ratios["Sector"]: lines.append(f"- Sector: {ratios['Sector']}")
    if ratios["Industry"]: lines.append(f"- Industry: {ratios['Industry']}")
    if ratios["Currency"]: lines.append(f"- Currency: {ratios['Currency']}")
    bot.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
    return search_again_prompt(chat_id)

def candidate_universe_for(symbol):
    s = symbol.upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return INDIA_LARGE_UNIVERSE
    # default to US large caps for others
    return US_LARGE_UNIVERSE

def send_peer_comparison(chat_id, symbol):
    t, used = fetch_ticker(symbol)
    if t is None:
        bot.send_message(chat_id, f"❌ Could not load data for {symbol}.")
        return search_again_prompt(chat_id)

    # Base company info
    try:
        base_info = t.info or {}
    except Exception:
        base_info = {}
    base_sector = base_info.get("sector")
    base_industry = base_info.get("industry")

    # Build universe and filter by sector/industry
    peers = []
    universe = candidate_universe_for(used)
    # ensure the main symbol is included
    if used not in universe:
        universe = [used] + universe

    for tick in universe:
        try:
            ti = yf.Ticker(tick)
            info = ti.info or {}
            if (base_sector and info.get("sector")==base_sector) or (base_industry and info.get("industry")==base_industry) or (tick==used):
                peers.append({
                    "ticker": tick,
                    "price": info.get("currentPrice"),
                    "pe": info.get("trailingPE"),
                    "pb": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "margin": info.get("profitMargins"),
                    "mcap": info.get("marketCap"),
                    "name": info.get("shortName") or info.get("longName") or tick
                })
        except Exception:
            continue

    # Deduplicate and keep top 6 by market cap
    uniq = {}
    for p in peers:
        uniq[p["ticker"]] = p
    peers = list(uniq.values())
    peers = sorted(peers, key=lambda x: (x["ticker"]!=used, -(x["mcap"] or 0)))[:6]

    if len(peers) <= 1:
        bot.send_message(chat_id, "🏢 Peer data not sufficient to compare for this company.")
        return search_again_prompt(chat_id)

    # Render as a neat monospace table
    header = f"🏢 *Peer Comparison* — base: {symbol} ({used})"
    fmt = "{:<12} {:>10} {:>8} {:>8} {:>8} {:>9} {:>10}"
    lines = [header, "```\n"+fmt.format("Ticker","Price","P/E","P/B","ROE%","Margin%","MktCap")]
    for p in peers:
        lines.append(fmt.format(
            p["ticker"][:12],
            human_number(p["price"]) if p["price"] else "—",
            f"{p['pe']:.2f}" if isinstance(p["pe"],(int,float)) else "—",
            f"{p['pb']:.2f}" if isinstance(p["pb"],(int,float)) else "—",
            f"{p['roe']*100:.1f}" if isinstance(p["roe"],(int,float)) else "—",
            f"{p['margin']*100:.1f}" if isinstance(p["margin"],(int,float)) else "—",
            human_number(p["mcap"]) if p["mcap"] else "—"
        ))
    lines.append("```")
    bot.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
    return search_again_prompt(chat_id)

def send_pros_cons(chat_id, symbol):
    t, used = fetch_ticker(symbol)
    if t is None:
        bot.send_message(chat_id, f"❌ Could not load data for {symbol}.")
        return search_again_prompt(chat_id)

    ratios = collect_ratios(t)
    pros, cons = [], []

    # ------- Heuristics (tweak thresholds as you wish) -------
    rg = ratios.get("Revenue Growth")
    pm = ratios.get("Profit Margin")
    om = ratios.get("Operating Margin")
    roe = ratios.get("ROE")
    de = ratios.get("Debt/Equity")
    pe_f = ratios.get("P/E (forward)")
    pe_t = ratios.get("P/E (trailing)")
    pb = ratios.get("P/B")
    mcap = ratios.get("Market Cap")
    beta = ratios.get("Beta")

    if rg is not None and rg > 0: pros.append("Positive revenue growth")
    if pm is not None and pm > 0.10: pros.append("Healthy profit margins")
    if om is not None and om > 0.15: pros.append("Strong operating efficiency")
    if roe is not None and roe > 0.15: pros.append("High return on equity")
    if mcap is not None and mcap > 1e11: pros.append("Large market cap (relative stability)")
    if de is not None and de < 1: pros.append("Reasonable debt-to-equity")

    if rg is not None and rg < 0: cons.append("Declining revenues")
    if roe is not None and roe < 0.05: cons.append("Weak ROE")
    if pm is not None and pm < 0.05: cons.append("Thin profit margins")
    if de is not None and de > 2: cons.append("High leverage (Debt/Equity)")
    # Valuation flags: very high P/E or P/B relative flags
    if (pe_f and isinstance(pe_f,(int,float)) and pe_f > 35) or (pe_t and isinstance(pe_t,(int,float)) and pe_t > 35):
        cons.append("Rich valuation (high P/E)")
    if pb and isinstance(pb,(int,float)) and pb > 6:
        cons.append("High price-to-book")
    if beta and isinstance(beta,(int,float)) and beta > 1.3:
        cons.append("More volatile than market (high beta)")

    if not pros: pros = ["No obvious strengths from public ratios"]
    if not cons: cons = ["No major weaknesses flagged from ratios"]

    lines = [f"✅ *Pros & Cons* — {symbol} ({used})", "*Pros:*"]
    lines += [f"- {p}" for p in pros]
    lines += ["", "*Cons:*"]
    lines += [f"- {c}" for c in cons]
    bot.send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
    return search_again_prompt(chat_id)

# ✅ After each stocks action, keep user ready to query again
def search_again_prompt(chat_id):
    bot.send_message(chat_id, "What next? Type a name or tap an option 👇🏻.", reply_markup=main_menu_keyboard())
    st = chat_state.get(chat_id, {})
    if st.get("mode") != "invest":
        chat_state[chat_id] = {"stage":"await_query"}

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("Bot starting...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


# In[ ]:





# In[ ]:





# In[ ]:




