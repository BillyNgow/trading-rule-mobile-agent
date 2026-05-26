
import os, time, requests
import pandas as pd
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PUBLIC_REPORT_URL = os.getenv("PUBLIC_REPORT_URL", "")

if not API_KEY:
    raise ValueError("Missing ALPHA_VANTAGE_API_KEY.")

TICKERS_FILE = "tickers.txt"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NASDAQ_TICKERS = {"AMZN","MSFT","GOOGL","META","NVDA","AVGO","NFLX","COST","LRCX","AMD","CSCO","AAPL"}

def fetch_daily(symbol):
    r = requests.get("https://www.alphavantage.co/query", params={
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY,
    }, timeout=30)
    data = r.json()
    if "Time Series (Daily)" not in data:
        print(f"Error for {symbol}: {data}")
        return None
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        "1. open": "Open", "2. high": "High", "3. low": "Low",
        "4. close": "Close", "5. volume": "Volume"
    })
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_indicators(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["AvgVolume20"] = df["Volume"].rolling(20).mean()
    df["High20"] = df["High"].rolling(20).max()
    df["Low20"] = df["Low"].rolling(20).min()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI14"] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    return df

def safe_round(v, digits=2):
    try:
        if pd.isna(v): return ""
        return round(float(v), digits)
    except Exception:
        return ""

def score_setup(row):
    close, sma20, sma50 = row["Close"], row["SMA20"], row["SMA50"]
    volume, avgvol = row["Volume"], row["AvgVolume20"]
    high20, low20, rsi = row["High20"], row["Low20"], row["RSI14"]
    score, notes = 0, []

    if pd.notna(close) and pd.notna(sma20) and pd.notna(sma50) and close > sma20 > sma50:
        score += 20; trend = "Clean"
    elif pd.notna(close) and pd.notna(sma50) and close > sma50:
        score += 10; trend = "Mixed"; notes.append("Trend not fully clean")
    else:
        trend = "Weak"; notes.append("Trend not clean")

    sr = "Weak"
    if pd.notna(high20) and pd.notna(low20) and high20 > low20:
        pos = (close - low20) / (high20 - low20)
        if 0.25 <= pos <= 0.75:
            score += 20; sr = "Clear"
        else:
            score += 10; sr = "Check"; notes.append("Near 20-day range edge")

    volratio = volume / avgvol if pd.notna(avgvol) and avgvol > 0 else 0
    if volratio >= 1.2:
        score += 15; vol = "Confirm"
    elif volratio >= 0.8:
        score += 8; vol = "Neutral"; notes.append("Volume not strong")
    else:
        vol = "Weak"; notes.append("Weak volume")

    dist = ((close - sma20) / sma20) * 100 if pd.notna(sma20) and sma20 else 999
    if dist <= 5 and pd.notna(rsi) and rsi <= 65:
        score += 20; chasing = "Low"
    elif dist <= 8 and pd.notna(rsi) and rsi <= 70:
        score += 10; chasing = "Borderline"; notes.append("Slightly extended")
    else:
        chasing = "High"; notes.append("Chasing risk")

    risk = close - low20 if pd.notna(low20) else 0
    reward = high20 - close if pd.notna(high20) else 0
    rr = reward / risk if risk > 0 else 0
    if rr >= 2:
        score += 15; rr_state = "Pass"
    elif rr >= 1.5:
        score += 8; rr_state = "Close"; notes.append("R:R not fully 1:2")
    else:
        rr_state = "Fail"; notes.append("Poor estimated R:R")

    score += 5
    news = "Manual"

    if score >= 85: grade, status = "A", "Review First"
    elif score >= 70: grade, status = "B", "Review"
    elif score >= 50: grade, status = "C", "Reject"
    else: grade, status = "D", "Reject"

    return {
        "Score": score, "FinalGrade": grade, "Status": status,
        "CleanTrend": trend, "ClearSR": sr, "VolumeConfirmation": vol,
        "NotChasing": chasing, "RR_1_to_2_Feasible": rr_state,
        "NewsRisk": news, "VolumeRatio": round(volratio, 2),
        "DistanceFromSMA20Pct": round(dist, 2), "RR_Estimate": round(rr, 2),
        "Notes": "; ".join(notes)
    }

def tradingview_url(ticker):
    exchange = "NASDAQ" if ticker in NASDAQ_TICKERS else "NYSE"
    safe = ticker.replace(".", "-")
    return f"https://www.tradingview.com/chart/?symbol={exchange}%3A{safe}"

def icon(value, good, warn=None):
    warn = warn or []
    if value in good: return "✅"
    if value in warn: return "⚠️"
    return "❌"

def make_desktop_html(results, path):
    generated = datetime.now().strftime("%d %b %Y, %I:%M %p MYT")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Daily Rule Quality Report</title>
<style>body{{font-family:Arial;padding:24px;background:#f7f7f7}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #ddd;padding:8px;font-size:13px}}th{{background:#222;color:white}}</style>
</head><body><h1>Daily Rule Quality Report</h1><p>Generated: {generated}</p><p>Setup quality only. Not buy/sell signals.</p>{results.to_html(index=False)}</body></html>"""
    open(path, "w", encoding="utf-8").write(html)

def make_mobile_html(results, path):
    generated = datetime.now().strftime("%d %b %Y, %I:%M %p MYT")
    review = results[results["FinalGrade"].isin(["A", "B"])].copy()
    rejected = results[~results["FinalGrade"].isin(["A", "B"])].copy()

    def card(row):
        ticker = row["Ticker"]; grade = row["FinalGrade"]
        verdict = "Manual review only."
        if row["RR_1_to_2_Feasible"] != "Pass": verdict = "Watch only. R:R not fully 1:2."
        if grade in ["C","D"]: verdict = "Reject under current rules."
        return f"""
<section class="card grade-{grade}">
  <div class="top"><div><div class="ticker">{ticker}</div><div class="sub">Close {row['Close']} · {row['Date']}</div></div><div class="grade">{grade}</div><div class="score">{row['Score']}<small>Score</small></div></div>
  <div class="rules">
    <div><span>Trend</span><b>{icon(row['CleanTrend'], ['Clean'], ['Mixed'])} {row['CleanTrend']}</b></div>
    <div><span>S/R</span><b>{icon(row['ClearSR'], ['Clear'], ['Check'])} {row['ClearSR']}</b></div>
    <div><span>Volume</span><b>{icon(row['VolumeConfirmation'], ['Confirm'], ['Neutral'])} {row['VolumeConfirmation']} ({row['VolumeRatio']}x)</b></div>
    <div><span>Chasing</span><b>{icon(row['NotChasing'], ['Low'], ['Borderline'])} {row['NotChasing']}</b></div>
    <div><span>R:R</span><b>{icon(row['RR_1_to_2_Feasible'], ['Pass'], ['Close'])} {row['RR_Estimate']}</b></div>
    <div><span>News</span><b>⚠️ {row['NewsRisk']}</b></div>
  </div>
  <div class="verdict">{verdict}</div>
  <a class="btn" href="{tradingview_url(ticker)}">Open TradingView Chart</a>
</section>"""

    review_cards = "".join(card(r) for _, r in review.iterrows()) or '<div class="empty">No A/B candidates today.</div>'
    rejected_cards = "".join(card(r) for _, r in rejected.head(10).iterrows())
    top5 = "".join([f"<div class='rank'><b>{i+1}. {r['Ticker']}</b><span>{r['FinalGrade']} · {r['Score']} · {r['Status']}</span></div>" for i,(_,r) in enumerate(results.head(5).iterrows())])

    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><title>Daily Rule Quality Report</title>
<style>
:root{{--bg:#05070a;--panel:#10151c;--line:#273241;--text:#f4f7fb;--muted:#9aa4b2;--green:#39d353;--yellow:#f5d90a;--red:#ff5b6b;--blue:#58a6ff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#172033 0,#05070a 42%);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;padding:14px 14px 92px}}
.header{{position:sticky;top:0;z-index:9;margin:-14px -14px 14px;padding:18px 14px 12px;background:rgba(5,7,10,.92);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}h1{{font-size:20px;margin:0 0 6px;font-weight:800}}.meta{{font-size:13px;color:var(--muted)}}.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}}.metric{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:10px;text-align:center}}.metric b{{display:block;font-size:20px}}.metric span{{font-size:11px;color:var(--muted)}}
.title{{font-size:15px;font-weight:800;margin:18px 0 10px}}.card{{background:linear-gradient(180deg,#141b24,#0a0e14);border:1px solid var(--line);border-radius:20px;padding:14px;margin-bottom:12px;box-shadow:0 10px 28px rgba(0,0,0,.25)}}.card.grade-A{{border-color:rgba(57,211,83,.6)}}.card.grade-B{{border-color:rgba(245,217,10,.55)}}.card.grade-C,.card.grade-D{{border-color:rgba(255,91,107,.45)}}
.top{{display:grid;grid-template-columns:1fr auto auto;gap:10px;align-items:center}}.ticker{{font-size:24px;font-weight:900}}.sub{{font-size:12px;color:var(--muted);margin-top:2px}}.grade{{min-width:34px;text-align:center;border-radius:9px;padding:6px 8px;font-weight:900;color:#080a0d}}.grade-A .grade{{background:var(--green)}}.grade-B .grade{{background:var(--yellow)}}.grade-C .grade,.grade-D .grade{{background:var(--red);color:white}}.score{{font-size:22px;font-weight:900;text-align:right}}.score small{{display:block;font-size:11px;color:var(--muted)}}.rules{{margin-top:12px;background:rgba(255,255,255,.03);border-radius:14px;padding:10px}}.rules div{{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.06)}}.rules div:last-child{{border-bottom:0}}.rules span{{color:var(--muted);font-size:13px}}.rules b{{font-size:13px;text-align:right}}.verdict{{margin-top:12px;padding:10px;border-radius:12px;background:rgba(245,217,10,.08);border:1px solid rgba(245,217,10,.25);font-size:13px;color:#ffe66b}}.btn{{display:block;text-align:center;margin-top:10px;padding:11px;border-radius:12px;text-decoration:none;color:#dcecff;border:1px solid rgba(88,166,255,.55);background:rgba(88,166,255,.08);font-weight:700}}.rank{{display:flex;justify-content:space-between;gap:12px;padding:12px;background:var(--panel);border:1px solid var(--line);border-radius:14px;margin-bottom:8px}}.rank span{{color:var(--muted)}}.note{{margin-top:14px;background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.25);border-radius:16px;padding:12px;color:#dcecff;font-size:13px;line-height:1.45}}.empty{{padding:18px;background:var(--panel);border:1px dashed var(--line);border-radius:16px;color:var(--muted);text-align:center}}.nav{{position:fixed;bottom:0;left:0;right:0;background:rgba(5,7,10,.95);backdrop-filter:blur(14px);border-top:1px solid var(--line);display:flex;justify-content:space-around;padding:9px 8px 22px}}.nav a{{color:var(--muted);text-decoration:none;font-size:11px;text-align:center}}.nav b{{display:block;font-size:18px;margin-bottom:2px;color:var(--green)}}
</style></head><body>
<header class="header"><h1>Daily Rule Quality Report</h1><div class="meta">Generated: {generated} · Setup quality only</div><div class="summary"><div class="metric"><b>{len(results)}</b><span>Scanned</span></div><div class="metric"><b>{len(review)}</b><span>A/B</span></div><div class="metric"><b>{int((results['FinalGrade']=='A').sum())}</b><span>A</span></div><div class="metric"><b>{int((results['FinalGrade']=='B').sum())}</b><span>B</span></div></div></header>
<div id="ab" class="title">A/B Candidates</div>{review_cards}
<div id="top" class="title">Top 5 Overall</div>{top5}
<div id="rejected" class="title">Rejected / Low Quality</div>{rejected_cards}
<div id="notes" class="note"><b>Important:</b><br>This is not a buy/sell signal. Manual chart review, news check, and real R:R confirmation are required.</div>
<nav class="nav"><a href="#ab"><b>◆</b>A/B</a><a href="#top"><b>★</b>Top</a><a href="#rejected"><b>⊘</b>Reject</a><a href="#notes"><b>!</b>Notes</a></nav>
</body></html>"""
    open(path, "w", encoding="utf-8").write(html)

def send_telegram_text(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured.")
        return
    r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=30)
    if r.status_code != 200: print("Telegram message failed:", r.text)

def send_telegram_document(path, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    with open(path, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"document": f}, timeout=60)
    if r.status_code != 200: print("Telegram file failed:", r.text)

def telegram_summary(results):
    generated = datetime.now().strftime("%d %b %Y, %I:%M %p MYT")
    review = results[results["FinalGrade"].isin(["A","B"])]
    lines = [f"🟡 {r['Ticker']} | {r['FinalGrade']} | Score {r['Score']} | R:R {r['RR_Estimate']}" for _, r in review.iterrows()]
    candidates = "\n".join(lines) if lines else "No A/B candidates today."
    top = "\n".join([f"{i+1}. {r['Ticker']} | {r['FinalGrade']} | {r['Score']}" for i,(_,r) in enumerate(results.head(5).iterrows())])
    link = f"\n\nOpen mobile dashboard:\n{PUBLIC_REPORT_URL}" if PUBLIC_REPORT_URL else ""
    return f"<b>Daily Rule Quality Report</b>\n{generated}\n\n<b>A/B candidates:</b>\n{candidates}\n\n<b>Top 5:</b>\n{top}{link}\n\nNot buy/sell signals. Manual review required."

def main():
    tickers = [x.strip() for x in open(TICKERS_FILE, encoding="utf-8") if x.strip()]
    rows = []
    for symbol in tickers:
        print("Processing", symbol)
        df = fetch_daily(symbol)
        if df is None or len(df) < 60: continue
        df = add_indicators(df)
        latest = df.iloc[-1]
        rows.append({
            "Ticker": symbol, "Date": df.index[-1].date(), "Close": safe_round(latest["Close"]),
            "Volume": int(latest["Volume"]) if pd.notna(latest["Volume"]) else "",
            "SMA20": safe_round(latest["SMA20"]), "SMA50": safe_round(latest["SMA50"]),
            "SMA200": safe_round(latest["SMA200"]), "RSI14": safe_round(latest["RSI14"]),
            "ATR14": safe_round(latest["ATR14"]), **score_setup(latest)
        })
        time.sleep(15)
    results = pd.DataFrame(rows)
    if results.empty:
        send_telegram_text("Daily Rule Quality Report: no results generated.")
        return
    order = {"A":1,"B":2,"C":3,"D":4}
    results["GradeOrder"] = results["FinalGrade"].map(order)
    results = results.sort_values(["GradeOrder","Score"], ascending=[True,False]).drop(columns=["GradeOrder"])
    csv_path = os.path.join(OUTPUT_DIR, "daily_rule_report.csv")
    desktop_path = os.path.join(OUTPUT_DIR, "daily_rule_report.html")
    mobile_path = os.path.join(OUTPUT_DIR, "index.html")
    results.to_csv(csv_path, index=False)
    make_desktop_html(results, desktop_path)
    make_mobile_html(results, mobile_path)
    send_telegram_text(telegram_summary(results))
    send_telegram_document(csv_path, "CSV report")
    send_telegram_document(mobile_path, "Mobile HTML report")
    print("Saved reports to output/")

if __name__ == "__main__":
    main()
