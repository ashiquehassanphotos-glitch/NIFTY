"""Nifty 500 breakout scanner + backtest database (free EOD data via yfinance).
Run:  python scanner.py          (real data)   |   python scanner.py --demo   (synthetic, offline test)
"""
import sqlite3, sys
import numpy as np, pandas as pd

DB = "breakouts.db"
LOOKBACK, VOL_MULT, STOP_ATR, TARGET_R, MAX_HOLD = 55, 1.5, 2.0, 3.0, 30
MIN_TURNOVER = 5e7   # skip stocks trading under Rs 5 crore/day (20-day avg)
RS_DAYS = 126        # relative strength window (~6 months)
REQUIRE_UPTREND = True  # only take signals when Nifty is above its 200-DMA
COST = 0.004  # round-trip brokerage + STT + slippage, as fraction of price (tune it)


# ---------- data ----------
def nifty500():
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
    except Exception:  # NSE sometimes blocks cloud servers: use a copy saved next to this file
        df = pd.read_csv("nifty500list.csv")
    return [s + ".NS" for s in df["Symbol"]]


def load_prices(tickers):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        raw = yf.download(chunk, period="10y", auto_adjust=True, group_by="ticker",
                          progress=False, threads=True)
        for t in chunk:
            try:
                d = raw[t].dropna(how="all")
                if len(d) > 300:
                    out[t] = d
            except KeyError:
                pass
        print(f"downloaded {min(i + 50, len(tickers))}/{len(tickers)}")
    return out


def demo_prices():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2016-01-01", "2026-11-13")
    n, out = len(idx), {}
    for k in range(40):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n)))
        o = c * (1 + rng.normal(0, 0.004, n))
        h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, n)))
        l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, n)))
        v = rng.integers(10**5, 10**6, n) * (1 + 2 * (rng.random(n) < 0.05))
        out[f"DEMO{k}.NS"] = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)
    return out


# ---------- signals + backtest ----------
def feats(d):
    d = d.dropna().copy()
    pc = d.Close.shift()
    tr = pd.concat([d.High - d.Low, (d.High - pc).abs(), (d.Low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["hh"] = d.High.rolling(LOOKBACK).max().shift()
    d["volr"] = d.Volume / d.Volume.rolling(20).mean().shift()
    d["dma200"] = d.Close.rolling(200).mean()
    d["atrp"] = d.atr / d.Close * 100
    d["dist200"] = (d.Close / d.dma200 - 1) * 100
    return d


def scan(prices, nifty):
    regime = (nifty.Close > nifty.Close.rolling(200).mean())
    nret = nifty.Close.pct_change(RS_DAYS)
    trades, sigs = [], []
    for sym, raw in prices.items():
        d = feats(raw)
        d["regime"] = regime.reindex(d.index).ffill().fillna(False).astype(int)
        d["rs"] = d.Close.pct_change(RS_DAYS) - nret.reindex(d.index).ffill()
        d["turn"] = (d.Close * d.Volume).rolling(20).mean()
        ok = (d.rs > 0) & (d.turn >= MIN_TURNOVER)
        if REQUIRE_UPTREND:
            ok &= d.regime == 1
        sig = ((d.Close > d.hh) & (d.volr >= VOL_MULT) & (d.Close > d.dma200) & ok).values
        O, H, L, C = (d[x].values for x in ("Open", "High", "Low", "Close"))
        n = len(d)
        for i in np.flatnonzero(sig):
            row = dict(symbol=sym, date=d.index[i].strftime("%Y-%m-%d"), volr=d.volr.iat[i],
                       atrp=d.atrp.iat[i], dist200=d.dist200.iat[i], rs=d.rs.iat[i] * 100, regime=int(d.regime.iat[i]))
            if i == n - 1:  # today's live signal: entry unknown until tomorrow's open
                e = C[i]; s = e - STOP_ATR * d.atr.iat[i]
                sigs.append({**row, "entry": e, "stop": s, "target": e + TARGET_R * (e - s)})
                continue
            e = O[i + 1]; s = e - STOP_ATR * d.atr.iat[i]; risk = e - s
            if risk <= 0:
                continue
            tgt = e + TARGET_R * risk
            last = min(i + MAX_HOLD, n - 1)
            ex, reason, days = C[last], "time", last - i
            for j in range(i + 1, last + 1):
                if L[j] <= s:                      # stop checked first (conservative)
                    ex, reason, days = min(s, O[j]), "stop", j - i; break
                if H[j] >= tgt:
                    ex, reason, days = max(tgt, O[j]), "target", j - i; break
            if reason == "time" and i + MAX_HOLD > n - 1:
                continue                           # trade still open, skip
            r = (ex - e) / risk - COST * e / risk
            trades.append({**row, "entry": e, "stop": s, "r": r, "days": days,
                           "reason": reason, "failed3": int(reason == "stop" and days <= 3)})
    return pd.DataFrame(trades), pd.DataFrame(sigs)


# ---------- history lookup (used by the dashboard) ----------
def similar(trades, sig, same_stock=False):
    t = trades[(trades.regime == sig.regime)
               & trades.volr.between(sig.volr * 0.7, sig.volr * 1.3)
               & trades.atrp.between(sig.atrp * 0.6, sig.atrp * 1.4)]
    return t[t.symbol == sig.symbol] if same_stock else t


def summarize(t):
    if len(t) == 0:
        return None
    return dict(trades=len(t), win_rate=(t.r > 0).mean() * 100, avg_R=t.r.mean(),
                best_R=t.r.max(), worst_R=t.r.min(), avg_days=t.days.mean(),
                false_breakout_3d=t.failed3.mean() * 100)


def monte_carlo(r, n=40, sims=10_000):
    """Bootstrap n trades 10,000 times -> range of outcomes and drawdowns (in R)."""
    r = np.asarray(r)
    if len(r) < 5:
        return None
    paths = np.cumsum(r[np.random.default_rng(0).integers(0, len(r), (sims, n))], axis=1)
    dd = (np.maximum.accumulate(np.maximum(paths, 0), axis=1) - paths).max(axis=1)
    p = lambda a, q: float(np.percentile(a, q))
    return dict(final_p5=p(paths[:, -1], 5), final_median=p(paths[:, -1], 50),
                final_p95=p(paths[:, -1], 95), dd_median=p(dd, 50), dd_p95=p(dd, 95))


def run_scan(demo=False):
    if demo:
        prices = demo_prices(); nifty = prices["DEMO0.NS"]
    else:
        import yfinance as yf
        prices = load_prices(nifty500())
        nifty = yf.download("^NSEI", period="10y", auto_adjust=True, progress=False)
        nifty.columns = [c[0] if isinstance(c, tuple) else c for c in nifty.columns]
    trades, sigs = scan(prices, nifty)
    cols = ["symbol", "date", "volr", "atrp", "dist200", "rs", "regime", "entry", "stop", "target"]
    if sigs.empty:  # quiet day: keep the table's columns so saving and the dashboard still work
        sigs = pd.DataFrame(columns=cols)
    with sqlite3.connect(DB) as con:
        trades.to_sql("trades", con, if_exists="replace", index=False)
        sigs.to_sql("signals", con, if_exists="replace", index=False)
    return len(trades), len(sigs)


if __name__ == "__main__":
    t, s = run_scan("--demo" in sys.argv)
    print(f"{t} historical breakouts stored, {s} live signals today")
