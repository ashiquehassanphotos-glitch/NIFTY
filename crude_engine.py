"""Crude oil (WTI futures, Yahoo CL=F) trend-breakout signals + backtest. Long AND short."""
import numpy as np, pandas as pd

DON, TREND, STOP_ATR, TARGET_R, MAX_HOLD = 20, 100, 2.0, 3.0, 15
COST = 0.001  # round-trip cost/slippage as fraction of price (tune it)


def load(demo=False):
    if demo:
        from scanner import demo_prices
        return demo_prices()["DEMO0.NS"], 83.0
    import yfinance as yf
    d = yf.download("CL=F", period="10y", auto_adjust=True, progress=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    fx = yf.download("INR=X", period="10d", progress=False)["Close"].dropna()
    return d.dropna(), float(fx.iloc[-1].item() if hasattr(fx.iloc[-1], "item") else fx.iloc[-1])


def feats(d):
    d = d.dropna().copy()
    pc = d.Close.shift()
    tr = pd.concat([d.High - d.Low, (d.High - pc).abs(), (d.Low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["hi"] = d.High.rolling(DON).max().shift()
    d["lo"] = d.Low.rolling(DON).min().shift()
    d["ma"] = d.Close.rolling(TREND).mean()
    d["atrp"] = d.atr / d.Close * 100
    return d


def backtest(raw):
    d = feats(raw)
    O, H, L, C, A = (d[x].values for x in ("Open", "High", "Low", "Close", "atr"))
    n = len(d)
    longs = ((d.Close > d.hi) & (d.Close > d.ma)).values
    shorts = ((d.Close < d.lo) & (d.Close < d.ma)).values
    trades, live, busy = [], None, -1
    for i in np.flatnonzero(longs | shorts):
        if i <= busy:      # no overlapping trades: one position at a time
            continue
        side = 1 if longs[i] else -1
        if i == n - 1:     # today's live signal
            risk = STOP_ATR * A[i]
            live = dict(side=side, entry=C[i], stop=C[i] - side * risk,
                        target=C[i] + side * TARGET_R * risk, atrp=d.atrp.iat[i])
            continue
        e = O[i + 1]; risk = STOP_ATR * A[i]
        if not risk > 0:
            continue
        s, tgt = e - side * risk, e + side * TARGET_R * risk
        last = min(i + MAX_HOLD, n - 1)
        ex, reason, days = C[last], "time", last - i
        for j in range(i + 1, last + 1):
            if side == 1:
                if L[j] <= s: ex, reason, days = min(s, O[j]), "stop", j - i; break
                if H[j] >= tgt: ex, reason, days = max(tgt, O[j]), "target", j - i; break
            else:
                if H[j] >= s: ex, reason, days = max(s, O[j]), "stop", j - i; break
                if L[j] <= tgt: ex, reason, days = min(tgt, O[j]), "target", j - i; break
        if reason == "time" and i + MAX_HOLD > n - 1:
            continue
        busy = i + days
        trades.append(dict(date=d.index[i].strftime("%Y-%m-%d"), side="long" if side == 1 else "short",
                           atrp=d.atrp.iat[i], r=side * (ex - e) / risk - COST * e / risk,
                           days=days, reason=reason, failed3=int(reason == "stop" and days <= 3)))
    lastrow = d.iloc[-1]
    levels = dict(close=lastrow.Close, hi=lastrow.hi, lo=lastrow.lo, ma=lastrow.ma, atr=lastrow.atr)
    return pd.DataFrame(trades), live, levels


def summarize(t):
    if len(t) == 0:
        return None
    return dict(trades=len(t), win_rate=(t.r > 0).mean() * 100, avg_R=t.r.mean(),
                best_R=t.r.max(), worst_R=t.r.min(), avg_days=t.days.mean(),
                false_breakout_3d=t.failed3.mean() * 100)


def option_plan(side, entry, stop, target, mcx, prem, delta, theta, days):
    """Estimate a BOUGHT MCX option's premium when the underlying hits our stop/target.
    Delta approximation only: ignores gamma and implied-volatility changes."""
    d = 1 if side == "long" else -1            # long signal -> call, short signal -> put
    mv_stop = mcx * (stop - entry) / entry
    mv_tgt = mcx * (target - entry) / entry
    decay = theta * days
    p_stop = max(prem + abs(delta) * d * mv_stop - decay, 0.0)
    p_tgt = max(prem + abs(delta) * d * mv_tgt - decay, 0.0)
    loss, gain = prem - p_stop, p_tgt - prem
    return dict(p_stop=p_stop, p_tgt=p_tgt, loss=loss, gain=gain, rr=(gain / loss if loss > 0 else float("nan")))
