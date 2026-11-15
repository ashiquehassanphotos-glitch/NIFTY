"""Dashboard.  Run:  streamlit run app.py"""
import os, sqlite3, time
import pandas as pd, streamlit as st
from scanner import DB, run_scan, similar, summarize, monte_carlo

st.set_page_config(page_title="Nifty Breakout Scanner", layout="wide")
st.title("Nifty 500 breakout signals")

stale = (not os.path.exists(DB)) or time.time() - os.path.getmtime(DB) > 20 * 3600
if st.sidebar.button("Refresh data now") or stale:
    with st.spinner("Downloading prices and scanning Nifty 500 (takes a few minutes)..."):
        run_scan(demo=bool(os.environ.get("DEMO")))
    st.rerun() if not stale else None

con = sqlite3.connect(DB)
sigs = pd.read_sql("select * from signals", con)
trades = pd.read_sql("select * from trades", con)

cap = st.sidebar.number_input("Capital (₹)", value=1_000_000, step=100_000)
risk_pct = st.sidebar.number_input("Risk per trade (%)", value=0.5, step=0.1)
st.sidebar.caption("Suggest-only. Past stats describe history, not the future. Not financial advice.")

if sigs.empty:
    st.info("No breakout signals on the latest candle. Run scanner.py after market close.")
    st.stop()

st.caption(f"{len(sigs)} signals on the latest candle · {len(trades)} historical breakouts in database")

for _, s in sigs.sort_values("volr", ascending=False).iterrows():
    qty = int((cap * risk_pct / 100) // max(s.entry - s.stop, 0.01))
    with st.expander(f"{s.symbol.replace('.NS', '')}  ·  close ₹{s.entry:,.1f}  ·  volume {s.volr:.1f}x", expanded=True):
        a, b, c, d = st.columns(4)
        a.metric("Entry (approx)", f"₹{s.entry:,.1f}")
        b.metric("Stop (2×ATR)", f"₹{s.stop:,.1f}")
        c.metric("Target (3R)", f"₹{s.target:,.1f}")
        d.metric("Qty for your risk", f"{qty}")
        st.write(f"Regime: {'Nifty above 200-DMA' if s.regime else 'Nifty below 200-DMA'} · "
                 f"ATR {s.atrp:.1f}% · {s.dist200:.0f}% above 200-DMA")

        rows = {}
        for label, sub in [("Similar setups (all stocks)", similar(trades, s)),
                           (f"Same stock", similar(trades, s, same_stock=True))]:
            sm = summarize(sub)
            if sm:
                rows[label] = sm
        if rows:
            st.dataframe(pd.DataFrame(rows).T.round(2), use_container_width=True)
            n = rows.get("Similar setups (all stocks)", {}).get("trades", 0)
            if n < 30:
                st.warning(f"Only {n} similar past trades: treat these stats as weak evidence.")
            mc = monte_carlo(similar(trades, s).r.values)
            if mc:
                st.write(f"**10,000 replays of 40 such trades (in R):** final result 5th/median/95th = "
                         f"{mc['final_p5']:.1f} / {mc['final_median']:.1f} / {mc['final_p95']:.1f} · "
                         f"max drawdown median {mc['dd_median']:.1f}R, 95th pct {mc['dd_p95']:.1f}R")
        else:
            st.warning("No similar past setups found.")
