import os
import pandas as pd, streamlit as st
from crude_engine import load, backtest, summarize, option_plan, STOP_ATR, TARGET_R, MAX_HOLD
from scanner import monte_carlo

st.set_page_config(page_title="Crude Oil Signals", layout="wide")
st.title("Crude oil (WTI) trend-breakout signals")
st.caption("Stats are for the WTI futures price (Yahoo CL=F), not MCX option prices. "
           "Suggest-only. Past results describe history, not the future. Not financial advice.")


@st.cache_data(ttl=6 * 3600)
def get():
    raw, fx = load(demo=bool(os.environ.get("DEMO")))
    trades, live, lv = backtest(raw)
    return raw.Close.tail(150), fx, trades, live, lv


closes, fx, trades, live, lv = get()
st.line_chart(closes)
a, b, c, d = st.columns(4)
a.metric("WTI close", f"${lv['close']:.2f}")
b.metric("20-day high / low", f"${lv['hi']:.2f} / ${lv['lo']:.2f}")
c.metric("100-day average", f"${lv['ma']:.2f}")
d.metric("ATR (daily range)", f"${lv['atr']:.2f}")

if live is None:
    st.info(f"No signal on the latest candle. A close above ${lv['hi']:.2f} (with price above the 100-day average) "
            f"would trigger a long; a close below ${lv['lo']:.2f} (with price below it) would trigger a short.")
    st.stop()

side = "long" if live["side"] == 1 else "short"
opt = "CALL (buy)" if side == "long" else "PUT (buy)"
st.subheader(f"Signal: {side.upper()} in the underlying → buy a {opt}")
x1, x2, x3, x4 = st.columns(4)
x1.metric("Entry (next open)", f"${live['entry']:.2f}", f"≈ ₹{live['entry'] * fx:,.0f}/bbl")
x2.metric(f"Exit: stop ({STOP_ATR:g}×ATR)", f"${live['stop']:.2f}", f"≈ ₹{live['stop'] * fx:,.0f}/bbl")
x3.metric(f"Exit: target ({TARGET_R:g}R)", f"${live['target']:.2f}", f"≈ ₹{live['target'] * fx:,.0f}/bbl")
x4.metric("Exit: time stop", f"{MAX_HOLD} trading days")

same = trades[trades.side == side]
sm = summarize(same)
if sm:
    st.markdown(f"**Past {side} signals on this rule**")
    st.dataframe(pd.DataFrame({side: sm}).T.round(2), use_container_width=True)
    if sm["trades"] < 30:
        st.warning(f"Only {sm['trades']} past trades: weak evidence.")
    mc = monte_carlo(same.r.values)
    if mc:
        st.write(f"**10,000 replays of 40 such trades (in R):** final 5th/median/95th = "
                 f"{mc['final_p5']:.1f} / {mc['final_median']:.1f} / {mc['final_p95']:.1f} · "
                 f"max drawdown median {mc['dd_median']:.1f}R, 95th pct {mc['dd_p95']:.1f}R")

st.markdown("### Your MCX option plan (buying a " + ("call" if side == "long" else "put") + ")")
st.caption("Enter live numbers from your broker's option chain. Prefer an at-the-money or slightly in-the-money strike (delta about 0.5 to 0.65) with several weeks to expiry.")
r1, r2, r3 = st.columns(3)
mcx = r1.number_input("MCX crude futures price (₹/bbl)", value=float(round(live["entry"] * fx)), step=10.0)
prem = r2.number_input("Option premium (₹/bbl)", value=150.0, step=5.0)
delta = r3.number_input("Delta (from option chain)", value=0.5, step=0.05)
r4, r5, r6 = st.columns(3)
theta = r4.number_input("Time decay per day (₹/bbl, theta)", value=3.0, step=0.5)
days = r5.number_input("Expected days held", value=7, step=1)
lot = r6.number_input("Lot size (barrels), check MCX", value=100, step=10)
r7, r8 = st.columns(2)
cap = r7.number_input("Capital (₹)", value=500_000, step=50_000)
rp = r8.number_input("Max loss per trade (%)", value=1.0, step=0.25)

p = option_plan(side, live["entry"], live["stop"], live["target"], mcx, prem, delta, theta, days)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Buy near premium", f"₹{prem:,.0f}")
m2.metric("Exit if stop hit: premium ≈", f"₹{p['p_stop']:,.0f}", f"-₹{p['loss']:,.0f}", delta_color="inverse")
m3.metric("Exit at target: premium ≈", f"₹{p['p_tgt']:,.0f}", f"+₹{p['gain']:,.0f}")
m4.metric("Reward : risk on the option", f"{p['rr']:.1f} : 1" if p["rr"] == p["rr"] else "n/a")
if p["rr"] == p["rr"] and p["rr"] < 1.5:
    st.warning("Reward to risk on this option is below 1.5 after time decay. Consider skipping, or a different strike or expiry.")
by_stop = int((cap * rp / 100) // max(p["loss"] * lot, 1))
by_zero = int((cap * rp / 100) // max(prem * lot, 1))
st.write(f"**Lots within your loss limit:** {by_stop} if you exit at the stop level above · "
         f"{by_zero} if the option could go to zero (gap risk). Use the smaller number if unsure.")
st.caption("Premium levels are estimates from delta and a fixed daily decay. Real premiums also move with implied "
           "volatility and gamma, so they will differ. Not backtested on option prices (no free historical data). "
           "Exit when the crude price hits the stop or target, or at the time stop, and avoid holding through big "
           "scheduled events such as weekly US inventory data or OPEC meetings unless you accept the gap risk.")
