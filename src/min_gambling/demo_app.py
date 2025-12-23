import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import random
import time
import numpy as np
import streamlit as st

from min_gambling.env import SlotEnv
from min_gambling.qlearn import discretize

st.set_page_config(page_title="RL Slot: Learn to Quit", layout="centered")
st.title("RL Slot Machine Demo (Negative EV): Learns to Quit")

st.write(
    "This demo shows an agent that initially spins (uncertainty/exploration) and later learns to quit "
    "when the evidence suggests the machine is negative expected value."
)

# Load Q-table
try:
    Q = np.load("q_table.npy")
except FileNotFoundError:
    st.error("q_table.npy not found. Run training first: `python -m min_gambling.train` from repo root.")
    st.stop()

p_true = st.sidebar.slider("True win probability p (negative EV if < 0.10)", 0.01, 0.20, 0.09, 0.01)
horizon = st.sidebar.slider("Horizon", 50, 500, 200, 50)
bankroll0 = st.sidebar.slider("Initial bankroll", 10, 300, 100, 10)
speed = st.sidebar.slider("Step delay (sec)", 0.0, 1.0, 0.15, 0.05)

alpha0 = 8
beta0 = 2

if "env" not in st.session_state or st.sidebar.button("Reset episode"):
    st.session_state.env = SlotEnv(
        p_true=p_true, horizon=horizon, bankroll0=bankroll0, seed=random.randint(0, 1_000_000)
    )
    st.session_state.obs = st.session_state.env.reset()
    st.session_state.log = []

env = st.session_state.env
bankroll, t, wins, losses = st.session_state.obs

alpha = alpha0 + wins
beta = beta0 + losses
p_hat = alpha / (alpha + beta)

st.metric("Bankroll", f"{bankroll:.1f}")
st.write(f"t: **{t}/{horizon}** | wins: **{wins}** | losses: **{losses}** | posterior mean p̂: **{p_hat:.3f}**")

def agent_action(obs):
    bankroll, t, wins, losses = obs
    # keep bankroll_scale consistent with training (conservative)
    bankroll_scale = max(200.0, bankroll0 * 2.0)
    s = discretize(bankroll, t, wins, losses, horizon,
                   alpha0=alpha0, beta0=beta0,
                   bankroll_scale=bankroll_scale)
    return int(np.argmax(Q[s]))

def do_step():
    a = agent_action(st.session_state.obs)
    obs2, r, done = env.step(a)
    st.session_state.log.append((t, a, r, obs2[0], obs2[2], obs2[3]))
    st.session_state.obs = obs2
    return a, r, done

c1, c2 = st.columns(2)
step_btn = c1.button("Step once")
auto_btn = c2.button("Auto-run to end")

if step_btn:
    a, r, done = do_step()
    st.write(f"Action: **{'SPIN' if a==0 else 'QUIT'}** | Reward: **{r:.1f}**")
    if done:
        st.success("Episode ended.")

if auto_btn:
    done = False
    while not done:
        a, r, done = do_step()
        time.sleep(speed)
    st.success("Episode ended.")

if st.session_state.log:
    st.subheader("Recent steps (last 25)")
    for row in st.session_state.log[-25:]:
        tt, a, r, b, w, l = row
        st.write(f"t={tt:3d} | {('SPIN' if a==0 else 'QUIT'):4s} | r={r:5.1f} | bankroll={b:6.1f} | w={w} l={l}")
