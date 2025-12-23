import math
import random
import numpy as np
from .env import SlotEnv


def discretize(
    bankroll, t, wins, losses, horizon,
    alpha0=8, beta0=2,
    m_bins=20, n_bins=20, b_bins=10, t_bins=10,
    bankroll_scale=200.0,
):
    """
    Builds a compact state out of:
    - posterior mean of p (m)
    - evidence amount (n)
    - bankroll bucket
    - time bucket
    """
    alpha = alpha0 + wins
    beta = beta0 + losses

    m = alpha / (alpha + beta)  # posterior mean
    n = alpha + beta            # evidence strength

    m_idx = min(m_bins - 1, max(0, int(m * m_bins)))

    # squash evidence to [0,1] using log
    n_max = alpha0 + beta0 + horizon
    n_squash = math.log(1 + n) / math.log(1 + n_max)
    n_idx = min(n_bins - 1, max(0, int(n_squash * n_bins)))

    b_norm = bankroll / bankroll_scale
    b_idx = min(b_bins - 1, max(0, int(b_norm * b_bins)))

    t_norm = t / float(horizon)
    t_idx = min(t_bins - 1, max(0, int(t_norm * t_bins)))

    return (m_idx, n_idx, b_idx, t_idx)


def train_q_learning(
    episodes=8000,
    p_true=0.09,
    horizon=200,
    bankroll0=100,
    gamma=0.99,
    lr=0.12,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.9992,
    alpha0=8,
    beta0=2,
    seed=0,
):
    rng = random.Random(seed)
    def p_sampler(rng):
        # 80% bad (negative EV), 20% good (positive EV)
        return 0.09 if rng.random() < 0.80 else 0.12

    env = SlotEnv(p_sampler=p_sampler, horizon=horizon, bankroll0=bankroll0, seed=seed)

    m_bins, n_bins, b_bins, t_bins = 20, 20, 10, 10
    Q = np.zeros((m_bins, n_bins, b_bins, t_bins, 2), dtype=np.float32)

    bankroll_scale = max(200.0, bankroll0 * 2.0)  # conservative scaling

    eps = eps_start
    ep_returns = np.zeros(episodes, dtype=np.float32)
    ep_spin_rate = np.zeros(episodes, dtype=np.float32)
    ep_spins = np.zeros(episodes, dtype=np.int32)

    for ep in range(episodes):
        bankroll, t, wins, losses = env.reset()
        done = False
        G = 0.0
        spins = 0
        steps = 0

        while not done:
            s = discretize(bankroll, t, wins, losses, horizon,
                           alpha0=alpha0, beta0=beta0,
                           bankroll_scale=bankroll_scale)

            if rng.random() < eps:
                a = rng.choice([0, 1])
            else:
                a = int(np.argmax(Q[s]))

            (bankroll2, t2, wins2, losses2), r, done = env.step(a)

            s2 = discretize(bankroll2, t2, wins2, losses2, horizon,
                            alpha0=alpha0, beta0=beta0,
                            bankroll_scale=bankroll_scale)

            td_target = r + (0.0 if done else gamma * float(np.max(Q[s2])))
            Q[s + (a,)] += lr * (td_target - Q[s + (a,)])

            bankroll, t, wins, losses = bankroll2, t2, wins2, losses2

            G += r
            steps += 1
            if a == 0:
                spins += 1

        ep_returns[ep] = G
        ep_spins[ep] = spins
        ep_spin_rate[ep] = spins / max(1, steps)

        eps = max(eps_end, eps * eps_decay)

        if (ep + 1) % 500 == 0:
            print(f"[train] ep={ep+1}/{episodes}  eps={eps:.3f}  avg_return(last500)={ep_returns[max(0,ep-499):ep+1].mean():.2f}")

    return Q, ep_returns, ep_spin_rate, ep_spins


def evaluate_policy(Q, episodes=500, p_true=0.09, horizon=200, bankroll0=100, alpha0=8, beta0=2, seed=123):
    env = SlotEnv(p_true=p_true, horizon=horizon, bankroll0=bankroll0, seed=seed)
    bankroll_scale = max(200.0, bankroll0 * 2.0)

    returns = np.zeros(episodes, dtype=np.float32)
    spins_until_quit = np.zeros(episodes, dtype=np.int32)
    ruined = np.zeros(episodes, dtype=np.int32)

    for i in range(episodes):
        bankroll, t, wins, losses = env.reset()
        done = False
        G = 0.0
        spins = 0

        while not done:
            s = discretize(bankroll, t, wins, losses, horizon,
                           alpha0=alpha0, beta0=beta0,
                           bankroll_scale=bankroll_scale)
            a = int(np.argmax(Q[s]))
            (bankroll, t, wins, losses), r, done = env.step(a)
            G += r
            if a == 0:
                spins += 1

        returns[i] = G
        spins_until_quit[i] = spins
        ruined[i] = 1 if bankroll <= 0 else 0

    return returns, spins_until_quit, float(ruined.mean())
