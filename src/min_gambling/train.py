import os
import numpy as np
import matplotlib.pyplot as plt

from .qlearn import train_q_learning, evaluate_policy


def save_plots(ep_returns, ep_spin_rate, eval_returns, eval_spins, ruin_prob, outdir="results"):
    os.makedirs(outdir, exist_ok=True)

    plt.figure()
    plt.plot(ep_returns)
    plt.title("Training: Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig(os.path.join(outdir, "train_returns.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(ep_spin_rate)
    plt.title("Training: Fraction of Actions = SPIN")
    plt.xlabel("Episode")
    plt.ylabel("SPIN rate")
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(outdir, "train_spin_rate.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(eval_spins, bins=30)
    plt.title("Evaluation: Spins Until Quit")
    plt.xlabel("Spins")
    plt.ylabel("Count")
    plt.savefig(os.path.join(outdir, "eval_spins_hist.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(eval_returns, bins=30)
    plt.title("Evaluation: Episode Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.savefig(os.path.join(outdir, "eval_returns_hist.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.axis("off")
    plt.text(0.02, 0.75, f"Ruin probability (bankroll <= 0): {ruin_prob:.3f}", fontsize=14)
    plt.text(0.02, 0.55, f"Mean spins until quit: {float(np.mean(eval_spins)):.1f}", fontsize=14)
    plt.text(0.02, 0.35, f"Mean return: {float(np.mean(eval_returns)):.1f}", fontsize=14)
    plt.title("Evaluation Summary")
    plt.savefig(os.path.join(outdir, "eval_summary.png"), dpi=160, bbox_inches="tight")
    plt.close()


def main():
    # Core settings (MVP)
    p_true = 0.09      # negative EV (since < 0.10)
    horizon = 200
    bankroll0 = 100

    # Train
    Q, ep_returns, ep_spin_rate, _ = train_q_learning(
        episodes=8000,
        p_true=p_true,
        horizon=horizon,
        bankroll0=bankroll0,
        seed=0,
        alpha0=8, beta0=2,
    )

    np.save("q_table.npy", Q)

    # Evaluate
    eval_returns, eval_spins, ruin_prob = evaluate_policy(
        Q, episodes=500, p_true=p_true, horizon=horizon, bankroll0=bankroll0
    )

    # Save plots
    save_plots(ep_returns, ep_spin_rate, eval_returns, eval_spins, ruin_prob, outdir="results")

    print("Saved: q_table.npy and results/*.png")
    print(f"Ruin prob: {ruin_prob:.3f} | Mean spins: {eval_spins.mean():.1f} | Mean return: {eval_returns.mean():.1f}")


if __name__ == "__main__":
    main()
