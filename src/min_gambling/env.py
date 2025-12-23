import random
from typing import Callable, Optional


class SlotEnv:
    """
    Hidden win probability p_true per episode.
    Reward per SPIN: +9 on win, -1 on loss.
    True EV = 10*p_true - 1 (negative if p_true < 0.10)
    Actions: 0=SPIN, 1=QUIT
    Observation: (bankroll, t, wins, losses)
    """
    def __init__(
        self,
        p_true: Optional[float] = 0.09,
        p_sampler: Optional[Callable[[random.Random], float]] = None,
        horizon: int = 200,
        bankroll0: float = 100,
        seed: int = 0,
    ):
        self.p_true = None if p_sampler is not None else float(p_true)
        self.p_sampler = p_sampler
        self.horizon = int(horizon)
        self.bankroll0 = float(bankroll0)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.bankroll = self.bankroll0
        self.done = False
        self.wins = 0
        self.losses = 0

        # Sample machine per episode if sampler provided
        if self.p_sampler is not None:
            self.p_true = float(self.p_sampler(self.rng))

        return self._obs()

    def _obs(self):
        return (self.bankroll, self.t, self.wins, self.losses)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode already terminated; call reset().")

        if action == 1:  # QUIT
            self.done = True
            return self._obs(), 0.0, True

        # SPIN
        self.t += 1
        win = (self.rng.random() < self.p_true)

        if win:
            r = 9.0
            self.wins += 1
        else:
            r = -1.0
            self.losses += 1

        self.bankroll += r

        if self.t >= self.horizon or self.bankroll <= 0:
            self.done = True

        return self._obs(), r, self.done
