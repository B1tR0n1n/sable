"""Seeded randomness utilities for reproducible simulations."""

import random as _random
import numpy as np
from typing import Any, Sequence, TypeVar

T = TypeVar("T")


class SeededRandom:
    """Reproducible random number generator wrapping Python's random and numpy."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = _random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    @property
    def seed(self) -> int:
        """Return the seed used to initialize this RNG."""
        return self._seed

    def random(self) -> float:
        """Return a random float in [0, 1)."""
        return self._rng.random()

    def uniform(self, a: float, b: float) -> float:
        """Return a random float in [a, b]."""
        return self._rng.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        """Return a random integer in [a, b]."""
        return self._rng.randint(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        """Return a Gaussian random value."""
        return self._rng.gauss(mu, sigma)

    def choice(self, seq: Sequence[T]) -> T:
        """Return a random element from a non-empty sequence."""
        return self._rng.choice(seq)

    def choices(
        self,
        population: Sequence[T],
        weights: Sequence[float] | None = None,
        k: int = 1,
    ) -> list[T]:
        """Return k random elements with optional weights."""
        return self._rng.choices(population, weights=weights, k=k)

    def sample(self, population: Sequence[T], k: int) -> list[T]:
        """Return k unique random elements from the population."""
        return self._rng.sample(population, k)

    def shuffle(self, x: list) -> None:
        """Shuffle list in place."""
        self._rng.shuffle(x)

    def weighted_choice(self, items: Sequence[T], weights: Sequence[float]) -> T:
        """Return a single random element weighted by weights."""
        return self._rng.choices(items, weights=weights, k=1)[0]

    def np_random(self) -> np.random.Generator:
        """Return the underlying numpy random Generator."""
        return self._np_rng
