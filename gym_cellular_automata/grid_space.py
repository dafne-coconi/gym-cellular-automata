from functools import reduce
from operator import mul
from typing import Optional, Sequence

import numpy as np
from gymnasium.spaces import Space


class GridSpace(Space):
    r"""
    A Space for Cellular Automata Lattices.
    Arbitrary integers can be used as cell states.

    Example::

        >>> GridSpace(n=3, shape=(2, 2))
        >>> GridSpace(values=[-1, 0, 1], shape=(2,2))

    """

    def __init__(
        self,
        n: Optional[int] = None,
        values: Optional[Sequence[int]] = None,
        shape: tuple = tuple(),
        probs: Optional[Sequence[float]] = None,
        dtype: np.intc = np.int32,
        seed: int = None,
    ):
        super().__init__(shape, dtype, seed)

        assert shape, "Shape must be a non-empty tuple."

        if values is not None:
            self._from_values = True

            self.values = np.unique(np.array(values, dtype=dtype))
            self.n = len(self.values)

        elif n is not None:
            self._from_values = False

            assert n is not None and n > 0, "'n' must be a positive integer."
            self.n = n

            self.values = np.arange(self.n, dtype=dtype)

        else:
            raise ValueError("'n' or 'values' must be provided.")

        uniform = np.repeat(1.0, self.n) / self.n
        #self.probs = uniform if probs is None else probs
        self.probs = np.array([0.23, 0.76, 0.01])

        assert len(self.values) == len(
            self.probs
        ), "Unique values do NOT MATCH with assigned probabilities."

        self.size = reduce(mul, self.shape)

    def sample(self) -> np.ndarray:
        return self.np_random.choice(
            a=self.values, size=self.size, p=self.probs
        ).reshape(self.shape)
        
    def sample_det(self) -> np.ndarray:
        initial_g = self.np_random.choice(
            a=[0,1], size=self.size, p=[0.2,0.8]
        ).reshape(self.shape)
        i_r = int(np.random.choice(self.shape[0],1))
        i_c = int(np.random.choice(self.shape[1],1))
        #initial_g[i_r,i_c] = 2
        initial_g[7,5] = 2
        return initial_g

    def contains(self, x) -> bool:
        if isinstance(x, list):
            x = np.array(x, dtype=self.dtype)

        return set(np.unique(x)).issubset(set(self.values)) and self.shape == x.shape

    def __repr__(self):
        if self._from_values:
            return f"GridSpace(values={self.values}, shape={self.shape})"

        else:
            return f"GridSpace(n={self.n}, shape={self.shape})"

    def __eq__(self, other):
        return (
            isinstance(other, GridSpace)
            and (self.shape == other.shape)
            and np.all(self.values == other.values)
        )

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True
