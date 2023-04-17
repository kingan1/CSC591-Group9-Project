from typing import Optional, List

from data import Data
from data.col import Col
from data.row import Row
from distance import Distance, PDist, cosine_similarity
from predicate import ZitzlerPredicate
from utils import many, any
from .base import BaseOptimizer


class SwayOptimizer(BaseOptimizer):
    def __init__(self, distance_class: Distance = None, reuse: bool = True, far: float = 0.95, halves: int = 512,
                 rest: int = 10, i_min: float = 0.5, seed=None):
        super().__init__(seed)

        self._data: Optional[Data] = None

        self._distance_class = distance_class or PDist(p=2)

        self._reuse = reuse
        self._far = far
        self._halves = halves
        self._rest = rest
        self._i_min = i_min

    def _run(self, data: Data):
        self._data: Data = data

        return self._sway(self._data.cols.x)

    def _project(self, cols: List[Col], row: Row, a: Row, b: Row, c: float):
        return {
            "row": row,
            "x": cosine_similarity(
                a=self._distance_class.dist(cols, row, a),
                b=self._distance_class.dist(cols, row, b),
                c=c
            )
        }

    def _half(self, cols: List[Col], rows: List[Row], above=None):
        """
        divides data using 2 far points
        """
        some = many(rows, self._halves)

        a = above if above is not None and self._reuse else any(some)

        tmp = sorted([{"row": r, "d": self._distance_class.dist(cols, r, a)} for r in some], key=lambda x: x["d"])
        far = tmp[int((len(tmp) - 1) * self._far)]

        b, c = far["row"], far["d"]

        sorted_rows = sorted(map(lambda row: self._project(cols, row, a, b, c), rows), key=lambda x: x["x"])
        left, right = [], []

        for n, two in enumerate(sorted_rows):
            if (n + 1) <= (len(rows) / 2):
                left.append(two["row"])
            else:
                right.append(two["row"])

        evals = 1 if above is not None and self._reuse else 2

        return left, right, a, b, c, evals

    def _sway(self, cols: List[Col]):
        def worker(rows: List[Row], worse, evals0=None, above=None):
            if len(rows) <= len(self._data.rows) ** self._i_min:
                return rows, many(worse, self._rest * len(rows)), evals0

            l, r, a, b, c, evals_ = self._half(cols, rows, above)

            if ZitzlerPredicate.better(self._data.cols.y, b, a):
                l, r, a, b = r, l, b, a

            for x in r:
                worse.append(x)

            return worker(l, worse, evals_ + evals0, a)

        best, rest, evals = worker(self._data.rows, [], 0)

        return Data.clone(self._data, best), Data.clone(self._data, rest), evals
