from typing import Optional, List

from data import Data
from data.col import Col
from data.row import Row
from distance import Distance, PDist, cosine_similarity
from options import options
from predicate import ZitzlerPredicate
from utils import many, any


class SwayOptimizer:
    def __init__(self, distance_class: Distance = None):
        self._data: Optional[Data] = None
        self._distance_class = distance_class or PDist(p=2)

    def run(self, data: Data):
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
        some = many(rows, int(options["Halves"]))

        a = any(some) if above is not None and options["reuse"] else any(some)

        tmp = sorted([{"row": r, "d": self._distance_class.dist(cols, r, a)} for r in some], key=lambda x: x["d"])
        far = tmp[int((len(tmp) - 1) * options["Far"])]

        b, c = far["row"], far["d"]

        sorted_rows = sorted(map(lambda row: self._project(cols, row, a, b, c), rows), key=lambda x: x["x"])
        left, right = [], []

        for n, two in enumerate(sorted_rows):
            if (n + 1) <= (len(rows) / 2):
                left.append(two["row"])
            else:
                right.append(two["row"])

        evals = 1 if above is not None and options["reuse"] else 2

        return left, right, a, b, c, evals

    def _sway(self, cols: List[Col]):
        def worker(rows: List[Row], worse, evals0=None, above=None):
            if len(rows) <= len(self._data.rows) ** options["IMin"]:
                return rows, many(worse, options["Rest"] * len(rows)), evals0

            l, r, a, b, c, evals_ = self._half(cols, rows, above)

            if ZitzlerPredicate.better(self._data.cols.y, b, a):
                l, r, a, b = r, l, b, a

            for x in r:
                worse.append(x)

            return worker(l, worse, evals_ + evals0, a)

        best, rest, evals = worker(self._data.rows, [], 0)

        return Data.clone(self._data, best), Data.clone(self._data, rest), evals
