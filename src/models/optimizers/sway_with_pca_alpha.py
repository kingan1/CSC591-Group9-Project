import math
from typing import List, Optional

from sklearn.decomposition import PCA

from data import Data
from data.col import Col, Num, Sym
from distance import cosine_similarity, PDist, Distance
from predicate import ZitzlerPredicate
from utils import many, any


class SwayWithPCAAlphaOptimizer:
    def __init__(self, distance_class: Distance = None, reuse: bool = True, far: float = 0.95, halves: int = 512,
                 rest: int = 10, i_min: float = 0.5):
        self._data: Optional[Data] = None
        self._pca_rows: List[List[float]] = [[]]

        self._distance_class = distance_class or PDist(p=2)

        self._reuse = reuse
        self._far = far
        self._halves = halves
        self._rest = rest
        self._i_min = i_min

    def run(self, data: Data):
        self._data: Data = data
        self._run_pca(self._data.cols.x)

        return self._sway()

    def _run_pca(self, cols: List[Col]):
        input_ = [
            [
                col.normalize(row.cells[col.at]) if row.cells[col.at] != "?" else col.normalize(col.mid())
                for col in cols if isinstance(col, Num)
            ]
            for row in self._data.rows
        ]

        for i in range(len(self._data.rows)):
            for col in cols:
                if isinstance(col, Sym):
                    value = self._data.rows[i].cells[col.at]

                    input_[i] += [0.5 if value == "?" else (1 if key == value else 0) for key in col.has.keys()]

        pca_columns = max(1, int(math.log2(len(input_[0]))))

        pca = PCA(n_components=pca_columns)
        self._pca_rows = pca.fit_transform(input_)

    def _project(self, row_index: int, a_index: int, b_index: int, c: float):
        return {
            "row_index": row_index,
            "x": cosine_similarity(
                a=self._distance_class.raw_dist(self._pca_rows[row_index], self._pca_rows[a_index]),
                b=self._distance_class.raw_dist(self._pca_rows[row_index], self._pca_rows[b_index]),
                c=c
            )
        }

    def _half(self, row_indexes: List[int], above=None):
        """
        divides data using 2 far points
        """
        some = many(row_indexes, self._halves)

        a = above if above is not None and self._reuse else any(some)

        tmp = sorted(
            [
                {"row_index": i, "d": self._distance_class.raw_dist(self._pca_rows[i], self._pca_rows[a])}
                for i in some
            ],
            key=lambda x: x["d"]
        )

        far = tmp[int((len(tmp) - 1) * self._far)]

        b, c = far["row_index"], far["d"]

        sorted_rows = sorted(
            map(lambda row_index: self._project(row_index, a, b, c), row_indexes), key=lambda x: x["x"]
        )

        left, right = [], []

        for n, two in enumerate(sorted_rows):
            if (n + 1) <= (len(row_indexes) / 2):
                left.append(two["row_index"])
            else:
                right.append(two["row_index"])

        evals = 1 if above is not None and self._reuse else 2

        return left, right, a, b, c, evals

    def _sway(self):
        def worker(row_indexes: List[int], worse, evals0=None, above=None):
            if len(row_indexes) <= len(self._data.rows) ** self._i_min:
                return row_indexes, many(worse, self._rest * len(row_indexes)), evals0

            l, r, a, b, c, evals_ = self._half(row_indexes, above)

            if ZitzlerPredicate.better(self._data.cols.y, self._data.rows[b], self._data.rows[a]):
                l, r, a, b = r, l, b, a

            for x in r:
                worse.append(x)

            return worker(l, worse, evals_ + evals0, a)

        best, rest, evals = worker(list(range(len(self._data.rows))), [], 0)

        best = [self._data.rows[i] for i in best]
        rest = [self._data.rows[i] for i in rest]

        return Data.clone(self._data, best), Data.clone(self._data, rest), evals
