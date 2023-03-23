import math
from functools import cmp_to_key
from typing import Union, List, Dict

from cols import Cols
from num import Num
from options import options
from row import Row
from sym import Sym
from utils import any, csv, many, norm, rnd


class Data:

    def __init__(self):
        self.rows = list()
        self.cols = None

    def read(self, src: Union[str, List], rows=None) -> None:
        def f(t):
            self.add(t)
        if type(src) == str:
            csv(src, f)
        else:
            self.cols = Cols(src.cols.names)
            
            for row in rows:
                self.add(row)

    def add(self, t: Union[List, Row]):
        """
        Adds a new row and updates column headers.

        :param t: Row to be added
        """
        if self.cols:
            t = t if isinstance(t, Row) else Row(t)

            self.rows.append(t)
            self.cols.add(t)
        else:
            self.cols = Cols(t)

    @staticmethod
    def clone(data, ts={}) -> 'Data':
        """
        Returns a clone with the same structure as self.

        :param init: Initial data for the clone
        """
        data1 = Data()
        data1.add(data.cols.names)
        for _, t in enumerate(ts or {}):
            data1.add(t)
        return data1

    def stats(self, cols: List[Union[Sym, Num]] = None, nplaces: int = 2, what: str = "mid") -> Dict:
        """
        Returns mid or div of cols (defaults to i.cols.y).

        :param cols: Columns to collect statistics for
        :param nplaces: Decimal places to round the statistics
        :param what: Statistics to collect
        :return: Dict with all statistics for the columns
        """
        ret = dict(sorted({col.txt: rnd(getattr(col, what)(), nplaces) for col in cols or self.cols.y}.items()))
        ret["N"] = len(self.rows)
        return ret

    def cluster(self, rows: List[Row] = None, cols: List[Union[Sym, Num]] = None, above: Row = None):
        """
        Performs N-level bi clustering on the rows.

        :param rows: Data points to cluster
        :param min_: Clustering threshold value
        :param cols: Columns to cluster on
        :param above: Point chosen as A
        :return: Rows under the current node
        """
        rows = self.rows if rows is None else rows
        cols = self.cols.x if cols is None else cols

        node = {"data": self.clone(rows)}

        if len(rows) >= 2:
            left, right, node['A'], node['B'], node['mid'], node['c'] = self.half(rows, cols, above)

            node['left'] = self.cluster(left, cols, node['A'])
            node['right'] = self.cluster(right, cols, node['B'])

        return node

    def sway(self, cols=None):
        def worker(rows, worse, evals0=None, above=None):
            if len(rows) <= len(self.rows) ** options["min"]:
                return rows, many(worse, options['rest'] * len(rows)), evals0

            l, r, A, B, c, evals = self.half(rows, cols, above)

            if self.better(B, A):
                l, r, A, B = r, l, B, A

            for x in r:
                worse.append(x)

            return worker(l, worse, evals + evals0, A)

        best, rest, evals = worker(self.rows, [], 0)

        return Data.clone(self, best), Data.clone(self, rest), evals

    def better(self, row1, row2, s1=0, s2=0, ys=None, x=0, y=0):
        if not ys:
            ys = self.cols.y

        for col in ys:
            x = norm(col, row1.cells[col.at])
            y = norm(col, row2.cells[col.at])

            s1 = s1 - math.exp(col.w * (x - y) / len(ys))
            s2 = s2 - math.exp(col.w * (y - x) / len(ys))

        return s1 / len(ys) < s2 / len(ys)

    def betters(self, n=None):
        tmp = sorted(self.rows, key=cmp_to_key(lambda row1, row2: -1 if self.better(row1, row2) else 1))
        return tmp[1:n], tmp[n+1:] if n is not None else tmp

    def half(self, rows=None, cols=None, above=None):
        """
        divides data using 2 far points
        """

        def gap(r1, r2):
            return self.dist(r1, r2, cols)

        def cos(a, b, c):
            return (a ** 2 + c ** 2 - b ** 2) / (2 * c)

        def proj(r):
            return {'row': r, 'x': cos(gap(r, A), gap(r, B), c)}

        rows = rows or self.rows
        some = many(rows, options["Halves"])

        A = above if above and options["Reuse"] else any(some)

        tmp = sorted([{"row": r, "d": gap(r, A)} for r in some], key=lambda x: x["d"])
        far = tmp[int((len(tmp) - 1) * options["Far"])]

        B, c = far["row"], far["d"]

        sorted_rows = sorted(map(proj, rows), key=lambda x: x["x"])
        left, right = [], []

        for n, two in enumerate(sorted_rows):
            if (n + 1) <= (len(rows) / 2):
                left.append(two["row"])
            else:
                right.append(two["row"])

        evals = 1 if options["Reuse"] and above else 2

        return left, right, A, B, c, evals

    def tree(self, rows=None, cols=None, above=None):
        rows = rows if rows else self.rows

        here = {"data": Data.clone(self, rows)}

        if (len(rows)) >= 2 * ((len(self.rows)) ** options['min']):
            left, right, A, B, _, _ = self.half(rows, cols, above)
            here["left"] = self.tree(left, cols, A)
            here["right"] = self.tree(right, cols, B)
        return here

    def dist(self, t1, t2, cols=None):
        def dist1(col, x, y):
            if x == "?" and y == "?":
                return 1
            if type(col) is Sym:
                return 0 if x == y else 1
            x, y = norm(col, x), norm(col, y)
            if x == "?":
                x = 1 if y < 0.5 else 1
            if y == "?":
                y = 1 if x < 0.5 else 1
            return abs(x - y)

        d = 0
        cols = cols or self.cols.x

        for col in cols:
            d = d + dist1(col, t1.cells[col.at], t2.cells[col.at]) ** options["p"]

        return (d / len(cols)) ** (1 / options["p"])
