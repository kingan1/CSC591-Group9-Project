from typing import Tuple, List, Dict

from data import Data
from discretization import Range, value, bins
from models.explainers.base import BaseExplainer


class RangeExplainer(BaseExplainer):
    def __init__(self):
        super().__init__()

        self.tmp: List[Tuple[Range, int, float]] = []
        self.max_sizes: Dict[str, int] = {}

    def xpln(self, data: Data, best: Data, rest: Data):
        def v(has):
            return value(has, len(best.rows), len(rest.rows), "best")

        self.best = best
        self.rest = rest

        self.tmp: List[Tuple[Range, int, float]] = []
        self.max_sizes: Dict[str, int] = {}

        tmp, self.max_sizes = [], {}

        for _, ranges in enumerate(bins(data.cols.x, {"best": best.rows, "rest": rest.rows})):
            self.max_sizes[ranges[0].txt] = len(ranges)

            for _, range_ in enumerate(ranges):
                tmp.append({"range": range_, "max": len(ranges), "val": v(range_.y.has)})

        rule, most = RangeExplainer._first_n(sorted(tmp, key=lambda x: x["val"], reverse=True), self._score)

        return rule

    def _score(self, ranges: List[Range]):
        rule = RangeExplainer._rule(ranges, self.max_sizes)

        if rule:
            bestr = RangeExplainer.selects(rule, self.best.rows)
            restr = RangeExplainer.selects(rule, self.rest.rows)

            if len(bestr) + len(restr) > 0:
                return value(
                    has={
                        "best": len(bestr),
                        "rest": len(restr)
                    },
                    n_b=len(self.best.rows),
                    n_r=len(self.rest.rows),
                    s_goal="best"
                ), rule

        return None, None

    @staticmethod
    def _first_n(sorted_ranges: List[dict], score_fun):
        first = sorted_ranges[0]['val']

        def useful(range_):
            if range_['val'] > 0.05 and range_['val'] > first / 10:
                return range_

        sorted_ranges = [s for s in sorted_ranges if useful(s)]
        most: int = -1
        out: int = -1

        for n in range(len(sorted_ranges)):
            tmp, rule = score_fun([r['range'] for r in sorted_ranges[:n + 1]])

            if tmp is not None and tmp > most:
                out, most = rule, tmp

        return out, most

    @staticmethod
    def _rule(ranges, max_size):
        t = {}

        for _, range_ in enumerate(ranges):
            t[range_.txt] = t.get(range_.txt, [])
            t[range_.txt].append({"lo": range_.lo, "hi": range_.hi, "at": range_.at})

        return RangeExplainer._prune(t, max_size)

    @staticmethod
    def _prune(rule, max_size):
        n = 0
        new_rule = {}

        for txt, ranges in rule.items():
            n = n + 1

            if len(ranges) == max_size[txt]:
                n = n - 1
                rule[txt] = None
            else:
                new_rule[txt] = ranges

        if n > 0:
            return new_rule

        return None

    @staticmethod
    def selects(rule, rows):
        def disjunction(ranges, row):
            for rang in ranges:
                at = rang['at']
                x = row.cells[at]
                lo = rang['lo']
                hi = rang['hi']

                if x == '?' or (lo == hi and lo == x) or (lo <= x < hi):
                    return True

            return False

        def conjunction(row):
            for _, ranges in rule.items():
                if not disjunction(ranges, row):
                    return False
            return True

        def function(row):
            return row if conjunction(row) else None

        r = []

        for item in list(map(function, rows)):
            if item:
                r.append(item)

        return r
