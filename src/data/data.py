from functools import cmp_to_key
from typing import Union, List, Dict, Optional

from predicate import ZitzlerPredicate
from utils import csv, rnd
from .col import Num, Sym
from .cols import Cols
from .row import Row


class Data:
    def __init__(self, src: Union[str, List] = None, rows: Union[List, Row] = None):
        self.rows: List[Row] = []
        self.cols: Optional[Cols] = None

        if src or rows:
            self.read(src, rows)

    def read(self, src: Union[str, List], rows: Union[List, Row] = None):
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
    def clone(data: 'Data', ts: List = None) -> 'Data':
        """
        Returns a clone with the same structure as self.

        :param data: Initial data for the clone
        :param ts: List of Rows to add
        """
        if ts is None:
            ts = []

        data1 = Data()
        data1.add(data.cols.names)

        for _, t in enumerate(ts):
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

    def betters(self, n: int = None, predicate=None):
        if predicate is None:
            predicate = ZitzlerPredicate

        tmp = sorted(
            self.rows,
            key=cmp_to_key(lambda row1, row2: -1 if predicate.better(self.cols.y, row1, row2) else 1)
        )
        if n is None:
            return tmp
        

        return tmp[0:n], tmp[n:] 
