import math
from typing import List, Union

from data.col import Col
from data.row import Row
from utils import norm


class ZitzlerPredicate:
    @staticmethod
    def better(cols: List[Union[Col]], row1: Row, row2: Row, s1=0, s2=0, x=0, y=0):
        for col in cols:
            x = norm(col, row1.cells[col.at])
            y = norm(col, row2.cells[col.at])

            s1 = s1 - math.exp(col.w * (x - y) / len(cols))
            s2 = s2 - math.exp(col.w * (y - x) / len(cols))

        return s1 / len(cols) < s2 / len(cols)
