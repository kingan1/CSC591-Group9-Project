from typing import List

from data.col import Col
from data.row import Row


class Distance:
    def dist(self, cols: List[Col], row1: Row, row2: Row):
        raise NotImplementedError("Cannot create object of Distance")

    def raw_dist(self, data1: List[float], data2: List[float]):
        raise NotImplementedError("Cannot create object of Distance")


class PDist(Distance):
    def __init__(self, p: float):
        self.p = p

    def dist(self, cols: List[Col], row1: Row, row2: Row):
        d = 0

        for col in cols:
            d = d + col.dist(row1.cells[col.at], row2.cells[col.at]) ** self.p

        return (d / len(cols)) ** (1 / self.p)

    def raw_dist(self, data1: List[float], data2: List[float]):
        d = 0

        assert len(data1) == len(data2)

        for i in range(len(data1)):
            d = d + abs(data1[i] - data2[i]) ** self.p

        return (d / len(data1)) ** (1 / self.p)


def cosine_similarity(a: float, b: float, c: float):
    if c == 0:
        return 0

    return (a ** 2 + c ** 2 - b ** 2) / (2 * c)
