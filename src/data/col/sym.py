import collections
import math

from .base import Col


class Sym(Col):
    def __init__(self, at: int = 0, txt: str = ""):
        super().__init__(at=at, txt=txt)

        self.has = collections.defaultdict(int)

        self.most = 0
        self.mode = None

    def add(self, x: str, n: int = 1):
        """
        Updates counts of things seen so far

        :param x: Symbol to add
        :param n: Number of times to add
        """
        if x != "?":
            self.n = self.n + n
            self.has[x] = n + (self.has[x] or 0)

            if self.has[x] > self.most:
                self.most = self.has[x]
                self.mode = x

        return x

    def mid(self):
        """
        Returns the mode
        """
        return self.mode

    def div(self):
        """
        Returns the entropy
        """

        def fun(p):
            return p * math.log(p, 2)

        e = 0

        for _, n in self.has.items():
            e = e + fun(n / self.n)

        return -e

    def dist(self, data1, data2):
        if data1 == '?' and data2 == '?':
            return 1
        elif data1 == data2:
            return 0
        else:
            return 1
