import collections
import math
from typing import Union


class Sym:
    """
    Summarizes a stream of Symbols.
    """

    def __init__(self, at: int = 0, txt: str = ""):
        self.at = at
        self.txt = txt

        self.n = 0
        self.has = collections.defaultdict(int)
        self.most = 0
        self.mode = None

    def add(self, x: str, n=1):
        """
        Updates counts of things seen so far

        :param x: Symbol to add
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

    def dist(self, s1, s2):
        if s1 == '?' and s2 == '?':
            return 1
        elif s1 == s2:
            return 0
        else:
            return 1
