import re
from typing import List

from .col import Col, Sym, Num
from .row import Row


class Cols:
    """
    Factory for managing a set of NUMs or SYMs
    """

    def __init__(self, t: List):
        """
        Initializes a new Cols object, contains many columns

        :param t: Row to convert to NUMs or SYMs
        """
        self.names: List = t

        self.all: List[Col] = []

        self.x: List[Col] = []
        self.y: List[Col] = []

        self.klass = None

        for n, s in enumerate(t):
            s = s.strip()
            # Generate Nums and Syms from column names
            col = Num(n, s) if re.findall("^[A-Z]+", s) else Sym(n, s)
            self.all.append(col)

            if not re.findall("X$", s):
                if re.findall("!$", s):
                    self.klass = col
                # if it ends in "!", "+", or "-", append it to self.y, else append to self.x
                self.y.append(col) if re.findall("[!+-]$", s) else self.x.append(col)

    def add(self, row: Row) -> None:
        """
        Updates the columns with details from row

        :param row: Row to add
        """
        for _, t in enumerate([self.x, self.y]):
            for _, col in enumerate(t):
                col.add(row.cells[col.at])
