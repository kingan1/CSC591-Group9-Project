from math import inf, floor
from typing import Union, Optional, List

from num import Num
from options import options
from sym import Sym
from utils import copy


class Range:
    def __init__(self, at, txt, lo, hi=None):
        self.at = at
        self.txt = txt

        self.lo = lo
        self.hi = lo or hi or lo

        self.y = Sym()

    def extend(self, n: int, s: str):
        self.lo = min(n, self.lo)
        self.hi = max(n, self.hi)

        self.y.add(s)


def merge(col1: Union[Sym, Num], col2: Union[Sym, Num]) -> Union[Sym, Num]:
    new = copy(col1)

    if isinstance(col1, Sym):
        for x, n in col2.has.items():
            new.add(x, n)
    else:
        for _, n in col2.has.items():
            new.add(n)

        new.lo = min(col1.lo, col2.lo)
        new.hi = max(col1.hi, col2.hi)

    return new


def merge2(col1: Union[Sym, Num], col2: Union[Sym, Num]) -> Optional[Union[Sym, Num]]:
    new = merge(col1, col2)

    if new.div() <= ((col1.div() * col1.n) + (col2.div() * col2.n)) / new.n:
        return new


def merge_any(ranges0: List[Range]) -> List[Range]:
    def no_gaps(t: List[Range]):
        if not t:
            return t

        for j in range(1, len(t)):
            t[j].lo = t[j - 1].hi

        t[0].lo = -inf
        t[len(t) - 1].hi = inf

        return t

    ranges1 = []
    j = 0

    while j < len(ranges0):
        left = ranges0[j]
        right = None if j == len(ranges0) - 1 else ranges0[j + 1]

        if right is not None:
            y = merge2(left.y, right.y)

            if y is not None:
                j += 1
                left.hi, left.y = right.hi, y

        ranges1.append(left)
        j += 1
    return no_gaps(ranges0) if len(ranges0) == len(ranges1) else merge_any(ranges1)

def extend(range, n, s):
    range.lo = min(n, range.lo)
    range.hi = max(n, range.hi)
    range.y.add(s)


def bins(cols,rowss):
    def with1Col(col):
        def itself(x):
            return x
        n,ranges = withAllRows(col)
        ranges   = sorted(list(map(itself, ranges.values())),key = lambda x: x.lo) #-- keyArray to numArray, sorted
        if   type(col) == Sym:
            return ranges 
        else:
            return merges(ranges, n/options["bins"], options["D"]*col.div())
    def withAllRows(col):
        def xy(x,y):
            nonlocal n
            if x != "?":
                n = n + 1
                k = bin(col,x)
                ranges[k] = ranges.get(k, Range(col.at,col.txt,x))
                extend(ranges[k], x, y)
        #-----------
        n,ranges = 0,{}
        for y,rows in rowss.items():
            for _,row in enumerate(rows):
                xy(row.cells[col.at],y)
        return n, ranges 
    # end --------------
    ret = list(map(with1Col, cols))
    return ret

# -- Map `x` into a small number of bins. `SYM`s just get mapped
# -- to themselves but `NUM`s get mapped to one of `is.bins` values.
# -- Called by function `bins`.
def bin(col,x):
    if x=="?" or type(col) == Sym:
        return x
    tmp = (col.hi - col.lo)/(options["bins"] - 1)
    return col.hi == col.lo and 1 or floor(x/tmp + .5)*tmp


def value(has, n_b: int = 1, n_r: int = 1, s_goal: str = None) -> float:
    b, r = 0, 0

    for x, n in has.items():
        if x == s_goal:
            b = b + n
        else:
            r = r + n

    b, r = b / (n_b + 1 / inf), r / (n_r + 1 / inf)
    return b ** 2 / (b + r)


def merges(ranges0,nSmall,nFar):
    def noGaps(t):
        if not t:
            return t
        for j in range(1,len(t)):
            t[j].lo = t[j-1].hi
        t[0].lo  = -inf
        t[len(t)-1].hi =  inf
        return t
    def try2Merge(left,right,j):
        y = merged(left.y, right.y, nSmall, nFar)
        if y: 
            j = j+1 #-- next round, skip over right.
            left.hi, left.y = right.hi, y #end 
        return j , left 
    # end -------------
    ranges1,j,here = [],0, None
    while j < len(ranges0):
        here = ranges0[j]
        if j < len(ranges0)-1:
            j,here = try2Merge(here, ranges0[j+1], j)
        j=j+1
        ranges1.append(here)
    return noGaps(ranges0) if len(ranges0)==len(ranges1) else merges(ranges1,nSmall,nFar)

#   -- If (1) the parts are too small or
# -- (2) the whole is as good (or simpler) than the parts,
# -- then return the merge.
def merged(col1,col2,nSmall, nFar):
    new = merge(col1,col2)
    if nSmall and col1.n < nSmall or col2.n < nSmall:
        return new
    if nFar   and not type(col1) == Sym and abs(col1.div() - col2.div()) < nFar:
        return new
    if new.div() <= (col1.div()*col1.n + col2.div()*col2.n)/new.n:
        return new

# -- Merge two `cols`. Called by def `merged`.
def merge(col1,col2):
    new = copy(col1)
    if   type(col1) == Sym:
        for x,n in col2.has.items():
            new.add(x,n)
    else:
        for _,n in enumerate(col2.has):
            new.add(n)
        new.lo = min(col1.lo, col2.lo)
        new.hi = max(col1.hi, col2.hi)
    return new
