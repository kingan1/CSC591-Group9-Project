from typing import Optional, List
import random
import math
import numpy as np
from data import Data
from data.col import Col
from data.row import Row
from distance import Distance, PDist, cosine_similarity
from predicate import ZitzlerPredicate
from utils import many, any
from .base import BaseOptimizer
from sklearn.cluster import AgglomerativeClustering

class SwayWithAggloOptimizer:
    def __init__(self, distance_class: Distance = None, reuse: bool = True, far: float = 0.95, halves: int = 512,
                 rest: int = 10, i_min: float = 0.5):
        self._data: Optional[Data] = None

        self._distance_class = distance_class or PDist(p=2)

        self._reuse = reuse
        self._far = far
        self._halves = halves
        self._rest = rest
        self._i_min = i_min

    def run(self, data: Data):
        self._data: Data = data

        return self.sway_agglo(self._data.cols.x)
    
    def half_agglo(self, rows=None, cols=None):
        e = 0
        if not rows:
            rows = self._data.rows

        rows_numpy = np.array([r.cells for r in rows])
        agg = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward').fit(rows_numpy)
        left = []
        right = []
        for idx, label in enumerate(agg.labels_):
            if label == 0:
                left.append(rows[idx])
            else:
                right.append(rows[idx])

        e = 1 if self._reuse else 2
        return left, right, random.choices(left, k=10), random.choices(right, k=10), e
    
    def sway_agglo(self, cols=None):
        
        def worker(rows, worse,evals0):
            if len(rows) <= len(self._data.rows) ** self._i_min:
                return rows, many(worse, self._rest * len(rows)), evals0
            l, r, A, B,evals1 = self.half_agglo(rows, cols)
            if self.better_multiple(B, A):
                l, r, A, B = r, l, B, A
            for x in r:
                worse.append(x)
            return worker(l, worse, evals1 + evals0)

        rows = row_cleaning(self._data.rows)
        best, rest, evals = worker(rows, [],0)
        return Data.clone(self._data,best), Data.clone(self._data,rest),evals

    def better_multiple(self, rows1, rows2, s1=0, s2=0, ys=None, x=0, y=0):
        if not ys:
            ys = self._data.cols.y
        for col in ys:
            for row1, row2 in zip(rows1, rows2):
                x = col.normalize(row1.cells[col.at])
                y = col.normalize(row2.cells[col.at])
                s1 = s1 - math.exp(col.w * (x - y) / len(ys))
                s2 = s2 - math.exp(col.w * (y - x) / len(ys))
        return s1 / len(ys) < s2 / len(ys)
    
def row_cleaning(rows):
    cleaned_rows = []
    for row in rows:
        flag = True
        for cell in row.cells:
            if isinstance(cell, str) and cell.strip() == '?':
                flag = False
                break
        if flag:
            cleaned_rows.append(row)
    return cleaned_rows