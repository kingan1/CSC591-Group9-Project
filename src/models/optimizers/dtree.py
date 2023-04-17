from typing import Optional

from data import Data
from data.col.sym import Sym
from .base import BaseOptimizer
from sklearn import preprocessing, tree


class DtreeOptimizer(BaseOptimizer):
    def __init__(self, best=None, rest=None, seed=None):
        super().__init__(seed)
        self._data: Optional[Data] = None

        self._best= best
        self._rest = rest

    def _run(self, data: Data):
        self._data: Data = data

        self._clf = self._dtree()

        return self._classify_dtree()
    
    def _dtree(self):
        best_rows = []
        for r in self._best.rows:
            best_rows.append([r.cells[c.at] for c in self._best.cols.x] + ["best"])
        rest_rows = []
        for r in self._rest.rows:
            rest_rows.append([r.cells[c.at] for c in self._rest.cols.x]+ ["rest"] )


        X = best_rows + rest_rows
        le = preprocessing.LabelEncoder()
        for i,col in enumerate(self._rest.cols.x):
            if isinstance(col, Sym):
                new_cols = le.fit_transform([x[i] for x in X])
                for v, x in zip(new_cols, X):
                    x[i] = v

        X = list(filter(self._remove_missing, X))
        

        X_data = [x[:-1] for x in X]
        y = [x[-1] for x in X]
        
        clf = tree.DecisionTreeClassifier(random_state=0).fit(X_data, y)
        return clf
    
    def _remove_missing(self, X):
        return not "?" in X

    def _classify_dtree(self):
        X = []
        for r in self._data.rows:
            X.append([r.cells[c.at] for c in self._data.cols.x])
        le = preprocessing.LabelEncoder()
        for i,col in enumerate(self._data.cols.x):
            if type(col) == Sym:
                new_cols = le.fit_transform([x[i] for x in X])
                for v, x in zip(new_cols, X):
                    x[i] = v
        best = []
        rest = []
        for i, x in enumerate(X):
            if self._remove_missing(x):
                res = self._clf.predict([x])
                if res == "best":
                    best.append(self._data.rows[i])
                else:
                    rest.append(self._data.rows[i])


        return best, rest, 0

    