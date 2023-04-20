from typing import Optional

from data import Data
from data.col.sym import Sym
from models.explainers.base import BaseExplainer
from sklearn import preprocessing, tree

from utils import Random


class DTreeExplainer(BaseExplainer):
    def __init__(self, seed: int = None):
        super().__init__()
        self._clf = None

        self._data: Optional[Data] = None
        self._random = Random()

        if seed:
            self._random.set_seed(seed)

    def _xpln(self, data: Data, best: Data, rest: Data):
        self._data: Data = data

        self.best = best
        self.rest = rest

        self._clf = self._dtree()

        return self._clf

    def _dtree(self):
        best_rows = []
        for r in self.best.rows:
            best_rows.append([r.cells[c.at] for c in self.best.cols.x] + ["best"])

        rest_rows = []
        for r in self.rest.rows:
            rest_rows.append([r.cells[c.at] for c in self.rest.cols.x] + ["rest"])

        input_ = best_rows + rest_rows
        le = preprocessing.LabelEncoder()

        for i, col in enumerate(self.rest.cols.x):
            if isinstance(col, Sym):
                new_cols = le.fit_transform([x[i] for x in input_])

                for v, x in zip(new_cols, input_):
                    x[i] = v

        input_ = list(filter(self._remove_missing, input_))

        input_data = [x[:-1] for x in input_]
        y = [x[-1] for x in input_]

        clf = tree.DecisionTreeClassifier(random_state=0).fit(input_data, y)
        return clf

    @staticmethod
    def _remove_missing(value):
        return "?" not in value

    @staticmethod
    def selects(rule, data):
        input_ = []

        for r in data.rows:
            input_.append([r.cells[c.at] for c in data.cols.x])

        le = preprocessing.LabelEncoder()

        for i, col in enumerate(data.cols.x):
            if type(col) == Sym:
                new_cols = le.fit_transform([x[i] for x in input_])

                for v, x in zip(new_cols, input_):
                    x[i] = v

        best = []
        rest = []

        for i, x in enumerate(input_):
            if DTreeExplainer._remove_missing(x):
                res = rule.predict([x])

                if res == "best":
                    best.append(data.rows[i])
                else:
                    rest.append(data.rows[i])

        return best
