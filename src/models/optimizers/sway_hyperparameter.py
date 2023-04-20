from typing import Optional, List

from data import Data
from data.col import Col
from data.row import Row
from distance import Distance, PDist, cosine_similarity
from predicate import HyperparameterPredicate
from utils import many, any
from .base import BaseOptimizer
from .sway import SwayOptimizer
from itertools import product


class SwayHyperparameterOptimizer(BaseOptimizer):
    def __init__(self, p: float = 2, reuse: bool = True, far: float = 0.95, halves: int = 512,
                 rest: int = 10, i_min: float = 0.5, file: str = None, sway2: bool = False, seed=None):
        super().__init__(seed)
        
        self._data: Optional[Data] = None

        self._distance_class = PDist(p)

        self._reuse = reuse
        self._far = far
        self._halves = halves
        self._rest = rest
        self._i_min = i_min
        self._file = file
        self._sway2 = sway2
        self._p = p
        self._options = None

    def _run(self, data):
        evals = 0
        if not self._options:
            evals = self._explore_parameters()
        
        
        best, rest, evals_1 = SwayOptimizer(
                reuse=self._options["reuse"],
                far=self._options["Far"],
                halves=self._options["Halves"],
                rest=self._options["Rest"],
                i_min=self._options["IMin"],
                p=self._options["P"]
            ).run(data)
        return best, rest, evals + evals_1

    def _project(self, cols: List[Col], row: Row, a: Row, b: Row, c: float):
        return {
            "row": row,
            "x": cosine_similarity(
                a=self._distance_class.dist(cols, row, a),
                b=self._distance_class.dist(cols, row, b),
                c=c
            )
        }

    def _half(self, cols: List[Col], rows: List[Row], above=None):
        """
        divides data using 2 far points
        """
        some = many(rows, self._halves)

        a = above if above is not None and self._reuse else any(some)

        tmp = sorted([{"row": r, "d": self._distance_class.dist(cols, r, a)} for r in some], key=lambda x: x["d"])
        far = tmp[int((len(tmp) - 1) * self._far)]

        b, c = far["row"], far["d"]

        sorted_rows = sorted(map(lambda row: self._project(cols, row, a, b, c), rows), key=lambda x: x["x"])
        left, right = [], []

        for n, two in enumerate(sorted_rows):
            if (n + 1) <= (len(rows) / 2):
                left.append(two["row"])
            else:
                right.append(two["row"])

        evals = 1 if above is not None and self._reuse else 2

        return left, right, a, b, c, evals

    def _sway(self, cols: List[Col]):
        def worker(rows: List[Row], worse, evals0=None, above=None):
            if len(rows) <= len(self._data.rows) ** self._i_min:
                return rows, many(worse, self._rest * len(rows)), evals0

            l, r, a, b, c, evals_ = self._half(cols, rows, above)
            new_data = Data(self._file)

            [better, gs_evals_] = HyperparameterPredicate.better(self._data.cols, b, a, new_data, SwayOptimizer)
            evals_ += gs_evals_
            if better:
                l, r, a, b = r, l, b, a

            for x in r:
                worse.append(x)

            return worker(l, worse, evals_ + evals0, a)

        best, rest, evals = worker(self._data.rows, [], 0)

        return Data.clone(self._data, best), Data.clone(self._data, rest), evals
    


    def _explore_parameters(self):
        if self._sway2 and not self._options:
            print("refreshing sway")
            # list of parameters used by sway, as well as example values to sample
            params = { 
                "Far":  [i/100 for i in range(70,100,5)],
                "Halves":  [i for i in range(100, 700,100)],
                "IMin":  [i/10 for i in range(0,8,2)],
                "Max": [i for i in range(1, 150, 25)], # not used
                "P":  [1+(i/10) for i in range(10)],
                "Rest":  [i for i in range(1,5)],
                "reuse":  [True,False], 
            }
            # types of each parameter
            types = { 
                "Far":  float,
                "Halves":  int,
                "IMin":  float,
                "Max": int,
                "P":  int,
                "Rest":  int,
                "reuse":  bool
            }

            # get each combination of parameters
            permutations_dicts = [dict(zip(params.keys(), v)) for v in product(*params.values())]
            
            # this is used to create a sample CSV for our parameters
            test_params = {}
            for k,v in params.items():
                test_params[k] = v[0]

            with open("gridsearch_params.csv", "w") as fp:
                fp.write(",".join(test_params.keys()) + "\n")
                fp.write(",".join([str(c) for c in test_params.values()]))
            
            # create a data object of all combinations of hyperparameters
            test_data = Data("gridsearch_params.csv")
            data=Data(src=test_data,rows=[list(v.values()) for v in permutations_dicts])

            # get the best combination of hyperparameters
            # self._sway()
            self._data = data
            best, _, n_evals = self._sway(data.cols.x)
            

            # set the hyperparameters as the "average" of the hyperparameters in best
            res = best.stats(best.cols.x)
            res.pop("N")
            res = {k: types[k](v) for k,v in res.items()}
            print("new: ", res)
            print()
            self._options = res
            
            return n_evals
        
        # these are optimized for auto2.csv
        self._options = {'Far': 0.95, 'Halves': 500, 'IMin': 0.0, 'P': 1, 'Rest': 2, 'reuse': False}
        return 0

