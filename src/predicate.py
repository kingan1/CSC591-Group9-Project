import math
from typing import List, Union
from data.col import Col
from data.row import Row
from utils import norm
from options import options


class ZitzlerPredicate:
    @staticmethod
    def better(cols: List[Union[Col]], row1: Row, row2: Row, s1=0, s2=0, x=0, y=0):
        for col in cols:
            x = norm(col, row1.cells[col.at])
            y = norm(col, row2.cells[col.at])

            s1 = s1 - math.exp(col.w * (x - y) / len(cols))
            s2 = s2 - math.exp(col.w * (y - x) / len(cols))

        return s1 / len(cols) < s2 / len(cols)
    

class HyperparameterPredicate:
    @staticmethod
    def better(cols: List[Union[Col]], row1: Row, row2: Row, data=None, opt=None):
        
        def get_options(row):
            global options
            # gets a fresh options dictionary, with the given
            #  hyperparameters changed
            options = options.t.copy()
            
            for item,col in zip(row,[c.name for c in cols]):
                options[col] = item
            return options
        
        # performs sway on the data file with the first set of hyperparameters
        # data=Data(options["file"])
        # best,rest,evals = data.sway(options_new = get_options(row1))
        options = get_options(row1)
        best, rest, evals = opt(
                reuse=options["reuse"],
                far=options["Far"],
                halves=options["Halves"],
                rest=options["Rest"],
                i_min=options["IMin"]
        ).run(data)
        
        # records the best.stats()
        row_best = [0 for _ in data.cols.names]
        # for each y column
        for key, val in best.stats().items():
            for ys in data.cols.y:
                if ys.txt == key:
                    # set the row[y column.at] = stats for that column
                    row_best[ys.at] = val

        # performs sway on the data file with the second set of hyperparameters
        best2,rest2,evals2 = data.sway(options_new = get_options(row2))
        row_best2 = [0 for _ in data.cols.names]
        # for each y column
        for key, val in best2.stats().items():
            for ys in data.cols.y:
                if ys.txt == key:
                    # set the row[y column.at] = stats for that column
                    row_best2[ys.at] = val

        # return if the results of better(sway(first hyperparameters), sway(second))
        res = data.better(row_best,row_best2)
        # the number of gridsearch evals should take into account the number of evals
        #  each individual sway took
        gs_evals = evals+evals2
        return [res, gs_evals]

