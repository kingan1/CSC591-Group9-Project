from itertools import combinations
from typing import Dict

from tabulate import tabulate

from data import Data
from explain import Explain, selects
from models.optimizers import SwayOptimizer, SwayWithPCAOptimizer
from models.optimizers.base import BaseOptimizer
from options import options
from stats import cliffs_delta, bootstrap

help_ = """

project: multi-goal semi-supervised algorithms
(c) Group 9
  
USAGE: python3 main.py [OPTIONS] [-g ACTIONS]
  
OPTIONS:
  -b  --bins        initial number of bins           = 16
  -c  --cliff       cliff's delta threshold          = .147
  -d  --D           different is over sd*d           = .35
  -F  --Far         distance to distant              = .95
  -h  --help        show help                        = false
  -H  --Halves      search space for clustering      = 512
  -I  --IMin        size of smallest cluster         = .5
  -M  --Max         numbers                          = 512
  -p  --P           dist coefficient                 = 2
  -R  --Rest        how many of rest to sample       = 10
  -r  --reuse       child splits reuse a parent pole = true
  -x  --Bootstrap   number of samples to bootstrap   = 512    
  -o  --Conf        confidence interval              = 0.05
  -f  --file        file to generate table of        = ../data/auto2.csv
  -n  --Niter       number of iterations to run      = 20
  -w  --wColor      output with color                = true
"""


class ResultsGenerator:
    def __init__(self, data_src: str, optimizers: Dict[str, BaseOptimizer], n_iters=20):
        self._data = None
        self._data_src = data_src
        self._n_iters = n_iters

        self._optimizers = optimizers

        self._results = self._get_results()
        self._n_evals = self._get_n_evals()

        self._comparisons = self._get_comparisons()

    def _get_comparisons(self):
        comparisons = [[["all", "all"], None], ]

        for optimizer in self._optimizers:
            comparisons.append([["all", optimizer], None])

        for combination in combinations(list(self._optimizers.keys()), 2):
            comparisons.append([list(combination), None])

        for optimizer in self._optimizers:
            comparisons.append([[optimizer, "top"], None])

        return comparisons

    def _get_results(self):
        optimizers = ["all", ] + \
                     list(self._optimizers.keys()) + \
                     [i + "_xpln" for i in list(self._optimizers.keys())] + \
                     ["top", ]

        return {optimizer: [] for optimizer in optimizers}

    def _get_n_evals(self):
        optimizers = ["all", ] + \
                     list(self._optimizers.keys()) + \
                     [i + "_xpln" for i in list(self._optimizers.keys())] + \
                     ["top", ]

        return {optimizer: 0 for optimizer in optimizers}

    def run(self):
        i = 0

        while i < self._n_iters:
            self._data = Data(self._data_src)

            self._results["all"].append(self._data)
            self._n_evals["all"] += 0

            for o_name, optimizer in self._optimizers.items():
                rule = -1

                while rule == -1:
                    best, rest, evals = optimizer.run(data=self._data)

                    x = Explain(best, rest)
                    rule, _ = x.xpln(self._data, best, rest)

                    if rule == -1:
                        continue

                    xpln_data = Data.clone(self._data, selects(rule, self._data.rows))

                    self._results[o_name].append(best)
                    self._results[o_name + "_xpln"].append(xpln_data)

                    self._n_evals[o_name] += evals
                    self._n_evals[o_name + "_xpln"] += evals

                    if o_name == "sway":
                        top2, _ = self._data.betters(len(best.rows))
                        top = Data.clone(self._data, top2)

                        self._results['top'].append(top)
                        self._n_evals["top"] += len(self._data.rows)

            self._update_comparisons(i)

            i += 1

    def _update_comparisons(self, iter_: int):
        for i in range(len(self._comparisons)):
            [base, diff], result = self._comparisons[i]

            if result is None:
                self._comparisons[i][1] = ["=" for _ in range(len(self._data.cols.y))]

            for k in range(len(self._data.cols.y)):
                if self._comparisons[i][1][k] == "=":
                    base_y, diff_y = self._results[base][iter_].cols.y[k], self._results[diff][iter_].cols.y[k]
                    equals = bootstrap(base_y.has(), diff_y.has()) and cliffs_delta(base_y.has(), diff_y.has())

                    if not equals:
                        if i == 0:
                            print("WARNING: all to all {} {} {}".format(i, k, "false"))
                            print(f"all to all comparison failed for {self._results[base][iter_].cols.y[k].txt}")

                        self._comparisons[i][1][k] = "≠"

    def print_table(self, color: bool):
        headers = [y.txt for y in self._data.cols.y]
        table = []

        for k, v in self._results.items():
            # set the row equal to the average stats
            stats = get_stats(v)
            stats_list = [k] + [stats[y] for y in headers]

            # adds on the average number of evals
            stats_list += [self._n_evals[k] / self._n_iters]

            table.append(stats_list)

        if color:
            for i in range(len(headers)):
                # get the value of the 'y[i]' column for each algorithm
                header_vals = [v[i + 1] for v in table]

                # if the 'y' value is minimizing, use min else use max
                fun = max if headers[i][-1] == "+" else min

                # change the table to have green text if it is the "best" for that column
                table[header_vals.index(fun(header_vals))][i + 1] = '\033[92m' + str(
                    table[header_vals.index(fun(header_vals))][i + 1]) + '\033[0m'

        print(tabulate(table, headers=headers + ["Avg evals"], numalign="right"))
        print()

        # generates the =/!= table
        table = []

        # for each comparison of the algorithms
        #    append the = / !=
        for [base, diff], result in self._comparisons:
            table.append([f"{base} to {diff}"] + result)

        print(tabulate(table, headers=headers, numalign="right"))


def get_stats(data_array):
    # gets the average stats, given the data array objects
    res = {}

    # accumulate the stats
    for item in data_array:
        stats = item.stats()

        # update the stats
        for k, v in stats.items():
            res[k] = res.get(k, 0) + v

    # right now, the stats are summed. change it to average
    for k, v in res.items():
        res[k] /= options["Niter"]

    return res


def main():
    """
    `main` runs each algorithm for 20 iterations, on the given file dataset.

    It accumulates the results per each iteration, and compares the algorithms
    using cliffsDelta and bootstrap

    It then produces summatory stats, including a mean table (for each algorithm,
    summarize each y column and number of iterations)
    And a table comparing each algorithm to each other using cliffsDelta and bootstrap
    """

    options.parse_cli_settings(help_)

    if options["help"]:
        print(help_)
    else:
        optimizers = {
            "sway": SwayOptimizer(
                reuse=options["reuse"],
                far=options["Far"],
                halves=options["Halves"],
                rest=options["Rest"],
                i_min=options["IMin"]
            ),
            "sway_pca": SwayWithPCAOptimizer(
                reuse=options["reuse"],
                far=options["Far"],
                halves=options["Halves"],
                rest=options["Rest"],
                i_min=options["IMin"]
            ),
        }

        rg = ResultsGenerator(data_src=options["file"], optimizers=optimizers)
        rg.run()

        rg.print_table(color=options["wColor"])


main()
