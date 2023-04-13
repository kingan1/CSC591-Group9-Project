from tabulate import tabulate

from data import Data
from explain import Explain, selects
from models.optimizers import SwayOptimizer, SwayWithPCAOptimizer
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
        results = {"all": [], "sway": [], "sway2": [], "xpln": [], "xpln2": [], "top": []}
        n_evals = {"all": 0, "sway": 0, "sway2": 0, "xpln": 0, "xpln2": 0, "top": 0}

        comparisons = [
            [["all", "all"], None],
            [["all", "sway"], None],
            [["all", "sway2"], None],
            [["sway", "sway2"], None],
            [["sway", "xpln"], None],
            [["sway2", "xpln"], None],
            [["sway", "top"], None]
        ]

        count = 0
        data = None

        # do a while loop because sometimes explain can return -1
        while count < options["Niter"]:
            # read in the data
            data = Data(options["file"])

            # get the "all" and "sway" results
            best, rest, evals_sway = SwayOptimizer(
                reuse=options["reuse"],
                far=options["Far"],
                halves=options["Halves"],
                rest=options["Rest"],
                i_min=options["IMin"]
            ).run(data)

            # get "sway2" results
            best2, rest2, evals_sway2 = SwayWithPCAOptimizer(
                reuse=options["reuse"],
                far=options["Far"],
                halves=options["Halves"],
                rest=options["Rest"],
                i_min=options["IMin"]
            ).run(data)

            # get the "xpln" results
            x = Explain(best, rest)
            rule, _ = x.xpln(data, best, rest)

            # get the "xpln2" results
            x2 = Explain(best2, rest2)
            rule2, _ = x2.xpln(data, best2, rest2)

            # if it was able to find a rule
            if rule != -1 and rule2 != -1:
                # get the best rows of that rule
                data1 = Data.clone(data, selects(rule, data.rows))
                data2 = Data.clone(data, selects(rule2, data.rows))

                results['all'].append(data)
                results['sway'].append(best)
                results['sway2'].append(best2)
                results['xpln'].append(data1)
                results['xpln2'].append(data2)

                # get the "top" results by running the betters algorithm
                top2, _ = data.betters(len(best.rows))
                top = Data.clone(data, top2)
                results['top'].append(top)

                # accumulate the number of evals
                # for all: 0 evaluations 
                n_evals["all"] += 0
                n_evals["sway"] += evals_sway
                n_evals["sway2"] += evals_sway2

                # xpln uses the same number of evals since it just uses the data from
                # sway to generate rules, no extra evals needed
                n_evals["xpln"] += evals_sway
                n_evals["xpln2"] += evals_sway2
                n_evals["top"] += len(data.rows)

                # update comparisons
                for i in range(len(comparisons)):
                    [base, diff], result = comparisons[i]

                    # if they haven't been initialized, mark with true until we can prove false

                    if result is None:
                        comparisons[i][1] = ["=" for _ in range(len(data.cols.y))]

                    # for each column
                    for k in range(len(data.cols.y)):
                        # if not already marked as false
                        if comparisons[i][1][k] == "=":
                            # check if it is false
                            base_y, diff_y = results[base][count].cols.y[k], results[diff][count].cols.y[k]
                            equals = bootstrap(base_y.has(), diff_y.has()) and cliffs_delta(base_y.has(), diff_y.has())

                            if not equals:
                                if i == 0:
                                    # should never fail for all to all, unless sample size is large
                                    print("WARNING: all to all {} {} {}".format(i, k, "false"))
                                    print(f"all to all comparison failed for {results[base][count].cols.y[k].txt}")

                                comparisons[i][1][k] = "≠"
                count += 1

        # generate the stats table
        headers = [y.txt for y in data.cols.y]
        table = []

        # for each algorithm's results
        for k, v in results.items():
            # set the row equal to the average stats
            stats = get_stats(v)
            stats_list = [k] + [stats[y] for y in headers]

            # adds on the average number of evals
            stats_list += [n_evals[k] / options["Niter"]]

            table.append(stats_list)

        if options["wColor"]:
            # updates stats table to have the best result per column highlighted
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
        for [base, diff], result in comparisons:
            table.append([f"{base} to {diff}"] + result)

        print(tabulate(table, headers=headers, numalign="right"))


main()
