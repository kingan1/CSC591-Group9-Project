from tabulate import tabulate

from data import Data
from explain import Explain, selects
from models.optimizers import SwayOptimizer, SwayHyperparameterOptimizer
from models.optimizers.dtree import DtreeOptimizer
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
  -w  --wColor      output with color                = false
  -s  --sway2       refresh the sway2 parameters     = false
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


def mean(lst):
    return sum(lst) / len(lst)

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
        results = {"all": [], "sway1": [], "sway2": [], "xpln1": [], "xpln2": [], "top": []}
        n_evals = {"all": 0, "sway1": 0, "sway2": 0, "xpln1": 0, "xpln2": 0, "top": 0}
        comparisons = [[["all", "all"],None], 
                       [["all", "sway1"],None], 
                       [["all", "sway2"],None],
                       [["sway1", "sway2"],None],  
                       [["sway1", "xpln1"],None],   
                       [["sway2", "xpln2"],None], 
                       [["sway1", "top"],None]]
        ranks = {"all": 0, "sway1": 0, "sway2": 0, "xpln1": 0, "xpln2": 0, "top": 0}

        count = 0
        # sway2_options = explore_parameters()
        data = None

        # read in the data
        data = Data(options["file"])

        # initialize the sway hyperparameter optimizer
        swayHyperparameter = SwayHyperparameterOptimizer(
                    reuse=options["reuse"],
                    far=options["Far"],
                    halves=options["Halves"],
                    rest=options["Rest"],
                    i_min=options["IMin"],
                    file=options["file"],
                    sway2=options["sway2"]
                )
        
        # get the "top" results by running the betters algorithm
        all_ranked, _ = data.betters(len(data.rows))
        # for each row, rank it normalized from 1-100
        for idx, row in enumerate(all_ranked):
            row.rank = 1 + (idx/len(data.rows))*99
            
        

        # do a while loop because sometimes explain can return -1
        while count < options["Niter"]:

            # get the "all" and "sway" results
            best, rest, evals_sway = SwayOptimizer(
                reuse=options["reuse"],
                far=options["Far"],
                halves=options["Halves"],
                rest=options["Rest"],
                i_min=options["IMin"]
            ).run(data)

            # get the "xpln" results
            x = Explain(best, rest)
            rule, _ = x.xpln(data, best, rest)

            # if it was able to find a rule
            if rule != -1:
                best_xpln2, _, _ = DtreeOptimizer(best=best, rest=rest).run(data)
                xpln2 = Data.clone(data, best_xpln2)
                # get the best rows of that rule
                data1 = Data.clone(data, selects(rule, data.rows))


                best2, _, evals_sway2 = swayHyperparameter.run(data)

                top2, _ = data.betters(len(best.rows))
                top = Data.clone(data, top2)

                results['all'].append(data)
                results['sway1'].append(best)
                results['xpln1'].append(data1)
                results['xpln2'].append(xpln2)
                results['top'].append(top)
                results['sway2'].append(best2)

                
                ranks['all'] += (mean([r.rank for r in data.rows]))
                ranks['sway1'] += (mean([r.rank for r in best.rows]))
                ranks['xpln1'] +=(mean([r.rank for r in data1.rows]))
                ranks['xpln2']+=(mean([r.rank for r in xpln2.rows]))
                ranks['sway2']+=(mean([r.rank for r in best2.rows]))
                ranks['top']+=(mean([r.rank for r in top.rows]))

                
                
                

                # accumulate the number of evals
                # for all: 0 evaluations 
                n_evals["all"] += 0
                n_evals["sway1"] += evals_sway
                n_evals["sway2"] += evals_sway2

                # xpln uses the same number of evals since it just uses the data from
                # sway to generate rules, no extra evals needed
                n_evals["xpln1"] += evals_sway
                n_evals["xpln2"] += evals_sway
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
                        if True: #comparisons[i][1][k] == "=":
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

            # adds on average rank of rows
            stats_list += [ranks[k] / options["Niter"]]

            table.append(stats_list)

        
        # generates the best algorithm/beat sway table
        maxes = []
        # each algorithm
        h = [v[0] for v in table]
        for i in range(len(headers)):
            # get the value of the 'y[i]' column for each algorithm
            header_vals = [v[i+1] for v in table]
            # if the 'y' value is minimizing, use min else use max
            fun = max if headers[i][-1] == "+" else min
            # vals is sway's result for y[i] and sway2's result for y[i]
            # used to say if our sway2 algorithm is better than sway
            vals = [table[h.index("sway1")][i+1],table[h.index("sway2")][i+1]]

            
            vals_x = [table[h.index("xpln1")][i+1],table[h.index("xpln2")][i+1]]
            # appends [y column name, 
            #          what algorithm is the best for that y, 
            #          if sway2 better than sway2]
            maxes.append([headers[i],
                          table[header_vals.index(fun(header_vals))][0],
                           vals.index(fun(vals)) == 1,
                           vals_x.index(fun(vals_x)) == 1])

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

        print(tabulate(table, headers=headers + ["Avg evals", "Avg rank"], numalign="right"))
        print()

            
        m_headers = ["Best", "Beat Sway?", "Beat Xpln?"]
        print(tabulate(maxes, headers=m_headers,numalign="right"))
        print()
        
        # generates the =/!= table
        table=[]
        # for each comparison of the algorithms
        #    append the = / !=
        for [base, diff], result in comparisons:
            table.append([f"{base} to {diff}"] + result)

        print(tabulate(table, headers=headers, numalign="right"))


main()
