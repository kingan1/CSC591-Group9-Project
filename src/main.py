
from explain import Explain, selects
from data import Data
from options import options
from stats import cliffsDelta, bootstrap
from tabulate import tabulate

help = """

project: multi-goal semi-supervised algorithms
(c) Group 9
  
USAGE: python3 main.py [OPTIONS] [-g ACTIONS]
  
OPTIONS:
  -b  --bins        initial number of bins           = 16
  -c  --cliff       cliff's delta threshold          = .147
  -d  --d           different is over sd*d           = .35
  -F  --Far         distance to distant              = .95
  -h  --help        show help                        = false
  -H  --Halves      search space for clustering      = 512
  -m  --min         size of smallest cluster         = .5
  -M  --Max         numbers                          = 512
  -p  --p           dist coefficient                 = 2
  -r  --rest        how many of rest to sample       = 4
  -R  --Reuse       child splits reuse a parent pole = true
  -x  --bootstrap   number of samples to bootstrap   = 512    
  -o  --conf        confidence interval              = 0.05
  -f  --file        file to generate table of        = ../data/auto2.csv
  -n  --niter       number of iterations to run      = 20
"""

def get_stats(data_array):
    res = {}
    # accumulate the stats
    for item in data_array:
        stats = item.stats()
        
        for k,v in stats.items():
            res[k] = res.get(k,0) + v

        
    for k,v in res.items():
        res[k] /= options["niter"]
    return res

def main():
    """
    `main` fills in the settings, updates them from the command line, runs
    the start up actions (and before each run, it resets the random number seed and settongs);
    and, finally, returns the number of test crashed to the operating system.

    :param funs: list of actions to run
    :param saved: dictionary to store options
    :param fails: number of failed functions
    """

    options.parse_cli_settings(help)


    if options["help"]:
        print(help)
    else:

        results = {"all": [], "sway": [], "xpln": [], "top": []}
        y_cols = Data(options["file"])
        headers = [y.txt for y in y_cols.cols.y]
        comparisons = [[["all", "all"],None], [["all", "sway"],None], [["sway", "xpln"],None], [["sway", "top"],None]]
        count = 0
        while count < options["niter"]:
            data=Data(options["file"])
            best,rest,evals = data.sway()
            x = Explain(best, rest)
            rule,most= x.xpln(data,best,rest)
            if rule != -1:
                data1= Data(data,selects(rule,data.rows))

                results['all'].append(data)
                results['sway'].append(best)
                results['xpln'].append(data1)
                    
                top2,_ = data.betters(len(best.rows))
                top = Data(data,top2)
                
                results['top'].append(top)
                # update comparisons
                for i in range(len(comparisons)):
                    [base, diff], result = comparisons[i]
                    # if they haven't been initialized, mark with true until we can prove false
                    if result == None:
                        comparisons[i][1] = ["=" for _ in range(len(data.cols.y))]
                    # for each column
                    for k in range(len(data.cols.y)):
                        # if not already marked as false
                        if comparisons[i][1][k] == "=":
                            # check if it is false
                            base_y, diff_y = results[base][count].cols.y[k],results[diff][count].cols.y[k]
                            equals = bootstrap(base_y.has(), diff_y.has()) and cliffsDelta(base_y.has(), diff_y.has())
                            if not equals:
                                comparisons[i][1][k] = "≠"
                count += 1

        table = []
        for k,v in results.items():
            stats = get_stats(v)
            stats_list = [k] + [stats[y] for y in headers]
            
            table.append(stats_list)
        print(tabulate(table, headers=headers,numalign="right"))
        print()


        table=[]
        for [base, diff], result in comparisons:
            table.append([f"{base} to {diff}"] + result)
        print(tabulate(table, headers=headers,numalign="right"))


main()
