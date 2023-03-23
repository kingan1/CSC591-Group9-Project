
from explain import Explain, selects
from data import Data
from options import options
from stats import cliffsDelta, bootstrap
data_file = "../data/auto2.csv"

help = """

xpln: multi-goal semi-supervised explanation
(c) Group 9
  
USAGE: python3 main.py [OPTIONS] [-g ACTIONS]
  
OPTIONS:
  -b  --bins    initial number of bins       = 16
  -c  --cliff  cliff's delta threshold      = .147
  -d  --d       different is over sd*d       = .35
  -F  --Far     distance to distant          = .95
  -h  --help    show help                    = false
  -H  --Halves  search space for clustering  = 512
  -m  --min     size of smallest cluster     = .5
  -M  --Max     numbers                      = 512
  -p  --p       dist coefficient             = 2
  -r  --rest    how many of rest to sample   = 4
  -R  --Reuse   child splits reuse a parent pole = true
  -s  --seed    random number seed           = 937162211
  -x  --bootstrap   number of samples to bootstrap   = 512    
  -o  --conf   confidence interval                   = 0.05
  -h  --cohen   cohen's D value                      = 0.35
"""
n_iter = 20

def get_stats(data_array):
    res = {}
    # accumulate the stats
    for item in data_array:
        stats = item.stats()
        
        for k,v in stats.items():
            res[k] = res.get(k,0) + v

        
    for k,v in res.items():
        res[k] /= n_iter
    return res

def get_equals(k, k2, results):
    # for each iteration, must equal
    for i in range(n_iter):
        k_data = results[k][i]
        k2_data = results[k2][i]
        # for each y column
        for k_y, k2_y in zip(k_data.cols.y, k2_data.cols.y):
            equals = bootstrap(k_y.has(), k2_y.has()) and cliffsDelta(k_y.has(), k2_y.has())
            if not equals:
                return False
    return True

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


    if options['help']:
        print(help)
    else:

        results = {"all": [], "sway": [], "xpln": [], "top": []}
        
        count=0
        while count < n_iter:
            data=Data()
            data.read(data_file)
            best,rest,evals = data.sway()
            x = Explain(best, rest)
            rule,most= x.xpln(data,best,rest)
            if rule != -1:
                data1= Data()
                data1.read(data,selects(rule,data.rows))
                results['all'].append(data)

                results['sway'].append(best)

                results['xpln'].append(data1)
                    
                top2,_ = data.betters(len(best.rows))
                top = Data()
                top.read(data,top2)
                
                results['top'].append(top)

                count += 1

        print("sanity check")
        for k,v in results.items():
            # results is an array of data, have to combine the stats
            stats = get_stats(v)
            print(f"{k} {stats}")

        comparisons = [["all", "all"], ["all", "sway"], ["sway", "xpln"], ["sway", "top"]]
        # for each pair of keys, do equals as bootstrap and cliffsdelta
        for k, k2 in comparisons:
            printable = f"{k} to {k2}"
            valid = True
            valid = get_equals(k,k2, results)
            printable += "=" if valid else "!="
            print(printable)
main()
