
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
  -b  --Bins        initial number of bins           = 16
  -c  --Cliff       cliff's delta threshold          = .147
  -d  --D           different is over sd*d           = .35
  -F  --Far         distance to distant              = .95
  -h  --help        show help                        = false
  -H  --Halves      search space for clustering      = 512
  -m  --Min         size of smallest cluster         = .5
  -M  --Max         numbers                          = 512
  -p  --P           dist coefficient                 = 2
  -R  --Rest        how many of rest to sample       = 4
  -r  --reuse       child splits reuse a parent pole = false
  -x  --Bootstrap   number of samples to bootstrap   = 512    
  -o  --Conf        confidence interval              = 0.05
  -f  --file        file to generate table of        = ../data/auto2.csv
  -n  --Niter       number of iterations to run      = 20
  -s  --sway2       refresh the sway2 parameters     = false
"""

from itertools import product

def explore_parameters():
    if options['sway2']:
        print("refreshing sway")
        # use steps to specify steps for each range of values
        steps = {"1000": 100,"100":10, '10': 1}
        # list of parameters used by sway, as well as example values to sample
        params = { 
            "Far":  [i/100 for i in range(70,100,steps["10"]*5)],
            "Halves":  [i for i in range(100, 600, steps["1000"])],
            "Min":  [i/10 for i in range(0,8,steps['10']*2)],
            "Max": [i for i in range(1, 150, 25)],
            "P":  [1+(i/10) for i in range(10)],
            "Rest":  [i for i in range(1,5)],
            "reuse":  [True,False], 
        }
        # types of each parameter
        types = { 
            "Far":  float,
            "Halves":  int,
            "Min":  float,
            "Max": int,
            "P":  int,
            "Rest":  int,
            "reuse":  bool
        }

        print(params)
        # get each combination of parameters
        permutations_dicts = [dict(zip(params.keys(), v)) for v in product(*params.values())]
        print(f"{len(permutations_dicts)} items")
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
        best,_,evals = data.sway(method="gs")
        print(f"{evals} evals")

        # set the hyperparameters as the "average" of the hyperparameters in best
        res = best.stats(best.cols.x)
        res.pop("N")
        res = {k: types[k](v) for k,v in res.items()}
        print("new: ", res)
        print()
        
        return get_options(res)
    
    # these are optimized for auto2.csv
    print("not refreshing sway")
    finalized = {'Far': 0.8, 'Halves': 700, 'Max': 1, 'Min': 0.0, 'P': 1, 'Rest': 4, 'reuse': False}
    return get_options(finalized)

def get_stats(data_array):
    res = {}
    # accumulate the stats
    for item in data_array:
        stats = item.stats()
        
        for k,v in stats.items():
            res[k] = res.get(k,0) + v

        
    for k,v in res.items():
        res[k] /= options["Niter"]
    return res

def get_options(new_options):
    global options
    options2 = options.t.copy()
    
    for k,v in new_options.items():
        options2[k] = v
    print("using options", options2)
    assert len(options2) == 16
    return options2

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

        results = {"all": [], "sway": [], "sway2": [], "xpln": [], "top": []}
        comparisons = [[["all", "all"],None], 
                       [["all", "sway"],None], 
                       [["all", "sway2"],None],
                       [["sway", "sway2"],None],  
                       [["sway", "xpln"],None],   
                       [["sway2", "xpln"],None], 
                       [["sway", "top"],None]]
        count = 0
        sway2_options = explore_parameters()
        data=None
        while count < options["Niter"]:
            data=Data(options["file"])
            best,rest,_ = data.sway()
            x = Explain(best, rest)
            rule,_= x.xpln(data,best,rest)
            if rule != -1:
                data1= Data(data,selects(rule,data.rows))

                results['all'].append(data)
                results['sway'].append(best)
                results['xpln'].append(data1)
                best2,_,_ = data.sway(options_new=sway2_options)
                results['sway2'].append(best2)
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

        # generate the stats table
        headers = [y.txt for y in data.cols.y]
        table = []
        # for each algorithm's results
        for k,v in results.items():
            # set the row equal to the average stats
            stats = get_stats(v)
            stats_list = [k] + [stats[y] for y in headers]
            
            table.append(stats_list)
        print(tabulate(table, headers=headers,numalign="right"))
        print()

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
            vals = [table[h.index("sway")][i+1],table[h.index("sway2")][i+1]]
            # appends [y column name, 
            #          what algorithm is the best for that y, 
            #          if sway2 better than sway2]
            maxes.append([headers[i],
                          table[header_vals.index(fun(header_vals))][0],
                           vals.index(fun(vals)) == 1])
            
        m_headers = ["Best", "Beat Sway?"]
        print(tabulate(maxes, headers=m_headers,numalign="right"))
        print()
        
        # generates the =/!= table
        table=[]
        # for each comparison of the algorithms
        #    append the = / !=
        for [base, diff], result in comparisons:
            table.append([f"{base} to {diff}"] + result)
        print(tabulate(table, headers=headers,numalign="right"))


main()
