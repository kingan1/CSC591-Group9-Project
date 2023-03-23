
from explain import Explain, show_rule, selects
from data import Data
from options import options
from discretization import bins, value
from num import Num
from sym import Sym
from utils import adds, set_seed, rint, rand, rnd, csv, cliffsDelta, showTree, diffs

help = """

xpln: multi-goal semi-supervised explanation
(c) Group 9
  
USAGE: python3 main.py [OPTIONS] [-g ACTIONS]
  
OPTIONS:
  -b  --bins    initial number of bins       = 16
  -c  --cliffs  cliff's delta threshold      = .147
  -d  --d       different is over sd*d       = .35
  -f  --file    data file                    = ../data/auto93.csv
  -F  --Far     distance to distant          = .95
  -g  --go      start-up action              = nothing
  -h  --help    show help                    = false
  -H  --Halves  search space for clustering  = 512
  -m  --min     size of smallest cluster     = .5
  -M  --Max     numbers                      = 512
  -p  --p       dist coefficient             = 2
  -r  --rest    how many of rest to sample   = 4
  -R  --Reuse   child splits reuse a parent pole = true
  -s  --seed    random number seed           = 937162211
"""


def main(saved=None, fails=None):
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
        
        results = {"all": {}, "sway": {}, "xpln": {}, "top": {}}
        n_iter = 20
        count=0
        while count < n_iter:
            data=Data()
            data.read(options['file'])
            best,rest,evals = data.sway()
            x = Explain(best, rest)
            rule,most= x.xpln(data,best,rest)
            if rule != -1:
                data1= Data()
                data1.read(data,selects(rule,data.rows))
                for k,v in data.stats().items():
                    results['all'][k] = results['all'].get(k, 0) + v

                for k,v in best.stats().items():
                    results['sway'][k] = results['sway'].get(k, 0) + v

                for k,v in data1.stats().items():
                    results['xpln'][k] = results['xpln'].get(k, 0) + v
                    

                top2,_ = data.betters(len(best.rows))
                top = Data()
                top.read(data,top2)
                
                for k,v in top.stats().items():
                    results['top'][k] = results['top'].get(k, 0) + v

                top2,_ = data.betters(len(best.rows))
                top = Data()
                top.read(data,top2)
                count += 1
        
        for k,v in results.items():
            for k2,_ in results[k].items():
                results[k][k2] /= n_iter
            print(k, v)
main()
