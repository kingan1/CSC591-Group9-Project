import os
import pandas as pd
from data.data import Data
from data.col.num import Num
from options import options
from stats import cliffs_delta, bootstrap
from tabulate import tabulate

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
  -s  --sway2       refresh the sway2 parameters     = false
"""
options.parse_cli_settings(help_)
df = []
for f1 in os.listdir("../data"):
    f = "../data/" + f1

    data = Data(f)

    # name | num rows | num x col | num y col
    sum_df = [f1.replace(".csv",""), len(data.rows), len(data.cols.x), len(data.cols.y)]
    df.append(sum_df)

print(tabulate(df, headers=["Dataset", "Number of rows", "Number of x", "Number of y"], numalign="right", tablefmt="latex"))


    


