# CSC591-Group9-Project


## How to run

- Install python atleast 3.9
- Install requirements: `pip install -r requirements.txt`
- Run `cd src`
- Run `python main.py` to generate tables for the `auto2.csv` file
- Run `python main.py --help` to view possible configuration values
- The hyperparameters for sway2 are optimized for `auto2.csv`. If you want to optimize them for another dataset, set the `--sway2 true` flag. This will run hyperparameter tuning on the given file, which will take more time

```
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
```

## Structure

- Data files are located at [data](data)
- Output files are located at [out](out) and are named using the csv file name
- Python files are located at [src](src)
- A [script](generate_out.sh) is used to generate all output. It iterates over all data files, runs them through our program, and concatenates output to the output directory
    - It does this in parallel to speed up times, so it will start all processes in the background, concatenating their output and how long the process takes to `out/data_file.out`


## Description of algorithms

### main.py

`main.py` compares each algorithm over 20 iterations.

To produce the top table:
- For the given number of iterations:
    - run data.read to get the "all" result
    - run data.sway to get the "sway" result
    - run sway with the optimal hyperparameters from `auto2.csv` to get the "sway2" result
    - run the explain algorithm, from HW6, to get the "xpln" result
    - run the betters algorithm, from previous homeworks, to get the "top" result
    - saves all of these data's into a table
- after all iterations complete, for each algorithm:
    - combine the stats. This is done by summing up each mean for each column, and dividing by the number of iterations
    - produces a table that for each `y` column, prints out the best algorithm for that column, and also if our `sway2` algorithm beats the `sway` algorithm
    - produces a table that performs bootstrap and cliffsdelta to determine if each combination of algorithms are considered equal
    - records how long it took for the algorithm to run

Example: `auto2.csv`, which is outputted to [auto2.out](out/auto2.out)

```
         CityMPG+    HighwayMPG+    Weight-    Class-
-----  ----------  -------------  ---------  --------
all            21             28       3040      17.7
sway        29.85          34.75     2110.5      8.82
sway2       39.15           43.9    1981.25     9.195
xpln         29.1           33.8       2261    10.525
top            33           40.6       2054      8.96

             Best    Beat Sway?
-----------  ------  ------------
CityMPG+     sway2   True
HighwayMPG+  sway2   True
Weight-      sway2   True
Class-       sway    False

               CityMPG+    HighwayMPG+    Weight-    Class-
-------------  ----------  -------------  ---------  --------
all to all     =           =              =          =
all to sway    ≠           ≠              ≠          ≠
all to sway2   ≠           ≠              ≠          ≠
sway to sway2  ≠           ≠              ≠          ≠
sway to xpln   ≠           ≠              ≠          ≠
sway2 to xpln  ≠           ≠              ≠          ≠
sway to top    ≠           ≠              ≠          ≠

real    1m44.796s
user    0m0.000s
sys     0m0.000s

```


To produce the bottom table:
- inside the same for loop as the main loop above::
    - after getting all of the results
    - for each comparison `base`,`diff` (all to all, all to sway, sway to xpln, sway to top)
        - originally "set" the comparison result to True, or equals, for each `y` column

        - for each `y` column, check if the `base` data is statistically different than the `diff` data
            - if it is not equal, set the comparison result for that y value to false

Example: `auto2.csv`, which is outputted to [auto2.out](out/auto2.out)

```
              CityMPG+    HighwayMPG+    Weight-    Class-
------------  ----------  -------------  ---------  --------
all to all    =           =              =          =
all to sway   ≠           ≠              ≠          ≠
sway to xpln  ≠           ≠              ≠          ≠
sway to top   ≠           ≠              ≠          ≠
```

### sway2

See [gridsearch](gridsearch.md)

### xpln2
TODO
