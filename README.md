# CSC591-Group9-Project


## How to run

- Install python atleast 3.9
- Install requirements: `pip install -r requirements.txt`
- Run `cd src`
- Run `python main.py` to generate tables for the `auto2.csv` file
- Run `python main.py --help` to view possible configuration values

```
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
```

## Structure

- Data files are located at [data](data)
- Output files are located at [out](out) and are named using the csv file name
- Python files are located at [src](src)
- A [script](generate_out.sh) is used to generate all output. It iterates over all data files, runs them through our program, and concatenates output to the output directory


## Description of algorithms

### main.py

`main.py` compares each algorithm over 20 iterations.

To produce the top table:
- For the given number of iterations:
    - run data.read to get the "all" result
    - run data.sway to get the "sway" result
    - run the explain algorithm, from HW6, to get the "xpln" result
    - run the betters algorithm, from previous homeworks, to get the "top" result
    - saves all of these data's into a table
- after all iterations complete, for each algorithm:
    - combine the stats. This is done by summing up each mean for each column, and dividing by the number of iterations

Example: `auto2.csv`, which is outputted to [auto2.out](out/auto2.out)

```
        CityMPG+    HighwayMPG+    Weight-    Class-
----  ----------  -------------  ---------  --------
all           21             28       3040      17.7
sway        30.2             35       2112     8.745
xpln        29.5           33.6    2281.75     9.795
top           33             40     2052.5       8.9

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
TODO
### xpln2
TODO
