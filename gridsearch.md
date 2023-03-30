# explanation of gridsearch


## Overview

- One thing we could do for sway2 is to do hyperparameter tuning.
- To do this, I got all combinations of hyperparameters, then performed sway
- I did hyperparameter tuning for auto2.csv, then applied those parameters to all other algorithms
- the result is sway2


## In depth

Steps of how it does it:

- I noted all of the parameters that are used when data.sway is called ("Far", "Halves", "Min", "Max", "P", "Rest", "reuse")
- I then created a dictionary of array of values these could take on (had to redo this multiple times to optimize for space/time)
- I then got all combinations of these parameters, and created a data object with it
- I performed sway on this data:
    - I passed in method="gs", so the betters function would be called as `gridsearch_better`



### sway

- It performed sway on the data, but the differences were:
    - instead of calling Data.better, it called Data.gridsearch_better
    - for every iteration, the number of evals were equal to the # of evals of sway of each hyperparameter it chose
- Sway is now doing sway over combinations of hyperparameters
- an "evaluation" is creating a new Data object with the `auto2.csv` file, calling sway, then the `y` value is the stats on the `better` half
- Data object now has a self.options, to support the gridsearch being able to customize the parameters


### gridsearch_better

- Takes in 2 rows of hyperparameters, to test which of those are "better"
    - creates a data object with `auto2.csv`, and performs sway with the hyperparameters of `row1`
    - creates a row object equal to the `best.stats()`
    - Does the same for `row2`
    - returns the better of those two rows
- increments self.gs_evals as the number of evals taken by sway of both hyperparameters
- in sway, returns both the number of evals taken by each individual data.sway, and the number of evals taken by the gridsearch.sway
