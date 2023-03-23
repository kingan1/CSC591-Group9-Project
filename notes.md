# notes from in class

	• Hes given sway1&xpln1,we write sway2&xpln2 (maybenotxpln2)
	• Hinting towards using PCA or svm
	• Xpln1 was learned from sway's results then applied to all data
	• Sklearn discretizer, clustering, rule based
	• Doesn't necessarily need to beat top
		○ Top is "upper bound", how close can we get to it
	• Can’t compare directly to table, the = and != are results of
		○ cliffDelta & bootstrap
	• Within table, mark "best" or "worst" with a color

Bonuses
	• February study
		○ Using a budget B in january, try it again in february with a lesser budget
	• Ablation
		○ Change/disable one idea from the model
	• HPO
		○ Hyperparameter optimization with minimal sampling method



Ideas
	• https://www.sciencedirect.com/science/article/pii/S0957417422000501


Improve sway/explain
	• Diff evaluation function
		○ Domination predicates: 
			§ euclidean distance to worst
		○ Boolean domination
			§ Do an initial bit of sampling for "easy" or "hard" goals to get, only focus on the easy or hard ones
			§ "camels back"
		○ Weight objectives in different ways
		○ February study
			§ Run sway once, run it again with a "seed" to sway
			§ Use the rules to select a subset, good way to prune large spaces
		○ Things we've learned from prior runs of sway

	• Diff hyperparameters
		○ Gridsearch
		○ Use sway to optimize sway
			§ Rn have global config, would have to pass around the config
		○ autoML, optuner, hyperopt
			§ Have expensive number of evaluations
			§ Check the number of evaluations

	• Cluster differently (ex: DBScan)
		○ DBScan is very good, assumes clusters are dense so clusters densest groups together
		○ In general sklearn is good for clustering
		○ https://dl.acm.org/doi/10.1145/1830483.1830575

	• For explain, create rules from multiple features
		○ Like using principal components, regression in high dimensional spaces
		

	• Alternative for zitzlers
		○ Tchebycheff, approach on penalty based methods
		○ MOEM/D

	• Find better way to split cluster in half ( right now an above point is passed in from previous rounds)
		○ Instead, find 2 distant points again instead of reusing it
		○ Andres thing 
			§ Not clear how to combine symbolic and numeric diversity
			§ Before sway, build the whole tree and find ways to remove high variance subtrees in X space
			§ Then pass that to sway
			§ Optimizing predictions for very small data sets 
				□ Tiny.cc/seai
		○ Sway is a recursive biclustering, so the clustering we need to do should also be recursive biclustering
			§ With dbscan, find the most dense cluster and say that’s the "best"
			§ Do smth in y space to work out and see if we want to use that or the lesss dense
			§ Need to combine clustering and pruning

	• Dimensionality reduction - scikits opticsClustering, dbscan, bruterforce
		○ Sway is a symbolic analogue to PCA
		○ You'll find names like CFS and information gain, normally they're supervised
			§ Aka blow our budget
		○ But we can discretize our Y variables into bins and use that for supervised learning
			§ One of the bins were "year < 1997 < 1993"
			§ Look at diversity in that bin
			§ Now we decide what attributes matter the most
			§ Then for feature selection do the ones with most diversity
		○ Or run sway and label the clusters "best" or "rest"
			§ Then do feature selection
		

	• Increase budget / B thresholds


For sway2
	• Input data
	• Output mean Y values seen across 20 runs
	• Our budget has been small, like 5 10 20 (REST)
	• See 



Questions
	1. Can we use decision tree for feature selection?
		a. You could Blow your evaluations budget
		b. Or, run sway once to get best and rest
Use that to figure out what features matter the most, weight it on distance