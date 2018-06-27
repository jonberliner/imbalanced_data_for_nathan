# imbalanced_data_for_nathan
comparing different methods for imbalanced data with linear regression

requires sklearn, scipy, numpy, pandas, matplotlib
run example with synthetic data with `python main.py`
atm, need to manually rewrite main script to take in your real data instead of synthetic

## files
+ **main.py**: defines and runs experiments with different balancing methods
+ **balanced_samplers.py**: functions for sampling data to get class balance
+ **util.py**: contains function for cross validation
+ Hi Nathan.  Lebron is coming to Philly

## data
+ **nathan_rat_data.csv**
####
data shared on g-drive, columns are as follows:

1: brain region ID
2: mouse ID
3: neuron ID
4: neuron activity (au)
5: time and identity of lever press (1 = right, 2 = left)
6: time and identity of CS (1= positive, 2=negative)

Data is sampled at 10hz. You should probably z-score each neurons activity to normalize
