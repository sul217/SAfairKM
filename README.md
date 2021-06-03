# A Stochastic Alternating Balance $'k'$-Means Algorithm for Fair Clustering

This is the numerical implementation for a recent work **A Stochastic Alternating Balance $'k'$-Means Algorithm for Fair Clustering**.  

## 1. Package requirements

The code was implemented using Python 3.6
- numpy
- collections
- time
- matplotlib
- random
- scipy
- sklearn
- multiprocessing


## 2. Project goal
We designed and implemented a novel stochastic alternating balance fair $k$-means (SAfairKM) algorithm, inspired from the classical mini-batch $k$-means algorithm, which essentially consists of alternatively taking pure mini-batch $k$-means updates and swap-based balance improvement updates. Then, we frame it to a Pareto front version in order to construct a good approximation of the entire Pareto fronts defining the best trade-offs between fair $k$-means cost and balance.


## 3. Files description

`SAfairKM.ipynb `: demonstrates how to run the proposed algorithm for the given synthetic and real datasets.

`utils.py`: contains all the necessary functions and class for the algorithm implementation. 

The 'data' folder includes two real datasets--- Adult and Bank datasets. 

## 5. Examples
The figure below shows the full trade-off between accuracy and fairness w.r.t. disparate impact using Adult income dataset and taking gender as the sensitive attribute. $f_1(x)$ and $f_2(x)$ refer to prediction loss and squared convariance approximation for disparate impact. 

