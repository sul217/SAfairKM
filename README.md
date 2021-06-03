# A Stochastic Alternating Balance <img src="https://latex.codecogs.com/svg.latex?\Large&space;k" title="\Large k"/>-Means Algorithm for Fair Clustering


This is the numerical implementation for a recent work **A Stochastic Alternating Balance <img src="https://latex.codecogs.com/svg.latex?\Large&space;k" title="\Large k"/>-Means Algorithm for Fair Clustering**.  

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
We designed and implemented a novel stochastic alternating balance fair <img src="https://latex.codecogs.com/svg.latex?\large&space;k" title="\large k"/>-means (SAfairKM) algorithm, inspired from the classical mini-batch <img src="https://latex.codecogs.com/svg.latex?\large&space;k" title="\large k"/>-means algorithm, which essentially consists of alternatively taking pure mini-batch $k$-means updates and swap-based balance improvement updates. Then, we frame it to a Pareto front version in order to construct a good approximation of the entire Pareto fronts defining the best trade-offs between fair $k$-means cost and balance.


## 3. Files description

`SAfairKM.ipynb `: demonstrates how to run the proposed algorithm for the given synthetic and real datasets.

`utils.py`: contains all the necessary functions and class for the algorithm implementation. 

The data folder includes two real datasets: Adult and Bank datasets. 

## 4. Examples

<img src="data/Comparison_Pareto_front_dataSynthetic2-equal_numpts592.pdf" width="600px" style="float: right;">
