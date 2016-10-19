# GGS
Greedy Gaussian Segmentation (GGS) is a Python solver for efficiently segmenting multivariate time series data. For implementation details, please see our paper at [http://stanford.edu/~boyd/papers/ggs.html](http://stanford.edu/~boyd/papers/ggs.html).

----

The GGS Solver takes a T-by-n data matrix and breaks the T timestamps on an n-dimensional vector into segments over which the data is well explained as indepedent samples from a multivariate Gaussian distribution. It does so by formulating a covariance-regularized maximum likelihood problem and solving it using a greedy heuristic, with full details described in the [paper](http://stanford.edu/~boyd/papers/ggs.html).

This package has three main functions:

```
RunGGS(data, K, lamb, features = [], verbose = False)
```

Finds K breakpoints in the data for a given regularization parameter lambda

----


```
FindHyperparams(data, Kmax=25, lambList = [0.1, 1, 10], features = [], verbose = False)
```

Runs 10-fold cross validation, and returns the train and test set likelihood for every (K, lambda) pair up to Kmax.

----

```
FindMeanCovs(data, breakpoints, lamb, features = [], verbose = False)
```
Finds the means and regularized covariances of each segment, given a set of breakpoints

----









References
==========
[Greedy Gaussian Segmentation of Time Series Data -- D. Hallac, P. Nystrup, and S. Boyd][ggs]



[ggs]: http://stanford.edu/~boyd/papers/ggs.html "Greedy Gaussian Segmentation of Time Series Data -- D. Hallac, P. Nystrup, and S. Boyd"


Authors
------
David Hallac, Peter Nystrup, and Stephen Boyd.



