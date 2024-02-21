# Nonnegative Matrix Factorization (NMF) for Time Series Forecasting

This github repository contains the codes and scripts to generate the results of the numerical experiments reported in [1], where we introduce a novel methodologyto forecast several time series with non-negative and possibly missing entries, based on low-rank decompositions and matrix completion. We refer the interest reader to [1], for the theoretical analysis of the algorithms, referring in particular to statistical guarantees on uniqueness and robustness of the solutions. We implemented accelerated PALM [2] for Masked AMF and accelerated HALS for Masked NNMF [4]. The code is partially based on the one developed by H. Javadi and A. Montanari, see [5], contained in the Python file ``NMF.py`` and the implementation of the bechmarks is based on https://machinelearningmastery.com/random-forest-for-time-series-forecasting/. 

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Code

``Experiments.ipynb`` is a Python notebook explaining the code implemented for daily electricity consumption data-sets of 370 Portuguese customers during the period 2011-2014, see [6]. In order to obtain the numerical results reported in the directory ``results`` for the real-world and synthetic data-sets contained in the directory ``data``: 

```
./allrun_main.sh
```

## Bibliography

[1] Y. De Castro and L. Mencarelli, Time series prediction from partial observations via Nonnegative Matrix Factorization, 2024.

[2] A. Trindade. ElectricityLoadDiagrams20112014. UCI Machine Learning Repository, 2015. DOI: https://doi.org/10.24432/C58C86.

[3] J. Bolte, S. Sabach, and M. Teboulle. Proximal alternating linearized minimization for nonconvex and nonsmooth problems. Mathematical Programming, 146(1–2):459–494, 2014.

[4] N. Gillis and F. Glineur. Accelerated multiplicative updates and hierarchical ALS algorithms for nonnegative matrix factorization. Neural Computation, 24(4):1085–1105, 2012. doi: 10.1162/ NECO\_a\_00256. URL https://doi.org/10.1162/NECO_a_00256.

[5] H. Javadi and A. Montanari. Nonnegative matrix factorization via archetypal analysis. Journal of the American Statistical Association, 115(530):896–907, 2020.

[6] A. Trindade. ElectricityLoadDiagrams20112014. UCI Machine Learning Repository, 2015. DOI: https://doi.org/10.24432/C58C86.

