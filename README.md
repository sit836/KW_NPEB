# Kiefer-Wolfowitz Nonparametric Empirical Bayes

Compute the Kiefer-Wolfowitz nonparametric maximum likelihood estimator for mixtures.

This repository makes available the code in Python for the work of Koenker and Mizera (
2014): [Convex Optimization, Shape Constraints, Compound Decisions, And Empirical Bayes Rules](http://www.stat.ualberta.ca/~mizera/Preprints/brown.pdf)
.

## Making Predictions With No Features - A Basic Usage

Given a training set T = {y_i}, the algorithm provides a way to construct a predictor of future y-values such that the
sum of squared errors between observations and predictors is minimized.

## Getting Started

### Prerequisites

You will need:

* python (>= 3.6)
* pip (>= 19.0.3)
* MOSEK (>=8.1.30)

Important about MOSEK:

* MOSEK is a commercial optimization software. Please visit [MOSEK](https://www.mosek.com/) for license information.
* PIP:

```
pip install -f https://download.mosek.com/stable/wheel/index.html Mosek --user
``` 

For different ways of installation, please visit
their [installation page](https://docs.mosek.com/8.1/pythonapi/install-interface.html).

* MOSEK needs to be installed in the GLOBAL environment.

### Installing

```
pip install kwnpeb
```

## Examples

* [simple](https://github.com/sit836/KW_NPEB/tree/master/examples/simple) - The basic usage
* [bayesball](https://github.com/sit836/KW_NPEB/tree/master/examples/bayesball) - In-season prediction of batting
  averages with the 2005 Major League baseball

## Contributors

* [Sile Tao](https://ca.linkedin.com/in/sile-tao-95523941)
* [Li Zhang](https://ca.linkedin.com/in/li-zhang-0350833b)
* [Guanqi Huang](https://ca.linkedin.com/in/guanqi-huang)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
