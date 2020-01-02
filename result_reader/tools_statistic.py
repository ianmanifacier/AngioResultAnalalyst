#!/usr/bin/python
# coding: utf-8

import numpy as np

# generate gaussian data
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import kstest
from scipy.stats import randint

def isThisDistributionFlat(data, alpha=0.05, details=False):
    """
    source : https://stackoverflow.com/questions/22392562/how-can-check-the-distribution-of-a-variable-in-python
    """
    lowerbound = 0
    upperbound = 10
    kstest(data, randint.cdf, args=(lowerbound,upperbound))
    """
    #args is a tuple containing the extra parameter required by ss.randint.cdf, in this case, lower bound and upper bound

    cdf: The cdf parameter perform the Kolmogorov-Smirnov test for goodness of fit.
    This performs a test of the distribution F(x) of an observed
    random variable against a given distribution G(x). Under the null
    hypothesis the two distributions are identical, F(x)=G(x). The
    alternative hypothesis can be either 'two-sided' (default), 'less'
    or 'greater'. The KS test is only valid for continuous distributions.
    Parameters"""


def ShapiroWilkTestOfNormality(data, alpha=0.05, details=False):
    """ The Shapiro-Wilk test evaluates a data sample and quantifies how likely it is that the data
    was drawn from a Gaussian distribution, named for Samuel Shapiro and Martin Wilk.
    In practice, the Shapiro-Wilk test is believed to be a reliable test of normality,
    although there is some suggestion that the test may be suitable for smaller samples
    of data, e.g. thousands of observations or fewer.
    
    - Statistic: A quantity calculated by the test that can be interpreted in the context of the test via
    comparing it to critical values from the distribution of the test statistic.
    - p-value: Used to interpret the test, in this case whether the sample was drawn from a Gaussian distribution.
    
    - details: if true details about the 

    source : https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    """
    # normality test
    stat, p = shapiro(data)
    if details:
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    if p > alpha: # p > alpha: fail to reject H0.
        if details:
            print("Sample looks Gaussian (normal) according to Shapiro test(fail to reject H0)")
        return True
    else: # p <= alpha: reject H0, not normal.
        if details:
            print('Sample does not look Gaussian (reject H0)')
        return False


def AgostinoTestOfNormality(data, alpha=0.05, details=False):
    """
    The D’Agostino’s K^2 test calculates summary statistics from the data, namely kurtosis and skewness, to determine if the data distribution departs from the normal distribution, named for Ralph D’Agostino.

    Skew is a quantification of how much a distribution is pushed left or right, a measure of asymmetry in the distribution.
    Kurtosis quantifies how much of the distribution is in the tail. It is a simple and commonly used statistical test for normality.
    
    source : https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    """
    # normality test
    stat, p = normaltest(data) # The D’Agostino’s K^2 test is available via the normaltest() SciPy function and returns the test statistic and the p-value.
    if details:
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        if details:
            print('Sample looks Gaussian (fail to reject H0)')
        return True
    else:
        if details:
            print('Sample does not look Gaussian (reject H0)')
        return False


def AndersonDarlingNormalityTest(data, alpha=0.05, details=False):
    result = anderson(data)
    if details:
        print('Statistic: %.3f' % result.statistic)
    dataLooksNormal = True
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            if details:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            if details:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
            dataLooksNormal = False
    return dataLooksNormal


def isThisDistributionNormal(data, alpha=0.05, details=False):
    """ This function performs a normality on the distribution, in other words, it
    test to see if the distribution is normal (Gaussian).
    source : https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

    We use 2 technics to check if data is Gaussian:
    Graphical Methods. These are methods for plotting the data and qualitatively evaluating whether the data looks Gaussian.
        - Statistical Tests. These are methods that calculate statistics on
        the data and quantify how likely it is that the data was drawn from a
        Gaussian distribution.
        - Statistical Tests. These are methods that calculate statistics on the
        data and quantify how likely it is that the data was drawn from a Gaussian distribution.
    Methods of this type are often called normality tests.
    """
    passShapiroWilkTest = ShapiroWilkTestOfNormality(data, alpha=alpha, details=details)
    passAgostinoK2Test = AgostinoTestOfNormality(data, alpha=alpha, details=details)
    passAndersonDarlingTest = AndersonDarlingNormalityTest(data, alpha=alpha, details=details)
    Gaussian = passShapiroWilkTest and passAgostinoK2Test and passAndersonDarlingTest

    if Gaussian:
        """ We use Parametric Statistical Methods:  """
        print("data is/looks Gaussian (no hard fail)")
    else:
        """ Use Nonparametric Statistical Methods """
        print("Data is non Gaussian")



# seed the random number generator
seed(1)
# generate univariate observations
data = 5 * randn(100) + 10
# summarize
print('mean=%.3f stdv=%.3f' % (mean(data), std(data)))
isThisDistributionNormal(data, alpha=0.05, details=True)