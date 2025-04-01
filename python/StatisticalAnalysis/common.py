"""
Common functions for basic statistics.
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, skew, kurtosis, levene, f_oneway


# Functions
def normality_test(df):
    """
    This function performs a normality test on the data.
    """
    # Ignore nan values
    df = df.dropna()
    return shapiro(df)

def skew_test(df):
    """
    This function performs a skew test on the data.
    """
    # Ignore nan values
    df = df.dropna()
    return skew(df)

def kurtosis_test(df):
    """
    This function performs a kurtosis test on the data.
    """
    # Ignore nan values
    df = df.dropna()
    return kurtosis(df)

def homogeneity_test(group_A, group_B, group_C):
    """
    This function performs a homogeneity test on the data.
    """
    # Ignore nan values
    group_A = group_A.dropna()
    group_B = group_B.dropna()
    group_C = group_C.dropna()

    stat, p = levene(group_A, group_B, group_C)

    return stat, p

def one_way_anova(group_A, group_B, group_C):
    """
    This function performs a one-way ANOVA on the data.
    """
    # Ignore nan values
    group_A = group_A.dropna()
    group_B = group_B.dropna()
    group_C = group_C.dropna()

    stat, p = f_oneway(group_A, group_B, group_C)

    return stat, p










