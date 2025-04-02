"""
Common functions for basic statistics.
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, skew, kurtosis, levene, f_oneway
import pingouin as pg


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

def one_way_or_Nway_anova(data=None, dv=None, between=None, ss_type=2, detailed=False, effsize='np2'):
    """
    Performs a one-way or N-way ANOVA using pingouin.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        dv (str): Name of the column containing the dependent variable.
        between (str or list): Name of the column(s) containing the
                               between-subject factor(s).
        ss_type (int): How sum of squares is calculated for unbalanced designs
                       (1, 2, or 3). Default is 2.
        detailed (bool): If True, returns a detailed ANOVA table.
                         Default is False for one-way, True for N-way.
        effsize (str): Effect size to compute ('np2' for partial eta-squared
                       or 'n2' for eta-squared). Default is 'np2'.

    Returns:
        pd.DataFrame: ANOVA summary table.

    Raises:
        ValueError: If input parameters are invalid or data is unsuitable.
        ImportError: If statsmodels is required but not installed for certain
                     ANOVA types (e.g., 3+ factors, unbalanced 2-way).

    Notes:
        - Missing values are automatically removed by pingouin.anova.
        - For unbalanced designs with 2+ factors or designs with 3+ factors,
          pingouin internally uses statsmodels.
        - See pingouin.anova documentation for more details.
    """
    # Basic validation (can be expanded)
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    if dv is None or dv not in data.columns:
        raise ValueError(f"Dependent variable '{dv}' not found in DataFrame columns.")
    if between is None:
        raise ValueError("'between' factor(s) must be specified.")

    # Check if between columns exist in the DataFrame
    if isinstance(between, str):
        if between not in data.columns:
            raise ValueError(f"Between factor '{between}' not found in DataFrame columns.")
    elif isinstance(between, list):
        for factor in between:
            if factor not in data.columns:
                raise ValueError(f"Between factor '{factor}' not found in DataFrame columns.")
    else:
        raise ValueError("'between' must be a string or a list of strings.")

    # Call pingouin's anova function
    try:
        aov_table = pg.anova(
            data=data,
            dv=dv,
            between=between,
            ss_type=ss_type,
            detailed=detailed, # pingouin handles default based on N-way
            effsize=effsize
        )
        return aov_table
    except Exception as e:
        # Re-raise the exception for clarity or handle specific pingouin/statsmodels errors
        print(f"An error occurred during ANOVA calculation: {e}")
        raise









