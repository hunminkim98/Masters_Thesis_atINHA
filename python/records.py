"""
This file contains the code for the statistical analysis of the records.
"""

import pandas as pd
import numpy as np
from DataProcessing.readExcel import *
from DataProcessing.visualization import *
from StatisticalAnalysis.common import *

# Data directory path
data_dir = r"D:\석사\석사4차\Masters_Thesis_atINHA\data\10mRecords"

# Read all excels
all_data_dict = read_all_excels(data_dir)

# Split the data into two groups
group_A_df, group_B_df, group_C_df = split_data(all_data_dict)

# Split the data by timepoint
pre_A, post1_A, post2_A = split_by_timepoint(group_A_df)
pre_B, post1_B, post2_B = split_by_timepoint(group_B_df)
pre_C, post1_C, post2_C = split_by_timepoint(group_C_df)

# Normality test of Group A
print("\n" + "="*50 + "\nNormality Test of Group A:\n" + "="*50)
print(normality_test(pre_A))
print(normality_test(post1_A))
print(normality_test(post2_A))

# homogeneity test of Group A
print("\n" + "="*50 + "\nHomogeneity Test of Group A:\n" + "="*50)
stat, p = homogeneity_test(pre_A, post1_A, post2_A)
print(f"Statistic: {stat}, p-value: {p}")

# qq_plot(pre_A, title="Q-Q Plot of Group A Pre")
# qq_plot(post1_A, title="Q-Q Plot of Group A Post1")
# qq_plot(post2_A, title="Q-Q Plot of Group A Post2")

# Normality test of Group B
print("\n" + "="*50 + "\nNormality Test of Group B:\n" + "="*50)
print(normality_test(pre_B))
print(normality_test(post1_B))
print(normality_test(post2_B))

# homogeneity test of Group B
print("\n" + "="*50 + "\nHomogeneity Test of Group B:\n" + "="*50)
stat, p = homogeneity_test(pre_B, post1_B, post2_B)
print(f"Statistic: {stat}, p-value: {p}")

# qq_plot(pre_B, title="Q-Q Plot of Group B Pre")
# qq_plot(post1_B, title="Q-Q Plot of Group B Post1")
# qq_plot(post2_B, title="Q-Q Plot of Group B Post2")

# Normality test of Group C
print("\n" + "="*50 + "\nNormality Test of Group C:\n" + "="*50)
print(normality_test(pre_C))
print(normality_test(post1_C))
print(normality_test(post2_C))

# homogeneity test of Group C
print("\n" + "="*50 + "\nHomogeneity Test of Group C:\n" + "="*50)
stat, p = homogeneity_test(pre_C, post1_C, post2_C)
print(f"Statistic: {stat}, p-value: {p}")

# qq_plot(pre_C, title="Q-Q Plot of Group C Pre")
# qq_plot(post1_C, title="Q-Q Plot of Group C Post1")
# qq_plot(post2_C, title="Q-Q Plot of Group C Post2")

# Normality test of all groups
all_data = pd.concat([pre_A, post1_A, post2_A, pre_B, post1_B, post2_B, pre_C, post1_C, post2_C])

# One dimensional data
single_column = pd.Series(all_data.values.flatten())
single_column = single_column.dropna()  # NaN 값 제거
single_column = single_column.reset_index(drop=True)  # 인덱스 재설정

# Normality test of one dimensional data
print("\n" + "="*50 + "\nNormality Test of One Dimensional Data:\n" + "="*50)
print(normality_test(single_column))
qq_plot(single_column, title="Q-Q Plot of One Dimensional Data of Records")

# Skewness test of one dimensional data
# print("\n" + "="*50 + "\nSkewness Test of One Dimensional Data:\n" + "="*50)
# print(skew_test(single_column))
# density_plot(single_column, title="Density Plot of One Dimensional Data of Records")

# Kurtosis test of one dimensional data
# print("\n" + "="*50 + "\nKurtosis Test of One Dimensional Data:\n" + "="*50)
# print(kurtosis_test(single_column))

# Homogeneity test of all groups
GroupA = pd.concat([pre_A, post1_A, post2_A])
GroupA = pd.Series(GroupA.values.flatten())
GroupA = GroupA.dropna()
GroupA = GroupA.reset_index(drop=True)

GroupB = pd.concat([pre_B, post1_B, post2_B])
GroupB = pd.Series(GroupB.values.flatten())
GroupB = GroupB.dropna()
GroupB = GroupB.reset_index(drop=True)

GroupC = pd.concat([pre_C, post1_C, post2_C])
GroupC = pd.Series(GroupC.values.flatten())
GroupC = GroupC.dropna()
GroupC = GroupC.reset_index(drop=True)

print("\n" + "="*50 + "\nHomogeneity Test of All Groups:\n" + "="*50)
stat, p = homogeneity_test(GroupA, GroupB, GroupC)
print(f"Statistic: {stat}, p-value: {p}")

# One way ANOVA for baseline of all groups (Pre)
print("\n" + "="*50 + "\nOne Way ANOVA for Baseline of All Groups:\n" + "="*50)
stat, p = one_way_anova(pre_A, pre_B, pre_C)
print(f"Statistic: {stat}, p-value: {p}")



































