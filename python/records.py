"""
This file contains the code for the statistical analysis of the records.
"""

import pandas as pd
import numpy as np
from DataProcessing.readExcel import *
from DataProcessing.visualization import *
from StatisticalAnalysis.common import *
import pingouin as pg

#############################################
############ Data Preparation ###############
#############################################

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


#############################################
############# Normality Test ################
#############################################


# Normality test of Group A
print("\n" + "="*50 + "\nNormality Test of Group A:\n" + "="*50)
print(normality_test(pre_A))
print(normality_test(post1_A))
print(normality_test(post2_A))

# Normality test of Group B
print("\n" + "="*50 + "\nNormality Test of Group B:\n" + "="*50)
print(normality_test(pre_B))
print(normality_test(post1_B))
print(normality_test(post2_B))

# Normality test of Group C
print("\n" + "="*50 + "\nNormality Test of Group C:\n" + "="*50)
print(normality_test(pre_C))
print(normality_test(post1_C))
print(normality_test(post2_C))

# Normality test of all groups
all_data = pd.concat([pre_A, post1_A, post2_A, pre_B, post1_B, post2_B, pre_C, post1_C, post2_C])

# One dimensional data
single_column = pd.Series(all_data.values.flatten())
single_column = single_column.dropna()
single_column = single_column.reset_index(drop=True)

# Normality test
print("\n" + "="*50 + "\nNormality Test of One Dimensional Data:\n" + "="*50)
stat, p = normality_test(single_column)

# Save the result
output_path = "normality_test_result.csv"
normality_test_result = pd.DataFrame({'Statistic': [stat], 'p-value': [p]})
normality_test_result.to_csv(output_path, index=False)
print(f"Normality test result saved to {output_path}")

# qq_plot(single_column, title="Q-Q Plot of One Dimensional Data of Records")
# qq_plot_with_pingouin(single_column)

# qq_plot(pre_A, title="Q-Q Plot of Group A Pre")
# qq_plot(post1_A, title="Q-Q Plot of Group A Post1")
# qq_plot(post2_A, title="Q-Q Plot of Group A Post2")

# qq_plot(pre_B, title="Q-Q Plot of Group B Pre")
# qq_plot(post1_B, title="Q-Q Plot of Group B Post1")
# qq_plot(post2_B, title="Q-Q Plot of Group B Post2")

# qq_plot(pre_C, title="Q-Q Plot of Group C Pre")
# qq_plot(post1_C, title="Q-Q Plot of Group C Post1")
# qq_plot(post2_C, title="Q-Q Plot of Group C Post2")

# Skewness test of one dimensional data
# print("\n" + "="*50 + "\nSkewness Test of One Dimensional Data:\n" + "="*50)
# print(skew_test(single_column))
# density_plot(single_column, title="Density Plot of One Dimensional Data of Records")

# Kurtosis test of one dimensional data
# print("\n" + "="*50 + "\nKurtosis Test of One Dimensional Data:\n" + "="*50)
# print(kurtosis_test(single_column))

#############################################
############# Homogeneity Test #############
#############################################


# homogeneity test of Group A
print("\n" + "="*50 + "\nHomogeneity Test of Group A:\n" + "="*50)
stat, p = homogeneity_test(pre_A, post1_A, post2_A)
print(f"Statistic: {stat}, p-value: {p}")


# homogeneity test of Group B
print("\n" + "="*50 + "\nHomogeneity Test of Group B:\n" + "="*50)
stat, p = homogeneity_test(pre_B, post1_B, post2_B)
print(f"Statistic: {stat}, p-value: {p}")


# homogeneity test of Group C
print("\n" + "="*50 + "\nHomogeneity Test of Group C:\n" + "="*50)
stat, p = homogeneity_test(pre_C, post1_C, post2_C)
print(f"Statistic: {stat}, p-value: {p}")

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

# Save the result
output_path = "homogeneity_test_result.csv"
homogeneity_test_result = pd.DataFrame({'Statistic': [stat], 'p-value': [p]})
homogeneity_test_result.to_csv(output_path, index=False)
print(f"Homogeneity test result saved to {output_path}")

#############################################
####### One Way ANOVA for baseline ##########
#############################################


# One way ANOVA for baseline of all groups (Pre)
print("\n" + "="*50 + "\nOne Way ANOVA for Baseline of All Groups:\n" + "="*50)
stat, p = one_way_anova(pre_A, pre_B, pre_C)
print(f"Statistic: {stat}, p-value: {p}")


# One way ANOVA with Pingouin for baseline of all groups 
# Combine pre-test data into a single DataFrame with a group identifier
# Create DataFrame directly from pre_A, then assign columns
df_pre_A = pd.DataFrame(pre_A)
df_pre_A.columns = ['Score'] 
df_pre_A['Group'] = 'MAE'

df_pre_B = pd.DataFrame(pre_B)
df_pre_B.columns = ['Score']
df_pre_B['Group'] = 'DI'

df_pre_C = pd.DataFrame(pre_C)
df_pre_C.columns = ['Score']
df_pre_C['Group'] = 'Control'

combined_pre_data = pd.concat([df_pre_A, df_pre_B, df_pre_C], ignore_index=True)

print(f"Combined Pre-Data for ANOVA:\n{combined_pre_data.head()}\n...") # Print head for verification
print(f"Combined Pre-Data columns: {combined_pre_data.columns}")
print(f"Combined Pre-Data shape: {combined_pre_data.shape}")

aov = pg.anova(
    data=combined_pre_data,
    dv='Score',
    between='Group',
    detailed=True,
    effsize='np2'
)
print("\n" + "="*50 + "\nOne Way ANOVA with Pingouin for Baseline of All Groups:\n" + "="*50)
print(aov)

# Save the result
output_path = "one_way_anova_result.csv"
one_way_anova_result = pd.DataFrame({'Statistic': [stat], 'p-value': [p]})
one_way_anova_result.to_csv(output_path, index=False)
print(f"One way ANOVA result saved to {output_path}")

# Visualization of one way ANOVA results for baseline of all groups
anova_boxplot(
    dataframe=combined_pre_data,
    x_axis='Group',  # The grouping variable used in ANOVA
    y_axis='Score',  # The dependent variable used in ANOVA
    title='Baseline Score Distribution by Group', # A descriptive title
    palette='Set3' # Optional: specify a color palette
)


####################################################################
######### Sphericity Test for two-way mixed design ANOVA ###########
####################################################################

# Assuming group_A_df, group_B_df, group_C_df are loaded and have columns like 'Pre', 'Post1', 'Post2'
# If column names differ, adjust them in the 'value_vars' list below.

# --- Create Long-Format DataFrame ---

all_dfs_long = []
for df, group_name in zip([group_A_df, group_B_df, group_C_df], ['MAE', 'DI', 'Control']):
    # Add SubjectID (assuming index corresponds to subject within the group)
    df_processed = df.reset_index().rename(columns={'index': 'GroupSubjectID'})
    # Create a truly unique ID across groups
    df_processed['UniqueID'] = group_name + '_' + df_processed['GroupSubjectID'].astype(str)
    # Add Group Name
    df_processed['Group'] = group_name
    # Melt to long format
    # IMPORTANT: Replace ['Pre', 'Post1', 'Post2'] with the actual column names for timepoints in your DataFrames
    df_long = pd.melt(
        df_processed,
        id_vars=['UniqueID', 'Group', 'GroupSubjectID'], # Keep original ID if needed, use UniqueID as primary
        value_vars=['Pre', 'Post1', 'Post2'], # <-- ADJUST THESE NAMES IF NEEDED
        var_name='Time',
        value_name='Score'
    )
    all_dfs_long.append(df_long)

# Combine all groups
long_data = pd.concat(all_dfs_long, ignore_index=True)

# Drop rows with NaN scores, as sphericity test requires complete cases for each subject
long_data_complete = long_data.dropna(subset=['Score'])

# --- Debugging Sphericity Input ---
# Check how many subjects have data for all time points using the new UniqueID
subject_counts = long_data_complete.groupby('UniqueID')['Time'].nunique()
complete_subjects_ids = subject_counts[subject_counts == long_data_complete['Time'].nunique()].index
long_data_complete_subjects = long_data_complete[long_data_complete['UniqueID'].isin(complete_subjects_ids)]

# print("\n" + "="*50 + "\nDebugging Sphericity Input Data:\n" + "="*50)
# print(f"Original long_data shape: {long_data.shape}")
# print(f"Shape after dropna(subset=['Score']): {long_data_complete.shape}")
# print(f"Number of unique subjects (UniqueID) after dropna: {long_data_complete['UniqueID'].nunique()}")
# print(f"Number of time points being compared: {long_data_complete['Time'].nunique()}")
# print(f"Number of subjects with data for ALL time points: {len(complete_subjects_ids)}")

if len(complete_subjects_ids) > 1:
    print("Variance within each time point for subjects with complete data:")
    print(long_data_complete_subjects.groupby('Time')['Score'].var())
    
    # --- Check variance of differences ---
    print("\nCalculating variance of differences between time points...")
    try:
        # Pivot the data to wide format for easy difference calculation using UniqueID
        wide_complete = long_data_complete_subjects.pivot(index='UniqueID', columns='Time', values='Score')
        # Calculate differences (ensure correct order based on your time points)
        diff_post1_pre = wide_complete['Post1'] - wide_complete['Pre']
        diff_post2_pre = wide_complete['Post2'] - wide_complete['Pre']
        diff_post2_post1 = wide_complete['Post2'] - wide_complete['Post1']
        
        # print(f"Variance of (Post1 - Pre): {diff_post1_pre.var():.6f}")
        # print(f"Variance of (Post2 - Pre): {diff_post2_pre.var():.6f}")
        # print(f"Variance of (Post2 - Post1): {diff_post2_post1.var():.6f}")
        
        if diff_post1_pre.var() == 0 or diff_post2_pre.var() == 0 or diff_post2_post1.var() == 0:
            print("\n*** Warning: Variance of differences is zero for at least one pair. This causes Mauchly's test calculation issues. ***")
            print("    However, zero variance implies perfect sphericity, hence p=1.0 is the expected outcome.")

    except Exception as diff_err:
        print(f"Could not calculate variance of differences: {diff_err}")
    # --- End check variance of differences ---
        
else:
    print("Not enough subjects with complete data across all time points to calculate variance reliably.")
# Use the data with only complete subjects for the test
sphericity_input_data = long_data_complete_subjects 
# --- End Debugging ---

print("\n" + "="*50 + "\nLong-Format Data for Repeated Measures:\n" + "="*50)
print(sphericity_input_data) # Print all data
print(f"Shape of data used for sphericity test: {sphericity_input_data.shape}")




# Save the data for subsequent analysis
output_path = "sphericity_data.csv"
sphericity_input_data.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")


# --- Perform Sphericity Test ---
# We test the sphericity assumption for the within-subject factor 'Time'.

print("\n" + "="*50 + "\nSphericity Test (Mauchly's):\n" + "="*50)
# *** Use the filtered data and the correct UniqueID ***
# spher, W, chi2, dof, pval = sphericity_test(
#     data=sphericity_input_data, # <-- Use data with only complete subjects
#     dv='Score',      # Dependent variable column
#     within='Time',   # Within-subject factor column
#     subject='UniqueID' # Subject identifier column <-- Use UniqueID
spher, W, chi2, dof, pval = pg.sphericity(
    data=sphericity_input_data,
    dv='Score',
    within='Time',
    subject='UniqueID'
)
print(f"Sphericity assumption met: {spher}")
print(f"W: {W:.4f}")
print(f"chi2({dof}): {chi2:.4f}")
print(f"p-value: {pval:.4f}")

if not spher:
    print("\nNote: Sphericity assumption violated (p <= 0.05). Consider corrections (e.g., Greenhouse-Geisser) for ANOVA.")

# Save the result of sphericity test
output_path = "sphericity_test_result.csv"
sphericity_test_result = pd.DataFrame(
    {
        'Sphericity': [spher],
        'W': [W],
        'chi2': [chi2],
        'dof': [dof],
        'p-value': [pval]
    }
)
sphericity_test_result.to_csv(output_path, index=False)
print(f"Sphericity test result saved to {output_path}")

#######################################################################
################# Two-Way 3x3 Mixed Design ANOVA ######################
#######################################################################

aov = pg.mixed_anova(
    data=sphericity_input_data,
    dv='Score',
    between='Group',
    within='Time',
    subject='UniqueID',
    correction=False  # Greenhouse-Geisser correction
)
print("\n" + "="*50 + "\nTwo-Way 3x3 Mixed Design ANOVA:\n" + "="*50)
print(aov)


# Save the result of two-way mixed design ANOVA
output_path = "two_way_mixed_anova_result.csv"
two_way_mixed_anova_result = pd.DataFrame(aov)
two_way_mixed_anova_result.to_csv(output_path, index=False)
print(f"Two-way mixed design ANOVA result saved to {output_path}")

# Post-hoc tests based on ANOVA results
print("\n" + "="*50 + "\nPost-hoc Tests for Significant Effects:\n" + "="*50)

# Post-hoc for between-subject factor (Group)
posthoc_between = pg.pairwise_tukey(data=sphericity_input_data, dv='Score', between='Group')
print("\nPost-hoc (Tukey HSD) for between-subject factor (Group):\n")
print(posthoc_between)

# Post-hoc for within-subject factor (Time)
posthoc_within = pg.pairwise_tests(data=sphericity_input_data, dv='Score', within='Time', 
                                  subject='UniqueID', padjust='bonf', parametric=True)
print("\nPost-hoc (Paired t-tests with Bonferroni correction) for within-subject factor (Time):\n")
print(posthoc_within)

# Post-hoc for interaction effect: Separate analysis for each group
print("\nPost-hoc analysis for interaction: Separate repeated measures for each group:\n")
for group in sphericity_input_data['Group'].unique():
    group_data = sphericity_input_data[sphericity_input_data['Group'] == group]
    
    print(f"\n--- Group: {group} ---")
    # One-way repeated measures ANOVA for this group
    rm_anova = pg.rm_anova(data=group_data, dv='Score', within='Time', subject='UniqueID')
    print(rm_anova)
    
    # Save group RM-ANOVA results
    rm_anova.to_csv(f"rm_anova_{group}_group.csv", index=False)
    
    # If significant time effect in this group, run post-hoc
    if rm_anova.loc[0, 'p-unc'] < 0.05:
        posthoc = pg.pairwise_tests(data=group_data, dv='Score', within='Time', 
                                   subject='UniqueID', padjust='bonf', parametric=True)
        print("\nPost-hoc comparisons for this group:")
        print(posthoc)
        
        # Save group post-hoc results
        posthoc.to_csv(f"posthoc_within_time_{group}_group.csv", index=False)

# Save post-hoc test results for overall analysis
posthoc_between.to_csv("posthoc_between_groups.csv", index=False)
posthoc_within.to_csv("posthoc_within_time_overall.csv", index=False)
print("Post-hoc test results saved to CSV files")

# Post-hoc analysis for interaction: Compare groups at each time point
print("\n" + "="*50 + "\nInteraction Post-hoc: Comparing Groups at Each Time Point:\n" + "="*50)
interaction_posthoc_results = []

for time_point in sphericity_input_data['Time'].unique():
    print(f"\n--- Time Point: {time_point} ---")
    # Filter data for this time point
    time_data = sphericity_input_data[sphericity_input_data['Time'] == time_point]
    
    # One-way ANOVA for groups at this time point
    time_aov = pg.anova(data=time_data, dv='Score', between='Group', detailed=True)
    print(f"ANOVA for groups at {time_point}:")
    print(time_aov)
    
    # Save ANOVA results for this time point
    time_aov.to_csv(f"anova_groups_at_{time_point}.csv", index=False)
    
    # If significant group effect at this time point, run post-hoc
    if time_aov.loc[0, 'p-unc'] < 0.05:
        time_posthoc = pg.pairwise_tukey(data=time_data, dv='Score', between='Group')
        print(f"\nPost-hoc (Tukey HSD) for groups at {time_point}:")
        print(time_posthoc)
        
        # Save post-hoc results for this time point
        time_posthoc.to_csv(f"posthoc_groups_at_{time_point}.csv", index=False)
        
        # Collect results for summary
        time_posthoc['Time'] = time_point
        interaction_posthoc_results.append(time_posthoc)

# Combine all interaction post-hoc results if available
if interaction_posthoc_results:
    all_interaction_posthoc = pd.concat(interaction_posthoc_results, ignore_index=True)
    all_interaction_posthoc.to_csv("interaction_posthoc_all_timepoints.csv", index=False)
    print("\nInteraction post-hoc analysis results saved to CSV files")

# 상호작용 그래프 생성
mixed_anova_interaction_plot(
    data=sphericity_input_data, 
    dv='Score', 
    between='Group', 
    within='Time', 
    subject='UniqueID', 
    error_bars='sd',  # Standard Deviation
    title='Changes in Score Over Time by Group',
    aov_results=aov  # ANOVA 결과 전달하여 p-값 표시
)

# # 분포 비교 그래프 생성
# mixed_anova_distribution_plot(
#     data=sphericity_input_data, 
#     dv='Score', 
#     between='Group', 
#     within='Time', 
#     subject='UniqueID',
#     layout='facet',  # 시간별로 패싯 그리드 생성
#     title='Score Distribution Analysis'
# )


































