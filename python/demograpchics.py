"""
Demographics of the dataset.
"""
import pandas as pd
import numpy as np
from DataProcessing.readExcel import *
from DataProcessing.visualization import *

# Functions
## Calculate the number of participants
def calculate_number_of_participants(demographics_dict):
    """
    Calculate the number of participants in the dataset.
    """

    # Get the number of participants from each excel file
    name_of_groups = list(demographics_dict.keys())
    number_of_participants = [len(df) for df in demographics_dict.values()]

    # Create a dataframe
    num_of_subjs = pd.DataFrame({"Group": name_of_groups, "Number of Participants": number_of_participants})

    return num_of_subjs, sum(number_of_participants)


## Calculate the average and std of the age
def calculate_avg_and_std_of_age(demographics_dict):
    """
    Calculate the average and std of the age.
    """

    # Create a list to store the average and std of the age
    avg_and_std_of_age = []

    # Get the age from each group
    name_of_groups = list(demographics_dict.keys())
    for group in name_of_groups:
        age = demographics_dict[group]["Age"]

        # Calculate the average and std of the age
        avg_age = np.mean(age)
        std_age = np.std(age)

        # group dataframe
        group_age = pd.DataFrame({"Group": [group], # dataframe expected to non-scalar!
                                 "Average_Age": [avg_age],
                                 "Standard_Deviation_of_Age": [std_age]})
        
        # Add the group dataframe to the list
        avg_and_std_of_age.append(group_age)

    # Concatenate the list of dataframes
    avg_and_std_of_age = pd.concat(avg_and_std_of_age)

    return avg_and_std_of_age


## Calculate the average and std of the height
def calculate_avg_and_std_of_height(demographics_dict):
    """
    Calculate the average and std of the height.
    """

    # Create a list to store the average and std of the height
    avg_and_std_of_height = []

    # Get the height from each group
    name_of_groups = list(demographics_dict.keys())

    for group in name_of_groups:
        height = demographics_dict[group]["Height(m)"]

        # Calculate ave and std of heights
        avg_height = np.mean(height)
        std_height = np.std(height)
        
        # Group dataframe
        group_height = pd.DataFrame({"Group": [group],
                                "Average_Height": [avg_height],
                                "Standard_Deviation_of_Height": [std_height]})

        # Add the group dataframe into the list
        avg_and_std_of_height.append(group_height)

    # Concatenate the list of dataframes
    avg_and_std_of_height = pd.concat(avg_and_std_of_height)

    return avg_and_std_of_height


## Calculate the average and std of the mass
def Calculate_avg_and_std_of_mass(demographics_dict):
    """
    Calculate the average and std of the mass.
    """

    # Create a list to store the average and std of the mass
    avg_and_std_of_mass = []

    # Get the mass from each group
    name_of_group = list(demographics_dict.keys())
    
    for group in name_of_group:
        mass = demographics_dict[group]["Weight(kg)"]

        # Calculate avg and std of the masses
        avg_mass = np.mean(mass)
        std_mass = np.std(mass)

        # Group dataframe
        group_mass = pd.DataFrame({"Group": [group],
                                "Average_Weight": [avg_mass],
                                "Standard_Deviation_of_Weight": [std_mass]})

        # Add the group dataframe into the list
        avg_and_std_of_mass.append(group_mass)

    # Concatenate the list of dataframe
    avg_and_std_of_mass = pd.concat(avg_and_std_of_mass)

    return avg_and_std_of_mass


## For testing
if __name__ == "__main__":
    # demographics directory path
    demographics_dir = r"D:\석사\석사4차\Masters_Thesis_atINHA\data\Demographics"

    # Color palette
    palette = ['skyblue', 'lightcoral', 'lightgreen']

    # Read all excels
    demographics_dict = read_all_excels(demographics_dir)

    # Calculate the number of participants
    num_of_subjs, total_number_of_participants = calculate_number_of_participants(demographics_dict)
    print("\n" + "="*50 + "\nNumber of participants:\n" + "="*50)
    print(num_of_subjs)
    print(f"Total number of participants: {total_number_of_participants}")

    # Calculate the average and std of the age
    avg_and_std_of_age = calculate_avg_and_std_of_age(demographics_dict)
    # Print each dataframe separately with clear separation
    print("\n" + "="*50 + "\nAvg and std of the age:\n" + "="*50)
    print(avg_and_std_of_age)

    # # bar plot with std
    # box_plot_with_std_from_dataframe(
    #     dataframe=avg_and_std_of_age, 
    #     x_axis="Group", 
    #     y_axis="Average_Age", 
    #     std_col="Standard_Deviation_of_Age", 
    #     palette=palette,
    #     title="Average Age by Group", 
    #     x_label="Group", 
    #     y_label="Average Age"
    # )

    # Calculate the avg and std of the height in m
    avg_and_std_of_height = calculate_avg_and_std_of_height(demographics_dict)
    print("\n" + "="*50 + "\nAvg and Std of the height:\n" + "="*50)
    print(avg_and_std_of_height)

    # # Box plot for average height
    # box_plot_with_std_from_dataframe(
    #     dataframe=avg_and_std_of_height, 
    #     x_axis="Group", 
    #     y_axis="Average_Height", 
    #     std_col="Standard_Deviation_of_Height", 
    #     palette=palette,
    #     title="Average Height by Group", 
    #     x_label="Group", 
    #     y_label="Average Height"
    # )

    # Calculate the avg and std of the Weight in kg
    avg_and_std_of_mass = Calculate_avg_and_std_of_mass(demographics_dict)
    print("\n" + "="*50 + "\nAvg and std of the weight:\n" + "="*50)
    print(avg_and_std_of_mass)

    # # Box plot for average weight
    # box_plot_with_std_from_dataframe(
    #     dataframe=avg_and_std_of_mass, 
    #     x_axis="Group", 
    #     y_axis="Average_Weight", 
    #     std_col="Standard_Deviation_of_Weight", 
    #     palette=palette,
    #     title="Average Weight by Group", 
    #     x_label="Group", 
    #     y_label="Average Weight"
    # )
