import pandas as pd
import os
import glob

# Functions
## Read header
def read_header(file_path):
    """
    Read the header with customized number of rows
    """

    # Read the header row
    header_row = 1 # Number of rows to read as header
    header_df = pd.read_excel(file_path, header=None, nrows=header_row)

    # Extract values from the first row
    header = header_df.iloc[0].tolist()

    return header, header_row

## Read data
def read_data(file_path, header_row, column_names):
    """
    Read the data using the provided column names, skipping the header row.
    """

    # Read the excel file, skipping the header row(s) and assigning column names
    data_df = pd.read_excel(file_path, skiprows=header_row, header=None, names=column_names)
    return data_df


## Read all raw data
def read_all_data(file_path):
    """
    Read all raw data from the excel file.
    """

    # Read all raw data
    all_data_df = pd.read_excel(file_path, header=None)

    return all_data_df

## Read all excel files in the directory
def read_all_excels(directory_path):
    """
    Read all excels in the directory and save them in dictionary.
    """

    # Get all excel files in the directory
    excel_files = glob.glob(os.path.join(directory_path, "*.xlsx")) or glob.glob(os.path.join(directory_path, "*.xls"))
    
    # Read all excels
    all_data_dict = {}
    for file in excel_files:
        file_name = os.path.basename(file) # Get the file name
        file_name = file_name.split(".")[0] # Remove the file extension
        header, header_row = read_header(file)
        data_df = read_data(file, header_row, header)
        all_data_dict[file_name] = data_df

    return all_data_dict


## Split the data into two groups
def split_data(all_data_dict):
    """
    Split the data into three groups.
    
    Args:
        all_data_dict: Dictionary containing the data for each group
        
    Returns:
        Three DataFrames, one for each group
    """
    group_A_df = all_data_dict["GroupA_MAE"]
    group_B_df = all_data_dict["GroupB_DI"]
    group_C_df = all_data_dict["GroupC_Control"]
    
    return group_A_df, group_B_df, group_C_df

def split_by_timepoint(df):
    """
    Split the dataframe by Pre, Post1, and Post2 columns.
    
    Args:
        df: DataFrame containing Pre, Post1, and Post2 columns
        
    Returns:
        Three Series/DataFrames for Pre, Post1, and Post2 data
    """
    # Extract each timepoint column, preserving the 'Name' column if needed
    pre_data = df[['Pre']]
    post1_data = df[['Post1']]
    post2_data = df[['Post2']]
    
    return pre_data, post1_data, post2_data

# For testing
# if __name__ == "__main__":

    # # Excel file path
    # file_path = r"D:\\석사\\석사4차\\Masters_Thesis_atINHA\\data\\10mRecords\\GroupA_MAE.xlsx"

    # # Read header
    # header, header_row = read_header(file_path)
    # print(f"header: \n{header}")

    # # Read data using the extracted header as column names
    # df = read_data(file_path, header_row, header)
    # print(f"df: \n{df}")

    # # Read all raw data
    # # all_data_df = read_all_data(file_path)
    # # print(f"all_data_df: \n{all_data_df}")

    ## Read all excels in the directory
    # directory_path = r"D:\\석사\\석사4차\\Masters_Thesis_atINHA\\data\\10mRecords"
    # all_data_dict = read_all_excels(directory_path)

    # # Print each dataframe separately with clear separation
    # print("\n" + "="*50 + "\nALL DATA DICTIONARY:\n" + "="*50)
    # for file_name, df in all_data_dict.items():
    #         print(f"\n\n{'-'*20} {file_name} {'-'*20}")
    #         print(df)

    # # Split the data into two groups
    # group_A_df, group_B_df, group_C_df = split_data(all_data_dict)

    # # Print each dataframe separately with clear separation
    # print("\n" + "="*50 + "\nGROUP A DATAFRAME:\n" + "="*50)
    # print(group_A_df)

    # print("\n" + "="*50 + "\nGROUP B DATAFRAME:\n" + "="*50)
    # print(group_B_df)

    # print("\n" + "="*50 + "\nGROUP C DATAFRAME:\n" + "="*50)
    # print(group_C_df)

    # # Demonstrate splitting by timepoint for Group A
    # print("\n" + "="*50 + "\nSPLIT BY TIMEPOINT (GROUP A):\n" + "="*50)
    # pre_A, post1_A, post2_A = split_by_timepoint(group_A_df)
    
    # print("\nPre data:")
    # print(pre_A)
    
    # print("\nPost1 data:")
    # print(post1_A)
    
    # print("\nPost2 data:")
    # print(post2_A)