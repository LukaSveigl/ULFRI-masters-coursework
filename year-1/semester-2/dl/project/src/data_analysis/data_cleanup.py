# data_cleanup.py
# 
# This file contains a script for cleaning the disease prediction training dataset. The script uses pandas 
# for data manipulation.
# 
# The script first loads the training dataset from a CSV file. It then identifies and drops any columns 
# that have 'unnamed' in their column name (case insensitive), which are often artifacts of the data 
# export process and do not contain useful information.
# 
# After the cleanup process, the script saves the cleaned dataset back to the original CSV file, 
# overwriting the original uncleaned data.
# 
# This script is essential for the disease diagnosis system, as it ensures that the training data for the 
# disease prediction model is clean and free of unnecessary columns.
import pandas as pd

if __name__ == "__main__":
    # Load the data
    train_dataset = pd.read_csv("../../data/disease_prediction/Training.csv")

    # Drop the final unnamed column
    train_dataset.drop(train_dataset.columns[train_dataset.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    # Save the cleaned data
    train_dataset.to_csv("../../data/disease_prediction/Training.csv", index=False)
    