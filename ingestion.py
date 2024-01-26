"""
Looks for new data to ingest

Author: Samuel Castan
Date: July, 2023
"""
import pandas as pd
import os
import json


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    """
    Checks for datasets, compile them together, and write to an output file
    """

    # Set the file extension to search for
    extension = '.csv'

    # Empty list to store the file names to import briefly
    file_names = []

    # Loop through all the files in the directory
    for filename in os.listdir(input_folder_path):
        # Check if the file matches the desired extension
        if filename.endswith(extension):
            # append filename to list for further reading
            file_names.append(filename)

    # Empty dataframe to store all the files to import
    df = pd.DataFrame()

    # Concatenating new data into a dataframe
    for file in file_names:
        _ = pd.read_csv(os.path.join(input_folder_path, file))
        df = pd.concat([df, _])

    # Removing duplicated records
    df = df.drop_duplicates()

    # Ingest dataframe into output folder
    df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)

    # Log files used for further steps
    with open(
        os.path.join(
            output_folder_path,
            "ingestedfiles.txt"),
        "w"
    ) as text_file:
        text_file.write(",".join(file_names))


if __name__ == '__main__':
    merge_multiple_dataframe()
