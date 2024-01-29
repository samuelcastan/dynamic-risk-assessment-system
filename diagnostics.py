
import pandas as pd
import numpy as np
import timeit
import os
import json
import joblib

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


# Function to get model predictions
def model_predictions():
    
    model = joblib.load(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "r")

    data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    X = data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    
    predictions = model.predict(X)
    
    return predictions


# Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    model = joblib.load(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "r")
    data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X = data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    
    columns = X.columns
    
    statistics = {}
    
    for column in columns:
        mean = X[column].mean()
        median = X[column].median()
        std = X[column].std()
        statistics[column] = {"mean": mean, "median": median, "std": std}

    return statistics

# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

# Function to check dependencies
def outdated_packages_list():
    pass
    # get a list of


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    # execution_time()
    # outdated_packages_list()





