
import pandas as pd
import timeit
import os
import json
import joblib
import subprocess

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


# Function to get model predictions
def model_predictions():

    model = joblib.load(
        os.path.join(
            prod_deployment_path,
            "trainedmodel.pkl"),
        "r")

    data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    X = data[["lastmonth_activity",
              "lastyear_activity",
              "number_of_employees"]]

    predictions = model.predict(X)

    return predictions


# Function to get summary statistics
def dataframe_summary():

    data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    X = data[["lastmonth_activity",
              "lastyear_activity",
              "number_of_employees"]]

    columns = X.columns

    statistics = {}

    for column in columns:
        mean = X[column].mean()
        median = X[column].median()
        std = X[column].std()
        statistics[column] = {"mean": mean, "median": median, "std": std}

    return statistics


def missing_values_ptg():

    data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    X = data[["lastmonth_activity",
              "lastyear_activity",
              "number_of_employees"]]

    length = len(X)

    percentages = (X.isna().sum() / length) * 100

    return percentages


# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    timing_values = {
        "training.py": None,
        "ingestion.py": None
    }

    for script in timing_values.keys():
        start_time = timeit.default_timer()
        os.system('python {}'.format(script))
        timing = timeit.default_timer() - start_time
        timing_values[script] = timing

    return timing_values


# Function to check dependencies
def outdated_packages_list():
    pass

    result = subprocess.run(['pip', 'list', '--outdated'], capture_output=True, text=True)

    if result.returncode == 0:
        # Save the results to a text file
        with open('outdated_packages.txt', 'w') as file:
            file.write(result.stdout)
    else:
        print("Failed to check for outdated packages. Error:", result.stderr)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_values_ptg()
    execution_time()
    outdated_packages_list()
