import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

# Function for model scoring


def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    data = pd.read_csv(
        os.path.join(test_data_path, "testdata.csv")
    )

    with open(os.path.join(output_model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    y_test = data.pop("exited")
    X_test = data[['lastmonth_activity',
                   'lastyear_activity', 'number_of_employees']]

    y_pred = model.predict(X_test)

    f1score = f1_score(y_test, y_pred)

    with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as file:
        file.write(str(f1score))


if __name__ == '__main__':

    score_model()
