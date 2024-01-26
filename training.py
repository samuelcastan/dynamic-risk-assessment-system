import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


# Function for training the model
def train_model():

    # read data to train on
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    y = df.pop('exited')
    X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]

    model = RandomForestClassifier(
        n_estimators=1000
    )

    # fit the logistic regression to your data\
    model.fit(X, y)

    # write the trained model to your workspace in a
    # file called trainedmodel.pkl
    pickle.dump(
        model,
        open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb')
    )


if __name__ == '__main__':
    train_model()
