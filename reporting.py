import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import os


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


# Function for reporting
def score_model():
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    model = joblib.load(os.path.join(prod_deployment_path, "trainedmodel.pkl"))

    X = df[["lastmonth_activity",
            "lastyear_activity",
            "number_of_employees"]]

    y = df["exited"]

    y_pred = model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=[
            "Not Exited",
            "Exited"],
        yticklabels=[
            "Not Exited",
            "Exited"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('confusion_matrix.png'))


if __name__ == '__main__':
    score_model()
