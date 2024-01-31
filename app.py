from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
from diagnostics import model_predictions, dataframe_summary, missing_values_ptg, execution_time
from scoring import score_model
import json
import os


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


# Welcoming
@app.route("/")
def index():
    return "Welcome to the API!"


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():

    data_path = request.json.get(
        'data_path', os.path.join(
            test_data_path, "testdata.csv"))

    # Call the model_predictions function with the specified data_path
    predictions = model_predictions(data_path)
    # Return the predictions as JSON
    return jsonify(predictions)


# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def stats():
    # check the score of the deployed model

    f1_score = score_model()
    return jsonify(f1_score)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary_stats():
    # check means, medians, and modes for each column
    statistics = dataframe_summary()
    return statistics


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # check timing and percent NA values
    na_percentages = missing_values_ptg()
    timing_values = execution_time()

    result = {
        "na_percetange": na_percentages,
        "timing_values": timing_values
    }

    return result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
