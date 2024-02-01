from ingestion import merge_multiple_dataframe
from training import train_model
import scoring
import deployment
import diagnostics
import reporting
import json
import os

with open("config.json", "r") as f:
    config = json.load(f)

prod_deployment_path = config["prod_deployment_path"]
input_folder_path = config["input_folder_path"]

# Check and read new data
# first, read ingestedfiles.txt
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as file:
    data = file.read()
    ingestedfiles = data.split(",")

# second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt
extension = ".csv"
file_names = []

for filename in os.listdir(input_folder_path):
    if filename.endswith(extension):
        file_names.append(filename)

new_files = [file for file in file_names if file not in ingestedfiles]

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) > 0:
    merge_multiple_dataframe()
    train_model()
    
else:
    print("No new data to ingest")

# Checking for model drift
# check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data



# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the
# process here


# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
