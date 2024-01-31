import requests

data = {
    "data_path": "testdata/testdata.csv"
}

# Make a POST request to the "/prediction" endpoint
response = requests.post("http://0.0.0.0:8000/prediction", json=data)

# response = requests.post("http://0.0.0.0:8000/prediction")

print(response.text)
