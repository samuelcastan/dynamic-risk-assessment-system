import requests

# Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000/"

data = {
    "data_path": "testdata/testdata.csv"
}

# Call each API endpoint and store the responses
response = requests.post(url=URL + "prediction", json=data)
# response = requests.get(url=URL + "")

print(response.text)