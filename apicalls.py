import requests

# Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000/"

data = {
    "data_path": "testdata/testdata.csv"
}

# Call each API endpoint and store the responses
response1 = requests.post(url=URL + "prediction", json=data).text
response2 = requests.get(url=URL + "scoring").text
response3 = requests.get(url=URL + "summarystats").text
response4 = requests.get(url=URL + "diagnostics").text

# combine all API responses
responses = {
    "prediction": response1,
    "scoring": response2,
    "summary_stats": response3,
    "diagnostics": response4
}

print(responses)
