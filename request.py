import requests
data = {
    "user_name": "joshua"
  }

url = "http://127.0.0.1:5000/predict_api"
response = requests.post(url, json=data)
print("Churn: "+ str(response.json()))
