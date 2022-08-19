import json
from random import randrange
import requests
API_URL = "http://localhost:8000/generate"
def query(payload):
    response = requests.request("POST", API_URL, json=payload)
    return json.loads(response.content.decode("utf-8"))
data = query({"inputs": "test", "parameters": {"max_new_tokens": 20, "do_sample": False}})
print(data)
