import json
from multiprocessing.pool import ThreadPool

import requests
API_URL = "http://localhost:8000/generate"
def query(payload):
    response = requests.request("POST", API_URL, json=payload)
    return json.loads(response.content.decode("utf-8"))

batch_size = 32

with ThreadPool(batch_size) as pool:
    pool.map(
        lambda : query({"inputs": "test", "parameters": {"max_new_tokens": 20, "do_sample": False}}),
        range(batch_size)
    )
