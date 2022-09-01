import json
from multiprocessing.pool import ThreadPool
from random import randrange
import requests
API_URL = "http://localhost:8000/generate"
batch_size = 32

def query(payload):
    response = requests.request("POST", API_URL, json=payload)
    print(json.loads(response.content.decode("utf-8")))

def random_input():
    original_text = "Translate to chinese. EN: I like soup. CN:"
    return original_text[:randrange(1,len(original_text))]

with ThreadPool(batch_size) as pool:
    pool.map(
        lambda _: query({"inputs": random_input(), "parameters": {"max_new_tokens": 20, "do_sample": False}}),
        range(batch_size)
    )