import redis
import time
import pickle

r = redis.Redis(host="localhost", port=6379, db=0)

p = r.pubsub()
p.subscribe(["query"])

while True:
    for message in p.listen():
        print("Message", message)
        if message["type"] == "message":
            (topic, inputs, parameters) = pickle.loads(message["data"])
            r.publish(topic, pickle.dumps({"output": inputs}))
