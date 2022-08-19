import datetime
import redis
import pickle
import os
import uuid
import threading
from queue import Queue, Empty
import time
import subprocess
import sys
from flask import Flask, jsonify, make_response, request
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(process)d - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__file__)

QUEUE_SIZE = 32


q = Queue()
r = redis.Redis(host="localhost", port=6379, db=0)

max_input_tokens = 1000

shards = 16
name = "bigscience/bloom"

shards = 2
name = "bigscience/bigscience-small-testing"


app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate():
    logger = app.logger
    start = datetime.datetime.now()
    body = request.json

    qsize = q.qsize()
    logger.info(f"Queue size {qsize}")

    if qsize >= QUEUE_SIZE:
        return make_response({"error": "Queue full , try again later"}, 503)
    if "inputs" not in body:
        return make_response({"error": "`inputs` is required"}, 400)

    inputs = body.get("inputs", "Hello")
    parameters = body.get("parameters", {})

    parameters = {
        "do_sample": parameters.get("do_sample", None),
        "temperature": parameters.get("temperature", None),
        "top_k": parameters.get("top_k", None),
        "top_p": parameters.get("top_p", None),
        "max_new_tokens": parameters.get("max_new_tokens", 20),
    }

    if parameters["max_new_tokens"] > 512:
        return make_response(
            {"error": "You cannot generate more than 100 new tokens, at least for now"},
            400,
        )

    topic = str(uuid.uuid4()).encode("utf-8")
    p = r.pubsub()
    p.subscribe([topic])

    q.put(1)
    r.publish("query", pickle.dumps((topic, inputs, parameters)))

    for message in p.listen():
        # print("Message", message)
        if message["type"] == "message":
            q.get()
            out = pickle.loads(message["data"])
            if "error" in out:
                return make_response(out, 400)
            elapsed = datetime.datetime.now() - start
            logger.info(f"Input {repr(inputs)}")
            logger.info(f"Parameters {parameters}")
            logger.info(f"Output {repr(out)}")
            logger.info(
                f"Ran in {elapsed} ({elapsed/parameters['max_new_tokens']}/token)"
            )
            return make_response(jsonify([{"generated_text": out["output"]}]), 200)


if __name__ == "__main__":
    app.run("127.0.0.1", port=8000)
