import datetime
import pickle
import os
import uuid
import threading
from queue import Queue, Empty
import time
import threading
import zmq
import uvicorn
import subprocess
import sys
from flask import Flask, jsonify, make_response, request
import logging

logging.basicConfig(level=logging.DEBUG, format='%(process)d - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__file__)

QUEUE_SIZE=32


q = Queue()
context = zmq.Context()

max_input_tokens = 1000

shards = 16
name = "bigscience/bloom"

shards = 2
name = "bigscience/bigscience-small-testing"


# def server_loop(q):
#     while True:
#         (topic, inputs, parameters, s) = q.get()
#         pub.send_pyobj((topic, inputs, parameters))
#         while True:
#             out = sub.recv_pyobj()
#             rtopic = out.pop("topic")
#             if rtopic == topic:
#                 s.put(out)
#                 break
# 
# 
# t = threading.Thread(target=server_loop, args=(q, )).start()

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
        "max_new_tokens": parameters.get("max_new_tokens", 20)
    }


    if parameters["max_new_tokens"] > 512:
        return make_response({"error": "You cannot generate more than 100 new tokens, at least for now"}, 400)


    topic = str(uuid.uuid4()).encode("utf-8")
    logger.info(f"Sending {inputs}")

    sub_port = 5560
    sub =  context.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{sub_port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"%s" % topic)

    port = 5559
    pub = context.socket(zmq.PUB)
    pub.connect(f"tcp://localhost:{port}")

    msg = b"query %s"  % pickle.dumps((topic, inputs, parameters))
    logger.info(f"Sending {msg}")
    pub.send(msg)
    out = sub.recv()
    _, messagedata = out.split()
    out = pickle.loads(messagedata)
    if "error" in out:
        return make_response(out, 400)
    elapsed = datetime.datetime.now() - start
    logger.info(f"Input {repr(inputs)}")
    logger.info(f"Parameters {parameters}")
    logger.info(f"Output {repr(out)}")
    logger.info(f"Ran in {elapsed} ({elapsed/parameters['max_new_tokens']}/token)")
    return make_response(jsonify([{"generated_text": out["output"]}]), 200)




    # command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(shards), "generate.py", "--name",name, "--max-input-tokens", str(max_input_tokens)]
    # print(" ".join(command))
    # subprocess.Popen(command)
    # print("Waiting for shards to connect")
    # for i in range(shards):
    #     response = sub.recv()
    #     print("Shard {response} connected");
if __name__ == "__main__":
    app.run("127.0.0.1", port=8000)
