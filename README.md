# BLOOM parallel test

## DIRTY solution

install `transformers` branch: `thomas/dirty_bloom_tp`

```
pip -e git+https://github.com/huggingface/transformers.git@thomas/add_custom_kernels#egg=transformers
```

Alternatively,
For the custom kernel:
```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout thomas/add_custom_kernels
python setup.py build_ext --inplace # Might have to edit `setup.py` to remove the torch import
pip install -e .
```


### RUN

This will require `redis` to be installed on the machine.
Redis is the easiest way to communicate through pubsub to all the various processes without causing too much issues for NCCL 
or the webserver threading/circuit breaking model.

```
python -m torch.distributed.run --nproc_per_node=8 generate.py --name bigscience/bloom --max-input-tokens=1000 --save-path=/data/models/
```
```
python server.py
```


### USE

```
curl -X POST -d '{"inputs": "This is a test", "parameters": {"max_new_tokens": 20, "temperature": 0.4}}' http://localhost:8000/generate -H "content-type: application/json"
```
