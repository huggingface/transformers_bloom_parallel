# BLOOM parallel test

## DIRTY solution

install `transformers` branch: `thomas/dirty_bloom_tp`

### RUN

`python -m torch.distributed.run --nproc_per_node=2 generate.py`

## FX solution

### SETUP

install `optimum`, branch: `thomas/make_tensor_parallel_via_fx`

install `transformers`, branch: `thomas/make_tp_work_with_bloom` (we only try to support bloom for now)

### RUN

`python -m torch.distributed.run --nproc_per_node=2 main.py`