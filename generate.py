import contextlib
import datetime
import os
from pathlib import Path

import torch
import torch.distributed
import torch.distributed.distributed_c10d
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, AutoConfig, LogitsProcessorList
from transformers.modeling_utils import no_init_weights

from shard_model import shard_model


def initialize_torch_distributed():
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    if torch.cuda.is_available():
        # initialized `torch.distributed`
        # Set the device id.
        assert world_size <= torch.cuda.device_count(), "Each process is one gpu"
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        backend = "gloo"


    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method)

    return torch.distributed.distributed_c10d._get_default_group()

# override print with this function
def print_rank_0(*texts):
    process_group = torch.distributed.distributed_c10d._get_default_group()
    if process_group.rank() == 0:
        print(*texts)

@contextlib.contextmanager
def set_default_dtype(dtype):
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved_dtype)

# Necessary for generate
class TensorParallelShardedLogitsProcessor(LogitsProcessor):
    def __init__(self, process_group: torch.distributed.ProcessGroup):
        super().__init__()
        self.process_group = process_group

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores.contiguous()
        logits_from_tp_ranks = [torch.empty_like(logits) for _ in range(self.process_group.size())]
        torch.distributed.all_gather(logits_from_tp_ranks, logits, group=self.process_group)
        return torch.cat(logits_from_tp_ranks, dim=-1)

def main():
    shard_directory = "/home/thomas_wang_huggingface_co/models" # "/Users/thomas/code/bigscience/transformers_bloom_tensor_parallel/models"
    model_name = "bigscience/bloom"
    dtype = torch.bfloat16
    max_length = 50

    process_group = initialize_torch_distributed()
    tp_rank = process_group.rank()
    tp_world_size = process_group.size()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    print_rank_0("Loaded tokenizer!")
    start = datetime.datetime.now()
    print_rank_0("Loading model")

    # shard state_dict
    if tp_rank == 0:
        # TODO @thomasw21 do some caching
        shard_state_dict_paths = shard_model(model_name, Path(shard_directory), tp_world_size=tp_world_size, dtype=dtype)
        shard_state_dict_paths = [str(path.absolute()) for path in shard_state_dict_paths]
    else:
        shard_state_dict_paths = [None] * tp_world_size

    torch.distributed.broadcast_object_list(shard_state_dict_paths, src=0, group=process_group)
    shard_state_dict_path = shard_state_dict_paths[tp_rank]

    config = AutoConfig.from_pretrained(model_name, tp_parallel=True)

    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    with set_default_dtype(dtype):
        with no_init_weights():
            # we can probably set the device to `meta` here?
            model = AutoModelForCausalLM.from_config(config).to(dtype)
    torch.distributed.barrier(group=process_group)
    print_rank_0(f"Initialized model")
    model.load_state_dict(torch.load(shard_state_dict_path, map_location=device))
    model.to(device)
    torch.distributed.barrier(group=process_group)
    print_rank_0(f"Loaded model in {datetime.datetime.now() - start}")

    for name, parameters in model.named_parameters():
        print_rank_0(name, parameters.dtype, parameters.shape)

    while True:
        # Getting input
        accumulating_test = tp_rank == 0 # only tp_rank=0 gets the test
        torch.distributed.barrier(group=process_group)
        texts = []
        while accumulating_test:
            text = input(
                '''Enter the paragraph (Enter for to validate new input line and Ctrl-c to start generating the prompt):''')
            if text == "":
                accumulating_test = False
            else:
                texts.append(text)
        torch.distributed.barrier(group=process_group)
        torch.distributed.broadcast_object_list(texts, src=0, group=process_group)

        text = "\n".join(texts)

        # getting generation
        input_ids = tokenizer(text, return_tensors='pt').to(device)

        # Greedy generation
        greedy_output = model.generate(
            **input_ids,
            max_length=max_length,
            do_sample=False,
            logits_processor=LogitsProcessorList([
                TensorParallelShardedLogitsProcessor(process_group=process_group)
            ])
        )

        # print generation
        print_rank_0(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()