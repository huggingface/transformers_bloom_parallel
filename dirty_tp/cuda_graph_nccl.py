# python -m torch.distributed.run --nproc_per_node=1 --master_port 7000 transformers_bloom_tensor_parallel/dirty_tp/cuda_graph_nccl.py
# torchrun --nproc_per_node=2 --master_port 7000 transformers_bloom_tensor_parallel/dirty_tp/cuda_graph_nccl.py
# NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1  torchrun --nproc_per_node=2 --master_port 7000 transformers_bloom_tensor_parallel/dirty_tp/cuda_graph_nccl.py
# NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 are necessary or else we get `what():  CUDA error: invalid argument` error`
import contextlib
import datetime
import os
from pathlib import Path

import torch
import torch.backends
import torch.distributed
import torch.distributed.distributed_c10d
from torch._C._autograd import ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, AutoConfig, LogitsProcessorList
from transformers.modeling_utils import no_init_weights

from shard_model import shard_model


# os.environ['TRANSFORMERS_CACHE'] = '/home/thomas_wang_huggingface_co/.cache/huggingface/transformers'
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
    master_port = os.getenv('MASTER_PORT', '7001')
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
        logits_tp_shard = scores.contiguous()
        batch_size, vocab_tp_shard_size = logits_tp_shard.shape
        tp_world_size = self.process_group.size()
        vocab_size = tp_world_size * vocab_tp_shard_size
        logits = torch.empty(batch_size, vocab_size, dtype=logits_tp_shard.dtype, device=logits_tp_shard.device)
        torch.distributed.all_gather(
            list(logits.view(batch_size, tp_world_size, vocab_tp_shard_size).permute(1,0,2)),
            logits_tp_shard,
            group=self.process_group,
            async_op=True
        )
        return logits

def main():

    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.

    # shard_directory = "/home/nouamane_huggingface_co/projects/llm-ultra-fast/models" # "/Users/thomas/code/bigscience/transformers_bloom_tensor_parallel/models"
    shard_directory = "/home/thomas_wang_huggingface_co/models" # "/Users/thomas/code/bigscience/transformers_bloom_tensor_parallel/models"
    model_name = "bigscience/bigscience-small-testing" #"bigscience/bloom"
    # model_name = "bigscience/bloom" #"bigscience/bloom"
    dtype = torch.bfloat16

    process_group = initialize_torch_distributed()
    tp_rank = process_group.rank()
    tp_world_size = process_group.size()

    tensorboard_folder = f"./tb_logs/tb_pt_tp_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_tp-rank-{tp_rank}-of-{tp_world_size}"

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

    config = AutoConfig.from_pretrained(model_name, slow_but_exact=False, tp_parallel=True)

    if torch.cuda.is_available():
        device="cuda"

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        # torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark=True
        # torch.backends.cudnn.deterministic=True
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

    # for name, parameters in model.named_parameters():
    #     print_rank_0(name, parameters.dtype, parameters.shape)

    model.eval()

    while True:
        # Getting input
        accumulating_text = tp_rank == 0 # only tp_rank=0 gets the test
        torch.distributed.barrier(group=process_group)
        texts = ['Hello my name is']

        # Broadcast input to every ranks
        num_text_segment = torch.tensor(len(texts), device=device, dtype=torch.long)
        torch.distributed.broadcast(num_text_segment, src=0)
        # Early return if texts is empty
        if num_text_segment == 0:
            continue
        if tp_rank != 0:
            texts = [None] * num_text_segment
        torch.distributed.broadcast_object_list(texts, src=0, group=process_group)

        text = " ".join(texts)

        # getting generation
        input_ids = tokenizer(text, return_tensors='pt').to(device)
        original_tokens = len(input_ids["input_ids"])

        # Greedy generation
        torch.distributed.barrier(group=process_group)

        if tp_rank == 0 and False:
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=tensorboard_trace_handler(tensorboard_folder),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
        else:
            prof = contextlib.nullcontext()

        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(12):
                with torch.no_grad():
                # with contextlib.nullcontext():
                    greedy_output = model.generate(
                        **input_ids,
                        max_new_tokens=10,
                        do_sample=False,
                        logits_processor=LogitsProcessorList([
                            TensorParallelShardedLogitsProcessor(process_group=process_group)
                        ])
                    )

        torch.cuda.current_stream().wait_stream(s)


        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                greedy_output = model.generate(
                    **input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([
                        TensorParallelShardedLogitsProcessor(process_group=process_group)
                    ])
                )
            torch.distributed.barrier(group=process_group, async_op=True) # this is important or else: RuntimeError: CUDA error: capturing stream has unjoined work

        inp = tokenizer(text, return_tensors='pt').to(device)
        with prof:
            for _ in range(3):
                input_ids["input_ids"].copy_(inp["input_ids"])
                input_ids["attention_mask"].copy_(inp["attention_mask"])
                print_rank_0(input_ids)
                print_rank_0("Start timer")

                start = datetime.datetime.now()

                g.replay() # replay the graph and updates outputs

                print_rank_0(greedy_output)
                
                print_rank_0(f"Pipeline took {datetime.datetime.now() - start} seconds")
                print_rank_0(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

        break


if __name__ == "__main__":
    main()