import contextlib
import datetime
import os
import pickle
from pathlib import Path
import json

import torch
import torch.backends
import torch.distributed
import torch.distributed.distributed_c10d
from torch._C._autograd import ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    AutoConfig,
    LogitsProcessorList,
)
from transformers.modeling_utils import no_init_weights

from shard_model import shard_model, match_suffix
from utils import unroll_parameters

BATCH_SIZE = 32


def initialize_torch_distributed():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

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
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, rank=rank, init_method=init_method
    )

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


def safe_receive(r, p, tokenizer, max_input_tokens):
    for message in p.listen():
        if message["type"] == "message":
            (topic, inputs, parameters) = pickle.loads(message["data"])
            input_ids = tokenizer(inputs, return_tensors="pt")
            if input_ids["input_ids"].shape[1] > max_input_tokens:
                print(f"Ignored prompt was too long {input_ids.shape[1]}")
                r.publish(
                    topic,
                    pickle.dumps(
                        {
                            "error": f"This is a long prompt ({original_tokens} tokens long). We're limiting to {args.max_input_tokens}."
                        }
                    ),
                )
                continue
            return topic, inputs, parameters


def main(args):
    shard_directory = os.getenv("MODELS_CACHE", "/data/models/")
    model_name = args.name
    dtype = torch.bfloat16
    max_new_tokens = 20

    process_group = initialize_torch_distributed()
    tp_rank = process_group.rank()
    tp_world_size = process_group.size()

    # oprint = print
    # def print_server(*args, **kwargs):
    #     oprint(f"[Rank {tp_rank}]", *args, **kwargs)
    # print = print_server

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    print_rank_0("Loaded tokenizer!")
    start = datetime.datetime.now()
    print_rank_0("Loading model")

    # shard state_dict
    if tp_rank == 0:
        # TODO @thomasw21 do some caching
        shard_state_dict_paths = shard_model(
            model_name, Path(shard_directory), tp_world_size=tp_world_size, dtype=dtype
        )
        shard_state_dict_paths = [
            str(path.absolute()) for path in shard_state_dict_paths
        ]
    else:
        shard_state_dict_paths = [None] * tp_world_size

    torch.distributed.broadcast_object_list(
        shard_state_dict_paths, src=0, group=process_group
    )
    shard_state_dict_path = shard_state_dict_paths[tp_rank]

    config = AutoConfig.from_pretrained(
        model_name, slow_but_exact=False, tp_parallel=True
    )
    config.pad_token_id = 3

    device = "cuda"

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    with set_default_dtype(dtype):
        with no_init_weights():
            # we can probably set the device to `meta` here?
            model = AutoModelForCausalLM.from_config(config).to(dtype)

    torch.distributed.barrier(group=process_group)
    print_rank_0(f"Initialized model")
    state_dict = torch.load(shard_state_dict_path)
    # TODO @thomasw21: HACK in order to transpose all weight prior
    for key in state_dict.keys():
        do_transpose = False
        if not match_suffix(key, "weight"):
            continue

        for potential_suffix in [
            "self_attention.query_key_value.weight",
            "self_attention.dense.weight",
            "dense_h_to_4h.weight",
            "dense_4h_to_h.weight",
        ]:
            if match_suffix(key, potential_suffix):
                do_transpose = True

        if do_transpose:
            state_dict[key] = state_dict[key].transpose(1, 0).contiguous()

    model.load_state_dict(state_dict)
    model.to(device)
    torch.distributed.barrier(group=process_group)
    model = model.eval()
    print_rank_0(f"Loaded model in {datetime.datetime.now() - start}")

    if tp_rank == 0:
        r = redis.Redis(host="localhost", port=6379, db=0)

        p = r.pubsub()
        p.subscribe(["query"])

    print(f"Ready in {datetime.datetime.now() - start}")
    torch.distributed.barrier(group=process_group)
    accumulating_text = tp_rank == 0  # only tp_rank=0 gets the test
    while True:
        # Getting input
        torch.distributed.barrier(group=process_group)
        if tp_rank == 0:
            print(f"Waiting for new objects")
            items = [safe_receive(r, p, tokenizer, args.max_input_tokens)]

            sub.setsockopt(zmq.RCVTIMEO, 0)
            while len(items) < BATCH_SIZE:
                try:
                    print("Waiting for the batch")
                    item = safe_receive(r, p, tokenizer, args.max_input_tokens)
                    items.append(item)
                except zmq.error.Again:
                    break
            sub.setsockopt(zmq.RCVTIMEO, -1)
            print(f"Received batch of {len(items)}")
        else:
            items = []
        torch.distributed.barrier(group=process_group)

        # Broadcast input to every ranks
        num_text_segment = torch.tensor(len(items), device=device, dtype=torch.long)
        torch.distributed.broadcast(num_text_segment, src=0)

        if tp_rank == 0:
            texts = items
        else:
            texts = [None] * num_text_segment
        torch.distributed.broadcast_object_list(texts, src=0, group=process_group)

        topics, inputss, parameterss = zip(*texts)
        input_ids = tokenizer(list(inputss), return_tensors="pt", padding=True).to(
            device
        )

        next_id_choosers, stopping_criterias = unroll_parameters(parameterss)

        torch.distributed.barrier(group=process_group)

        all_input_ids = [
            input_ids["input_ids"][i : i + 1]
            for i in range(input_ids["input_ids"].shape[0])
        ]

        # As long as we still have something there.
        with torch.no_grad():
            while True:
                # for k, v in input_ids.items():
                #     if isinstance(v, (list, tuple)):
                #         print_rank_0(k, v[0][0].shape)
                #     else:
                #         print_rank_0(k, v.shape)
                outputs = model.forward(**input_ids, use_cache=True)

                keep_ids = []
                keep_past_ids = []
                next_input_ids = []
                something_has_exited = False
                for i in range(input_ids["input_ids"].shape[0]):
                    logits = outputs.logits[i : i + 1]
                    next_id_chooser = next_id_choosers[i]
                    stopping_criteria = stopping_criterias[i]
                    all_ids = all_input_ids[i]
                    topic = topics[i]

                    next_ids = next_id_chooser(all_ids, logits[:, -1])
                    all_ids = torch.cat([all_ids, next_ids], dim=1)
                    all_input_ids[i] = all_ids

                    if stopping_criteria(all_ids):
                        output = tokenizer.decode(all_ids[0], skip_special_tokens=True)
                        print_rank_0(topic, repr(output))
                        something_has_exited = True
                        if tp_rank == 0:
                            pub.send(b"%s %s" % (topic, pickle.dumps(output)))
                    else:
                        keep_ids.append(i)
                        keep_past_ids.extend(
                            [
                                j
                                for j in range(
                                    i, i + config.n_head // process_group.size()
                                )
                            ]
                        )
                        next_input_ids.append(next_ids)

                if not keep_ids:
                    break

                if something_has_exited:
                    input_ids["attention_mask"] = input_ids["attention_mask"][keep_ids]
                    input_ids["past_key_values"] = [
                        (key[keep_past_ids], value[keep_past_ids])
                        for key, value in outputs["past_key_values"]
                    ]
                    next_id_choosers = [next_id_choosers[i] for i in keep_ids]
                    stopping_criterias = [stopping_criterias[i] for i in keep_ids]
                    all_input_ids = [all_input_ids[i] for i in keep_ids]
                    topics = [topics[i] for i in keep_ids]
                else:
                    input_ids["past_key_values"] = outputs["past_key_values"]

                input_ids["input_ids"] = torch.cat(next_input_ids, dim=0)
                input_ids["attention_mask"] = torch.cat(
                    [
                        input_ids["attention_mask"],
                        torch.ones((input_ids["attention_mask"].shape[0], 1)).to(
                            input_ids["attention_mask"].device
                        ),
                    ],
                    dim=1,
                )

        # # Use profiler only when asked for
        # if args.profile and tp_rank == 0:
        #     prof = profile(
        #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #         # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #         on_trace_ready=tensorboard_trace_handler(tensorboard_folder),
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True
        #     )
        # else:
        #     prof = contextlib.nullcontext()

        # start = datetime.datetime.now()
        # with prof:
        #     greedy_output = model.generate(
        #         **input_ids,
        #         **parameters,
        #         logits_processor=LogitsProcessorList([
        #             TensorParallelShardedLogitsProcessor(process_group=process_group)
        #         ])
        #     )
        #     torch.distributed.barrier(group=process_group)
        # stop = datetime.datetime.now()

        # # print generation
        # output_tokens = greedy_output[0]
        # output = tokenizer.decode(output_tokens, skip_special_tokens=True)

        # actual_new_tokens = len(output_tokens) - input_ids["input_ids"].shape[1]
        # print_rank_0(f"Took {stop - start} ({(stop - start) / actual_new_tokens} / token) ({actual_new_tokens})")

        # if tp_rank == 0:
        #     msg = {"topic": topic, "output": output}
        #     print(f"Sending {msg}")
        #     pub.send_pyobj(msg)


if __name__ == "__main__":
    torch.manual_seed(0)
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--name", required=True, type=str, help="model_name")
    parser.add_argument(
        "--workers",
        required=True,
        type=int,
        help="How many publishers exist and we need to connect to",
    )
    parser.add_argument(
        "--max-input-tokens",
        required=True,
        type=int,
        help="Maximum prompt length (in tokens)",
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    main(args)
