import os
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.fx import symbolic_trace

import torch
import torch.distributed

from optimum.fx.optimization import compose
from optimum.fx.parallelization import ApplyTensorParallelismModel, ApplyTensorParallelismAlibi


def experiment_with_graph(graph_module):
    for node in graph_module.graph.nodes:
        print("//////" * 100)
        print(node, node.op, node.target, node.args)

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

    return rank, world_size, torch.distributed.distributed_c10d._get_default_group()

def main():
    model_name = "bigscience/bloom-350m"
    tp_rank, tp_world_size, process_group = initialize_torch_distributed()

    # TODO: Offload everything to `disk` so that we don't never load anything in memory before running the parallelization
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True) # pretraining_tp=2, slow_but_exact=True,

    # trace model
    traced = symbolic_trace(
        deepcopy(model),
        input_names=["input_ids", "attention_mask"],
    )

    tp_transformation = ApplyTensorParallelismModel(
        process_group=process_group,
        mlp_h_to_4h_target_suffix="mlp.dense_h_to_4h",
        mlp_4h_to_h_target_suffix="mlp.dense_4h_to_h",
        attention_query_key_values_target_suffix="self_attention.query_key_value",
        attention_dense_target_suffix="self_attention.dense",
        lm_head_target_suffix="lm_head",
        word_embeddings_target_suffix="transformers.word_embeddings"
    )
    alibi_tp_transformation = ApplyTensorParallelismAlibi(
        process_group=process_group,
    )
    transformation = compose(tp_transformation, alibi_tp_transformation)

    transformed_model = transformation(deepcopy(traced))
    if tp_rank == 0:
        # print(transformed_model.code)
        pass

    ### DEBUG check that we have the same weights between original model and sharded model
    first_qkv_module_from_sharded = transformed_model.get_submodule("transformer.h.0.self_attention.query_key_value")
    weight_list = [torch.empty_like(first_qkv_module_from_sharded.weight) for _ in range(tp_world_size)]
    bias_list = [torch.empty_like(first_qkv_module_from_sharded.bias) for _ in range(tp_world_size)]
    torch.distributed.all_gather(weight_list, first_qkv_module_from_sharded.weight, group=process_group)
    torch.distributed.all_gather(bias_list, first_qkv_module_from_sharded.bias, group=process_group)
    new_weight = torch.cat(weight_list, dim=0)
    new_bias = torch.cat(bias_list, dim=0)
    if tp_rank == 0:
        first_qkv_module = model.get_submodule("transformer.h.0.self_attention.query_key_value")
        torch.testing.assert_close(new_weight, first_qkv_module.weight)
        torch.testing.assert_close(new_bias, first_qkv_module.bias)
    torch.distributed.barrier(process_group)
    ###

    # test forward
    # texts = ["Hello my name is", "I love this", " ".join(["Hello my name is"] * 32)] # padding batch
    # texts = ["Hello my name is", "I love my dog"] # no padding batch, short
    texts = [" ".join(["Hello my name is"] * 32), " ".join(["Hello my name is"] * 32)] # np padding batch. long
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)
    print(input_ids)

    # Move everything to cuda if possible
    if torch.cuda.is_available():
        model.cuda()
        transformed_model.cuda()
        input_ids.to("cuda")

    model.eval()
    transformed_model.eval()

    with torch.no_grad():
        results = transformed_model(**input_ids)

        # We need to gather all the attentions
        if "attentions" in results:
            all_attentions = []
            for attention in results["attentions"]:
                attention_from_tp_ranks = [torch.empty_like(attention) for _ in range(tp_world_size)]
                torch.distributed.all_gather(attention_from_tp_ranks, attention, group=process_group)
                all_attentions.append(torch.cat(attention_from_tp_ranks, dim=1))
            results["attentions"] = all_attentions

        # Test consistency across ranks
        logits_from_tp_ranks = [torch.empty_like(results["logits"]) for _ in range(tp_world_size)]
        torch.distributed.all_gather(logits_from_tp_ranks, results["logits"], group=process_group)
        if tp_rank == 0:
            master_rank_logits = logits_from_tp_ranks[0]
            for other_logits in logits_from_tp_ranks:
                torch.testing.assert_close(master_rank_logits, other_logits, atol=0.0, rtol=0.0)
        torch.distributed.barrier(process_group)

        if tp_rank == 0:
            # this should return everything perfectly
            print(tp_world_size)
            model = AutoModelForCausalLM.from_pretrained(model_name, pretraining_tp=tp_world_size, slow_but_exact=True, output_hidden_states=True,
                                                         output_attentions=True) # pretraining_tp=tp_world_size, slow_but_exact=True,

            # we compare results with TP=1 model
            baseline_results = model(**input_ids)

            # logits
            print(" /////" * 20 + " LOGITS " + "/////" * 20)
            print(results["logits"] - baseline_results["logits"])

            # hidden_states
            print(" /////" * 20 + " HIDDEN_STATES[1] " + "/////" * 20)
            print(results["hidden_states"][1] - baseline_results["hidden_states"][1])

            # hidden_states
            print(" /////" * 20 + " HIDDEN_STATES[-2] " + "/////" * 20)
            print(results["hidden_states"][-2] - baseline_results["hidden_states"][-2])

            # hidden_states
            print(" /////" * 20 + " HIDDEN_STATES[-1] " + "/////" * 20)
            print(results["hidden_states"][-1] - baseline_results["hidden_states"][-1])

            # torch.testing.assert_close(results["hidden_states"][0], baseline_results["hidden_states"][0])
            # torch.testing.assert_close(results["attentions"][0], baseline_results["attentions"][0])
            # torch.testing.assert_close(results["hidden_states"][1], baseline_results["hidden_states"][1])
            # torch.testing.assert_close(results["attentions"][1], baseline_results["attentions"][1])
            # torch.testing.assert_close(results["hidden_states"][2], baseline_results["hidden_states"][2])
            # torch.testing.assert_close(results["attentions"][2], baseline_results["attentions"][2])
            # torch.testing.assert_close(results["hidden_states"][-2], baseline_results["hidden_states"][-2])
            # torch.testing.assert_close(results["hidden_states"][-1], baseline_results["hidden_states"][-1])
            torch.testing.assert_close(results["logits"], baseline_results["logits"]) # Maybe rtol doesn't really matter?

if __name__ == "__main__":
    main()
