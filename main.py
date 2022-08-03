import os

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

    # # initialized `torch.distributed`
    # # Set the device id.
    # device = rank % torch.cuda.device_count()
    # torch.cuda.set_device(device)

    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    process_group = torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        init_method=init_method)

    return rank, world_size, process_group

def main():
    model_name = "bigscience/bloom-350m"
    tp_rank, tp_world_size, process_group = initialize_torch_distributed()

    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

    # trace model
    traced = symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask"],
    )

    tp_transformation = ApplyTensorParallelismModel(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        process_group=process_group,
        mlp_h_to_4h_target_suffix="mlp.dense_h_to_4h",
        mlp_4h_to_h_target_suffix="mlp.dense_4h_to_h",
        attention_query_key_values_target_suffix="self_attention.query_key_value",
        attention_dense_target_suffix="self_attention.dense",
    )
    alibi_tp_transformation = ApplyTensorParallelismAlibi(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        process_group=process_group,
    )
    transformation = compose(tp_transformation, alibi_tp_transformation)

    transformed_model = transformation(traced)
    print(transformed_model.code)

    # test forward
    texts = ["Hello my name is", "I love this"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)

    model.eval()
    transformed_model.eval()
    with torch.no_grad():
        results = transformed_model(**input_ids)

        if tp_rank == 0:
            # we compare results with TP=1 model
            baseline_results = model(**input_ids)

            # logits
            print(" /////" * 20 + " LOGITS " + "/////" * 20)
            print(results["logits"])
            print(baseline_results["logits"])
            print(results["logits"] - baseline_results["logits"])

            # hidden_states
            print(" /////" * 20 + " HIDDEN_STATES[1] " + "/////" * 20)
            print(results["hidden_states"][1])
            print(baseline_results["hidden_states"][1])
            print(results["hidden_states"][1] - baseline_results["hidden_states"][1])

            # hidden_states
            print(" /////" * 20 + " HIDDEN_STATES[-1] " + "/////" * 20)
            print(results["hidden_states"][-1])
            print(baseline_results["hidden_states"][-1])
            print(results["hidden_states"][-1] - baseline_results["hidden_states"][-1])

            torch.testing.assert_close(results["hidden_states"][0], baseline_results["hidden_states"][0])
            torch.testing.assert_close(results["hidden_states"][-1], baseline_results["hidden_states"][-1])
            torch.testing.assert_close(results["logits"], baseline_results["logits"])

if __name__ == "__main__":
    main()
