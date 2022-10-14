import torch

from huggingface_hub import hf_hub_download
from pathlib import Path
from tqdm import tqdm


def match_suffix(text, suffix):
    return text[-len(suffix) :] == suffix


def prepare_weights(model_name: str, save_path: Path, tp_world_size: int):
    save_paths = [
        save_path / f"{model_name}_tp-rank-{tp_rank}-of-{tp_world_size}.pty"
        for tp_rank in range(tp_world_size)
    ]

    if all(save_path.exists() for save_path in save_paths):
        print("Weights are already prepared")
        return save_paths

    shards_state_dicts = [{} for _ in range(tp_world_size)]

    filenames = [f"pytorch_model_000{i:02d}-of-00072.bin" for i in range(1, 73)]


    for filename in tqdm(filenames):
        weight_path = hf_hub_download(model_name, filename=filename)
        state_dict = torch.load(weight_path, map_location="cpu")

        keys = list(state_dict.keys())
        for state_name in keys:
            state = state_dict[state_name]
            if any(
                match_suffix(state_name, candidate)
                for candidate in [
                    "self_attention.query_key_value.weight",
                    "self_attention.query_key_value.bias",
                    "mlp.dense_h_to_4h.weight",
                    "mlp.dense_h_to_4h.bias",
                    "word_embeddings.weight",
                    "lm_head.weight",
                ]
            ):
                output_size = state.shape[0]
                assert output_size % tp_world_size == 0
                block_size = output_size // tp_world_size
                sharded_weights = torch.split(state, block_size, dim=0)
                assert len(sharded_weights) == tp_world_size
                for tp_rank, shard in enumerate(sharded_weights):
                    assert shard.shape[0] == block_size
                    if match_suffix(state_name, "lm_head.weight"):
                        shards_state_dicts[tp_rank][state_name] = shard.detach().clone()
                    else:
                        shards_state_dicts[tp_rank][
                            "transformer." + state_name
                        ] = shard.detach().clone()
            elif any(
                match_suffix(state_name, candidate)
                for candidate in [
                    "self_attention.dense.weight",
                    "mlp.dense_4h_to_h.weight",
                    "lm_head.weight",
                ]
            ):
                input_size = state.shape[1]
                assert input_size % tp_world_size == 0
                block_size = input_size // tp_world_size
                sharded_weights = torch.split(state, block_size, dim=1)
                assert len(sharded_weights) == tp_world_size
                for tp_rank, shard in enumerate(sharded_weights):
                    assert shard.shape[1] == block_size
                    if match_suffix(state_name, "lm_head.weight"):
                        shards_state_dicts[tp_rank][state_name] = shard.detach().clone()
                    else:
                        shards_state_dicts[tp_rank][
                            "transformer." + state_name
                        ] = shard.detach().clone()
            elif any(
                match_suffix(state_name, candidate)
                for candidate in [
                    "self_attention.dense.bias",
                    "mlp.dense_4h_to_h.bias",
                ]
            ):
                shards_state_dicts[0][
                    "transformer." + state_name
                ] = state.detach().clone()
                for tp_rank in range(1, tp_world_size):
                    shards_state_dicts[tp_rank][
                        "transformer." + state_name
                    ] = torch.zeros_like(state)
            else:
                # We duplicate parameters across tp ranks
                for tp_rank in range(tp_world_size):
                    shards_state_dicts[tp_rank][
                        "transformer." + state_name
                    ] = state.detach().clone()

            del state_dict[state_name]  # delete key from state_dict
            del state  # delete tensor

    # we save state_dict
    for tp_rank, (save_path, shard_state_dict) in enumerate(
        zip(save_paths, shards_state_dicts)
    ):
        save_paths.append(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            print(f"Skipping {save_path} as it already exists")
        else:
            torch.save(shard_state_dict, save_path)

    return save_paths


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="The model to use",
    )
    parser.add_argument("--save-path", required=True, type=str)
    parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args()

    prepare_weights(args.model, Path(args.save_path), args.world_size)
