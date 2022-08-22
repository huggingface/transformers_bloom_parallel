import torch
import torch.distributed
from transformers.generation_logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper, LogitNormalization, LogitsProcessor

class TensorParallelShardedLogitsProcessor(LogitsProcessor):
    def __init__(self, process_group: torch.distributed.ProcessGroup):
        super().__init__()
        self.process_group = process_group

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits_tp_shard = scores.contiguous()
        batch_size, vocab_tp_shard_size = logits_tp_shard.shape
        tp_world_size = self.process_group.size()
        vocab_size = tp_world_size * vocab_tp_shard_size
        shards = [torch.empty(batch_size, vocab_tp_shard_size, dtype=logits_tp_shard.dtype, device=logits_tp_shard.device) for _ in range(tp_world_size)]
        torch.distributed.all_gather(
            shards,
            logits_tp_shard,
            group=self.process_group
        )
        return torch.stack(shards, dim=1).view(batch_size, vocab_size)

class Sampling():
    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_tokens

class Greedy():
    def __call__(self, logits):
        return logits.argmax(dim=-1)


class NextIdChooser:
    def __init__(self, temperature=1.0, top_k=None, top_p= None, do_sample=False, **kwargs):
        warpers = LogitsProcessorList()
        sampling = do_sample
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
            sampling = True
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
            sampling = True
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
            sampling = True

        self.warpers = warpers
        self.choice = Sampling() if sampling else Greedy()

    def __call__(self, input_ids, scores):
        scores = self.warpers(input_ids, scores)
        next_ids = self.choice(scores)
        return next_ids.unsqueeze(-1)

class StoppingCriteria:
    def __init__(self, max_new_tokens=20, **kwargs):
        self.max_new_tokens = max_new_tokens
        self.current_tokens = 0

    def __call__(self, all_ids):
        self.current_tokens += 1
        return self.current_tokens >= self.max_new_tokens

def unroll_parameters(parameterss):
    next_ids_choosers = []
    stopping_criterias = []
    for parameters in parameterss:
        next_ids_choosers.append(NextIdChooser(**parameters))
        stopping_criterias.append(StoppingCriteria(**parameters))

    return next_ids_choosers, stopping_criterias
