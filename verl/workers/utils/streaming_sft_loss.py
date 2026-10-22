"""State-token focal loss for streaming rounds and ordinary NTP loss for offline QA."""

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id


class StreamSFTLoss:
    def __init__(self, state_token_ids, gamma=2.0, auto_alpha=True):
        if not state_token_ids or len(set(state_token_ids)) != len(state_token_ids):
            raise ValueError("Streaming SFT requires nonempty, distinct single-token state IDs")
        if gamma < 0:
            raise ValueError("Focal gamma must be non-negative")
        self.state_token_ids = tuple(state_token_ids)
        self.gamma = float(gamma)
        self.auto_alpha = bool(auto_alpha)

    @torch.no_grad()
    def prepare_batch(self, data: TensorDict, dp_group=None):
        input_ids = data["input_ids"]
        loss_mask = data["loss_mask"].values().bool()
        state_mask = data["stream_state_mask"].values().bool()
        if (state_mask & ~loss_mask).any():
            raise ValueError("Streaming state targets must also be included in loss_mask")
        if loss_mask[input_ids.offsets()[:-1]].any():
            raise ValueError("The first token of every sequence must be masked to avoid cross-sample targets")
        if tu.get_non_tensor_data(data, "batch_num_tokens", 0) <= 0:
            raise ValueError("Streaming SFT batch has no supervised tokens")

        alpha = torch.ones(len(self.state_token_ids), dtype=torch.float32, device=get_device_id())
        if self.auto_alpha:
            targets = input_ids.values()[state_mask]
            counts = torch.stack([(targets == token_id).sum() for token_id in self.state_token_ids])
            counts = counts.to(device=get_device_id(), dtype=torch.float32)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            num_present = (counts > 0).sum().clamp_min(1)
            alpha = torch.where(counts > 0, counts.sum() / (num_present * counts.clamp_min(1)), alpha)

        # One set of global class weights is shared by every accumulated micro-batch.
        tu.assign_non_tensor_data(data, "stream_class_alpha", tuple(alpha.tolist()))

    def __call__(self, model_output, data: TensorDict, dp_group=None):
        log_probs = model_output["log_probs"].values().float()
        targets = torch.roll(data["input_ids"].values(), shifts=-1, dims=0)
        loss_mask = torch.roll(data["loss_mask"].values().bool(), shifts=-1, dims=0)
        state_mask = torch.roll(data["stream_state_mask"].values().bool(), shifts=-1, dims=0)[loss_mask]
        log_probs = log_probs[loss_mask]
        targets = targets[loss_mask]

        ce_loss = -log_probs
        focal = (1.0 - log_probs.exp().clamp(max=1.0)).pow(self.gamma)
        alpha = tu.get_non_tensor_data(data, "stream_class_alpha", None)
        if alpha is None:
            raise RuntimeError("StreamSFTLoss.prepare_batch must run before micro-batch loss computation")
        weights = torch.ones_like(ce_loss)
        for token_id, class_alpha in zip(self.state_token_ids, alpha, strict=True):
            weights = torch.where(state_mask & (targets == token_id), class_alpha * focal, weights)

        # FSDP averages rank gradients; sum micro-batches using the global token denominator.
        token_count = tu.get_non_tensor_data(data, "batch_num_tokens", 0)
        dp_size = tu.get_non_tensor_data(data, "dp_size", 1)
        loss = (weights * ce_loss).sum() * dp_size / token_count
        return loss, {}
