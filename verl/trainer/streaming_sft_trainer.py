"""Mixed streaming and offline Qwen3-VL SFT using the existing verl trainer and FSDP engine."""

import logging
import os
import sys

import hydra

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.dataset.streaming_video import STATE_TOKENS
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.workers.utils.streaming_sft_loss import StreamSFTLoss

logger = logging.getLogger(__name__)


class StreamingSFTTrainer(SFTTrainer):
    def _build_config(self):
        super()._build_config()
        if self.model_config.hf_config.model_type not in ("qwen3_vl", "qwen3_vl_moe"):
            raise ValueError("Streaming SFT currently supports Qwen3-VL models")
        if self.engine_config.strategy not in ("fsdp", "fsdp2"):
            raise ValueError("Streaming SFT requires the FSDP or FSDP2 engine")
        if self.model_config.use_remove_padding or self.engine_config.ulysses_sequence_parallel_size != 1:
            raise ValueError("Streaming SFT requires model.use_remove_padding=false and sequence parallel size 1")
        if self.engine_config.use_fused_kernels != self.model_config.use_fused_kernels:
            raise ValueError("engine.use_fused_kernels must match model.use_fused_kernels")
        fused_backend = self.model_config.fused_kernel_options.get("impl_backend")
        if self.model_config.use_fused_kernels and fused_backend not in ("torch", "triton", "liger"):
            raise ValueError("Streaming SFT fused_kernel_options.impl_backend must be torch, triton or liger")
        if self.model_config.lora_rank > 0 or self.model_config.lora_adapter_path:
            raise ValueError("Streaming SFT currently requires full SFT so state embeddings and lm_head are trained")
        if self.config.data.pad_mode != "no_padding":
            raise ValueError("Streaming SFT requires data.pad_mode=no_padding for the verl collator")
        if self.model_config.processor is None:
            raise ValueError("Qwen3-VL streaming SFT requires a multimodal processor")

        tokenizer = self.model_config.tokenizer
        self.state_token_ids = []
        for token in STATE_TOKENS:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or ids[0] not in tokenizer.all_special_ids:
                raise ValueError(f"Add {token} through model.additional_special_tokens before training")
            self.state_token_ids.append(ids[0])
        logger.warning(
            "Streaming SFT acceleration config: attention=%s, use_liger=%s, use_fused_kernels=%s, fused_backend=%s",
            self.model_config.override_config.get("attn_implementation", "flash_attention_2"),
            self.model_config.use_liger, self.model_config.use_fused_kernels, fused_backend,
        )

    def _build_engine(self):
        super()._build_engine()
        loss_config = self.config.get("streaming_loss", {})
        self.loss_fn = StreamSFTLoss(
            self.state_token_ids,
            gamma=loss_config.get("gamma", 2.0),
            auto_alpha=loss_config.get("auto_alpha", True),
            supervised_only=loss_config.get("supervised_only", True),
        )
        self.training_client.set_loss_fn(self.loss_fn)

    def _build_dataloader(self):
        dp_size = self.engine.get_data_parallel_size()
        batch_size = self.config.data.train_batch_size
        if batch_size < dp_size or batch_size % dp_size:
            raise ValueError("data.train_batch_size must be a positive multiple of the data-parallel world size")
        if not self.config.data.use_dynamic_bsz:
            micro_batch_size = self.config.data.micro_batch_size_per_gpu
            if micro_batch_size <= 0 or (batch_size // dp_size) % micro_batch_size:
                raise ValueError("The per-rank batch size must be divisible by data.micro_batch_size_per_gpu")
        super()._build_dataloader()
        self.train_sampler.seed = self.config.trainer.seed
        if not len(self.train_dataloader):
            raise ValueError("No complete training batch; reduce data.train_batch_size or provide more valid samples")


@hydra.main(config_path="config", config_name="streaming_sft_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    initialize_global_process_group()
    try:
        trainer = StreamingSFTTrainer(config)
        trainer.fit()
    except Exception:
        logger.exception("Streaming SFT failed on rank %s", os.environ.get("RANK", "?"))
        sys.stdout.flush()
        sys.stderr.flush()
        # Collective cleanup can block while peers wait for this rank; let torchrun terminate them.
        os._exit(1)
    finally:
        destroy_global_process_group()


if __name__ == "__main__":
    main()
