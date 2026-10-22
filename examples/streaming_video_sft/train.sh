#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

# Priority: trailing Hydra overrides > environment variables > defaults below.

# Distributed launch.
PYTHON_BIN="${PYTHON_BIN:-python}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Dataset list and media roots; Atomic/MotionBench FPS stays source-specific.
DATASET_CONFIG="${DATASET_CONFIG:-$project_root/examples/streaming_video_sft/datasets.yaml}"
STREAMING_VIDEO_ROOT="${STREAMING_VIDEO_ROOT:-/workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData}"
STREAMING_ANNOTATIONS="${STREAMING_ANNOTATIONS:-$STREAMING_VIDEO_ROOT/annotations_curated}"
VIDEOCHAT3_ANNOTATION_ROOT="${VIDEOCHAT3_ANNOTATION_ROOT:-/workspace/s/lzw/datasets/VideoChat3/videochat3_data_annotations}"
VIDEOCHAT3_MEDIA_ROOT="${VIDEOCHAT3_MEDIA_ROOT:-/workspace/s/lzw/datasets/VideoChat3/media}"
STREAMING_FPS="${STREAMING_FPS:-4.0}"
STREAMING_TRUNCATION="${STREAMING_TRUNCATION:-filter}"
STREAMING_MAX_REFETCH="${STREAMING_MAX_REFETCH:-100}"
VIDEOCHAT3_FPS="${VIDEOCHAT3_FPS:-2.0}"
OFFLINE_QA_FPS="${OFFLINE_QA_FPS:-1.0}"
# Offline video QA only; streaming keeps every sampled frame.
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-256}"
FRAME_MIN_PIXELS="${FRAME_MIN_PIXELS:-$((28 * 28))}"
FRAME_MAX_PIXELS="${FRAME_MAX_PIXELS:-$((448 * 448))}"
MISSING_MEDIA="${MISSING_MEDIA:-skip}"

# Input length and DataLoader; token budget controls dynamic micro-batches.
MAX_LENGTH="${MAX_LENGTH:-49152}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$MAX_LENGTH}"
LENGTH_BUCKET_MICRO_BATCHES="${LENGTH_BUCKET_MICRO_BATCHES:-true}"
MAX_PADDED_TOKEN_LEN_PER_GPU="${MAX_PADDED_TOKEN_LEN_PER_GPU:-$MAX_TOKEN_LEN_PER_GPU}"
MAX_VISION_PATCHES_PER_GPU="${MAX_VISION_PATCHES_PER_GPU:-$((MAX_TOKEN_LEN_PER_GPU * 4))}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_DATASET_CONFIG="${VAL_DATASET_CONFIG:-null}"

# Model and acceleration.
MODEL_PATH="${MODEL_PATH:-/workspace/s/lzw/hf/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
USE_LIGER="${USE_LIGER:-true}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-true}"
FUSED_KERNEL_BACKEND="${FUSED_KERNEL_BACKEND:-triton}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

# Training and optimizer. MICRO_BATCH_SIZE is used only with USE_DYNAMIC_BSZ=false.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
SEED="${SEED:-1}"

# FSDP engine; model_dtype is the weight-loading dtype, separate from mixed precision.
ENGINE_STRATEGY="${ENGINE_STRATEGY:-fsdp2}"
ENGINE_DTYPE="${ENGINE_DTYPE:-bfloat16}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
FSDP_SIZE="${FSDP_SIZE:--1}"
RESHARD_AFTER_FORWARD="${RESHARD_AFTER_FORWARD:-false}"
GRADIENT_ACCUMULATION_NO_SYNC="${GRADIENT_ACCUMULATION_NO_SYNC:-true}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-false}"
OFFLOAD_POLICY="${OFFLOAD_POLICY:-false}"

# State-token focal loss; ordinary answers retain cross entropy.
SFT_SUPERVISED_ONLY="${SFT_SUPERVISED_ONLY:-true}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
AUTO_ALPHA="${AUTO_ALPHA:-true}"

# Experiment, logs and checkpoint retention.
PROJECT_NAME="${PROJECT_NAME:-streaming-video-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-vl-2b-streaming-baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/s/lzw/output/StreamMLLM}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/Qwen3VL-2B-baseline}"
TRAIN_LOGGERS="${TRAIN_LOGGERS:-[console,tensorboard]}"
SAVE_FREQ="${SAVE_FREQ:-500}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
SAVE_CONTENTS="${SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
TEST_FREQ="${TEST_FREQ:--1}"

# Runtime environment inherited by all torchrun workers.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ "$NNODES" -gt 1 && -z "${RUN_TIMESTAMP:-}" ]]; then
  printf 'Set the same RUN_TIMESTAMP on every node for multi-node training.\n' >&2
  exit 1
fi
run_timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
output_dir="${OUTPUT_DIR%/}/$run_timestamp"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUTPUT_ROOT/logs/$run_timestamp}"
mkdir -p -- "$output_dir" "$TENSORBOARD_DIR"
printf 'Dataset manifest: %s\n' "$DATASET_CONFIG"
printf 'Checkpoint directory: %s\n' "$output_dir"
printf 'TensorBoard directory: %s\n' "$TENSORBOARD_DIR"

exec "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --module verl.trainer.streaming_sft_trainer \
  "data.dataset_config=$DATASET_CONFIG" \
  "data.streaming_video.annotations=$STREAMING_ANNOTATIONS" \
  "data.streaming_video.video_root=$STREAMING_VIDEO_ROOT" \
  "data.videochat3.annotation_root=$VIDEOCHAT3_ANNOTATION_ROOT" \
  "data.videochat3.video_root=$VIDEOCHAT3_MEDIA_ROOT" \
  "data.streaming_video.fps=$STREAMING_FPS" \
  "data.streaming_video.truncation=$STREAMING_TRUNCATION" \
  "data.streaming_video.max_refetch=$STREAMING_MAX_REFETCH" \
  "data.videochat3.fps=$VIDEOCHAT3_FPS" \
  "data.offline_video_qa.fps=$OFFLINE_QA_FPS" \
  "data.video_max_frames=$VIDEO_MAX_FRAMES" \
  "data.frame_min_pixels=$FRAME_MIN_PIXELS" \
  "data.frame_max_pixels=$FRAME_MAX_PIXELS" \
  "data.streaming_video.missing_video=$MISSING_MEDIA" \
  "data.offline_video_qa.missing_video=$MISSING_MEDIA" \
  "data.videochat3.missing_media=$MISSING_MEDIA" \
  "data.max_length=$MAX_LENGTH" \
  "data.use_dynamic_bsz=$USE_DYNAMIC_BSZ" \
  "data.max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU" \
  "data.length_bucket_micro_batches=$LENGTH_BUCKET_MICRO_BATCHES" \
  "data.max_padded_token_len_per_gpu=$MAX_PADDED_TOKEN_LEN_PER_GPU" \
  "data.max_vision_patches_per_gpu=$MAX_VISION_PATCHES_PER_GPU" \
  "data.num_workers=$NUM_WORKERS" \
  "data.train_max_samples=$TRAIN_MAX_SAMPLES" \
  "data.val_files=$VAL_DATASET_CONFIG" \
  "data.train_batch_size=$GLOBAL_BATCH_SIZE" \
  "data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE" \
  "model.path=$MODEL_PATH" \
  "model.override_config.attn_implementation=$ATTN_IMPLEMENTATION" \
  "model.use_liger=$USE_LIGER" \
  "model.use_fused_kernels=$USE_FUSED_KERNELS" \
  "model.fused_kernel_options.impl_backend=$FUSED_KERNEL_BACKEND" \
  "model.enable_gradient_checkpointing=$GRADIENT_CHECKPOINTING" \
  "trainer.total_epochs=$EPOCHS" \
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
  "trainer.seed=$SEED" \
  "optim.lr=$LEARNING_RATE" \
  "optim.lr_warmup_steps=$WARMUP_STEPS" \
  "optim.lr_warmup_steps_ratio=$WARMUP_RATIO" \
  "optim.lr_scheduler_type=$LR_SCHEDULER" \
  "optim.weight_decay=$WEIGHT_DECAY" \
  "optim.clip_grad=$CLIP_GRAD" \
  "engine.strategy=$ENGINE_STRATEGY" \
  "engine.dtype=$ENGINE_DTYPE" \
  "engine.model_dtype=$MODEL_DTYPE" \
  "engine.fsdp_size=$FSDP_SIZE" \
  "engine.reshard_after_forward=$RESHARD_AFTER_FORWARD" \
  "engine.use_no_sync_for_gradient_accumulation=$GRADIENT_ACCUMULATION_NO_SYNC" \
  "engine.param_offload=$PARAM_OFFLOAD" \
  "engine.optimizer_offload=$OPTIMIZER_OFFLOAD" \
  "engine.offload_policy=$OFFLOAD_POLICY" \
  "streaming_loss.gamma=$FOCAL_GAMMA" \
  "streaming_loss.supervised_only=$SFT_SUPERVISED_ONLY" \
  "streaming_loss.auto_alpha=$AUTO_ALPHA" \
  "trainer.project_name=$PROJECT_NAME" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.default_local_dir=$output_dir" \
  "trainer.logger=$TRAIN_LOGGERS" \
  "trainer.save_freq=$SAVE_FREQ" \
  "trainer.max_ckpt_to_keep=$MAX_CKPT_TO_KEEP" \
  "trainer.resume_mode=$RESUME_MODE" \
  "trainer.resume_from_path=$RESUME_FROM_PATH" \
  "trainer.test_freq=$TEST_FREQ" \
  "checkpoint.save_contents=$SAVE_CONTENTS" \
  "$@"
