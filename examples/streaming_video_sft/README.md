# Qwen3-VL 流式与离线多模态混合 SFT

本实现直接使用当前 verl 的 `SFTTrainer → TrainingWorker → FSDP/FSDP2`，通过一份数据集清单混合训练 `stream-meta-v1` 流式数据、VideoChat3、LLaVA-OneVision-2-Spatial 和 ET-Instruct 的图片与离线视频问答数据。流式每帧一个 user-assistant 轮次、完整会话单样本、所有 assistant 轮次监督；状态为 `</Silence>` / `</Response>`，不创建 Standby 标签。超过 `MAX_LENGTH` 的完整流式会话默认过滤。普通图片/视频问答样本不增加交互状态，回答采用普通 NTP Cross Entropy。采样 FPS 以 `train.sh` 和各来源配置为准。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `examples/streaming_video_sft/datasets.yaml` | 统一配置数据集类型、标注、视频根目录、重复次数及各数据集选项 |
| `verl/utils/dataset/mixed_video_sft.py` | 读取清单，组合流式与离线 QA 数据，复用原生 sampler |
| `verl/utils/dataset/streaming_video.py` | JSONL/JSON 解析、媒体路径解析、逐帧完整会话、超长过滤、全轮 Label Mask 和 Qwen3-VL mRoPE |
| `verl/utils/dataset/offline_video_qa.py` | 通用离线 QA 兼容及当前 assistant 目标编码 |
| `verl/utils/dataset/videochat3.py` | VideoChat3 文本、多图、离线视频消息与 JSONL 偏移索引 |
| `verl/utils/dataset/videochat3_media.py` | 根据 manifest 读取解压媒体或 tar 内的图像、GIF、MP4、帧目录 |
| `verl/utils/dataset/image_video_qa.py` | Spatial/ET-Instruct 格式转换、媒体占位符匹配、JSONL/JSON 数组字节偏移索引 |
| `examples/streaming_video_sft/videochat3_sources.json` | 从官方配置提取的 210 个标注文件与媒体根目录映射 |
| `verl/workers/utils/streaming_sft_loss.py` | 状态 Token Focal Loss、回答 Token CE、全局类别权重与归一化 |
| `verl/trainer/streaming_sft_trainer.py` | 复用原生 SFT Trainer 的训练入口及必要配置检查 |
| `verl/trainer/config/streaming_sft_trainer.yaml` | 模型、数据、FSDP2、优化器和 Checkpoint 配置 |
| `examples/streaming_video_sft/prepare_data.py` | 只读检查标注，或新建包含已解析媒体路径的 JSONL |
| `examples/streaming_video_sft/train.sh` | 单机/多机 torchrun 启动 |
| `examples/streaming_video_sft/infer.py` | 本地视频逐帧回放、跨帧 KV 增量推理、两状态决策、保留历史文本的窗口重建、JSONL 输出 |

通用框架只增加了配置可选的特殊 Token 注册、FSDP 包装前的 Embedding 扩展、分 micro-batch 前的 loss 准备 hook，以及多模态张量进入模型前的设备对齐。普通 SFT 不启用新增 Token 配置时不扩展词表。

## 使用 uv 创建、配置和管理环境

以下命令在 **Linux NVIDIA GPU 服务器的 Bash** 中执行，工作目录始终是本项目的 `projects/Training/verl-main`。使用仓库自己的 `pyproject.toml` 和 `uv.lock`，为流式 SFT 建立独立的 `.venv-streaming-sft` 环境，不使用 conda base。下面是供服务器执行的命令，本次文档更新没有创建环境或安装依赖。

### 1. 确认服务器与锁定版本

当前仓库的 `fsdp` extra 使用以下版本；这些值来自本地依赖文件，并非任意组合的安装建议：

| 项目 | 当前配置 |
| --- | --- |
| 系统与 Python | Linux x86_64 / aarch64，CPython 3.12；原生 wheel 使用 cp312 |
| PyTorch / torchvision / torchaudio | 2.13.0+cu130 / 0.28.0+cu130 / 2.11.0+cu130 |
| Transformers / TorchCodec | 5.12.1 / 0.16.0，使用单独的 fsdp extra 时 |
| 训练方式 | FSDP2、全参数 SFT、BF16、FlashAttention 2、Liger；verl 融合输出头默认关闭 |

服务器需要能运行 CUDA 13.0 系列 wheel 的 NVIDIA 驱动，以及 FFmpeg 的共享动态库。uv 管理 Python 与 Python 包，驱动和 FFmpeg 由系统提供。BF16 需要相应 GPU 支持。先检查：

```bash
nvidia-smi
uname -m
ffmpeg -version
ffprobe -version
```

Ubuntu / Debian 服务器缺少 FFmpeg 时，由有系统安装权限的使用者执行：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

TorchCodec 的视频解码需要可加载的 FFmpeg 共享库；只有静态 `ffmpeg` 可执行文件不够，见 [TorchCodec 安装说明](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)。

### 2. 安装 uv 并创建独立环境

若服务器尚未安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

接下来将第一行替换为服务器上本项目的实际代码目录。已经有 `pyproject.toml`，无需执行 `uv init`：

```bash
cd /实际代码目录/VideoLLM-stream/projects/Training/verl-main
# ==============================
# 1. 项目配置
# ==============================
export PROJECT_ROOT="$(pwd)"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv-streaming-sft"

# ==============================
# 2. 下载加速
# ==============================

# 并发下载数（uv 默认约为 50）
export UV_CONCURRENT_DOWNLOADS=128

# 网络不稳定时避免过早超时
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=5

# 中国大陆网络可启用清华 PyPI 镜像
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

# ==============================
# 3. 安装 Python 和项目依赖
# ==============================

# 已安装的 Python 3.12 会被复用
uv python install 3.12

# uv sync 会自动创建虚拟环境，无需单独 uv venv
uv sync \
    --frozen \
    --extra fsdp \
    --python 3.12

# ==============================
# 4. 激活环境
# ==============================
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

export PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"

"$PYTHON_BIN" --version
```

`UV_PROJECT_ENVIRONMENT` 明确选择环境目录，`--extra fsdp` 选择训练依赖，`--frozen` 使用现有锁文件且不重新解析版本。`uv sync` 会以 editable 方式安装当前 verl 代码，并安装 Hydra/OmegaConf、TensorDict、torchdata、Accelerate、Pillow 和 TorchCodec 等依赖。`fsdp` extra 已包含 wheelhouse 提供的 `flash-attn==2.8.3`、`liger-kernel>=0.8.2` 和随 Torch 安装的 Triton；流式 SFT 默认开启三项加速，无需增加依赖或更改锁文件。本任务的训练和 Transformers 推理 Demo 使用同一环境，不需要选择 vLLM、SGLang 或 Megatron extra。

uv 的参数含义见 [Python 管理](https://docs.astral.sh/uv/guides/install-python/)、[项目环境路径](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path) 和 [依赖同步](https://docs.astral.sh/uv/concepts/projects/sync/)。

### 3. 检查解释器、依赖和 GPU

```bash
uv pip check --python "$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import sys
from importlib.metadata import version

import torch
from flash_attn import flash_attn_func
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
from torchcodec.decoders import VideoDecoder
from transformers import AutoProcessor, AutoModelForImageTextToText

print("python:", sys.executable)
for package in (
    "verl", "torch", "transformers", "torchcodec", "tensordict", "torchdata", "hydra-core",
    "flash-attn", "liger-kernel", "triton",
):
    print(package, version(package))
print("torch CUDA:", torch.version.cuda)
assert sys.version_info[:2] == (3, 12), "Use the Python 3.12 environment"
assert torch.cuda.is_available(), "CUDA is unavailable in this environment"
print("GPU count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
```

这些命令检查依赖、FlashAttention/Liger 导入、TorchCodec 导入和 CUDA 可用性，尚未加载模型或解码实际视频。安装失败时先核对 Linux、Python 3.12、仓库配置的 PyTorch / wheelhouse 源及网络连接；不要直接混装其他 CUDA / PyTorch 版本。

### 4. 日常复用与管理

每个新终端先进入同一代码目录并重新选择环境：

```bash
cd /实际代码目录/VideoLLM-stream/projects/Training/verl-main
export PROJECT_ROOT="$(pwd)"
export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv-streaming-sft"
export PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

常用管理命令如下；同步和运行时保持同一个 `fsdp` extra：

```bash
# 按仓库锁文件重新同步已有环境
uv sync --frozen --extra fsdp --python 3.12

# 查看已安装版本、依赖一致性及可复现的版本清单
uv pip list --python "$PYTHON_BIN"
uv pip check --python "$PYTHON_BIN"
uv pip freeze --python "$PYTHON_BIN"

# 未激活环境时也可执行项目命令，仍需保留 UV_PROJECT_ENVIRONMENT
uv run --frozen --extra fsdp python --version

# 退出当前激活的 shell 环境
deactivate
```

`uv sync` 默认精确同步所选依赖，可能移除手动安装但未声明的包。需要持久变更依赖时，同时更新 `pyproject.toml` 和 `uv.lock`，再同步环境；普通训练不需要执行 `uv add`、`uv lock --upgrade` 或 `uv sync --all-extras`。`uv run --no-sync ...` 仅适用于已经同步好的环境，它会跳过同步检查。

当前实现支持全参数 SFT、FSDP/FSDP2、`ulysses_sequence_parallel_size=1`、`model.use_remove_padding=false`。`data.pad_mode=no_padding` 是数据的变长存储格式；引擎在模型前向时转换为带 attention mask 的独立序列，并不把不同样本拼成一条可互相注意的序列。LoRA 和序列并行暂不启用，以保证新增词表参数训练和多模态边界正确。

默认加速配置位于 `verl/trainer/config/streaming_sft_trainer.yaml`：

```yaml
model:
  use_liger: true
  use_fused_kernels: true
  fused_kernel_options:
    impl_backend: triton
  override_config:
    attn_implementation: flash_attention_2
engine:
  use_fused_kernels: ${model.use_fused_kernels}
```

FlashAttention 2 用于注意力；`use_liger` 通过现有模型实例补丁启用支持的 RMSNorm/RoPE 算子。verl 融合输出头默认使用 Triton 后端，直接计算逐 Token `log_probs`，继续交给 `StreamSFTLoss` 计算状态 Focal 和普通回答 CE。Liger 自带的 `fused_linear_cross_entropy` 前向保持关闭。动态类别权重、Label Mask 和全局监督 Token 归一化均保持原逻辑。Liger 的 Qwen3-VL 接口参考[锁定版本源码](https://github.com/linkedin/Liger-Kernel/blob/v0.8.2/src/liger_kernel/transformers/monkey_patch.py)。

模型与引擎的融合开关通过插值保持一致。启动日志中的 `Streaming SFT acceleration config` 应显示 `use_fused_kernels=True, fused_backend=triton`，模型补丁日志应出现 `Using Triton backend for fused kernels in Qwen3VLForConditionalGeneration`。已有启动命令若显式关闭过融合内核，请将 `model.use_fused_kernels=true engine.use_fused_kernels=true` 作为最后两个覆盖参数。同步配置后需重新启动训练，已有训练进程不会自动切换；实际分布式训练及性能由服务器验证。

## 统一数据集入口

训练配置的 `data.dataset_config` 指向 `examples/streaming_video_sft/datasets.yaml`。DataLoader 通过 `MixedVideoSFTDataset` 读取这份清单，按每项 `type` 选择数据适配器；新增、删减或调整数据源只需修改清单。当前启用 1 个流式来源、11 个 VideoChat3 来源、17 个 Spatial 子数据集和 1 个 ET-Instruct 来源。VideoChat3 名称保留 `videochat3_` 前缀，Spatial 使用各子数据集名称。下面是清单格式示例，实际启用列表以 `datasets.yaml` 为准：

```yaml
datasets:
  - name: streaming_interaction
    type: streaming
    media_type: video
    enabled: true
    annotations: ${data.streaming_video.annotations}
    video_root: ${data.streaming_video.video_root}
    max_samples: -1
    repeat: 1
    options:
      fps: ${data.streaming_video.fps}
      truncation: ${data.streaming_video.truncation}
      max_refetch: ${data.streaming_video.max_refetch}
      missing_video: ${data.streaming_video.missing_video}
  - name: "videochat3_STEM_AI2D_GPT4V_train_0-0_merged_5507"
    type: videochat3
    media_type: image
    enabled: true
    annotations: "${data.videochat3.annotation_root}/image/honey_meta_merged_no_think_stem_norepeat/STEM_AI2D_GPT4V_train_0-0_merged_5507_sr0.05.jsonl"
    video_root: "${data.videochat3.video_root}"
    max_samples: -1
    repeat: 1
    options:
      annotation_root: "${data.videochat3.annotation_root}"
      source_map: videochat3_sources.json
      fps: 1.0
      video_max_frames: ${data.videochat3.video_max_frames}
      missing_media: ${data.videochat3.missing_media}
  - name: "videochat3_atomic_motion_sft_4fps"
    type: videochat3
    media_type: video
    enabled: true
    annotations: "${data.videochat3.annotation_root}/motion-videos/01_atomic_motion/sft_sr1.jsonl"
    video_root: "${data.videochat3.video_root}"
    max_samples: -1
    repeat: 1
    options:
      annotation_root: "${data.videochat3.annotation_root}"
      source_map: videochat3_sources.json
      fps: 4.0
      video_max_frames: ${data.videochat3.video_max_frames}
      missing_media: ${data.videochat3.missing_media}
```

`type` 标识标注适配器，支持 `streaming`、`videochat3`、`offline_video_qa`、`llava-onevision2-spatial` 和 `et-instruct`；`media_type` 标识媒体模态。当前 Spatial 标注使用图片及多图输入，声明为 image；ET-Instruct 声明为 video。多图 QA 仍标为 image；VideoChat3 预抽帧目录属于 video，其帧率和读取形式继续由标注及 source_map 指定。流式标注内部的 `media_type=video/sequence` 用于区分文件与帧目录，不会被清单字段覆盖。

加载器检查 media_type 的合法性，流式与通用离线视频 QA 只允许 video；VideoChat3 在索引时按来源映射的类别检查声明是否一致。为了兼容既有自定义清单，省略字段时使用 auto；适配器也支持 text 或 mixed 声明，但当前清单不启用纯文本/GIF 数据源。媒体标识不改变目标 Token 或 Loss Mask。

`annotations` 可填写文件、目录、glob 或路径列表；目录递归加载 JSON/JSONL。清单中的相对 `annotations`、`video_root`、`options.path_map`、`options.source_map`、`options.annotation_root` 文件/目录路径以该清单所在目录为基准。每项 `options` 覆盖该类适配器的数据选项，流式与通用 `offline_video_qa` 的 `path_map`、`path_replacements` 也可以放在这里；VideoChat3 使用官方来源映射与本地 manifest。`enabled: false` 完整跳过该项；只训练一种数据时保留对应项，或将其他项禁用。启用项的标注路径无匹配时明确报错；跳过缺失媒体后完全为空的来源会记录告警并跳过，全部来源都为空时训练报错。

`repeat` 是正整数，用于重复该数据源的样本索引，增加其在拼接数据集中的相对数量；不复制视频或预处理图片。每个 epoch 由 verl 原生 sampler 在全体样本上 shuffle，**不保证每个 batch 中的两类数据占固定比例**。流式一条标注会话是一个候选样本，离线每个 assistant 回答是一个样本，因此 `repeat: 1` 并不代表两类数据数量各占一半。可用每项的 `max_samples` 限制样本数量，再配合 `repeat` 调整训练占比。

每项 `max_samples` 在重复前生效，取 `-1` 表示不限，正整数表示上限。全局 `data.train_max_samples` 在重复后按各来源样本数的比例确定性限量，并均匀覆盖来源的索引；每个启用来源至少保留一个样本，所以上限小于启用来源数时会报错。`data.val_files` 可指定另一份清单用于验证，默认 `null`；训练清单和验证清单分别加载。当前清单共 30 个来源，跳过完全缺失媒体的空来源后，以实际加载的非空来源数检查全局上限；短跑示例使用 256。

## LLaVA-OneVision-2-Spatial 与 ET-Instruct

默认清单使用服务器的真实目录：

| 数据 | 标注位置 | `video_root`（图片和视频共同使用的媒体根目录） |
| --- | --- | --- |
| Spatial | `/workspace/s/lzw/datasets/Stream-VideoLLM/Source/LLaVA-OneVision-2-Spatial/spatial_instruct/` | `/workspace/s/lzw/datasets/Stream-VideoLLM/Source/LLaVA-OneVision-2-Spatial/videos` |
| ET-Instruct | `/workspace/s/lzw/datasets/Stream-VideoLLM/Source/ET-Instruct/et_instruct_164k_txt.json` | `/workspace/s/lzw/datasets/Stream-VideoLLM/Source/ET-Instruct/videos` |

Spatial 的 `type` 为 `llava-onevision2-spatial`，`name` 为子数据集名；同名 `.part_*.jsonl` 分片归为一个来源，当前 36 个文件组成 17 个来源。保留 `messages`、多图顺序和回答文本，按 `<image>` 位置插入对应图片。提供了 `images` 但没有占位符时，把图片插入首个 user 消息。标注中的 `images/3D/...` 等路径直接拼到 `video_root`，例如最终路径为 `.../videos/images/3D/...`。`depth` 是辅助标注，不作为 RGB 图片输入，也不检查其旧绝对路径。

ET-Instruct 使用 `name: ET-Instruct`、`type: et-instruct`，一个来源读取全部 163,880 条标注及其任务。`human/gpt` 转为 `user/assistant`，`<image>` 按该标注的 `video` 字段解释为视频输入；例如 `how_to_step/PJi8ZEHAFcI.mp4` 解析为 `.../ET-Instruct/videos/how_to_step/PJi8ZEHAFcI.mp4`。问题和答案的时间表达原样保留，不把 `tgt` 等答案元数据加入输入。视频按完整视频范围抽帧，问题中指定的关注区间保持在文本中。

两种类型共用 `data.image_video_qa` 预处理设置，支持每项 `options` 覆盖。视频 FPS 默认继承 `data.offline_video_qa.fps`（启动脚本的 `OFFLINE_QA_FPS`），帧数上限继承 `VIDEO_MAX_FRAMES`，视频帧像素设置继承 `FRAME_MIN_PIXELS` / `FRAME_MAX_PIXELS`。图片像素设置默认与 VideoChat3 相同。`missing_media` 继承启动脚本的 `MISSING_MEDIA`，默认跳过缺失媒体的整条标注并输出来源统计。修改 `data.image_video_qa.system_prompt` 可覆盖默认的 `You are a helpful assistant.`；标注自带的 system 消息优先保留。

每个 assistant 回答构建一个普通 QA 样本，之前的会话作为前缀，system、user、图片及历史回答不计 loss，仅当前 assistant 目标参与标准交叉熵；`stream_state_mask` 全零。复用现有 VideoChat3 的多图/视频组装和 `OfflineVideoQASFTDataset._encode()`，使用相同的 Qwen3-VL 模板、视觉网格检查及 mRoPE。视频超长时按现有离线策略减少采样帧；保留单帧仍超长或图片问答超长时明确报错，答案不截断。

初始化只建立标注字节偏移和 assistant 轮次数索引，不进行 Tokenization 或解码。ET-Instruct 的 JSON 数组在建立索引时逐对象解析并记录 UTF-8 字节位置，读取样本时只 seek/read 对应对象；不为每条 QA 重新加载整个数组。会话缓存受 `cache_sessions` 限制，DataLoader worker 之间不共享打开的文件或解码器。开发交付只做静态检查；服务器上的媒体读取、Tokenization 和分布式训练需在已有训练环境验证。

## 流式数据格式

标注递归读取 JSONL，也支持 JSON 对象/数组。目录名包括 EventRetrospection、ProactiveInteraction、FuturePrediction 及其子任务；不依赖硬编码目录层数，分类、来源、任务等额外字段原样保留。

```json
{
  "schema": "stream-meta-v1",
  "source": "streamo",
  "video_source": "YouCook2",
  "classification": "Event Retrospection",
  "video_name": "0Mz4NTozNXw",
  "video_path": "",
  "media_type": "video",
  "sequence": {"storage": "video", "fps": null},
  "video_ranges": [[0.0, 162.0]],
  "prompts": [{"at": 162.0, "text": "Recall the earlier observed states relevant to this monitoring question."}],
  "timeline": [
    {"type": "silence", "interval": [0.0, 161.0]},
    {"type": "response", "at": 162.0, "text": "Earlier observed states: at 70 s: 0; at 93 s: 3."}
  ]
}
```

上例取自本地同类真实记录，问题作了缩写；空路径保持为空，需由使用者填写或显式映射。代码不会根据 `video_name` 猜测视频文件名。

时间始终使用秒，支持数值、数值字符串和 `MM:SS` / `HH:MM:SS`。默认 `fps=4`，每隔 0.25 秒采样一帧，每帧各自形成一个 user-assistant 轮次。从 0 秒开始的前四轮分别对应 0.25、0.5、0.75、1.0 秒；FPS 同时决定图像采样和状态决策频率，不再单独设置轮时长。

`video_ranges` 中相接区间先合并再采样，避免边界产生额外决策；有间隔的区间分别采样，保留原始绝对时间。最后不足一个采样周期时保留真实截止时间的一帧，不用未来帧补齐。问题和 Response 只在第一个采样时间不早于其 `at` 的轮出现；同一轮中的多个回答按顺序合并。Response 覆盖该轮 Silence，未标注轮保持 Silence。不进行语义补标。

`media_type=sequence` 时，`video_path` 必须指向真实帧目录，`sequence.fps` 必须存在。默认按自然文件名顺序读取帧，支持 jpg/jpeg/png/webp。若需明确文件名，可在标注的 `sequence.filename_pattern` 或 `data.streaming_video.sequence_filename_pattern` 中指定，例如 `{seconds:05d}_{centiseconds:02d}.jpg`；也支持 `{index:06d}` / `{frame_idx:06d}`，编号偏移由 `sequence.start_index` 指定。非均匀、缺帧或带其他图片的目录应先整理或提供明确 pattern。

## VideoChat3 Stage3 数据与媒体

依据 [官方标注目录](https://huggingface.co/datasets/lmwang/VideoChat3-Stage3-Training-Data/tree/main/videochat3_data_annotations)、[来源配置](https://huggingface.co/datasets/lmwang/VideoChat3-Stage3-Training-Data/blob/main/VideoChat3_Stage3_Training_Data.json) 和 [媒体目录](https://huggingface.co/datasets/lmwang/VideoChat3-Stage3-Training-Data/tree/main/data) 适配。服务器布局为：

```text
/workspace/s/lzw/datasets/VideoChat3/
  videochat3_data_annotations/
    text/Dolci-Instruct-SFT/*.jsonl
    image/<标注分组>/*.jsonl
    motion-videos/<子任务>/sft_sr1.jsonl
    video/motionbench_train_4998_sr1.jsonl
    video/vflash_video/*.jsonl
  media/                         # 对应官方 data/，仅本地根目录改名
    image/
      manifest.jsonl
      image-*.tar                # 可保持 tar 分片，也可解压到当前目录
      Caption/...                # 解压后的示例目录
      remake/...
      STEM/...
    motion-videos/
      manifest.jsonl
      motion-videos-*.tar
      <视频文件>                 # 解压时位于当前目录
    video/
      manifest.jsonl
      video-*.tar
      <MP4、GIF 或帧目录>
```

原始 `messages` 的 `content` 可以是字符串，也可以是列表。支持 `text`、`image_url: {url: ...}`、`video_url: {url: ...}` 及 `video_metadata`；一轮可含多图，视频可使用 MP4、GIF 或预抽帧资源。消息内媒体顺序保留，原有 `<IMG_CONTEXT>` / `<VIDEO_CONTEXT>` 占位符移除后由 Qwen3-VL Processor 重建视觉 Token，不迁移 VideoChat3 模型专用 Token。只使用当前 assistant 之前已经出现的媒体和文本，后续消息不进入该训练样本。

标注中的相对 URL **不能一律直接拼接 media 根目录**。例如 COYO 的 `coyo_.../0.jpg` 应解析到 `media/image/Caption/COYO-Recaption/coyo_.../0.jpg`，Co-Instruct 在 `media/image/remake/General/Co-Instruct/`；MP4/GIF 位于 `media/video/`，MotionBench/TVQA 使用 `media/video/<视频ID>/` 帧目录。这些前缀由 `videochat3_sources.json` 精确指定，覆盖官方当前发布的 210 个标注文件；文件内保存上游配置 URL 与 SHA256。`annotation_root` 用来计算标注的相对名称并查表，标注相对层级应保留。

解压后的文件优先读取。仅保留原始 tar 时，必须同时保留同类别的 `manifest.jsonl`，以 `media_root + source_filename` 精确定位 shard/member；不会搜索同名文件，也不需要训练时下载或解压。Motion 来源配置与 manifest 中的上传者绝对路径按其共同的 `videochat3_data_annotations/` 后缀对齐。读取 tar 使用 Python 标准库，MP4 使用已有 torchcodec，图像和 GIF 使用已有 Pillow，不新增依赖。帧目录按官方文件名字典序排列，并按标注源 FPS 选帧；片段边界取 `clip_start_time/clip_end_time`。

当前清单在使用者提供的 JSON 基础上移除了全部 16 个纯文本数据源，以及 TGIF count / transition 两个 GIF 数据源。保留图片 177 项、Motion 12 项、其他视频 3 项，共 192 项，均分别启用；Motion 4/8/30 FPS 版本继续保留。VideoChat3 普通视频默认 2 FPS，当前 TVQA 和 CinePile 的 `options.fps` 引用 `${data.videochat3.fps}`，默认 2 FPS；Atomic 保留 4/8/30 FPS，MotionBench 保留 4 FPS。`video_max_frames=256` 是每个视频的采样帧数上限，超过时在整个片段内均匀限帧，实际有效 FPS 会降低。每项 repeat=1，不额外进行数据增强或再次应用上游采样比例。

每个子数据集可独立调整 enabled、max_samples、repeat 和 options。各子数据集在同一个 worker 中共享按媒体根目录缓存的读取器，避免重复加载 manifest 或各自打开相同 tar。适配器及来源映射仍保留文本/GIF 兼容能力，但当前训练清单不加载这些数据源。

纯文本样本不要求媒体文件，也不传图片给 Processor；图片不加视频时间戳；视频帧带各自秒级时间戳。每个 assistant 回答独立构造一个样本，之前的 system/user、视觉与历史 assistant 全部 mask，当前答案和模板结束符全部使用 NTP/CE，`stream_state_mask` 全零。JSONL 仅索引字节偏移并缓存少量会话，避免常驻保存数 GB 的 Dolci 文本。

超长时只对视频帧均匀减帧，保留多图问答中的全部图片与完整答案；文本、多图或视频减到一帧后仍超长则报错，需要提高 Token 预算或降低每图像素。`missing_media: skip` 默认跳过缺少实际媒体的记录并统计数量，`error` 则停止；来源映射配置错误与训练时的媒体解码错误仍会报错。

离线适配器在建立索引时跳过空回答、缺少必需字段及不支持的角色或内容类型。流式索引只检查 schema、video_ranges 是否存在和媒体路径，不展开时间线、不读取视频尺寸或编码会话；时间线的完整校验在取样时执行，格式错误会报错。公共读取器会跳过 JSONL 的 JSON/UTF-8 解析错误和非对象记录，并保留原文件字节偏移；无法解析的整份 JSON 文件会跳过并告警。索引为空的数据源会从混合数据中跳过，全部为空时停止。原始标注不会改写。

## 通用离线视频 QA 兼容格式

`type: offline_video_qa` 继续兼容 LLaVA 等数据，但默认清单已由 VideoChat3 替换。需要这类来源时可另加清单项并配置其标注和 `video_root`。常见标注使用 `video` 与 `conversations`：

```json
{
  "id": "example",
  "video": "原始标注中提供的相对视频路径.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nDescribe what happens in this video."},
    {"from": "gpt", "value": "The answer from the original annotation."}
  ]
}
```

上例只展示字段结构，`video` 必须使用原始标注的真实路径。适配器也接受 `messages` 的 `role/content` 格式，以及 `video` / `video_path` 配合 `question` / `answer` 的单轮格式；不要求离线数据转换成 `stream-meta-v1`，不添加 Silence/Response 标签。

通用离线 QA 默认 `fps=1`、`video_max_frames=256`，按 `ceil(duration * fps)` 决定帧数、上限 256 帧，再在全片均匀取样。每个 assistant 回答构成一个训练样本，前序对话作为上下文，后续对话不进入该样本。输入帧、时间信息、system/user 及历史 assistant 文本不计 loss，只监督当前答案和模板结束符，采用普通自回归下一 Token 预测。超长时逐步减少全片均匀采样帧，不截断答案；一帧仍超长则报错。

离线任务允许看到完整视频；混合训练不改变流式样本“只看当前及历史”的因果边界。`video` / `videos` 可使用单元素路径列表，多视频记录不在当前适配范围内。

LLaVA 的不同子集可能在 `video` 中保留不同层级。默认 `video_root` 与原始相对路径直接拼接，所以解压目录必须保持这两者能够组成真实文件路径；如原始路径已经含 `videos/`，应将该数据项的根目录改为 TrainingData，或按实际下载结构配置精确前缀替换。需要不同根目录的子集可以拆成多项。如果实际目录明确采用 `data_source/video` 层级，可显式设置该项 `options.prepend_data_source: true`；默认关闭。代码不搜索同名视频、不根据 `id` 猜文件名，也不自动下载媒体。

## 视频路径配置与缺失媒体

服务器默认位置：

```text
/workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData/
  annotations_curated/
  videos/
```

路径规则：

1. 可选 `path_map` 按 `source/video_source/video_name`、原始 `video_path`、`video_name` 的优先级查找使用者提供的精确映射；空路径只有显式映射才能补齐。
2. 可选 `path_replacements` 对路径前缀作最长匹配替换。
3. 相对路径再拼接 `video_root`，绝对路径直接使用。

清单中流式数据的默认 `video_root` 是 TrainingData 根目录，默认没有前缀重写。当前 `annotations_curated` 的视频路径使用 `videos/...`，例如 `videos/streamo/...mp4` 会解析到 `TrainingData/videos/streamo/...mp4`。VideoChat3 使用单独的 `/workspace/s/lzw/datasets/VideoChat3/media` 和来源映射规则。使用其他标注中的 `videos_pool/...` 等前缀时，应按真实布局配置 `options.path_replacements` 或精确映射。

`path_map` 可以是 Hydra 字典，也可以是 JSON 文件路径。JSON 结构为 `{"source/video_source/video_name": "使用者填写的实际文件或目录路径"}`。视频与帧目录都使用相同规则。

`missing_video=skip` 默认跳过空路径和不存在的媒体，并打印跳过数；`error` 则报告标注文件/行号后停止。若没有任何可用样本，训练明确报错，不生成虚构样本。流式样本预处理中的 `IndexError`、`RuntimeError`、`VideoFrameTimeError` 会记录告警后跳过整条样本并补取其他会话；其他未处理的字段、答案或模板错误继续报错。标注尾部取整超出媒体时长的兼容范围由 `data.streaming_video.end_tolerance` 控制，默认一个采样周期；仅复用最后历史帧，超过容差抛出 `VideoFrameTimeError` 并按上述规则跳过样本。

先只读检查本地标注，无需视频或训练依赖：

```bash
python examples/streaming_video_sft/prepare_data.py \
  --annotations ../../../datasets/Stream-VideoLLM/annotations_curated \
  --fps 4 --round-seconds 1 \
  --inspect-only --limit-sessions 64
```

在服务器检查真实路径并新建过滤后的标注：

```bash
python examples/streaming_video_sft/prepare_data.py \
  --annotations /workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData/annotations_curated \
  --video-root /workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData \
  --path-map /workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData/paths.json \
  --fps 4 --round-seconds 1 \
  --missing-video skip \
  --output /workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData/resolved_train.jsonl
```

`paths.json` 需先由使用者填写实际映射；没有映射需要时省略该选项。`--path-replacements` 接受 JSON 对象或 JSON 文件。检查统计中的 `conversation_samples` 是标注会话数，`frame_rounds` 与 `sampled_frames` 是逐帧交互数；这些统计尚未应用 Tokenizer/Processor 的长度过滤。`--output` 只创建新文件，存在时停止，不覆盖原标注。输出仍是 stream-meta-v1，不生成逐轮独立 QA 或预解码图片。

## 因果性、模板与监督

流式每条有效标注对应一个完整训练样本：system 后依次放入所有采样帧的 user-assistant 交互。每条 user 消息包含 `<Xs>` 时间戳、当前一帧图像及此时到达的问题；assistant 输出 `</Silence>` 或 `</Response> 回答文本`。不逐轮拆样本，不设置历史窗口，也不使用离线数据的帧数上限。

视频使用 torchcodec 的精确按播放时间取帧接口，并检查源帧 PTS 不晚于请求时间。如果接口返回未来帧，仅在此异常路径通过 `get_frame_at()` 按真实 PTS 二分定位最近的历史帧，保持 `PTS <= requested`，不使用平均 FPS 推算帧号，也不放宽未来帧容差。恢复成功时记录视频、请求时间、原始 PTS 与选中 PTS；无可用历史帧时抛出 `VideoFrameTimeError`。帧序列使用 `floor(t * source_fps)`。每帧带自身的 `<Xs>` 时间提示，不采用 `time-1` 偏移。未来问题不会提前注入。完整会话通过因果注意力训练，各轮只能关注自身及之前的 token；各帧作为独立图像输入视觉编码器。

取样时使用 Processor 保存的 [Qwen3-VL 官方模板](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/chat_template.json)，通过 Tokenizer 做纯文本渲染，再一次编码完整流式会话，根据消息边界定位所有 assistant 正文与结束符。`labels` 默认全部为 `-100`，仅所有 assistant 的状态、回答和结束符保留对应 Token ID；system、user、图像及角色头均不计 loss。`loss_mask` 与 `labels != -100` 保持一致，每轮首个状态 Token 单独标记到 `stream_state_mask`，历史回答同样参与监督。纯文本渲染避开 Processor 对模板未使用的 `enable_thinking` 发出的参数位置警告，离线与推理路径使用同样方式。

`MAX_LENGTH` 限制包含视觉 Token 的完整序列。初始化只扫描标注并建立候选索引。DataLoader 取样后展开逐帧会话，按媒体尺寸计算视觉 Token 数，执行一次 Tokenization 并构造 labels，然后按 `max(len(input_ids), len(labels)) > MAX_LENGTH` 抛出 `MaxLengthError`。合格的同一份编码直接用于后续训练，超长会话整条丢弃；不截断、不删历史、不降采样。只有通过检查的样本才解码完整帧并执行一次视觉预处理，实际网格必须与长度估计一致。处理顺序参考 [Streamo 长度检查](https://github.com/maifoundations/Streamo/blob/main/swift/llm/template/base.py#L1177-L1220)。

默认 `STREAMING_TRUNCATION=filter`，遇到 `MaxLengthError` 后按候选索引确定性补取同一流式数据源的其他会话，参考 [Streamo LazyLLMDataset](https://github.com/maifoundations/Streamo/blob/main/swift/llm/dataset/utils.py)。流式视频预处理出现 `IndexError`、`RuntimeError` 或 `VideoFrameTimeError` 时，也直接丢弃整条会话并补取，不返回部分帧或空样本；日志记录 rank、来源内索引、标注位置、视频路径、异常类型和原因，同时重置当前读取器，清理失败的解码状态及帧缓存。这样每个 DataLoader 槽位返回一条有效样本，保持各 rank 的 batch 大小和训练步数一致。`STREAMING_MAX_REFETCH=100` 限制每槽最多检查的候选数，候选均不可用时明确报错；`STREAMING_TRUNCATION=error` 仍直接抛出首次超长错误，上述三类预处理异常仍允许补取。其他未处理的预处理异常及模型训练阶段的异常继续报错。训练入口遇到致命异常时打印完整堆栈、刷新日志并以非零状态立即退出，由 torchrun 终止其余 rank，避免异常后的进程组清理继续等待集合通信。

流式 `len(dataset)`、`max_samples`、sampler 和 epoch 步数均按候选会话槽位计数，补取可能使有效会话在同一 epoch 重复出现。worker 会记住已判定的超长及上述三类预处理异常样本索引，以跳过重复处理，超长首次过滤及每累计 1000 条打印一次日志；这些计数属于当前 worker，不是全局唯一过滤数量。初始化仍需读取各来源的标注并检查路径，整个流程不再做全量流式 Tokenization；每次实际读取的流式候选至多编码一次，不跨 epoch 缓存所有有效编码。

Qwen3-VL 使用项目现有 `get_rope_index`，产生视觉三轴与文本位置的 `[4, L]` 位置编码。训练显式提供位置编码，因此关闭 Processor 的 `mm_token_type_ids`；推理依靠模型自动生成位置编码，保留该字段并在追加 Response Token 时同步扩展。两条路径使用相同时间/消息格式。

特殊 Token 在数据编码前注册到 Tokenizer，并与 Processor 共享。模型加载原始权重后、FSDP 包装前按需扩展 Embedding 和 lm_head；Qwen 预留词表行只增不缩。Checkpoint 保存修改后的 Tokenizer、Processor 和模型配置。

## Loss

沿用 Streamo 中“只对状态 Token 加 focal 权重”的机制，状态集合改成数据实际拥有的两类，并由逐 Token 的 `stream_state_mask` 明确区分两种监督：

```text
离线 QA 所有目标 Token、流式回答及结束符：CE_i = -log p(target_i)
流式 stream_state_mask 指定的 Silence/Response：alpha_c * (1 - p(target_i))^gamma * CE_i
gamma 默认 2
alpha_c = 当前全局 batch 的状态 Token 总数 / (出现的状态类别数 * 类别 c 的数量)
```

每个完整流式会话有多少个逐帧轮次，就有多少个受监督的状态 Token，所有轮次均参与类别计数。离线 QA 的 `stream_state_mask` 全零，即使答案文字包含 `</Silence>` / `</Response>` 也使用普通 CE。计数在拆 micro-batch 前计算，并跨数据并行组同步，整个梯度累积 batch 使用相同 alpha。最终将加权 CE 求和，除以流式与离线数据的全局有效监督 Token 数，再结合 FSDP 的梯度平均缩放；不会按 micro-batch 重复平均，也不按输入总长度稀释 loss。`streaming_loss.auto_alpha=false` 可关闭类别平衡。

## 训练命令与参数管理

### 1. 在 train.sh 中集中设置参数

先完成前面的 uv 环境配置，再打开 `examples/streaming_video_sft/train.sh`。脚本顶部按用途列出默认值，通常只需编辑这些变量，无需在启动命令后重复填写几十个 `key=value`：

| 脚本分组 | 关键变量及默认值 |
| --- | --- |
| 分布式启动 | `NNODES=1`、`NODE_RANK=0`、`NPROC_PER_NODE=8`、`MASTER_ADDR=127.0.0.1`、`MASTER_PORT=29500` |
| 数据集入口与路径 | `DATASET_CONFIG`、`STREAMING_ANNOTATIONS`、`STREAMING_VIDEO_ROOT`、`VIDEOCHAT3_ANNOTATION_ROOT`、`VIDEOCHAT3_MEDIA_ROOT` |
| 抽帧与超长处理 | `STREAMING_FPS=4.0`、`STREAMING_TRUNCATION=filter`、`STREAMING_MAX_REFETCH=100`、`VIDEOCHAT3_FPS=2.0`、`OFFLINE_QA_FPS=1.0` |
| 帧数与分辨率 | 离线视频 `VIDEO_MAX_FRAMES=256`；共享 `FRAME_MIN_PIXELS=28*28`、`FRAME_MAX_PIXELS=448*448`、`MISSING_MEDIA=skip` |
| 序列与 DataLoader | `MAX_LENGTH=32768`、`USE_DYNAMIC_BSZ=true`、`MAX_TOKEN_LEN_PER_GPU=$MAX_LENGTH`、`NUM_WORKERS=4`、`TRAIN_MAX_SAMPLES=-1` |
| 模型与加速 | `MODEL_PATH`、`ATTN_IMPLEMENTATION=flash_attention_2`、`USE_LIGER=true`、`USE_FUSED_KERNELS=true`、`FUSED_KERNEL_BACKEND=triton`、`GRADIENT_CHECKPOINTING=true` |
| 训练与优化器 | `GLOBAL_BATCH_SIZE=128`、`MICRO_BATCH_SIZE=1`、`EPOCHS=1`、`TOTAL_TRAINING_STEPS=null`、`LEARNING_RATE=1e-5`、`WARMUP_STEPS=200`、`WARMUP_RATIO=0.0`、`LR_SCHEDULER=cosine`、`WEIGHT_DECAY=0.01`、`CLIP_GRAD=1.0`、`SEED=1` |
| FSDP 引擎 | `ENGINE_STRATEGY=fsdp2`、`ENGINE_DTYPE=bfloat16`、`MODEL_DTYPE=fp32`、`FSDP_SIZE=-1`、`RESHARD_AFTER_FORWARD=false`、`GRADIENT_ACCUMULATION_NO_SYNC=true`；三项 CPU offload 默认关闭 |
| 流式 Loss | `FOCAL_GAMMA=2.0`、`AUTO_ALPHA=true` |
| 实验与日志 | `PROJECT_NAME=streaming-video-sft`、`EXPERIMENT_NAME=qwen3-vl-2b-streaming-baseline`、`OUTPUT_ROOT=/workspace/s/lzw/output/StreamMLLM`、`TRAIN_LOGGERS=[console,tensorboard]` |
| Checkpoint 与验证 | `SAVE_FREQ=500`、`MAX_CKPT_TO_KEEP=1`、`SAVE_CONTENTS=[model,optimizer,extra,hf_model]`、`RESUME_MODE=disable`、`RESUME_FROM_PATH=null`、`VAL_DATASET_CONFIG=null`、`TEST_FREQ=-1` |

变量采用 `NAME="${NAME:-默认值}"`，保留环境变量覆盖。优先级为 **命令末尾的 Hydra 参数 > 环境变量 > train.sh 默认值**。例如 `GLOBAL_BATCH_SIZE=64` 可临时改变 batch；脚本末尾追加 `optim.lr=5e-6` 可临时覆盖学习率。`streaming_sft_trainer.yaml` 保留对应基础配置，供直接调用 Python 入口和 Hydra 使用。

默认模型为本地 Qwen3-VL-2B-Instruct 快照。`MODEL_DTYPE=fp32` 是加载权重的类型，`ENGINE_DTYPE=bfloat16` 是训练混合精度；它们控制不同阶段。`USE_DYNAMIC_BSZ=true` 时按 Token 预算拆分 micro-batch，`MICRO_BATCH_SIZE` 仅用于关闭动态 batch 后的静态模式。

默认固定 warmup 200 步；如需按总步数比例设置，使用 `WARMUP_STEPS=-1 WARMUP_RATIO=0.05`，避免固定步数覆盖比例设置。

默认数据路径：

```text
STREAMING_VIDEO_ROOT=/workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData
STREAMING_ANNOTATIONS=$STREAMING_VIDEO_ROOT/annotations_curated
VIDEOCHAT3_ANNOTATION_ROOT=/workspace/s/lzw/datasets/VideoChat3/videochat3_data_annotations
VIDEOCHAT3_MEDIA_ROOT=/workspace/s/lzw/datasets/VideoChat3/media
```

`datasets.yaml` 继续管理数据源列表、启用状态、模态、标注相对名称、重复次数和来源特例。混合加载器为清单提供运行时 `data` 配置，清单中的 `${data.*}` 引用会在加载时解析。例如 `${data.streaming_video.fps}` 使用脚本设置的流式 FPS，`${data.videochat3.annotation_root}` 使用脚本设置的标注根目录。通用 FPS、离线帧数上限、流式超长处理和路径均使用引用；Atomic 4/8/30 FPS 和 MotionBench 4 FPS 保留各自数值。自定义清单若填写固定数值，会覆盖对应全局设置；希望随脚本变化时使用同样的引用格式。

先抽查流式标注与真实路径，去掉 `--limit-sessions 128` 可检查全部会话：

```bash
"$PYTHON_BIN" examples/streaming_video_sft/prepare_data.py \
  --annotations /workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData/annotations_curated \
  --video-root /workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData \
  --fps 4 --round-seconds 1 \
  --missing-video error --limit-sessions 128
```

该命令不修改标注，也不解码视频。当前默认清单有 193 个来源（1 个流式、192 个 VideoChat3）；未下载的来源在清单设置 `enabled: false`，或选用只包含已下载来源的清单。

### 2. 单机 8 卡正式混合 SFT

默认参数已写入脚本，启动只需：

```bash
uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh
```

也可在同一训练环境中直接使用 `bash examples/streaming_video_sft/train.sh`。脚本使用 `PYTHON_BIN` 指定的解释器，默认 `python`，启动 `torch.distributed.run`；torchrun 子进程继承相同环境。本脚本不使用 `VERL_USE_UV`。

后台运行并保存控制台日志：

```bash
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/train_${RUN_TIMESTAMP}.log"

nohup env RUN_TIMESTAMP="$RUN_TIMESTAMP" \
  uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo "PID: $!"
echo "Log: $LOG_FILE"
```

脚本自动为 Checkpoint 添加日期子目录，并默认设置 `TENSORBOARD_DIR=$OUTPUT_ROOT/logs/$RUN_TIMESTAMP`；启动日志会打印数据清单、Checkpoint 与 TensorBoard 路径。TensorBoard 可由 TrainingHub 读取。修改 `TENSORBOARD_DIR` 环境变量可单独指定日志目录。

默认全局 batch 128 分到 8 卡，每卡每次更新处理 16 个样本；动态模式根据各样本长度决定 micro-batch 数，不固定累积次数。不需要单独设置 `gradient_accumulation_steps`。全局 batch 必须是数据并行 world size 的正整数倍；静态模式还要求每卡 batch 能被 `MICRO_BATCH_SIZE` 整除。

### 3. 单卡运行两个 optimizer step

短跑参数用环境变量临时覆盖，仍使用同一脚本：

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
GLOBAL_BATCH_SIZE=2 \
MAX_LENGTH=8192 \
NUM_WORKERS=0 \
TRAIN_MAX_SAMPLES=256 \
TOTAL_TRAINING_STEPS=2 \
SAVE_FREQ=1 \
MAX_CKPT_TO_KEEP=2 \
uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh
```

`MAX_TOKEN_LEN_PER_GPU` 未单独设置时随 `MAX_LENGTH` 使用 8192。默认清单有 193 个启用来源，全局样本上限至少为 193，才能保证每来源一条。`TRAIN_MAX_SAMPLES` 在各来源建立索引后限制混合样本数，短跑仍会扫描启用标注；要减少初始化工作，使用较小清单或各来源的 `max_samples`。单卡容量仍取决于实际序列与视觉输入，以上命令尚未在服务器验证。

### 4. 仅训练流式主动交互数据

将以下内容保存为 `examples/streaming_video_sft/datasets.streaming-only.yaml`：

```yaml
datasets:
  - name: streaming_interaction
    type: streaming
    media_type: video
    enabled: true
    annotations: ${data.streaming_video.annotations}
    video_root: ${data.streaming_video.video_root}
    max_samples: -1
    repeat: 1
    options:
      fps: ${data.streaming_video.fps}
      truncation: ${data.streaming_video.truncation}
      max_refetch: ${data.streaming_video.max_refetch}
      missing_video: ${data.streaming_video.missing_video}
```

切换清单和实验名称即可启动：

```bash
DATASET_CONFIG="$PROJECT_ROOT/examples/streaming_video_sft/datasets.streaming-only.yaml" \
OUTPUT_DIR=/workspace/s/lzw/output/StreamMLLM/Qwen3VL-2B-streaming-only \
EXPERIMENT_NAME=qwen3-vl-2b-streaming-only \
uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh
```

自定义混合清单继续使用相同字段，不改变状态 Focal 与离线 NTP Loss。

### 5. 多机示例：2 节点，每节点 8 卡

各节点使用相同代码、环境、模型及数据路径，Checkpoint 输出目录放在共享存储。以下主节点地址和日期仅为命令示例，需要填写实际值；两台机器必须使用同一个 `RUN_TIMESTAMP`。

两台机器均设置：

```bash
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
export RUN_TIMESTAMP=20261022_120000
```

主节点：

```bash
NODE_RANK=0 uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh
```

另一节点：

```bash
NODE_RANK=1 uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh
```

默认全局 batch 128 分到 16 卡，每卡处理 8 个样本。数据清单、模型、输出目录与训练参数在各节点保持一致；多机未设置统一日期时脚本会报错。

### 6. Checkpoint 输出与断点恢复

默认目录：

```text
/workspace/s/lzw/output/StreamMLLM/
  logs/<YYYYMMDD_HHMMSS>/          # TensorBoard events
  Qwen3VL-2B-baseline/
    <YYYYMMDD_HHMMSS>/
      latest_checkpointed_iteration.txt
      global_step_500/
        huggingface/              # 完整 HF 权重、配置及 Tokenizer/Processor
        ...                       # 模型分片、优化器和训练进度
```

`OUTPUT_DIR` 是实验权重根目录，脚本追加日期子目录。默认每 500 个 optimizer step 保存一次，保留最近 1 个 Checkpoint，包含模型、优化器、训练状态与 HF 导出。`MAX_CKPT_TO_KEEP=null` 保留全部。

恢复时指定已经存在的完整 `global_step_<N>` 目录，并保持原运行的模型、清单、GPU 拓扑、batch 和其他训练参数：

```bash
RESUME_MODE=resume_path \
RESUME_FROM_PATH=/workspace/s/lzw/output/StreamMLLM/Qwen3VL-2B-baseline/20261022_120000/global_step_500 \
uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh
```

默认 `RESUME_MODE=disable` 开始新实验。`resume_path` 从指定目录恢复，结果写入本次日期目录；`auto` 只检查本次输出目录，因此继续同一个目录时还需填写原来的 `OUTPUT_DIR` 与 `RUN_TIMESTAMP`。`huggingface/` 可用于推理或初始化新模型，恢复优化器与进度需要完整 Checkpoint 目录。

`TOTAL_TRAINING_STEPS` 是整个运行的目标 optimizer step 数，恢复时表示累计总步数，不是额外增加的步数；`EPOCHS` 应覆盖足够的 batch。

### 7. 覆盖参数与输入预算

常用配置直接修改脚本，临时实验可用环境变量：

```bash
VIDEO_MAX_FRAMES=128 FRAME_MAX_PIXELS=65536 LEARNING_RATE=5e-6 \
uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh
```

低频或来源专用参数继续支持 Hydra 覆盖，位置在命令最后，优先级最高：

```bash
uv run --frozen --extra fsdp bash examples/streaming_video_sft/train.sh \
  data.streaming_video.cache_frames=256 \
  optim.lr=5e-6
```

`VIDEO_MAX_FRAMES` 通过 `data.video_max_frames` 仅限制离线视频 QA 和 VideoChat3 视频，默认 256，流式会话不设帧数上限。共享的 `FRAME_MIN_PIXELS/FRAME_MAX_PIXELS` 默认 784 / 200704，是逐帧像素面积预算，实际宽高按 Processor 网格对齐。VideoChat3 图片 QA 仍用独立的 `data.videochat3.min_pixels/max_pixels`，默认 4096 / 147456。

`MAX_LENGTH` 限制完整输入序列：流式超限默认过滤整条会话；离线继续均匀减少视频帧，并保留完整监督答案。训练参数 `ROUND_SECONDS`、`WINDOW_ROUNDS` 已移除；旧清单中的对应选项应删除。静态模式下 `MAX_TOKEN_LEN_PER_GPU` 不参与 micro-batch 划分；动态模式下它必须能容纳过滤后的最长样本。

流式、离线 QA 和 VideoChat3 均先根据图像尺寸计算视觉 token 长度，再对保留图片执行一次视觉预处理。流式完整会话使用一次文本编码生成全轮监督掩码；离线完整对话和监督前缀复用相同的视觉 token 长度并核对前缀。实际 `image_grid_thw` 必须与预估一致。

动态模式默认启用 `LENGTH_BUCKET_MICRO_BATCHES=true`，按长度的 2 次幂区间分桶，将相近长度样本放入同一 micro-batch，同时约束有效 token 总数、补齐后的 `batch_size × max_seq_len` 和未合并的视觉 patch 数。`MAX_PADDED_TOKEN_LEN_PER_GPU` 默认等于 `MAX_TOKEN_LEN_PER_GPU`，`MAX_VISION_PATCHES_PER_GPU` 在脚本中默认为其四倍；直接使用 YAML 时视觉预算默认为 131072。超预算的单个样本或不可拆分组会报错，不会删帧或裁剪目标。各 rank 同步 micro-batch 数量，额外拆分保持每个 micro-batch 非空。上述预算只在启用长度分桶的动态模式下生效。

`SFT_SUPERVISED_ONLY=true` 默认只在监督 token 对应的前一位置执行词表投影和 log-prob 计算，并跳过 entropy 的前向与反向计算；完整上下文继续参与 Transformer 和视觉编码器的训练。流式的监督位置包括所有 assistant 轮次，开关不改变监督范围。Triton 后端使用 log-prob 专用分支；torch/liger 后端使用分块 linear/CE。输出恢复为原有 token 布局，状态 focal loss、普通答案 CE、全局监督 token 分母和类别权重保持不变。MFU 估算中的输出头部分按监督 token 数计数。

比较原组批和完整输出头路径时，可设置 `LENGTH_BUCKET_MICRO_BATCHES=false SFT_SUPERVISED_ONLY=false`；保持相同的全局 batch、FPS、像素预算与数据清单。新增路径目前只做静态检查，尚未在服务器验证数值、吞吐和 GPU 满载情况。

需要验证时设置 `VAL_DATASET_CONFIG=/实际路径/validation_datasets.yaml TEST_FREQ=100`，验证清单使用同样格式。`data.train_files` 默认引用 `data.dataset_config`。旧字段 `max_frames`、视频 `min_pixels/max_pixels` 仍可作为兼容入口；同时填写时新字段优先。

直接调用 Python 入口时使用 YAML 配置默认值，需要自行通过 torchrun 启动；自动日期目录和 TensorBoard 目录由 `train.sh` 设置。

## 逐帧主动推理 Demo（增量 KV Cache）

使用上文配置的训练虚拟环境和导出的 Hugging Face checkpoint。当前 Demo 默认对齐 `train.sh`：**1 FPS、每帧一次决策、1,024–200,704 像素、32,768 token、BF16、FlashAttention 2、seed=1**。若该 checkpoint 训练时通过环境变量或 Hydra 覆盖了这些参数，命令也应使用对应实际值。

```bash
cd /workspace/s/lzw/projects/StreamMLLM/projects/Training/verl-main
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" examples/streaming_video_sft/infer.py \
  --model /实际训练checkpoint/huggingface \
  --video /实际存在的本地视频.mp4 \
  --question '持续观察视频，当出现需要报告的新事件时主动回答。' \
  --question-time 0 \
  --fps 1 --max-context-tokens 32768 \
  --frame-min-pixels 1024 --frame-max-pixels 200704 \
  --dtype bfloat16 --attn-implementation flash_attention_2 \
  --context-policy sliding_window --seed 1 \
  --output /新建的输出目录/stream_predictions.jsonl
```

模型、视频与输出路径均需替换为实际路径。`--model` 必须是带训练后特殊 Token 和权重的 HF 目录，不能用原始基础模型代替。问题省略时在开始回放前从终端输入一次；`--question-time` 指定原始到达时间，只有到达该时刻的首个采样轮次才注入问题。`--start` / `--end` 选择回放范围，`--max-rounds` 限制本次处理轮数。`--realtime` 按视频时钟等待，处理落后时继续顺序推理，不跳帧。单实例跨卡可设置 `CUDA_VISIBLE_DEVICES=0,1` 并加 `--device-map auto`；单卡示例沿用 `--device auto`。

每条视频创建一个独立 `DynamicCache`。窗口首帧 prefill 系统提示、当前时间/指令、当前图像和 assistant 前缀；后续轮只处理新增 user 消息和当前图像，并持续传入已有 `past_key_values`。历史帧不会再次经过 processor、视觉编码器或 Transformer 前向。当前图像使用原生 `get_rope_index`，三维 mRoPE 接续当前窗口位置；额外的文本位置平面用于标记 KV 槽位。接口对应锁定的 Transformers 5.12.1 原生 [Qwen3-VL](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)、[Qwen3-VL-MoE](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py) 和 [DynamicCache](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/cache_utils.py)。

每轮只在 `</Silence>` / `</Response>` 两个状态 logits 中取 argmax，随后立即输出 `event="state"`。Silence 会写入状态及原生 assistant 结束标记；Response 先将状态写入同一缓存，再逐 token 生成正文并写回缓存，遇到 EOS 或 `--max-new-tokens` 后闭合轮次，完成时输出 `event="answer"`。答案正文抑制两个交互状态 Token，原始生成 IDs 用于后续缓存与重建，显示用的 decode/strip 不改写历史，也不会额外插入空格。

默认 `--context-policy sliding_window` 在“已有 KV + 当前输入 + 状态/正文/结束标记预算”超出上下文时，从当前轮重建窗口：清空旧 KV，位置从 0 开始，正常构建系统提示，再加入**全部历史 user/assistant 文本、原始问题到达时间、历史帧时间、模型状态和答案**，最后加入当前帧及本轮新问题。历史图像和视觉占位不带入，过去的问题不会丢失，也不会改写为重建时间。重建后的文本和当前帧仍放不下时明确报错，保留已有 JSONL 记录；`--context-policy error` 则超限直接报错。

`--window-rounds`、`--video-max-frames` 保留为可选的视觉窗口上限，默认都不设置；显式设置后，达到上限也按上述规则从当前轮重建并保留全部文本。`--context-policy error` 下触及这些上限同样报错。训练脚本的 `VIDEO_MAX_FRAMES=128` 仅约束离线视频分支，不会自动成为本 Demo 的流式窗口上限。`--system-prompt-file` 可指定训练时相同的自定义提示，默认沿用训练 helper 的两状态系统提示。默认答案解码为 `--temperature 0 --top-p 1 --top-k 0 --repetition-penalty 1`；可覆盖这些参数和 `--max-new-tokens`（默认 256），SFT 没有与生成温度直接对应的训练参数。

JSONL 和控制台事件包括：

| 事件 | 内容 |
| --- | --- |
| `state` | 即时状态，以及当前 `start/timestamp/frame_timestamps`、视觉窗口轮数/帧数、`input_tokens/new_input_tokens/cached_tokens_reused`、重建标记/原因/窗口起点、已保存文本量和状态判断耗时 |
| `answer` | 与同一视频及 `timestamp` 对应的正文、截断标记、轮次闭合后的 KV 长度和该轮处理耗时；不重复 `state` 字段 |
| `summary` | 正常结束时的总轮数、原始问题时间与首次观测时间、累计/峰值/最终 KV token、旧 KV 丢弃量、重建次数、实际编码图像数和本次完整推理协议 |

`state.kv_cache_tokens` 是状态确定时的缓存长度，尚不含该轮状态和正文；窗口首帧的 `cached_tokens_reused=0`，后续帧应大于 0。正常结束时，`summary.cache_statistics.vision_frames_encoded` 应等于处理轮数，且 `processed_tokens = final_cached_tokens + evicted_tokens`。输出文件只新建，不覆盖；正常结束、异常或中断时均释放会话 KV。当前只完成静态核验，真实 GPU 缓存数值一致性、重建行为、解码质量和实时吞吐尚未实测。

## 参考依据与验证范围

主要依据是项目 `docs/Streamo_vs_ms-swift_3.10_20261019/README.md`、`streamo_development.diff` 和 `changed_sources.json` 中的 `stream_focal_loss`、converter、streaming preprocessor、`inference.py`。采用了状态词加权监督和帧到消息的交互方式；未移植 ms-swift Trainer/注册系统、版本演进差异、Standby 弱标签，或原实现的时间偏移、提前问题注入和文件缓存方式。

数据字段与任务按本地 `annotations_refinev2`、`docs/VideoLLM-Stream设计方案.md` 及 `docs/数据集.xlsx` 表二核对。标注检查、Python 语法/静态接口和启动脚本检查在本地完成；未安装任何依赖或运行 GPU 训练。服务器模型路径、真实视频解码、完整分布式反向、HF Checkpoint 再加载及生成质量尚需在服务器执行上面的短跑命令验证。没有增加测试文件、评测系统或部署组件。

2026-10-22 的完整会话训练改为 DataLoader 内单次编码与懒过滤，保留全轮状态监督、Focal/CE 与全局 Token 归一化。3 个修改的 Python 文件通过 AST/编译、符号引用、120 列和空白检查，`train.sh` 通过 `bash -n`；已静态核对初始化没有完整会话预处理调用、流式会话只有一个文本编码调用点，以及参数引用。未运行项目代码、测试或训练，未创建测试文件、安装依赖或新环境。真实模板编码、媒体解码、混合 DataLoader、分布式反向、数值与吞吐尚未实测。

2026-10-20 的 VideoChat3 适配曾读取官方 210 个 JSONL 各自的首条记录，观察到字符串及 text/image_url/video_url 列表；这是格式抽查，未全量验证所有记录。来源映射覆盖全部 210 个已发布标注文件，当前启用项以 `datasets.yaml` 为准。

当前训练入口使用 `annotations_curated`。此前只读标注检查覆盖 67 个 JSONL、192901 条会话，视频路径为非空的 `videos/...` 相对路径，3 个文件为空。旧版按秒分轮、逐轮拆样本的计数不代表当前逐帧轮数或有效会话数；初始化日志报告候选数量，超长过滤数量在实际取样时记录。
