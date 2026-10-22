# Copyright 2026 The VideoLLM-Stream authors
# Licensed under the Apache License, Version 2.0.

"""Offline video QA samples with answer-only next-token supervision."""

import bisect
import logging
import math
from pathlib import Path

from verl.utils.dataset.streaming_video import (
    DEFAULT_VIDEO_ROOT,
    EmptyDatasetError,
    VideoFrameReader,
    load_annotations,
    load_path_mapping,
    resolve_video_path,
)
from verl.utils.dataset.video_sft_processing import encode_image_text, image_token_lengths, process_sft_images

logger = logging.getLogger(__name__)
DEFAULT_SYSTEM_PROMPT = "You are a helpful video assistant. Answer the user's question based on the video."


def normalize_offline_record(record):
    """Accept LLaVA conversations, role/content messages, or question/answer records."""
    raw_video = record.get("video_path") or record.get("video") or record.get("videos")
    if isinstance(raw_video, list):
        if len(raw_video) > 1:
            raise ValueError("Offline video QA expects one video per record")
        raw_video = raw_video[0] if raw_video else None
    if raw_video is not None and not isinstance(raw_video, str):
        raise ValueError("video/video_path must be a string or a single-element string list")
    raw_messages = record.get("conversations") or record.get("messages")
    if not raw_messages:
        if "question" not in record or "answer" not in record:
            raise ValueError("Expected conversations, messages, or question/answer fields")
        raw_messages = [
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ]
    messages = []
    for message in raw_messages:
        role = message.get("role", message.get("from", ""))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported offline conversation role: {role!r}")
        content = message.get("content", message.get("value"))
        if isinstance(content, list):
            texts = []
            for part in content:
                if part.get("type") == "text":
                    texts.append(part["text"])
                elif part.get("type") in ("video", "video_url") and role == "user":
                    video = part.get("video", part.get("video_url"))
                    video = video.get("url") if isinstance(video, dict) else video
                    if raw_video and video and raw_video != video:
                        raise ValueError("Conflicting video paths in the offline record")
                    raw_video = raw_video or video
                else:
                    raise ValueError("Offline messages support text and one video reference")
            content = "\n".join(texts)
        if not isinstance(content, str):
            raise ValueError("Offline conversation text must be a string")
        if role == "user":
            content = content.replace("<video>", "").replace("<image>", "").strip()
        if role == "assistant" and not content.strip():
            raise ValueError("Offline assistant answers must not be empty")
        messages.append({"role": role, "content": content})
    first_user = next((i for i, message in enumerate(messages) if message["role"] == "user"), None)
    targets = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    if first_user is None or not targets or targets[0] < first_user:
        raise ValueError("Offline video QA requires a user question before the first assistant answer")
    if raw_video is not None and not isinstance(raw_video, str):
        raise ValueError("The offline video reference must be a path string")
    return raw_video, messages, targets


def sample_offline_timestamps(duration, fps=1.0, max_frames=256):
    """Use ceil(duration * fps) frames, capped and uniformly spread over the complete video."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Expected positive video duration, got {duration}")
    if not math.isfinite(fps) or fps <= 0 or max_frames <= 0:
        raise ValueError("Offline fps and max_frames must be positive")
    count = min(max_frames, max(1, math.ceil(duration * fps)))
    if count == 1:
        return [duration / 2]
    last = math.nextafter(duration, -math.inf)
    return [index * last / (count - 1) for index in range(count)]


class OfflineVideoQASFTDataset:
    """One sample per assistant turn; the full video and earlier turns form its masked prefix."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if processor is None:
            raise ValueError("OfflineVideoQASFTDataset requires a Qwen3-VL processor")
        self.tokenizer, self.processor = tokenizer, processor
        self.config = config or {}
        self.options = dict(self.config.get("offline_video_qa") or {})
        shared = self.config.get("streaming_video") or {}
        self.fps = float(self.options.get("fps", 1.0))
        self.max_frames = int(self.options.get("video_max_frames", self.options.get("max_frames", 256)))
        sample_offline_timestamps(1.0, self.fps, self.max_frames)
        self.max_length = int(self.options.get("max_length", self.config.get("max_length", 8192)))
        self.pad_mode = self.config.get("pad_mode", "no_padding")
        if self.pad_mode not in ("right", "no_padding"):
            raise ValueError("Offline video QA supports right or no_padding")
        self.system_prompt = self.options.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        self.chat_kwargs = dict(self.config.get("apply_chat_template_kwargs", {}))
        self.chat_kwargs.setdefault("enable_thinking", False)
        frame_min_pixels = shared.get("frame_min_pixels", shared.get("min_pixels", 28 * 28))
        frame_max_pixels = shared.get("frame_max_pixels", shared.get("max_pixels", 448 * 448))
        self.image_kwargs = dict(
            min_pixels=int(self.options.get("frame_min_pixels", self.options.get("min_pixels", frame_min_pixels))),
            max_pixels=int(self.options.get("frame_max_pixels", self.options.get("max_pixels", frame_max_pixels))),
        )
        path_map = load_path_mapping(self.options.get("path_map"))
        replacements = dict(self.options.get("path_replacements") or {})
        missing_video = self.options.get("missing_video", "skip")
        if missing_video not in ("skip", "error"):
            raise ValueError("missing_video must be skip or error")
        self.records, self.cumulative_turns = [], []
        self.reader = None
        self.skipped = 0
        self.invalid_records = 0
        total = 0
        for record, reference, line_number in load_annotations(parquet_files):
            try:
                raw_video, messages, targets = normalize_offline_record(record)
                if raw_video and self.options.get("prepend_data_source", False) and not Path(raw_video).is_absolute():
                    if not record.get("data_source"):
                        raise ValueError("prepend_data_source requires a data_source field")
                    raw_video = str(Path(record["data_source"]) / raw_video)
                path_record = dict(record, video_path=raw_video, video_name=record.get("id", ""))
                path = resolve_video_path(
                    path_record, self.options.get("video_root", f"{DEFAULT_VIDEO_ROOT}/videos"), path_map, replacements
                )
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
                self.invalid_records += 1
                logger.warning("Skipping invalid offline QA annotation at %s:%d: %s", reference[0], line_number, error)
                continue
            if not path or not Path(path).is_file():
                self.skipped += 1
                if missing_video == "error":
                    raise FileNotFoundError(f"Missing offline video at {reference[0]}:{line_number}: {path!r}")
                continue
            if max_samples > 0:
                targets = targets[: max_samples - total]
            total += len(targets)
            self.records.append((path, messages, targets, f"{reference[0]}:{line_number}"))
            self.cumulative_turns.append(total)
            if max_samples > 0 and total >= max_samples:
                break
        logger.warning(
            "Offline video QA dataset: %d answer samples, %d records, "
            "%d missing videos skipped, %d invalid records skipped",
            total, len(self.records), self.skipped, self.invalid_records,
        )
        if total == 0:
            raise EmptyDatasetError(
                "No usable offline video QA samples. Check annotation files and video_root/path_map."
            )

    def __len__(self):
        return self.cumulative_turns[-1]

    def __getitem__(self, item):
        if item < 0 or item >= len(self):
            raise IndexError(item)
        record_index = bisect.bisect_right(self.cumulative_turns, item)
        target_index = item - (self.cumulative_turns[record_index - 1] if record_index else 0)
        path, conversation, targets, location = self.records[record_index]
        if self.reader is None:
            self.reader = VideoFrameReader(cache_frames=int(self.options.get("cache_frames", 64)))
        # Opening through the shared reader retains its worker-local decoder for the requested frames.
        self.reader.read(path, [0.0])
        metadata = self.reader.decoder.metadata
        duration = metadata.end_stream_seconds - metadata.begin_stream_seconds
        timestamps = sample_offline_timestamps(duration, self.fps, self.max_frames)
        images = self.reader.read(path, timestamps)
        while True:
            messages = [dict(message) for message in conversation[: targets[target_index] + 1]]
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            first_user = next(message for message in messages if message["role"] == "user")
            content = [{"type": "text", "text": f"Video duration: {duration:g}s.\n{first_user['content']}"}]
            for timestamp, image in zip(timestamps, images, strict=True):
                content.extend([
                    {"type": "text", "text": f"<{timestamp:g}s>"},
                    {"type": "image", "image": image},
                ])
            first_user["content"] = content
            result = self._encode(messages, images, location)
            if result is not None:
                return result
            if len(images) == 1:
                raise ValueError(
                    f"Offline video QA at {location} with one frame exceeds "
                    f"max_length={self.max_length}; increase max_length or reduce image max_pixels"
                )
            count = max(1, len(images) // 2)
            indices = (
                [len(images) // 2] if count == 1
                else [round(index * (len(images) - 1) / (count - 1)) for index in range(count)]
            )
            timestamps = [timestamps[index] for index in indices]
            images = [images[index] for index in indices]

    def _encode(self, messages, images, location, image_kwargs=None):
        """Encode a complete assistant target; return None if the intact example exceeds the budget."""
        import torch
        import torch.nn.functional as F

        text = self.tokenizer.apply_chat_template(
            messages, chat_template=self.processor.chat_template,
            tokenize=False, add_generation_prompt=False, **self.chat_kwargs,
        )
        image_kwargs = self.image_kwargs if image_kwargs is None else image_kwargs
        image_lengths = image_token_lengths(self.processor, images, image_kwargs)
        inputs = encode_image_text(self.processor, text, image_lengths)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]
        if len(input_ids) > self.max_length:
            return None
        prefix = self.tokenizer.apply_chat_template(
            messages[:-1], chat_template=self.processor.chat_template,
            tokenize=False, add_generation_prompt=True, **self.chat_kwargs,
        )
        prefix_ids = encode_image_text(self.processor, prefix, image_lengths)["input_ids"][0]
        if not torch.equal(input_ids[: len(prefix_ids)], prefix_ids):
            raise ValueError(f"Chat template does not preserve the generation prefix at {location}")
        inputs.update(process_sft_images(self.processor, images, image_kwargs, image_lengths))
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[len(prefix_ids) :] = 1
        if not bool(loss_mask.any()):
            raise ValueError(f"The assistant target must contain at least one token at {location}")
        labels = input_ids.masked_fill(~loss_mask.bool(), -100)
        inputs.pop("mm_token_type_ids", None)
        from verl.models.transformers.qwen3_vl import get_rope_index

        vision_positions = get_rope_index(
            self.processor, input_ids=input_ids, attention_mask=attention_mask,
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        position_ids = torch.cat((torch.arange(len(input_ids)).unsqueeze(0), vision_positions), dim=0)
        result = dict(
            input_ids=input_ids, position_ids=position_ids, labels=labels, loss_mask=loss_mask,
            stream_state_mask=torch.zeros_like(input_ids), multi_modal_inputs=inputs,
        )
        if self.pad_mode == "right":
            padding = self.max_length - len(input_ids)
            result["input_ids"] = F.pad(input_ids, (0, padding), value=self.tokenizer.pad_token_id or 0)
            result["position_ids"] = F.pad(position_ids, (0, padding), value=0)
            result["labels"] = F.pad(labels, (0, padding), value=-100)
            result["loss_mask"] = F.pad(loss_mask, (0, padding), value=0)
            result["stream_state_mask"] = F.pad(result["stream_state_mask"], (0, padding), value=0)
            result["attention_mask"] = F.pad(attention_mask, (0, padding), value=0)
        return result
