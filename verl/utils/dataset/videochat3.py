# Copyright 2026 The VideoLLM-Stream authors
# Licensed under the Apache License, Version 2.0.

"""VideoChat3 text, image and offline-video conversations with ordinary NTP supervision."""

import bisect
import json
import logging
from collections import OrderedDict
from pathlib import Path

from .offline_video_qa import OfflineVideoQASFTDataset, sample_offline_timestamps
from .streaming_video import EmptyDatasetError, load_annotations
from .videochat3_media import VideoChat3Media

logger = logging.getLogger(__name__)


def normalize_videochat3_record(record):
    messages = []
    for message in record["messages"]:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported VideoChat3 role: {role!r}")
        parts = message["content"]
        if isinstance(parts, str):
            parts = [{"type": "text", "text": parts}]
        content = []
        for part in parts:
            kind = part["type"]
            if kind == "text":
                text = part["text"]
                if role == "user":
                    text = text.replace("<IMG_CONTEXT>", "").replace("<VIDEO_CONTEXT>", "")
                content.append({"type": "text", "text": text})
            elif kind in {"image_url", "video_url"} and role == "user":
                reference = part[kind]
                path = reference["url"] if isinstance(reference, dict) else reference
                if not isinstance(path, str) or not path.strip():
                    raise ValueError("VideoChat3 media references must be nonempty path strings")
                content.append({
                    "type": "image" if kind == "image_url" else "video", "path": path,
                    "metadata": dict(part.get("video_metadata") or {}),
                })
            else:
                raise ValueError(f"Unsupported VideoChat3 content {kind!r} in role {role!r}")
        if role != "user":
            content = "".join(part["text"] for part in content)
            if role == "assistant" and not content.strip():
                raise ValueError("VideoChat3 assistant answers must not be empty")
        messages.append({"role": role, "content": content})
    targets = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    first_user = next((i for i, message in enumerate(messages) if message["role"] == "user"), None)
    if first_user is None or not targets or targets[0] < first_user:
        raise ValueError("VideoChat3 requires a user message before its assistant targets")
    return messages, targets


class VideoChat3SFTDataset(OfflineVideoQASFTDataset):
    """Index JSONL offsets instead of retaining the multi-gigabyte text annotations in memory."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if processor is None:
            raise ValueError("VideoChat3SFTDataset requires a Qwen3-VL processor")
        self.tokenizer, self.processor = tokenizer, processor
        self.options = dict(config.get("videochat3") or {})
        self.media_type = self.options.get("media_type", "auto")
        if self.media_type not in {"image", "video", "text", "mixed", "auto"}:
            raise ValueError(f"Unsupported VideoChat3 media_type: {self.media_type!r}")
        self.annotation_root = Path(self.options["annotation_root"]).expanduser().resolve()
        with Path(self.options["source_map"]).open(encoding="utf-8") as stream:
            self.source_map = json.load(stream)["sources"]
        self.media = VideoChat3Media.shared(self.options["video_root"], self.options.get("cache_archives", 2))
        self.fps = float(self.options.get("fps", 2.0))
        self.max_frames = int(self.options.get("video_max_frames", self.options.get("max_frames", 256)))
        sample_offline_timestamps(1.0, self.fps, self.max_frames)
        self.max_length = int(self.options.get("max_length", config.get("max_length", 8192)))
        self.pad_mode = config.get("pad_mode", "no_padding")
        if self.pad_mode not in {"right", "no_padding"}:
            raise ValueError("VideoChat3 supports right or no_padding")
        self.system_prompt = self.options.get("system_prompt") or "You are a helpful assistant."
        self.chat_kwargs = dict(config.get("apply_chat_template_kwargs") or {})
        self.chat_kwargs.setdefault("enable_thinking", False)
        self.image_kwargs = dict(
            min_pixels=int(self.options.get("min_pixels", 4096)),
            max_pixels=int(self.options.get("max_pixels", 147456)),
        )
        self.frame_kwargs = dict(
            min_pixels=int(self.options.get("frame_min_pixels", self.options.get("min_pixels", 28 * 28))),
            max_pixels=int(self.options.get("frame_max_pixels", self.options.get("max_pixels", 448 * 448))),
        )
        self.missing_media = self.options.get("missing_media", "skip")
        if self.missing_media not in {"skip", "error"}:
            raise ValueError("missing_media must be skip or error")
        self.references, self.cumulative_turns, self.sessions = [], [], OrderedDict()
        self.skipped = 0
        self.invalid_records = 0
        total = 0
        for record, reference, line_number in load_annotations(parquet_files):
            location = f"{reference[0]}:{line_number}"
            try:
                source_key = Path(reference[0]).resolve().relative_to(self.annotation_root).as_posix()
                source = self.source_map[source_key]
                source_media_type = {
                    "image": "image", "video": "video", "motion-videos": "video", "text": "text",
                }[source_key.split("/", 1)[0]]
                if self.media_type not in {"auto", "mixed", source_media_type}:
                    raise ValueError(
                        f"Declared media_type={self.media_type!r}, but source {source_key!r} is {source_media_type!r}"
                    )
            except (KeyError, ValueError) as error:
                raise ValueError(f"Invalid VideoChat3 source at {location}: {error}") from error
            try:
                messages, targets = normalize_videochat3_record(record)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
                self.invalid_records += 1
                logger.warning("Skipping invalid VideoChat3 annotation at %s: %s", location, error)
                continue
            messages = self._resolve_media(messages, source)
            if messages is None:
                self.skipped += 1
                continue
            count = len(targets)
            if max_samples > 0:
                count = min(count, max_samples - total)
            total += count
            self.references.append((reference, source_key, count))
            self.cumulative_turns.append(total)
            if max_samples > 0 and total >= max_samples:
                break
        logger.warning(
            "VideoChat3: %d answer samples, %d records, %d missing-media records skipped, %d invalid records skipped",
            total, len(self.references), self.skipped, self.invalid_records,
        )
        if not total:
            raise EmptyDatasetError("No usable VideoChat3 samples. Check annotations, media root and manifest files.")

    def _resolve_media(self, messages, source):
        for message in messages:
            if message["role"] != "user":
                continue
            for part in message["content"]:
                if part["type"] == "text":
                    continue
                resource = self.media.resolve(source, part["path"])
                if resource is None:
                    if self.missing_media == "error":
                        raise FileNotFoundError(f"Missing VideoChat3 media: {source['media_subdir']}/{part['path']}")
                    return None
                part["resource"] = resource
                if part["type"] == "video":
                    part["metadata"].setdefault("video_backend", source.get("video_read_type", "decord"))
        return messages

    def _session(self, index):
        if index not in self.sessions:
            (filename, position), source_key, count = self.references[index]
            with open(filename, "rb") as stream:
                if filename.lower().endswith(".jsonl"):
                    stream.seek(position)
                    record = json.loads(stream.readline().decode("utf-8-sig"))
                else:
                    records = json.load(stream)
                    record = records[position] if isinstance(records, list) else records
            messages, targets = normalize_videochat3_record(record)
            messages = self._resolve_media(messages, self.source_map[source_key])
            if messages is None:
                raise FileNotFoundError(f"VideoChat3 media disappeared after indexing: {filename}:{position}")
            self.sessions[index] = (messages, targets[:count], f"{filename}@{position}")
        self.sessions.move_to_end(index)
        session = self.sessions[index]
        while len(self.sessions) > max(int(self.options.get("cache_sessions", 8)), 1):
            self.sessions.popitem(last=False)
        return session

    def __getitem__(self, item):
        if item < 0 or item >= len(self):
            raise IndexError(item)
        session_index = bisect.bisect_right(self.cumulative_turns, item)
        target_index = item - (self.cumulative_turns[session_index - 1] if session_index else 0)
        conversation, targets, location = self._session(session_index)
        conversation = conversation[: targets[target_index] + 1]
        visuals = {}
        for i, message in enumerate(conversation):
            if message["role"] != "user":
                continue
            for j, part in enumerate(message["content"]):
                if part["type"] == "image":
                    visuals[i, j] = (None, [self.media.image(part["resource"])])
                elif part["type"] == "video":
                    visuals[i, j] = self.media.video(part["resource"], part["metadata"], self.fps, self.max_frames)
        image_kwargs = (
            self.frame_kwargs if any(times is not None for times, _ in visuals.values()) else self.image_kwargs
        )
        while True:
            messages, images = [], []
            for i, message in enumerate(conversation):
                if message["role"] != "user":
                    messages.append(dict(message))
                    continue
                content = []
                for j, part in enumerate(message["content"]):
                    if part["type"] == "text":
                        content.append(dict(part))
                        continue
                    times, frames = visuals[i, j]
                    for k, image in enumerate(frames):
                        if times is not None:
                            content.append({"type": "text", "text": f"<{times[k]:g}s>"})
                        content.append({"type": "image", "image": image})
                        images.append(image)
                messages.append({"role": "user", "content": content})
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            result = self._encode(messages, images, location, image_kwargs=image_kwargs)
            if result is not None:
                return result
            reduced = False
            for key, (times, frames) in visuals.items():
                if times is not None and len(frames) > 1:
                    count = max(1, len(frames) // 2)
                    indices = (
                        [len(frames) // 2] if count == 1
                        else [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
                    )
                    visuals[key] = ([times[i] for i in indices], [frames[i] for i in indices])
                    reduced = True
            if not reduced:
                raise ValueError(
                    f"Image/video QA example at {location} exceeds max_length={self.max_length}; "
                    "increase the token budget or reduce image max_pixels. Targets are never truncated."
                )
