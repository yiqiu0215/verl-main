# Copyright 2026 The VideoLLM-Stream authors
# Licensed under the Apache License, Version 2.0.

"""Ordinary image/video QA from Spatial JSONL and ET-Instruct JSON annotations."""

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path

from .offline_video_qa import sample_offline_timestamps
from .streaming_video import EmptyDatasetError
from .videochat3 import VideoChat3SFTDataset
from .videochat3_media import VideoChat3Media

logger = logging.getLogger(__name__)


def indexed_qa_records(paths):
    for path in paths:
        path = Path(path)
        filename = str(path)
        if path.suffix.lower() == ".jsonl":
            with path.open("rb") as stream:
                line_number = 0
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    line_number += 1
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line.decode("utf-8-sig"))
                    except ValueError as error:
                        logger.warning("Skipping invalid QA annotation at %s:%d: %s", filename, line_number, error)
                        continue
                    yield record, (filename, offset, len(line)), f"{filename}:{line_number}"
            continue

        with path.open("rb") as stream:
            byte_offset = 3 if stream.read(3) == b"\xef\xbb\xbf" else 0
        with path.open(encoding="utf-8-sig", newline="") as stream:
            text = stream.read()
        whitespace = re.compile(r"\s*")
        position = whitespace.match(text, 0).end()
        if text[position : position + 1] != "[":
            raise ValueError(f"QA JSON annotations must contain an array: {filename}")
        position = whitespace.match(text, position + 1).end()
        decoder = json.JSONDecoder()
        previous = 0
        record_index = 0
        if text[position : position + 1] != "]":
            while True:
                record, end = decoder.raw_decode(text, position)
                byte_offset += len(text[previous:position].encode("utf-8"))
                length = len(text[position:end].encode("utf-8"))
                yield record, (filename, byte_offset, length), f"{filename}[{record_index}]"
                byte_offset += length
                previous = end
                record_index += 1
                position = whitespace.match(text, end).end()
                if text[position : position + 1] == "]":
                    break
                if text[position : position + 1] != ",":
                    raise ValueError(f"Expected a comma between QA records in {filename} at character {position}")
                position = whitespace.match(text, position + 1).end()
        if text[position + 1 :].strip():
            raise ValueError(f"Unexpected content after the QA JSON array: {filename}")


def normalize_image_video_record(record, video_root):
    visuals = []
    image_paths = record.get("images") or []
    video_paths = record.get("videos") or []
    if not isinstance(image_paths, list) or not isinstance(video_paths, list):
        raise ValueError("QA images/videos must be lists of paths")
    if record.get("video"):
        video_paths = [record["video"], *video_paths]
    for kind, paths in (("image", image_paths), ("video", video_paths)):
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("QA media paths must be nonempty strings")
            path = Path(raw_path).expanduser()
            path = path if path.is_absolute() else Path(video_root) / path
            visuals.append({"type": kind, "resource": {"path": str(path)}, "metadata": {}})
    if not visuals:
        raise ValueError("Image/video QA annotations must reference at least one media file")

    messages = []
    cursor = 0
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    conversation = record.get("messages") or record.get("conversations")
    for message in conversation:
        role = message["role"] if "role" in message else roles[message["from"]]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported QA role: {role!r}")
        text = message["content"] if "content" in message else message["value"]
        if not isinstance(text, str):
            raise ValueError("QA message content must be text")
        if role != "user":
            if role == "assistant" and not text.strip():
                raise ValueError("QA assistant answers must not be empty")
            messages.append({"role": role, "content": text})
            continue
        content = []
        for part in re.split(r"(<image>|<video>)", text):
            if part in {"<image>", "<video>"}:
                if cursor >= len(visuals):
                    raise ValueError("More QA media placeholders than referenced files")
                visual = visuals[cursor]
                # ET-Instruct uses <image> for its single video input.
                if part == "<video>" and visual["type"] != "video":
                    raise ValueError("A video placeholder must reference a video")
                content.append(visual)
                cursor += 1
            elif part:
                content.append({"type": "text", "text": part})
        messages.append({"role": role, "content": content})
    targets = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    first_user = next((index for index, message in enumerate(messages) if message["role"] == "user"), None)
    if first_user is None or not targets or targets[0] < first_user:
        raise ValueError("QA annotations require a user message before their assistant targets")
    if cursor == 0:
        messages[first_user]["content"] = visuals + messages[first_user]["content"]
    elif cursor != len(visuals):
        raise ValueError("QA media placeholders must match the number of referenced files")
    return messages, targets


class ImageVideoQASFTDataset(VideoChat3SFTDataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if processor is None:
            raise ValueError("Image/video QA requires a Qwen3-VL processor")
        self.tokenizer, self.processor = tokenizer, processor
        self.options = dict(config.get("image_video_qa") or {})
        self.video_root = self.options["video_root"]
        self.media_type = self.options.get("media_type", "auto")
        self.media = VideoChat3Media.shared(self.video_root)
        self.fps = float(self.options.get("fps", 1.0))
        self.max_frames = int(self.options.get("video_max_frames", 128))
        sample_offline_timestamps(1.0, self.fps, self.max_frames)
        self.max_length = int(self.options.get("max_length", config.get("max_length", 8192)))
        self.pad_mode = config.get("pad_mode", "no_padding")
        if self.pad_mode not in {"right", "no_padding"}:
            raise ValueError("Image/video QA supports right or no_padding")
        self.system_prompt = self.options.get("system_prompt") or "You are a helpful assistant."
        self.chat_kwargs = dict(config.get("apply_chat_template_kwargs") or {})
        self.chat_kwargs.setdefault("enable_thinking", False)
        self.image_kwargs = dict(
            min_pixels=int(self.options.get("min_pixels", 4096)),
            max_pixels=int(self.options.get("max_pixels", 147456)),
        )
        self.frame_kwargs = dict(
            min_pixels=int(self.options.get("frame_min_pixels", 784)),
            max_pixels=int(self.options.get("frame_max_pixels", 200704)),
        )
        self.missing_media = self.options.get("missing_media", "skip")
        if self.missing_media not in {"skip", "error"}:
            raise ValueError("missing_media must be skip or error")
        self.references, self.cumulative_turns, self.sessions = [], [], OrderedDict()
        skipped = invalid_records = total = 0
        for record, reference, location in indexed_qa_records(parquet_files):
            try:
                messages, targets = normalize_image_video_record(record, self.video_root)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
                invalid_records += 1
                logger.warning("Skipping invalid image/video QA annotation at %s: %s", location, error)
                continue
            missing = self._missing_media(messages)
            if missing:
                if self.missing_media == "error":
                    raise FileNotFoundError(f"Missing QA media at {location}: {missing}")
                skipped += 1
                continue
            count = len(targets)
            if max_samples > 0:
                count = min(count, max_samples - total)
            total += count
            self.references.append((reference, count, location))
            self.cumulative_turns.append(total)
            if max_samples > 0 and total >= max_samples:
                break
        logger.warning(
            "Image/video QA: %d answer samples, %d records, %d missing-media records skipped, "
            "%d invalid records skipped", total, len(self.references), skipped, invalid_records,
        )
        if not total:
            raise EmptyDatasetError("No usable image/video QA samples. Check annotations and video_root.")

    def _missing_media(self, messages):
        for message in messages:
            if message["role"] != "user":
                continue
            for part in message["content"]:
                if part["type"] == "text":
                    continue
                if self.media_type not in {"auto", "mixed", part["type"]}:
                    raise ValueError(f"Declared media_type={self.media_type!r}, got {part['type']!r}")
                path = part["resource"]["path"]
                if not Path(path).is_file():
                    return path
        return None

    def _session(self, index):
        if index not in self.sessions:
            (filename, offset, length), count, location = self.references[index]
            with open(filename, "rb") as stream:
                stream.seek(offset)
                record = json.loads(stream.read(length).decode("utf-8-sig"))
            messages, targets = normalize_image_video_record(record, self.video_root)
            missing = self._missing_media(messages)
            if missing:
                raise FileNotFoundError(f"QA media disappeared after indexing at {location}: {missing}")
            self.sessions[index] = (messages, targets[:count], location)
        self.sessions.move_to_end(index)
        session = self.sessions[index]
        while len(self.sessions) > max(int(self.options.get("cache_sessions", 8)), 1):
            self.sessions.popitem(last=False)
        return session
