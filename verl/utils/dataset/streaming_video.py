# Copyright 2026 The VideoLLM-Stream authors
# Licensed under the Apache License, Version 2.0.

"""Full-session, per-frame streaming supervision for stream-meta-v1 annotations."""

import bisect
import glob
import json
import logging
import math
import random
import re
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_TOKENS = ("</Silence>", "</Response>")
STATE_NAMES = dict(zip(("silence", "response"), STATE_TOKENS, strict=True))
DEFAULT_VIDEO_ROOT = "/workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData"
DEFAULT_SYSTEM_PROMPT = """You are a streaming video assistant. Each round contains one video frame with its absolute
timestamp in seconds. Use only the current frame and previous observations.
User instructions become available only when they arrive. After each frame output exactly one state:
</Silence>: no relevant event or no answer or update is currently needed.
</Response>: sufficient evidence is available; immediately follow this token with your answer.
Silence must not contain an answer. Never describe unseen future frames as observations."""


class EmptyDatasetError(ValueError):
    pass


class MaxLengthError(ValueError):
    pass


def parse_time(value):
    if isinstance(value, str) and ":" in value:
        seconds = 0.0
        for component in value.split(":"):
            seconds = seconds * 60 + float(component)
    else:
        seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Expected finite nonnegative seconds, got {value!r}")
    return seconds


def sample_rounds(start, end, fps=4.0):
    """Sample one frame per round at start + 1/fps, ..., end, retaining a partial final interval."""
    start, end = parse_time(start), parse_time(end)
    if not math.isfinite(fps) or fps <= 0 or end < start:
        raise ValueError(f"Invalid sampling range/fps: {start}, {end}, {fps}")
    rounds = []
    count = max(1, math.ceil((end - start) * fps - 1e-9))
    for index in range(count):
        left = start + index / fps
        right = end if index == count - 1 else start + (index + 1) / fps
        rounds.append(dict(start=left, timestamp=right, frame_timestamps=[right]))
    return rounds


def annotation_files(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = set()
    for value in paths:
        path = Path(value).expanduser()
        if path.is_dir():
            files.update(path.rglob("*.jsonl"))
            files.update(path.rglob("*.json"))
        elif path.is_file():
            files.add(path)
        else:
            files.update(Path(match) for match in glob.glob(str(path), recursive=True))
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL annotations found in {paths}")
    return sorted(files)


def load_annotations(paths):
    """Yield valid JSON objects with their original offsets/indices; log and skip malformed records."""
    for path in annotation_files(paths):
        skipped = 0
        if path.suffix.lower() == ".jsonl":
            with path.open("rb") as stream:
                line_number = 0
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    line_number += 1
                    if line.strip():
                        try:
                            record = json.loads(line.decode("utf-8-sig"))
                            if not isinstance(record, dict):
                                raise ValueError("Annotation record must be a JSON object")
                        except ValueError as error:
                            skipped += 1
                            logger.warning("Skipping invalid annotation at %s:%d: %s", path, line_number, error)
                            continue
                        yield record, (str(path), offset), line_number
        else:
            try:
                with path.open(encoding="utf-8-sig") as stream:
                    records = json.load(stream)
                if not isinstance(records, (dict, list)):
                    raise ValueError("JSON annotation file must contain an object or an array of objects")
            except ValueError as error:
                logger.warning("Skipping invalid JSON annotation file %s: %s", path, error)
                continue
            if isinstance(records, dict):
                records = [records]
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    skipped += 1
                    logger.warning("Skipping invalid annotation at %s:%d: expected a JSON object", path, index + 1)
                    continue
                yield record, (str(path), index), index + 1
        if skipped:
            logger.warning("Annotation reader: %d malformed records skipped in %s", skipped, path)


def load_path_mapping(value):
    if isinstance(value, (str, Path)) and value:
        with Path(value).open(encoding="utf-8-sig") as stream:
            value = json.load(stream)
    return dict(value or {})


def resolve_video_path(record, video_root=DEFAULT_VIDEO_ROOT, path_map=None, path_replacements=None):
    """Resolve only explicit paths or explicitly supplied mappings; never guess a video filename."""
    raw = str(record.get("video_path") or "").strip()
    mapping = path_map or {}
    scoped_name = "/".join(str(record.get(key) or "") for key in ("source", "video_source", "video_name"))
    for key in (scoped_name, raw, str(record.get("video_name") or "")):
        if key and key in mapping:
            raw = str(mapping[key] or "").strip()
            break
    if not raw:
        return None
    for old, new in sorted((path_replacements or {}).items(), key=lambda item: -len(item[0])):
        if raw == old or raw.startswith(old.rstrip("/\\") + "/") or raw.startswith(old.rstrip("/\\") + "\\"):
            raw = str(new).rstrip("/\\") + raw[len(old.rstrip("/\\")) :]
            break
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(video_root).expanduser() / path
    return str(path)


def build_stream_turns(record, fps=4.0):
    if record.get("schema") != "stream-meta-v1":
        raise ValueError(f"Expected stream-meta-v1, got {record.get('schema')!r}")
    ranges = record.get("video_ranges") or []
    if not ranges:
        raise ValueError("video_ranges must explicitly specify the observed video intervals")
    intervals = []
    previous_end = -1.0
    for start, end in sorted((parse_time(start), parse_time(end)) for start, end in ranges):
        if start < previous_end - 1e-8:
            raise ValueError("video_ranges must not overlap")
        if intervals and abs(start - previous_end) < 1e-8:
            intervals[-1][1] = end
        else:
            intervals.append([start, end])
        previous_end = end
    turns = [turn for start, end in intervals for turn in sample_rounds(start, end, fps)]
    timestamps = [turn["timestamp"] for turn in turns]
    for turn in turns:
        turn.update(prompts=[], state="silence", response=STATE_TOKENS[0])
    for prompt in record.get("prompts") or []:
        at = parse_time(prompt["at"])
        index = bisect.bisect_left(timestamps, at)
        if index < len(turns) and prompt.get("text"):
            turns[index]["prompts"].append(str(prompt["text"]))
    response_texts = {}
    for event in record.get("timeline") or []:
        state = event["type"].lower()
        if state not in STATE_NAMES:
            raise ValueError(f"Unsupported timeline state {state!r}")
        if state == "response":
            at = parse_time(event["at"])
            index = bisect.bisect_left(timestamps, at)
            if index == len(turns):
                raise ValueError(f"Response at {at}s exceeds video_ranges endpoint {timestamps[-1]}s")
            text = str(event.get("text") or "").strip()
            if not text:
                raise ValueError(f"Response at {at}s has no answer text")
            response_texts.setdefault(index, []).append(text)
        elif "interval" in event:
            start, end = map(parse_time, event["interval"])
            if end < start:
                raise ValueError(f"Invalid timeline interval: {event['interval']}")
            indices = range(
                bisect.bisect_left(timestamps, start), bisect.bisect_right(timestamps, end)
            )
            for index in indices:
                turns[index]["state"] = state
                turns[index]["response"] = STATE_NAMES[state]
        else:
            index = bisect.bisect_left(timestamps, parse_time(event["at"]))
            if index < len(turns):
                turns[index]["state"] = state
                turns[index]["response"] = STATE_NAMES[state]
    for index, turn in enumerate(turns):
        if index in response_texts:
            turn["state"] = "response"
            turn["response"] = STATE_TOKENS[1] + " " + "\n".join(response_texts[index])
    return turns


def build_user_message(turn, images, question=None):
    text = f"<{turn['timestamp']:g}s>"
    if question:
        text += "\n" + question
    content = [{"type": "text", "text": text}]
    for _, image in zip(turn["frame_timestamps"], images, strict=True):
        content.append({"type": "image", "image": image})
    return {"role": "user", "content": content}


class VideoFrameReader:
    """Decode requested observations only; keep one worker-local decoder and a bounded image cache."""

    def __init__(self, cache_frames=64, sequence_filename_pattern=None, end_tolerance=0.0, seek_mode="exact"):
        self.cache_frames = int(cache_frames)
        self.sequence_filename_pattern = sequence_filename_pattern
        self.end_tolerance = float(end_tolerance)
        self.seek_mode = seek_mode
        if not math.isfinite(self.end_tolerance) or self.end_tolerance < 0:
            raise ValueError("end_tolerance must be finite and nonnegative")
        self.path = None
        self.decoder = None
        self.frames = None
        self.cache = OrderedDict()

    def _open(self, path, media_type="video", sequence=None):
        sequence = sequence or {}
        pattern = sequence.get("filename_pattern", self.sequence_filename_pattern)
        source_key = (path, media_type, pattern)
        if source_key != self.path:
            self.path, self.decoder, self.frames = source_key, None, None
            self.cache.clear()
            if media_type == "video":
                from torchcodec.decoders import VideoDecoder

                self.decoder = VideoDecoder(path, dimension_order="NHWC", device="cpu", seek_mode=self.seek_mode)
            elif media_type == "sequence":
                self.frames = sorted(
                    (p for p in Path(path).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}),
                    key=lambda p: [
                        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", p.name)
                    ],
                )
                if not self.frames:
                    raise FileNotFoundError(f"No sequence images in {path}")
            else:
                raise ValueError(f"Unsupported media_type: {media_type}")

    def _sequence_frame_path(self, path, timestamp, sequence):
        source_fps = float(sequence.get("fps") or 0)
        if source_fps <= 0:
            raise ValueError(f"sequence.fps must be specified for {path}")
        duration = len(self.frames) / source_fps
        if timestamp > duration + self.end_tolerance + 1e-6:
            raise ValueError(f"Timestamp {timestamp}s exceeds image sequence duration: {path}")
        frame_index = min(math.floor(timestamp * source_fps), len(self.frames) - 1)
        pattern = sequence.get("filename_pattern", self.sequence_filename_pattern)
        if not pattern:
            return self.frames[frame_index]
        frame_time = frame_index / source_fps
        return Path(path) / pattern.format(
            index=frame_index + int(sequence.get("start_index", 0)),
            frame_idx=frame_index + int(sequence.get("start_index", 0)),
            seconds=int(frame_time),
            centiseconds=int(round((frame_time % 1) * 100)),
        )

    def frame_sizes(self, path, timestamps, media_type="video", sequence=None):
        from PIL import Image

        sequence = sequence or {}
        self._open(path, media_type, sequence)
        if media_type == "video":
            width, height = self.decoder.metadata.width, self.decoder.metadata.height
            if not width or not height:
                raise ValueError(f"Video metadata must contain frame dimensions: {path}")
            return [(width, height)] * len(timestamps)
        sizes, sizes_by_path = [], {}
        for timestamp in timestamps:
            frame_path = self._sequence_frame_path(path, timestamp, sequence)
            if frame_path not in sizes_by_path:
                with Image.open(frame_path) as image:
                    sizes_by_path[frame_path] = image.size
            sizes.append(sizes_by_path[frame_path])
        return sizes

    def read(self, path, timestamps, media_type="video", sequence=None):
        from PIL import Image

        sequence = sequence or {}
        self._open(path, media_type, sequence)
        images = []
        for timestamp in timestamps:
            key = (float(timestamp), json.dumps(sequence, sort_keys=True))
            if key not in self.cache:
                if media_type == "video":
                    begin = self.decoder.metadata.begin_stream_seconds
                    end = self.decoder.metadata.end_stream_seconds
                    requested = begin + timestamp
                    if requested > end + self.end_tolerance + 1e-6:
                        raise ValueError(
                            f"Annotation timestamp {timestamp}s exceeds media duration {end - begin}s: {path}"
                        )
                    frame = self.decoder.get_frame_played_at(min(requested, math.nextafter(end, -math.inf)))
                    if frame.pts_seconds > requested:
                        raise ValueError(f"Decoder returned a future frame for {timestamp}s: {path}")
                    image = Image.fromarray(frame.data.numpy()).convert("RGB")
                else:
                    frame_path = self._sequence_frame_path(path, timestamp, sequence)
                    with Image.open(frame_path) as source_image:
                        image = source_image.convert("RGB")
                self.cache[key] = image
            self.cache.move_to_end(key)
            images.append(self.cache[key])
            while len(self.cache) > max(self.cache_frames, 1):
                self.cache.popitem(last=False)
        return images


def iter_video_rounds(path, fps=4.0, start=0.0, end=None):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    reader = VideoFrameReader(cache_frames=1, end_tolerance=1.0 / fps)
    reader._open(str(path))
    duration = reader.decoder.metadata.end_stream_seconds - reader.decoder.metadata.begin_stream_seconds
    end = duration if end is None else min(float(end), duration)
    for turn in sample_rounds(start, end, fps):
        yield turn, reader.read(str(path), turn["frame_timestamps"])


class StreamingVideoSFTDataset:
    """One complete conversation per annotation, supervising every assistant response."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if processor is None:
            raise ValueError("StreamingVideoSFTDataset requires a Qwen3-VL processor")
        self.tokenizer, self.processor = tokenizer, processor
        self.state_ids = {}
        for state, token in STATE_NAMES.items():
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Register {token} as a single special token before building the streaming dataset")
            self.state_ids[state] = token_ids[0]
        message_tokens = [
            self.tokenizer.encode(token, add_special_tokens=False) for token in ("<|im_start|>", "<|im_end|>")
        ]
        if any(len(ids) != 1 for ids in message_tokens):
            raise ValueError("Streaming SFT requires Qwen3-VL message boundary tokens")
        self.message_start_id, self.message_end_id = (ids[0] for ids in message_tokens)
        self.assistant_prefix_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.config = config or {}
        self.options = self.config.get("streaming_video", {})
        self.fps = float(self.options.get("fps", 4.0))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.max_length = int(self.config.get("max_length", 8192))
        self.truncation = self.options.get("truncation", "filter")
        if self.truncation not in ("filter", "error"):
            raise ValueError("Streaming truncation must be filter or error; conversations are never truncated")
        self.max_refetch = int(self.options.get("max_refetch", 100))
        if self.max_refetch < 1:
            raise ValueError("streaming_video.max_refetch must be positive")
        self.pad_mode = self.config.get("pad_mode", "no_padding")
        if self.pad_mode not in ("right", "no_padding"):
            raise ValueError("Streaming video supports right or no_padding")
        self.system_prompt = self.options.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        self.chat_kwargs = dict(self.config.get("apply_chat_template_kwargs", {}))
        self.chat_kwargs.setdefault("enable_thinking", False)
        self.image_kwargs = dict(
            min_pixels=int(self.options.get("frame_min_pixels", self.options.get("min_pixels", 28 * 28))),
            max_pixels=int(self.options.get("frame_max_pixels", self.options.get("max_pixels", 448 * 448))),
        )
        self.path_map = load_path_mapping(self.options.get("path_map"))
        self.path_replacements = dict(self.options.get("path_replacements") or {})
        self.missing_video = self.options.get("missing_video", "skip")
        if self.missing_video not in ("skip", "error"):
            raise ValueError("missing_video must be skip or error")
        self.references = []
        self.sessions = OrderedDict()
        self.reader = None
        self.skipped = 0
        self.invalid_records = 0
        self.overlong_indices = set()
        for record, reference, line_number in load_annotations(parquet_files):
            try:
                path = resolve_video_path(
                    record, self.options.get("video_root", DEFAULT_VIDEO_ROOT), self.path_map, self.path_replacements
                )
                media_type = record.get("media_type", "video")
                if media_type not in ("video", "sequence"):
                    raise ValueError(f"Unsupported streaming media_type: {media_type!r}")
                available = path and (Path(path).is_dir() if media_type == "sequence" else Path(path).is_file())
                if available:
                    if record.get("schema") != "stream-meta-v1":
                        raise ValueError(f"Expected stream-meta-v1, got {record.get('schema')!r}")
                    if not record.get("video_ranges"):
                        raise ValueError("video_ranges must explicitly specify the observed video intervals")
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
                self.invalid_records += 1
                logger.warning("Skipping invalid streaming annotation at %s:%d: %s", reference[0], line_number, error)
                continue
            if not available:
                self.skipped += 1
                if self.missing_video == "error":
                    raise FileNotFoundError(f"Missing video_path/media at {reference[0]}:{line_number}: {path!r}")
                continue
            location = f"{reference[0]}:{line_number}"
            self.references.append((reference, path, location))
            if max_samples > 0 and len(self.references) >= max_samples:
                break
        logger.warning(
            "Streaming dataset: %d candidate full-session samples, %d missing media skipped, "
            "%d invalid headers skipped; tokenization and max_length=%d filtering run lazily in DataLoader",
            len(self.references), self.skipped, self.invalid_records, self.max_length,
        )
        if not self.references:
            raise EmptyDatasetError("No candidate streaming conversations. Check annotations and media paths.")

    def __len__(self):
        return len(self.references)

    def _session(self, session_index):
        if session_index not in self.sessions:
            (filename, position), path, _ = self.references[session_index]
            if filename.lower().endswith(".jsonl"):
                with open(filename, "rb") as stream:
                    stream.seek(position)
                    record = json.loads(stream.readline().decode("utf-8-sig"))
            else:
                with open(filename, encoding="utf-8-sig") as stream:
                    records = json.load(stream)
                record = records[position] if isinstance(records, list) else records
            turns = build_stream_turns(record, self.fps)
            self.sessions[session_index] = (record, path, turns)
        self.sessions.move_to_end(session_index)
        session = self.sessions[session_index]
        while len(self.sessions) > max(int(self.options.get("cache_sessions", 8)), 1):
            self.sessions.popitem(last=False)
        return session

    def _encode_conversation(self, turns, image_lengths, location):
        import torch

        from .video_sft_processing import encode_image_text

        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in turns:
            messages.append(build_user_message(turn, [None], "\n".join(turn["prompts"])))
            messages.append({"role": "assistant", "content": turn["response"]})
        text = self.tokenizer.apply_chat_template(
            messages, chat_template=self.processor.chat_template,
            tokenize=False, add_generation_prompt=False, **self.chat_kwargs,
        )
        inputs = encode_image_text(self.processor, text, image_lengths)
        input_ids = inputs["input_ids"][0]
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = inputs["attention_mask"][0]
        # Locate Qwen message boundaries in one encoding, without repeatedly tokenizing growing prefixes.
        starts = (input_ids == self.message_start_id).nonzero(as_tuple=True)[0].tolist()
        ends = (input_ids == self.message_end_id).nonzero(as_tuple=True)[0].tolist()
        if len(starts) != len(messages) or len(ends) != len(messages):
            raise ValueError(f"Expected one Qwen message boundary pair per message at {location}")
        labels = torch.full_like(input_ids, -100)
        stream_state_mask = torch.zeros_like(input_ids)
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if start >= end or (index + 1 < len(starts) and end >= starts[index + 1]):
                raise ValueError(f"Invalid Qwen message boundaries at {location}")
            if messages[index]["role"] != "assistant":
                continue
            body_start = start + len(self.assistant_prefix_ids)
            if input_ids[start:body_start].tolist() != self.assistant_prefix_ids:
                raise ValueError(f"Unsupported assistant template prefix at {location}")
            turn = turns[index // 2 - 1]
            if body_start >= end or input_ids[body_start].item() != self.state_ids[turn["state"]]:
                raise ValueError(f"Each assistant response must start with its annotated state token at {location}")
            labels[body_start : end + 1] = input_ids[body_start : end + 1]
            stream_state_mask[body_start] = 1
        length = max(len(input_ids), len(labels))
        if length > self.max_length:
            raise MaxLengthError(
                f"Complete streaming conversation at {location} has {length} tokens, "
                f"exceeding max_length={self.max_length}"
            )
        inputs.update(labels=labels, loss_mask=(labels != -100).long(), stream_state_mask=stream_state_mask)
        return inputs

    def __getitem__(self, item):
        if item < 0 or item >= len(self):
            raise IndexError(item)
        n_try = min(self.max_refetch, len(self)) if self.truncation == "filter" else 1
        # Deterministic replacements need no extra RNG state in StatefulDataLoader checkpoints.
        offsets = random.Random(item).sample(range(1, len(self)), n_try - 1)
        for index in [item] + [(item + offset) % len(self) for offset in offsets]:
            if index in self.overlong_indices:
                continue
            try:
                return self._encode_item(index)
            except MaxLengthError as error:
                if self.truncation == "error":
                    raise
                self.overlong_indices.add(index)
                count = len(self.overlong_indices)
                if count == 1 or count % 1000 == 0:
                    logger.warning(
                        "Streaming lazy filter: %d overlong conversations skipped in this worker; %s", count, error
                    )
        raise ValueError(
            f"No valid streaming conversation in {n_try} candidate slots at {self.references[item][2]}; "
            "increase max_length, reduce FPS/image pixels, or increase streaming_video.max_refetch"
        )

    def _encode_item(self, item):
        import torch
        import torch.nn.functional as F

        from .video_sft_processing import image_token_lengths_from_sizes, process_sft_images

        record, path, turns = self._session(item)
        location = self.references[item][2]
        if self.reader is None:
            tolerance = self.options.get("end_tolerance")
            self.reader = VideoFrameReader(
                self.options.get("cache_frames", 64), self.options.get("sequence_filename_pattern"),
                1.0 / self.fps if tolerance is None else tolerance,
            )
        timestamps = [turn["timestamp"] for turn in turns]
        media_type, sequence = record.get("media_type", "video"), record.get("sequence")
        sizes = self.reader.frame_sizes(path, timestamps, media_type, sequence)
        image_lengths = image_token_lengths_from_sizes(self.processor, sizes, self.image_kwargs)
        inputs = self._encode_conversation(turns, image_lengths, location)
        images = self.reader.read(path, timestamps, media_type, sequence)
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        labels = inputs.pop("labels")
        loss_mask = inputs.pop("loss_mask")
        stream_state_mask = inputs.pop("stream_state_mask")
        inputs.update(process_sft_images(self.processor, images, self.image_kwargs, image_lengths))
        inputs.pop("mm_token_type_ids", None)
        from verl.models.transformers.qwen3_vl import get_rope_index

        vision_positions = get_rope_index(
            self.processor, input_ids=input_ids, attention_mask=attention_mask,
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        position_ids = torch.cat((torch.arange(len(input_ids)).unsqueeze(0), vision_positions), dim=0)
        result = dict(
            input_ids=input_ids, position_ids=position_ids, labels=labels, loss_mask=loss_mask,
            stream_state_mask=stream_state_mask, multi_modal_inputs=inputs,
        )
        if self.pad_mode == "right":
            padding = self.max_length - len(input_ids)
            result["input_ids"] = F.pad(input_ids, (0, padding), value=self.tokenizer.pad_token_id or 0)
            result["position_ids"] = F.pad(position_ids, (0, padding), value=0)
            result["labels"] = F.pad(labels, (0, padding), value=-100)
            result["loss_mask"] = F.pad(loss_mask, (0, padding), value=0)
            result["stream_state_mask"] = F.pad(stream_state_mask, (0, padding), value=0)
            result["attention_mask"] = F.pad(attention_mask, (0, padding), value=0)
        return result
