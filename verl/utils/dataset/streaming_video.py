# Copyright 2026 The VideoLLM-Stream authors
# Licensed under the Apache License, Version 2.0.

"""Causal, multi-frame round supervision for stream-meta-v1 annotations."""

import bisect
import glob
import json
import logging
import math
import re
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_TOKENS = ("</Silence>", "</Response>")
STATE_NAMES = dict(zip(("silence", "response"), STATE_TOKENS, strict=True))
DEFAULT_VIDEO_ROOT = "/workspace/s/lzw/datasets/Stream-VideoLLM/TrainingData"
DEFAULT_SYSTEM_PROMPT = """You are a streaming video assistant. Each round contains chronological frames from a
completed time interval, with absolute timestamps in seconds. Use only this round and previous observations.
User instructions become available only when they arrive. After the frames of each round output exactly one state:
</Silence>: no relevant event or no answer or update is currently needed.
</Response>: sufficient evidence is available; immediately follow this token with your answer.
Silence must not contain an answer. Never describe unseen future frames as observations."""


class EmptyDatasetError(ValueError):
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


def sample_rounds(start, end, fps=4.0, round_seconds=1.0):
    """Sample right-closed rounds: (0, 1] contains frames at .25, .5, .75 and 1 seconds."""
    start, end = parse_time(start), parse_time(end)
    if not math.isfinite(fps) or fps <= 0 or end < start or not math.isfinite(round_seconds) or round_seconds <= 0:
        raise ValueError(f"Invalid sampling range/fps/round_seconds: {start}, {end}, {fps}, {round_seconds}")
    frames_per_round = fps * round_seconds
    if frames_per_round < 1 or not math.isclose(frames_per_round, round(frames_per_round), abs_tol=1e-8):
        raise ValueError("fps * round_seconds must be a positive integer number of frames")
    rounds = []
    count = max(1, math.ceil((end - start) / round_seconds - 1e-9))
    for index in range(count):
        left = start + index * round_seconds
        right = min(start + (index + 1) * round_seconds, end)
        frame_count = math.floor((right - left) * fps + 1e-9)
        timestamps = [min(left + (i + 1) / fps, right) for i in range(frame_count)]
        if not timestamps or right - timestamps[-1] > 1e-8:
            timestamps.append(right)
        else:
            timestamps[-1] = right
        rounds.append(dict(start=left, timestamp=right, frame_timestamps=timestamps))
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


def build_stream_turns(record, fps=4.0, round_seconds=1.0):
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
    turns = [turn for start, end in intervals for turn in sample_rounds(start, end, fps, round_seconds)]
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
    text = f"<{turn['start']:g}s-{turn['timestamp']:g}s>"
    if question:
        text += "\n" + question
    content = [{"type": "text", "text": text}]
    for timestamp, image in zip(turn["frame_timestamps"], images, strict=True):
        content.append({"type": "text", "text": f"<{timestamp:g}s>"})
        content.append({"type": "image", "image": image})
    return {"role": "user", "content": content}


class VideoFrameReader:
    """Decode requested observations only; keep one worker-local decoder and a bounded image cache."""

    def __init__(self, cache_frames=64, sequence_filename_pattern=None, end_tolerance=0.0):
        self.cache_frames = int(cache_frames)
        self.sequence_filename_pattern = sequence_filename_pattern
        self.end_tolerance = float(end_tolerance)
        if not math.isfinite(self.end_tolerance) or self.end_tolerance < 0:
            raise ValueError("end_tolerance must be finite and nonnegative")
        self.path = None
        self.decoder = None
        self.frames = None
        self.cache = OrderedDict()

    def read(self, path, timestamps, media_type="video", sequence=None):
        from PIL import Image

        sequence = sequence or {}
        pattern = sequence.get("filename_pattern", self.sequence_filename_pattern)
        source_key = (path, media_type, pattern)
        if source_key != self.path:
            self.path, self.decoder, self.frames = source_key, None, None
            self.cache.clear()
            if media_type == "video":
                from torchcodec.decoders import VideoDecoder

                self.decoder = VideoDecoder(path, dimension_order="NHWC", device="cpu", seek_mode="exact")
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
                    source_fps = float(sequence.get("fps") or 0)
                    if source_fps <= 0:
                        raise ValueError(f"sequence.fps must be specified for {path}")
                    duration = len(self.frames) / source_fps
                    if timestamp > duration + self.end_tolerance + 1e-6:
                        raise ValueError(f"Timestamp {timestamp}s exceeds image sequence duration: {path}")
                    frame_index = math.floor(timestamp * source_fps)
                    frame_index = min(frame_index, len(self.frames) - 1)
                    if pattern:
                        frame_time = frame_index / source_fps
                        frame_path = Path(path) / pattern.format(
                            index=frame_index + int(sequence.get("start_index", 0)),
                            frame_idx=frame_index + int(sequence.get("start_index", 0)),
                            seconds=int(frame_time),
                            centiseconds=int(round((frame_time % 1) * 100)),
                        )
                    else:
                        frame_path = self.frames[frame_index]
                    with Image.open(frame_path) as source_image:
                        image = source_image.convert("RGB")
                self.cache[key] = image
            self.cache.move_to_end(key)
            images.append(self.cache[key])
            while len(self.cache) > max(self.cache_frames, 1):
                self.cache.popitem(last=False)
        return images


def iter_video_rounds(path, fps=4.0, start=0.0, end=None, round_seconds=1.0):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    reader = VideoFrameReader(cache_frames=max(1, math.ceil(fps * round_seconds)), end_tolerance=1.0 / fps)
    from torchcodec.decoders import VideoDecoder

    reader.path = (str(path), "video", None)
    reader.decoder = VideoDecoder(str(path), dimension_order="NHWC", device="cpu", seek_mode="exact")
    duration = reader.decoder.metadata.end_stream_seconds - reader.decoder.metadata.begin_stream_seconds
    end = duration if end is None else min(float(end), duration)
    for turn in sample_rounds(start, end, fps, round_seconds):
        yield turn, reader.read(str(path), turn["frame_timestamps"])


class StreamingVideoSFTDataset:
    """One sample per completed multi-frame round; supervise only its assistant response."""

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
        self.config = config or {}
        self.options = self.config.get("streaming_video", {})
        self.fps = float(self.options.get("fps", 4.0))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.round_seconds = float(self.options.get("round_seconds", 1.0))
        sample_rounds(0.0, self.round_seconds, self.fps, self.round_seconds)
        self.video_max_frames = int(self.options.get("video_max_frames", self.options.get("max_frames", 256)))
        if self.video_max_frames < round(self.fps * self.round_seconds):
            raise ValueError("video_max_frames must accommodate all frames in the current round")
        self.window_rounds = int(self.options.get("window_rounds", 64))
        if self.window_rounds <= 0:
            raise ValueError("window_rounds must be positive")
        self.max_length = int(self.config.get("max_length", 8192))
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
        self.references, self.cumulative_turns = [], []
        self.sessions = OrderedDict()
        self.reader = None
        self.skipped = 0
        self.invalid_records = 0
        total = 0
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
                    turns = build_stream_turns(record, self.fps, self.round_seconds)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
                self.invalid_records += 1
                logger.warning("Skipping invalid streaming annotation at %s:%d: %s", reference[0], line_number, error)
                continue
            if not available:
                self.skipped += 1
                if self.missing_video == "error":
                    raise FileNotFoundError(f"Missing video_path/media at {reference[0]}:{line_number}: {path!r}")
                continue
            count = len(turns)
            if max_samples > 0:
                count = min(count, max_samples - total)
            total += count
            self.references.append((reference, path))
            self.cumulative_turns.append(total)
            if max_samples > 0 and total >= max_samples:
                break
        logger.warning(
            "Streaming dataset: %d round samples, %d sessions, %d missing media skipped, %d invalid records skipped",
            total, len(self.references), self.skipped, self.invalid_records,
        )
        if total == 0:
            raise EmptyDatasetError("No usable streaming samples. Check annotations and video_root/path_map.")

    def __len__(self):
        return self.cumulative_turns[-1]

    def _session(self, session_index):
        if session_index not in self.sessions:
            (filename, position), path = self.references[session_index]
            if filename.lower().endswith(".jsonl"):
                with open(filename, "rb") as stream:
                    stream.seek(position)
                    record = json.loads(stream.readline().decode("utf-8-sig"))
            else:
                with open(filename, encoding="utf-8-sig") as stream:
                    records = json.load(stream)
                record = records[position] if isinstance(records, list) else records
            turns = build_stream_turns(record, self.fps, self.round_seconds)
            self.sessions[session_index] = (record, path, turns)
        self.sessions.move_to_end(session_index)
        session = self.sessions[session_index]
        while len(self.sessions) > max(int(self.options.get("cache_sessions", 8)), 1):
            self.sessions.popitem(last=False)
        return session

    def __getitem__(self, item):
        import torch
        import torch.nn.functional as F

        if item < 0 or item >= len(self):
            raise IndexError(item)
        session_index = bisect.bisect_right(self.cumulative_turns, item)
        turn_index = item - (self.cumulative_turns[session_index - 1] if session_index else 0)
        record, path, turns = self._session(session_index)
        if self.reader is None:
            tolerance = self.options.get("end_tolerance")
            self.reader = VideoFrameReader(
                self.options.get("cache_frames", 64), self.options.get("sequence_filename_pattern"),
                1.0 / self.fps if tolerance is None else tolerance,
            )
        start = max(0, turn_index - self.window_rounds + 1)
        frame_count = sum(len(turn["frame_timestamps"]) for turn in turns[start : turn_index + 1])
        while frame_count > self.video_max_frames:
            frame_count -= len(turns[start]["frame_timestamps"])
            start += 1
        images = self.reader.read(
            path, [timestamp for turn in turns[start : turn_index + 1] for timestamp in turn["frame_timestamps"]],
            record.get("media_type", "video"), record.get("sequence"),
        )
        while True:
            messages = [{"role": "system", "content": self.system_prompt}]
            earlier = [
                f"Earlier instruction received at <{turn['timestamp']:g}s>:\n{question}"
                for turn in turns[:start] for question in turn["prompts"]
            ]
            image_index = 0
            for offset, turn in enumerate(turns[start : turn_index + 1]):
                questions = (earlier if offset == 0 else []) + turn["prompts"]
                frame_count = len(turn["frame_timestamps"])
                messages.append(build_user_message(
                    turn, images[image_index : image_index + frame_count], "\n".join(questions)
                ))
                image_index += frame_count
                messages.append({"role": "assistant", "content": turn["response"]})
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **self.chat_kwargs
            )
            prefix = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True, **self.chat_kwargs
            )
            inputs = dict(self.processor(
                text=[text], images=images, return_tensors="pt", add_special_tokens=False,
                return_mm_token_type_ids=False, **self.image_kwargs
            ))
            input_ids = inputs.pop("input_ids")[0]
            attention_mask = inputs.pop("attention_mask")[0]
            # Prefix image tokens must be expanded exactly as they are in the complete training example.
            prefix_ids = self.processor(
                text=[prefix], images=images, return_tensors="pt", add_special_tokens=False,
                return_mm_token_type_ids=False, **self.image_kwargs
            )["input_ids"][0]
            if not torch.equal(input_ids[: len(prefix_ids)], prefix_ids):
                raise ValueError(
                    "Chat template does not preserve the generation prefix; cannot construct a safe label mask"
                )
            if len(input_ids) <= self.max_length:
                break
            if start == turn_index:
                raise ValueError(
                    f"Current round, arrived instructions and complete target require {len(input_ids)} tokens, "
                    f"exceeding max_length={self.max_length}; increase it or reduce image max_pixels"
                )
            images = images[len(turns[start]["frame_timestamps"]) :]
            start += 1
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[len(prefix_ids) :] = 1
        if not bool(loss_mask.any()):
            raise ValueError("The assistant target must contain at least one token")
        state_id = self.state_ids[turns[turn_index]["state"]]
        if input_ids[len(prefix_ids)].item() != state_id:
            raise ValueError("Assistant supervision must start with the annotated streaming state token")
        stream_state_mask = torch.zeros_like(input_ids)
        stream_state_mask[len(prefix_ids)] = 1
        inputs.pop("mm_token_type_ids", None)
        from verl.models.transformers.qwen3_vl import get_rope_index

        vision_positions = get_rope_index(
            self.processor, input_ids=input_ids, attention_mask=attention_mask,
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        position_ids = torch.cat((torch.arange(len(input_ids)).unsqueeze(0), vision_positions), dim=0)
        result = dict(
            input_ids=input_ids, position_ids=position_ids, loss_mask=loss_mask,
            stream_state_mask=stream_state_mask, multi_modal_inputs=inputs,
        )
        if self.pad_mode == "right":
            padding = self.max_length - len(input_ids)
            result["input_ids"] = F.pad(input_ids, (0, padding), value=self.tokenizer.pad_token_id or 0)
            result["position_ids"] = F.pad(position_ids, (0, padding), value=0)
            result["loss_mask"] = F.pad(loss_mask, (0, padding), value=0)
            result["stream_state_mask"] = F.pad(stream_state_mask, (0, padding), value=0)
            result["attention_mask"] = F.pad(attention_mask, (0, padding), value=0)
        return result
