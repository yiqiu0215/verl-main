# Copyright 2026 The VideoLLM-Stream authors
# Licensed under the Apache License, Version 2.0.

"""Build map-style mixed SFT data from a single, explicit dataset manifest."""

import bisect
import logging
from pathlib import Path

from omegaconf import OmegaConf

from .streaming_video import EmptyDatasetError, StreamingVideoSFTDataset, annotation_files

logger = logging.getLogger(__name__)


def _resolve_path(value, directory):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else directory / path)


def _sample_limit(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value == 0 or value < -1:
        raise ValueError(f"{name} must be -1 or a positive integer; use enabled: false to disable a source")
    return value


class MixedVideoSFTDataset:
    """Concatenate streaming rounds and ordinary offline QA, preserving their own label masks."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        data_config = (
            OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config or {})
        )
        manifest_path = parquet_files if parquet_files is not None else data_config.get("dataset_config")
        if OmegaConf.is_config(manifest_path):
            manifest_path = OmegaConf.to_container(manifest_path, resolve=True)
        if isinstance(manifest_path, (list, tuple)):
            if len(manifest_path) != 1:
                raise ValueError("MixedVideoSFTDataset expects one manifest YAML path in train_files or val_files")
            manifest_path = manifest_path[0]
        if not isinstance(manifest_path, (str, Path)) or not str(manifest_path).strip():
            raise ValueError("Set data.dataset_config/train_files to a dataset manifest YAML path")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest does not exist: {self.manifest_path}")
        manifest = OmegaConf.to_container(
            OmegaConf.merge({"data": data_config}, OmegaConf.load(self.manifest_path)), resolve=True
        )
        if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
            raise ValueError(f"Dataset manifest must contain a datasets list: {self.manifest_path}")
        max_samples = _sample_limit(max_samples, "Global max_samples")
        self.datasets, self.sources = [], []
        names = set()
        for entry in manifest["datasets"]:
            if not isinstance(entry, dict):
                raise ValueError("Every datasets entry must be a mapping")
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f"Dataset names must be nonempty and unique, got {name!r}")
            names.add(name)
            enabled = entry.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError("Dataset enabled must be true or false")
            if not enabled:
                continue
            kind = entry.get("type")
            if kind not in ("streaming", "offline_video_qa", "videochat3"):
                raise ValueError(f"Dataset {name!r}: type must be streaming, offline_video_qa or videochat3")
            media_type = entry.get("media_type", "auto")
            if media_type not in ("image", "video", "text", "mixed", "auto"):
                raise ValueError(f"Dataset {name!r}: unsupported media_type {media_type!r}")
            if kind in ("streaming", "offline_video_qa") and media_type not in ("video", "auto"):
                raise ValueError(f"Dataset {name!r}: {kind} requires media_type=video")
            paths = entry.get("annotations")
            if isinstance(paths, str):
                paths = [paths]
            if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p.strip() for p in paths):
                raise ValueError(f"Dataset {name!r}: annotations must be a path or a nonempty list of paths")
            files = set()
            for path in paths:
                # Validate every declared path; a typo must not silently remove one shard/source.
                matched = [
                    file for file in annotation_files(_resolve_path(path, self.manifest_path.parent))
                    if file.is_file() and file.suffix.lower() in (".json", ".jsonl")
                ]
                if not matched:
                    raise FileNotFoundError(f"Dataset {name!r}: no JSON/JSONL annotations match {path!r}")
                files.update(matched)
            limit = _sample_limit(entry.get("max_samples", -1), f"Dataset {name!r} max_samples")
            repeat = entry.get("repeat", 1)
            if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
                raise ValueError(f"Dataset {name!r}: repeat must be a positive integer")
            options = entry.get("options") or {}
            if not isinstance(options, dict):
                raise ValueError(f"Dataset {name!r}: options must be a mapping")
            option_key = "streaming_video" if kind == "streaming" else kind
            source_config = OmegaConf.merge(
                OmegaConf.create(data_config), {option_key: options}
            )
            source_options = source_config[option_key]
            source_options["media_type"] = media_type
            if entry.get("video_root") is not None:
                source_options["video_root"] = _resolve_path(entry["video_root"], self.manifest_path.parent)
            elif source_options.get("video_root"):
                source_options["video_root"] = _resolve_path(source_options["video_root"], self.manifest_path.parent)
            if isinstance(source_options.get("path_map"), str) and source_options["path_map"]:
                source_options["path_map"] = _resolve_path(source_options["path_map"], self.manifest_path.parent)
            for path_key in ("annotation_root", "source_map"):
                if source_options.get(path_key):
                    source_options[path_key] = _resolve_path(source_options[path_key], self.manifest_path.parent)
            if kind == "streaming":
                dataset_class = StreamingVideoSFTDataset
            elif kind == "videochat3":
                from .videochat3 import VideoChat3SFTDataset

                dataset_class = VideoChat3SFTDataset
            else:
                from .offline_video_qa import OfflineVideoQASFTDataset

                dataset_class = OfflineVideoQASFTDataset
            try:
                dataset = dataset_class(
                    parquet_files=sorted(files), tokenizer=tokenizer, config=source_config,
                    processor=processor, max_samples=limit,
                )
            except EmptyDatasetError as error:
                logger.warning("Skipping empty dataset %r (%s): %s", name, kind, error)
                continue
            except (ValueError, FileNotFoundError) as error:
                raise type(error)(f"Dataset {name!r} ({kind}): {error}") from error
            if len(dataset) == 0:
                raise ValueError(f"Enabled dataset {name!r} has no usable samples")
            self.datasets.append(dataset)
            self.sources.append(dict(name=name, type=kind, media_type=media_type, samples=len(dataset), repeat=repeat))
        if not self.datasets:
            raise ValueError("Dataset manifest must enable at least one nonempty source")

        lengths = [source["samples"] * source["repeat"] for source in self.sources]
        total = sum(lengths)
        if 0 < max_samples < total:
            if max_samples < len(lengths):
                raise ValueError("Global max_samples must be at least the number of enabled sources to preserve mixing")
            # Reserve one sample per source, then distribute the remaining capacity proportionally.
            remaining = max_samples - len(lengths)
            capacities = [length - 1 for length in lengths]
            capacity = sum(capacities)
            allocated = [1 + remaining * count // capacity for count in capacities]
            remainders = [remaining * count % capacity for count in capacities]
            for index in sorted(range(len(lengths)), key=lambda i: (-remainders[i], i))[: max_samples - sum(allocated)]:
                allocated[index] += 1
            lengths = allocated
        self.cumulative_sizes = []
        total = 0
        for source, length in zip(self.sources, lengths, strict=True):
            total += length
            self.cumulative_sizes.append(total)
            source["mixed_samples"] = length
            logger.warning(
                "Mixed SFT source %s (%s, %s): %d base samples, repeat=%d, %d selected samples",
                source["name"], source["type"], source["media_type"], source["samples"], source["repeat"], length,
            )
        logger.warning("Mixed SFT: %d samples from %d sources in %s", total, len(self.sources), self.manifest_path)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, item):
        if item < 0 or item >= len(self):
            raise IndexError(item)
        source_index = bisect.bisect_right(self.cumulative_sizes, item)
        start = self.cumulative_sizes[source_index - 1] if source_index else 0
        offset = item - start
        source = self.sources[source_index]
        # Small global caps cover each source uniformly instead of only using its first annotations.
        index = (
            offset * source["samples"] // source["mixed_samples"]
            if source["mixed_samples"] < source["samples"] else offset % source["samples"]
        )
        return self.datasets[source_index][index]
