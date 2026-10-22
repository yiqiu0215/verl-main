# Copyright 2026 The VideoLLM-Stream authors
# Licensed under the Apache License, Version 2.0.

"""Read the published VideoChat3 layout, either extracted or in its original tar shards."""

import json
import math
import tarfile
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from .offline_video_qa import sample_offline_timestamps
from .streaming_video import parse_time

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class VideoChat3Media:
    @classmethod
    @lru_cache(maxsize=8)
    def shared(cls, root, cache_archives=2):
        return cls(root, cache_archives)

    def __init__(self, root, cache_archives=2):
        self.root = Path(root).expanduser()
        self.cache_archives = int(cache_archives)
        if self.cache_archives < 1:
            raise ValueError("cache_archives must be positive")
        self.indices, self.archives = {}, OrderedDict()

    @staticmethod
    def _source_key(root, filename):
        # Motion manifests retain the uploader's absolute prefix; the source config uses this relative suffix.
        marker = "videochat3_data_annotations/"
        if marker in root:
            root = marker + root.split(marker, 1)[1]
        return root.rstrip("/"), filename

    def resolve(self, source, filename):
        path = Path(filename).expanduser()
        if path.is_absolute():
            return {"path": str(path)} if path.exists() else None
        path = self.root / source["media_subdir"] / filename
        if path.exists():
            return {"path": str(path)}
        category = source["media_subdir"].split("/")[0]
        if category not in self.indices:
            manifest = self.root / category / "manifest.jsonl"
            index = {}
            if manifest.is_file():
                with manifest.open(encoding="utf-8") as stream:
                    for line in stream:
                        item = json.loads(line)
                        key = self._source_key(item["media_root"], item["source_filename"])
                        index[key] = (item["shard"], item["member"])
            self.indices[category] = index
        key = self._source_key(source["media_root"], filename)
        entry = self.indices[category].get(key)
        if entry is None:
            return None
        shard, member = entry
        archive = self.root / category / shard
        return {"archive": str(archive), "member": member} if archive.is_file() else None

    def _archive(self, path):
        if path not in self.archives:
            self.archives[path] = tarfile.open(path, "r:")
        self.archives.move_to_end(path)
        while len(self.archives) > self.cache_archives:
            _, archive = self.archives.popitem(last=False)
            archive.close()
        return self.archives[path]

    @contextmanager
    def open(self, resource):
        if "path" in resource:
            with open(resource["path"], "rb") as stream:
                yield stream
        else:
            stream = self._archive(resource["archive"]).extractfile(resource["member"])
            if stream is None:
                raise ValueError(f"Expected a media file, got {resource}")
            with stream:
                yield stream

    def image(self, resource):
        from PIL import Image

        with self.open(resource) as stream, Image.open(stream) as image:
            return image.convert("RGB")

    @staticmethod
    def _timestamps(duration, metadata, fps, max_frames):
        start = parse_time(metadata.get("clip_start_time") or 0.0)
        end = metadata.get("clip_end_time")
        end = duration if end is None else min(parse_time(end), duration)
        if end <= start:
            raise ValueError(f"Invalid video clip: {start}s to {end}s, duration {duration}s")
        return [start + t for t in sample_offline_timestamps(end - start, fps, max_frames)]

    def video(self, resource, metadata, fps, max_frames):
        from PIL import Image

        backend = metadata.get("video_backend", "")
        filename = resource.get("path", resource.get("member", ""))
        if backend in {"img", "img2", "frame", "frame2", "indexed_jpg"}:
            if "path" in resource:
                frames = sorted(
                    (p for p in Path(resource["path"]).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda p: p.name,
                )
                resources = [{"path": str(path)} for path in frames]
            else:
                prefix = resource["member"].rstrip("/") + "/"
                archive = self._archive(resource["archive"])
                names = sorted(
                    item.name for item in archive.getmembers()
                    if item.isfile() and item.name.startswith(prefix)
                    and Path(item.name).suffix.lower() in IMAGE_SUFFIXES
                )
                resources = [{"archive": resource["archive"], "member": name} for name in names]
            source_fps = float(metadata.get("fps") or 0)
            if not resources or not math.isfinite(source_fps) or source_fps <= 0:
                raise ValueError(f"Frame video requires images and positive metadata fps: {resource}")
            duration = len(resources) / source_fps
            times = self._timestamps(duration, metadata, fps, max_frames)
            images = [self.image(resources[min(math.floor(t * source_fps), len(resources) - 1)]) for t in times]
            return times, images
        if backend == "gif" or filename.lower().endswith(".gif"):
            with self.open(resource) as stream, Image.open(stream) as gif:
                source_fps = float(metadata.get("fps") or 0)
                if source_fps <= 0:
                    duration = float(metadata.get("duration") or 0)
                    if duration <= 0:
                        raise ValueError(f"GIF requires metadata fps or duration: {resource}")
                    source_fps = gif.n_frames / duration
                times = self._timestamps(gif.n_frames / source_fps, metadata, fps, max_frames)
                images = []
                for timestamp in times:
                    gif.seek(min(math.floor(timestamp * source_fps), gif.n_frames - 1))
                    images.append(gif.convert("RGB"))
                return times, images
        from torchcodec.decoders import VideoDecoder

        with self.open(resource) as stream:
            decoder = VideoDecoder(stream, dimension_order="NHWC", device="cpu", seek_mode="exact")
            begin, end = decoder.metadata.begin_stream_seconds, decoder.metadata.end_stream_seconds
            times = self._timestamps(end - begin, metadata, fps, max_frames)
            images = [
                Image.fromarray(
                    decoder.get_frame_played_at(min(begin + t, math.nextafter(end, -math.inf))).data.numpy()
                ).convert("RGB")
                for t in times
            ]
        return times, images
