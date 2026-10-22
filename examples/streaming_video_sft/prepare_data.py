"""Inspect stream-meta-v1 annotations or export records with explicitly resolved media paths."""

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

# Load the standard-library-only adapter without importing verl's Ray/Torch package initializer.
_module_path = Path(__file__).resolve().parents[2] / "verl/utils/dataset/streaming_video.py"
_spec = importlib.util.spec_from_file_location("streaming_video_data", _module_path)
streaming = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(streaming)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="+", required=True, help="JSONL/JSON files, directories or globs")
    parser.add_argument("--video-root", default=streaming.DEFAULT_VIDEO_ROOT)
    parser.add_argument("--path-map", help="JSON path map, keyed by source/video_source/video_name, path or video_name")
    parser.add_argument("--path-replacements", help="JSON object/file containing old-prefix to new-prefix replacements")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--round-seconds", type=float, default=1.0)
    parser.add_argument("--missing-video", choices=("skip", "error"), default="skip")
    parser.add_argument(
        "--inspect-only", action="store_true", help="Inspect annotation labels without requiring media files"
    )
    parser.add_argument(
        "--output", help="New output JSONL with resolved media paths; existing files are never overwritten"
    )
    parser.add_argument(
        "--limit", "--limit-sessions", type=int, default=0, help="Maximum annotation records to inspect (0: all)"
    )
    args = parser.parse_args()
    if args.inspect_only and args.output:
        parser.error("--output requires checking that media exists; omit --inspect-only")
    path_map = streaming.load_path_mapping(args.path_map)
    replacements = {}
    if args.path_replacements:
        replacements = (
            json.loads(args.path_replacements)
            if args.path_replacements.lstrip().startswith("{")
            else streaming.load_path_mapping(args.path_replacements)
        )
    counts = Counter()
    output = Path(args.output).open("x", encoding="utf-8") if args.output else None
    try:
        for record, reference, line_number in streaming.load_annotations(args.annotations):
            counts["records"] += 1
            path = streaming.resolve_video_path(record, args.video_root, path_map, replacements)
            counts["empty_video_path"] += int(not path)
            exists = path and (Path(path).is_dir() if record.get("media_type") == "sequence" else Path(path).is_file())
            counts["missing_media"] += int(not exists)
            if not exists and not args.inspect_only:
                if args.missing_video == "error":
                    raise FileNotFoundError(f"Missing media at {reference[0]}:{line_number}: {path!r}")
                counts["skipped"] += 1
            else:
                turns = streaming.build_stream_turns(record, args.fps, args.round_seconds)
                counts["accepted"] += 1
                counts["round_samples"] += len(turns)
                counts["sampled_frames"] += sum(len(turn["frame_timestamps"]) for turn in turns)
                counts.update({f"state_{key}": value for key, value in Counter(t["state"] for t in turns).items()})
                if output:
                    resolved = dict(record, video_path=path)
                    output.write(json.dumps(resolved, ensure_ascii=False) + "\n")
            if args.limit and counts["records"] >= args.limit:
                break
    finally:
        if output:
            output.close()
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
