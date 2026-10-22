#!/usr/bin/env python3
"""Replay a local video with a trained Qwen3-VL streaming SFT checkpoint."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exported Hugging Face SFT checkpoint directory")
    parser.add_argument("--video", required=True, help="Local video file")
    parser.add_argument("--question", help="User instruction; read from stdin when omitted")
    parser.add_argument("--question-time", type=float, default=0.0, help="Absolute time when the instruction arrives")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--window-rounds", type=int, default=64)
    parser.add_argument("--video-max-frames", type=int, default=256)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum answer length after </Response>")
    parser.add_argument("--max-rounds", type=int, help="Optional replay limit")
    parser.add_argument("--frame-min-pixels", "--min-pixels", type=int, default=28 * 28)
    parser.add_argument("--frame-max-pixels", "--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a device such as cuda:0")
    parser.add_argument("--device-map", help="Transformers device map, e.g. auto, for a model spanning GPUs")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--temperature", type=float, default=0.0, help="Answer sampling temperature; 0 is greedy")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--realtime", action="store_true", help="Pace round arrival using the video clock")
    parser.add_argument("--output", type=Path, help="Create a new JSONL file; existing files are not overwritten")
    args = parser.parse_args()
    if args.fps <= 0 or args.start < 0 or args.question_time < 0:
        parser.error("fps must be positive; start and question-time must be nonnegative")
    if args.end is not None and args.end < args.start:
        parser.error("end must be greater than or equal to start")
    if args.window_rounds < 1 or args.max_new_tokens < 1 or args.max_context_tokens <= args.max_new_tokens + 1:
        parser.error("window-rounds and max-new-tokens must be positive, with enough max-context-tokens")
    if args.video_max_frames < 1:
        parser.error("video-max-frames must be positive")
    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error("max-rounds must be positive")
    if not 0 < args.frame_min_pixels <= args.frame_max_pixels:
        parser.error("pixels must satisfy 0 < frame-min-pixels <= frame-max-pixels")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be nonnegative and top-p must be in (0, 1]")
    if not Path(args.video).is_file():
        parser.error(f"video does not exist: {args.video}")
    if not Path(args.model).is_dir():
        parser.error(f"model must be a local Hugging Face checkpoint directory: {args.model}")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def encode_context(processor, rounds, question, question_observed_at, system_prompt):
    from verl.utils.dataset.streaming_video import build_user_message

    messages = [{"role": "system", "content": system_prompt}]
    images = []
    for index, turn in enumerate(rounds):
        turn_question = None
        if question_observed_at is not None:
            if turn["timestamp"] == question_observed_at:
                turn_question = question
            elif index == 0 and turn["timestamp"] > question_observed_at:
                turn_question = f"Earlier instruction received at <{question_observed_at:g}s>:\n{question}"
        messages.append(build_user_message(turn, turn["images"], question=turn_question))
        images.extend(turn["images"])
        if turn["assistant"] is not None:
            messages.append({"role": "assistant", "content": turn["assistant"]})
    text = processor.tokenizer.apply_chat_template(
        messages, chat_template=processor.chat_template,
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return processor(
        text=[text], images=images, return_tensors="pt", add_special_tokens=False, return_mm_token_type_ids=True
    )


def write_event(event, output):
    line = json.dumps(event, ensure_ascii=False)
    print(line, flush=True)
    if output is not None:
        output.write(line + "\n")
        output.flush()


def main():
    args = parse_args()
    if args.question is None:
        args.question = input("User instruction: ").strip()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from verl.utils.dataset.streaming_video import DEFAULT_SYSTEM_PROMPT, STATE_TOKENS, iter_video_rounds

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype_name = args.dtype
    if dtype_name == "auto":
        if device.startswith("cuda"):
            dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype_name = "float32"
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.frame_min_pixels,
        max_pixels=args.frame_max_pixels,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer
    state_ids = []
    for token in STATE_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] not in tokenizer.all_special_ids:
            raise ValueError(
                f"{token} must be one registered special token. Load the SFT checkpoint with its saved tokenizer."
            )
        state_ids.append(ids[0])
    if len(set(state_ids)) != len(STATE_TOKENS):
        raise ValueError("The streaming states must have distinct token IDs")

    load_kwargs = {"dtype": getattr(torch, dtype_name), "attn_implementation": args.attn_implementation}
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True, **load_kwargs)
    if model.config.model_type not in {"qwen3_vl", "qwen3_vl_moe"}:
        raise ValueError(f"Expected a Qwen3-VL checkpoint, found {model.config.model_type}")
    if args.device_map is None:
        model.to(device)
    model.eval()
    for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
        if embedding is None or embedding.weight.shape[0] <= max(state_ids):
            raise ValueError("Checkpoint embeddings do not contain the trained streaming state tokens")
    input_device = model.get_input_embeddings().weight.device
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)
    if not eos_ids or any(token_id in state_ids for token_id in eos_ids):
        raise ValueError("Checkpoint must define conversation EOS tokens distinct from the streaming states")
    text_config = getattr(model.config, "text_config", model.config)
    model_context = getattr(text_config, "max_position_embeddings", args.max_context_tokens)
    context_limit = min(args.max_context_tokens, model_context)
    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file is not None
        else DEFAULT_SYSTEM_PROMPT
    )
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0],
        "suppress_tokens": state_ids,
        "use_cache": True,
        "min_length": 0,
        "min_new_tokens": 0,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "num_return_sequences": 1,
    }
    if args.temperature > 0:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    rounds = []
    question_observed_at = None
    replay_started = time.perf_counter()
    round_count = 0
    with ExitStack() as stack, torch.inference_mode():
        output = None
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(args.output.open("x", encoding="utf-8"))
        video_rounds = iter_video_rounds(args.video, fps=args.fps, start=args.start, end=args.end)
        stack.callback(video_rounds.close)
        for turn, images in video_rounds:
            timestamp = turn["timestamp"]
            if args.realtime:
                delay = timestamp - args.start - (time.perf_counter() - replay_started)
                if delay > 0:
                    time.sleep(delay)
            if question_observed_at is None and timestamp >= args.question_time:
                question_observed_at = timestamp
            rounds.append({**turn, "images": images, "assistant": None})
            rounds = rounds[-args.window_rounds :]
            while sum(len(history["images"]) for history in rounds) > args.video_max_frames:
                rounds.pop(0)
            while True:
                inputs = encode_context(processor, rounds, args.question, question_observed_at, system_prompt)
                input_length = inputs["input_ids"].shape[-1]
                if input_length + 1 + args.max_new_tokens <= context_limit:
                    break
                if len(rounds) == 1:
                    raise ValueError(
                        f"Current round needs {input_length} input tokens plus answer budget, exceeding "
                        f"context limit {context_limit}. Reduce frame-max-pixels/max-new-tokens "
                        "or increase max-context-tokens."
                    )
                # Remove a complete historical round without truncating any image or the current round.
                rounds.pop(0)
            inputs = inputs.to(input_device)
            prediction = model(**inputs, logits_to_keep=1, use_cache=False)
            state_index = prediction.logits[0, -1, state_ids].argmax().item()
            state = STATE_TOKENS[state_index]
            del prediction
            write_event(
                {
                    "event": "state",
                    "video": str(Path(args.video)),
                    "start": turn["start"],
                    "timestamp": timestamp,
                    "frame_timestamps": turn["frame_timestamps"],
                    "state": state,
                    "context_rounds": len(rounds),
                    "context_frames": sum(len(history["images"]) for history in rounds),
                    "input_tokens": input_length,
                },
                output,
            )
            answer = ""
            if state == "</Response>":
                state_tensor = inputs["input_ids"].new_tensor([[state_ids[state_index]]])
                answer_inputs = dict(inputs)
                answer_inputs["input_ids"] = torch.cat([inputs["input_ids"], state_tensor], dim=-1)
                answer_inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 1))], dim=-1
                )
                if "mm_token_type_ids" in inputs:
                    answer_inputs["mm_token_type_ids"] = torch.cat(
                        [inputs["mm_token_type_ids"], inputs["mm_token_type_ids"].new_zeros((1, 1))], dim=-1
                    )
                generated = model.generate(**answer_inputs, **generation_kwargs)
                answer_ids = generated[0, input_length + 1 :]
                answer = tokenizer.decode(
                    answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).strip()
                answer_truncated = len(answer_ids) >= args.max_new_tokens and answer_ids[-1].item() not in eos_ids
                write_event(
                    {
                        "event": "answer",
                        "video": str(Path(args.video)),
                        "timestamp": timestamp,
                        "answer": answer,
                        "answer_truncated": answer_truncated,
                    },
                    output,
                )
            rounds[-1]["assistant"] = state + (f" {answer}" if answer else "")
            round_count += 1
            if args.max_rounds is not None and round_count >= args.max_rounds:
                break
    if round_count == 0:
        raise ValueError("The selected video interval contains no decodable rounds")


if __name__ == "__main__":
    main()
