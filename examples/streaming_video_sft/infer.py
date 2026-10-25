#!/usr/bin/env python3
"""Replay a local video with a trained Qwen3-VL checkpoint and cross-frame KV reuse."""

import argparse
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exported Hugging Face SFT checkpoint directory")
    parser.add_argument("--video", required=True, help="Local video file")
    parser.add_argument("--question", help="User instruction; read from stdin when omitted")
    parser.add_argument("--question-time", type=float, default=0.0, help="Absolute time when the instruction arrives")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--window-rounds", type=int, help="Optional visual rounds per KV window; unlimited by default")
    parser.add_argument(
        "--video-max-frames", type=int, help="Optional visual frames per KV window; unlimited by default",
    )
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--context-policy", choices=["sliding_window", "error"], default="sliding_window")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum answer length after </Response>")
    parser.add_argument("--max-rounds", type=int, help="Optional replay limit")
    parser.add_argument("--frame-min-pixels", "--min-pixels", type=int, default=32 * 32)
    parser.add_argument("--frame-max-pixels", "--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a device such as cuda:0")
    parser.add_argument("--device-map", help="Transformers device map, e.g. auto, for a model spanning GPUs")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "eager", "flash_attention_2"], default="flash_attention_2",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Answer sampling temperature; 0 is greedy")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--realtime", action="store_true", help="Pace round arrival using the video clock")
    parser.add_argument("--output", type=Path, help="Create a new JSONL file; existing files are not overwritten")
    args = parser.parse_args()
    if (not all(math.isfinite(value) for value in (args.fps, args.start, args.question_time))
            or args.fps <= 0 or args.start < 0 or args.question_time < 0):
        parser.error("fps must be positive; start and question-time must be nonnegative")
    if args.end is not None and (not math.isfinite(args.end) or args.end < args.start):
        parser.error("end must be greater than or equal to start")
    if args.max_new_tokens < 1 or args.max_context_tokens <= args.max_new_tokens + 1:
        parser.error("max-new-tokens must be positive, with enough max-context-tokens")
    if args.window_rounds is not None and args.window_rounds < 1:
        parser.error("window-rounds must be positive")
    if args.video_max_frames is not None and args.video_max_frames < 1:
        parser.error("video-max-frames must be positive")
    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error("max-rounds must be positive")
    if not 0 < args.frame_min_pixels <= args.frame_max_pixels:
        parser.error("pixels must satisfy 0 < frame-min-pixels <= frame-max-pixels")
    if (not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1
            or args.top_k < 0 or not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0):
        parser.error("Invalid temperature, top-p, top-k or repetition-penalty")
    if not Path(args.video).is_file():
        parser.error(f"video does not exist: {args.video}")
    if not Path(args.model).is_dir():
        parser.error(f"model must be a local Hugging Face checkpoint directory: {args.model}")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


class StreamingSession:
    def __init__(self, model, processor, args, state_tokens, state_ids, eos_ids, system_prompt):
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.generation.logits_process import (
            LogitsProcessorList, RepetitionPenaltyLogitsProcessor, SuppressTokensLogitsProcessor,
            TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
        )

        self.torch, self.model, self.processor, self.args = torch, model, processor, args
        self.tokenizer, self.system_prompt = processor.tokenizer, system_prompt
        self.state_tokens, self.state_ids, self.eos_ids = state_tokens, state_ids, eos_ids
        self.input_device = model.get_input_embeddings().weight.device
        self.past_key_values = DynamicCache(config=model.config)
        self.input_ids = torch.empty((1, 0), dtype=torch.long, device=self.input_device)
        self.next_position = 0
        self.window_rounds, self.window_frames, self.window_resets = 0, 0, 0
        self.window_start_timestamp = None
        self.processed_tokens, self.peak_tokens, self.vision_frames = 0, 0, 0
        self.evicted_tokens, self.evicted_rounds = 0, 0
        self.text_history_ids, self.question_history = [], []
        system_text = self.render_messages([dict(role="system", content=system_prompt)], False)
        self.system_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
        assistant_text = self.render_messages([dict(role="assistant", content=state_tokens[0])], False)
        _, marker, suffix = assistant_text.partition(state_tokens[0])
        self.assistant_suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        if not marker or not self.assistant_suffix_ids or self.assistant_suffix_ids[0] not in eos_ids:
            raise ValueError("The checkpoint chat template must close assistant turns with a conversation EOS")
        text_config = getattr(model.config, "text_config", model.config)
        self.context_limit = min(
            args.max_context_tokens, getattr(text_config, "max_position_embeddings", args.max_context_tokens),
        )
        self.round_reserve = 1 + args.max_new_tokens + len(self.assistant_suffix_ids)
        if self.context_limit <= self.round_reserve:
            raise ValueError("The model context limit leaves no room for the streaming prompt")
        self.answer_processors = LogitsProcessorList()
        if args.repetition_penalty != 1.0:
            self.answer_processors.append(RepetitionPenaltyLogitsProcessor(args.repetition_penalty))
        self.answer_processors.append(SuppressTokensLogitsProcessor(state_ids))
        if args.temperature > 0:
            self.answer_processors.append(TemperatureLogitsWarper(args.temperature))
            if args.top_k > 0:
                self.answer_processors.append(TopKLogitsWarper(args.top_k))
            if args.top_p < 1:
                self.answer_processors.append(TopPLogitsWarper(args.top_p))
        self.model.model.rope_deltas = None

    def render_messages(self, messages, add_generation_prompt):
        return self.tokenizer.apply_chat_template(
            messages, chat_template=self.processor.chat_template, tokenize=False,
            add_generation_prompt=add_generation_prompt, enable_thinking=False,
        )

    def encode_frame(self, user_message):
        torch = self.torch
        # Use an explicit system message, then remove its IDs when appending an incremental turn.
        text = self.render_messages([
            dict(role="system", content=self.system_prompt), user_message,
        ], True)
        images = [item["image"] for item in user_message["content"] if item["type"] == "image"]
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", add_special_tokens=False,
            return_mm_token_type_ids=True,
        )
        offset = len(self.system_ids)
        if inputs["input_ids"][0, :offset].tolist() != self.system_ids:
            raise ValueError("The checkpoint chat template must preserve the system-message prefix")
        if self.input_ids.shape[-1]:
            inputs["input_ids"] = inputs["input_ids"][:, offset:]
            inputs["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, offset:]
        elif self.text_history_ids:
            history = inputs["input_ids"].new_tensor([self.text_history_ids])
            inputs["input_ids"] = torch.cat([
                inputs["input_ids"][:, :offset], history, inputs["input_ids"][:, offset:],
            ], dim=-1)
            token_types = inputs["mm_token_type_ids"]
            inputs["mm_token_type_ids"] = torch.cat([
                token_types[:, :offset], token_types.new_zeros(history.shape), token_types[:, offset:],
            ], dim=-1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs

    def prefill(self, inputs):
        torch = self.torch
        inputs = {name: value.to(self.input_device) for name, value in inputs.items()}
        input_ids = inputs["input_ids"]
        previous_length, query_length = self.input_ids.shape[-1], input_ids.shape[-1]
        if self.past_key_values.get_seq_length() != previous_length:
            raise RuntimeError("Streaming KV cache and token history are out of sync")
        if inputs.get("image_grid_thw") is not None:
            positions, _ = self.model.model.get_rope_index(
                input_ids=input_ids, mm_token_type_ids=inputs["mm_token_type_ids"],
                image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs["attention_mask"],
            )
            position_advance = int(positions.max().item()) + 1
        else:
            positions = torch.arange(query_length, device=self.input_device).view(1, 1, -1).expand(3, 1, -1)
            position_advance = query_length
        # Plane 0 indexes KV slots; planes 1-3 continue the window's native multimodal positions.
        text_positions = torch.arange(
            previous_length, previous_length + query_length, device=self.input_device,
        ).view(1, 1, -1)
        inputs["position_ids"] = torch.cat([text_positions, positions + self.next_position], dim=0)
        inputs["attention_mask"] = input_ids.new_ones((1, previous_length + query_length))
        prediction = self.model(
            **inputs, past_key_values=self.past_key_values, logits_to_keep=1,
            use_cache=True, return_dict=True,
        )
        self.past_key_values = prediction.past_key_values
        if self.past_key_values is None or self.past_key_values.get_seq_length() != previous_length + query_length:
            raise RuntimeError("Qwen3-VL did not return the complete incremental KV cache")
        self.input_ids = torch.cat([self.input_ids, input_ids], dim=-1)
        self.next_position += position_advance
        self.processed_tokens += query_length
        self.peak_tokens = max(self.peak_tokens, self.input_ids.shape[-1])
        if inputs.get("image_grid_thw") is not None:
            self.vision_frames += inputs["image_grid_thw"].shape[0]
        return prediction.logits[:, -1, :]

    def prefill_text(self, token_ids):
        input_ids = self.input_ids.new_tensor([token_ids])
        return self.prefill(dict(input_ids=input_ids, attention_mask=self.torch.ones_like(input_ids)))

    def decode_answer(self, state_index):
        torch = self.torch
        state, state_id = self.state_tokens[state_index], self.state_ids[state_index]
        answer_ids, truncated = [], False
        if state == "</Response>":
            logits = self.prefill_text([state_id])
            for _ in range(self.args.max_new_tokens):
                scores = self.answer_processors(self.input_ids.to(logits.device), logits.float())
                if self.args.temperature > 0:
                    token_id = torch.multinomial(torch.softmax(scores, dim=-1), 1).item()
                else:
                    token_id = scores.argmax(dim=-1).item()
                if token_id in self.eos_ids:
                    break
                answer_ids.append(token_id)
                logits = self.prefill_text([token_id])
            else:
                truncated = True
            self.prefill_text(self.assistant_suffix_ids)
        else:
            self.prefill_text([state_id, *self.assistant_suffix_ids])
        answer = self.tokenizer.decode(
            answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        return answer, truncated, [state_id, *answer_ids, *self.assistant_suffix_ids]

    def restart_window(self):
        from transformers.cache_utils import DynamicCache

        self.evicted_tokens += self.input_ids.shape[-1]
        self.evicted_rounds += self.window_rounds
        self.past_key_values = None
        self.input_ids = self.input_ids.new_empty((1, 0))
        self.past_key_values = DynamicCache(config=self.model.config)
        self.next_position = 0
        self.window_rounds, self.window_frames = 0, 0
        self.window_start_timestamp = None
        self.window_resets += 1
        self.model.model.rope_deltas = None

    def step(self, turn, images, question=None, question_time=None):
        from verl.utils.dataset.streaming_video import build_user_message

        started = time.perf_counter()
        user_message = build_user_message(turn, images, question=question)
        inputs = self.encode_frame(user_message)
        cached_length = self.input_ids.shape[-1]
        reset_reason = None
        if cached_length:
            if cached_length + inputs["input_ids"].shape[-1] + self.round_reserve > self.context_limit:
                reset_reason = "max_context_tokens"
            elif self.args.window_rounds is not None and self.window_rounds >= self.args.window_rounds:
                reset_reason = "window_rounds"
            elif (self.args.video_max_frames is not None
                  and self.window_frames + len(images) > self.args.video_max_frames):
                reset_reason = "video_max_frames"
        if reset_reason is not None:
            if self.args.context_policy == "error":
                raise ValueError(f"Streaming window exceeds {reset_reason}; context-policy=error")
            self.restart_window()
            cached_length = 0
            inputs = self.encode_frame(user_message)
        new_input_length = inputs["input_ids"].shape[-1]
        if cached_length + new_input_length + self.round_reserve > self.context_limit:
            raise ValueError(
                f"Streaming context requires {cached_length} cached and {new_input_length} new input tokens "
                f"plus {self.round_reserve} state/answer/closing tokens; limit={self.context_limit}. "
                "All history text is retained; reduce image/answer budgets or increase max-context-tokens."
            )
        if not cached_length:
            self.window_start_timestamp = turn["timestamp"]
        logits = self.prefill(inputs)
        state_index = logits[0, self.state_ids].argmax().item()
        del logits
        if question is not None:
            self.question_history.append(dict(
                timestamp=question_time, observed_at=turn["timestamp"], question=question,
            ))
        yield dict(
            event="state", video=str(Path(self.args.video)), start=turn["start"], timestamp=turn["timestamp"],
            frame_timestamps=turn["frame_timestamps"], state=self.state_tokens[state_index],
            context_rounds=self.window_rounds + 1, context_frames=self.window_frames + len(images),
            input_tokens=cached_length + new_input_length, new_input_tokens=new_input_length,
            cached_tokens_reused=cached_length, kv_cache_tokens=self.input_ids.shape[-1],
            mrope_next_position=self.next_position, window_id=self.window_resets,
            window_reset=reset_reason is not None, window_reset_reason=reset_reason,
            window_start_timestamp=self.window_start_timestamp, text_history_tokens=len(self.text_history_ids),
            state_latency_sec=time.perf_counter() - started,
        )
        answer, truncated, assistant_ids = self.decode_answer(state_index)
        self.window_rounds += 1
        self.window_frames += len(images)
        history_question = (
            f"User question received at <{question_time:g}s>:\n{question}" if question is not None else None
        )
        history_message = build_user_message(turn, images, question=history_question)
        history_message["content"] = [item for item in history_message["content"] if item["type"] == "text"]
        history_text = self.render_messages([
            dict(role="system", content=self.system_prompt), history_message,
        ], True)
        history_ids = self.tokenizer.encode(history_text, add_special_tokens=False)
        self.text_history_ids.extend(history_ids[len(self.system_ids):])
        self.text_history_ids.extend(assistant_ids)
        if self.state_tokens[state_index] == "</Response>":
            yield dict(
                event="answer", video=str(Path(self.args.video)), timestamp=turn["timestamp"],
                answer=answer, answer_truncated=truncated, kv_cache_tokens=self.input_ids.shape[-1],
                wall_time_sec=time.perf_counter() - started,
            )

    def close(self):
        self.past_key_values = None
        self.input_ids = None
        self.model.model.rope_deltas = None


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
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

    from verl.utils.dataset.streaming_video import DEFAULT_SYSTEM_PROMPT, STATE_TOKENS, iter_video_rounds

    set_seed(args.seed)
    if STATE_TOKENS != ("</Silence>", "</Response>"):
        raise ValueError("The Demo requires the two-state streaming protocol")
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
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)
    if not eos_ids or any(token_id in state_ids for token_id in eos_ids):
        raise ValueError("Checkpoint must define conversation EOS tokens distinct from the streaming states")
    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file is not None
        else DEFAULT_SYSTEM_PROMPT
    )
    question_observed_at = None
    replay_started = time.perf_counter()
    round_count = 0
    with ExitStack() as stack, torch.inference_mode():
        output = None
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(args.output.open("x", encoding="utf-8"))
        session = StreamingSession(model, processor, args, STATE_TOKENS, state_ids, eos_ids, system_prompt)
        stack.callback(session.close)
        video_rounds = iter_video_rounds(args.video, fps=args.fps, start=args.start, end=args.end)
        stack.callback(video_rounds.close)
        for turn, images in video_rounds:
            timestamp = turn["timestamp"]
            if args.realtime:
                delay = timestamp - args.start - (time.perf_counter() - replay_started)
                if delay > 0:
                    time.sleep(delay)
            new_question = question_observed_at is None and timestamp >= args.question_time - 1e-9
            if new_question:
                question_observed_at = timestamp
            for event in session.step(
                turn, images, question=args.question if new_question else None, question_time=args.question_time,
            ):
                write_event(event, output)
            round_count += 1
            if args.max_rounds is not None and round_count >= args.max_rounds:
                break
        if round_count == 0:
            raise ValueError("The selected video interval contains no decodable rounds")
        write_event(dict(
            event="summary", video=str(Path(args.video)), rounds=round_count,
            wall_time_sec=time.perf_counter() - replay_started, question_history=session.question_history,
            cache_statistics=dict(
                processed_tokens=session.processed_tokens, peak_cached_tokens=session.peak_tokens,
                final_cached_tokens=session.input_ids.shape[-1], evicted_tokens=session.evicted_tokens,
                evicted_rounds=session.evicted_rounds, window_resets=session.window_resets,
                vision_frames_encoded=session.vision_frames, retained_text_tokens=len(session.text_history_ids),
            ),
            protocol=dict(
                name="qwen3-vl-stream-demo-kv-v1", model_path=str(Path(args.model).resolve()),
                sample_fps=args.fps, infer_fps=args.fps, frames_per_decision=1, states=list(STATE_TOKENS),
                state_decoding="argmax_over_state_token_ids", system_prompt=system_prompt, enable_thinking=False,
                min_pixels=args.frame_min_pixels, max_pixels=args.frame_max_pixels,
                max_context_tokens=session.context_limit, max_new_tokens=args.max_new_tokens,
                context_policy=args.context_policy, dtype=dtype_name, attn_implementation=args.attn_implementation,
                kv_cache="DynamicCache", incremental_prefill=True, cross_frame_cache_reuse=True,
                cache_eviction="full_window_restart" if args.context_policy == "sliding_window" else "none",
                history_tokens="original_generated_ids", restart_history="all_text_without_past_images",
                mrope_positions="window_relative", frame_time="interval_right_endpoint", time_text="<Xs>",
                window_rounds=args.window_rounds, video_max_frames=args.video_max_frames,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty, seed=args.seed,
            ),
        ), output)


if __name__ == "__main__":
    main()
