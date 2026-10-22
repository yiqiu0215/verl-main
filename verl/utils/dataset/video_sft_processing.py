# Copyright 2026 The VideoLLM-Stream authors
# Licensed under the Apache License, Version 2.0.

"""Text encoding with Qwen3-VL image lengths, before allocating visual patches."""


def image_token_lengths(processor, images, image_kwargs):
    return image_token_lengths_from_sizes(processor, (image.size for image in images), image_kwargs)


def image_token_lengths_from_sizes(processor, sizes, image_kwargs):
    image_processor = processor.image_processor
    merge_length = image_processor.merge_size**2
    lengths_by_size = {}
    lengths = []
    for size in sizes:
        if size not in lengths_by_size:
            width, height = size
            patches = image_processor.get_number_of_image_patches(height, width, image_kwargs)
            lengths_by_size[size] = int(patches) // merge_length
        lengths.append(lengths_by_size[size])
    return lengths


def encode_image_text(processor, text, image_lengths):
    if image_lengths:
        token = processor.image_token
        parts = text.split(token)
        if len(parts) != len(image_lengths) + 1:
            raise ValueError("Image placeholders must match the number of sampled images")
        text = (
            "".join(part + token * length for part, length in zip(parts[:-1], image_lengths, strict=True)) + parts[-1]
        )
    return dict(
        processor(text=[text], return_tensors="pt", add_special_tokens=False, return_mm_token_type_ids=False)
    )


def process_sft_images(processor, images, image_kwargs, image_lengths):
    if not images:
        return {}
    inputs = dict(processor.image_processor(images=images, return_tensors="pt", **image_kwargs))
    actual_lengths = (inputs["image_grid_thw"].prod(dim=-1) // processor.image_processor.merge_size**2).tolist()
    if actual_lengths != image_lengths:
        raise ValueError("Image token length estimation must match the processed image grids")
    return inputs
