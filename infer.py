#!/usr/bin/env python3
"""Clean single-image inference for the MLX Unlimited-OCR port.

Wraps mlx-vlm's deepseekocr Model + DeepseekOCRProcessor with:
  - proper detokenization (no byte-BPE Ġ/Ċ artifacts)
  - gundam (crop tiling) vs base mode selection
"""
import argparse, time
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.utils import generate_step
from PIL import Image


def build_inputs(model, processor, prompt, image_path, mode):
    if mode == "gundam":
        base_size, image_size, cropping = 1024, 640, True
    else:  # base
        base_size, image_size, cropping = 1024, 1024, False
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    img = Image.open(image_path).convert("RGB")
    out = processor.process_one(
        prompt=prompt, images=[img], inference_mode=True,
        base_size=base_size, image_size=image_size, cropping=cropping,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx_unlimited_bf16")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown.")
    ap.add_argument("--mode", choices=["gundam", "base"], default="gundam")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    model, processor = load(args.model)
    tok = processor.tokenizer

    inp = build_inputs(model, processor, args.prompt, args.image, args.mode)
    input_ids = inp["input_ids"]
    pixel_values = inp["images"]
    kwargs = dict(
        images_seq_mask=inp["images_seq_mask"],
        images_spatial_crop=inp["images_spatial_crop"],
    )

    # prime the vision features + prefill via model.__call__ inside generate_step
    t0 = time.time()
    detok = tok.detokenizer
    detok.reset()
    n = 0
    eos_ids = set(getattr(tok, "eos_token_ids", []) or [tok.eos_token_id])
    text_parts = []
    sampler = lambda logits: mx.argmax(logits, axis=-1)
    for (token, _logprobs) in generate_step(
        input_ids[0], model, pixel_values=pixel_values,
        max_tokens=args.max_tokens, sampler=sampler, **kwargs,
    ):
        tid = token.item() if hasattr(token, "item") else int(token)
        if tid in eos_ids:
            break
        detok.add_token(tid)
        n += 1
    detok.finalize()
    text = detok.text
    dt = time.time() - t0
    print("===== OUTPUT =====")
    print(text)
    print("==================")
    print(f"{n} tokens in {dt:.1f}s  ({n/max(dt,1e-9):.1f} tok/s)")


if __name__ == "__main__":
    main()
