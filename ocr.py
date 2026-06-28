#!/usr/bin/env python3
"""MLX Unlimited-OCR: clean OCR + bounding-box visualization.

- Detokenizes the byte-level BPE stream correctly (no Ġ/Ċ artifacts).
- Parses <|det|>label[x1,y1,x2,y2]<|/det|> layout boxes.
- Draws the boxes on the source image (faithful to the HF reference).
"""
import argparse
import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlx_vlm import load
from mlx_vlm.generate import stream_generate


# ----- byte-level (GPT-2 style) decoder: maps Ġ->space, Ċ->newline, etc. -----
@lru_cache()
def _byte_decoder():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def bpe_decode(tokens_concat: str) -> str:
    bd = _byte_decoder()
    buf = bytearray()
    out = []
    for ch in tokens_concat:
        if ch in bd:
            buf.append(bd[ch])
        else:  # token char outside the byte map (e.g. exotic special token) -> keep literal
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            out.append(ch)
    if buf:
        out.append(buf.decode("utf-8", errors="replace"))
    return "".join(out)


# ----- box parsing (mirrors modeling_unlimitedocr.re_match) -----
def parse_boxes(text):
    refs = []
    ref_pattern = r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>"
    for label, box in re.findall(ref_pattern, text, re.DOTALL):
        refs.append((label.strip(), box))
    det_pattern = r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>"
    for label, box in re.findall(det_pattern, text, re.DOTALL):
        refs.append((label.strip(), box))
    return refs


def draw_boxes(image, refs):
    W, H = image.size
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    rng = np.random.default_rng(0)
    for label, box in refs:
        try:
            coords = eval(box)
        except Exception:
            continue
        if coords and isinstance(coords[0], (int, float)):
            coords = [coords]
        color = tuple(int(c) for c in rng.integers(0, 220, size=3))
        for x1, y1, x2, y2 in coords:
            x1 = int(x1 / 999 * W); y1 = int(y1 / 999 * H)
            x2 = int(x2 / 999 * W); y2 = int(y2 / 999 * H)
            width = 4 if label == "title" else 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            od.rectangle([x1, y1, x2, y2], fill=color + (28,))
            ty = max(0, y1 - 16)
            tb = draw.textbbox((0, 0), label, font=font)
            draw.rectangle([x1, ty, x1 + (tb[2] - tb[0]) + 2, ty + (tb[3] - tb[1]) + 2],
                           fill=(255, 255, 255))
            draw.text((x1 + 1, ty), label, font=font, fill=color)
    img.paste(overlay, (0, 0), overlay)
    return img


def run(model, processor, image_path, prompt, max_tokens):
    tok = processor.tokenizer
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    ids = []
    for r in stream_generate(model, processor, prompt, image=[image_path],
                             max_tokens=max_tokens, temperature=0.0, verbose=False):
        if r.token is not None:
            ids.append(int(r.token))
    pieces = tok.convert_ids_to_tokens(ids)
    text = bpe_decode("".join(pieces))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx_unlimited_bf16")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown.")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-image", default=None)
    args = ap.parse_args()

    model, processor = load(args.model)
    text = run(model, processor, args.image, args.prompt, args.max_tokens)

    # clean markdown = strip the grounding/det tags
    clean = re.sub(r"<\|/?(ref|det|grounding)\|>", "", text)
    clean = re.sub(r"^\s*(doc|title|text|aside_text|page_number|page_footnote)\s*\[[^\]]*\]\s*",
                   "", clean, flags=re.MULTILINE)
    clean = re.sub(r"[A-Za-z_]+\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\]", "", clean)

    print(clean)

    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(clean)
        print(f"\n[saved markdown -> {args.out_md}]")

    refs = parse_boxes(text)
    if args.out_image:
        annotated = draw_boxes(Image.open(args.image), refs)
        annotated.save(args.out_image)
        print(f"[saved {len(refs)} boxes -> {args.out_image}]")


if __name__ == "__main__":
    main()
