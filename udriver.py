#!/usr/bin/env python3
"""Faithful Unlimited-OCR generation driver for MLX.

Adds the Unlimited-OCR "long-horizon" decode mechanics on top of the
mlx-vlm DeepSeek-OCR model (same architecture):

  * PrefillRingCache  - full attention over all image+prompt (prefill) tokens,
    then a fixed-size ring buffer over the most recent `window` generated
    tokens. Memory is bounded => effectively unbounded output length.
  * NoRepeatNgram     - sliding-window n-gram repetition blocker
    (ngram_size=35, window=128 single / 1024 multi-page), the same logit
    processor the reference uses to avoid degenerate loops on long output.
  * infer / infer_multi - single image (gundam/base) and multi-page parsing.
"""
import re
from functools import lru_cache

import mlx.core as mx
import numpy as np
from PIL import Image


# ----------------------------- clean detokenize -----------------------------
@lru_cache()
def _byte_decoder():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def bpe_decode(s):
    bd = _byte_decoder(); buf = bytearray(); out = []
    for ch in s:
        if ch in bd:
            buf.append(bd[ch])
        else:
            if buf:
                out.append(buf.decode("utf-8", "replace")); buf = bytearray()
            out.append(ch)
    if buf:
        out.append(buf.decode("utf-8", "replace"))
    return "".join(out)


# ------------------------- sliding-window ring cache -------------------------
class PrefillRingCache:
    """Keep ALL prefill K/V; ring-buffer the last `window` decode tokens."""
    def __init__(self, window):
        self.window = window
        self.keys = None
        self.values = None
        self.offset = 0            # absolute position (drives RoPE)
        self.prefill_len = None
        self.ring_pos = 0
        self.decoded = 0

    def update_and_fetch(self, keys, values):
        B, H, L, D = keys.shape
        if self.keys is None:                       # prefill
            self.keys, self.values = keys, values
            self.prefill_len = L
            self.offset = L
            return self.keys, self.values
        for t in range(L):                          # decode (L is usually 1)
            kt = keys[:, :, t:t + 1, :]
            vt = values[:, :, t:t + 1, :]
            if self.window is None or self.decoded < self.window:
                self.keys = mx.concatenate([self.keys, kt], axis=2)
                self.values = mx.concatenate([self.values, vt], axis=2)
            else:
                slot = self.prefill_len + self.ring_pos
                self.keys[:, :, slot:slot + 1, :] = kt
                self.values[:, :, slot:slot + 1, :] = vt
                self.ring_pos = (self.ring_pos + 1) % self.window
            self.decoded += 1
            self.offset += 1
        return self.keys, self.values


# --------------------------- n-gram no-repeat --------------------------------
def apply_no_repeat_ngram(logits, seq, ngram_size, window):
    if ngram_size <= 0 or window <= 0 or len(seq) < ngram_size:
        return logits
    start = max(0, len(seq) - window)
    end = len(seq) - ngram_size + 1
    if end <= start:
        return logits
    prefix = tuple(seq[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
    banned = set()
    for i in range(start, end):
        ng = seq[i:i + ngram_size]
        if ngram_size == 1 or tuple(ng[:-1]) == prefix:
            banned.add(ng[-1])
    if banned:
        idx = mx.array(sorted(banned))
        logits[0, idx] = -float("inf")
    return logits


def make_ngram_logits_processor(ngram_size=35, window=128):
    """mlx-lm-compatible processor: (tokens, logits)->logits. Blocks n-gram repeats
    within `window` of the most recent generated tokens (same rule as the reference)."""
    def proc(tokens, logits):
        seq = tokens.tolist()
        if len(seq) < ngram_size:
            return logits
        start = max(0, len(seq) - window)
        end = len(seq) - ngram_size + 1
        if end <= start:
            return logits
        prefix = tuple(seq[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
        banned = set()
        for i in range(start, end):
            ng = seq[i:i + ngram_size]
            if ngram_size == 1 or tuple(ng[:-1]) == prefix:
                banned.add(ng[-1])
        if banned:
            idx = mx.array(sorted(banned))
            if logits.ndim == 2:
                logits[0, idx] = -float("inf")
            else:
                logits[idx] = -float("inf")
        return logits
    return proc


# ------------------------------ generation -----------------------------------
def _lm_parts(model):
    lm = model.language_model
    return lm.model.embed_tokens, lm.model.layers, lm.model.norm, lm.lm_head


def _prefill_logits(model, inputs_embeds, caches):
    _, layers, norm, lm_head = _lm_parts(model)
    h = inputs_embeds
    for layer, c in zip(layers, caches):
        h = layer(h, "causal", c)
    h = norm(h[:, -1:, :])
    return lm_head(h)[:, -1, :]   # (1, vocab)


def _decode_logits(model, token, caches):
    embed, layers, norm, lm_head = _lm_parts(model)
    h = embed(mx.array([[token]]))
    for layer, c in zip(layers, caches):
        h = layer(h, None, c)
    h = norm(h)
    return lm_head(h)[:, -1, :]


def generate_tokens(model, processor, inputs, max_tokens, window, ngram_size, ngram_window):
    embeds = model.get_input_embeddings(
        inputs["input_ids"], inputs["images"],
        inputs["images_spatial_crop"], inputs["images_seq_mask"],
    ).inputs_embeds
    n_layers = len(model.language_model.model.layers)
    caches = [PrefillRingCache(window) for _ in range(n_layers)]

    tok = processor.tokenizer
    _e = getattr(tok, "eos_token_ids", None) or tok.eos_token_id
    eos = set(_e) if isinstance(_e, (list, tuple, set)) else {_e}

    logits = _prefill_logits(model, embeds, caches)
    seq = []
    for _ in range(max_tokens):
        logits = apply_no_repeat_ngram(logits, seq, ngram_size, ngram_window)
        token = int(mx.argmax(logits, axis=-1).item())
        if token in eos:
            break
        seq.append(token)
        logits = _decode_logits(model, token, caches)
        mx.eval(logits)
    pieces = tok.convert_ids_to_tokens(seq)
    return seq, bpe_decode("".join(pieces))


# ------------------------------- public API ----------------------------------
def _build_inputs(processor, prompt, image_path, mode):
    base_size, image_size, cropping = (
        (1024, 640, True) if mode == "gundam" else (1024, 1024, False)
    )
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    img = Image.open(image_path).convert("RGB")
    return processor.process_one(
        prompt=prompt, images=[img], inference_mode=True,
        base_size=base_size, image_size=image_size, cropping=cropping,
    )


def infer(model, processor, image_path, prompt="<image>\n<|grounding|>Convert the document to markdown.",
          mode="gundam", max_tokens=8192, window=128, ngram_size=35, ngram_window=128):
    inputs = _build_inputs(processor, prompt, image_path, mode)
    _, text = generate_tokens(model, processor, inputs, max_tokens, window, ngram_size, ngram_window)
    return text


def infer_multi(model, processor, image_paths, prompt="<image>\nMulti page parsing.",
                max_tokens=8192, window=128, ngram_size=35, ngram_window=1024):
    """Multi-page: base mode (image_size=1024), wider n-gram window (matches reference)."""
    results = []
    for p in image_paths:
        results.append(infer(model, processor, p, prompt=prompt, mode="base",
                             max_tokens=max_tokens, window=window,
                             ngram_size=ngram_size, ngram_window=ngram_window))
    return results


# ------------------------------ box drawing ----------------------------------
def parse_boxes(text):
    refs = []
    for label, box in re.findall(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", text, re.DOTALL):
        refs.append((label.strip(), box))
    for label, box in re.findall(r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>", text, re.DOTALL):
        refs.append((label.strip(), box))
    return refs


def draw_boxes(image, refs):
    from PIL import ImageDraw, ImageFont
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
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4 if label == "title" else 2)
            od.rectangle([x1, y1, x2, y2], fill=color + (28,))
            ty = max(0, y1 - 16)
            tb = draw.textbbox((0, 0), label, font=font)
            draw.rectangle([x1, ty, x1 + (tb[2] - tb[0]) + 2, ty + (tb[3] - tb[1]) + 2], fill=(255, 255, 255))
            draw.text((x1 + 1, ty), label, font=font, fill=color)
    img.paste(overlay, (0, 0), overlay)
    return img


def parse_layout(text):
    """Parse regions WITH their following text: [(label, [x1,y1,x2,y2], text), ...]."""
    out = []
    pat = (r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*\[([^\]]+)\]\s*<\|/det\|>"
           r"(.*?)(?=<\|det\|>|<\|ref\|>|$)")
    for label, box, body in re.findall(pat, text, re.DOTALL):
        try:
            coords = [int(float(x)) for x in box.split(",")]
            if len(coords) == 4:
                out.append((label.strip().lower(), coords, body.strip()))
        except Exception:
            continue
    return out


_FIG_CAP = re.compile(r"^\s*(fig(?:ure)?\.?|table)\s*\.?\s*\d", re.I)


def extract_figures_smart(image, text, out_dir, prefix="fig", top_margin=0.04):
    """Robust figure extraction: model `image` boxes + caption-anchored fallback.

    For every figure/table caption the model finds, if no `image` box sits directly
    above it, the figure is reconstructed as the page-width band between the region
    above and the caption — recovering plain-background plots the model under-detects.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    W, H = image.size
    regions = parse_layout(text)
    img_boxes = [c for l, c, _ in regions if l in ("image", "figure")]
    captions = [c for l, c, t in regions
                if l in ("image_caption", "table_caption") or _FIG_CAP.match(t)]
    # content width from all detected regions (fallback to full page)
    xs1 = [c[0] for _, c, _ in regions] or [20]
    xs2 = [c[2] for _, c, _ in regions] or [979]
    cx1, cx2 = min(xs1), max(xs2)

    boxes = list(img_boxes)
    for cap in sorted(captions, key=lambda c: c[1]):  # top to bottom
        cap_top = cap[1]
        # already an image directly above this caption? then skip
        if any(b[3] <= cap_top + 10 and b[3] >= cap_top - 320 for b in img_boxes):
            continue
        # nearest region bottom above the caption (else a top margin)
        aboves = [c[3] for _, c, _ in regions if c[3] < cap_top - 10]
        top = max(aboves) + 5 if aboves else int(top_margin * 999)
        if cap_top - top < 40:          # too thin to be a figure
            continue
        boxes.append([cx1, top, cx2, cap_top - 3])

    # dedupe near-identical boxes, then crop
    saved, seen = [], []
    i = 0
    for x1, y1, x2, y2 in boxes:
        key = (round(x1, -1), round(y1, -1), round(x2, -1), round(y2, -1))
        if key in seen:
            continue
        seen.append(key)
        px1, py1 = int(x1 / 999 * W), int(y1 / 999 * H)
        px2, py2 = int(x2 / 999 * W), int(y2 / 999 * H)
        if px2 <= px1 or py2 <= py1:
            continue
        p = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
        image.convert("RGB").crop((px1, py1, px2, py2)).save(p)
        saved.append(p); i += 1
    return saved


def extract_images(image, refs, out_dir, prefix="fig", labels=("image", "figure")):
    """Crop every region the model tagged as a figure/image and save each as a file.
    Returns the list of saved paths. Coords are 0-999 normalized (same as draw_boxes)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    W, H = image.size
    saved = []
    i = 0
    for label, box in refs:
        if label.lower() not in labels:
            continue
        try:
            coords = eval(box)
        except Exception:
            continue
        if coords and isinstance(coords[0], (int, float)):
            coords = [coords]
        for x1, y1, x2, y2 in coords:
            x1 = int(x1 / 999 * W); y1 = int(y1 / 999 * H)
            x2 = int(x2 / 999 * W); y2 = int(y2 / 999 * H)
            if x2 <= x1 or y2 <= y1:
                continue
            p = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
            image.convert("RGB").crop((x1, y1, x2, y2)).save(p)
            saved.append(p); i += 1
    return saved


def to_markdown(text):
    clean = re.sub(r"<\|/?(ref|det|grounding)\|>", "", text)
    clean = re.sub(r"[A-Za-z_]+\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\]", "", clean)
    return clean.strip()
