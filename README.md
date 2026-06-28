# Unlimited-OCR → MLX

An Apple-Silicon (MLX) port of [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR),
a 3B vision-language OCR model for one-shot long-horizon document parsing.

## What this is

Unlimited-OCR is, by the authors' own description, **DeepSeek-OCR "pushed one step
further."** Its architecture is identical to DeepSeek-OCR:

```
image ─► SAM ViT-B (windowed)  ┐
                               ├─► concat ─► linear projector (2048→1280) ─► tokens ─┐
image ─► CLIP-L/14 ────────────┘                                                      │
                                                                                      ▼
                       DeepSeek-V2 MoE LM  (12 layers · 64 routed experts, 6 active · 2 shared)
```

Because the architecture matches, this port **reuses `mlx-vlm`'s built-in
`deepseekocr` model** for the network itself, and adds the Unlimited-OCR–specific
*long-horizon decoding* on top:

| Unlimited-OCR feature | Where it lives here |
|---|---|
| SAM+CLIP encoder, projector, DeepSeek-V2 MoE LM | `mlx-vlm` `deepseekocr` (weights converted to MLX) |
| Sliding-window ring KV cache (full-attention prefill + 128-tok decode window → bounded memory, unbounded output) | `udriver.PrefillRingCache` |
| Sliding-window n-gram no-repeat (size 35, window 128/1024) | `udriver.apply_no_repeat_ngram` |
| Gundam (1024/640 + crop tiling) & Base (1024) modes | `udriver._build_inputs` |
| Multi-page / PDF parsing | `udriver.infer_multi`, `batch_pages.py` |
| Grounding-box parsing + visualization | `udriver.parse_boxes` / `draw_boxes` |
| Clean byte-level BPE detokenization (no `Ġ`/`Ċ` artifacts) | `udriver.bpe_decode` |

The converted model declares `model_type: "deepseekocr"`, so it loads on a stock
`pip install mlx-vlm` with **no patching required**.

> **Weights are not in this repo** (they're 8.5 GB). `setup_model.sh` downloads
> `baidu/Unlimited-OCR` from Hugging Face and converts it to MLX locally.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install mlx-vlm pymupdf pillow
./setup_model.sh --4bit          # download + convert (bf16 + 4-bit); omit --4bit for bf16 only
python ocr.py --image page.png --out-md page.md --out-image page_boxed.png
```

## Setup (manual)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install mlx-vlm pymupdf pillow
```

The MLX model lives in `mlx_unlimited_bf16/` (bf16, ~6.2 GB; ~8 GB peak RAM).
`mlx_unlimited_4bit/` is a ~2.3 GB quantized variant (~3.8 GB peak RAM) for 8–16 GB Macs.

### Which to use (measured on M-series, 24 GB)

| | bf16 | 4-bit |
|---|---|---|
| Disk / peak RAM | 6.2 GB / 8.0 GB | 2.3 GB / **3.8 GB** |
| Decode speed | ~12 tok/s (short) · ~45–70 (long) | ~same |
| OCR fidelity | best | slightly worse on hard regions |

**Quantization here saves memory, not time.** Throughput is bounded by the
high-precision vision encoder (SAM+CLIP), the per-page prefill, and the
129,280-vocab `lm_head` — none of which 4-bit shrinks much (the MoE has few
active params). On a 16 GB+ Mac prefer **bf16** for accuracy; pick 4-bit only when
RAM-constrained. Short outputs look slow because the fixed per-page vision-encode
cost isn't amortized; long pages reach 45–70 tok/s.

## Convert from the original (reproduce)

```bash
huggingface-cli download baidu/Unlimited-OCR --local-dir hf_model
python -c "import json;p='hf_model/config.json';c=json.load(open(p));c['model_type']='deepseekocr';c.pop('auto_map',None);json.dump(c,open(p,'w'),indent=2)"
python -m mlx_vlm convert --hf-path hf_model --mlx-path mlx_unlimited_bf16 --dtype bfloat16
# 4-bit:
python -m mlx_vlm convert --hf-path hf_model --mlx-path mlx_unlimited_4bit --dtype bfloat16 -q --q-bits 4
```

## Usage

### Single image (faithful long-horizon path: ring cache + n-gram)
```python
from mlx_vlm import load
import udriver
model, processor = load("mlx_unlimited_bf16")
text = udriver.infer(model, processor, "page.png", mode="gundam", max_tokens=8192)
print(udriver.to_markdown(text))
```

### OCR + bounding-box image
```bash
python ocr.py --image page.png --out-md page.md --out-image page_boxed.png
```

### Whole PDF (render → OCR every page → per-page md + boxed png + combined md)
```bash
python -c "import fitz,os;d=fitz.open('doc.pdf');[d[i].get_pixmap(matrix=fitz.Matrix(200/72,200/72)).save(f'pdf_pages/page_{i+1:02d}.png') for i in range(d.page_count)]"
python batch_pages.py --out-dir ocr_out
```

### Prompts
- `<|grounding|>Convert the document to markdown.` — layout-aware markdown + boxes
- `Free OCR.` — plain text, no layout
- `Locate <|ref|>some text<|/ref|> in the image.` — text grounding → `[[x1,y1,x2,y2]]` (0–999 normalized)

## Files
- `mlx_unlimited_bf16/`, `mlx_unlimited_4bit/` — converted MLX models
- `udriver.py` — faithful Unlimited-OCR decode (ring cache, n-gram, multi-page, boxes)
- `ocr.py` — fast single-image OCR + box image (mlx-vlm `stream_generate`)
- `batch_pages.py` — whole-folder / whole-PDF batch driver

## Notes
- `udriver.infer` reproduces the reference's bounded-memory decode (eager loop, ~12 tok/s).
  `ocr.py` / `batch_pages.py` use mlx-vlm's optimized `stream_generate` (~70 tok/s); for
  per-page outputs (well under the window+prefill budget) the two are equivalent in quality.
- Original model: MIT-licensed. Architecture credit: DeepSeek-OCR, DeepSeek-OCR-2, PaddleOCR.
