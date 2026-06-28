#!/usr/bin/env python3
"""OCR every page image in a folder -> per-page markdown + boxed PNG + combined .md.

Uses the fast mlx-vlm stream_generate path with clean byte-level detokenization
and the Unlimited-OCR box visualization.
"""
import argparse, glob, os, re, sys, time

from PIL import Image
from mlx_vlm import load
from mlx_vlm.generate import stream_generate

import udriver  # bpe_decode, parse_boxes, draw_boxes, to_markdown


def ocr_page(model, processor, image_path, prompt, max_tokens):
    tok = processor.tokenizer
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    ids = []
    ngram = udriver.make_ngram_logits_processor(ngram_size=35, window=128)
    for r in stream_generate(model, processor, prompt, image=[image_path],
                             max_tokens=max_tokens, temperature=0.0, verbose=False,
                             logits_processors=[ngram]):
        if r.token is not None:
            ids.append(int(r.token))
    text = udriver.bpe_decode("".join(tok.convert_ids_to_tokens(ids)))
    return text, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx_unlimited_bf16")
    ap.add_argument("--pages-glob", default="pdf_pages/page_*.png")
    ap.add_argument("--out-dir", default="ocr_out")
    ap.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown.")
    ap.add_argument("--free-ocr", action="store_true",
                    help="layout-free plain text (no boxes, fewer tokens, faster)")
    ap.add_argument("--max-tokens", type=int, default=6144)
    args = ap.parse_args()
    if args.free_ocr:
        args.prompt = "<image>\nFree OCR."

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "boxed"), exist_ok=True)
    pages = sorted(glob.glob(args.pages_glob))
    print(f"[loading model] {args.model}", flush=True)
    model, processor = load(args.model)

    combined = []
    for i, p in enumerate(pages, 1):
        t = time.time()
        text, n = ocr_page(model, processor, p, args.prompt, args.max_tokens)
        md = udriver.to_markdown(text)
        stem = os.path.splitext(os.path.basename(p))[0]
        refs = [] if args.free_ocr else udriver.parse_boxes(text)
        if not args.free_ocr:
            udriver.draw_boxes(Image.open(p), refs).save(
                os.path.join(args.out_dir, "boxed", f"{stem}_boxed.png"))
        with open(os.path.join(args.out_dir, f"{stem}.md"), "w") as f:
            f.write(md)
        combined.append(f"\n\n<!-- ===== {stem} ===== -->\n\n{md}")
        print(f"PAGE {i}/{len(pages)} {stem}: {n} tok, {len(refs)} boxes, {time.time()-t:.1f}s", flush=True)

    with open(os.path.join(args.out_dir, "full_document.md"), "w") as f:
        f.write("".join(combined))
    print(f"DONE -> {args.out_dir}/full_document.md  (+ per-page .md and boxed/)", flush=True)


if __name__ == "__main__":
    main()
