#!/usr/bin/env bash
# Download baidu/Unlimited-OCR and convert it to MLX (bf16 + optional 4-bit).
# Weights are NOT shipped in this repo — this script builds them locally.
set -euo pipefail

python3 -m pip install -q "mlx-vlm" "huggingface_hub" pymupdf pillow

echo "[1/3] downloading baidu/Unlimited-OCR ..."
hf download baidu/Unlimited-OCR --local-dir hf_model \
  --include "*.safetensors" "*.json" "tokenizer.json"

echo "[2/3] routing model_type -> deepseekocr (Unlimited-OCR == DeepSeek-OCR arch) ..."
python3 - <<'PY'
import json
p = "hf_model/config.json"; c = json.load(open(p))
c["model_type"] = "deepseekocr"; c.pop("auto_map", None)
json.dump(c, open(p, "w"), indent=2)
PY

echo "[3/3] converting to MLX bf16 ..."
python3 -m mlx_vlm convert --hf-path hf_model --mlx-path mlx_unlimited_bf16 --dtype bfloat16

if [[ "${1:-}" == "--4bit" ]]; then
  echo "      + 4-bit ..."
  python3 -m mlx_vlm convert --hf-path hf_model --mlx-path mlx_unlimited_4bit \
    --dtype bfloat16 -q --q-bits 4 --q-group-size 64
fi

# strip remote-code refs so loading needs no torch
python3 - <<'PY'
import json, glob, os
for d in ["mlx_unlimited_bf16", "mlx_unlimited_4bit"]:
    for f in ("config.json", "tokenizer_config.json", "processor_config.json"):
        p = os.path.join(d, f)
        if not os.path.exists(p): continue
        c = json.load(open(p)); c.pop("auto_map", None)
        if str(c.get("processor_class", "")).startswith("Unlimited"):
            c["processor_class"] = "DeepseekOCRProcessor"
        json.dump(c, open(p, "w"), indent=2, ensure_ascii=False)
    for junk in glob.glob(os.path.join(d, "modeling_*.py")) + \
                glob.glob(os.path.join(d, "configuration_*.py")) + \
                [os.path.join(d, "deepencoder.py"), os.path.join(d, "conversation.py")]:
        if os.path.exists(junk): os.remove(junk)
PY
echo "done -> mlx_unlimited_bf16/  (run: python ocr.py --image page.png --out-image out.png)"
