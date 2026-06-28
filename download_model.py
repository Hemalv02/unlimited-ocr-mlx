import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download
p = snapshot_download(
    "baidu/Unlimited-OCR",
    local_dir="hf_model",
    allow_patterns=["*.safetensors", "*.json", "*.py", "tokenizer.json", "*.txt"],
)
print("DOWNLOADED TO", p)
