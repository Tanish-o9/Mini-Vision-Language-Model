import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class VLMJsonlDataset(Dataset):
    """
    Each JSONL row:
      Stage 1: {"image": "images/abc.jpg", "caption": "A dog runs."}
      Stage 2: {"image": "images/abc.jpg", "question": "What is ...?", "answer": "..."}

    Optional cached feature row:
      {"image": "...", "feature": "features/abc.pt", ...}
    Feature tensor must have shape [49, 768].
    """

    def __init__(self, jsonl_path, processor, stage, tokenizer=None):
        self.jsonl_path = Path(jsonl_path).resolve()
        self.base_dir = self.jsonl_path.parent
        self.processor = processor
        self.stage = stage
        self.tokenizer = tokenizer
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if not self.rows:
            raise ValueError(f"No records found in {jsonl_path}")

    def __len__(self):
        return len(self.rows)

    def _resolve(self, value):
        p = Path(value)
        return p if p.is_absolute() else self.base_dir / p

    def __getitem__(self, index):
        row = self.rows[index]
        result = {}

        if row.get("feature"):
            feature_path = self._resolve(row["feature"])
            result["image_features"] = torch.load(feature_path, map_location="cpu", weights_only=True)
        else:
            image_path = self._resolve(row["image"])
            with Image.open(image_path) as im:
                image = im.convert("RGB")
                result["pixel_values"] = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        if self.stage == "stage1":
            user_text = "Describe this image in one clear sentence."
            answer = row["caption"]
        else:
            user_text = row["question"]
            answer = row["answer"]

        if self.tokenizer and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_text}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"USER: {user_text}\nASSISTANT:"

        return {"prompt": prompt, "answer": str(answer), **result}


def collate_vlm_batch(batch):
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]
    collated = {"prompt": prompts, "answer": answers}

    if "pixel_values" in batch[0]:
        collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    elif "image_features" in batch[0]:
        collated["image_features"] = torch.stack([item["image_features"] for item in batch])

    return collated
