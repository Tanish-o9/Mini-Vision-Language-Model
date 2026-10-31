"""
Optional: precompute frozen CLIP patch features to avoid running CLIP each epoch.
Input JSONL rows must contain an 'image' path.
Output rows contain a relative 'feature' path plus original fields.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    clip = CLIPModel.from_pretrained(args.model).to(device=device, dtype=dtype).eval()
    for p in clip.parameters():
        p.requires_grad = False

    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    features_dir = Path(args.features_dir).resolve()
    features_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with output_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(tqdm(rows, desc="Caching CLIP features")):
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = input_path.parent / image_path
            with Image.open(image_path) as im:
                pixels = processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"].to(device, dtype)
            with torch.no_grad():
                # Penultimate layer (hidden_states[-2]) preserves richer spatial details (LLaVA-style)
                features = clip.vision_model(pixel_values=pixels, output_hidden_states=True).hidden_states[-2][:, 1:, :][0]
            feature_file = features_dir / f"{input_path.stem}_{i:07d}.pt"
            torch.save(features.cpu().float(), feature_file)
            new_row = dict(row)
            new_row["feature"] = str(feature_file)
            out.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    print(f"Cached {len(rows)} feature files. New manifest: {output_path}")


if __name__ == "__main__":
    main()
