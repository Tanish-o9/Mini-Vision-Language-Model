import argparse
import collections
import csv
import json
import re
import time
from pathlib import Path

import yaml
from PIL import Image

from src.model import MiniVLM


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_token_f1(prediction, target):
    pred_tokens = normalize(prediction).split()
    target_tokens = normalize(target).split()
    if not pred_tokens or not target_tokens:
        return 1.0 if pred_tokens == target_tokens else 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    return (2 * precision * recall) / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Held-out VQA JSONL")
    parser.add_argument("--checkpoint", default="checkpoints/stage2")
    parser.add_argument("--output", default="outputs/evaluation.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    model = MiniVLM(
        cfg["model"]["vision_model"],
        cfg["model"]["language_model"],
        use_lora=False,
        resample_tokens=cfg.get("model", {}).get("resample_tokens"),
    )
    model.load_projector(Path(args.checkpoint) / "projector.pt")
    model.load_lora(Path(args.checkpoint) / "lora")
    model.eval()

    manifest = Path(args.jsonl).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct, total, total_f1, elapsed = 0, 0, 0.0, 0.0
    results = []
    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            start = time.perf_counter()
            prediction = model.answer(image, row["question"], cfg["model"]["max_new_tokens"])
            elapsed += time.perf_counter() - start

        target = str(row["answer"])
        match = normalize(prediction) == normalize(target)
        f1 = compute_token_f1(prediction, target)
        correct += int(match)
        total_f1 += f1
        total += 1
        results.append({
            "question": row["question"],
            "target": target,
            "prediction": prediction,
            "normalized_exact_match": match,
            "token_f1": round(f1, 4),
        })

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [
            "question", "target", "prediction", "normalized_exact_match", "token_f1"
        ])
        writer.writeheader()
        writer.writerows(results)

    accuracy = 100 * correct / total if total else 0.0
    mean_f1 = 100 * total_f1 / total if total else 0.0
    avg_latency = elapsed / total if total else 0.0
    print(f"Samples: {total}")
    print(f"Normalized exact-match accuracy: {accuracy:.2f}%")
    print(f"Mean Token-F1 score: {mean_f1:.2f}%")
    print(f"Average end-to-end latency: {avg_latency:.3f} seconds/sample")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
