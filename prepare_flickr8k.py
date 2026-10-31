"""
Convert the standard Flickr8k dataset into the Stage-1 JSONL format.

Expected files:
  Flickr8k/
    Images/                 (all jpg images)
    captions.txt            OR Flickr8k.token.txt

The caption file format may be:
  image.jpg#0<TAB>A caption
or:
  image,caption
"""
import argparse
import csv
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--captions-file", required=True)
    parser.add_argument("--output-dir", default="data/flickr8k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    captions_file = Path(args.captions_file)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with captions_file.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        f.seek(0)
        if "\t" in first:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                filename = parts[0].split("#")[0]
                caption = parts[1].strip()
                if caption and (images_dir / filename).exists():
                    records.append({"image": str(images_dir / filename), "caption": caption})
        else:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("image") or row.get("image_name")
                caption = row.get("caption") or row.get("comment")
                if filename and caption and (images_dir / filename).exists():
                    records.append({"image": str(images_dir / filename), "caption": caption})

    if not records:
        raise RuntimeError("No valid image-caption rows found. Check paths and caption file format.")

    random.Random(args.seed).shuffle(records)
    split = int(len(records) * 0.9)
    train_rows, val_rows = records[:split], records[split:]

    for name, rows in [("train.jsonl", train_rows), ("val.jsonl", val_rows)]:
        with (output_dir / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created {len(train_rows)} train and {len(val_rows)} validation records in {output_dir}")


if __name__ == "__main__":
    main()
