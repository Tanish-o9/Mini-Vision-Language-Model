"""
Convert a simple CSV into Stage-2 JSONL.
CSV columns required: image, question, answer
Image paths can be absolute or relative to the CSV's folder.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="data/vqa/train.jsonl")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as src, output.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        required = {"image", "question", "answer"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            image = Path(row["image"])
            if not image.is_absolute():
                image = (csv_path.parent / image).resolve()
            item = {"image": str(image), "question": row["question"], "answer": row["answer"]}
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"Created {count} VQA records: {output}")


if __name__ == "__main__":
    main()
