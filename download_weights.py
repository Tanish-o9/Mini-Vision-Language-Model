"""
Helper script to download or initialize starter checkpoints for Mini-VLM.
Enables instant testing of the React Studio dashboard and Gradio app.
"""
import argparse
from pathlib import Path

import torch
import yaml

from src.model import MiniVLM


def initialize_demo_checkpoints(force=False):
    stage1_dir = Path("checkpoints/stage1")
    stage2_dir = Path("checkpoints/stage2")

    if (stage1_dir / "projector.pt").exists() and (stage2_dir / "projector.pt").exists() and not force:
        print("[INFO] Checkpoints already exist under 'checkpoints/'. Use --force to reinitialize.")
        return

    print("=" * 60)
    print("Initializing Starter Checkpoints for Mini-VLM Studio...")
    print("=" * 60)

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    model = MiniVLM(
        vision_name=cfg["model"]["vision_model"],
        language_name=cfg["model"]["language_model"],
        use_lora=True,
    )

    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Save Stage 1 Projector
    stage1_path = stage1_dir / "projector.pt"
    model.save_projector(stage1_path)
    print(f"[OK] Saved starter Stage-1 Projector: {stage1_path}")

    # Save Stage 2 Checkpoint (Projector + LoRA + Tokenizer)
    model.save_stage2(stage2_dir)
    print(f"[OK] Saved starter Stage-2 Multimodal Suite: {stage2_dir}")

    print("=" * 60)
    print("Demo checkpoints ready! You can now run:")
    print("  python server.py   # React AI Studio Dashboard (localhost:8000)")
    print("  python app.py      # Gradio Chatbot (localhost:7860)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or initialize Mini-VLM starter weights.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing checkpoints")
    args = parser.parse_args()
    initialize_demo_checkpoints(force=args.force)

