"""
Self-contained smoke test for Mini Vision-Language Model.
Verifies end-to-end dataset loading, model forward, loss backward,
multi-turn inference, and checkpoint save/load.
"""
import json
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image

from src.data import VLMJsonlDataset, collate_vlm_batch
from src.model import MiniVLM


def run_smoke_test():
    print("=" * 60)
    print("Running Mini-VLM End-to-End Smoke Test")
    print("=" * 60)

    temp_dir = Path(tempfile.mkdtemp(prefix="vlm_smoke_"))
    try:
        # 1. Create a dummy test image
        img_path = temp_dir / "sample.jpg"
        dummy_img = Image.new("RGB", (224, 224), color=(73, 109, 137))
        dummy_img.save(img_path)
        print("[OK] Created synthetic test image.")

        # 2. Create sample JSONL files
        stage1_jsonl = temp_dir / "stage1.jsonl"
        stage2_jsonl = temp_dir / "stage2.jsonl"

        with stage1_jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"image": str(img_path), "caption": "A blue solid background."}) + "\n")
            f.write(json.dumps({"image": str(img_path), "caption": "A solid color square."}) + "\n")

        with stage2_jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"image": str(img_path), "question": "What color is this?", "answer": "Blue."}) + "\n")
            f.write(json.dumps({"image": str(img_path), "question": "Is it red?", "answer": "No, it is blue."}) + "\n")

        print("[OK] Created sample Stage 1 and Stage 2 JSONL datasets.")

        # 3. Initialize MiniVLM
        vision_name = "openai/clip-vit-base-patch32"
        language_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"[OK] Initializing MiniVLM with vision='{vision_name}', language='{language_name}'...")
        model = MiniVLM(vision_name, language_name, use_lora=True)

        # 4. Test dataset & dynamic batching collator
        dataset = VLMJsonlDataset(stage1_jsonl, model.processor, stage="stage1", tokenizer=model.tokenizer)
        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        batch_items = [dataset[0], dataset[1]]
        batch = collate_vlm_batch(batch_items)
        assert len(batch["prompt"]) == 2
        assert "pixel_values" in batch
        print("[OK] Verified VLMJsonlDataset and collate_vlm_batch (batch_size=2).")

        # 5. Test model forward & loss computation
        loss = model(
            prompt=batch["prompt"],
            answer=batch["answer"],
            pixel_values=batch.get("pixel_values"),
        )
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        print(f"[OK] Verified forward pass with ChatML prompt (Loss = {loss.item():.4f}).")

        # 6. Test backward pass & optimizer step
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        print("[OK] Verified backward pass and gradient step.")

        # 7. Test multi-turn conversational inference
        history = [("Is this image bright?", "Yes, it is moderately bright.")]
        answer = model.answer(dummy_img, "What is the primary color?", history=history, max_new_tokens=30)
        print(f"[OK] Verified multi-turn inference (Model Answer: '{answer}').")

        # 8. Test Checkpoint saving and reloading
        ckpt_dir = temp_dir / "checkpoint_test"
        model.save_stage2(ckpt_dir)
        assert (ckpt_dir / "projector.pt").exists(), "Projector file missing in checkpoint."
        assert (ckpt_dir / "lora").exists(), "LoRA directory missing in checkpoint."

        reload_model = MiniVLM(vision_name, language_name, use_lora=False)
        reload_model.load_projector(ckpt_dir / "projector.pt")
        reload_model.load_lora(ckpt_dir / "lora")
        print("[OK] Verified saving and reloading Stage 2 checkpoint.")

        print("=" * 60)
        print("ALL SMOKE TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_smoke_test()

