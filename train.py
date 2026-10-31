import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.data import VLMJsonlDataset, collate_vlm_batch
from src.model import MiniVLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", default=None, help="Optional validation JSONL for evaluating validation loss")
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--language-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Warmup ratio for cosine LR scheduler")
    parser.add_argument("--resample-tokens", type=int, default=None, help="Downsample 49 tokens to 16, 25, 36, etc.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit QLoRA quantization")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--output", default=None)
    parser.add_argument("--init-projector", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    vision_name = args.vision_model or cfg["model"]["vision_model"]
    language_name = args.language_model or cfg["model"]["language_model"]
    epochs = args.epochs or cfg["training"]["epochs_" + args.stage]
    batch_size = args.batch_size or cfg["training"].get("batch_size", 1)
    resample_tokens = args.resample_tokens or cfg.get("model", {}).get("resample_tokens")
    output = args.output or (
        cfg["paths"]["stage1_checkpoint"] if args.stage == "stage1"
        else cfg["paths"]["stage2_checkpoint"]
    )

    model = MiniVLM(
        vision_name, language_name,
        use_lora=(args.stage == "stage2"),
        lora_r=cfg["training"]["lora_r"],
        lora_alpha=cfg["training"]["lora_alpha"],
        lora_dropout=cfg["training"]["lora_dropout"],
        resample_tokens=resample_tokens,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    if args.stage == "stage2":
        projector_path = args.init_projector or cfg["paths"]["stage1_checkpoint"]
        if not Path(projector_path).exists():
            raise FileNotFoundError(
                f"Stage-1 projector not found: {projector_path}. Train stage1 first."
            )
        model.load_projector(projector_path)

    dataset = VLMJsonlDataset(args.train_jsonl, model.processor, args.stage, tokenizer=model.tokenizer)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_vlm_batch
    )

    val_loader = None
    if args.val_jsonl:
        val_dataset = VLMJsonlDataset(args.val_jsonl, model.processor, args.stage, tokenizer=model.tokenizer)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_vlm_batch
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["training"]["learning_rate_" + args.stage],
        weight_decay=0.01,
    )
    accumulation = cfg["training"]["gradient_accumulation_steps"]

    # Cosine Annealing Learning Rate Scheduler with Warmup
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else cfg.get("training", {}).get("warmup_ratio", 0.05)
    total_steps = (len(loader) // accumulation + 1) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler("cuda") if device_type == "cuda" else None

    # CSV Logging Setup
    log_dir = Path("outputs")
    log_dir.mkdir(parents=True, exist_ok=True)
    history_file = log_dir / f"{args.stage}_history.csv"
    if not history_file.exists():
        with history_file.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

    model.train()
    optimizer.zero_grad(set_to_none=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        running = 0.0
        progress = tqdm(loader, desc=f"{args.stage} epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.bfloat16):
                loss = model(
                    prompt=batch["prompt"],
                    answer=batch["answer"],
                    pixel_values=batch.get("pixel_values"),
                    image_features=batch.get("image_features"),
                    max_answer_tokens=cfg["training"]["max_answer_tokens"],
                )

            if scaler is not None:
                scaler.scale(loss / accumulation).backward()
            else:
                (loss / accumulation).backward()
            running += loss.item()

            if step % accumulation == 0 or step == len(loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        train_mean = running / max(len(loader), 1)
        print(f"Epoch {epoch+1}: mean loss = {train_mean:.4f}")

        val_mean = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    with torch.amp.autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.bfloat16):
                        v_loss = model(
                            prompt=val_batch["prompt"],
                            answer=val_batch["answer"],
                            pixel_values=val_batch.get("pixel_values"),
                            image_features=val_batch.get("image_features"),
                            max_answer_tokens=cfg["training"]["max_answer_tokens"],
                        )
                    val_loss += v_loss.item()
            val_mean = val_loss / max(len(val_loader), 1)
            print(f"Epoch {epoch+1}: validation loss = {val_mean:.4f}")

            if val_mean < best_val_loss:
                best_val_loss = val_mean
                best_output = (
                    Path(output).with_name(f"{Path(output).stem}_best.pt")
                    if args.stage == "stage1"
                    else Path(output).with_name(f"{Path(output).name}_best")
                )
                if args.stage == "stage1":
                    model.save_projector(best_output)
                else:
                    model.save_stage2(best_output)
                print(f"-> Saved new best checkpoint (val loss {val_mean:.4f}) to: {best_output}")
            model.train()

        # Log epoch summary to CSV
        with history_file.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch + 1,
                round(train_mean, 5),
                round(val_mean, 5) if val_mean is not None else "",
                f"{current_lr:.2e}",
            ])

    if args.stage == "stage1":
        model.save_projector(output)
    else:
        model.save_stage2(output)
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
