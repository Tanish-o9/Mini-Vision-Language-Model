"""
FastAPI Backend Server for Mini-VLM React Dashboard.
Provides REST APIs for:
- Multimodal conversational chat (/api/chat)
- Training metrics & convergence history (/api/metrics)
- System diagnostics & GPU memory (/api/status)
- Model configuration (/api/config)
- Static distribution serving for production React build
"""
import base64
import csv
import io
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from src.model import MiniVLM

app = FastAPI(title="Mini-VLM Studio API", version="1.0.0")

# Enable CORS for local frontend development (Vite @ localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_CACHE = {
    "model": None,
    "checkpoint": None,
    "stage": None,
}


@app.on_event("startup")
def preload_model():
    print("[INIT] Pre-warming Mini-VLM into GPU memory for instantaneous inference...")
    try:
        get_model()
        print("[OK] Mini-VLM warm and ready to serve chat requests instantly!")
    except Exception as e:
        print(f"[WARN] Startup warm-up encountered: {e}")


def get_model(checkpoint: str = "checkpoints/stage2", stage: str = "stage2"):
    global MODEL_CACHE
    if MODEL_CACHE["model"] is not None and MODEL_CACHE["checkpoint"] == checkpoint and MODEL_CACHE["stage"] == stage:
        return MODEL_CACHE["model"]

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    resample_tokens = cfg.get("model", {}).get("resample_tokens")
    model = MiniVLM(
        vision_name=cfg["model"]["vision_model"],
        language_name=cfg["model"]["language_model"],
        use_lora=False,
        lora_r=cfg["training"]["lora_r"],
        lora_alpha=cfg["training"]["lora_alpha"],
        lora_dropout=cfg["training"]["lora_dropout"],
        resample_tokens=resample_tokens,
    )

    ckpt_path = Path(checkpoint)
    if stage == "stage1":
        if ckpt_path.exists():
            model.load_projector(ckpt_path)
    else:
        proj = ckpt_path / "projector.pt"
        lora = ckpt_path / "lora"
        if proj.exists():
            model.load_projector(proj)
        if lora.exists():
            model.load_lora(lora)

    model.eval()
    MODEL_CACHE["model"] = model
    MODEL_CACHE["checkpoint"] = checkpoint
    MODEL_CACHE["stage"] = stage
    return model


class ChatRequest(BaseModel):
    image_base64: str
    secondary_image_base64: Optional[str] = None
    question: str
    history: Optional[List[List[str]]] = None
    max_new_tokens: Optional[int] = 80
    stage: Optional[str] = "stage2"
    checkpoint: Optional[str] = "checkpoints/stage2"


class ChatResponse(BaseModel):
    answer: str
    latency_seconds: float
    device: str


@app.get("/api/status")
def get_system_status():
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Only"
    vram_alloc = 0.0
    vram_res = 0.0
    if cuda_available:
        vram_alloc = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        vram_res = round(torch.cuda.memory_reserved() / (1024 ** 3), 2)

    return {
        "status": "online",
        "device": "cuda" if cuda_available else "cpu",
        "gpu_name": gpu_name,
        "vram_allocated_gb": vram_alloc,
        "vram_reserved_gb": vram_res,
        "model_loaded": MODEL_CACHE["model"] is not None,
        "current_checkpoint": MODEL_CACHE["checkpoint"],
        "current_stage": MODEL_CACHE["stage"],
        "torch_version": torch.__version__,
    }


@app.get("/api/config")
def get_config():
    if Path("config.yaml").exists():
        return yaml.safe_load(open("config.yaml", encoding="utf-8"))
    return {}


@app.get("/api/metrics")
def get_metrics():
    metrics = {"stage1": [], "stage2": []}
    for stage in ["stage1", "stage2"]:
        csv_file = Path("outputs") / f"{stage}_history.csv"
        if csv_file.exists():
            with csv_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics[stage].append({
                        "epoch": int(row.get("epoch", 0)),
                        "train_loss": float(row.get("train_loss", 0.0)) if row.get("train_loss") else None,
                        "val_loss": float(row.get("val_loss", 0.0)) if row.get("val_loss") else None,
                        "learning_rate": row.get("learning_rate", ""),
                    })
    return metrics


@app.post("/api/chat", response_model=ChatResponse)
def chat_predict(req: ChatRequest):
    import time
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="Image data is required.")
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        header, b64data = (
            req.image_base64.split(",", 1) if "," in req.image_base64 else ("", req.image_base64)
        )
        img_bytes = base64.b64decode(b64data)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        secondary_image = None
        if req.secondary_image_base64:
            h2, b64_2 = (
                req.secondary_image_base64.split(",", 1) if "," in req.secondary_image_base64 else ("", req.secondary_image_base64)
            )
            img_bytes_2 = base64.b64decode(b64_2)
            secondary_image = Image.open(io.BytesIO(img_bytes_2)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    try:
        model = get_model(checkpoint=req.checkpoint or "checkpoints/stage2", stage=req.stage or "stage2")
        start = time.perf_counter()
        answer = model.answer(
            image=image,
            question=req.question,
            history=req.history,
            max_new_tokens=req.max_new_tokens or 80,
            secondary_image=secondary_image,
        )
        latency = time.perf_counter() - start
        return ChatResponse(answer=answer, latency_seconds=round(latency, 3), device=model.device_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Serve compiled React frontend if exists
dist_dir = Path("frontend/dist")
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Mini-VLM Studio Server at http://localhost:8000")
    print("React AI Studio Dashboard: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

