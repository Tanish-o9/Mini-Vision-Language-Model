<div align="center">

# 👁️ Mini Vision-Language Model (LLaVA-Style)

### A Modular, Lightweight, and Production-Grade Multimodal Learning System
Built with **Frozen CLIP ViT-B/32** + **Trainable MLP Projector** + **Qwen2.5-0.5B-Instruct (+ LoRA)**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![CI/CD Smoke Test](https://img.shields.io/github/actions/workflow/status/Tanish-o9/Mini-Vision-Language-Model/smoke-test.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI%2FCD%20Tests)](https://github.com/Tanish-o9/Mini-Vision-Language-Model/actions)
[![React Dashboard](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](frontend/)
[![Open In Colab](https://img.shields.io/badge/Google%20Colab-Open%20Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Tanish-o9/Mini-Vision-Language-Model/blob/main/notebooks/Quickstart_Colab.ipynb)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](docker-compose.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.45%2B-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![GitHub Profile](https://img.shields.io/badge/Author-Tanish--o9-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Tanish-o9)

<br/>

[Overview](#-overview) •
[Architecture](#-architecture--dataflow) •
[Two-Stage Pipeline](#-two-stage-training-paradigm) •
[Quickstart](#-quickstart--smoke-test) •
[Step-by-Step Guide](#-complete-step-by-step-pipeline) •
[Evaluation](#-evaluation--benchmarks) •
[License](#-license)

</div>

---

## 📌 Overview

**Mini Vision-Language Model** is an educational, resource-efficient implementation of modern Vision-Language Models (VLMs) following the **LLaVA (Large Language and Vision Assistant)** design paradigm.

Unlike giant multimodal models requiring enterprise clusters, this project pairs **CLIP ViT-B/32** with Alibaba Cloud's **Qwen2.5-0.5B-Instruct**, enabling end-to-end multimodal alignment, instruction fine-tuning, and multi-turn visual dialogue on **consumer GPUs (even under 4 GB VRAM)**.

### 🌟 Key Highlights & Enhancements
- 🎯 **Penultimate Layer Features**: Extracts representations from CLIP's penultimate layer (`hidden_states[-2]`), preserving localized spatial details rather than text-collapsed final features.
- 💬 **Native Qwen ChatML Formatting**: Built-in `tokenizer.apply_chat_template()` support ensures generation stays aligned with Qwen's instruction priors.
- ⚡ **PyTorch Native AMP & GradScaler**: Hardware-accelerated FP16 training with numerical underflow/overflow prevention.
- 📈 **Cosine LR Scheduler with Warmup**: Smooth gradient transitions preventing divergence during early projector alignment.
- 📦 **Dynamic Batch Collation**: Custom `collate_vlm_batch` enables true variable-length batched training (`batch_size >= 1`).
- 🤖 **Multi-Turn Chatbot UI**: Interactive Gradio web interface supporting multi-turn conversational dialogue with stateful image context.
- 📉 **4-bit / 8-bit QLoRA Option**: Integrated `bitsandbytes` support for training and inference on low-memory setups.
- 📊 **Dual-Metric Evaluation**: Evaluates models on both **Normalized Exact Match** and **Token-Level F1 Score**.

---

## 🏗️ Architecture & Dataflow

```
 [ Input Image (224x224 RGB) ]
               │
               ▼
     [ Frozen CLIP ViT-B/32 ]
               │ (Extract Penultimate Layer [-2])
               ▼
    [ 49 Patch Tokens × 768-dim ]
               │
               ▼ (Optional: AdaptiveAvgPool2d Spatial Resampler)
    [ Spatial Tokens × 768-dim ]
               │
               ▼
     [ Trainable 2-Layer MLP ] ── Linear(768, 896) ➔ GELU ➔ Linear(896, 896)
               │
               ▼
    [ Visual Tokens (N × 896-dim) ]
               │
               ├──────────────────────────────────────────────┐
               │                                              │
               ▼                                              ▼
     [ Concatenated Input Embeddings ] <── [ Qwen Token Embeddings (ChatML Prompt + Answer) ]
               │
               ▼
     [ Qwen2.5-0.5B-Instruct ]  <── Frozen base weights, trainable LoRA on q_proj & v_proj (Stage 2)
               │
               ▼
     [ Autoregressive Response ]
```

---

## 🔄 Two-Stage Training Paradigm

Modern VLMs use a disciplined two-stage sequence to bridge visual perception and language understanding:

| Dimension | Stage 1: Feature Alignment | Stage 2: Visual Instruction Tuning (VQA) |
| :--- | :--- | :--- |
| **Objective** | Teach the MLP projector to translate CLIP visual patches into LLM soft-tokens. | Teach the model to follow complex user instructions and answer questions. |
| **Dataset Format** | Image-caption pairs (e.g. Flickr8k) | VQA triplets (`image, question, answer`) |
| **Trainable Modules** | **Projector only** (CLIP & Qwen base frozen) | **Projector + Qwen LoRA adapters** (`q_proj`, `v_proj`) |
| **Loss Supervision** | Cross-entropy strictly on **caption tokens** (prefix masked with `-100`) | Cross-entropy strictly on **answer tokens** (prefix masked with `-100`) |
| **Saved Artifacts** | `checkpoints/stage1/projector.pt` | `checkpoints/stage2/` (`projector.pt`, `lora/`, `tokenizer/`) |

---

## ⚡ Quickstart & Smoke Test

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Tanish-o9/Mini-Vision-Language-Model.git
cd Mini-Vision-Language-Model

# Create & activate environment
python -m venv .venv

# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Verified Smoke Test
Verify that your PyTorch installation, CUDA drivers, Hugging Face downloads, dynamic collation, forward/backward passes, and multi-turn generation work end-to-end:

```bash
python smoke_test.py
```

<details>
<summary><b>🔍 Click to view real verified test output</b></summary>

```text
============================================================
Running Mini-VLM End-to-End Smoke Test
============================================================
[OK] Created synthetic test image.
[OK] Created sample Stage 1 and Stage 2 JSONL datasets.
[OK] Initializing MiniVLM with vision='openai/clip-vit-base-patch32', language='Qwen/Qwen2.5-0.5B-Instruct'...
Loading weights: 100%|##########| 398/398 [00:00<00:00, 37957.50it/s]
Loading weights: 100%|##########| 290/290 [00:01<00:00, 171.32it/s]
[OK] Verified VLMJsonlDataset and collate_vlm_batch (batch_size=2).
[OK] Verified forward pass with ChatML prompt (Loss = 11.2371).
[OK] Verified backward pass and gradient step.
[OK] Verified multi-turn inference.
[OK] Verified saving and reloading Stage 2 checkpoint.
============================================================
ALL SMOKE TESTS PASSED SUCCESSFULLY!
============================================================
```
</details>

---

## 🚀 Complete Step-by-Step Pipeline

### Step 1: Prepare Stage-1 Captioning Data
Convert the standard Flickr8k dataset into train/validation JSONL splits:
```bash
python prepare_flickr8k.py --images-dir "D:/datasets/Flickr8k/Images" --captions-file "D:/datasets/Flickr8k/Flickr8k.token.txt"
```
*Outputs generated in:* `data/flickr8k/train.jsonl` and `data/flickr8k/val.jsonl`.

### Step 2: (Optional) Precompute & Cache CLIP Features
Precomputes penultimate layer features (`.pt` files) to accelerate training by $3\times$ to $5\times$:
```bash
python cache_features.py --input-jsonl data/flickr8k/train.jsonl --output-jsonl data/flickr8k/train_cached.jsonl
```

### Step 3: Train Stage 1 (Projector Alignment)
Train the projector with PyTorch Native AMP, Cosine Warmup, and Best Checkpointing:
```bash
python train.py --stage stage1 --train-jsonl data/flickr8k/train_cached.jsonl --val-jsonl data/flickr8k/val.jsonl --epochs 2
```
*Saves:* `checkpoints/stage1/projector.pt` (and `projector_best.pt`).  
*Epoch Logs:* `outputs/stage1_history.csv`.

### Step 4: Prepare Stage-2 VQA Data
Convert any 3-column CSV (`image,question,answer`) into Stage-2 JSONL format:
```bash
python prepare_vqa_jsonl.py --csv data/vqa_train.csv --output data/vqa/train.jsonl
```

### Step 5: Train Stage 2 (LoRA Visual Instruction Tuning)
Loads the Stage-1 projector and fine-tunes the projector + Qwen LoRA adapters on VQA pairs:
```bash
python train.py --stage stage2 --train-jsonl data/vqa/train.jsonl --val-jsonl data/vqa/val.jsonl --epochs 2
```

> **💡 Low VRAM Tip (4-bit QLoRA):**  
> If running on a GPU with $\le 6\text{ GB}$ VRAM, simply add `--load-in-4bit`:
> ```bash
> python train.py --stage stage2 --train-jsonl data/vqa/train.jsonl --load-in-4bit
> ```

### Step 6: Evaluate on Held-Out Test Set
Evaluate performance using both Normalized Exact Match and Token-Level F1 Score:
```bash
python evaluate.py --jsonl data/vqa/test.jsonl --checkpoint checkpoints/stage2
```
*Outputs:* Prints normalized exact-match accuracy, mean token-F1 score, and latency; exports per-sample predictions to `outputs/evaluation.csv`.

### Step 7: Launch the React AI Studio Dashboard 🖥️
Run the FastAPI backend server to serve the full compiled **React AI Studio Dashboard** (NOT Streamlit — full React 19 + Vite experience!):
```bash
python server.py
```
Open **`http://localhost:8000`** in your browser to explore:
- 💬 **Multimodal Playground**: Drag-and-drop images with full multi-turn conversational chat bubbles.
- 📊 **Training Analytics**: Loss & convergence curves across Stage 1 and Stage 2 with interactive SVG charts.
- 🧠 **Architecture Inspector**: Live visualization of tensor flow from $224 \times 224$ to CLIP, Projector, and Qwen embeddings.
- ⚙️ **Diagnostics**: Real-time CUDA / VRAM memory allocation stats and active checkpoint details.

*(Optional: For live hot-reloading frontend development)*:
```bash
cd frontend
npm run dev   # Runs on http://localhost:5173
```

*(Alternative: Quick Gradio Demo)*:
```bash
python app.py --stage stage2 --checkpoint checkpoints/stage2
```

### Step 8: Run in Google Colab 🚀
Don't have a local GPU? Train and experiment on Google Colab's free T4 GPU in 1 click:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tanish-o9/Mini-Vision-Language-Model/blob/main/notebooks/Quickstart_Colab.ipynb)

### Step 9: Containerized Deployment (Docker) 🐳
Launch the backend API and compiled React Studio Dashboard in an isolated container:
```bash
# Build & start container
docker-compose up --build

# Open in browser: http://localhost:8000
```

---

## 📁 Repository Structure

```
Mini-Vision-Language-Model/
├── .github/workflows/       # GitHub Actions automated CI/CD pipeline
│   └── smoke-test.yml       # Automated test suite on push/PR
├── notebooks/               # 1-Click Interactive Jupyter/Colab notebooks
│   └── Quickstart_Colab.ipynb
├── frontend/                # Full React 19 + Vite AI Studio Dashboard
│   ├── src/                 # React components (Playground, Dual Comparison, Analytics)
│   ├── dist/                # Production-ready compiled web bundle
│   └── package.json         # Node.js dependencies
├── Dockerfile               # Multi-stage production container image
├── docker-compose.yml       # One-command container orchestration
├── server.py                # FastAPI backend server with REST endpoints & static server
├── download_weights.py      # Quickstart checkpoint downloader & demo initializer
├── config.yaml              # Centralized hyperparameters & paths
├── requirements.txt         # Pinned Python package dependencies
├── LICENSE                  # MIT Open Source License
├── smoke_test.py            # Automated end-to-end self-test script
├── train.py                 # Stage 1 & Stage 2 training engine with AMP & LR scheduler
├── evaluate.py              # Test evaluation (Exact Match + Token F1 score + Latency)
├── app.py                   # Multi-turn Gradio Chatbot Web UI
├── cache_features.py        # Precomputes CLIP penultimate patch features to .pt
├── prepare_flickr8k.py      # Flickr8k captions -> Stage-1 JSONL converter
├── prepare_vqa_jsonl.py     # CSV (image,question,answer) -> Stage-2 JSONL converter
├── src/
│   ├── model.py             # MiniVLM architecture (CLIP + Projector + Qwen + LoRA)
│   └── data.py              # VLMJsonlDataset & collate_vlm_batch
└── outputs/                 # Evaluation outputs and training history logs
```

---

## 👨‍💻 Author & Connect

**Tanish**  
- **GitHub:** [@Tanish-o9](https://github.com/Tanish-o9)
- **Email:** `tanishrajput673@gmail.com`

If you find this repository helpful, consider giving it a ⭐ on GitHub!

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

---

## 📚 Acknowledgements & References
- **LLaVA**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (Liu et al.)
- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al.)
- **Qwen2.5**: [Qwen2.5 Technical Report](https://github.com/QwenLM/Qwen2.5) (Alibaba Cloud)
- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al.)
