# Mini Vision-Language Model (LLaVA-Style)

A lightweight, modular, and educational **Vision-Language Model (VLM)** implementation inspired by the **LLaVA (Large Language and Vision Assistant)** architecture. It aligns a frozen visual encoder with an instruction-tuned small language model (SLM) using an intermediate projection network and Low-Rank Adaptation (LoRA).

---

## 🏗️ Model Architecture & Dataflow

```
[ Input Image (224x224 RGB) ]
              │
              ▼
    [ Frozen CLIP ViT-B/32 ]
              │ (Extract Penultimate Layer Hidden States [-2])
              ▼
   [ 49 Patch Tokens × 768-dim ]
              │
              ▼ (Optional: AdaptiveAvgPool2d Resampler)
   [ Spatial Tokens × 768-dim ]
              │
              ▼
    [ Trainable 2-Layer MLP ] ── Linear(768, 896) -> GELU -> Linear(896, 896)
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
    [ Qwen2.5-0.5B-Instruct ]  <── Base weights frozen, LoRA adapters on q_proj & v_proj (Stage 2)
              │
              ▼
    [ Autoregressive Response ]
```

### Core Components
1. **Vision Encoder**: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) (Frozen). Extracts patch tokens from the **penultimate layer** (`hidden_states[-2]`), preserving rich localized spatial features.
2. **Projector**: Trainable 2-Layer MLP (`Linear(768, 896) -> GELU -> Linear(896, 896)`). Maps visual representations into Qwen's word embedding space.
3. **Language Model**: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Frozen base, trainable LoRA in Stage 2).
4. **Prompt Format**: Native Qwen **ChatML** template (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).

---

## 🔄 Two-Stage Training Paradigm

| Feature | Stage 1: Feature Alignment | Stage 2: Visual Instruction Tuning (VQA) |
| :--- | :--- | :--- |
| **Objective** | Teach the projector to translate visual features into LLM soft-tokens | Teach the model to follow instructions and answer visual questions |
| **Dataset Format** | Image-caption pairs (e.g. Flickr8k) | VQA triplets (`image, question, answer`) |
| **Trainable Weights** | **Projector only** (CLIP & Qwen frozen) | **Projector + Qwen LoRA** (`q_proj`, `v_proj`) |
| **Loss Supervision** | Cross-entropy on **caption tokens only** (prefix masked with `-100`) | Cross-entropy on **answer tokens only** (prefix masked with `-100`) |
| **Saved Artifacts** | `checkpoints/stage1/projector.pt` | `checkpoints/stage2/` (`projector.pt`, `lora/`, `tokenizer/`) |

---

## 🚀 Quickstart & Setup

### 1. Installation
Python 3.10, 3.11, or 3.13 is recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### 2. Verified End-to-End Smoke Test
Run the automated smoke test script to verify environment compatibility, Hugging Face downloads, dynamic batch collation, forward/backward passes, multi-turn inference, and checkpoint reloading:

```bash
python smoke_test.py
```

#### Real Verified Execution Output:
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
[OK] Verified multi-turn inference (Model Answer: '...').
[OK] Verified saving and reloading Stage 2 checkpoint.
============================================================
ALL SMOKE TESTS PASSED SUCCESSFULLY!
============================================================
```

---

## 📊 Complete Step-by-Step Pipeline

### Step 1: Prepare Stage-1 Captioning Data
Format standard Flickr8k images and captions into train/val JSONL splits:
```bash
python prepare_flickr8k.py --images-dir "D:/datasets/Flickr8k/Images" --captions-file "D:/datasets/Flickr8k/Flickr8k.token.txt"
```
*Outputs:* `data/flickr8k/train.jsonl` and `data/flickr8k/val.jsonl`.

### Step 2: (Optional) Precompute & Cache CLIP Features
Saves the 49 patch embeddings per image to disk (`.pt` files) so training can skip the vision encoder forward pass:
```bash
python cache_features.py --input-jsonl data/flickr8k/train.jsonl --output-jsonl data/flickr8k/train_cached.jsonl
```

### Step 3: Train Stage 1 (Projector Alignment)
Train the projector using PyTorch Native AMP, Cosine LR Scheduler with warmup, and validation checkpointing:
```bash
python train.py --stage stage1 --train-jsonl data/flickr8k/train_cached.jsonl --val-jsonl data/flickr8k/val.jsonl --epochs 2
```
*Checkpoints:* `checkpoints/stage1/projector.pt` (and `projector_best.pt`).  
*Metrics Log:* `outputs/stage1_history.csv`.

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

> **Low VRAM Option (4-bit QLoRA):**
> If training on a GPU with $\le 6\text{ GB}$ VRAM, add `--load-in-4bit`:
> ```bash
> python train.py --stage stage2 --train-jsonl data/vqa/train.jsonl --load-in-4bit
> ```

*Checkpoints:* `checkpoints/stage2/` (`projector.pt`, `lora/`, `tokenizer/`).  
*Metrics Log:* `outputs/stage2_history.csv`.

### Step 6: Evaluate on Held-Out Test Set
Evaluate accuracy across normalized exact match and token-level F1 score:
```bash
python evaluate.py --jsonl data/vqa/test.jsonl --checkpoint checkpoints/stage2
```
*Outputs:* Prints normalized exact-match accuracy, mean token-F1 score, and average latency; exports per-sample predictions to `outputs/evaluation.csv`.

### Step 7: Launch Conversational Web Demo
Launch the Gradio multi-turn chatbot interface:
```bash
python app.py --stage stage2 --checkpoint checkpoints/stage2
```
Open `http://127.0.0.1:7860` in your browser, upload an image, and converse with the model with full dialogue history context!

---

## 🛠️ Key Technical Features

1. **Native ChatML Support**: Uses `tokenizer.apply_chat_template()` for seamless alignment with Qwen2.5-Instruct's conversation priors.
2. **Penultimate Layer Features**: Extracts CLIP's `hidden_states[-2]`, avoiding the text-collapsed final representation and capturing richer spatial geometry.
3. **Dynamic Batching & Collation**: Custom `collate_vlm_batch` in `src/data.py` supports variable-length text padding, attention masking, and `batch_size >= 1`.
4. **PyTorch Native AMP & GradScaler**: Uses `torch.amp.autocast` and `GradScaler` for numeric stability and fast FP16 compute on CUDA.
5. **Learning Rate Scheduler**: Cosine decay with configurable warmup ratio (default 5%) for smooth projector training.
6. **Multi-Turn Chatbot**: Gradio UI supports sequential dialogue about uploaded images, tracking chat history across turns.
7. **Visual Token Resampling**: Optional downsampling via `resample_tokens` in `config.yaml` (e.g. 16 or 25 tokens) for lower latency and reduced VRAM footprint.
8. **QLoRA 4-bit / 8-bit Quantization**: Optional `load_in_4bit` integration for consumer GPU setups ($< 4\text{ GB}$ VRAM).
9. **CSV Training History**: Automatically logs epoch-level train loss, validation loss, and learning rates to `outputs/`.

---

## 📁 Repository Structure

```
Mini Vision-Language Model/
├── config.yaml              # Centralized hyperparameters & paths
├── requirements.txt         # Pinned Python package dependencies
├── smoke_test.py            # Automated end-to-end pipeline test
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

## 📜 Citation & Credits
- **LLaVA**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (Liu et al.)
- **Vision Encoder**: [CLIP](https://github.com/openai/CLIP) (Radford et al.)
- **Language Backbone**: [Qwen2.5](https://github.com/QwenLM/Qwen2.5) (Alibaba Cloud)
- **Parameter-Efficient Tuning**: [PEFT](https://github.com/huggingface/peft) (Hugging Face)
