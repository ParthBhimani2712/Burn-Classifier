# burn/degree

AI-assisted burn wound classification and segmentation. A fine-tuned
**DINOv2-large + LoRA** backbone with a **SegFormer** decoder head predicts
burn severity (1st / 2nd / 3rd degree) and produces a pixel-level map of the
affected tissue, served through a FastAPI backend and a single-page frontend.

> ⚠️ **Research demo only.** This is not a medical device and must not be
> used for diagnosis. It is a portfolio project demonstrating transfer
> learning, parameter-efficient fine-tuning, and multi-task learning on a
> small dataset.

---

## Project structure

```
burn-classifier/
├── backend/
│   ├── model.py          # BurnSOTAModel (inference-only)
│   ├── inference.py      # preprocessing, TTA, overlay generation
│   └── main.py           # FastAPI app (API + static frontend)
├── static/
│   └── index.html        # single-page frontend
├── burn_final.pth        # ← your trained weights go here
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

For CPU-only (works anywhere, ~5–10 s per inference):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For CUDA 12.1 (GPU, <1 s per inference):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Drop in the trained weights

Copy your `burn_final.pth` (produced by the training script) into the project
root:

```
burn-classifier/burn_final.pth
```

If you want to use a different path:

```bash
export BURN_CHECKPOINT=/absolute/path/to/burn_final.pth
```

### 4. Run

```bash
cd burn-classifier
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. On first startup, `transformers` will download
the DINOv2-large weights (~1.2 GB) — this takes a minute and only happens
once.

---

## Environment variables

| Variable           | Default                          | Purpose                                    |
| ------------------ | -------------------------------- | ------------------------------------------ |
| `BURN_CHECKPOINT`  | `./burn_final.pth`               | Path to the trained checkpoint.            |
| `BURN_USE_TTA`     | `1`                              | Set to `0` to disable 4-way TTA (faster).  |

---

## API

### `GET /api/health`

```json
{
  "status": "ok",
  "device": "cuda",
  "model_loaded": true,
  "error": null,
  "tta": true
}
```

### `POST /api/predict`

Multipart form with a single `file` field (image).

```bash
curl -X POST http://localhost:8000/api/predict \
     -F "file=@sample.jpg"
```

Response (truncated):

```json
{
  "predicted_class": 1,
  "predicted_class_name": "2nd degree",
  "confidence": 0.874,
  "probabilities": {
    "1st degree": 0.08,
    "2nd degree": 0.874,
    "3rd degree": 0.046
  },
  "per_class_area_pct": {
    "1st degree": 12.4,
    "2nd degree": 34.1,
    "3rd degree": 2.7
  },
  "original_image": "data:image/png;base64,...",
  "overlay_image":  "data:image/png;base64,...",
  "mask_image":     "data:image/png;base64,...",
  "image_width": 1024,
  "image_height": 768,
  "tta_used": true,
  "latency_ms": 412
}
```

---

## Deployment notes

This project is size-constrained by its weights:

- **DINOv2-large** (~1.2 GB, downloaded once from HuggingFace Hub at runtime).
- Your `burn_final.pth` (hundreds of MB — too large for a normal git commit).

Three realistic deployment routes:

### A. Hugging Face Spaces *(recommended for a resume demo)*

- Best fit: the platform is designed for ML demos, free CPU tier available,
  upgradable to GPU.
- Upload `burn_final.pth` to the Space directly (Spaces support large files
  without Git LFS).
- Either keep the FastAPI app as-is (Docker Space) or convert to a Gradio
  interface if you want a one-file option.

### B. Render / Railway / Fly.io

- Put the checkpoint on an object store (S3, GCS, HF Hub) and download at
  startup, OR use Git LFS. Free CPU tiers are tight on RAM — you may need
  a paid plan for DINOv2-large.

### C. Docker + your own VPS

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ backend/
COPY static/ static/
COPY burn_final.pth .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## How it matches the training setup

| Training detail                                      | Inference mirror                               |
| ---------------------------------------------------- | ---------------------------------------------- |
| DINOv2-large backbone + LoRA (rank 8, α=16) in Q/K/V/Out | LoRA injected **before** `load_state_dict`.     |
| 224×224 input, ImageNet normalisation                | Same preprocessing.                            |
| Classification head: learned query + MultiheadAttention pooling | Same module.                           |
| Segmentation head: SegFormer MLP decoder fusing layers 5/11/17/23 | Same module.                           |
| Eval TTA: identity + H-flip + V-flip + HV-flip       | Same flips, softmax averaged.                  |
| Backbone `gradient_checkpointing_enable()`           | Explicitly disabled at load.                   |
| `model.train()` during training                      | `model.eval()` + `torch.no_grad()` throughout. |

---

## For your resume

Short description you can lift:

> **Burn Wound Classifier** — End-to-end multi-task deep learning project:
> fine-tuned DINOv2-large with LoRA adapters for joint classification and
> segmentation of burn wounds (1st/2nd/3rd degree). Two-stage training with
> focal + dice losses, MixUp/CutMix, EMA, and cosine warm-restart schedules
> on a single T4 GPU. Deployed as a FastAPI + vanilla-JS web app with
> test-time augmentation at inference.

**Keywords worth highlighting:** PyTorch, Transformers, DINOv2, LoRA
(parameter-efficient fine-tuning), multi-task learning, SegFormer, mixed
precision, gradient checkpointing, EMA, FastAPI.
