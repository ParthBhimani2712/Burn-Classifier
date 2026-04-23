# 🔥 burn/degree — AI Burn Wound Classifier & Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/DINOv2-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LoRA-purple?style=for-the-badge"/>
</p>

<p align="center">
  <b>🚀 <a href="https://huggingface.co/spaces/parthbhimani27/Burn_Detection">Live Demo on Hugging Face Spaces</a></b>
</p>

> ⚠️ **Research demo only.** This is not a medical device and must not be used for clinical diagnosis. It is a portfolio project demonstrating transfer learning, parameter-efficient fine-tuning, and multi-task learning on a small dataset.

---

## 📌 Overview

**burn/degree** is an end-to-end deep learning system for burn wound **classification** and **segmentation**. It predicts burn severity across three clinical grades — 1st, 2nd, and 3rd degree — and produces a pixel-level segmentation overlay of the affected tissue area.

The system is powered by a fine-tuned **DINOv2-large** backbone enhanced with **LoRA adapters**, paired with a **SegFormer** decoder head for segmentation. The backend is served as a **FastAPI** REST API with a lightweight single-page frontend.

---

## ✨ Features

- **Multi-task inference** — joint classification (severity grade) and segmentation (pixel mask) in a single forward pass
- **Parameter-efficient fine-tuning** — LoRA (rank 8, α=16) applied to Q/K/V/Output projections of DINOv2-large
- **Test-Time Augmentation (TTA)** — 4-way flip ensemble (identity, H-flip, V-flip, HV-flip) for improved robustness
- **Per-class area breakdown** — reports percentage of tissue area per burn grade
- **Confidence scores** — softmax probabilities for all three classes
- **Visual output** — returns the original image, segmentation mask, and blended overlay, all as base64 PNGs

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Backbone | DINOv2-large (ViT-L/14) |
| Fine-tuning strategy | LoRA — rank 8, α=16, applied to Q/K/V/Out projections |
| Classification head | Learned query + MultiheadAttention pooling |
| Segmentation head | SegFormer MLP decoder fusing layers 5 / 11 / 17 / 23 |
| Input resolution | 224 × 224, ImageNet normalisation |
| Training losses | Focal loss + Dice loss |
| Augmentations | MixUp, CutMix, TTA at inference |
| Regularisation | EMA, cosine warm-restart LR schedule, gradient checkpointing |
| Training hardware | Single NVIDIA T4 GPU |

---

## 📁 Project Structure

```
burn-classifier/
├── backend/
│   ├── model.py          # BurnSOTAModel (inference-only)
│   ├── inference.py      # Preprocessing, TTA, overlay generation
│   └── main.py           # FastAPI app (API + static frontend)
├── static/
│   └── index.html        # Single-page frontend
├── burn_final.pth        # ← Trained weights (not included in repo)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/burn-classifier.git
cd burn-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

**CPU-only** (works anywhere, ~5–10 s per inference):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**CUDA 12.1** (GPU, <1 s per inference):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Add trained weights

Place your `burn_final.pth` in the project root:

```
burn-classifier/burn_final.pth
```

To use a custom checkpoint path:

```bash
export BURN_CHECKPOINT=/absolute/path/to/burn_final.pth
```

### 5. Run the server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

> On first startup, `transformers` will download DINOv2-large weights (~1.2 GB from HuggingFace Hub). This only happens once.

---

## ⚙️ Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `BURN_CHECKPOINT` | `./burn_final.pth` | Path to the trained checkpoint |
| `BURN_USE_TTA` | `1` | Set to `0` to disable 4-way TTA (faster inference) |

---

## 🌐 API Reference

### `GET /api/health`

Returns model and device status.

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

Submit an image for burn classification and segmentation.

**Request:** Multipart form with a `file` field.

```bash
curl -X POST http://localhost:8000/api/predict \
     -F "file=@sample.jpg"
```

**Response:**

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

## ☁️ Deployment

The live demo is hosted on **Hugging Face Spaces** (Docker runtime):

🔗 [https://huggingface.co/spaces/parthbhimani27/Burn_Detection](https://huggingface.co/spaces/parthbhimani27/Burn_Detection)

### Deploy your own

**Option A — Hugging Face Spaces** *(recommended)*
- Upload the repo as a Docker Space
- Upload `burn_final.pth` directly through the Spaces UI (no Git LFS required)
- Free CPU tier available; upgradable to GPU

**Option B — Render / Railway / Fly.io**
- Host `burn_final.pth` on S3, GCS, or HF Hub and download at startup
- Note: free CPU tiers may be RAM-constrained for DINOv2-large

**Option C — Docker + VPS**

```dockerfile
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

## 🔁 Training vs. Inference Consistency

| Training detail | Inference mirror |
|---|---|
| DINOv2-large + LoRA (rank 8, α=16) in Q/K/V/Out | LoRA injected **before** `load_state_dict` |
| 224×224 input, ImageNet normalisation | Identical preprocessing |
| Classification head: learned query + MHA pooling | Same module |
| Segmentation head: SegFormer MLP decoder (layers 5/11/17/23) | Same module |
| TTA: identity + H-flip + V-flip + HV-flip | Same flips, softmax-averaged |
| `gradient_checkpointing_enable()` during training | Explicitly disabled at load |
| `model.train()` during training | `model.eval()` + `torch.no_grad()` at inference |

---

## 🛠️ Tech Stack

`PyTorch` · `Transformers (HuggingFace)` · `DINOv2` · `LoRA / PEFT` · `SegFormer` · `FastAPI` · `Uvicorn` · `Pillow` · `NumPy`

---

## 📄 License

This project is released for educational and portfolio purposes. The model weights are not intended for commercial or clinical use.

---

## 🙋 Author

**Parth Bhimani**
[Hugging Face](https://huggingface.co/parthbhimani27) · [GitHub](https://github.com/<your-username>)
