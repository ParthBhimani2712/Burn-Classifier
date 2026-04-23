"""
FastAPI server for the burn wound classifier.

Routes
------
GET  /                 → serves the single-page frontend
GET  /api/health       → { status, device, model_loaded }
POST /api/predict      → multipart form with `file` (image) → prediction JSON

Run locally:
    cd burn-classifier
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError


from inference_v2 import load_model, predict

# ────────────────────────── config ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# allow overriding via env var; default looks in project root
CHECKPOINT_PATH = os.environ.get(
    "BURN_CHECKPOINT",
    str(BASE_DIR / "burn_v2_final.pth"),
)
USE_TTA = os.environ.get("BURN_USE_TTA", "1") == "1"
MAX_UPLOAD_MB = 10
MAX_SIDE = 1024  # downscale huge uploads before inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────── model load ───────────────────────────────────────
print("=" * 60)
print(f"  device          : {DEVICE}")
print(f"  checkpoint      : {CHECKPOINT_PATH}")
print(f"  TTA at inference: {USE_TTA}")
print("=" * 60)

MODEL = None
MODEL_ERROR = None
try:
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Place your trained burn_final.pth in the project root, or set "
            "the BURN_CHECKPOINT env var to its absolute path."
        )
    MODEL = load_model(CHECKPOINT_PATH, device=DEVICE)
except Exception as e:  # noqa: BLE001 — we want to surface this in /api/health
    MODEL_ERROR = str(e)
    print(f"[startup] ⚠ model failed to load: {e}")

# ────────────────────────── app ──────────────────────────────────────────────
app = FastAPI(
    title="Burn Wound Classifier",
    description="EfficientNet-B3 + U-Net decoder for burn-degree "
                "classification and segmentation.",
    version="2.0.0",
)

# CORS open by default; tighten allow_origins if you split frontend/backend hosts.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if MODEL is not None else "model_unavailable",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "error": MODEL_ERROR,
        "tta": USE_TTA,
    }


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, f"Model unavailable: {MODEL_ERROR}")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image too large (> {MAX_UPLOAD_MB} MB).")

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()  # force decode now so we fail fast on broken files
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not decode image.")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Image error: {e}")

    # downscale very large uploads to keep inference + response size reasonable
    if max(pil.size) > MAX_SIDE:
        pil.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    t0 = time.time()
    try:
        result = predict(MODEL, pil, device=DEVICE, use_tta=USE_TTA)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Inference failed: {e}")

    result["latency_ms"] = int((time.time() - t0) * 1000)
    return JSONResponse(result)


# ────────────────────────── static frontend ──────────────────────────────────
# Serve index.html at "/" and static assets under their own paths.
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/")
    async def root():
        index = STATIC_DIR / "index.html"
        if not index.exists():
            return JSONResponse({"error": "frontend not built"}, status_code=500)
        return FileResponse(index)
else:
    @app.get("/")
    async def root_placeholder():
        return {
            "message": "Burn wound classifier API running. "
                       "Frontend not found at static/.",
            "endpoints": ["/api/health", "/api/predict"],
        }
