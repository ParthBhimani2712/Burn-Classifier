"""
v2 inference: classification + binary segmentation, with the wound region
coloured by the *image-level* predicted severity.

Drops in alongside the v1 backend code with no changes to main.py — see the
import swaps at the bottom of the patch instructions.
"""

import io
import base64
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp

# ─────────────────────────── architecture ────────────────────────────────────
# Must match the training script EXACTLY — same encoder, same head structure.
ENCODER = "efficientnet-b3"
NUM_CLASSES = 3
NUM_SEG_CLASSES = 1
IMG_SIZE = 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ["1st degree", "2nd degree", "3rd degree"]


class BurnMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=None,           # we'll load our own weights
            in_channels=3,
            classes=NUM_SEG_CLASSES,
            activation=None,
        )
        encoder_out_ch = self.unet.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(encoder_out_ch, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        features = self.unet.encoder(x)
        cls_logits = self.cls_head(features[-1])
        decoder_out = self.unet.decoder(features)
        seg_logits = self.unet.segmentation_head(decoder_out)
        return cls_logits, seg_logits


def load_model(checkpoint_path: str, device: torch.device) -> BurnMultiTask:
    """Load the v2 checkpoint produced by train_v2.py."""
    print(f"[model] instantiating BurnMultiTask ({ENCODER}) on {device}")
    model = BurnMultiTask()

    print(f"[model] loading weights from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        weights = state["model"]
    else:
        weights = state

    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"[model] ⚠ {len(missing)} missing keys (first: {missing[0]})")
    if unexpected:
        print(f"[model] ⚠ {len(unexpected)} unexpected keys (first: {unexpected[0]})")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[model] ready.")
    return model


# ─────────────────────────── preprocessing ───────────────────────────────────
def preprocess(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return tensor.to(device)


# ─────────────────────────── 4-way TTA ──────────────────────────────────────
_TTA = [
    (lambda x: x,                          lambda y: y),
    (lambda x: torch.flip(x, dims=[3]),    lambda y: torch.flip(y, dims=[-1])),
    (lambda x: torch.flip(x, dims=[2]),    lambda y: torch.flip(y, dims=[-2])),
    (lambda x: torch.flip(x, dims=[2, 3]), lambda y: torch.flip(y, dims=[-2, -1])),
]


@torch.no_grad()
def _predict_with_tta(model, x):
    cls_probs, seg_probs = [], []
    for fwd, inv in _TTA:
        cl, sl = model(fwd(x))
        cls_probs.append(F.softmax(cl, dim=1))
        seg_probs.append(inv(torch.sigmoid(sl)).squeeze(1))
    return torch.stack(cls_probs).mean(0), torch.stack(seg_probs).mean(0)


# ─────────────────────────── colour palette ─────────────────────────────────
# Severity-coded RGB: yellow → orange → dark red.
# Background and unburned-skin regions are NOT painted (they show through as the
# original image), which is much cleaner visually than colouring them.
SEVERITY_COLORS = {
    0: (255, 200, 60),    # 1st degree → yellow-amber
    1: (240, 120, 40),    # 2nd degree → orange
    2: (160, 30, 30),     # 3rd degree → dark red
}


# ─────────────────────────── main entrypoint ────────────────────────────────
@torch.no_grad()
def predict(model, pil_image: Image.Image, device: torch.device,
            use_tta: bool = True, seg_threshold: float = 0.5) -> Dict[str, Any]:
    image_rgb = pil_image.convert("RGB")
    orig_w, orig_h = image_rgb.size

    x = preprocess(image_rgb, device)

    if use_tta:
        cls_probs, seg_probs = _predict_with_tta(model, x)
    else:
        cls_logits, seg_logits = model(x)
        cls_probs = F.softmax(cls_logits, dim=1)
        seg_probs = torch.sigmoid(seg_logits).squeeze(1)

    # upsample seg map to original resolution for a pixel-perfect overlay
    seg_probs_up = F.interpolate(
        seg_probs.unsqueeze(1), size=(orig_h, orig_w),
        mode="bilinear", align_corners=False,
    ).squeeze(1).squeeze(0).cpu().numpy()        # (H, W) float in [0, 1]

    cls_np = cls_probs.squeeze(0).cpu().numpy()
    pred_idx = int(cls_np.argmax())

    # build the visuals
    overlay_img = build_overlay(
        image_rgb, seg_probs_up, pred_idx,
        threshold=seg_threshold, max_alpha=0.6,
    )
    heatmap_img = build_heatmap_only(seg_probs_up, pred_idx)

    # area stats: pixels above threshold = wound area
    wound_pct = float((seg_probs_up > seg_threshold).mean() * 100.0)
    mean_conf_in_wound = (
        float(seg_probs_up[seg_probs_up > seg_threshold].mean())
        if (seg_probs_up > seg_threshold).any() else 0.0
    )

    return {
        "predicted_class": pred_idx,
        "predicted_class_name": CLASS_NAMES[pred_idx],
        "confidence": float(cls_np[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(cls_np[i]) for i in range(NUM_CLASSES)},
        "wound_area_pct": round(wound_pct, 2),
        "mean_seg_confidence": round(mean_conf_in_wound, 3),
        "original_image": pil_to_b64(image_rgb),
        "overlay_image": pil_to_b64(overlay_img),
        "mask_image": pil_to_b64(heatmap_img),
        "image_width": orig_w,
        "image_height": orig_h,
        "tta_used": bool(use_tta),
        # kept for frontend backward compatibility (old UI reads this key)
        "per_class_area_pct": {
            CLASS_NAMES[i]: (round(wound_pct, 2) if i == pred_idx else 0.0)
            for i in range(NUM_CLASSES)
        },
    }


# ─────────────────────────── visualisation ──────────────────────────────────
def build_overlay(pil_image: Image.Image, seg_probs: np.ndarray,
                  pred_class: int, threshold: float = 0.5,
                  max_alpha: float = 0.6) -> Image.Image:
    """
    Paint the wound region in the colour of the predicted severity, with alpha
    proportional to per-pixel segmentation confidence. Unburned skin and
    background show through unchanged.
    """
    img = np.asarray(pil_image).astype(np.float32)         # (H, W, 3)
    color = np.array(SEVERITY_COLORS[pred_class], dtype=np.float32)  # (3,)

    # alpha is 0 below threshold, then ramps linearly from 0→max_alpha
    # between (threshold, 1.0). Smooth fade keeps boundaries from looking pixelated.
    a = np.clip((seg_probs - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    a = (a * max_alpha)[..., None]                          # (H, W, 1)

    blended = (1.0 - a) * img + a * color
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def build_heatmap_only(seg_probs: np.ndarray, pred_class: int) -> Image.Image:
    """
    Pure heatmap on a dark background — for the 'segmentation' tab in the UI.
    Uses a yellow→orange→red gradient driven by per-pixel confidence, keyed
    to the predicted severity colour.
    """
    h, w = seg_probs.shape
    color = np.array(SEVERITY_COLORS[pred_class], dtype=np.float32)
    bg = np.array([22, 18, 14], dtype=np.float32)

    # interpolate from background to severity colour by confidence
    a = seg_probs[..., None].astype(np.float32)
    canvas = (1.0 - a) * bg + a * color
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def pil_to_b64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    img = pil_image
    if max(img.size) > 1280:
        img = img.copy()
        img.thumbnail((1280, 1280), Image.LANCZOS)
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
