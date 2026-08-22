"""
detector.py

Hugging Face's contribution to this project: a pretrained zero-shot object
detection model (OWL-ViT2 / OWL-ViT) and the processor that prepares inputs
for it.

This module only loads the model and runs a forward pass. It deliberately
returns the raw output tensors (logits, pred_boxes) instead of calling the
library's post-processing helper — all of the box math lives in
postprocess.py using plain PyTorch.

Note: we use "google/owlv2-base-patch16-ensemble" (OWL-ViT2) rather than the
original "google/owlvit-base-patch32". Testing on this project's cluttered
"I Spy"-style images showed owlvit-base-patch32 produces very low confidence
scores (mostly 0.02-0.05) even for objects clearly present in the image,
which made a sensible confidence threshold impractical. owlv2-base-patch16
-ensemble is Google's improved successor with the same zero-shot API shape
and noticeably higher, more usable confidence scores on the same images.
"""

from functools import lru_cache

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_NAME = "google/owlv2-base-patch16-ensemble"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the OWL-ViT2 processor + model (loaded once per process)."""
    processor = Owlv2Processor.from_pretrained(MODEL_NAME)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def run_detection(image: Image.Image, text_query: str):
    """Run OWL-ViT on an image with a single text query.

    Returns:
        logits: FloatTensor of shape (num_queries,) — one score per
            predicted box for our single text query.
        pred_boxes: FloatTensor of shape (num_queries, 4) — boxes in
            normalized (center_x, center_y, width, height) format.
    """
    processor, model = load_model()

    inputs = processor(text=[[text_query]], images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Batch size is 1 and there is exactly one text query, so squeeze both
    # the batch dimension and the per-query class dimension.
    logits = outputs.logits[0, :, 0]
    pred_boxes = outputs.pred_boxes[0]

    return logits, pred_boxes
