"""
postprocess.py

PyTorch's contribution to this project: turning OWL-ViT's raw output
tensors into an actual answer, using nothing but tensor operations.

Everything here is written by hand instead of using the library helper
`post_process_object_detection()` or `torchvision.ops.nms`:
  - box format conversion (center_x, center_y, w, h) -> (x1, y1, x2, y2)
  - scaling normalized boxes to real pixel coordinates
  - confidence filtering via sigmoid on the raw logits
  - non-max suppression, implemented from scratch
"""

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (center_x, center_y, w, h) to (x1, y1, x2, y2).

    `boxes` has shape (N, 4) in normalized [0, 1] coordinates.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def scale_to_image(boxes_xyxy: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """Scale normalized (x1, y1, x2, y2) boxes to pixel coordinates."""
    scale = torch.tensor(
        [image_width, image_height, image_width, image_height],
        dtype=boxes_xyxy.dtype,
    )
    return boxes_xyxy * scale


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Intersection-over-union of one box against a set of boxes.

    `box` has shape (4,), `boxes` has shape (N, 4). Both in (x1, y1, x2, y2)
    pixel coordinates. Returns an (N,) tensor of IoU values.
    """
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    intersection = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - intersection
    return intersection / union.clamp(min=1e-8)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.3) -> torch.Tensor:
    """Greedy non-max suppression, implemented from scratch.

    `boxes` has shape (N, 4) in pixel (x1, y1, x2, y2) coordinates,
    `scores` has shape (N,). Returns a LongTensor of indices to keep,
    ordered from highest to lowest score.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current.item())

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def get_best_detection(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    image_width: int,
    image_height: int,
    confidence_threshold: float = 0.1,
    iou_threshold: float = 0.3,
):
    """Turn raw model output into a single best detection, or None.

    Returns a dict {"box": [x1, y1, x2, y2], "score": float} in pixel
    coordinates, or None if nothing clears `confidence_threshold`.
    """
    scores = torch.sigmoid(logits)

    above_threshold = scores >= confidence_threshold
    if not torch.any(above_threshold):
        return None

    kept_scores = scores[above_threshold]
    kept_boxes_norm = pred_boxes[above_threshold]

    boxes_xyxy = cxcywh_to_xyxy(kept_boxes_norm)
    boxes_pixels = scale_to_image(boxes_xyxy, image_width, image_height)

    keep_indices = nms(boxes_pixels, kept_scores, iou_threshold=iou_threshold)
    if keep_indices.numel() == 0:
        return None

    best_index = keep_indices[0]
    return {
        "box": boxes_pixels[best_index].tolist(),
        "score": kept_scores[best_index].item(),
    }
