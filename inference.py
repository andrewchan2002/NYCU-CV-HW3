"""Inference script — generates test-results.json for CodaBench submission.

Improvements over baseline:
  - Test-Time Augmentation (TTA): runs inference on original + H-flipped image,
    then merges predictions for an ensemble effect.
  - Per-class NMS after TTA merge prevents duplicate detections across views.

Usage:
    python inference.py
    python inference.py --checkpoint checkpoints/best_model.pth --score-thresh 0.5
    python inference.py --no-tta   # disable TTA for faster but weaker results
"""

import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools import mask as coco_mask_util
from torchvision.ops import nms

from model import get_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--test-dir", default="test_release")
    parser.add_argument("--test-json", default="test_image_name_to_ids.json")
    parser.add_argument("--output", default="test-results.json")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable test-time augmentation")
    return parser.parse_args()


def encode_mask(binary_mask_np):
    """Convert a (H, W) uint8 numpy mask to COCO RLE dict."""
    arr = np.asfortranarray(binary_mask_np.astype(np.uint8))
    rle = coco_mask_util.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _predict(model, img_t, device):
    """Single forward pass; returns CPU tensors."""
    with torch.no_grad():
        pred = model([img_t.to(device)])[0]
    return {k: v.cpu() for k, v in pred.items()}


def predict_with_tta(model, img_t, device):
    """Two-view TTA: original + horizontal flip.

    Flip predictions are adjusted back to original coordinates before merging.
    Per-class NMS is applied to the combined candidate set.
    """
    pred_orig = _predict(model, img_t, device)

    img_flip = img_t.flip(-1)  # horizontal flip (C, H, W)
    pred_flip = _predict(model, img_flip, device)

    # Map flipped coordinates back to original space
    w = img_t.shape[-1]
    boxes_flip = pred_flip["boxes"].clone()
    boxes_flip[:, [0, 2]] = w - boxes_flip[:, [2, 0]]
    masks_flip = pred_flip["masks"].flip(-1)  # (N, 1, H, W)

    boxes = torch.cat([pred_orig["boxes"], boxes_flip])
    scores = torch.cat([pred_orig["scores"], pred_flip["scores"]])
    labels = torch.cat([pred_orig["labels"], pred_flip["labels"]])
    masks = torch.cat([pred_orig["masks"], masks_flip])

    return boxes, scores, labels, masks


def predict_single(model, img_t, device):
    """Single-view inference without TTA."""
    pred = _predict(model, img_t, device)
    return pred["boxes"], pred["scores"], pred["labels"], pred["masks"]


def apply_per_class_nms(boxes, scores, labels, masks, iou_thresh):
    """NMS applied independently per class to avoid suppressing different-class overlaps."""
    keep_all = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        keep = nms(boxes[idx], scores[idx], iou_thresh)
        keep_all.append(idx[keep])
    if not keep_all:
        empty = torch.zeros(0, dtype=torch.long)
        return empty
    return torch.cat(keep_all)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.test_json, "r") as f:
        test_info = json.load(f)
    print(f"Test images: {len(test_info)}")

    model = get_model(pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    print(f"TTA: {'disabled' if args.no_tta else 'enabled (H-flip)'}")

    results = []

    for item in test_info:
        image_id = item["id"]
        file_name = item["file_name"]
        img_path = os.path.join(args.test_dir, file_name)

        if not os.path.exists(img_path):
            print(f"  WARN: {img_path} missing — skipped")
            continue

        img = Image.open(img_path).convert("RGB")
        img_t = TF.to_tensor(img)

        if args.no_tta:
            boxes, scores, labels, masks = predict_single(model, img_t, device)
        else:
            boxes, scores, labels, masks = predict_with_tta(model, img_t, device)

        # Per-class NMS on merged TTA candidates
        if len(boxes) > 0:
            keep = apply_per_class_nms(boxes, scores, labels, masks, args.nms_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]

        # Final score threshold
        valid = scores >= args.score_thresh
        boxes = boxes[valid].numpy()
        labels = labels[valid].numpy()
        scores = scores[valid].numpy()
        masks = masks[valid, 0].numpy()  # (N, H, W) float

        for i in range(len(scores)):
            binary = (masks[i] > 0.5).astype(np.uint8)
            rle = encode_mask(binary)
            x1, y1, x2, y2 = boxes[i]
            results.append({
                "image_id":    int(image_id),
                "category_id": int(labels[i]),
                "segmentation": rle,
                "bbox":  [round(float(x1), 2), round(float(y1), 2),
                          round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(scores[i]), 6),
            })

        print(f"  [{image_id:3d}] {file_name}: {len(scores)} detections")

    with open(args.output, "w") as f:
        json.dump(results, f)

    print(f"\nTotal detections: {len(results)}")
    print(f"Saved to: {args.output}")
    print("Next step: zip the file and upload to CodaBench.")


if __name__ == "__main__":
    main()
