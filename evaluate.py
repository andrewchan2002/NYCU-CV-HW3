"""Local AP50 evaluation on the validation split.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth --score-thresh 0.05
    python evaluate.py --tta   # enable TTA for a more accurate local estimate
"""

import argparse

import numpy as np
import torch
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from dataset import CellDataset, get_transform
from inference import apply_per_class_nms, predict_single, predict_with_tta
from model import get_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default="train")
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true",
                        help="Enable TTA during evaluation (slower, more accurate)")
    return parser.parse_args()


def build_coco_gt(dataset, indices):
    images, anns = [], []
    ann_id = 1
    for local_id, global_id in enumerate(indices):
        img_t, target = dataset[global_id]
        h, w = img_t.shape[1], img_t.shape[2]
        images.append({"id": local_id, "height": h, "width": w})
        for j in range(len(target["labels"])):
            m = target["masks"][j].numpy().astype(np.uint8)
            rle = coco_mask_util.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = target["boxes"][j].tolist()
            anns.append({
                "id": ann_id, "image_id": local_id,
                "category_id": int(target["labels"][j]),
                "segmentation": rle,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
            })
            ann_id += 1
    cats = [{"id": i, "name": f"class{i}"} for i in range(1, 5)]
    return {"images": images, "annotations": anns, "categories": cats}


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    val_indices = ckpt.get("val_indices", None)

    dataset = CellDataset(args.data_dir, transforms=get_transform(train=False))

    if val_indices is None:
        n = len(dataset)
        n_val = max(1, int(n * 0.1))
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        val_indices = idx[n - n_val:]
        print(f"No val_indices in checkpoint — using seed-42 split ({len(val_indices)} images)")
    else:
        print(f"Using val split from checkpoint ({len(val_indices)} images)")

    model = get_model(pretrained=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device)
    model.eval()
    print(f"TTA: {'enabled' if args.tta else 'disabled'}")

    coco_gt_dict = build_coco_gt(dataset, val_indices)
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    dt_list = []
    for local_id, global_id in enumerate(tqdm(val_indices, desc="Eval")):
        img_t, _ = dataset[global_id]

        if args.tta:
            boxes, scores, labels, masks = predict_with_tta(model, img_t, device)
        else:
            boxes, scores, labels, masks = predict_single(model, img_t, device)

        if len(boxes) > 0:
            keep = apply_per_class_nms(boxes, scores, labels, masks, args.nms_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]

        valid = scores >= args.score_thresh
        boxes = boxes[valid].numpy()
        labels = labels[valid].numpy()
        scores = scores[valid].numpy()
        masks = masks[valid, 0].numpy()

        for i in range(len(scores)):
            binary = (masks[i] > 0.5).astype(np.uint8)
            rle = coco_mask_util.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = boxes[i]
            dt_list.append({
                "image_id": local_id,
                "category_id": int(labels[i]),
                "segmentation": rle,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i]),
            })

    if not dt_list:
        print("No detections — AP50 = 0.0")
        return

    coco_dt = coco_gt.loadRes(dt_list)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.iouThrs = np.array([0.5])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    print(f"\nAP50 = {ev.stats[0]:.4f}")


if __name__ == "__main__":
    main()
