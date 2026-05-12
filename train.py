"""Training script for medical cell instance segmentation.

Improvements over baseline:
  - AdamW optimizer (better convergence than SGD for this task)
  - OneCycleLR schedule (built-in warmup + annealing in one scheduler)
  - Mixed-precision training via AMP (faster, same quality)
  - Gradient accumulation for effective larger batch size
  - Enhanced augmentation (affine, Gaussian noise) in dataset.py
  - Better model RPN settings in model.py

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 2 --lr 1e-3 --amp
    python train.py --resume checkpoints/last_model.pth
"""

import argparse
import contextlib
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CellDataset, collate_fn, get_transform
from model import count_parameters, get_model, get_param_groups


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="train")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="LR for new heads; backbone gets lr * 0.1")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch-size * accum-steps)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training (requires CUDA)")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Compute AP50 on val set every N epochs (0 = disable)")
    parser.add_argument("--backbone", default="resnet50",
                        choices=["resnet50", "resnet101", "resnet152"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _amp_context(enabled):
    if enabled:
        return torch.amp.autocast(device_type="cuda")
    return contextlib.nullcontext()


def train_one_epoch(model, optimizer, loader, device, epoch,
                    scaler=None, accum_steps=1, scheduler=None, print_freq=10):
    model.train()
    total_loss = 0.0
    n = len(loader)
    use_amp = scaler is not None
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with _amp_context(use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values()) / accum_steps

        raw_loss = losses.item() * accum_steps
        if not math.isfinite(raw_loss):
            print(f"  [skip] non-finite loss at batch {i}")
            optimizer.zero_grad()
            continue

        if scaler:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        # Optimization step after accumulating enough gradients
        last_batch = (i + 1) == n
        if (i + 1) % accum_steps == 0 or last_batch:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += raw_loss

        if (i + 1) % print_freq == 0 or last_batch:
            parts = " ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items())
            print(f"  Ep{epoch} [{i+1}/{n}] total={raw_loss:.4f} | {parts}")

    return total_loss / max(n, 1)


@torch.no_grad()
def validate_loss(model, loader, device):
    model.train()  # keep train mode to get losses
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        if math.isfinite(losses.item()):
            total += losses.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def compute_val_ap50(model, dataset, val_indices, device, score_thresh=0.05):
    """Run COCO AP50 evaluation on val split without TTA."""
    from pycocotools import mask as coco_mask_util
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from inference import apply_per_class_nms, predict_single

    model.eval()

    images_gt, anns_gt, ann_id = [], [], 1
    for local_id, global_id in enumerate(val_indices):
        img_t, target = dataset[global_id]
        h, w = img_t.shape[1], img_t.shape[2]
        images_gt.append({"id": local_id, "height": h, "width": w})
        for j in range(len(target["labels"])):
            m = target["masks"][j].numpy().astype(np.uint8)
            rle = coco_mask_util.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = target["boxes"][j].tolist()
            anns_gt.append({
                "id": ann_id, "image_id": local_id,
                "category_id": int(target["labels"][j]),
                "segmentation": rle,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
            })
            ann_id += 1

    cats = [{"id": i, "name": f"class{i}"} for i in range(1, 5)]
    coco_gt = COCO()
    coco_gt.dataset = {"images": images_gt, "annotations": anns_gt, "categories": cats}
    coco_gt.createIndex()

    dt_list = []
    for local_id, global_id in enumerate(val_indices):
        img_t, _ = dataset[global_id]
        boxes, scores, labels, masks = predict_single(model, img_t, device)
        if len(boxes) > 0:
            keep = apply_per_class_nms(boxes, scores, labels, masks, 0.5)
            boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
        valid = scores >= score_thresh
        b, s, l, m = boxes[valid].numpy(), scores[valid].numpy(), labels[valid].numpy(), masks[valid, 0].numpy()
        for i in range(len(s)):
            binary = (m[i] > 0.5).astype(np.uint8)
            rle = coco_mask_util.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = b[i]
            dt_list.append({
                "image_id": local_id,
                "category_id": int(l[i]),
                "segmentation": rle,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s[i]),
            })

    if not dt_list:
        return 0.0

    import io, contextlib as _cl
    coco_dt = coco_gt.loadRes(dt_list)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.params.iouThrs = np.array([0.5])
    with _cl.redirect_stdout(io.StringIO()):  # suppress COCO verbose output
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


# ---------------------------------------------------------------------------
# Plot helpers (called at end of training)
# ---------------------------------------------------------------------------
def plot_training_curves(train_losses, val_losses, ap50_ep, ap50_vals,
                         output_dir=".", backbone="resnet50"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _smooth(arr, w=3):
        a = np.array(arr, dtype=float)
        if len(a) < w:
            return a
        kernel = np.ones(w) / w
        padded = np.pad(a, (w // 2, w // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:len(a)]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "legend.fontsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 150,
    })

    n = len(train_losses)
    ep = np.arange(1, n + 1)
    bb_label = backbone.replace("resnet", "ResNet-")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Mask R-CNN ({bb_label}) — Training Curves ({n} Epochs)", fontsize=15, y=1.01)

    ax = axes[0]
    ax.plot(ep, _smooth(train_losses), label="Train Loss", color="#2196F3", linewidth=2)
    ax.plot(ep, _smooth(val_losses),   label="Val Loss",   color="#FF5722", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.set_xlim(0, n)
    ax.set_ylim(0, 3.0)

    ax = axes[1]
    if ap50_ep:
        ax.plot(ap50_ep, ap50_vals, marker="o", color="#4CAF50",
                linewidth=2.5, markersize=7, label="Val AP50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP50")
    ax.set_title("Validation AP50")
    ax.legend(loc="lower right")
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0.0, 0.65)

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        print("  WARNING: --amp ignored (CUDA not available)")

    # --- Datasets ---
    train_ds_aug = CellDataset(args.data_dir, transforms=get_transform(train=True))
    train_ds_clean = CellDataset(args.data_dir, transforms=get_transform(train=False))

    n_total = len(train_ds_aug)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_set = torch.utils.data.Subset(train_ds_aug, train_idx)
    val_set = torch.utils.data.Subset(train_ds_clean, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    # --- Model ---
    model = get_model(pretrained=True, backbone=args.backbone)
    model.to(device)
    print(f"Trainable params: {count_parameters(model):,}")

    # Differential LR: pretrained backbone/FPN/RPN lower, new heads higher
    pretrained_params, head_params = get_param_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": pretrained_params, "lr": args.lr * 0.1},
            {"params": head_params,       "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # OneCycleLR handles warmup + annealing in one scheduler;
    # pct_start=0.1 means 10% of training is warmup
    steps_per_epoch = len(train_loader) // max(args.accum_steps, 1)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr],
        total_steps=total_steps,
        pct_start=0.1,
    )

    scaler = torch.amp.GradScaler() if use_amp else None

    start_epoch = 0
    best_val_loss = float("inf")
    best_ap50 = 0.0
    train_loss_history: list = []
    val_loss_history:   list = []
    ap50_ep_history:    list = []
    ap50_score_history: list = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_ap50 = ckpt.get("best_ap50", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}")

    # Dataset without augmentation for AP50 eval (same as val_set but accessed by index)
    val_ds_clean = CellDataset(args.data_dir, transforms=get_transform(train=False))

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            scaler=scaler, accum_steps=args.accum_steps,
            scheduler=scheduler,
        )
        val_loss = validate_loss(model, val_loader, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        lr_head = optimizer.param_groups[1]["lr"]
        lr_bb = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} "
              f"lr_head={lr_head:.6f} lr_bb={lr_bb:.6f} t={elapsed:.0f}s")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_ap50": best_ap50,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }
        if scaler:
            ckpt["scaler"] = scaler.state_dict()

        torch.save(ckpt, os.path.join(args.output_dir, "last_model.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, "best_loss_model.pth"))
            print(f"  => Best val loss: {best_val_loss:.4f} — saved best_loss_model.pth")

        # Periodic AP50 evaluation (the actual target metric)
        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1):
            ap50 = compute_val_ap50(model, val_ds_clean, val_idx, device)
            print(f"  => Val AP50: {ap50:.4f}")
            ap50_ep_history.append(epoch + 1)
            ap50_score_history.append(ap50)
            if ap50 > best_ap50:
                best_ap50 = ap50
                ckpt["best_ap50"] = best_ap50
                torch.save(ckpt, os.path.join(args.output_dir, "best_model.pth"))
                print(f"  => New best AP50: {best_ap50:.4f} — saved best_model.pth")

    print(f"\nDone. Best val loss: {best_val_loss:.4f} | Best AP50: {best_ap50:.4f}")
    if args.eval_every == 0:
        print("  (AP50 eval was disabled; best_model.pth not saved — use best_loss_model.pth)")

    plot_training_curves(train_loss_history, val_loss_history,
                         ap50_ep_history, ap50_score_history,
                         backbone=args.backbone)


if __name__ == "__main__":
    main()
