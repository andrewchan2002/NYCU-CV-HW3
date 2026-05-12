"""Dataset and data augmentation for medical cell instance segmentation."""

import os
import random

import cv2
import numpy as np
import tifffile
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class CellDataset(Dataset):
    """Medical cell instance segmentation dataset.

    Each sample folder contains:
      image.tif       - RGB image
      class1.tif, ... - instance masks (pixel value = instance ID, 0 = bg)
    """

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        img_dir = os.path.join(self.root_dir, self.image_dirs[idx])
        img = Image.open(os.path.join(img_dir, "image.tif")).convert("RGB")

        boxes, masks, labels = [], [], []

        for class_id in range(1, 5):
            mask_path = os.path.join(img_dir, f"class{class_id}.tif")
            if not os.path.exists(mask_path):
                continue

            mask = tifffile.imread(mask_path).astype(np.int32)
            for inst_id in np.unique(mask):
                if inst_id == 0:
                    continue
                binary_mask = (mask == inst_id).astype(np.uint8)
                rows, cols = np.where(binary_mask)
                if rows.size == 0:
                    continue
                ymin, ymax = int(rows.min()), int(rows.max())
                xmin, xmax = int(cols.min()), int(cols.max())
                if xmax <= xmin or ymax <= ymin:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                masks.append(binary_mask)
                labels.append(class_id)

        img_tensor = TF.to_tensor(img)
        h, w = img_tensor.shape[1], img_tensor.shape[2]

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            masks_t = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            iscrowd = torch.zeros(len(labels), dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            labels_t = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "masks": masks_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, _, w = image.shape
            image = TF.hflip(image)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h, _ = image.shape
            image = TF.vflip(image)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
            if target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-2)
        return image, target


class RandomPhotometricDistort:
    """Random brightness, contrast, saturation, hue jitter."""

    def __init__(self, brightness=(0.7, 1.3), contrast=(0.7, 1.3),
                 saturation=(0.7, 1.3), hue=(-0.15, 0.15)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(*self.brightness))
        if random.random() < 0.5:
            image = TF.adjust_contrast(image, random.uniform(*self.contrast))
        if random.random() < 0.5:
            image = TF.adjust_saturation(image, random.uniform(*self.saturation))
        if random.random() < 0.5:
            image = TF.adjust_hue(image, random.uniform(*self.hue))
        return image, target


class RandomAffine:
    """Random rotation + translation applied jointly to image and masks.

    Boxes are recomputed from warped masks so they stay tight.
    Masks that disappear entirely after warping are dropped.
    """

    def __init__(self, degrees=10, translate=(0.05, 0.05), prob=0.3):
        self.degrees = degrees
        self.translate = translate
        self.prob = prob

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        _, h, w = image.shape
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate[0], self.translate[0]) * w
        ty = random.uniform(-self.translate[1], self.translate[1]) * h

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty

        # Warp image: tensor (C,H,W) float[0,1] → numpy uint8 → warp → back
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.warpAffine(img_np, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        image = TF.to_tensor(img_np)

        if target["masks"].numel() == 0:
            return image, target

        orig_labels = target["labels"]
        orig_iscrowd = target["iscrowd"]
        keep_masks, keep_boxes, keep_idx = [], [], []

        for i, mask_np in enumerate(target["masks"].numpy()):
            warped = cv2.warpAffine(mask_np.astype(np.uint8), M, (w, h),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT)
            ys, xs = np.where(warped > 0)
            if ys.size == 0:
                continue
            keep_idx.append(i)
            keep_masks.append(warped)
            keep_boxes.append([int(xs.min()), int(ys.min()),
                                int(xs.max()), int(ys.max())])

        if keep_masks:
            boxes_t = torch.as_tensor(keep_boxes, dtype=torch.float32)
            target["masks"] = torch.as_tensor(np.stack(keep_masks), dtype=torch.uint8)
            target["boxes"] = boxes_t
            target["labels"] = orig_labels[keep_idx]
            target["iscrowd"] = orig_iscrowd[keep_idx]
            target["area"] = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["area"] = torch.zeros(0, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(0, dtype=torch.int64)

        return image, target


class RandomGaussianNoise:
    """Add random Gaussian noise to the image (does not affect targets)."""

    def __init__(self, sigma_range=(2, 8), prob=0.3):
        self.sigma_range = sigma_range
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            sigma = random.uniform(*self.sigma_range) / 255.0
            noise = torch.randn_like(image) * sigma
            image = (image + noise).clamp(0.0, 1.0)
        return image, target


class RandomElasticDeformation:
    """Elastic deformation simulating realistic cell shape variation.

    Applies a smooth random displacement field to both image and masks,
    preserving spatial correspondence. Masks that disappear after warping
    are dropped.
    """

    def __init__(self, alpha=10.0, sigma=5.0, prob=0.3):
        self.alpha = alpha  # displacement magnitude in pixels
        self.sigma = sigma  # Gaussian smoothing of displacement field
        self.prob = prob

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        _, h, w = image.shape
        ksize = int(6 * self.sigma) | 1  # nearest odd kernel size

        dx = np.random.uniform(-self.alpha, self.alpha, (h, w)).astype(np.float32)
        dy = np.random.uniform(-self.alpha, self.alpha, (h, w)).astype(np.float32)
        dx = cv2.GaussianBlur(dx, (ksize, ksize), self.sigma)
        dy = cv2.GaussianBlur(dy, (ksize, ksize), self.sigma)

        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        map_x = np.clip(grid_x + dx, 0, w - 1)
        map_y = np.clip(grid_y + dy, 0, h - 1)

        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.remap(img_np, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        image = TF.to_tensor(img_np)

        if target["masks"].numel() == 0:
            return image, target

        orig_labels = target["labels"]
        orig_iscrowd = target["iscrowd"]
        keep_masks, keep_boxes, keep_idx = [], [], []

        for i, mask_np in enumerate(target["masks"].numpy()):
            warped = cv2.remap(mask_np.astype(np.uint8), map_x, map_y,
                               interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT)
            ys_nz, xs_nz = np.where(warped > 0)
            if ys_nz.size == 0:
                continue
            keep_idx.append(i)
            keep_masks.append(warped)
            keep_boxes.append([int(xs_nz.min()), int(ys_nz.min()),
                                int(xs_nz.max()), int(ys_nz.max())])

        if keep_masks:
            boxes_t = torch.as_tensor(keep_boxes, dtype=torch.float32)
            target["masks"] = torch.as_tensor(np.stack(keep_masks), dtype=torch.uint8)
            target["boxes"] = boxes_t
            target["labels"] = orig_labels[keep_idx]
            target["iscrowd"] = orig_iscrowd[keep_idx]
            target["area"] = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["area"] = torch.zeros(0, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(0, dtype=torch.int64)

        return image, target


class RandomRotation90:
    """Random 90/180/270-degree rotation applied jointly to image and masks."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        k = random.randint(1, 3)  # number of 90-degree CCW rotations

        # torch.rot90 operates on the last two dims for (C, H, W)
        image = torch.rot90(image, k, dims=[1, 2])
        _, h, w = image.shape  # new dimensions after rotation

        if target["masks"].numel() > 0:
            target["masks"] = torch.rot90(target["masks"], k, dims=[1, 2])

            if target["boxes"].numel() > 0:
                # Recompute boxes from rotated masks (simpler than coordinate math for 90-deg)
                new_boxes = []
                for mask in target["masks"].numpy():
                    ys, xs = np.where(mask > 0)
                    if ys.size == 0:
                        new_boxes.append([0, 0, 1, 1])
                    else:
                        new_boxes.append([int(xs.min()), int(ys.min()),
                                          int(xs.max()), int(ys.max())])
                boxes_t = torch.as_tensor(new_boxes, dtype=torch.float32)
                target["boxes"] = boxes_t
                target["area"] = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

        return image, target


def get_transform(train=True):
    transforms = []
    if train:
        transforms += [
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            RandomRotation90(0.5),
            RandomPhotometricDistort(),
            RandomAffine(degrees=10, translate=(0.05, 0.05), prob=0.3),
            RandomGaussianNoise(sigma_range=(2, 8), prob=0.3),
            RandomElasticDeformation(alpha=10.0, sigma=5.0, prob=0.3),
        ]
    return Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))
