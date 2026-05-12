"""Mask R-CNN model for medical cell instance segmentation."""

from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 5  # 0=background, 1=class1, 2=class2, 3=class3, 4=class4

_RPN_KWARGS = dict(
    box_detections_per_img=500,
    box_score_thresh=0.05,
    box_nms_thresh=0.5,
    rpn_pre_nms_top_n_train=2000,
    rpn_post_nms_top_n_train=2000,
    rpn_pre_nms_top_n_test=2000,
    rpn_post_nms_top_n_test=2000,
)


def get_model(num_classes=NUM_CLASSES, pretrained=True, backbone="resnet50"):
    """Build Mask R-CNN with FPN backbone.

    backbone: "resnet50" (FPN V2, COCO pretrained), "resnet101", or "resnet152"
              (FPN, ImageNet pretrained backbone only)
    """
    if backbone == "resnet50":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=5,
            **_RPN_KWARGS,
        )
        # COCO weights use 91 classes → replace heads for num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels_mask, dim_reduced=256, num_classes=num_classes
        )
    elif backbone in ("resnet101", "resnet152"):
        from torchvision.models import ResNet101_Weights, ResNet152_Weights
        from torchvision.models.detection import MaskRCNN
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

        bb_weights = {
            "resnet101": ResNet101_Weights.IMAGENET1K_V2,
            "resnet152": ResNet152_Weights.IMAGENET1K_V2,
        }[backbone] if pretrained else None
        bb = resnet_fpn_backbone(backbone, weights=bb_weights, trainable_layers=5)
        # MaskRCNN initialises heads for num_classes directly; no replacement needed
        model = MaskRCNN(bb, num_classes=num_classes, **_RPN_KWARGS)
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose: resnet50, resnet101, resnet152"
        )

    model.rpn.nms_thresh = 0.7
    return model


def get_param_groups(model):
    """Split parameters: backbone/FPN/RPN at lower LR, new heads at full LR."""
    pretrained_params, head_params = [], []
    new_modules = {model.roi_heads.box_predictor, model.roi_heads.mask_predictor}

    for param in model.parameters():
        if not param.requires_grad:
            continue
        is_new = any(param is p for m in new_modules for p in m.parameters())
        (head_params if is_new else pretrained_params).append(param)

    return pretrained_params, head_params


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
