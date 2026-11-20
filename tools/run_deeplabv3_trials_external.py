"""
Standalone DeepLabV3 fine-tuning trial runner.

This script mirrors the DeepLabV3 part of the project notebook, but is easier
to run on another PC/GPU. It trains selected trials, saves loss logs, evaluates
IoU on the selected COCO val2017 records, and writes summary CSV files.

Expected project layout:
  data/coco/annotations/instances_train2017.json
  data/coco/annotations/instances_val2017.json
  data/coco/train2017/*.jpg
  data/coco/val2017/*.jpg
  data/selected/train_2000_deeplabv3.json
  data/selected/train_5000_deeplabv3.json
  data/selected/train_10000_deeplabv3.json
  data/selected/train_50000_deeplabv3.json
  data/selected/val_1000_primary_objects.json

Example:
  python tools/run_deeplabv3_trials_external.py \
    --coco-root data/coco \
    --selected-dir data/selected \
    --output-dir external_deeplabv3_trials \
    --trials A B C D \
    --download-missing-images \
    --workers 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, random_split
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

SEED = 42
IGNORE_INDEX = 255

VOC_CATEGORIES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
VOC_NAME_TO_INDEX = {name: idx for idx, name in enumerate(VOC_CATEGORIES)}
COCO_TO_VOC_NAME = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorbike",
    "airplane": "aeroplane",
    "bus": "bus",
    "train": "train",
    "boat": "boat",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "bottle": "bottle",
    "chair": "chair",
    "couch": "sofa",
    "potted plant": "pottedplant",
    "dining table": "diningtable",
    "tv": "tvmonitor",
}
COCO_TO_VOC_INDEX = {
    coco_name: VOC_NAME_TO_INDEX[voc_name] for coco_name, voc_name in COCO_TO_VOC_NAME.items()
}

DEEPLAB_TRAINING_TRIALS = [
    {
        "trial": "A",
        "name": "head_only_train2000_epoch1",
        "record_file": "train_2000_deeplabv3.json",
        "records": 2000,
        "epochs": 1,
        "image_size": 320,
        "batch_size": 4,
        "mode": "head_only",
        "loss": "cross_entropy",
        "lr_head": 1e-3,
        "lr_backbone": None,
        "gradient_accumulation_steps": 1,
        "frozen_batch_norm": False,
    },
    {
        "trial": "B",
        "name": "layer4_balanced_train2000_epoch3",
        "record_file": "train_2000_deeplabv3.json",
        "records": 2000,
        "epochs": 3,
        "image_size": 384,
        "batch_size": 2,
        "mode": "layer4_plus_head",
        "loss": "weighted_ce_plus_dice",
        "lr_head": 2e-4,
        "lr_backbone": 5e-5,
        "gradient_accumulation_steps": 1,
        "frozen_batch_norm": False,
    },
    {
        "trial": "C",
        "name": "primary_object_resized_train5000",
        "record_file": "train_5000_deeplabv3.json",
        "records": 5000,
        "epochs": 3,
        "image_size": 384,
        "batch_size": 2,
        "mode": "staged_head_then_layer4",
        "loss": "weighted_ce_plus_dice",
        "lr_head": 2e-4,
        "lr_backbone": 5e-5,
        "gradient_accumulation_steps": 1,
        "frozen_batch_norm": False,
    },
    {
        "trial": "D",
        "name": "staged_train10000_gentle_model_soup",
        "record_file": "train_10000_deeplabv3.json",
        "records": 10000,
        "epochs": "staged",
        "image_size": 384,
        "batch_size": 2,
        "mode": "staged_head_then_layer4",
        "loss": "weighted_ce_plus_dice",
        "lr_head": 1e-4,
        "lr_backbone": 2e-5,
        "gradient_accumulation_steps": 1,
        "frozen_batch_norm": False,
    },
    {
        "trial": "E",
        "name": "staged_train50000_frozenbn_accum2_model_soup",
        "record_file": "train_50000_deeplabv3.json",
        "records": 50000,
        "epochs": "staged",
        "image_size": 384,
        "batch_size": 1,
        "mode": "staged_head_then_layer4",
        "loss": "weighted_ce_plus_dice",
        "lr_head": 1e-4,
        "lr_backbone": 2e-5,
        "gradient_accumulation_steps": 2,
        "frozen_batch_norm": True,
    },
]
TRIAL_BY_NAME = {row["trial"]: row for row in DEEPLAB_TRAINING_TRIALS}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_coco_indices(coco: dict[str, Any]) -> dict[str, Any]:
    images = {int(img["id"]): img for img in coco.get("images", [])}
    categories = {int(cat["id"]): cat for cat in coco.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    annotations_by_id: dict[int, dict[str, Any]] = {}
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        annotations_by_image.setdefault(image_id, []).append(ann)
        annotations_by_id[int(ann["id"])] = ann
    return {
        "images": images,
        "categories": categories,
        "annotations_by_image": annotations_by_image,
        "annotations_by_id": annotations_by_id,
    }


def annotation_category_name(annotation: dict[str, Any], categories: dict[int, dict[str, Any]]) -> str:
    return categories[int(annotation["category_id"])]["name"]


def annotation_to_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


def binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def build_semantic_mask_for_image(
    annotations: list[dict[str, Any]],
    height: int,
    width: int,
    categories: dict[int, dict[str, Any]],
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    sorted_annotations = sorted(annotations, key=lambda ann: float(ann.get("area", 0.0)), reverse=True)

    for ann in sorted_annotations:
        if ann.get("iscrowd", 0):
            continue
        name = annotation_category_name(ann, categories)
        if name not in COCO_TO_VOC_INDEX:
            ann_mask = annotation_to_mask(ann, height, width)
            mask[ann_mask] = ignore_index

    for ann in sorted_annotations:
        if ann.get("iscrowd", 0):
            continue
        name = annotation_category_name(ann, categories)
        label = COCO_TO_VOC_INDEX.get(name)
        if label is None:
            continue
        ann_mask = annotation_to_mask(ann, height, width)
        mask[ann_mask] = int(label)
    return mask


class CocoDeepLabDataset:
    def __init__(
        self,
        records_json: Path,
        annotations_json: Path,
        images_dir: Path,
        image_size: int = 384,
        augment: bool = False,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.records = load_json(records_json)
        self.coco = load_json(annotations_json)
        self.indices = build_coco_indices(self.coco)
        self.images_dir = Path(images_dir)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.ignore_index = int(ignore_index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        record = self.records[idx]
        image_path = self.images_dir / record["file_name"]
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        annotations = self.indices["annotations_by_image"].get(int(record["image_id"]), [])
        mask = build_semantic_mask_for_image(
            annotations,
            height=height,
            width=width,
            categories=self.indices["categories"],
            ignore_index=self.ignore_index,
        )

        image = np.array(
            Image.fromarray(image).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        )
        mask = np.array(
            Image.fromarray(mask).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        )

        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor, record


def deeplab_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    images, masks, records = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0), list(records)


def build_deeplabv3(pretrained: bool = True) -> torch.nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    return deeplabv3_resnet50(weights=weights)


def freeze_for_mode(model: torch.nn.Module, mode: str) -> torch.nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    if mode == "head_only":
        modules = [model.classifier]
        if getattr(model, "aux_classifier", None) is not None:
            modules.append(model.aux_classifier)
    elif mode == "layer4_plus_head":
        modules = [model.backbone.layer4, model.classifier]
        if getattr(model, "aux_classifier", None) is not None:
            modules.append(model.aux_classifier)
    elif mode == "all":
        modules = [model]
    else:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
    return model


def freeze_batch_norm(model: torch.nn.Module) -> torch.nn.Module:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    return model


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return int(total), int(trainable)


def foreground_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_clamped = target.clone()
    target_clamped[~valid] = 0
    one_hot = F.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probs = probs * valid
    one_hot = one_hot * valid
    foreground = torch.arange(num_classes, device=logits.device) > 0
    probs_fg = probs[:, foreground]
    one_hot_fg = one_hot[:, foreground]
    intersection = (probs_fg * one_hot_fg).sum(dim=(0, 2, 3))
    union = probs_fg.sum(dim=(0, 2, 3)) + one_hot_fg.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    present = one_hot_fg.sum(dim=(0, 2, 3)) > 0
    if present.sum() == 0:
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


def make_class_weights(background_weight: float = 0.15) -> torch.Tensor:
    weights = torch.ones(len(VOC_CATEGORIES), dtype=torch.float32)
    weights[0] = float(background_weight)
    return weights


def build_optimizer_for_mode(
    model: torch.nn.Module,
    mode: str,
    lr_head: float,
    lr_backbone: float | None,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if mode == "head_only":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr_head,
            weight_decay=weight_decay,
        )
    if mode == "layer4_plus_head":
        backbone_params = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
        head_params = list(model.classifier.parameters())
        if getattr(model, "aux_classifier", None) is not None:
            head_params += list(model.aux_classifier.parameters())
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_backbone if lr_backbone is not None else lr_head},
                {"params": head_params, "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_head,
        weight_decay=weight_decay,
    )


def build_stage_plan(mode: str, epochs: int | str) -> list[dict[str, Any]]:
    if mode != "staged_head_then_layer4":
        return [{"stage": mode, "mode": mode, "epochs": int(epochs)}]
    total_epochs = 2 if epochs == "staged" else int(epochs)
    head_epochs = max(1, total_epochs // 2)
    layer4_epochs = max(1, total_epochs - head_epochs)
    return [
        {"stage": "head_warmup", "mode": "head_only", "epochs": head_epochs},
        {"stage": "layer4_adaptation", "mode": "layer4_plus_head", "epochs": layer4_epochs},
    ]


def compute_deeplab_training_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    ce_loss: torch.nn.Module,
    loss_name: str,
    dice_weight: float,
) -> torch.Tensor:
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    loss = ce_loss(logits, masks)
    if loss_name == "weighted_ce_plus_dice":
        loss = loss + dice_weight * foreground_dice_loss(logits, masks)
    return loss


def evaluate_training_validation_loss(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ce_loss: torch.nn.Module,
    loss_name: str,
    dice_weight: float,
    frozen_batch_norm: bool = False,
) -> float:
    if frozen_batch_norm:
        freeze_batch_norm(model)
    model.eval()
    running_loss = 0.0
    item_count = 0
    with torch.inference_mode():
        for images, masks, _record in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)["out"]
            loss = compute_deeplab_training_loss(logits, masks, ce_loss, loss_name, dice_weight)
            running_loss += float(loss.detach().cpu()) * images.shape[0]
            item_count += images.shape[0]
    return running_loss / max(1, item_count)


def train_deeplab_one_experiment(
    trial_config: dict[str, Any],
    records_json: Path,
    annotations_json: Path,
    images_dir: Path,
    output_checkpoint: Path,
    log_path: Path,
    device: torch.device,
    workers: int,
    weight_decay: float = 1e-4,
    background_weight: float = 0.15,
    dice_weight: float = 0.5,
) -> pd.DataFrame:
    dataset = CocoDeepLabDataset(
        records_json,
        annotations_json,
        images_dir,
        image_size=trial_config["image_size"],
        augment=True,
    )
    val_size = max(1, min(100, int(0.05 * len(dataset))))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=trial_config["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=deeplab_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=trial_config["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=deeplab_collate_fn,
    )

    model = build_deeplabv3(pretrained=True).to(device)
    ce_weights = make_class_weights(background_weight).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        weight=ce_weights if trial_config["loss"] == "weighted_ce_plus_dice" else None,
    )

    stage_plan = build_stage_plan(trial_config["mode"], trial_config["epochs"])
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_rows = []
    best_val_loss = float("inf")
    global_epoch = 0
    trial = trial_config["trial"]
    print(f"\n=== Trial {trial}: {trial_config['name']} ===")
    print(f"records={len(dataset)} image_size={trial_config['image_size']} batch_size={trial_config['batch_size']}")
    print(f"loss={trial_config['loss']} lr_head={trial_config['lr_head']} lr_backbone={trial_config['lr_backbone']}")
    print(f"train split={train_size} hold-out split={val_size}")

    for stage in stage_plan:
        active_mode = stage["mode"]
        stage_name = stage.get("stage", active_mode)
        model = freeze_for_mode(model, active_mode)
        if trial_config["frozen_batch_norm"]:
            model = freeze_batch_norm(model)
        total_params, trainable_params = count_parameters(model)
        optimizer = build_optimizer_for_mode(
            model,
            active_mode,
            trial_config["lr_head"],
            trial_config["lr_backbone"],
            weight_decay,
        )

        for stage_epoch in range(1, int(stage["epochs"]) + 1):
            global_epoch += 1
            model.train()
            if trial_config["frozen_batch_norm"]:
                freeze_batch_norm(model)
            running_loss = 0.0
            item_count = 0
            optimizer.zero_grad(set_to_none=True)
            epoch_start = time.perf_counter()

            for step, (images, masks, _record) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(images)["out"]
                loss = compute_deeplab_training_loss(
                    logits,
                    masks,
                    ce_loss,
                    trial_config["loss"],
                    dice_weight,
                )
                accum_steps = int(trial_config["gradient_accumulation_steps"])
                (loss / accum_steps).backward()
                if step % accum_steps == 0 or step == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.detach().cpu()) * images.shape[0]
                item_count += images.shape[0]

            train_loss = running_loss / max(1, item_count)
            val_loss = evaluate_training_validation_loss(
                model,
                val_loader,
                device,
                ce_loss,
                trial_config["loss"],
                dice_weight,
                frozen_batch_norm=trial_config["frozen_batch_norm"],
            )
            epoch_seconds = time.perf_counter() - epoch_start
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "trial": trial,
                        "trial_name": trial_config["name"],
                        "epoch": global_epoch,
                        "stage": stage_name,
                        "stage_epoch": stage_epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "num_classes": len(VOC_CATEGORIES),
                        "classes": VOC_CATEGORIES,
                        "record_file": str(records_json),
                        "image_size": trial_config["image_size"],
                    },
                    output_checkpoint,
                )

            row = {
                "trial": trial,
                "trial_name": trial_config["name"],
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": stage_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "is_best_checkpoint": is_best,
                "epoch_seconds": epoch_seconds,
                "batch_size": trial_config["batch_size"],
                "gradient_accumulation_steps": trial_config["gradient_accumulation_steps"],
                "effective_batch_size": trial_config["batch_size"] * trial_config["gradient_accumulation_steps"],
                "image_size": trial_config["image_size"],
                "mode": active_mode,
                "loss": trial_config["loss"],
                "lr_head": trial_config["lr_head"],
                "lr_backbone": trial_config["lr_backbone"],
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "checkpoint": str(output_checkpoint),
            }
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(log_path, index=False)
            print(
                f"trial={trial} epoch={global_epoch} stage={stage_name} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"best={best_val_loss:.4f} time={epoch_seconds/60:.1f} min"
            )

    return pd.DataFrame(log_rows)


def load_deeplab_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    return model


def preprocess_for_deeplab(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()


def run_deeplab_object_iou_eval(
    records_json: Path,
    annotations_json: Path,
    images_dir: Path,
    output_csv: Path,
    device: torch.device,
    checkpoint: Path | None = None,
    model_name: str = "DeepLabV3_pretrained",
    limit: int | None = None,
) -> pd.DataFrame:
    records = load_json(records_json)
    if limit is not None:
        records = records[:limit]
    coco = load_json(annotations_json)
    indices = build_coco_indices(coco)

    model = build_deeplabv3(pretrained=True)
    if checkpoint is not None:
        model = load_deeplab_checkpoint_into_model(model, checkpoint, device)
    model.to(device)
    model.eval()

    rows = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {model_name} on {len(records)} records...")
    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            image_path = images_dir / record["file_name"]
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
            height, width = image_rgb.shape[:2]
            annotation = indices["annotations_by_id"][int(record["annotation_id"])]
            gt_mask = annotation_to_mask(annotation, height, width)
            target_label = COCO_TO_VOC_INDEX.get(record["category_name"])

            start = time.perf_counter()
            tensor = preprocess_for_deeplab(image_rgb).to(device)
            logits = model(tensor)["out"]
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            pred_label_map = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            elapsed = time.perf_counter() - start

            if target_label is None:
                iou = np.nan
                pred_pixels = np.nan
            else:
                pred_mask = pred_label_map == int(target_label)
                iou = binary_iou(pred_mask, gt_mask)
                pred_pixels = int(pred_mask.sum())

            rows.append(
                {
                    "model": model_name,
                    "image_id": int(record["image_id"]),
                    "category_name": record["category_name"],
                    "supported_deeplab_class": target_label is not None,
                    "target_label": target_label,
                    "iou": iou,
                    "runtime_mean_seconds": elapsed,
                    "gt_pixels": int(gt_mask.sum()),
                    "pred_pixels": pred_pixels,
                }
            )
            if index % 100 == 0 or index == len(records):
                pd.DataFrame(rows).to_csv(output_csv, index=False)
                print(f"  {index}/{len(records)}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def summarize_object_iou(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "records_total": len(df),
        "records_with_iou": int(df["iou"].notna().sum()),
        "mean_iou": float(df["iou"].mean()),
        "median_iou": float(df["iou"].median()),
        "std_iou": float(df["iou"].std()),
        "mean_time_seconds": float(df["runtime_mean_seconds"].mean()),
        "median_time_seconds": float(df["runtime_mean_seconds"].median()),
        "total_time_seconds": float(df["runtime_mean_seconds"].sum()),
    }


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    return checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def create_deeplab_model_soup_checkpoint(
    finetuned_checkpoint_path: Path,
    output_checkpoint_path: Path,
    alpha: float,
) -> Path:
    pretrained_state = build_deeplabv3(pretrained=True).state_dict()
    fine_checkpoint = torch.load(finetuned_checkpoint_path, map_location="cpu")
    fine_state = extract_state_dict(fine_checkpoint)
    mixed = {}
    for key, pre_value in pretrained_state.items():
        fine_value = fine_state.get(key)
        if fine_value is None:
            mixed[key] = pre_value
        elif hasattr(pre_value, "dtype") and str(pre_value.dtype).startswith("torch.float"):
            mixed[key] = (1.0 - float(alpha)) * pre_value + float(alpha) * fine_value
        else:
            mixed[key] = fine_value
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": mixed,
            "alpha": float(alpha),
            "source_finetuned_checkpoint": str(finetuned_checkpoint_path),
        },
        output_checkpoint_path,
    )
    return output_checkpoint_path


def plot_loss_curves(loss_logs: list[Path], output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping loss plot because matplotlib is unavailable: {exc}")
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for log_path in loss_logs:
        if not log_path.exists():
            continue
        df = pd.read_csv(log_path)
        if df.empty:
            continue
        label = str(df["trial"].iloc[0])
        axes[0].plot(df["epoch"], df["train_loss"], marker="o", label=label)
        axes[1].plot(df["epoch"], df["val_loss"], marker="o", label=label)
    axes[0].set_title("Training Loss")
    axes[1].set_title("Train Hold-Out Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Trial")
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trial_iou(summary_csv: Path, output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping IoU plot because matplotlib is unavailable: {exc}")
        return
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    output_png.parent.mkdir(parents=True, exist_ok=True)
    labels = df["trial"].fillna(df["model"]).astype(str)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(labels, df["mean_iou"], color="#4C78A8")
    ax.set_ylim(0, max(0.75, math.ceil(float(df["mean_iou"].max()) * 10) / 10))
    ax.set_ylabel("Mean IoU on val1000")
    ax.set_xlabel("DeepLabV3 Trial")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def collect_required_image_names(selected_dir: Path, trials: list[str], eval_records: Path) -> tuple[set[str], set[str]]:
    train_names: set[str] = set()
    for trial in trials:
        config = TRIAL_BY_NAME[trial]
        for row in load_json(selected_dir / config["record_file"]):
            train_names.add(row["file_name"])
    val_names = {row["file_name"] for row in load_json(eval_records)}
    return train_names, val_names


def download_one(split: str, file_name: str, destination: Path, base_url: str) -> tuple[str, bool, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return file_name, True, "exists"
    url = f"{base_url.rstrip('/')}/{split}/{file_name}"
    try:
        urllib.request.urlretrieve(url, destination)
        return file_name, True, "downloaded"
    except Exception as exc:
        if destination.exists() and destination.stat().st_size == 0:
            destination.unlink()
        return file_name, False, str(exc)


def ensure_images(
    file_names: set[str],
    split: str,
    image_dir: Path,
    download_missing: bool,
    download_workers: int,
    coco_image_base_url: str,
) -> None:
    missing = sorted(name for name in file_names if not (image_dir / name).exists())
    if not missing:
        print(f"{split}: all required images are available ({len(file_names)} files).")
        return
    if not download_missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{split}: {len(missing)} images are missing in {image_dir}. "
            f"First missing files: {preview}. Add --download-missing-images to fetch them."
        )

    print(f"{split}: downloading {len(missing)} missing images into {image_dir}")
    failures = []
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = [
            pool.submit(download_one, split, name, image_dir / name, coco_image_base_url)
            for name in missing
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            name, ok, message = future.result()
            if not ok:
                failures.append((name, message))
            if idx % 500 == 0 or idx == len(futures):
                print(f"  download progress: {idx}/{len(futures)}")
    if failures:
        example = "; ".join(f"{name}: {msg}" for name, msg in failures[:5])
        raise RuntimeError(f"Image download failed for {len(failures)} files. Examples: {example}")


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external DeepLabV3 fine-tuning trials.")
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--selected-dir", type=Path, default=Path("data/selected"))
    parser.add_argument("--output-dir", type=Path, default=Path("external_deeplabv3_trials"))
    parser.add_argument(
        "--trials",
        nargs="+",
        choices=sorted(TRIAL_BY_NAME),
        default=["A", "B", "C", "D"],
        help="Trials to run. Default A-D because E is the large 50k final run.",
    )
    parser.add_argument("--eval-records", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--download-missing-images", action="store_true")
    parser.add_argument("--coco-image-base-url", default="http://images.cocodataset.org")
    parser.add_argument("--skip-training", action="store_true", help="Use existing checkpoints in output-dir.")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Retrain even if a checkpoint already exists.")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-pretrained", action="store_true")
    parser.add_argument("--model-soup-alpha", type=float, default=0.5)
    parser.add_argument("--soup-trials", nargs="+", default=["E"], choices=sorted(TRIAL_BY_NAME))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    coco_root = args.coco_root
    train_images = coco_root / "train2017"
    val_images = coco_root / "val2017"
    train_ann = coco_root / "annotations" / "instances_train2017.json"
    val_ann = coco_root / "annotations" / "instances_val2017.json"
    eval_records = args.eval_records or (args.selected_dir / "val_1000_primary_objects.json")

    required_paths = [train_ann, val_ann, eval_records]
    for trial in args.trials:
        required_paths.append(args.selected_dir / TRIAL_BY_NAME[trial]["record_file"])
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing_paths))

    train_images.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    train_names, val_names = collect_required_image_names(args.selected_dir, args.trials, eval_records)
    ensure_images(
        train_names,
        "train2017",
        train_images,
        args.download_missing_images,
        args.download_workers,
        args.coco_image_base_url,
    )
    ensure_images(
        val_names,
        "val2017",
        val_images,
        args.download_missing_images,
        args.download_workers,
        args.coco_image_base_url,
    )

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Trials: {', '.join(args.trials)}")

    trial_summary_rows = []
    loss_log_paths = []

    if args.eval_pretrained and not args.skip_evaluation:
        baseline_csv = tables_dir / "deeplabv3_pretrained_val1000_external.csv"
        baseline_df = run_deeplab_object_iou_eval(
            eval_records,
            val_ann,
            val_images,
            baseline_csv,
            device=device,
            checkpoint=None,
            model_name="DeepLabV3_pretrained",
            limit=args.eval_limit,
        )
        row = summarize_object_iou(baseline_df, "DeepLabV3_pretrained")
        row["trial"] = "pretrained"
        trial_summary_rows.append(row)

    for trial in args.trials:
        config = TRIAL_BY_NAME[trial]
        checkpoint_path = checkpoint_dir / f"deeplabv3_trial_{trial}_{config['name']}.pth"
        log_path = tables_dir / f"deeplabv3_trial_{trial}_{config['name']}_loss_log.csv"
        eval_csv = tables_dir / f"deeplabv3_trial_{trial}_{config['name']}_val1000.csv"
        loss_log_paths.append(log_path)

        if not args.skip_training:
            if checkpoint_path.exists() and not args.overwrite:
                print(f"Checkpoint already exists for trial {trial}; using it: {checkpoint_path}")
            else:
                train_deeplab_one_experiment(
                    config,
                    args.selected_dir / config["record_file"],
                    train_ann,
                    train_images,
                    checkpoint_path,
                    log_path,
                    device=device,
                    workers=args.workers,
                )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for trial {trial} does not exist: {checkpoint_path}. "
                "Run without --skip-training or point output-dir to existing checkpoints."
            )

        if not args.skip_evaluation:
            df = run_deeplab_object_iou_eval(
                eval_records,
                val_ann,
                val_images,
                eval_csv,
                device=device,
                checkpoint=checkpoint_path,
                model_name=f"DeepLabV3_trial_{trial}_{config['name']}",
                limit=args.eval_limit,
            )
            row = summarize_object_iou(df, f"DeepLabV3_trial_{trial}_{config['name']}")
            row["trial"] = trial
            row["checkpoint"] = str(checkpoint_path)
            row["loss_log"] = str(log_path)
            row["record_file"] = config["record_file"]
            row["epochs"] = config["epochs"]
            row["image_size"] = config["image_size"]
            row["batch_size"] = config["batch_size"]
            row["gradient_accumulation_steps"] = config["gradient_accumulation_steps"]
            row["effective_batch_size"] = config["batch_size"] * config["gradient_accumulation_steps"]
            row["mode"] = config["mode"]
            row["loss"] = config["loss"]
            row["lr_head"] = config["lr_head"]
            row["lr_backbone"] = config["lr_backbone"]
            trial_summary_rows.append(row)

        if trial in args.soup_trials and not args.skip_evaluation:
            soup_checkpoint = checkpoint_dir / (
                f"deeplabv3_trial_{trial}_{config['name']}_soup_alpha_{args.model_soup_alpha:.3f}.pth"
            )
            soup_csv = tables_dir / (
                f"deeplabv3_trial_{trial}_{config['name']}_soup_alpha_{args.model_soup_alpha:.3f}_val1000.csv"
            )
            create_deeplab_model_soup_checkpoint(checkpoint_path, soup_checkpoint, args.model_soup_alpha)
            soup_df = run_deeplab_object_iou_eval(
                eval_records,
                val_ann,
                val_images,
                soup_csv,
                device=device,
                checkpoint=soup_checkpoint,
                model_name=f"DeepLabV3_trial_{trial}_soup_alpha_{args.model_soup_alpha:.3f}",
                limit=args.eval_limit,
            )
            row = summarize_object_iou(soup_df, f"DeepLabV3_trial_{trial}_soup_alpha_{args.model_soup_alpha:.3f}")
            row["trial"] = f"{trial}_soup"
            row["checkpoint"] = str(soup_checkpoint)
            row["loss_log"] = str(log_path)
            row["record_file"] = config["record_file"]
            row["epochs"] = config["epochs"]
            row["image_size"] = config["image_size"]
            row["batch_size"] = config["batch_size"]
            row["gradient_accumulation_steps"] = config["gradient_accumulation_steps"]
            row["effective_batch_size"] = config["batch_size"] * config["gradient_accumulation_steps"]
            row["mode"] = f"{config['mode']} + model_soup_alpha_{args.model_soup_alpha:.3f}"
            row["loss"] = config["loss"]
            row["lr_head"] = config["lr_head"]
            row["lr_backbone"] = config["lr_backbone"]
            trial_summary_rows.append(row)

    if trial_summary_rows:
        summary_df = pd.DataFrame(trial_summary_rows)
        summary_path = tables_dir / "deeplabv3_trial_summary_val1000.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\nTrial summary:")
        cols = ["trial", "model", "records_with_iou", "mean_iou", "median_iou", "mean_time_seconds"]
        print(summary_df[cols].to_string(index=False))
        print(f"\nSaved summary: {summary_path}")
        plot_trial_iou(summary_path, figures_dir / "deeplabv3_trial_iou_val1000.png")

    plot_loss_curves(loss_log_paths, figures_dir / "deeplabv3_trial_loss_curves.png")
    print(f"Done. Outputs are in: {output_dir}")


if __name__ == "__main__":
    main()
