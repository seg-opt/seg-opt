"""Fit and visualize a small DINOv3 linear segmentation probe.

This is an intentionally small demonstration, not a replacement for a full
train/validation/test experiment.  It freezes DINOv3, caches dense features
from a deterministic subset of the lunar dataset, fits the repository's 1x1
segmentation head, and evaluates it on held-out frames.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from src.data.dataset import (
    EXCLUDED_ID_FILES,
    _mask_from_ground,
    filter_pairs,
    find_pairs,
    load_excluded_ids,
)
from src.models import DINOv3Segmenter, load_dinov3


CLASS_NAMES = ("background", "sky", "small_rock", "large_rock")
CLASS_COLORS = np.asarray(
    [
        (34, 34, 34),
        (72, 149, 239),
        (72, 201, 113),
        (239, 83, 80),
    ],
    dtype=np.uint8,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small DINOv3 semantic-segmentation demo"
    )
    parser.add_argument(
        "--model-id",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../../datasets/artificial_lunar_landscape"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dinov3_demo"),
    )
    parser.add_argument("--train-images", type=int, default=128)
    parser.add_argument("--eval-images", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument(
        "--unlabeled-images",
        type=Path,
        nargs="*",
        default=[],
        help="optional real images for a qualitative prediction sheet",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def find_dataset_pairs(root: Path) -> list[tuple[Path, Path]]:
    image_dir = root / "images" / "render"
    mask_dir = root / "images" / "ground"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"expected images/render and images/ground under {root.resolve()}"
        )

    pairs = find_pairs(image_dir, mask_dir)
    excluded = load_excluded_ids(root / name for name in EXCLUDED_ID_FILES)
    return filter_pairs(pairs, excluded)


def decode_mask(mask: Image.Image) -> np.ndarray:
    return _mask_from_ground(np.asarray(mask.convert("RGB")))


def load_batch(
    pairs: list[tuple[Path, Path]],
    processor: object,
) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
    images: list[Image.Image] = []
    masks: list[np.ndarray] = []
    for image_path, mask_path in pairs:
        with Image.open(image_path) as image_file:
            images.append(image_file.convert("RGB").copy())
        with Image.open(mask_path) as mask_file:
            masks.append(decode_mask(mask_file))

    encoded = processor(images=images, return_tensors="pt")
    pixels = encoded["pixel_values"]
    labels = torch.stack(
        [
            F.interpolate(
                torch.from_numpy(mask)[None, None].float(),
                size=pixels.shape[-2:],
                mode="nearest",
            )[0, 0].long()
            for mask in masks
        ]
    )
    return pixels, labels, [np.asarray(image) for image in images]


def extract_features(
    model: DINOv3Segmenter,
    processor: object,
    pairs: list[tuple[Path, Path]],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_batches = []
    label_batches = []
    for start in range(0, len(pairs), batch_size):
        pixels, labels, _ = load_batch(pairs[start : start + batch_size], processor)
        pixels = pixels.to(device)
        with torch.inference_mode():
            features = model._vit_features(pixels)
        patch_labels = F.interpolate(
            labels[:, None].float(),
            size=features.shape[-2:],
            mode="nearest",
        )[:, 0].long()
        feature_batches.append(features.float().cpu())
        label_batches.append(patch_labels.cpu())
        print(
            f"extracted {min(start + batch_size, len(pairs))}/{len(pairs)} frames",
            flush=True,
        )
    return torch.cat(feature_batches), torch.cat(label_batches)


def fit_head(
    model: DINOv3Segmenter,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    features = features.to(device)
    labels = labels.to(device)
    counts = torch.bincount(labels.flatten(), minlength=len(CLASS_NAMES)).float()
    weights = counts.sum().sqrt() / counts.clamp_min(1).sqrt()
    weights /= weights.mean()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = []
    model.classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model.classifier(features)
        loss = F.cross_entropy(logits, labels, weight=weights)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"epoch {epoch + 1:03d}/{epochs}: loss={history[-1]:.4f}", flush=True)
    model.classifier.eval()
    return history


def update_confusion(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    indices = labels.flatten().cpu() * len(CLASS_NAMES) + predictions.flatten().cpu()
    confusion += torch.bincount(
        indices,
        minlength=len(CLASS_NAMES) ** 2,
    ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))


def colorize(labels: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[labels]


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))


def make_example_sheet(
    examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    tile_size = 224
    header = 24
    columns = ("input", "ground truth", "prediction", "prediction overlay")
    sheet = Image.new(
        "RGB",
        (len(columns) * tile_size, len(examples) * (tile_size + header)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, (frame_id, image, labels, prediction) in enumerate(examples):
        image = resize_rgb(image, (tile_size, tile_size))
        ground_truth = colorize(labels)
        predicted = colorize(prediction)
        overlay = np.asarray(
            Image.blend(Image.fromarray(image), Image.fromarray(predicted), alpha=0.48)
        )
        tiles = (image, ground_truth, predicted, overlay)
        top = row * (tile_size + header)
        for column, (name, tile) in enumerate(zip(columns, tiles)):
            left = column * tile_size
            sheet.paste(Image.fromarray(tile), (left, top + header))
            label = f"{frame_id}: {name}" if column == 0 else name
            draw.text((left + 5, top + 5), label, fill="black")
    sheet.save(output_path)


def predict_unlabeled(
    model: DINOv3Segmenter,
    processor: object,
    image_paths: list[Path],
    device: torch.device,
    output_path: Path,
) -> None:
    images = []
    for path in image_paths:
        with Image.open(path) as image_file:
            images.append(image_file.convert("RGB").copy())
    pixels = processor(images=images, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        predictions = model(pixels.to(device)).argmax(dim=1).cpu().numpy()

    tile_size = 224
    header = 24
    columns = ("input", "prediction", "prediction overlay")
    sheet = Image.new(
        "RGB",
        (len(columns) * tile_size, len(images) * (tile_size + header)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, (path, image, prediction) in enumerate(
        zip(image_paths, images, predictions)
    ):
        image_array = resize_rgb(np.asarray(image), (tile_size, tile_size))
        predicted = colorize(prediction)
        overlay = np.asarray(
            Image.blend(
                Image.fromarray(image_array),
                Image.fromarray(predicted),
                alpha=0.48,
            )
        )
        top = row * (tile_size + header)
        for column, (name, tile) in enumerate(
            zip(columns, (image_array, predicted, overlay))
        ):
            left = column * tile_size
            sheet.paste(Image.fromarray(tile), (left, top + header))
            label = f"{path.name}: {name}" if column == 0 else name
            draw.text((left + 5, top + 5), label, fill="black")
    sheet.save(output_path)


def evaluate(
    model: DINOv3Segmenter,
    processor: object,
    pairs: list[tuple[Path, Path]],
    batch_size: int,
    device: torch.device,
    example_count: int,
) -> tuple[torch.Tensor, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]]:
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
    examples = []
    for start in range(0, len(pairs), batch_size):
        current_pairs = pairs[start : start + batch_size]
        pixels, labels, original_images = load_batch(current_pairs, processor)
        with torch.inference_mode():
            logits = model(pixels.to(device))
            predictions = logits.argmax(dim=1).cpu()
        update_confusion(confusion, predictions, labels)

        for pair, image, label, prediction in zip(
            current_pairs,
            original_images,
            labels.numpy(),
            predictions.numpy(),
        ):
            if len(examples) < example_count:
                examples.append((pair[0].stem[-4:], image, label, prediction))
        print(
            f"evaluated {min(start + batch_size, len(pairs))}/{len(pairs)} frames",
            flush=True,
        )
    return confusion, examples


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, object]:
    confusion = confusion.double()
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = intersection / union.clamp_min(1)
    return {
        "pixel_accuracy": float(intersection.sum() / confusion.sum().clamp_min(1)),
        "mean_iou": float(iou.mean()),
        "class_iou": {
            name: float(value) for name, value in zip(CLASS_NAMES, iou)
        },
        "confusion_matrix": confusion.long().tolist(),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    pairs = find_dataset_pairs(args.dataset_root)
    required = args.train_images + args.eval_images
    if required > len(pairs):
        raise ValueError(f"requested {required} frames, but only {len(pairs)} are available")
    random.Random(args.seed).shuffle(pairs)
    train_pairs = pairs[: args.train_images]
    eval_pairs = pairs[args.train_images : required]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    print(f"dataset: {args.dataset_root.resolve()} ({len(pairs)} usable frames)")
    print(f"model: {args.model_id}")
    print(f"split: {len(train_pairs)} probe-train / {len(eval_pairs)} held-out")
    started = time.perf_counter()

    model, processor = load_dinov3(args.model_id, num_classes=len(CLASS_NAMES))
    if not isinstance(model, DINOv3Segmenter) or model.backbone_type != "dinov3_vit":
        raise TypeError("this demo currently requires a DINOv3 ViT backbone")
    model.to(device).eval()
    features, patch_labels = extract_features(
        model,
        processor,
        train_pairs,
        args.batch_size,
        device,
    )
    loss_history = fit_head(
        model,
        features,
        patch_labels,
        args.epochs,
        args.learning_rate,
        device,
    )
    confusion, examples = evaluate(
        model,
        processor,
        eval_pairs,
        args.batch_size,
        device,
        min(args.examples, len(eval_pairs)),
    )
    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {
            "model_id": args.model_id,
            "device": str(device),
            "train_images": len(train_pairs),
            "eval_images": len(eval_pairs),
            "epochs": args.epochs,
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "elapsed_seconds": time.perf_counter() - started,
            "train_frame_ids": [pair[0].stem[-4:] for pair in train_pairs],
            "eval_frame_ids": [pair[0].stem[-4:] for pair in eval_pairs],
            "unlabeled_images": [str(path) for path in args.unlabeled_images],
        }
    )

    torch.save(model.classifier.state_dict(), args.output_dir / "linear_head.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    make_example_sheet(examples, args.output_dir / "examples.png")
    if args.unlabeled_images:
        predict_unlabeled(
            model,
            processor,
            args.unlabeled_images,
            device,
            args.output_dir / "unlabeled_examples.png",
        )
    print(json.dumps({key: metrics[key] for key in ("pixel_accuracy", "mean_iou", "class_iou")}, indent=2))
    print(f"artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
