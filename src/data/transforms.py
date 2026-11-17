import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance


DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_IMAGE_WIDTH = 768


def _validate_pair(image: np.ndarray, mask: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 image, got shape {image.shape}")
    if mask.ndim != 2:
        raise ValueError(f"expected an HxW mask, got shape {mask.shape}")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "image and mask must have the same spatial shape, got "
            f"{image.shape[:2]} and {mask.shape}"
        )


def resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    height: int = DEFAULT_IMAGE_HEIGHT,
    width: int = DEFAULT_IMAGE_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize an aligned image/mask pair using modality-appropriate interpolation."""
    _validate_pair(image, mask)
    if height <= 0 or width <= 0:
        raise ValueError("resize height and width must be positive")

    size = (width, height)
    resized_image = Image.fromarray(image).resize(size, Image.Resampling.BILINEAR)
    resized_mask = Image.fromarray(mask.astype(np.int32)).resize(
        size,
        Image.Resampling.NEAREST,
    )
    return (
        np.asarray(resized_image, dtype=np.uint8).copy(),
        np.asarray(resized_mask, dtype=np.int64).copy(),
    )


@dataclass(frozen=True)
class EvalSegmentationTransform:
    """Deterministic full-frame preprocessing for validation and test data."""

    height: int = DEFAULT_IMAGE_HEIGHT
    width: int = DEFAULT_IMAGE_WIDTH

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image, mask = resize_pair(image, mask, self.height, self.width)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


@dataclass(frozen=True)
class TrainSegmentationTransform:
    """Full-frame resize followed by the canonical baseline augmentations."""

    height: int = DEFAULT_IMAGE_HEIGHT
    width: int = DEFAULT_IMAGE_WIDTH
    flip_probability: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    gamma_min: float = 0.85
    gamma_max: float = 1.15

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        if not 0.0 <= self.brightness <= 1.0:
            raise ValueError("brightness must be in [0, 1]")
        if not 0.0 <= self.contrast <= 1.0:
            raise ValueError("contrast must be in [0, 1]")
        if not 0.0 < self.gamma_min <= self.gamma_max:
            raise ValueError("gamma range must be positive and ordered")

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        transformed = Image.fromarray(image)
        if self.brightness:
            factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            transformed = ImageEnhance.Brightness(transformed).enhance(factor)
        if self.contrast:
            factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            transformed = ImageEnhance.Contrast(transformed).enhance(factor)

        pixels = np.asarray(transformed, dtype=np.float32) / 255.0
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return np.clip(pixels**gamma * 255.0, 0.0, 255.0).astype(np.uint8)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image, mask = resize_pair(image, mask, self.height, self.width)
        if random.random() < self.flip_probability:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
        image = self._photometric(image)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
