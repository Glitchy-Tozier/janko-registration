from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def get_datetime_str() -> str:
    now = datetime.now()  # Get current time  # noqa: DTZ005

    timestamp_string = now.strftime("%y%m%d_%H%M")  # Convert to YYMMDDHHMM string
    # print(timestamp_string)  # Output example: 2608241430
    return timestamp_string


# Config


@dataclass
class GlobalConfig:
    synthetic_data_dir: Path
    synthetic_resolution: tuple[int, int]
    labels_column: str
    test_split_fraction: float
    model_dir: Path
    real_data_dir: Path


@dataclass
class GeneratorConfig:
    background_dir: Path
    base_image_width: int
    base_image_height: int
    piano_white_min_brightness: int
    piano_black_max_brightness: int
    color_variance: int
    piano_width_min_fraction: float
    piano_width_max_fraction: float
    rotation_min_degrees: float
    rotation_max_degrees: float
    perspective_strength: float
    brightness_range: tuple[float, float]
    contrast_range: tuple[float, float]
    blur_probability: float
    noise_probability: float
    flip_x_probability: float
    flip_y_probability: float
    jpeg_probability: float
    overlay_alpha_range: tuple[float, float]


@dataclass
class FeaturesConfig:
    keep_colors: bool
    add_grayscale: bool
    add_canny: bool
    add_sobel: bool
    show_previews: bool

    @property
    def count(self) -> int:
        return self.keep_colors * 3 + sum(
            (
                self.add_grayscale,
                self.add_canny,
                self.add_sobel,
            )
        )


@dataclass
class HeatmapConfig:
    unpadded_resolution: tuple[int, int]
    sigma: float
    border_width_multiplier: float


@dataclass
class Config:
    global_config: GlobalConfig
    generator: GeneratorConfig
    features: FeaturesConfig
    heatmap: HeatmapConfig

    def __init__(self, yml_path: str | Path = "config.yaml") -> None:
        with Path(yml_path).open("r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        global_cfg = data["global_config"]
        generator_cfg = data["generator"]
        features_cfg = data["features"]
        heatmap_cfg = data["heatmap"]

        self.global_config = GlobalConfig(
            synthetic_data_dir=Path(global_cfg["synthetic_data_dir"]),
            test_split_fraction=float(global_cfg["test_split_fraction"]),
            labels_column=str(global_cfg["labels_column"]),
            synthetic_resolution=tuple(global_cfg["synthetic_resolution"]),
            model_dir=Path(global_cfg["model_dir"]),
            real_data_dir=Path(global_cfg["real_data_dir"]),
        )

        self.generator = GeneratorConfig(
            background_dir=Path(generator_cfg["background_dir"]),
            base_image_width=int(generator_cfg["base_image_width"]),
            base_image_height=int(generator_cfg["base_image_height"]),
            piano_white_min_brightness=int(generator_cfg["piano_white_min_brightness"]),
            piano_black_max_brightness=int(generator_cfg["piano_black_max_brightness"]),
            color_variance=int(generator_cfg["color_variance"]),
            piano_width_min_fraction=float(generator_cfg["piano_width_min_fraction"]),
            piano_width_max_fraction=float(generator_cfg["piano_width_max_fraction"]),
            rotation_min_degrees=float(generator_cfg["rotation_min_degrees"]),
            rotation_max_degrees=float(generator_cfg["rotation_max_degrees"]),
            perspective_strength=float(generator_cfg["perspective_strength"]),
            brightness_range=tuple(generator_cfg["brightness_range"]),
            contrast_range=tuple(generator_cfg["contrast_range"]),
            blur_probability=float(generator_cfg["blur_probability"]),
            noise_probability=float(generator_cfg["noise_probability"]),
            flip_x_probability=float(generator_cfg["flip_x_probability"]),
            flip_y_probability=float(generator_cfg["flip_y_probability"]),
            jpeg_probability=float(generator_cfg["jpeg_probability"]),
            overlay_alpha_range=tuple(generator_cfg["overlay_alpha_range"]),
        )

        self.features = FeaturesConfig(
            show_previews=bool(features_cfg["show_previews"]),
            keep_colors=bool(features_cfg["keep_colors"]),
            add_grayscale=bool(features_cfg["add_grayscale"]),
            add_canny=bool(features_cfg["add_canny"]),
            add_sobel=bool(features_cfg["add_sobel"]),
        )

        self.heatmap = HeatmapConfig(
            unpadded_resolution=tuple(heatmap_cfg["unpadded_resolution"]),
            sigma=float(heatmap_cfg["sigma"]),
            border_width_multiplier=float(heatmap_cfg["border_width_multiplier"]),
        )
