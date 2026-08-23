from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GlobalConfig:
    synthetic_resolution: tuple[int, int]


@dataclass
class GeneratorConfig:
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
class Config:
    global_config: GlobalConfig
    generator: GeneratorConfig

    def __init__(self, yml_path: str | Path = "config.yaml") -> None:
        with Path(yml_path).open("r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        global_cfg = data["global_config"]
        generator_cfg = data["generator"]

        self.global_config = GlobalConfig(
            synthetic_resolution=tuple(global_cfg["synthetic_resolution"]),
        )

        self.generator = GeneratorConfig(
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
