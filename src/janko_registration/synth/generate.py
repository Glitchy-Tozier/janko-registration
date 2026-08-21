from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from janko_registration.geometry.janko import (
    KeyGeometry,
    PianoGeometry,
    create_full_janko_geometry,
)
from janko_registration.piano.janko_piano_base import BLACK_INDICES


@dataclass
class GeneratorConfig:
    """Configuration controlling synthetic image generation."""

    image_width: int = 1920
    image_height: int = 1080

    piano_white_min_brightness: int = 150
    piano_black_max_brightness: int = 100
    color_variance: int = 10  # up to plus or minus that number to every RGB number

    # Desired width of the complete piano as a fraction of the image width.
    piano_width_min_fraction: float = 0.5
    piano_width_max_fraction: float = 2.0

    rotation_min_degrees: float = -25.0
    rotation_max_degrees: float = 25.0

    # Maximum perspective displacement as a fraction of the piano size.
    perspective_strength: float = 0.05

    brightness_range: tuple[float, float] = (0.5, 1.5)
    contrast_range: tuple[float, float] = (0.5, 1.5)

    blur_probability: float = 0.40
    noise_probability: float = 0.40

    flip_x_probability: float = 0.5
    flip_y_probability: float = 0.5

    jpeg_probability: float = 0.50

    # Opacity range for the texture/image overlaid on the piano.
    # The actual opacity varies spatially between these values.
    overlay_alpha_range: tuple[float, float] = (0.0, 0.3)


def transform_points(
    canonical_points: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    """
    Transform 2D points from canonical piano coordinates into image pixels.

    A homography is a 3x3 matrix. To use it with 2D points, we first
    turn (x, y) into homogeneous coordinates (x, y, 1).

    The matrix multiplication gives us:

        [x', y', w']

    We then divide x' and y' by w' to return to ordinary 2D coordinates:

        x = x' / w'
        y = y' / w'
    """
    homogeneous_points = np.concatenate(
        [
            canonical_points,
            np.ones(
                (len(canonical_points), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    transformed_homogeneous_points = homogeneous_points @ homography.T

    # Every transformed point now has the form (x', y', w').
    # Dividing by w' converts homogeneous coordinates back to (x, y).
    normalized_points = (
        transformed_homogeneous_points / transformed_homogeneous_points[:, 2:3]
    )

    return normalized_points[:, :2]


def rotation_matrix(
    angle_degrees: float,
) -> np.ndarray:
    """
    Return a 2D rotation matrix for the given angle in degrees.

    The standard 2D rotation matrix is:

        [ cos(theta)  -sin(theta) ]
        [ sin(theta)   cos(theta) ]

    Multiplying a point by this matrix rotates it around the origin.
    """
    angle_radians = math.radians(angle_degrees)

    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)

    return np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=np.float64,
    )


def make_random_homography(
    piano_geometry: PianoGeometry,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a random transformation from canonical piano space to image space.

    The transformation includes:
        - scaling
        - rotation
        - translation
        - perspective distortion
    """
    canonical_bounding_box = piano_geometry.bounding_box

    # ------------------------------------------------------------
    # Find the center and width of the canonical piano.
    # ------------------------------------------------------------

    canonical_center = canonical_bounding_box.mean(axis=0)

    canonical_width = (
        canonical_bounding_box[:, 0].max() - canonical_bounding_box[:, 0].min()
    )

    # Move the bounding-box corners so that the piano center is at (0, 0).
    # This lets us rotate and scale around the piano's own center.
    centered_bounding_box = canonical_bounding_box - canonical_center

    # ------------------------------------------------------------
    # Choose the target size in pixels.
    # ------------------------------------------------------------

    target_piano_width_pixels = rng.uniform(
        config.image_width * config.piano_width_min_fraction,
        config.image_width * config.piano_width_max_fraction,
    )

    scale_factor = target_piano_width_pixels / canonical_width

    scaled_bounding_box = centered_bounding_box * scale_factor

    # ------------------------------------------------------------
    # Rotate the piano.
    # ------------------------------------------------------------

    rotation_angle_degrees = rng.uniform(
        config.rotation_min_degrees,
        config.rotation_max_degrees,
    )

    rotated_bounding_box = (
        scaled_bounding_box @ rotation_matrix(rotation_angle_degrees).T
    )

    # ------------------------------------------------------------
    # Translate the piano.
    #
    # The center is intentionally allowed to lie outside the image.
    # That naturally creates partially cropped pianos.
    # ------------------------------------------------------------

    piano_center_x_pixels = rng.uniform(
        -0.25 * config.image_width,
        1.25 * config.image_width,
    )

    piano_center_y_pixels = rng.uniform(
        -0.25 * config.image_height,
        1.25 * config.image_height,
    )

    transformed_bounding_box = rotated_bounding_box.copy()

    transformed_bounding_box[:, 0] += piano_center_x_pixels
    transformed_bounding_box[:, 1] += piano_center_y_pixels

    # ------------------------------------------------------------
    # Add perspective distortion.
    # ------------------------------------------------------------

    rotated_piano_width_pixels = (
        rotated_bounding_box[:, 0].max() - rotated_bounding_box[:, 0].min()
    )

    rotated_piano_height_pixels = (
        rotated_bounding_box[:, 1].max() - rotated_bounding_box[:, 1].min()
    )

    maximum_horizontal_perspective_shift = (
        rotated_piano_width_pixels * config.perspective_strength
    )

    maximum_vertical_perspective_shift = (
        rotated_piano_height_pixels * config.perspective_strength
    )

    perspective_offsets = rng.uniform(
        low=[
            -maximum_horizontal_perspective_shift,
            -maximum_vertical_perspective_shift,
        ],
        high=[
            maximum_horizontal_perspective_shift,
            maximum_vertical_perspective_shift,
        ],
        size=transformed_bounding_box.shape,
    )

    destination_corners = transformed_bounding_box + perspective_offsets

    # ------------------------------------------------------------
    # Build the homography.
    #
    # The source points MUST be the original canonical coordinates.
    # The destination points are their desired image-space positions.
    #
    # OpenCV then finds the 3x3 matrix H that maps source -> destination.
    # ------------------------------------------------------------

    homography = cv2.getPerspectiveTransform(
        canonical_bounding_box.astype(np.float32),
        destination_corners.astype(np.float32),
    )

    return homography


def load_backgrounds(
    background_directory: Path,
) -> list[np.ndarray]:
    """
    Load all supported background images from a directory.

    Images that OpenCV cannot read are skipped with a warning.
    """
    if not background_directory.exists():
        return []

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    backgrounds = []

    for image_path in sorted(background_directory.iterdir()):
        if image_path.suffix.lower() not in supported_extensions:
            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(f"Warning: could not read {image_path}")
            continue

        backgrounds.append(image)

    return backgrounds


def prepare_background(
    source_background: np.ndarray,
    target_image_width: int,
    target_image_height: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resize, randomly flip and randomly crop a background to exactly fill the target image.

    The source image and target image can have completely different
    aspect ratios. We therefore scale the source until it covers the
    entire target, then take a random crop from the result.
    """
    source_height, source_width = source_background.shape[:2]

    resize_scale = max(
        target_image_width / source_width,
        target_image_height / source_height,
    )

    resized_width = max(
        1,
        round(source_width * resize_scale),
    )

    resized_height = max(
        1,
        round(source_height * resize_scale),
    )

    resized_background = cv2.resize(
        source_background,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    # After resizing, these are the possible ranges for the
    # top-left corner of the target crop.
    maximum_crop_x = resized_width - target_image_width
    maximum_crop_y = resized_height - target_image_height

    crop_x = (
        0
        if maximum_crop_x <= 0
        else int(
            rng.integers(
                0,
                maximum_crop_x + 1,
            )
        )
    )

    crop_y = (
        0
        if maximum_crop_y <= 0
        else int(
            rng.integers(
                0,
                maximum_crop_y + 1,
            )
        )
    )

    cropped_background = resized_background[
        crop_y : crop_y + target_image_height,
        crop_x : crop_x + target_image_width,
    ]

    # This should only happen for unusual input dimensions.
    # The resize is a final safety net.
    if cropped_background.shape[:2] != (
        target_image_height,
        target_image_width,
    ):
        cropped_background = cv2.resize(
            cropped_background,
            (
                target_image_width,
                target_image_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    flipped_background = cropped_background
    if rng.random() <= config.flip_x_probability:
        flipped_background = cv2.flip(flipped_background, 0)
    if rng.random() <= config.flip_y_probability:
        flipped_background = cv2.flip(flipped_background, 1)

    return flipped_background


def choose_background(
    backgrounds: list[np.ndarray],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Choose a background image.
    """

    bg_idx = int(rng.integers(0, len(backgrounds)))
    source_background = backgrounds[bg_idx]

    return prepare_background(
        source_background,
        target_image_width=config.image_width,
        target_image_height=config.image_height,
        config=config,
        rng=rng,
    )


def draw_piano(
    image: np.ndarray,
    piano_geometry: PianoGeometry,
    homography: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw the transformed Janko piano onto an OpenCV image.

    Returns:
        A binary mask indicating which pixels belong to the piano.
    """

    piano_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    def random_shift_RGB(number: int) -> int:
        summand = rng.integers(-config.color_variance, config.color_variance)
        modified_number = number + summand
        clipped_number = np.clip(modified_number, 0, 255)

        return int(clipped_number)

    for key in piano_geometry.keys:
        transformed_polygons = [
            transform_points(
                polygon,
                homography,
            )
            for polygon in key.polygons
        ]

        is_white = key.index % 12 not in BLACK_INDICES
        gray_value = (
            rng.integers(config.piano_white_min_brightness, 255, endpoint=True)
            if is_white
            else rng.integers(0, config.piano_black_max_brightness, endpoint=True)
        )

        fill_color = (
            random_shift_RGB(gray_value),
            random_shift_RGB(gray_value),
            random_shift_RGB(gray_value),
        )

        for polygon in transformed_polygons:
            polygon_pixels = np.round(polygon).astype(np.int32)

            # Draw the actual piano.
            cv2.fillPoly(
                image,
                [polygon_pixels],
                fill_color,
                lineType=cv2.LINE_AA,
            )

            # Draw the same polygon into a separate mask.
            cv2.fillPoly(
                piano_mask,
                [polygon_pixels],
                255,
                lineType=cv2.LINE_AA,
            )

    return piano_mask


def make_spatial_alpha_map(
    image_width: int,
    image_height: int,
    alpha_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a smoothly varying spatial opacity map.

    Rather than assigning a completely independent alpha value to
    every pixel, a small random map is generated and enlarged.
    This creates broad, natural-looking changes in opacity.
    """
    alpha_min, alpha_max = alpha_range

    alpha_small = rng.uniform(
        alpha_min,
        alpha_max,
        size=(8, 8),
    ).astype(np.float32)

    alpha = cv2.resize(
        alpha_small,
        (
            image_width,
            image_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    alpha = cv2.GaussianBlur(
        alpha,
        ksize=(0, 0),
        sigmaX=max(image_width, image_height) / 30.0,
    )

    return np.clip(
        alpha,
        0.0,
        1.0,
    )


def apply_piano_overlay(
    image: np.ndarray,
    overlay: np.ndarray,
    piano_mask: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Blend an image over the piano with smoothly varying opacity.

    The overlay is only applied where the piano mask is present.

    The existing `blur_probability` is reused to decide whether the
    overlay itself receives Gaussian blur.
    """
    height, width = image.shape[:2]

    overlay = cv2.resize(
        overlay,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Optionally blur the overlay.
    if rng.random() < config.blur_probability:
        blur_sigma = rng.uniform(0.3, 2.0)

        overlay = cv2.GaussianBlur(
            overlay,
            ksize=(0, 0),
            sigmaX=blur_sigma,
        )

    # Generate spatially varying opacity.
    alpha = make_spatial_alpha_map(
        image_width=width,
        image_height=height,
        alpha_range=config.overlay_alpha_range,
        rng=rng,
    )

    # Restrict the overlay to the piano.
    piano_mask_float = piano_mask.astype(np.float32) / 255.0

    alpha *= piano_mask_float

    # Convert HxW -> HxWx1 so it broadcasts over BGR channels.
    alpha = alpha[:, :, None]

    # Alpha compositing.
    image_float = image.astype(np.float32)
    overlay_float = overlay.astype(np.float32)

    result = image_float * (1.0 - alpha) + overlay_float * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def polygon_is_fully_visible(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    """
    Return whether every sampled polygon point lies inside the image.
    """
    x_coordinates = polygon[:, 0]
    y_coordinates = polygon[:, 1]

    return bool(
        np.all(x_coordinates >= 0)
        and np.all(x_coordinates < image_width)
        and np.all(y_coordinates >= 0)
        and np.all(y_coordinates < image_height)
    )


def key_is_fully_visible(
    key_geometry: KeyGeometry,
    homography: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    """
    Return whether every polygon belonging to a key is fully visible.
    """
    for polygon in key_geometry.polygons:
        transformed_polygon = transform_points(
            polygon,
            homography,
        )

        if not polygon_is_fully_visible(
            transformed_polygon,
            image_width,
            image_height,
        ):
            return False

    return True


def get_visible_key_indices(
    piano_geometry: PianoGeometry,
    homography: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[int]:
    """
    Return the indices of all keys that are completely visible.
    """
    return [
        key_geometry.index
        for key_geometry in piano_geometry.keys
        if key_is_fully_visible(
            key_geometry,
            homography,
            image_width,
            image_height,
        )
    ]


def get_canonical_bbox_for_keys(
    piano_geometry: PianoGeometry,
    visible_key_indices: list[int],
) -> np.ndarray:
    """
    Return the canonical bounding rectangle around selected keys.

    The returned points are ordered:
        top-left
        top-right
        bottom-right
        bottom-left
    """
    selected_keys = [
        key_geometry
        for key_geometry in piano_geometry.keys
        if key_geometry.index in visible_key_indices
    ]

    if not selected_keys:
        raise ValueError("No fully visible keys.")

    selected_points = np.concatenate(
        [
            polygon
            for key_geometry in selected_keys
            for polygon in key_geometry.polygons
        ],
        axis=0,
    )

    min_x = selected_points[:, 0].min()
    max_x = selected_points[:, 0].max()
    min_y = selected_points[:, 1].min()
    max_y = selected_points[:, 1].max()

    return np.array(
        [
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y],
            [min_x, min_y],
        ],
        dtype=np.float64,
    )


def apply_appearance_effects(
    image: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly modify brightness, contrast, blur, noise and compression.
    """
    result = image.astype(np.float32)

    # ------------------------------------------------------------
    # Brightness
    # ------------------------------------------------------------

    brightness_factor = rng.uniform(*config.brightness_range)

    result *= brightness_factor

    # ------------------------------------------------------------
    # Contrast
    # ------------------------------------------------------------

    contrast_factor = rng.uniform(*config.contrast_range)

    # Shift the image so that 127.5 is the midpoint,
    # multiply the distance from that midpoint, then shift back.
    result = (result - 127.5) * contrast_factor + 127.5

    result = np.clip(
        result,
        0,
        255,
    )

    # ------------------------------------------------------------
    # Blur
    # ------------------------------------------------------------

    if rng.random() < config.blur_probability:
        blur_sigma = rng.uniform(
            0.3,
            2.0,
        )

        result = cv2.GaussianBlur(
            result,
            ksize=(0, 0),
            sigmaX=blur_sigma,
        )

    # ------------------------------------------------------------
    # Sensor-like noise
    # ------------------------------------------------------------

    if rng.random() < config.noise_probability:
        noise_sigma = rng.uniform(
            1.0,
            10.0,
        )

        noise = rng.normal(
            0,
            noise_sigma,
            size=result.shape,
        )

        result += noise

    result = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    # ------------------------------------------------------------
    # JPEG compression artifacts
    # ------------------------------------------------------------

    if rng.random() < config.jpeg_probability:
        jpeg_quality = int(
            rng.integers(
                20,
                95,
            )
        )

        encode_success, encoded_image = cv2.imencode(
            ".jpg",
            result,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                jpeg_quality,
            ],
        )

        if encode_success:
            result = cv2.imdecode(
                encoded_image,
                cv2.IMREAD_COLOR,
            )

    return result


def generate_sample_image(
    piano_geometry: PianoGeometry,
    backgrounds: list[np.ndarray],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    np.ndarray,
]:
    """
    Generate one synthetic image and its complete ground-truth metadata.

    Returns:
        image
        four target corner points
        indices of fully visible keys
        homography used to place the piano
    """

    # ------------------------------------------------------------
    # Janko Piano homography
    # ------------------------------------------------------------
    maximum_attempts = 1000

    for attempt_number in range(
        1,
        maximum_attempts + 1,
    ):
        homography = make_random_homography(
            piano_geometry,
            config,
            rng,
        )

        visible_key_indices = get_visible_key_indices(
            piano_geometry,
            homography,
            image_width=config.image_width,
            image_height=config.image_height,
        )

        shows_sufficient_keys = len(visible_key_indices) >= 30

        even_idx_count = sum([True for v in visible_key_indices if v % 2 == 0])
        odd_idx_count = sum([True for v in visible_key_indices if v % 2 == 1])
        odd_even_diff = abs(even_idx_count - odd_idx_count)
        odd_even_diff_acceptable = odd_even_diff <= 5

        if shows_sufficient_keys and odd_even_diff_acceptable:
            # print("Found fitting image after", attempt_number, "attempts.")
            break

    else:
        raise RuntimeError(
            "Could not generate a sample with at least "
            "two fully visible keys after "
            f"{maximum_attempts} attempts."
        )

    # ------------------------------------------------------------
    # Background.
    # ------------------------------------------------------------

    image = choose_background(
        backgrounds,
        config,
        rng,
    )

    # ------------------------------------------------------------
    # Draw the opaque piano and obtain a piano mask.
    # ------------------------------------------------------------

    piano_mask = draw_piano(
        image,
        piano_geometry,
        homography,
        config,
        rng,
    )

    # ------------------------------------------------------------
    # Add a second randomly selected image over the piano.
    #
    # We deliberately reuse choose_background() here. It already
    # performs random image selection, resizing, cropping and flipping.
    # ------------------------------------------------------------

    overlay = choose_background(
        backgrounds,
        config,
        rng,
    )

    image = apply_piano_overlay(
        image,
        overlay,
        piano_mask,
        config,
        rng,
    )

    # ------------------------------------------------------------
    # Ground-truth bounding box for the visible keys.
    # ------------------------------------------------------------

    canonical_visible_key_bbox = get_canonical_bbox_for_keys(
        piano_geometry,
        visible_key_indices,
    )

    target_corners = transform_points(
        canonical_visible_key_bbox,
        homography,
    )

    # ------------------------------------------------------------
    # Global image effects.
    # ------------------------------------------------------------

    image = apply_appearance_effects(
        image,
        config,
        rng,
    )

    return (
        image,
        target_corners,
        visible_key_indices,
        homography,
    )


def generate_dataset(
    output_directory: Path,
    background_directory: Path,
    sample_count: int,
    config: GeneratorConfig,
    seed: int | None = None,
) -> None:
    """
    Generate and save a complete synthetic training dataset.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_directory = output_directory / "images"
    image_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_path = output_directory / "labels.jsonl"

    random_generator = np.random.default_rng(seed)

    print("Creating canonical Janko geometry ...", end=" ")
    piano_geometry = create_full_janko_geometry()
    print(
        f" Done!\nLoading backgrounds from {background_directory} ...",
        end=" ",
        flush=True,
    )

    backgrounds = load_backgrounds(background_directory)
    print(f"Loaded {len(backgrounds)} background(s).")

    with labels_path.open(
        "w",
        encoding="utf-8",
    ) as labels_file:
        for sample_index in range(sample_count):
            (image, target_corners, visible_key_indices, homography) = (
                generate_sample_image(
                    piano_geometry,
                    backgrounds,
                    config,
                    random_generator,
                )
            )

            filename = f"{sample_index:06d}.png"
            image_path = image_directory / filename
            write_success = cv2.imwrite(
                str(image_path),
                image,
            )

            if not write_success:
                raise RuntimeError(f"Could not write image to {image_path}")

            metadata = {
                "image": f"images/{filename}",
                "corners": target_corners.tolist(),
                "visible_keys": visible_key_indices,
                "homography": homography.tolist(),
            }

            labels_file.write(json.dumps(metadata) + "\n")

            if (sample_index + 1) % 10 == 0:
                print("▉", end="", flush=True)

            if (sample_index + 1) % 100 == 0:
                print(f" Generated {sample_index + 1}/{sample_count}")


def main() -> None:
    """Parse command-line arguments and generate the requested dataset."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of synthetic images to generate.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic"),
        help="Directory in which to store the dataset.",
    )

    parser.add_argument(
        "--backgrounds",
        type=Path,
        default=Path("data/backgrounds"),
        help="Directory containing source background images.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible generation.",
    )

    args = parser.parse_args()

    config = GeneratorConfig()

    generate_dataset(
        output_directory=args.output,
        background_directory=args.backgrounds,
        sample_count=args.count,
        config=config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
