from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from janko_registration.geometry.janko import (
    KeyGeometry,
    PianoGeometry,
    create_full_janko_geometry,
)
from janko_registration.utils import Config


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
    transformed_points = cv2.perspectiveTransform(
        canonical_points.reshape(-1, 1, 2),
        homography,
    )

    return transformed_points.reshape(-1, 2)


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
    angle_radians = np.radians(angle_degrees)

    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)

    return np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=np.float64,
    )


def make_random_homography(
    piano_geometry: PianoGeometry,
    config: Config,
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
        config.generator.base_image_width * config.generator.piano_width_min_fraction,
        config.generator.base_image_width * config.generator.piano_width_max_fraction,
    )

    scale_factor = target_piano_width_pixels / canonical_width

    scaled_bounding_box = centered_bounding_box * scale_factor

    # ------------------------------------------------------------
    # Rotate the piano.
    # ------------------------------------------------------------

    rotation_angle_degrees = rng.uniform(
        config.generator.rotation_min_degrees,
        config.generator.rotation_max_degrees,
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
        -0.25 * config.generator.base_image_width,
        1.25 * config.generator.base_image_width,
    )

    piano_center_y_pixels = rng.uniform(
        -0.25 * config.generator.base_image_height,
        1.25 * config.generator.base_image_height,
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
        rotated_piano_width_pixels * config.generator.perspective_strength
    )

    maximum_vertical_perspective_shift = (
        rotated_piano_height_pixels * config.generator.perspective_strength
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
    config: Config,
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

    for idx, image_path in enumerate(sorted(background_directory.iterdir())):
        if image_path.suffix.lower() not in supported_extensions:
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: could not read {image_path}")
            continue

        source_height, source_width = image.shape[:2]

        resize_scale = max(
            config.generator.base_image_width / source_width,
            config.generator.base_image_height / source_height,
        )

        resized_width = max(
            1,
            round(source_width * resize_scale),
        )

        resized_height = max(
            1,
            round(source_height * resize_scale),
        )

        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        backgrounds.append(image)

        if (idx + 1) % 10 == 0:
            print("█", end="", flush=True)

        if (idx + 1) % 100 == 0:
            print(f" Loaded {idx + 1} backgrounds")

    print(end=" ", flush=True)
    return backgrounds


def check_vertical_clipping(
    piano_geometry: PianoGeometry,
    transformed_polygons_by_key: list[list[np.ndarray]],
    image_width: int,
    image_height: int,
) -> bool:
    """
    A function that checks whether no visible keys are cut off at the top/bottom:

    Returns `true` if there is at least one "on-screen" key that is cut off.
    Returns `false` if visible keys are only cut off at the sides of the picture.
    """

    for key_geometry, transformed_polygons in zip(
        piano_geometry.keys,
        transformed_polygons_by_key,
    ):
        # print("Analyzing key", key_geometry.index)

        # Treat all sub-polygons as parts of the same key.
        key_points = np.concatenate(transformed_polygons, axis=0)

        min_x = key_points[:, 0].min()
        max_x = key_points[:, 0].max()
        min_y = key_points[:, 1].min()
        max_y = key_points[:, 1].max()

        # Check wheather the key has some horizontal overlap with the image.
        horizontally_visible = max_x >= 0 and min_x <= image_width

        if not horizontally_visible:
            # print("Key is NOT horizontally visible → skip checking for vertical clipping")
            continue

        vertically_clipped = min_y < 0 or max_y > image_height

        if vertically_clipped:
            # print("Key is visible and vertically clipped at the top or bottom")
            return True

        # print("No vertical clipping detected for this key")

    return False


def prepare_background(
    source_background: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly flip and crop a background to exactly fill the target image.

    The background is already resized when it is loaded. The random crop
    still changes the position of the background for every sample.
    """
    resized_height, resized_width = source_background.shape[:2]

    # After resizing, these are the possible ranges for the
    # top-left corner of the target crop.
    maximum_crop_x = resized_width - config.generator.base_image_width
    maximum_crop_y = resized_height - config.generator.base_image_height

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

    cropped_background = source_background[
        crop_y : crop_y + config.generator.base_image_height,
        crop_x : crop_x + config.generator.base_image_width,
    ].copy()

    # This should only happen for unusual input dimensions.
    # The resize is a final safety net.
    if cropped_background.shape[:2] != (
        config.generator.base_image_height,
        config.generator.base_image_width,
    ):
        cropped_background = cv2.resize(
            cropped_background,
            (
                config.generator.base_image_width,
                config.generator.base_image_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    flipped_background = cropped_background
    if rng.random() <= config.generator.flip_x_probability:
        flipped_background = cv2.flip(flipped_background, 0)
    if rng.random() <= config.generator.flip_y_probability:
        flipped_background = cv2.flip(flipped_background, 1)

    return flipped_background


def choose_background(
    backgrounds: list[np.ndarray],
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Choose a background image.
    """

    bg_idx = int(rng.integers(0, len(backgrounds)))
    source_background = backgrounds[bg_idx]

    return prepare_background(
        source_background,
        config=config,
        rng=rng,
    )


def draw_piano(
    image: np.ndarray,
    piano_geometry: PianoGeometry,
    transformed_polygons_by_key: list[list[np.ndarray]],
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw the transformed Janko piano onto an OpenCV image.

    Returns:
        A binary mask indicating which pixels belong to the piano.
    """

    piano_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    def random_shift_RGB(number: int) -> int:
        summand = rng.integers(
            -config.generator.color_variance, config.generator.color_variance
        )
        modified_number = number + summand
        clipped_number = np.clip(modified_number, 0, 255)

        return int(clipped_number)

    for key, transformed_polygons in zip(
        piano_geometry.keys,
        transformed_polygons_by_key,
    ):
        gray_value = (
            rng.integers(
                config.generator.piano_white_min_brightness, 255, endpoint=True
            )
            if key.is_white
            else rng.integers(
                0, config.generator.piano_black_max_brightness, endpoint=True
            )
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

    return np.clip(
        alpha,
        0.0,
        1.0,
    )


def apply_piano_overlay(
    image: np.ndarray,
    overlay: np.ndarray,
    piano_mask: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Blend an image over the piano with smoothly varying opacity.

    The overlay is only applied where the piano mask is present.

    The existing `blur_probability` is reused to decide whether the
    overlay itself receives Gaussian blur.
    """
    height, width = image.shape[:2]

    # Optionally blur the overlay.
    if rng.random() < config.generator.blur_probability:
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
        alpha_range=config.generator.overlay_alpha_range,
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
    transformed_polygons: list[np.ndarray],
    image_width: int,
    image_height: int,
) -> bool:
    """
    Return whether every polygon belonging to a key is fully visible.
    """
    for transformed_polygon in transformed_polygons:
        if not polygon_is_fully_visible(
            transformed_polygon,
            image_width,
            image_height,
        ):
            return False

    return True


def get_visible_key_indices(
    piano_geometry: PianoGeometry,
    visible_octave_span: int,
    transformed_polygons_by_key: list[list[np.ndarray]],
    image_width: int,
    image_height: int,
) -> tuple[list[int], list[int]]:
    """
    Return the indices of...

    1. all keys that are completely visible,
    2. the keys that belong to the desired nr of octaves, starting from the left.
    """

    visible_key_geoms = [
        key_geometry
        for key_geometry, transformed_polygons in zip(
            piano_geometry.keys,
            transformed_polygons_by_key,
        )
        if key_is_fully_visible(
            key_geometry,
            transformed_polygons,
            image_width,
            image_height,
        )
    ]

    visible_indices_full = [vkg.index for vkg in visible_key_geoms]

    # print([vkg.key_char for vkg in visible_key_geoms])
    try:
        octaves_start_idx = [vkg.key_char for vkg in visible_key_geoms].index("C")
    except ValueError:
        octaves_start_idx = 9999  # Fallback value

    octaves_end_idx = octaves_start_idx + 12 * visible_octave_span
    visible_indices_octaves = [
        vkg.index
        for vkg in visible_key_geoms
        if vkg.index >= octaves_start_idx and vkg.index < octaves_end_idx
    ]

    return visible_indices_full, visible_indices_octaves


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
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly modify brightness, contrast, blur, noise and compression.
    """
    result = image.astype(np.float32)

    # ------------------------------------------------------------
    # Brightness
    # ------------------------------------------------------------

    brightness_factor = rng.uniform(*config.generator.brightness_range)

    result *= brightness_factor

    # ------------------------------------------------------------
    # Contrast
    # ------------------------------------------------------------

    contrast_factor = rng.uniform(*config.generator.contrast_range)

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

    if rng.random() < config.generator.blur_probability:
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

    if rng.random() < config.generator.noise_probability:
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

    if rng.random() < config.generator.jpeg_probability:
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
    config: Config,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
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
    desired_visible_octaves = 3
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

        transformed_polygons_by_key = [
            [
                transform_points(
                    polygon,
                    homography,
                )
                for polygon in key_geometry.polygons
            ]
            for key_geometry in piano_geometry.keys
        ]

        visible_indices_full, visible_indices_octaves = get_visible_key_indices(
            piano_geometry,
            desired_visible_octaves,
            transformed_polygons_by_key,
            image_width=config.generator.base_image_width,
            image_height=config.generator.base_image_height,
        )

        shows_sufficient_keys = (
            len(visible_indices_octaves) >= desired_visible_octaves * 12
        )

        has_vertical_clipping = check_vertical_clipping(
            piano_geometry,
            transformed_polygons_by_key,
            image_width=config.generator.base_image_width,
            image_height=config.generator.base_image_height,
        )

        if shows_sufficient_keys and not has_vertical_clipping:
            # print("Found fitting image after", attempt_number, "attempts.")
            break

    else:
        raise RuntimeError(
            "Could not generate a sample image of appropriate"
            f"quality keys after {maximum_attempts} attempts."
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
        transformed_polygons_by_key,
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

    canonical_visible_bbox_full = get_canonical_bbox_for_keys(
        piano_geometry,
        visible_indices_full,
    )
    target_corners_full = transform_points(
        canonical_visible_bbox_full,
        homography,
    )

    canonical_visible_bbox_octaves = get_canonical_bbox_for_keys(
        piano_geometry,
        visible_indices_octaves,
    )
    target_corners_octaves = transform_points(
        canonical_visible_bbox_octaves,
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

    image = cv2.resize(image, config.global_config.synthetic_resolution)

    return (
        image,
        target_corners_full,
        visible_indices_full,
        target_corners_octaves,
        visible_indices_octaves,
        homography,
    )


def generate_dataset(
    sample_count: int,
    config: Config,
    seed: int | None = None,
) -> None:
    """
    Generate and save a complete synthetic training dataset.
    """
    background_directory = config.generator.background_dir
    synthetic_data_dir = config.global_config.synthetic_data_dir
    labels_path = synthetic_data_dir / "labels.jsonl"
    image_directory = synthetic_data_dir / "images"
    image_directory.mkdir(parents=True, exist_ok=True)

    random_generator = np.random.default_rng(seed)

    print("Creating canonical Janko geometry ...", end=" ")
    piano_geometry = create_full_janko_geometry()
    print("Done!\n")

    backgrounds = load_backgrounds(background_directory, config)
    print(f"Loaded {len(backgrounds)} backgrounds.\n")

    with labels_path.open(
        "w",
        encoding="utf-8",
    ) as labels_file:
        for sample_index in range(sample_count):
            (
                image,
                target_corners_full,
                visible_indices_full,
                target_corners_octaves,
                visible_indices_octaves,
                homography,
            ) = generate_sample_image(
                piano_geometry,
                backgrounds,
                config,
                random_generator,
            )

            filename = f"{sample_index:06d}.jpg"
            image_path = image_directory / filename
            write_success = cv2.imwrite(
                str(image_path),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            if not write_success:
                raise RuntimeError(f"Could not write image to {image_path}")

            metadata = {
                "image_loc": f"images/{filename}",
                "target_corners_full": target_corners_full.tolist(),
                "visible_indices_full": visible_indices_full,
                "target_corners_octaves": target_corners_octaves.tolist(),
                "visible_indices_octaves": visible_indices_octaves,
                "homography": homography.tolist(),
            }

            labels_file.write(json.dumps(metadata) + "\n")

            if (sample_index + 1) % 10 == 0:
                print("█", end="", flush=True)

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
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible generation.",
    )

    args = parser.parse_args()

    config = Config()

    generate_dataset(
        sample_count=args.count,
        config=config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
