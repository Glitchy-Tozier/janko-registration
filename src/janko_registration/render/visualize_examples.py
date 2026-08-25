from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from janko_registration.neural.train_neural_network import NN_v1, NN_v2, NN_v3
from janko_registration.utils import Config


def load_synthetic_data(
    source_dir: Path, start_idx: int, desired_img_count: int, config: Config
) -> pd.DataFrame:
    """
    Load all images from a directory.

    Images that OpenCV cannot read are skipped with a warning.
    """

    labels_path = source_dir / "labels.jsonl"

    if not labels_path.exists():
        raise RuntimeError("Asserted synthetic data directory doesn't exist.")

    print("\nLoading synthetic data ...")
    rows = []

    with labels_path.open("r", encoding="utf-8") as labels_file:
        labels = list(labels_file)

    available_nr = len(labels[start_idx:])
    desired_end_idx = start_idx + desired_img_count
    max_possible_end_idx = start_idx + available_nr
    desired_labels = labels[start_idx : min(desired_end_idx, max_possible_end_idx)]

    for idx, line in enumerate(desired_labels):
        if idx == desired_img_count:
            break

        data = json.loads(line)
        image_name = data["image_loc"]
        image_path = source_dir / image_name

        image: np.ndarray = cv2.imread(image_path)

        if image is None:
            print(f"Warning: could not read {image_path}")
            continue

        # NumPy is:
        #   H x W x C
        #
        # PyTorch wants:
        #   C x H x W
        #
        # So we "rotate" dimensions
        timage = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        data["image"] = image
        data["timage"] = timage
        data["corners"] = torch.tensor(data["corners"], dtype=torch.float32)

        rows.append(data)

        if (idx + 1) % 10 == 0:
            print("█", end="", flush=True)

        if (idx + 1) % 100 == 0:
            print(f" Loaded {idx + 1}/{desired_img_count}")

    df = pd.DataFrame(rows)
    print("Dataframe shape  :", df.shape)
    print("Dataframe columns:", df.columns)

    return df


def load_real_pictures(
    source_dir: Path, desired_img_count: int, config: Config
) -> pd.DataFrame:
    """
    Load all actual images from a directory.

    Images that OpenCV cannot read are skipped with a warning.
    """
    if not source_dir.exists():
        raise RuntimeError("Asserted real data directory doesn't exist.")

    print("\nLoading real data ...")
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    image_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    rows = []

    for idx, image_path in enumerate(image_paths):
        if idx == desired_img_count:
            break

        data = {"image_loc": image_path.name}

        image_original: np.ndarray = cv2.imread(image_path)

        if image_original is None:
            print(f"Warning: could not read {image_path}")
            continue

        image_resized: np.ndarray = cv2.resize(
            image_original,
            config.global_config.synthetic_resolution,
            interpolation=cv2.INTER_AREA,
        )

        # NumPy is:
        #   H x W x C
        #
        # PyTorch wants:
        #   C x H x W
        #
        # So we "rotate" dimensions
        timage = (
            torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float()
            / 255.0
        )
        data["image_original"] = image_original
        data["image_resized"] = image_resized
        data["timage"] = timage

        rows.append(data)

        if (idx + 1) % 10 == 0:
            print("█", end="", flush=True)

        if (idx + 1) % 100 == 0:
            print(f" Loaded {idx + 1}/{desired_img_count}")

    df = pd.DataFrame(rows)
    print("Dataframe shape  :", df.shape)
    print("Dataframe columns:", df.columns)

    return df


def draw_points_on_picture(
    image: np.ndarray,
    points: torch.Tensor,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    # Ensure points are integer pixel coordinates
    points = np.asarray(points).round().astype(np.int32)

    # Draw onto a separate overlay
    overlay = image.copy()

    # Semi-transparent filled rectangle/quadrilateral
    cv2.fillPoly(
        overlay,
        [points],
        color=color,
    )

    # Semi-transparent boundary lines
    cv2.polylines(
        overlay,
        [points],
        isClosed=True,
        color=color,
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    # Blend overlay with the image
    image = cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
    )

    # Optional: draw the predicted points themselves
    # for x, y in points:
    for point in points:
        cv2.circle(
            image,
            point,
            # (int(x), int(y)),
            radius=5,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return image


def save_prediction_on_picture(
    model_filename_stem: str,
    image_name: str,
    image: np.ndarray,
    true_points: torch.Tensor | None,
    pred_points: torch.Tensor,
    config: Config,
) -> None:
    # original image resolution
    o_width = image.shape[1]
    o_height = image.shape[0]
    # target resolution
    t_width = config.generator.base_image_width
    t_height = config.generator.base_image_height

    print(f"Image has a shape of {o_width}x{o_height} →", end=" ")
    if o_width != t_width or o_height != t_height:
        print(f"scaling image to {t_width}x{t_height}.")
        image = cv2.resize(image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
    else:
        print("keeping image at this resolution.")

    # OpenCV uses BGR colors, not RGB.
    blue = (255, 0, 0)
    green = (0, 255, 0)
    # Draw predicted and actual shapes onto image (if they exist)
    if true_points is not None:
        image = draw_points_on_picture(image, true_points, blue)
    image = draw_points_on_picture(image, pred_points, green)

    image_path = Path(
        f"data/visualizations/{model_filename_stem}/{image_name.split('/')[-1]}"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(image_path, image):
        print(f"Saved image to {image_path}\n")
    else:
        print(f"Could not save image to {image_path}\n")


def main() -> None:
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=f"Name of the model that shall be fetched from the models directory (currently '{config.global_config.model_dir}').",
    )
    parser.add_argument(
        "--train_start_idx",
        type=int,
        default=0,
        help="Number of synthetic example images to load. Use -1 to load all.",
    )
    parser.add_argument(
        "--nr_train_examples",
        type=int,
        default=0,
        help="Number of synthetic example images to load. Use -1 to load all.",
    )
    parser.add_argument(
        "--test_start_idx",
        type=int,
        default=800,
        help="Number of synthetic example images to load. Use -1 to load all.",
    )
    parser.add_argument(
        "--nr_test_examples",
        type=int,
        default=0,
        help="Number of synthetic example images to load. Use -1 to load all.",
    )
    parser.add_argument(
        "--nr_real_examples",
        type=int,
        default=0,
        help="Number of real example images to load. Use -1 to load all.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Prepare synthetic dataset
    # ------------------------------------------------------------

    df_train = load_synthetic_data(
        config.global_config.synthetic_data_dir,
        args.train_start_idx,
        args.nr_train_examples,
        config,
    )
    df_test = load_synthetic_data(
        config.global_config.synthetic_data_dir,
        args.test_start_idx,
        args.nr_test_examples,
        config,
    )
    df_synth = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    print(df_synth.shape)

    # ------------------------------------------------------------
    # Prepare real dataset
    # ------------------------------------------------------------

    df_real = load_real_pictures(
        config.global_config.real_data_dir,
        args.nr_real_examples,
        config,
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    torch.manual_seed(123)
    model = NN_v2()

    model_path = config.global_config.model_dir / args.model_name
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("\nModel:")
    print(model)

    for _, row in df_synth.iterrows():
        features = row["timage"]
        labels = row["corners"]
        with torch.no_grad():
            pred_points = model(features).squeeze(0)

        pred_points_sup = pred_points.clone()  # scaled up
        pred_points_sup[:, 0] *= config.generator.base_image_width
        pred_points_sup[:, 1] *= config.generator.base_image_height

        true_points = labels.clone()
        true_points_sdown = labels.clone()  # scaled down
        true_points_sdown[:, 0] /= config.generator.base_image_width
        true_points_sdown[:, 1] /= config.generator.base_image_height

        for pred, true in zip(pred_points_sup.view([-1]), true_points.view([-1])):
            # print(pred, true)
            pass

        save_prediction_on_picture(
            model_path.stem,
            row["image_loc"],
            row["image"],
            true_points,
            pred_points_sup,
            config,
        )

    for _, row in df_real.iterrows():
        features = row["timage"]
        with torch.no_grad():
            pred_points = model(features).squeeze(0)

        pred_points_sup = pred_points.clone()  # scaled up
        pred_points_sup[:, 0] *= config.generator.base_image_width
        pred_points_sup[:, 1] *= config.generator.base_image_height

        for pred, true in zip(pred_points_sup.view([-1]), true_points.view([-1])):
            # print(pred, true)
            pass
        save_prediction_on_picture(
            model_path.stem,
            row["image_loc"],
            row["image_original"],
            None,
            pred_points_sup,
            config,
        )


if __name__ == "__main__":
    main()
