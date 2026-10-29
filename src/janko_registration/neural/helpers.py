from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from janko_registration.utils import Config, get_datetime_str


def get_device() -> torch.device:
    # return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class PictureDataset(Dataset):
    @staticmethod
    def prepare_features(image: np.ndarray, config: Config) -> np.ndarray:
        """
        Extract and combine image features into a single H x W x C array.

        Supported features:
            - Normalized BGR color channels
            - Grayscale
            - Canny edge detection
            - Sobel gradient magnitude
            - Probabilistic Hough line detection

        All output channels use uint8 values in the range 0-255.
        The resulting array can be normalized to 0-1 when converted
        to a PyTorch tensor.
        """
        img_color = cv2.normalize(
            image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)
        features = config.features
        result = []

        if features.keep_colors:
            result.append(img_color)

        needs_gray = (
            features.add_grayscale
            or features.add_canny
            or features.add_sobel
            or features.add_lines
        )
        if needs_gray:
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # Reused for Hough line detection
        if features.add_canny or features.add_lines:
            img_canny = cv2.Canny(
                img_gray,
                threshold1=1000,
                threshold2=500,
                apertureSize=5,
            )

        if features.add_grayscale:
            result.append(img_gray[..., None])

        if features.add_canny:
            result.append(img_canny[..., None])

        if features.add_sobel:
            gx = cv2.Sobel(img_gray, cv2.CV_32F, dx=1, dy=0, ksize=3)
            gy = cv2.Sobel(img_gray, cv2.CV_32F, dx=0, dy=1, ksize=3)

            mag = cv2.magnitude(gx, gy)
            img_sobel = cv2.normalize(
                mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            ).astype(np.uint8)
            # mag = cv2.normalize(mag, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) # Old, wrong implementation

            result.append(img_sobel[..., None])

        # Probabilistic Hough line detection:
        # Detect long, approximately horizontal line segments.
        if features.add_lines:
            _height, width = img_gray.shape

            # Binary mask containing the selected line segments.
            line_mask = np.zeros_like(img_gray, dtype=np.uint8)

            # Initial parameters relative to image width.
            min_line_length = max(30, int(width * 0.12))
            max_line_gap = max(5, int(width * 0.05))

            min_line_count = 15
            for threshold in range(160, 20, -20):
                lines = cv2.HoughLinesP(
                    img_canny,
                    rho=1,
                    theta=np.pi / 180,
                    threshold=threshold,
                    minLineLength=min_line_length,
                    maxLineGap=max_line_gap,
                )

                if lines is not None and len(lines) >= min_line_count:
                    if features.show_previews:
                        print(threshold)
                    break
                else:
                    continue

            try:
                for x1, y1, x2, y2 in lines:
                    # Calculate the line's angle relative to horizontal.
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    # Treat lines pointing in opposite directions equally.
                    angle = min(angle, 180.0 - angle)

                    # Use max rotation angle from config, including a safety
                    # buffer due to perspective distortion
                    if angle <= config.generator.rotation_max_degrees + 10:
                        cv2.line(
                            line_mask,
                            (x1, y1),
                            (x2, y2),
                            color=255,
                            thickness=2,
                            lineType=cv2.LINE_8,
                        )
                if len(lines) < min_line_count:
                    print(
                        f"\n    config: add_lines: Could only detect {len(lines)} lines on image."
                    )
                    """ cv2.imshow("original", img_color)
                    cv2.imshow("grayscale", img_gray)
                    cv2.imshow("canny", img_canny)
                    cv2.imshow("lines", line_mask[..., None])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows() """
            except ValueError:
                print("\n    Could not detect any lines in image.")
            result.append(line_mask[..., None])

        # Validate and combine the feature channels
        if not result:
            raise ValueError("No image features enabled.")

        combined = np.concatenate(result, axis=2)

        # Previews
        if features.show_previews:
            # Flatten first 10 rows, first 10 columns across all channels
            for c in range(combined.shape[2]):
                channel = combined[..., c]
                df = pd.DataFrame(
                    channel[:10, :10], columns=[f"col_{i}" for i in range(10)]
                )
                print(f"\nChannel {c}:")
                print(df.to_string(float_format=lambda x: f"{x:.0f}"))

            cv2.imshow("original", image)
            cv2.imshow("normalized color", img_color)
            if "img_gray" in locals():
                cv2.imshow("grayscale", img_gray)
            if "img_canny" in locals():
                cv2.imshow("canny", img_canny)
            if "img_sobel" in locals():
                cv2.imshow("sobel", img_sobel)
            if "line_mask" in locals():
                cv2.imshow("hough_lines", line_mask)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return combined

    @staticmethod
    def load_synthetic_data(
        desired_img_count: int, print_output: bool, config: Config
    ) -> tuple[list, list]:
        """
        Load all images from a directory.

        Images that OpenCV cannot read are skipped with a warning.
        """

        source_dir = config.global_config.synthetic_data_dir
        labels_path = source_dir / "labels.jsonl"

        if not labels_path.exists():
            raise RuntimeError("Asserted synthetic data directory doesn't exist.")

        if print_output:
            print()

        X = []
        y = []

        with labels_path.open("r", encoding="utf-8") as labels_file:
            for idx, line in enumerate(labels_file):
                if idx == desired_img_count:
                    break

                metadata = json.loads(line)
                image_path = source_dir / metadata["image_loc"]
                image: np.ndarray = cv2.imread(image_path)

                if image is None:
                    print(f"Warning: could not read {image_path}")
                    continue

                image = PictureDataset.prepare_features(image, config)

                # NumPy is:
                #   H x W x C
                #
                # PyTorch wants:
                #   C x H x W
                #
                # So we "rotate" dimensions
                timage = torch.from_numpy(image).permute(2, 0, 1).float()
                timage.div_(255.0)  # Inplace division to avoid random memory bugs
                X.append(timage)
                # del image

                tfeatures = torch.tensor(
                    metadata[config.global_config.labels_column], dtype=torch.float32
                )
                y.append(tfeatures)

                if print_output and (idx + 1) % 20 == 0:
                    print(
                        f"\rLoading images ... loaded {idx + 1}/{desired_img_count} images.",
                        end="",
                    )
        print(" Done!")
        return X, y  # torch.tensor(X), torch.tensor(y)

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        # return self.labels.shape[0]
        return len(self.labels)


def create_dataloaders(
    N: int, batch_size: int, print_output: bool, config: Config
) -> tuple[DataLoader, DataLoader]:
    """A reusable way to load data, generate Datasets and create DataLoaders from them."""
    # Load data as lists
    X, y = PictureDataset.load_synthetic_data(N, print_output, config)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.global_config.test_split_fraction, shuffle=False
    )
    # Create Datasets
    train_ds = PictureDataset(X_train, y_train)
    test_ds = PictureDataset(X_test, y_test)
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    return (train_loader, test_loader)


def compute_loss_on_dataset(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    convert_to_points: bool,
    print_output: bool,
    config: Config,
) -> float:
    """Calculate mean absolute corner-coordinate error in pixels."""

    model.eval()
    # `reduction="sum"` is to integrate batch size
    criterion = torch.nn.L1Loss(reduction="sum")

    total_absolute_error = 0.0
    total_coordinates = 0

    if print_output:
        print("Calculating corner MAE ...")

    with torch.inference_mode():
        for features, labels in dataloader:
            if print_output:
                print("█", end="", flush=True)
            features = features.to(device)
            labels = labels.to(device)

            prediction = model(features)
            if convert_to_points:
                prediction = model.model_output_to_points(prediction, config)

            total_absolute_error += criterion(labels, prediction).item()
            total_coordinates += labels.numel()

    if total_coordinates == 0:
        raise ValueError("Cannot calculate corner MAE on an empty dataset.")

    mean_mae = round(total_absolute_error / total_coordinates, 2)

    if print_output:
        print("\nMean corner MAE over (test?) dataset:", mean_mae, "(in px)")

    return mean_mae


def save_model(model: torch.nn.Module, config: Config) -> None:
    """Save the model to a preconfigured path."""
    model_dir = config.global_config.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"{model.__class__.__name__}_{get_datetime_str()}.pth"
    model_path = model_dir / model_filename

    torch.save(model.state_dict(), model_path)
    print("💾 Saved model to", model_path)
