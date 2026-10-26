from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from janko_registration.utils import Config, FeaturesConfig, get_datetime_str


def get_device() -> torch.device:
    # return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class PictureDataset(Dataset):
    @staticmethod
    def prepare_features(image: np.ndarray, features: FeaturesConfig) -> np.ndarray:
        """
        Remove color information or add grayscale / edge detection layers to the image.
        Can be adapted in the global `config.yaml` file.
        """
        if features.keep_colors:
            result = [image]
        else:
            result = []

        if features.add_grayscale or features.add_canny or features.add_sobel:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if features.add_grayscale:
                result.append(img_gray[..., None])

            if features.add_canny:
                canny = cv2.Canny(img_gray, 50, 150, 5)
                result.append(canny[..., None])

            if features.add_sobel:
                gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)

                mag = cv2.magnitude(gx, gy)
                mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
                result.append(mag[..., None])

        result = np.concatenate(result, axis=2)

        if features.show_previews and (
            features.add_grayscale or features.add_canny or features.add_sobel
        ):
            if not features.keep_colors:
                cv2.imshow("original", image)
            if features.add_grayscale:
                cv2.imshow("grayscale", img_gray)
            if features.add_canny:
                cv2.imshow("canny", canny)
            if features.add_sobel:
                cv2.imshow("sobel", mag)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

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
            print("\nLoading data ...")

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

                image = PictureDataset.prepare_features(image, config.features)

                # NumPy is:
                #   H x W x C
                #
                # PyTorch wants:
                #   C x H x W
                #
                # So we "rotate" dimensions
                timage = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                X.append(timage)

                tcorners = torch.tensor(
                    metadata[config.global_config.labels_column], dtype=torch.float32
                )
                y.append(tcorners)

                if print_output:
                    if (idx + 1) % 20 == 0:
                        print("█", end="", flush=True)

                    if (idx + 1) % 1000 == 0:
                        print(f" Loaded {idx + 1}/{desired_img_count}")

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
