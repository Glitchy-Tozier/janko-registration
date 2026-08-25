from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from janko_registration.utils import Config


class PictureDataset(Dataset):
    @staticmethod
    def load_synthetic_data(
        source_dir: Path, desired_img_count: int
    ) -> tuple[list, list]:
        """
        Load all images from a directory.

        Images that OpenCV cannot read are skipped with a warning.
        """

        labels_path = source_dir / "labels.jsonl"

        if not labels_path.exists():
            raise RuntimeError("Asserted synthetic data directory doesn't exist.")

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

                # NumPy is:
                #   H x W x C
                #
                # PyTorch wants:
                #   C x H x W
                #
                # So we "rotate" dimensions
                timage = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                X.append(timage)

                tcorners = torch.tensor(metadata["corners"], dtype=torch.float32)
                y.append(tcorners)

                if (idx + 1) % 10 == 0:
                    print("█", end="", flush=True)

                if (idx + 1) % 100 == 0:
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
    N: int, batch_size: int, config: Config
) -> tuple[DataLoader, DataLoader]:
    """A reusable way to load data, generate Datasets and create DataLoaders from them."""
    # Load data as lists
    X, y = PictureDataset.load_synthetic_data(
        config.global_config.synthetic_data_dir,
        N,
    )
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.global_config.test_split_fraction, shuffle=False
    )
    # Create Datasets
    train_ds = PictureDataset(X_train, y_train)
    test_ds = PictureDataset(X_test, y_test)
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=32, shuffle=False, num_workers=4
    )

    return (train_loader, test_loader)
