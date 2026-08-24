from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from janko_registration.utils import Config, format_duration, get_datetime_str


class PictureDataset(Dataset):
    @staticmethod
    def load_synthetic_data(
        source_dir: Path, desired_img_count: int, debugging: bool = False
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
                image_name = metadata["image"]
                image_path = source_dir / image_name
                image: np.ndarray = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_COLOR,
                )

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
                    print(f" Loaded {idx + 1}/{len(y)}")

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


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        NUM_OUTPUTS = 8  # 2 x 4 coordinates

        self.features = torch.nn.Sequential(
            # Input:
            #   batch x 3 x 180 x 320
            #
            # Output:
            #   batch x 32 x 180 x 320
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            # batch x 32 x 90 x 160
            torch.nn.MaxPool2d(kernel_size=2),
            # batch x 64 x 90 x 160
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            # batch x 64 x 45 x 80
            torch.nn.MaxPool2d(kernel_size=2),
            # batch x 128 x 45 x 80
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            # batch x 128 x 22 x 40
            torch.nn.MaxPool2d(kernel_size=2),
            # batch x 256 x 22 x 40
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            # batch x 256 x 1 x 1
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.output = torch.nn.Sequential(
            # batch x 256 x 1 x 1
            # ->
            # batch x 256
            torch.nn.Flatten(),
            # batch x 256
            # ->
            # batch x 9
            torch.nn.Linear(256, NUM_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        x = x.view([-1, 4, 2])
        # print(x)
        return x


def compute_loss_on_testset(
    model: NeuralNetwork,
    dataloader: PictureDataset,
    criterion: torch.nn.L1Loss,
    config: Config,
) -> tuple[float, float]:
    model.eval()
    mae_1 = []
    mae_2 = []

    # Loop through batches of the dataset to prevent memory issues
    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            pred_points = model(features)  # [B, 4, 2]

        pred_points_sup = pred_points.clone()  # scaled up
        pred_points_sup[:, :, 0] *= config.generator.base_image_width
        pred_points_sup[:, :, 1] *= config.generator.base_image_height

        true_points = labels.clone()
        true_points_sdown = labels.clone()  # scaled down
        true_points_sdown[:, :, 0] /= config.generator.base_image_width
        true_points_sdown[:, :, 1] /= config.generator.base_image_height

        mae_1.append(criterion(pred_points, true_points_sdown))  # Loss function
        mae_2.append(criterion(pred_points_sup, true_points))  # Loss function

    print("\nMAE mean over test dataset:", np.mean(mae_1), "(downscaled)")
    print("MAE mean over test dataset:", np.mean(mae_2), "(in px)\n")

    return np.mean(mae_1), np.mean(mae_2)


def main() -> None:
    # Unfortunately my PC has an Intel graphics card.
    print("CUDA availibility:", torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_count",
        type=int,
        default=-1,
        help="Number of sample images to load. Use -1 to load all.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs.",
    )
    args = parser.parse_args()
    config = Config()

    # ------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------

    torch.manual_seed(123)
    model = NeuralNetwork()
    print("\nModel:")
    print(model)

    # ------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------

    X, y = PictureDataset.load_synthetic_data(
        config.global_config.synthetic_data_dir,
        args.data_count,
        debugging=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    train_ds = PictureDataset(X_train, y_train)
    test_ds = PictureDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=32, shuffle=False, num_workers=4
    )

    # ------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = args.epochs
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            pred_points = model(features)  # [B, 4, 2]
            true_points = labels.clone()  # .view(-1, 3, 3) # [B, 3, 3]
            true_points[:, :, 0] /= config.generator.base_image_width
            true_points[:, :, 1] /= config.generator.base_image_height

            # loss = F.mse_loss(pred_points, true_points)  # Loss function
            loss = criterion(pred_points, true_points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### LOGGING
            elapsed = time.perf_counter() - start_time
            completed_batches = epoch * batch_idx + batch_idx + 1
            batches_per_second = completed_batches / elapsed
            remaining_batches = num_epochs * len(train_loader) - completed_batches
            eta = remaining_batches / batches_per_second

            print(
                # f"\r"
                f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                f" | Batch: {batch_idx + 1:03d}/{len(train_loader):03d}"
                f" | Loss: {loss.item():.6f}"
                f" | ETA: {format_duration(eta)}",
                # end="",
                flush=True,
            )

        # model.eval()
        # Optional model evaluation
        # compute_loss_on_testset(model, test_loader, criterion, config)

    compute_loss_on_testset(model, test_loader, criterion, config)

    # Save model
    model_dir = config.global_config.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"model_{get_datetime_str()}.pth"
    model_path = model_dir / model_filename
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    main()
