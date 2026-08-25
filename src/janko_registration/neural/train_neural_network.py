from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from janko_registration.neural.helpers import PictureDataset, create_dataloaders
from janko_registration.utils import Config, format_duration, get_datetime_str


# This performed worst, likely due to me discarding locality with `AdaptiveAvgPool2d`.
# Best achieved loss: ~0.200
class NN_v1(torch.nn.Module):
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


# This performed best so far
# Best achieved loss: ~0.062 on the test dataset
class NN_v2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=7, padding=3),  # [B, 8, 540, 960]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # [B, 8, 270, 480]
            torch.nn.Conv2d(8, 16, kernel_size=7, padding=3),  # [B, 16, 270, 480]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # [B, 16, 135, 240]
            torch.nn.Conv2d(16, 32, kernel_size=7, padding=3),  # [B, 32, 135, 240]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # [B, 32, 67, 120]
            torch.nn.Conv2d(32, 32, kernel_size=7, padding=3),  # [B, 32, 67, 120]
            torch.nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Flatten(),  # [B, 257280]
            torch.nn.Linear(257280, 1024),  # [B, 1024]
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),  # [B, 256]
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),  # [B, 64]
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),  # [B, 8]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        return x.view(-1, 4, 2)


# This performed only slightly better than `NN_v1`
class NN_v3(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=7, padding=3),  # [B, 8, 540, 960]
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),  # [B, 16, 270, 480]
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                32, 64, kernel_size=3, padding=1, stride=2
            ),  # [B, 32, 135, 240]
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Flatten(),  # [B, 257280]
            torch.nn.Linear(17280, 1024),  # [B, 1024]
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),  # [B, 256]
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),  # [B, 64]
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),  # [B, 8]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        return x.view(-1, 4, 2)


def compute_loss_on_testset(
    model: torch.nn.Module,
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
    model = NN_v3()
    print("\nModel:")
    print(model)

    # ------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------

    train_loader, test_loader = create_dataloaders(
        N=args.data_count, batch_size=32, config=config
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
    model_filename = f"{model.__class__.__name__}_{get_datetime_str()}.pth"
    model_path = model_dir / model_filename
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    main()
