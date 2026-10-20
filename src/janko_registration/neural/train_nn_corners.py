from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from janko_registration.neural.helpers import (
    compute_corner_loss_on_dataset,
    create_dataloaders,
    get_device,
    save_model,
)
from janko_registration.utils import Config, format_duration


class CornerModelBase(torch.nn.Module):
    """
    A base class to reduce redundancy and provide reoccuring methods.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        return x.view(-1, 4, 2)

    def model_output_to_corners(self, x: torch.Tensor, config: Config) -> torch.Tensor:
        """
        A normalizing function ensuring that every model produces the desired final metric:
        The bounding box corners of the visible piano, in the original scale.
        """
        x_scaled_up = x.clone()
        x_scaled_up[:, :, 0] *= config.generator.base_image_width
        x_scaled_up[:, :, 1] *= config.generator.base_image_height
        return x_scaled_up


# This performed worst, likely due to me discarding locality with `AdaptiveAvgPool2d`.
# Best achieved loss: ~0.200
class NN_v1(CornerModelBase):
    def __init__(self, config: Config):
        super().__init__()

        NUM_OUTPUTS = 8  # 2 x 4 coordinates

        self.features = torch.nn.Sequential(
            # Input:
            #   batch x 3 x 180 x 320
            #
            # Output:
            #   batch x 32 x 180 x 320
            torch.nn.Conv2d(
                config.features.count, out_channels=32, kernel_size=3, padding=1
            ),
            torch.nn.ReLU(),
            # batch x 32 x 90 x 160
            torch.nn.MaxPool2d(kernel_size=2),
            # batch x 64 x 90 x 160
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # batch x 64 x 45 x 80
            torch.nn.MaxPool2d(kernel_size=2),
            # batch x 128 x 45 x 80
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # batch x 128 x 22 x 40
            torch.nn.MaxPool2d(kernel_size=2),
            # batch x 256 x 22 x 40
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
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


# This performed best so far
# Best achieved loss: ~0.062 on the test dataset
class NN_v2(CornerModelBase):
    def __init__(self, config: Config):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(config.features.count, 8, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, kernel_size=7, padding=3),
            torch.nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Flatten(),  # [B, 257280]
            torch.nn.LazyLinear(1024),  # [B, 1024]
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),  # [B, 256]
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),  # [B, 64]
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),  # [B, 8]
        )


# This performed only slightly better than `NN_v1`
class NN_v3(CornerModelBase):
    def __init__(self, config: Config):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(config.features.count, 16, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
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


# Maybe more linear Layers are what's needed?
class NN_v5(CornerModelBase):
    def __init__(self, config: Config):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(config.features.count, 16, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 32, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 32, kernel_size=7, padding=3),
            torch.nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(257280, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
        )


def main() -> None:
    device = get_device()
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    elif device.type == "xpu":
        print("GPU:", torch.xpu.get_device_name(device))

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
    model = NN_v2(config).to(device)
    print("\nModel:")
    print(model)
    print("Model device:", next(model.parameters()).device)

    # ------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------

    train_loader, test_loader = create_dataloaders(
        N=args.data_count, batch_size=32, config=config
    )

    # ------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = args.epochs
    start_time = time.perf_counter()
    epoch_losses = []
    best_epoch_loss = float("inf")

    try:
        for epoch in range(num_epochs):
            print()
            model.train()
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)

                prediction = model(features)  # [B, 4, 2]

                pred_corners = model.model_output_to_corners(prediction, config)
                true_corners = labels.clone()  # .view(-1, 3, 3) # [B, 3, 3]

                # loss = F.mse_loss(true_corners, pred_corners)  # Loss function
                loss = criterion(true_corners, pred_corners)

                l1 = F.l1_loss(true_corners, pred_corners)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### LOGGING
                elapsed = time.perf_counter() - start_time
                completed_batches = epoch * len(train_loader) + batch_idx + 1
                batches_per_second = completed_batches / elapsed
                remaining_batches = num_epochs * len(train_loader) - completed_batches
                eta = remaining_batches / batches_per_second

                if (batch_idx + 1) % 50 == 0:
                    print(
                        # f"\r"
                        f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                        f" | Batch: {batch_idx + 1:03d}/{len(train_loader):03d}"
                        f" | Loss: {loss.item():.3f} – Pixel MAE: {l1:.3f}"
                        f" | ETA: {format_duration(eta)}",
                        # end="",
                        flush=True,
                    )

            epoch_test_loss = compute_corner_loss_on_dataset(
                test_loader, model, device, config
            )
            epoch_losses.append(epoch_test_loss)

            if epoch_test_loss < best_epoch_loss - 1 and epoch_test_loss < 30:
                best_epoch_loss = epoch_test_loss
                save_model(model, config)

    except KeyboardInterrupt:
        compute_corner_loss_on_dataset(test_loader, model, device, config)
        save_model(model, config)

    print("\nCorner losses across all batches:")
    print(epoch_losses)


if __name__ == "__main__":
    main()
