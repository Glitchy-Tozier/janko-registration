from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from janko_registration.neural.helpers import (
    compute_loss_on_dataset,
    create_dataloaders,
    get_device,
    save_model,
)
from janko_registration.utils import Config, format_duration


class EdgeIntersectionModelBase(torch.nn.Module):
    """
    A base class to reduce redundancy and provide reoccuring methods.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.num_outputs = 4
        # self.x_res = config.generator.base_image_width
        self.y_res = config.generator.base_image_height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.output(x)
        x = x * self.y_res

        return x

    def model_output_to_points(self, x: torch.Tensor, config: Config) -> torch.Tensor:
        """
        Convert the predicted y-coordinates into the four edge intersection points.

        The points are ordered as:
            top-left
            top-right
            bottom-left
            bottom-right
        """
        # the y-coordinates are already properly scaled
        y_coordinates = x  # * config.generator.base_image_height
        x_coordinates = x.new_tensor(
            [
                0,
                config.generator.base_image_width,
                0,
                config.generator.base_image_width,
            ]
        ).expand(x.shape[0], -1)

        return torch.stack(
            (x_coordinates, y_coordinates),
            dim=-1,
        )


class EI_v01(EdgeIntersectionModelBase):
    def __init__(self, config: Config):
        super().__init__(config)

        kernel_size = 3
        padding = int((kernel_size - 1) / 2)
        dropout_rate = 0.1

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                config.features.count,
                8,
                kernel_size=kernel_size,
                padding=padding,
                stride=2,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=kernel_size, padding=padding, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            torch.nn.ReLU(),
        )

        self.output = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(256, self.num_outputs),
        )


def main() -> None:
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether additional output shall be shown.",
    )
    args = parser.parse_args()
    config = Config()
    device = get_device()

    if args.verbose:
        print("Using device:", device)

        if device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(device))
        elif device.type == "xpu":
            print("GPU:", torch.xpu.get_device_name(device))

    # ------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------

    torch.manual_seed(123)
    model = EI_v01(config).to(device)
    if args.verbose:
        print("\nModel:")
        print(model)
        print("Model device:", next(model.parameters()).device)

    # ------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------

    train_loader, test_loader = create_dataloaders(
        N=args.data_count, batch_size=32, print_output=args.verbose, config=config
    )

    # ------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------

    num_epochs = args.epochs

    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.SmoothL1Loss(beta=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    start_time = time.perf_counter()
    epoch_losses = []
    best_epoch_loss = float("inf")

    try:
        for epoch in range(num_epochs):
            if args.verbose:
                print()

            model.train()
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(device)
                labels = labels.to(device)

                prediction = model(features)  # [B, 4]

                loss = criterion(labels, prediction)
                l1 = F.l1_loss(labels, prediction)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### LOGGING
                if args.verbose and (batch_idx + 1) % 50 == 0:
                    elapsed = time.perf_counter() - start_time
                    completed_batches = epoch * len(train_loader) + batch_idx + 1
                    batches_per_second = completed_batches / elapsed
                    remaining_batches = (
                        num_epochs * len(train_loader) - completed_batches
                    )
                    eta = remaining_batches / batches_per_second

                    print(
                        # f"\r"
                        f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                        f" | Batch: {batch_idx + 1:03d}/{len(train_loader):03d}"
                        f" | LR: {optimizer.param_groups[0]['lr']:.1e}"
                        f" | Loss: {loss.item():.3f} – Pixel MAE: {l1:.3f}"
                        f" | ETA: {format_duration(eta)}",
                        # end="",
                        flush=True,
                    )

            epoch_test_loss = compute_loss_on_dataset(
                test_loader, model, device, False, args.verbose, config
            )
            epoch_losses.append(epoch_test_loss)

            if epoch_test_loss < best_epoch_loss - 1 and epoch_test_loss < 15:
                print("New best test set MAE loss:", epoch_test_loss)
                best_epoch_loss = epoch_test_loss
                save_model(model, config)

            scheduler.step()

    except KeyboardInterrupt:
        print(" → Detected CTRL+C → stopping execution, starting final test.")
        compute_loss_on_dataset(test_loader, model, device, False, True, config)

    if best_epoch_loss != epoch_losses[-1]:
        save_model(model, config)

    print("\ny-value losses across all batches:")
    print(epoch_losses)


if __name__ == "__main__":
    main()
