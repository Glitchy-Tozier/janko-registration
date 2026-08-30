from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from janko_registration.neural.helpers import PictureDataset, create_dataloaders
from janko_registration.utils import Config, format_duration, get_datetime_str


class NN_v4(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        if (
            config.heatmap.unpadded_resolution[0]
            != config.global_config.synthetic_resolution[0] // 2
            or config.heatmap.unpadded_resolution[1]
            != config.global_config.synthetic_resolution[1] // 2
        ):
            raise ValueError(
                f"The training data resolution currently is {config.global_config.synthetic_resolution}. "
                f"Thus, NN_v4 currently expects a heatmap resolution of {config.heatmap.unpadded_resolution}."
            )

        self.border = int(
            np.ceil(config.heatmap.sigma * config.heatmap.border_width_multiplier)
        )

        # The input will be downsampled by 2, so pad it by 2× the heatmap border.
        self.input_padding = 2 * self.border

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(config.features.count, 8, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            # [B, 4, 310, 520]
            # One channel per corner.
            torch.nn.Conv2d(128, 4, kernel_size=7, padding=3),
        )

        # Create gaussian kernel, used in the forward() method
        x = torch.arange(
            -self.border,
            self.border + 1,
            dtype=torch.float32,
        )
        gaussian_1d = torch.exp(-(x**2) / (2 * config.heatmap.sigma**2))
        gaussian_1d /= gaussian_1d.sum()

        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d /= gaussian_2d.sum()

        # One Gaussian kernel per corner.
        self.register_buffer(
            "gaussian_kernel",
            gaussian_2d[None, None].repeat(4, 1, 1, 1),
        )

    def forward(self, image_data: torch.Tensor) -> torch.Tensor:
        # [B, 3, 540, 960]
        image_data = F.pad(
            image_data,
            [self.input_padding] * 4,  # list of 4
        )

        # [B, 3, 620, 1040]
        x = self.layers(image_data)

        # Turn each corner's logits into a spatial probability distribution.
        x = F.softmax(x.flatten(2), dim=2).view_as(x)

        # Apply a fixed Gaussian blur with the same sigma as the target heatmaps.
        x = F.conv2d(
            x,
            self.gaussian_kernel,
            padding=self.border,
            groups=4,
        )

        # [B, 4, 310, 520]
        return x

    def model_output_to_corners(
        self, heatmap: torch.Tensor, config: Config
    ) -> torch.Tensor:
        """
        A normalizing function ensuring that every model produces the desired final metric:
        The bounding box corners of the visible piano, in the original scale.

        Converts [B, 4, H+2P, W+2P] model outputs into [B, 4, 2] corners with original scaling.

        The `heatmap` contains sigmoid values in [0, 1].
        """

        _batch_size, _num_corners, padded_height, padded_width = heatmap.shape

        unpadded_heatmap_width = padded_width - 2 * self.border
        unpadded_heatmap_height = padded_height - 2 * self.border

        # Convert sigmoid values into a spatial probability distribution.
        weights = heatmap.flatten(2)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        weights = weights.view_as(heatmap)

        # [H+2P, W+2P]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(padded_height, device=heatmap.device, dtype=heatmap.dtype),
            torch.arange(padded_width, device=heatmap.device, dtype=heatmap.dtype),
            indexing="ij",
        )

        # [B, 4]
        corner_x = (weights * grid_x).sum(dim=(2, 3)) - self.border
        corner_y = (weights * grid_y).sum(dim=(2, 3)) - self.border

        # [B, 4, 2]
        corners = torch.stack((corner_x, corner_y), dim=-1)

        # Heatmap coordinates → original image coordinates.
        corners[:, :, 0] *= config.generator.base_image_width / unpadded_heatmap_width
        corners[:, :, 1] *= config.generator.base_image_height / unpadded_heatmap_height

        return corners


def corners_to_heatmap(
    corners: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """
    Convert [B, 4, 2] corners into Gaussian heatmaps
    [B, 4, H+2P, W+2P].
    """

    unpadded_width, unpadded_height = config.heatmap.unpadded_resolution
    border = int(np.ceil(config.heatmap.sigma * config.heatmap.border_width_multiplier))

    corner_x = (
        corners[:, :, 0] * unpadded_width / config.generator.base_image_width + border
    )
    corner_y = (
        corners[:, :, 1] * unpadded_height / config.generator.base_image_height + border
    )

    grid_y, grid_x = torch.meshgrid(
        torch.arange(
            unpadded_height + 2 * border,
            device=corners.device,
            dtype=corners.dtype,
        ),
        torch.arange(
            unpadded_width + 2 * border,
            device=corners.device,
            dtype=corners.dtype,
        ),
        indexing="ij",
    )

    # [B, 4] → [B, 4, 1, 1]
    corner_x = corner_x[:, :, None, None]
    corner_y = corner_y[:, :, None, None]

    # [B, 4, H+2P, W+2P]
    distance_squared = (grid_x - corner_x) ** 2 + (grid_y - corner_y) ** 2

    # [B, 4, H+2P, W+2P]
    return torch.exp(-distance_squared / (2 * config.heatmap.sigma**2))


def compute_heatmap_conversion_loss(
    dataloader, model, iterations: int, cycles: int, config: Config
):
    print(
        "================================================================================="
    )
    for i, (_, l) in enumerate(dataloader):
        if i == cycles:
            break
        if i != 0:
            print(
                "---------------------------------------------------------------------------------"
            )
        criterion = torch.nn.L1Loss()
        corners = l.clone()
        for _ in range(iterations):
            loss = float(criterion(l, corners))
            print(round(loss, 3), end=" → ", flush=True)

            heatmap = corners_to_heatmap(corners, config)
            corners = model.model_output_to_corners(heatmap, config)
        print(round(loss, 3))
    print(
        "================================================================================="
    )


def compute_loss_on_testset(
    model: torch.nn.Module,
    dataloader: PictureDataset,
    criterion: torch.nn.L1Loss,
    config: Config,
) -> float:
    model.eval()
    mae = []

    # Loop through batches of the dataset to prevent memory issues
    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            prediction = model(features)  # [B, 4, H+2P, W+2P]
            pred_corners = model.model_output_to_corners(prediction, config)

        loss = criterion(pred_corners, labels)
        mae.append(loss)

    mean_mae = torch.stack(mae).mean().item()

    print("\nMean corner MAE over test dataset:", mean_mae, "(in px)")

    return mean_mae


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
    model = NN_v4(config)
    print("\nModel:")
    print(model)

    # ------------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------------

    train_loader, test_loader = create_dataloaders(
        N=args.data_count, batch_size=32, config=config
    )

    compute_heatmap_conversion_loss(train_loader, model, 8, 3, config)

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
            prediction = model(features)  # [B, 4, H+2P, W+2P]

            true_heatmap = corners_to_heatmap(labels, config)

            # Normalize both heatmaps into spatial probability distributions.
            prediction_probs = prediction.flatten(2)
            prediction_probs = prediction_probs / prediction_probs.sum(
                dim=2,
                keepdim=True,
            ).clamp_min(1e-8)

            target_probs = true_heatmap.flatten(2)
            target_probs = target_probs / target_probs.sum(
                dim=2,
                keepdim=True,
            ).clamp_min(1e-8)

            loss = F.kl_div(
                prediction_probs.clamp_min(1e-8).log(),
                target_probs,
                reduction="batchmean",
            )

            pred_corners = model.model_output_to_corners(prediction, config)
            true_corners = labels
            l1 = F.l1_loss(pred_corners, true_corners)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### LOGGING
            elapsed = time.perf_counter() - start_time
            completed_batches = epoch * len(train_loader) + batch_idx + 1
            batches_per_second = completed_batches / elapsed
            remaining_batches = num_epochs * len(train_loader) - completed_batches
            eta = remaining_batches / batches_per_second

            print(
                # f"\r"
                f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                f" | Batch: {batch_idx + 1:03d}/{len(train_loader):03d}"
                f" | Loss: {loss.item():.3f} – Pixel MAE: {l1:.3f}"
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
