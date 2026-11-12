"""CIFAR-10 UNet factory for IR-SDE conditioning.

The input is concatenated as [x_t, y] with x_t and y each 3 RGB channels,
so the total channel count is 6: 3 from x_t + 3 from y.

On a 3050 Ti 4G laptop, batch 128 usually uses about 1.5-2 GB VRAM;
if OOM occurs, reduce block_out_channels to (32, 64) or lower the batch size.
"""

from diffusers import UNet2DModel


def make_cifar_unet():
    """Return a compact UNet for CIFAR-10 [x_t, y] conditioning."""
    return UNet2DModel(
        sample_size=32,
        in_channels=6,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128),
        norm_num_groups=8,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
