"""On-demand MNIST corruption with a configurable noise mixture."""

import math

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class NoisyMNISTDataset(Dataset):
    """Return (corrupted, clean) images in [0, 1].

    Training redraws noise on every access. Testing uses seed + image index
    so comparisons use identical corruptions without caching noisy images.
    Each image has samples_per_image virtual slots with independent corruptions.
    Probabilities select one recipe per slot, rather than exact dataset counts.
    """

    DEFAULT_NOISE_PROBS = {
        "gaussian": 0.5,
        "salt_pepper": 0.5,
        "speckle": 0.0,
        "mixed": 0.0,
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        noise_probs=None,
        seed: int = 42,
        samples_per_image: int = 1,
    ):
        if (
            isinstance(samples_per_image, bool)
            or not isinstance(samples_per_image, int)
            or samples_per_image < 1
        ):
            raise ValueError("samples_per_image must be a positive integer")
        self.samples_per_image = samples_per_image
        self.train = train
        self.seed = seed
        self.noise_probs = dict(
            self.DEFAULT_NOISE_PROBS if noise_probs is None else noise_probs
        )
        if not self.noise_probs or any(
            name not in self.DEFAULT_NOISE_PROBS for name in self.noise_probs
        ):
            raise ValueError("noise_probs must contain supported noise names")
        weights = list(self.noise_probs.values())
        if any(not math.isfinite(p) or p < 0 for p in weights) or not math.isclose(
            sum(weights), 1.0, abs_tol=1e-6
        ):
            raise ValueError(
                "Noise probabilities must be finite, nonnegative and sum to 1"
            )
        self.noise_names = list(self.noise_probs)
        self.noise_weights = torch.tensor(weights, dtype=torch.float64)
        self.data = MNIST(root=root, train=train, download=True, transform=ToTensor())

    def __len__(self):
        return len(self.data) * self.samples_per_image

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError("Dataset index out of range")
        # Each source image occupies samples_per_image virtual sample slots.
        img, _ = self.data[idx // self.samples_per_image]
        generator = (
            None if self.train else torch.Generator().manual_seed(self.seed + idx)
        )
        choice = torch.multinomial(self.noise_weights, 1, generator=generator).item()
        noisy_img = self._corrupt(img, self.noise_names[choice], generator)
        return noisy_img.clamp(0, 1), img

    @staticmethod
    def _uniform(low, high, generator):
        return low + (high - low) * torch.rand((), generator=generator).item()

    def _corrupt(self, img, kind, generator):
        if kind == "gaussian":
            sigma = self._uniform(0.05, 0.5, generator)
            return img + sigma * torch.randn(img.shape, generator=generator)
        if kind == "salt_pepper":
            return self._salt_pepper(img, 0.01, 0.08, generator)
        if kind == "speckle":
            sigma = self._uniform(0.10, 0.30, generator)
            return img + img * sigma * torch.randn(img.shape, generator=generator)
        if kind == "mixed":
            sigma = self._uniform(0.03, 0.12, generator)
            noisy = img + sigma * torch.randn(img.shape, generator=generator)
            return self._salt_pepper(noisy, 0.01, 0.04, generator)
        raise ValueError(f"Unsupported noise type: {kind}")

    def _salt_pepper(self, img, low, high, generator):
        probability = self._uniform(low, high, generator)
        mask = torch.rand(img.shape, generator=generator)
        noisy = img.clone()
        noisy[mask < probability / 2] = 0
        noisy[mask > 1 - probability / 2] = 1
        return noisy
