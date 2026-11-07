"""Conditional predictor and deterministic IR-SDE restoration sampler.

The observed image y can come from any mixture configured in NoisyMNISTDataset.
Mixture probabilities affect data generation, not the Gaussian SDE coefficients.
"""

import math
from typing import List

import torch
from diffusers import DiffusionPipeline, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput


class IRSdeScheduler(SchedulerMixin, ConfigMixin):
    """常数回复速率 theta，离散时间步为 0,1,...,T。"""

    @register_to_config
    def __init__(self, T: int = 100, theta: float = 0.05, lam: float = 0.1,
                 prediction_type: str = "epsilon", clip_sample: bool = True):
        if isinstance(T, bool) or not isinstance(T, int) or T < 1:
            raise ValueError("T must be a positive integer.")
        if not all(math.isfinite(value) and value > 0 for value in (theta, lam)):
            raise ValueError("theta and lam must be positive.")
        if theta * T > 10:
            raise ValueError("theta*T must be <= 10 to avoid severe signal loss; use theta=5/T.")
        if prediction_type not in ("epsilon", "sample"):
            raise ValueError("prediction_type must be 'epsilon' or 'sample'")

        self.T = T
        t = torch.arange(T + 1, dtype=torch.float64)
        self.a = torch.exp(-theta * t)  # a_t = exp(-theta * t)
        self.v = lam**2 * (-torch.expm1(-2 * theta * t))  # v_t = lam^2 (1-a_t^2)
        self.train_timesteps = torch.arange(1, T + 1)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: int, y: torch.Tensor):
        """训练：在任一整数 t∈[1,T] 构造 x_t，网络预测传入的 noise。"""
        if x0.shape != noise.shape or x0.shape != y.shape:
            raise ValueError("x0, noise and y must have the same shape")
        t = torch.as_tensor(t, device=x0.device)
        if not torch.isfinite(t).all() or (t != t.round()).any():
            raise ValueError("t must contain finite integer timesteps")
        t = t.long().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x0.size(0))
        if t.numel() != x0.size(0) or (t < 1).any() or (t > self.T).any():
            raise ValueError("t 必须为 1,...,T，且长度等于 batch_size（或为标量）")
        shape = (-1,) + (1,) * (x0.ndim - 1)
        a = self.a[t.cpu()].to(x0).reshape(shape)
        v = self.v[t.cpu()].to(x0).reshape(shape)
        return y + a * (x0 - y) + v.sqrt() * noise

    def prediction_to_x0_noise(self, model_output, sample, t, y, clip=False):
        """Convert either network target to a consistent (x0, epsilon) pair."""
        t = torch.as_tensor(t, device=sample.device, dtype=torch.long).reshape(-1)
        shape = (-1,) + (1,) * (sample.ndim - 1)
        a = self.a.to(sample)[t].reshape(shape)
        sigma = self.v.to(sample)[t].sqrt().reshape(shape)
        if self.config.prediction_type == "sample":
            x0 = model_output
        else:
            x0 = y + (sample - y - sigma * model_output) / a
        if clip:
            x0 = x0.clamp(0, 1)
        # Recompute epsilon after clipping, so the DDIM path remains consistent.
        noise = (sample - y - a * (x0 - y)) / sigma
        return x0, noise

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        timesteps: List[int] = None,
        device: torch.device = None,
    ):
        """生成：选取 T→...→1，或者自定义降序节点；最终一定更新到 0。"""
        if timesteps is not None and num_inference_steps is not None:
            raise ValueError("Specify num_inference_steps or timesteps, not both")
        if timesteps is None:
            steps = self.T if num_inference_steps is None else num_inference_steps
            if not isinstance(steps, int) or not 1 <= steps <= self.T:
                raise ValueError("num_inference_steps 必须是 [1,T] 内的整数")
            nodes = torch.linspace(self.T, 1, steps).round().long()
        else:
            nodes = torch.as_tensor(timesteps)
            if not torch.isfinite(nodes).all() or (nodes != nodes.round()).any():
                raise ValueError("timesteps must contain finite integers")
            nodes = nodes.long()
            if nodes.ndim != 1 or nodes.numel() == 0 or nodes[0] != self.T:
                raise ValueError("timesteps 必须是以 T 开始的非空降序整数节点列表")
            if (
                (nodes < 1).any()
                or (nodes > self.T).any()
                or (nodes[:-1] <= nodes[1:]).any()
            ):
                raise ValueError("timesteps 必须严格递减，且每个节点在 [1,T] 内")
        self.timesteps = nodes.to(device)
        self.prev_timesteps = torch.cat(
            (self.timesteps[1:], self.timesteps.new_zeros(1))
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        y: torch.Tensor,
        prev_timestep: int,
    ):
        """从时间 i 跳至 j：利用预测噪声重构，j 可以小于 i-1。"""
        if any(not math.isfinite(float(t)) or float(t) != int(t)
               for t in (timestep, prev_timestep)):
            raise ValueError("Sampling timesteps must be finite integers")
        i, j = int(timestep), int(prev_timestep)
        if not 0 <= j < i <= self.T:
            raise ValueError("Require 0 <= prev_timestep < timestep <= T")
        if sample.shape != y.shape or sample.shape != model_output.shape:
            raise ValueError("sample, y and model_output must have the same shape")
        x0, noise = self.prediction_to_x0_noise(
            model_output, sample, i, y, clip=self.config.clip_sample
        )
        return SchedulerOutput(
            prev_sample=y + self.a[j].item() * (x0 - y)
            + self.v[j].sqrt().item() * noise
        )


def make_mnist_unet():
    """输入 [x_t, y]；单通道输出含义由 scheduler.prediction_type 指定。"""
    return UNet2DModel(
        sample_size=28,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64),
        norm_num_groups=8,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


class IRSdePipeline(DiffusionPipeline):
    """沿用 DDIMPipeline 的形式，但改用 IR-SDE 的条件初始化和跳步公式。"""

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: IRSdeScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        y,
        num_inference_steps: int = None,
        timesteps: List[int] = None,
        generator: torch.Generator = None,
        output_type: str = "pt",
    ):
        """y 为 [B,1,H,W]；可指定步数或明确列出生成节点。"""
        if output_type not in ("pt", "np", "pil"):
            raise ValueError("output_type 只能为 pt、np 或 pil")
        if y.ndim != 4 or y.shape[0] < 1 or y.shape[1] != 1:
            raise ValueError("y must have shape [B, 1, H, W] with B > 0")
        if not torch.isfinite(y).all():
            raise ValueError("y must contain finite pixels")
        self.unet.eval()
        y = y.to(device=self._execution_device, dtype=self.unet.dtype)
        self.scheduler.set_timesteps(num_inference_steps, timesteps, device=y.device)

        # x_T 近似服从 N(y, v_T I)，只有 a_T 很小时近似才好。
        noise = torch.randn(
            y.shape, device=y.device, dtype=y.dtype, generator=generator
        )
        x = y + self.scheduler.v[-1].sqrt().item() * noise

        for t, prev_t in self.progress_bar(
            zip(self.scheduler.timesteps, self.scheduler.prev_timesteps),
            total=len(self.scheduler.timesteps),
        ):
            prediction = self.unet(torch.cat([x, y], dim=1), t).sample
            x = self.scheduler.step(prediction, t, x, y, prev_t).prev_sample

        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite sampling result before clamping")
        x = x.clamp(0, 1)  # MNIST 数据范围 [0,1]
        if output_type != "pt":
            x = x.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                x = self.numpy_to_pil(x)
        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=x)
