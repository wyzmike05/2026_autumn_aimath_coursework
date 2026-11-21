"""CIFAR-10 RGB adapter for the shared IR-SDE pipeline."""

import torch
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from ir_sde.model import IRSdePipeline


class CIFAR10IRSdePipeline(IRSdePipeline):
    """Reuse the shared scheduler/pipeline interface with RGB input validation."""

    @torch.no_grad()
    def __call__(
        self,
        y,
        num_inference_steps: int = None,
        timesteps=None,
        generator: torch.Generator = None,
        output_type: str = "pt",
    ):
        if output_type not in ("pt", "np", "pil"):
            raise ValueError("output_type 只能为 pt、np 或 pil")
        if y.ndim != 4 or y.shape[0] < 1 or y.shape[1] != 3:
            raise ValueError("y must have shape [B, 3, H, W] with B > 0")
        if not torch.isfinite(y).all():
            raise ValueError("y must contain finite pixels")

        self.unet.eval()
        y = y.to(device=self._execution_device, dtype=self.unet.dtype)
        self.scheduler.set_timesteps(
            num_inference_steps,
            timesteps,
            device=y.device,
        )
        noise = torch.randn(
            y.shape,
            device=y.device,
            dtype=y.dtype,
            generator=generator,
        )
        x = y + self.scheduler.v[-1].sqrt().item() * noise

        for timestep, prev_timestep in self.progress_bar(
            zip(self.scheduler.timesteps, self.scheduler.prev_timesteps),
            total=len(self.scheduler.timesteps),
        ):
            prediction = self.unet(torch.cat([x, y], dim=1), timestep).sample
            x = self.scheduler.step(
                prediction,
                timestep,
                x,
                y,
                prev_timestep,
            ).prev_sample

        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite sampling result before clamping")
        x = x.clamp(0, 1)
        if output_type != "pt":
            x = x.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                x = self.numpy_to_pil(x)
        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=x)
