from __future__ import annotations

import os
from typing import Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class VPDiffusion(nn.Module):
    """Variance-preserving DDPM with DPS posterior sampling.

    Training objective:
        x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps
        minimize ||eps - eps_theta(x_t, t)||^2

    Score implied by the epsilon model:
        score_theta(x_t, t) = grad log p_t(x_t)
                           = -eps_theta(x_t, t) / sqrt(1-alpha_bar_t)

    DPS posterior correction for y = A x_0 + eta, eta ~ N(0, sigma_y^2 I):
        grad log p(y | x_t) approx grad_x_t log p(y | x0_hat(x_t))
        x_{t-1} mean <- DDPM_mean + zeta * posterior_var_t * grad log p(y | x_t)
    """

    def __init__(
        self,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        betas = make_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer("sqrt_recipm1_alpha_bars", torch.sqrt(1.0 / alpha_bars - 1.0))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alpha_bars, timesteps, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bars, timesteps, x0.shape) * noise
        )

    def predict_x0_from_eps(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        eps: torch.Tensor,
        *,
        clip: bool = True,
    ) -> torch.Tensor:
        x0 = (
            extract(self.sqrt_recip_alpha_bars, timesteps, xt.shape) * xt
            - extract(self.sqrt_recipm1_alpha_bars, timesteps, xt.shape) * eps
        )
        return torch.clamp(x0, -4.0, 4.0) if clip else x0

    def score_from_eps(self, eps: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        sigma_t = extract(self.sqrt_one_minus_alpha_bars, timesteps, x_shape)
        return -eps / torch.clamp(sigma_t, min=1e-8)

    def q_posterior_mean(self, x0_hat: torch.Tensor, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.posterior_mean_coef1, timesteps, xt.shape) * x0_hat
            + extract(self.posterior_mean_coef2, timesteps, xt.shape) * xt
        )

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        batch = x0.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, timesteps, noise)
        pred_noise = model(xt, timesteps)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddpm_sample(self, model: nn.Module, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        xt = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = model(xt, timesteps)
            x0_hat = self.predict_x0_from_eps(xt, timesteps, eps)
            mean = self.q_posterior_mean(x0_hat, xt, timesteps)
            var = extract(self.posterior_variance, timesteps, xt.shape)
            if step > 0:
                xt = mean + torch.sqrt(var) * torch.randn_like(xt)
            else:
                xt = mean
        return xt

    def dps_sample(
        self,
        model: nn.Module,
        measurement: torch.Tensor,
        *,
        operator: Callable[[torch.Tensor], torch.Tensor] | None = None,
        noise_std: float = 0.1,
        dps_scale: float = 1.0,
        sampler_noise_scale: float = 1.0,
        initial: torch.Tensor | None = None,
        return_trace: bool = False,
        show_progress: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
        """Posterior sample with diffusion posterior sampling.

        measurement is y. operator is A. If operator is None, A is identity.
        noise_std is sigma_y in y = A x0 + eta.
        """
        if operator is None:
            operator = lambda x: x

        model_was_training = model.training
        model.eval()
        sigma2 = max(float(noise_std) ** 2, 1e-8)
        xt = torch.randn_like(measurement) if initial is None else initial.clone()
        data_losses: list[float] = []

        steps = range(self.timesteps - 1, -1, -1)
        if show_progress:
            steps = tqdm(steps, desc="DPS reverse", leave=False)

        for step in steps:
            timesteps = torch.full((xt.shape[0],), step, device=xt.device, dtype=torch.long)
            xt_req = xt.detach().requires_grad_(True)
            eps = model(xt_req, timesteps)
            x0_hat = self.predict_x0_from_eps(xt_req, timesteps, eps)

            residual = operator(x0_hat) - measurement
            data_loss = residual.flatten(1).square().sum(dim=1).mean() / (2.0 * sigma2)
            grad_loss = torch.autograd.grad(data_loss, xt_req)[0]
            data_losses.append(float(data_loss.detach().cpu()))

            with torch.no_grad():
                mean = self.q_posterior_mean(x0_hat.detach(), xt_req.detach(), timesteps)
                var = extract(self.posterior_variance, timesteps, xt.shape)
                # grad log p(y|x_t) = -grad data_loss
                guided_mean = mean - dps_scale * var * grad_loss
                if step > 0:
                    xt = guided_mean + sampler_noise_scale * torch.sqrt(var) * torch.randn_like(xt)
                else:
                    xt = guided_mean

        if model_was_training:
            model.train()
        if return_trace:
            return xt, data_losses
        return xt
