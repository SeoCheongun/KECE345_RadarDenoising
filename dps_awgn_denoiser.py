# %%
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

CONFIG: dict[str, Any] = {
    "mode": "train_prior",  # train_prior, check, export, smoke_test
    "clean_dir": "extracted_data",
    "awgn_metadata": "extracted_data_awgn/metadata.csv",
    "output_dir": "runs/awgn_dps_full102",
    "checkpoint": "",
    "export_dir": "denoised_data_dps_full102",
    "snrs": "-10,-5,0,5,10",
    "labels": "",
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "window_size": 512,
    "windows_per_record": 4,
    "export_stride": 512,
    "timesteps": 100,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "base_channels": 96,
    "num_blocks": 6,
    "time_dim": 256,
    "epochs": 30,
    "batch_size": 4,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "dps_scale": 1.0,
    "dps_grad_clip": 1.0,
    "sampler_noise_scale": 0.0,
    "start_from_measurement": False,
    "max_records": 0,
    "max_train_items": 0,
    "max_eval_windows": 50,
    "max_export_files": 0,
    "max_export_windows_per_file": 0,
    "plot_count": 10,
    "show_plots_inline": True,
    "stft_window": 128,
    "stft_overlap": 0.90,
    "stft_nfft": 256,
    "spectrogram_db_floor": -45.0,
    "compress_mat": False,
    "overwrite": True,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0,
    "num_threads": 0,
}


SMOKE_CONFIG: dict[str, Any] = {
    "mode": "smoke_test",
    "output_dir": "runs/awgn_dps_full102_smoke",
    "checkpoint": "runs/awgn_dps_full102_smoke/best.pt",
    "max_records": 80,
    "max_train_items": 4,
    "max_eval_windows": 1,
    "epochs": 1,
    "batch_size": 1,
    "timesteps": 8,
    "base_channels": 16,
    "num_blocks": 2,
    "time_dim": 64,
    "plot_count": 1,
    "show_plots_inline": True,
    "device": "cpu",
    "num_threads": 4,
}


@dataclass(frozen=True)
class CleanRecord:
    path: Path
    label: str


@dataclass(frozen=True)
class AwgnPair:
    clean_path: Path
    noisy_path: Path
    label: str
    snr_db: float
    repeat: int


def as_namespace(config: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**config)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-102-channel clean-prior diffusion + DPS AWGN denoiser.")
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=value)
        else:
            parser.add_argument(f"--{key}", default=value)
    return parser.parse_args()


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def parse_float_set(text: str) -> set[float]:
    return {round(float(part.strip()), 6) for part in text.split(",") if part.strip()}


def parse_str_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def snr_tag(snr_db: float) -> str:
    value = int(snr_db) if float(snr_db).is_integer() else snr_db
    return f"snr_m{abs(value)}" if value < 0 else f"snr_{value}"


def build_clean_records(clean_dir: Path, labels: str = "", max_records: int = 0) -> list[CleanRecord]:
    label_filter = parse_str_set(labels)
    records: list[CleanRecord] = []
    for label_dir in sorted(path for path in clean_dir.iterdir() if path.is_dir()):
        label = label_dir.name
        if label_filter and label not in label_filter:
            continue
        for mat_path in sorted(label_dir.rglob("*.mat")):
            records.append(CleanRecord(path=mat_path, label=label))
    if max_records > 0:
        records = records[:max_records]
    if not records:
        raise RuntimeError(f"No clean .mat files found under {clean_dir}")
    return records


def build_awgn_pairs(args: argparse.Namespace, root: Path) -> list[AwgnPair]:
    metadata_path = resolve_path(args.awgn_metadata, root)
    snr_filter = parse_float_set(args.snrs)
    label_filter = parse_str_set(args.labels)
    pairs: list[AwgnPair] = []
    with metadata_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            snr_db = float(row["snr_db"])
            label = row["label"]
            if snr_filter and round(snr_db, 6) not in snr_filter:
                continue
            if label_filter and label not in label_filter:
                continue
            pairs.append(
                AwgnPair(
                    clean_path=resolve_path(row["source_file"], root),
                    noisy_path=resolve_path(row["output_file"], root),
                    label=label,
                    snr_db=snr_db,
                    repeat=int(row["repeat"]),
                )
            )
    if args.max_records > 0:
        pairs = pairs[: args.max_records]
    if not pairs:
        raise RuntimeError("No AWGN pair matched the current config.")
    return pairs


def split_by_key(items: list[Any], key_fn, train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Any], list[Any], list[Any]]:
    keys = sorted({str(key_fn(item)) for item in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_train = int(round(len(keys) * train_ratio))
    n_val = int(round(len(keys) * val_ratio))
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])
    train = [item for item in items if str(key_fn(item)) in train_keys]
    val = [item for item in items if str(key_fn(item)) in val_keys]
    test = [item for item in items if str(key_fn(item)) in test_keys]
    return train, val, test

def load_signal(path: Path) -> np.ndarray:
    data = loadmat(path)
    if "received_time_domain_signal" not in data:
        keys = [key for key in data if not key.startswith("__")]
        raise KeyError(f"{path} does not contain received_time_domain_signal. Keys: {keys}")
    signal = np.asarray(data["received_time_domain_signal"])
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    elif signal.ndim > 2:
        signal = signal.reshape(signal.shape[0], -1)
    return signal


def crop_or_pad(signal: np.ndarray, start: int, window_size: int) -> tuple[np.ndarray, int]:
    window = np.zeros((signal.shape[0], window_size), dtype=signal.dtype)
    end = min(signal.shape[1], start + window_size)
    actual = max(0, end - start)
    if actual > 0:
        window[:, :actual] = signal[:, start:end]
    return window, actual


def window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, max(1, stride)))
    final = length - window_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def clean_rms(window: np.ndarray) -> float:
    real_power = np.mean(window.real.astype(np.float64) ** 2)
    imag_power = np.mean(window.imag.astype(np.float64) ** 2)
    return float(math.sqrt(max((real_power + imag_power) / 2.0, 1e-12)))


def estimate_clean_rms_from_noisy(noisy_window: np.ndarray, snr_db: float) -> float:
    snr_linear = 10.0 ** (snr_db / 10.0)
    return clean_rms(noisy_window) / math.sqrt(1.0 + 1.0 / snr_linear)


def complex_to_tensor(window: np.ndarray, scale: float) -> torch.Tensor:
    stacked = np.concatenate([window.real, window.imag], axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(stacked / max(float(scale), 1e-8))


def tensor_to_complex(tensor: torch.Tensor, rows: int, scale: float, dtype: np.dtype) -> np.ndarray:
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False) * float(scale)
    real = array[:rows]
    imag = array[rows : 2 * rows]
    return (real + 1j * imag).astype(dtype, copy=False)


def normalized_awgn_std(snr_db: float) -> float:
    return float(10.0 ** (-snr_db / 20.0))

class CleanPriorWindowDataset(Dataset):
    def __init__(
        self,
        records: list[CleanRecord],
        *,
        window_size: int,
        windows_per_record: int,
        random_crop: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.window_size = window_size
        self.windows_per_record = max(1, windows_per_record)
        self.random_crop = random_crop
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_record

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index // self.windows_per_record]
        window_index = index % self.windows_per_record
        signal = load_signal(record.path)
        max_start = max(0, signal.shape[1] - self.window_size)
        if self.random_crop and max_start > 0:
            rng = np.random.default_rng(self.seed + index)
            start = int(rng.integers(0, max_start + 1))
        elif max_start > 0 and self.windows_per_record > 1:
            start = int(round(max_start * window_index / max(1, self.windows_per_record - 1)))
        else:
            start = max_start // 2 if max_start > 0 else 0
        window, _ = crop_or_pad(signal, start, self.window_size)
        scale = clean_rms(window)
        return {
            "x": complex_to_tensor(window, scale),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "path": str(record.path),
            "label": record.label,
            "start": start,
            "rows": signal.shape[0],
        }


class AwgnWindowDataset(Dataset):
    def __init__(self, pairs: list[AwgnPair], *, window_size: int, seed: int) -> None:
        self.pairs = pairs
        self.window_size = window_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        clean = load_signal(pair.clean_path)
        noisy = load_signal(pair.noisy_path)
        if clean.shape != noisy.shape:
            raise ValueError(f"Shape mismatch: {pair.clean_path} {clean.shape} vs {pair.noisy_path} {noisy.shape}")
        max_start = max(0, clean.shape[1] - self.window_size)
        start = max_start // 2 if max_start > 0 else 0
        clean_window, _ = crop_or_pad(clean, start, self.window_size)
        noisy_window, _ = crop_or_pad(noisy, start, self.window_size)
        scale = estimate_clean_rms_from_noisy(noisy_window, pair.snr_db)
        return {
            "clean": complex_to_tensor(clean_window, scale),
            "noisy": complex_to_tensor(noisy_window, scale),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "snr": torch.tensor(pair.snr_db, dtype=torch.float32),
            "label": pair.label,
            "start": start,
            "clean_path": str(pair.clean_path),
            "noisy_path": str(pair.noisy_path),
            "repeat": pair.repeat,
            "rows": clean.shape[0],
        }


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        args = timesteps.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.time_proj = nn.Linear(time_dim, channels)
        self.net = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            group_norm(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = x + self.time_proj(time_emb)[:, :, None]
        return x + self.net(h)


class EpsilonDenoiser1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_blocks: int,
        time_dim: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList([ResidualBlock1D(base_channels, time_dim, kernel_size) for _ in range(num_blocks)])
        self.out = nn.Sequential(
            group_norm(base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timesteps)
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out(h)


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class VPDiffusion(nn.Module):
    def __init__(self, timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer("sqrt_recipm1_alpha_bars", torch.sqrt(1.0 / alpha_bars - 1.0))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars))
        self.register_buffer("posterior_mean_coef2", (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars))

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alpha_bars, timesteps, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bars, timesteps, x0.shape) * noise
        )

    def predict_x0_from_eps(self, xt: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        x0 = (
            extract(self.sqrt_recip_alpha_bars, timesteps, xt.shape) * xt
            - extract(self.sqrt_recipm1_alpha_bars, timesteps, xt.shape) * eps
        )
        return torch.clamp(x0, -6.0, 6.0)

    def posterior_mean(self, x0_hat: torch.Tensor, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.posterior_mean_coef1, timesteps, xt.shape) * x0_hat
            + extract(self.posterior_mean_coef2, timesteps, xt.shape) * xt
        )

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        batch = x0.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, timesteps, noise)
        pred = model(xt, timesteps)
        return F.mse_loss(pred, noise)

    def dps_sample(
        self,
        model: nn.Module,
        measurement: torch.Tensor,
        *,
        noise_std: torch.Tensor,
        dps_scale: float,
        dps_grad_clip: float,
        sampler_noise_scale: float,
        start_from_measurement: bool,
    ) -> torch.Tensor:
        model.eval()
        xt = measurement.clone() if start_from_measurement else torch.randn_like(measurement)
        sigma2 = noise_std.float().square().reshape(-1, 1, 1).clamp_min(1e-8)

        for step in range(self.timesteps - 1, -1, -1):
            timesteps = torch.full((xt.shape[0],), step, device=xt.device, dtype=torch.long)
            xt_req = xt.detach().requires_grad_(True)
            eps = model(xt_req, timesteps)
            x0_hat = self.predict_x0_from_eps(xt_req, timesteps, eps)
            residual = x0_hat - measurement
            data_loss = (residual.square() / (2.0 * sigma2)).flatten(1).mean()
            grad = torch.autograd.grad(data_loss, xt_req)[0]
            if dps_grad_clip > 0:
                grad = torch.clamp(grad, -dps_grad_clip, dps_grad_clip)

            with torch.no_grad():
                mean = self.posterior_mean(x0_hat.detach(), xt_req.detach(), timesteps)
                var = extract(self.posterior_variance, timesteps, xt.shape)
                guided_mean = mean - dps_scale * var * grad
                if step > 0:
                    noise = torch.randn_like(xt) if sampler_noise_scale != 0 else 0.0
                    xt = guided_mean + sampler_noise_scale * torch.sqrt(var) * noise
                else:
                    xt = guided_mean
        return xt


def make_model(args: argparse.Namespace, in_channels: int, device: torch.device) -> EpsilonDenoiser1D:
    return EpsilonDenoiser1D(
        in_channels=in_channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        time_dim=args.time_dim,
    ).to(device)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
    in_channels: int,
    rows: int,
) -> None:
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_kwargs": {
            "in_channels": in_channels,
            "base_channels": args.base_channels,
            "num_blocks": args.num_blocks,
            "time_dim": args.time_dim,
        },
        "diffusion_kwargs": {
            "timesteps": args.timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
        },
        "data_kwargs": {
            "rows": rows,
            "in_channels": in_channels,
            "window_size": args.window_size,
            "all_channels": True,
        },
        "config": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def default_checkpoint(args: argparse.Namespace, root: Path) -> Path:
    if args.checkpoint:
        return resolve_path(args.checkpoint, root)
    return resolve_path(args.output_dir, root) / "best.pt"


def load_trained_prior(args: argparse.Namespace) -> tuple[EpsilonDenoiser1D, VPDiffusion, dict[str, Any]]:
    root = Path(".").resolve()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(default_checkpoint(args, root), device)
    model = EpsilonDenoiser1D(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    diffusion = VPDiffusion(**checkpoint["diffusion_kwargs"]).to(device)
    return model, diffusion, checkpoint

def train_prior(args: argparse.Namespace) -> None:
    root = Path(".").resolve()
    device = torch.device(args.device)
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clean_dir = resolve_path(args.clean_dir, root)
    records = build_clean_records(clean_dir, labels=args.labels, max_records=args.max_records)
    train_records, val_records, test_records = split_by_key(
        records, lambda item: item.path, args.train_ratio, args.val_ratio, args.seed
    )

    train_dataset = CleanPriorWindowDataset(
        train_records,
        window_size=args.window_size,
        windows_per_record=args.windows_per_record,
        random_crop=True,
        seed=args.seed,
    )
    val_dataset = CleanPriorWindowDataset(
        val_records,
        window_size=args.window_size,
        windows_per_record=1,
        random_crop=False,
        seed=args.seed,
    )
    if args.max_train_items > 0 and len(train_dataset) > args.max_train_items:
        train_dataset = Subset(train_dataset, list(range(args.max_train_items)))

    sample = train_dataset[0]
    rows = int(sample["rows"])
    in_channels = int(sample["x"].shape[0])
    if in_channels != rows * 2:
        raise RuntimeError("Channel mismatch. DPS version must use all complex rows as real+imag channels.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args, in_channels, device)
    diffusion = VPDiffusion(args.timesteps, args.beta_start, args.beta_end).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("")
    print("Training clean diffusion prior for DPS")
    print(f"  clean records: {len(records)}")
    print(f"  train windows: {len(train_dataset)}, val windows: {len(val_dataset)}, test records: {len(test_records)}")
    print(f"  rows used: {rows}/102, channels: {in_channels}, window: {args.window_size}")
    print(f"  device: {device}")

    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=True):
            x0 = batch["x"].to(device, non_blocking=True)
            loss = diffusion.training_loss(model, x0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_sum += float(loss.item()) * x0.shape[0]
            train_count += x0.shape[0]
        train_loss = train_sum / max(1, train_count)

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="validation", leave=False):
                x0 = batch["x"].to(device, non_blocking=True)
                loss = diffusion.training_loss(model, x0)
                val_sum += float(loss.item()) * x0.shape[0]
                val_count += x0.shape[0]
        val_loss = val_sum / max(1, val_count)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        save_checkpoint(output_dir / "latest.pt", model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss, args=args, in_channels=in_channels, rows=rows)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss, args=args, in_channels=in_channels, rows=rows)
            print(f"  saved best: {output_dir / 'best.pt'}")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def mse_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).square().flatten(1).mean(dim=1)


def db_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(numerator, 1e-30) / np.maximum(denominator, 1e-30))


def evaluate_dps_windows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(".").resolve()
    device = torch.device(args.device)
    model, diffusion, checkpoint = load_trained_prior(args)
    data_kwargs = checkpoint["data_kwargs"]
    pairs = build_awgn_pairs(args, root)
    _, _, test_pairs = split_by_key(pairs, lambda item: item.clean_path, args.train_ratio, args.val_ratio, args.seed)
    dataset = AwgnWindowDataset(test_pairs, window_size=int(data_kwargs["window_size"]), seed=args.seed)
    if args.max_eval_windows > 0 and len(dataset) > args.max_eval_windows:
        dataset = Subset(dataset, list(range(args.max_eval_windows)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("")
    print("Checking DPS AWGN denoising")
    print(f"  checkpoint: {default_checkpoint(args, root)}")
    print(f"  windows: {len(dataset)}")
    print(f"  rows used: {data_kwargs['rows']}/102")
    print(f"  device: {device}")

    rows: list[dict[str, Any]] = []
    index_offset = 0
    for batch in tqdm(loader, desc="DPS windows", leave=True):
        clean = batch["clean"].to(device)
        noisy = batch["noisy"].to(device)
        snr = batch["snr"].to(device)
        noise_std = torch.tensor([normalized_awgn_std(float(x)) for x in batch["snr"]], device=device)
        with torch.enable_grad():
            denoised = diffusion.dps_sample(
                model,
                noisy,
                noise_std=noise_std,
                dps_scale=args.dps_scale,
                dps_grad_clip=args.dps_grad_clip,
                sampler_noise_scale=args.sampler_noise_scale,
                start_from_measurement=args.start_from_measurement,
            )

        clean_power = clean.square().flatten(1).mean(dim=1).detach().cpu().numpy()
        noisy_mse = mse_tensor(noisy, clean).detach().cpu().numpy()
        denoised_mse = mse_tensor(denoised, clean).detach().cpu().numpy()
        noisy_snr = db_ratio(clean_power, noisy_mse)
        denoised_snr = db_ratio(clean_power, denoised_mse)
        gain = db_ratio(noisy_mse, denoised_mse)
        for i in range(clean.shape[0]):
            rows.append(
                {
                    "index": index_offset + i,
                    "snr_db": float(snr[i].detach().cpu()),
                    "label": str(batch["label"][i]),
                    "clean_path": str(batch["clean_path"][i]),
                    "noisy_path": str(batch["noisy_path"][i]),
                    "start": int(batch["start"][i]),
                    "scale": float(batch["scale"][i]),
                    "repeat": int(batch["repeat"][i]),
                    "noisy_mse": float(noisy_mse[i]),
                    "denoised_mse": float(denoised_mse[i]),
                    "noisy_snr_db": float(noisy_snr[i]),
                    "denoised_snr_db": float(denoised_snr[i]),
                    "snr_gain_db": float(gain[i]),
                }
            )
        index_offset += clean.shape[0]

    summary = summarize_by_snr(rows)
    return rows, summary


def summarize_by_snr(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["snr_db"])].append(row)
    summary = []
    for snr_db in sorted(grouped):
        items = grouped[snr_db]
        summary.append(
            {
                "snr_db": snr_db,
                "samples": len(items),
                "noisy_mse": float(np.mean([item["noisy_mse"] for item in items])),
                "denoised_mse": float(np.mean([item["denoised_mse"] for item in items])),
                "noisy_snr_db": float(np.mean([item["noisy_snr_db"] for item in items])),
                "denoised_snr_db": float(np.mean([item["denoised_snr_db"] for item in items])),
                "snr_gain_db": float(np.mean([item["snr_gain_db"] for item in items])),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def micro_doppler_spectrogram(signal: np.ndarray, *, window_size: int, overlap: float, nfft: int) -> np.ndarray:
    hop = max(1, int(round(window_size * (1.0 - overlap))))
    if signal.shape[1] < window_size:
        padded = np.zeros((signal.shape[0], window_size), dtype=signal.dtype)
        padded[:, : signal.shape[1]] = signal
        signal = padded
    starts = list(range(0, signal.shape[1] - window_size + 1, hop)) or [0]
    center = np.arange(window_size, dtype=np.float64) - (window_size - 1) / 2.0
    gaussian_window = np.exp(-0.5 * (center / (window_size / 6.0)) ** 2)
    frames = []
    for start in starts:
        frame = signal[:, start : start + window_size] * gaussian_window[None, :]
        spectrum = np.fft.fftshift(np.fft.fft(frame, n=nfft, axis=1), axes=1)
        frames.append(np.mean(np.abs(spectrum) ** 2, axis=0))
    return np.stack(frames, axis=1).astype(np.float32, copy=False)


def relative_db(spec: np.ndarray, floor_db: float) -> np.ndarray:
    db = 10.0 * np.log10(np.maximum(spec, 1e-30))
    db -= float(np.max(db))
    return np.maximum(db, floor_db)


def display_image_if_interactive(path: Path, args: argparse.Namespace) -> None:
    if not getattr(args, "show_plots_inline", False) or "get_ipython" not in globals():
        return
    try:
        from IPython.display import Image, display

        display(Image(filename=str(path)))
    except Exception as exc:
        print(f"Saved figure but could not display inline: {path} ({exc})")


def denoise_single_window_for_plot(
    args: argparse.Namespace,
    row: dict[str, Any],
    model: EpsilonDenoiser1D,
    diffusion: VPDiffusion,
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    data_kwargs = checkpoint["data_kwargs"]
    window_size = int(data_kwargs["window_size"])
    clean_full = load_signal(Path(row["clean_path"]))
    noisy_full = np.asarray(loadmat(row["noisy_path"])["received_time_domain_signal"])
    start = int(row["start"])
    clean_window, _ = crop_or_pad(clean_full, start, window_size)
    noisy_window, _ = crop_or_pad(noisy_full, start, window_size)
    scale = estimate_clean_rms_from_noisy(noisy_window, float(row["snr_db"]))
    measurement = complex_to_tensor(noisy_window, scale)[None].to(device)
    noise_std = torch.tensor([normalized_awgn_std(float(row["snr_db"]))], device=device)
    with torch.enable_grad():
        denoised_tensor = diffusion.dps_sample(
            model,
            measurement,
            noise_std=noise_std,
            dps_scale=args.dps_scale,
            dps_grad_clip=args.dps_grad_clip,
            sampler_noise_scale=args.sampler_noise_scale,
            start_from_measurement=args.start_from_measurement,
        )[0]
    denoised_window = tensor_to_complex(denoised_tensor, clean_window.shape[0], scale, noisy_window.dtype)
    return clean_window, noisy_window, denoised_window


def save_paper_visualizations(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    if not rows or args.plot_count <= 0:
        return
    root = Path(".").resolve()
    out_dir = resolve_path(args.output_dir, root)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model, diffusion, checkpoint = load_trained_prior(args)

    for plot_index, row in enumerate(rows[: args.plot_count]):
        clean_window, noisy_window, denoised_window = denoise_single_window_for_plot(args, row, model, diffusion, checkpoint)
        specs = [
            relative_db(
                micro_doppler_spectrogram(clean_window, window_size=args.stft_window, overlap=args.stft_overlap, nfft=args.stft_nfft),
                args.spectrogram_db_floor,
            ),
            relative_db(
                micro_doppler_spectrogram(noisy_window, window_size=args.stft_window, overlap=args.stft_overlap, nfft=args.stft_nfft),
                args.spectrogram_db_floor,
            ),
            relative_db(
                micro_doppler_spectrogram(denoised_window, window_size=args.stft_window, overlap=args.stft_overlap, nfft=args.stft_nfft),
                args.spectrogram_db_floor,
            ),
        ]
        titles = [
            "clean spectrogram",
            f"noisy spectrogram ({row['snr_db']:g} dB)",
            f"DPS denoised spectrogram (gain {row['snr_gain_db']:.2f} dB)",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
        for axis, spec, title in zip(axes, specs, titles):
            image = axis.imshow(
                spec,
                aspect="auto",
                origin="lower",
                cmap="jet",
                vmin=args.spectrogram_db_floor,
                vmax=0.0,
            )
            axis.set_title(title)
            axis.set_xlabel("STFT frame")
            axis.set_ylabel("Doppler bin")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="relative dB")
        fig.suptitle(f"Paper-style micro-Doppler spectrogram | rows used: {clean_window.shape[0]}/102")
        stem = f"dps_{plot_index:03d}_snr_{row['snr_db']:g}dB".replace("-", "m")
        figure_path = fig_dir / f"{stem}_paper_spectrogram.png"
        fig.savefig(figure_path, dpi=170)
        plt.close(fig)
        print(f"Saved figure: {figure_path}")
        display_image_if_interactive(figure_path, args)


def check_dps(args: argparse.Namespace) -> None:
    root = Path(".").resolve()
    out_dir = resolve_path(args.output_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate_dps_windows(args)
    write_csv(out_dir / "window_metrics.csv", rows)
    write_csv(out_dir / "summary_by_snr.csv", summary)
    (out_dir / "window_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary_by_snr.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_paper_visualizations(args, rows)

    print("")
    print("Summary by SNR")
    for row in summary:
        print(
            f"  {row['snr_db']:>5g} dB | noisy_mse={row['noisy_mse']:.5f} -> "
            f"dps_mse={row['denoised_mse']:.5f} | gain={row['snr_gain_db']:.2f} dB"
        )
    print(f"Saved DPS check outputs: {out_dir}")


def smoke_test(args: argparse.Namespace) -> None:
    config = vars(args).copy()
    config.update(SMOKE_CONFIG)
    smoke_args = as_namespace(config)
    root = Path(".").resolve()
    smoke_output = resolve_path(smoke_args.output_dir, root)
    smoke_check_output = smoke_output / "check"

    print("")
    print("Running Ctrl+Enter DPS smoke test")
    print("  1/2 tiny clean-prior training, rows 102/102")
    train_prior(smoke_args)

    print("")
    print("  2/2 DPS check and paper-style spectrogram")
    check_config = vars(smoke_args).copy()
    check_config["mode"] = "check"
    check_config["checkpoint"] = str(smoke_output / "best.pt")
    check_config["output_dir"] = str(smoke_check_output)
    check_args = as_namespace(check_config)
    check_dps(check_args)

    print("")
    print("Smoke test done")
    print(f"  checkpoint: {smoke_output / 'best.pt'}")
    print(f"  metrics: {smoke_check_output / 'summary_by_snr.csv'}")
    print(f"  figures: {smoke_check_output / 'figures'}")


def export_dps(args: argparse.Namespace) -> None:
    root = Path(".").resolve()
    device = torch.device(args.device)
    export_root = resolve_path(args.export_dir, root)
    export_root.mkdir(parents=True, exist_ok=True)
    model, diffusion, checkpoint = load_trained_prior(args)
    data_kwargs = checkpoint["data_kwargs"]
    window_size = int(data_kwargs["window_size"])
    awgn_root = resolve_path("extracted_data_awgn", root)
    pairs = build_awgn_pairs(args, root)
    if args.max_export_files > 0:
        pairs = pairs[: args.max_export_files]

    metrics: list[dict[str, Any]] = []
    for pair in tqdm(pairs, desc="export DPS .mat", leave=True):
        clean = load_signal(pair.clean_path)
        noisy_mat = loadmat(pair.noisy_path)
        noisy = np.asarray(noisy_mat["received_time_domain_signal"])
        if clean.shape != noisy.shape:
            raise ValueError(f"Shape mismatch: {pair.clean_path} vs {pair.noisy_path}")
        starts = window_starts(noisy.shape[1], window_size, args.export_stride)
        if args.max_export_windows_per_file > 0:
            starts = starts[: args.max_export_windows_per_file]
        accum = np.zeros(noisy.shape, dtype=np.complex128)
        weights = np.zeros(noisy.shape, dtype=np.float32)
        noise_std = torch.tensor([normalized_awgn_std(pair.snr_db)], device=device)

        for start in starts:
            noisy_window, actual = crop_or_pad(noisy, start, window_size)
            scale = estimate_clean_rms_from_noisy(noisy_window, pair.snr_db)
            measurement = complex_to_tensor(noisy_window, scale)[None].to(device)
            with torch.enable_grad():
                denoised_tensor = diffusion.dps_sample(
                    model,
                    measurement,
                    noise_std=noise_std,
                    dps_scale=args.dps_scale,
                    dps_grad_clip=args.dps_grad_clip,
                    sampler_noise_scale=args.sampler_noise_scale,
                    start_from_measurement=args.start_from_measurement,
                )[0]
            denoised_window = tensor_to_complex(denoised_tensor, noisy.shape[0], scale, noisy.dtype)
            accum[:, start : start + actual] += denoised_window[:, :actual].astype(np.complex128, copy=False)
            weights[:, start : start + actual] += 1.0

        mask = weights > 0
        denoised = noisy.copy()
        denoised[mask] = (accum[mask] / weights[mask]).astype(noisy.dtype, copy=False)
        rel = pair.noisy_path.resolve().relative_to(awgn_root.resolve())
        out_path = export_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_path}")
        payload = {key: value for key, value in noisy_mat.items() if not key.startswith("__")}
        payload["received_time_domain_signal"] = denoised
        payload["dps_snr_db"] = np.array([[pair.snr_db]], dtype=np.float64)
        payload["dps_repeat"] = np.array([[pair.repeat]], dtype=np.int32)
        payload["dps_rows_used"] = np.array([[noisy.shape[0]]], dtype=np.int32)
        savemat(out_path, payload, do_compression=args.compress_mat)

        noisy_mse = float(np.mean(np.abs(noisy[mask] - clean[mask]) ** 2))
        dps_mse = float(np.mean(np.abs(denoised[mask] - clean[mask]) ** 2))
        metrics.append(
            {
                "clean_file": str(pair.clean_path),
                "noisy_file": str(pair.noisy_path),
                "output_file": str(out_path),
                "snr_db": pair.snr_db,
                "repeat": pair.repeat,
                "rows_used": noisy.shape[0],
                "windows": len(starts),
                "noisy_mse": noisy_mse,
                "dps_mse": dps_mse,
                "mse_gain_db": 10.0 * math.log10(max(noisy_mse, 1e-30) / max(dps_mse, 1e-30)),
            }
        )
    (export_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved DPS denoised .mat files: {export_root}")



def main(config: dict[str, Any] | argparse.Namespace | None = None) -> None:
    if config is None:
        args = parse_cli_args()
    elif isinstance(config, argparse.Namespace):
        args = config
    else:
        merged = CONFIG.copy()
        merged.update(config)
        args = as_namespace(merged)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    if args.mode == "train_prior":
        train_prior(args)
    elif args.mode == "check":
        check_dps(args)
    elif args.mode == "export":
        export_dps(args)
    elif args.mode == "smoke_test":
        smoke_test(args)
    else:
        raise ValueError("mode must be train_prior, check, export, or smoke_test")


if __name__ == "__main__" and "get_ipython" not in globals():
    main()


# %%
# Ctrl+Enter smoke test cell.
if "get_ipython" in globals():
    CELL_CONFIG = CONFIG.copy()
    CELL_CONFIG.update(SMOKE_CONFIG)
    main(CELL_CONFIG)


# %%
# Ctrl+Enter real training cell.
if "get_ipython" in globals():
    CELL_CONFIG = CONFIG.copy()
    CELL_CONFIG["mode"] = "train_prior"
    CELL_CONFIG["output_dir"] = "runs/awgn_dps_full102"
    main(CELL_CONFIG)


# %%
# Ctrl+Enter check/visualize cell. Run after training creates best.pt.
if "get_ipython" in globals():
    CELL_CONFIG = CONFIG.copy()
    CELL_CONFIG["mode"] = "check"
    CELL_CONFIG["checkpoint"] = "runs/awgn_dps_full102/best.pt"
    CELL_CONFIG["output_dir"] = "runs/awgn_dps_full102/check"
    CELL_CONFIG["plot_count"] = 10
    main(CELL_CONFIG)


# %%
# Ctrl+Enter export cell. Run after the check figures look right.
if "get_ipython" in globals():
    CELL_CONFIG = CONFIG.copy()
    CELL_CONFIG["mode"] = "export"
    CELL_CONFIG["checkpoint"] = "runs/awgn_dps_full102/best.pt"
    CELL_CONFIG["export_dir"] = "denoised_data_dps_full102"
    main(CELL_CONFIG)
