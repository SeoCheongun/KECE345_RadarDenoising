# %%
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data_utils import (
    DEFAULT_SNR_LEVELS,
    RadarWindowDataset,
    add_awgn_torch,
    build_records,
    split_records,
)
from diffusion_model import EpsilonDenoiser1D
from diffusion_process import VPDiffusion


CONFIG = {
    "checkpoint": "runs/diffusion_dps/best.pt",
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "output_dir": "runs/diffusion_dps/eval",
    "auto_extract": True,
    "extractor": "",
    "batch_size": 4,
    "num_workers": 0,
    "split": "test",
    "max_eval": 32,
    "max_centroid": 512,
    "inverse_task": "denoise",  # denoise or inpaint
    "observed_fraction": 0.5,
    "dps_scale": 1.0,
    "start_from_measurement": False,
    "snr_levels": ",".join(str(x) for x in DEFAULT_SNR_LEVELS),
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DPS for radar inverse problems.")
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=value)
        else:
            parser.add_argument(f"--{key}", default=value)
    args, _ = parser.parse_known_args()
    return args


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_and_diffusion(checkpoint: dict, device: torch.device) -> tuple[EpsilonDenoiser1D, VPDiffusion]:
    model = EpsilonDenoiser1D(**checkpoint["model_kwargs"]).to(device)
    state = checkpoint.get("ema_model_state", checkpoint["model_state"])
    model.load_state_dict(state)
    model.eval()
    diffusion = VPDiffusion(**checkpoint["diffusion_kwargs"]).to(device)
    return model, diffusion


def tensor_features(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            x.mean(dim=-1),
            x.std(dim=-1),
            torch.sqrt(torch.mean(x.square(), dim=-1) + 1e-8),
            x.abs().amax(dim=-1),
        ],
        dim=1,
    )


def fit_centroids(loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    sums = None
    counts = torch.zeros(num_classes, device=device)
    for batch in tqdm(loader, desc="fit centroids", leave=False):
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        feat = tensor_features(x)
        if sums is None:
            sums = torch.zeros(num_classes, feat.shape[1], device=device)
        for cls in range(num_classes):
            mask = y == cls
            if mask.any():
                sums[cls] += feat[mask].sum(dim=0)
                counts[cls] += mask.sum()
    if sums is None:
        raise RuntimeError("No samples for centroid fitting.")
    return sums / counts.clamp_min(1.0)[:, None]


def classify_by_centroid(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    feat = tensor_features(x)
    dist = torch.cdist(feat, centroids)
    return dist.argmin(dim=1)


def make_mask_like(x: torch.Tensor, observed_fraction: float, generator: torch.Generator) -> torch.Tensor:
    if observed_fraction >= 1.0:
        return torch.ones_like(x)
    if observed_fraction <= 0.0:
        raise ValueError("observed_fraction must be > 0")
    return (torch.rand(x.shape, device=x.device, generator=generator) < observed_fraction).float()


def mse(x: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((x - target).square()).detach().cpu())



def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(args.data_root).resolve() / checkpoint_path
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, diffusion = build_model_and_diffusion(checkpoint, device)

    data_config = checkpoint["data_kwargs"]
    data_root = Path(args.data_root).resolve()
    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.is_absolute():
        extracted_dir = data_root / extracted_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = data_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records, label_names = build_records(
        data_root,
        extracted_dir,
        auto_extract=args.auto_extract,
        custom_extractor=args.extractor or None,
    )
    train_records, val_records, test_records = split_records(records, seed=args.seed)
    split_map = {"train": train_records, "val": val_records, "test": test_records}
    eval_records = split_map[args.split]

    train_dataset = RadarWindowDataset(
        train_records,
        window_size=data_config["window_size"],
        num_sensors=data_config["num_sensors"],
        windows_per_record=1,
        random_crop=False,
        seed=args.seed,
    )
    eval_dataset = RadarWindowDataset(
        eval_records,
        window_size=data_config["window_size"],
        num_sensors=data_config["num_sensors"],
        windows_per_record=1,
        random_crop=False,
        seed=args.seed,
    )

    if args.max_centroid > 0 and len(train_dataset) > args.max_centroid:
        train_dataset_for_centroids = Subset(train_dataset, list(range(args.max_centroid)))
    else:
        train_dataset_for_centroids = train_dataset
    if args.max_eval > 0 and len(eval_dataset) > args.max_eval:
        eval_dataset = Subset(eval_dataset, list(range(args.max_eval)))

    centroid_loader = DataLoader(
        train_dataset_for_centroids,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    centroids = fit_centroids(centroid_loader, len(label_names), device)

    snr_levels = [float(x.strip()) for x in args.snr_levels.split(",") if x.strip()]
    rng = torch.Generator(device=device).manual_seed(args.seed)
    rows = []

    print("")
    print("Evaluating inverse problem")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  task: {args.inverse_task}")
    print(f"  split: {args.split}, samples: {len(eval_dataset)}")
    print(f"  DPS scale: {args.dps_scale}")

    for snr_db in snr_levels:
        noisy_mse_sum = 0.0
        dps_mse_sum = 0.0
        clean_correct = 0
        noisy_correct = 0
        dps_correct = 0
        total = 0

        progress = tqdm(eval_loader, desc=f"SNR {snr_db:g} dB", leave=True)
        for batch in progress:
            x0 = batch["x"].to(device)
            labels = batch["label"].to(device)
            y_noisy, noise_std = add_awgn_torch(x0, snr_db, rng)

            if args.inverse_task == "denoise":
                measurement = y_noisy
                operator = lambda z: z
                baseline = y_noisy
            elif args.inverse_task == "inpaint":
                mask = make_mask_like(x0, args.observed_fraction, rng)
                measurement = mask * y_noisy
                operator = lambda z, mask=mask: mask * z
                baseline = measurement
            else:
                raise ValueError("--inverse_task must be denoise or inpaint")

            initial = measurement if args.start_from_measurement else None
            x_dps = diffusion.dps_sample(
                model,
                measurement,
                operator=operator,
                noise_std=noise_std,
                dps_scale=args.dps_scale,
                initial=initial,
            )

            batch_n = x0.shape[0]
            noisy_mse_sum += mse(baseline, x0) * batch_n
            dps_mse_sum += mse(x_dps, x0) * batch_n

            clean_pred = classify_by_centroid(x0, centroids)
            noisy_pred = classify_by_centroid(baseline, centroids)
            dps_pred = classify_by_centroid(x_dps, centroids)
            clean_correct += int((clean_pred == labels).sum().item())
            noisy_correct += int((noisy_pred == labels).sum().item())
            dps_correct += int((dps_pred == labels).sum().item())
            total += batch_n

            progress.set_postfix(
                noisy_mse=f"{noisy_mse_sum / total:.4f}",
                dps_mse=f"{dps_mse_sum / total:.4f}",
            )

        row = {
            "snr_db": snr_db,
            "samples": total,
            "noisy_mse": noisy_mse_sum / total,
            "dps_mse": dps_mse_sum / total,
            "clean_centroid_acc": clean_correct / total,
            "noisy_centroid_acc": noisy_correct / total,
            "dps_centroid_acc": dps_correct / total,
        }
        rows.append(row)
        print(row)

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
