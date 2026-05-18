# %%
"""Train action classifier on DPS-denoised spectrograms.

Two phases:
  Phase 1 (precompute): raw .mat -> AWGN -> DPS denoise -> spectrogram -> save to disk
  Phase 2 (train): load precomputed spectrograms -> train classifier

If precomputed cache exists, Phase 1 is skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from classifier_model import RadarActionClassifier2D
from data_utils import SignalRecord, build_records, choose_sensor_indices, save_label_map, split_records
from data_utils import add_awgn_torch
from diffusion_model import EpsilonDenoiser1D
from diffusion_process import VPDiffusion
from paper_micro_doppler import paper_micro_doppler_spectrogram


CONFIG = {
    "diffusion_checkpoint": "runs/diffusion_dps/best.pt",
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "output_dir": "runs/classifier_dps",
    "cache_dir": "runs/dps_spectrogram_cache",
    "auto_extract": True,
    "extractor": "",
    "epochs": 200,
    "batch_size": 2,
    "lr": 0.01,
    "weight_decay": 0.0,
    "snr_levels": "-15,-10,-5,0,5,10",
    "dps_scale": 0.3,
    "start_from_measurement": False,
    "base_channels": 64,
    "num_blocks": 7,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

SPECTROGRAM_DEFAULTS = {
    "chirp_rate_hz": 1000.0,
    "fast_time_axis": 0,
    "range_bin_radius": 1,
    "combine": "magnitude_sum",
    "nperseg": 128,
    "overlap_fraction": 0.90,
    "nfft": 128,
    "window": "gaussian",
    "gaussian_std": None,
    "eps": 1e-8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier on DPS-denoised spectrograms.")
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


def parse_snr_levels(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def load_raw_iq(path: Path) -> np.ndarray:
    mat = sio.loadmat(str(path))
    if "received_time_domain_signal" in mat:
        sig = mat["received_time_domain_signal"]
    else:
        arrays = [
            v for k, v in mat.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
        ]
        if not arrays:
            raise ValueError(f"No numeric array found in {path}")
        sig = arrays[0]
    if sig.ndim != 2:
        sig = sig.reshape(sig.shape[0], -1)
    return np.asarray(sig)


def load_diffusion_model(checkpoint_path: Path, device: torch.device) -> tuple[EpsilonDenoiser1D, VPDiffusion, dict]:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model = EpsilonDenoiser1D(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt.get("ema_model_state", ckpt["model_state"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    diffusion = VPDiffusion(**ckpt["diffusion_kwargs"]).to(device)
    return model, diffusion, ckpt.get("data_kwargs", {})


def denoise_full_signal(
    raw_iq: np.ndarray,
    sensor_idx: np.ndarray,
    snr_db: float,
    model: EpsilonDenoiser1D,
    diffusion: VPDiffusion,
    dps_scale: float,
    device: torch.device,
    *,
    seed: int = 42,
    start_from_measurement: bool = False,
) -> np.ndarray:
    """Add AWGN and run DPS denoising on the full signal at once. Returns denoised full IQ."""
    signal = raw_iq[sensor_idx]
    num_sensors = signal.shape[0]

    generator = torch.Generator(device=device).manual_seed(seed)

    stacked = np.concatenate([signal.real, signal.imag], axis=0).astype(np.float32)
    tensor = torch.from_numpy(stacked)
    rms = float(torch.sqrt(torch.mean(tensor.square()) + 1e-8).item())
    x0 = (tensor / rms).unsqueeze(0).to(device)

    y_noisy, noise_std = add_awgn_torch(x0, snr_db, generator)

    initial = y_noisy if start_from_measurement else None
    with torch.enable_grad():
        x_dps, _ = diffusion.dps_sample(
            model, y_noisy,
            noise_std=noise_std, dps_scale=dps_scale,
            initial=initial, return_trace=True, show_progress=False,
        )

    dps_np = x_dps.detach().cpu().squeeze(0).numpy()
    dps_signal = (dps_np[:num_sensors] + 1j * dps_np[num_sensors:]) * rms

    out = raw_iq.copy()
    out[sensor_idx] = dps_signal
    return out


def signal_to_spectrogram(sig: np.ndarray) -> torch.Tensor:
    _, _, spec_db, _, _ = paper_micro_doppler_spectrogram(
        sig.real.astype(np.float64),
        sig.imag.astype(np.float64),
        **SPECTROGRAM_DEFAULTS,
    )
    return torch.from_numpy(spec_db.copy()).float().unsqueeze(0)  # [1, F, T]


# =============================================================================
# Phase 1: Precompute DPS spectrograms to disk
# =============================================================================

def precompute_split(
    split_name: str,
    records: list[SignalRecord],
    snr_levels: list[float],
    *,
    prior: EpsilonDenoiser1D,
    diffusion: VPDiffusion,
    num_sensors: int,
    dps_scale: float,
    start_from_measurement: bool,
    device: torch.device,
    cache_dir: Path,
    seed: int,
) -> None:
    """Precompute DPS spectrograms for one split and save to disk."""
    split_dir = cache_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    items = [(i, snr) for i in range(len(records)) for snr in snr_levels]
    print(f"  Precomputing {split_name}: {len(items)} samples ({len(records)} records x {len(snr_levels)} SNRs)")

    xs = []
    labels = []
    for idx, (rec_idx, snr_db) in enumerate(tqdm(items, desc=f"DPS {split_name}", leave=True)):
        record = records[rec_idx]
        raw_iq = load_raw_iq(record.path)
        sensor_idx = choose_sensor_indices(raw_iq.shape[0], num_sensors)

        dps_iq = denoise_full_signal(
            raw_iq, sensor_idx, snr_db,
            prior, diffusion, dps_scale, device,
            seed=seed + idx,
            start_from_measurement=start_from_measurement,
        )
        spec = signal_to_spectrogram(dps_iq)
        xs.append(spec)
        labels.append(record.label_id)

        # Save periodically (every 50 samples) to avoid losing progress on crash
        if (idx + 1) % 50 == 0:
            _save_partial(split_dir, xs, labels, idx + 1)

    # Final save. Spectrogram time lengths can differ across records, so keep
    # them as a list and let collate_fn pad only within each mini-batch.
    _save_precomputed(split_dir / "data.pt", xs, labels)
    # Remove partial saves
    for p in split_dir.glob("partial_*.pt"):
        p.unlink()
    print(f"  Saved {split_name}: {len(xs)} spectrograms -> {split_dir / 'data.pt'}")


def _save_precomputed(path: Path, xs: list[torch.Tensor], labels: list[int], count: int | None = None) -> None:
    payload = {
        "x": [x.detach().cpu().contiguous() for x in xs],
        "label": torch.tensor(labels, dtype=torch.long),
    }
    if count is not None:
        payload["count"] = count
    torch.save(payload, path)


def _save_partial(split_dir: Path, xs: list[torch.Tensor], labels: list[int], count: int) -> None:
    _save_precomputed(split_dir / f"partial_{count}.pt", xs, labels, count=count)


def is_precomputed(cache_dir: Path) -> bool:
    return (cache_dir / "train" / "data.pt").is_file() and (cache_dir / "val" / "data.pt").is_file()


def try_resume_partial(split_dir: Path) -> int:
    """Check if there's a partial save we can resume from."""
    partials = sorted(split_dir.glob("partial_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if partials:
        last = partials[-1]
        data = torch.load(last, map_location="cpu", weights_only=False)
        return int(data.get("count", 0))
    return 0


# =============================================================================
# Phase 2: Load precomputed spectrograms and train
# =============================================================================

class PrecomputedDPSDataset(Dataset):
    """Load precomputed DPS spectrograms from disk."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        data_path = cache_dir / split / "data.pt"
        if not data_path.is_file():
            raise FileNotFoundError(f"Precomputed data not found: {data_path}")
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        raw_x = data["x"]
        if torch.is_tensor(raw_x):
            self.x = raw_x.float()
        else:
            self.x = [x.float() for x in raw_x]
        self.label = data["label"].long()

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[idx], "label": self.label[idx]}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    xs = [item["x"] for item in batch]
    labels = torch.stack([item["label"] for item in batch], dim=0).long()
    max_t = max(x.shape[-1] for x in xs)
    padded = [F.pad(x, (0, max_t - x.shape[-1])) if x.shape[-1] < max_t else x for x in xs]
    return {"x": torch.stack(padded, dim=0), "label": labels}


def run_epoch(
    model: RadarActionClassifier2D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc=desc, leave=True):
        x = batch["x"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        n = x.shape[0]
        total_loss += float(loss.detach()) * n
        correct += int((logits.argmax(1) == labels).sum().item())
        total += n
    return total_loss / max(1, total), correct / max(1, total)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = data_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = data_root / cache_dir

    # Load records
    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.is_absolute():
        extracted_dir = data_root / extracted_dir
    records, label_names = build_records(
        data_root, extracted_dir,
        auto_extract=args.auto_extract,
        custom_extractor=args.extractor or None,
    )
    train_records, val_records, test_records = split_records(records, seed=args.seed)
    snr_levels = parse_snr_levels(args.snr_levels)

    # --- Phase 1: Precompute if needed ---
    if not is_precomputed(cache_dir):
        print(f"\n[Phase 1] Precomputing DPS spectrograms to {cache_dir}")
        checkpoint_path = Path(args.diffusion_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = data_root / checkpoint_path
        prior, diffusion, data_kwargs = load_diffusion_model(checkpoint_path, device)
        num_sensors = int(data_kwargs.get("num_sensors", 32))

        precompute_split(
            "train", train_records, snr_levels,
            prior=prior, diffusion=diffusion,
            num_sensors=num_sensors, dps_scale=args.dps_scale,
            start_from_measurement=args.start_from_measurement,
            device=device, cache_dir=cache_dir, seed=args.seed,
        )
        precompute_split(
            "val", val_records, snr_levels,
            prior=prior, diffusion=diffusion,
            num_sensors=num_sensors, dps_scale=args.dps_scale,
            start_from_measurement=args.start_from_measurement,
            device=device, cache_dir=cache_dir, seed=args.seed + 1000,
        )
        # Free diffusion model memory
        del prior, diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("[Phase 1] Done.\n")
    else:
        print(f"\n[Phase 1] Cache found at {cache_dir}, skipping precompute.\n")

    # --- Phase 2: Train classifier ---
    print("[Phase 2] Training classifier on precomputed DPS spectrograms")
    train_dataset = PrecomputedDPSDataset(cache_dir, "train")
    val_dataset = PrecomputedDPSDataset(cache_dir, "val")

    sample = train_dataset[0]["x"]
    in_channels = int(sample.shape[0])
    classifier = RadarActionClassifier2D(
        in_channels=in_channels,
        num_classes=len(label_names),
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    save_label_map(output_dir / "label_map.json", label_names)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"  device: {device}")
    print(f"  dps_scale: {args.dps_scale}")
    print(f"  classes: {len(label_names)} -> {label_names}")
    print(f"  SNR levels: {snr_levels}")
    print(f"  train: {len(train_dataset)}, val: {len(val_dataset)}")
    print(f"  input shape: {tuple(sample.shape)}")

    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(classifier, train_loader, optimizer, device, f"epoch {epoch}/{args.epochs}")
        val_loss, val_acc = run_epoch(classifier, val_loader, None, device, "val")
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(row)

        torch.save({
            "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
            "classifier_state": classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "label_names": label_names,
            "classifier_kwargs": {"in_channels": in_channels, "num_classes": len(label_names),
                                  "base_channels": args.base_channels, "num_blocks": args.num_blocks},
            "config": vars(args),
        }, output_dir / "latest.pt")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
                "classifier_state": classifier.state_dict(),
                "label_names": label_names,
                "classifier_kwargs": {"in_channels": in_channels, "num_classes": len(label_names),
                                      "base_channels": args.base_channels, "num_blocks": args.num_blocks},
                "config": vars(args),
            }, output_dir / "best.pt")
            print(f"  -> best val_acc={best_acc:.4f}")

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Done. best_val_acc={best_acc:.4f}, output: {output_dir}")


if __name__ == "__main__":
    main()
