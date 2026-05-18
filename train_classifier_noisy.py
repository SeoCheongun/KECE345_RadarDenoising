# %%
"""Train action classifier on noisy (no DPS) spectrograms.

Pipeline per sample:
  raw .mat -> add AWGN at specified SNR -> paper micro-Doppler spectrogram -> classify
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
from data_utils import SignalRecord, build_records, save_label_map, split_records
from paper_micro_doppler import paper_micro_doppler_spectrogram


CONFIG = {
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "output_dir": "runs/classifier_noisy",
    "auto_extract": True,
    "extractor": "",
    "epochs": 200,
    "batch_size": 2,
    "lr": 0.01,
    "weight_decay": 0.0,
    "snr_levels": "-15,-10,-5,0,5,10",
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
    parser = argparse.ArgumentParser(description="Train classifier on noisy spectrograms (no DPS).")
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


def add_awgn_complex(signal: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    sig_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = (rng.randn(*signal.shape) + 1j * rng.randn(*signal.shape)) * np.sqrt(noise_power / 2.0)
    return signal + noise


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


def signal_to_spectrogram(sig: np.ndarray) -> torch.Tensor:
    _, _, spec_db, _, _ = paper_micro_doppler_spectrogram(
        sig.real.astype(np.float64),
        sig.imag.astype(np.float64),
        **SPECTROGRAM_DEFAULTS,
    )
    return torch.from_numpy(spec_db.copy()).float().unsqueeze(0)  # [1, F, T]


class NoisySpectrogramDataset(Dataset):
    """raw + AWGN -> spectrogram, one entry per (record, snr) pair."""

    def __init__(self, records: list[SignalRecord], snr_levels: list[float], seed: int) -> None:
        self.records = records
        self.items = [(i, snr) for i in range(len(records)) for snr in snr_levels]
        self.seed = seed
        self.cache: list[dict[str, torch.Tensor] | None] = [None] * len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.cache[idx] is not None:
            return self.cache[idx]
        rec_idx, snr_db = self.items[idx]
        record = self.records[rec_idx]
        sig = load_raw_iq(record.path)
        rng = np.random.RandomState(self.seed + idx)
        noisy = add_awgn_complex(sig, snr_db, rng)
        item = {
            "x": signal_to_spectrogram(noisy),
            "label": torch.tensor(record.label_id, dtype=torch.long),
        }
        self.cache[idx] = item
        return item


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

    train_dataset = NoisySpectrogramDataset(train_records, snr_levels, seed=args.seed)
    val_dataset = NoisySpectrogramDataset(val_records, snr_levels, seed=args.seed + 1000)

    sample = train_dataset[0]["x"]
    in_channels = int(sample.shape[0])
    model = RadarActionClassifier2D(
        in_channels=in_channels,
        num_classes=len(label_names),
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    save_label_map(output_dir / "label_map.json", label_names)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"\nTraining classifier on NOISY spectrograms (no DPS)")
    print(f"  device: {device}")
    print(f"  classes: {len(label_names)} -> {label_names}")
    print(f"  SNR levels: {snr_levels}")
    print(f"  train: {len(train_dataset)}, val: {len(val_dataset)}")
    print(f"  input shape: {tuple(sample.shape)}")

    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, f"epoch {epoch}/{args.epochs}")
        val_loss, val_acc = run_epoch(model, val_loader, None, device, "val")
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(row)

        torch.save({
            "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
            "classifier_state": model.state_dict(),
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
                "classifier_state": model.state_dict(),
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
