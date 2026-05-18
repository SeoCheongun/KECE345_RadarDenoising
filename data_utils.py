from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


DEFAULT_SNR_LEVELS = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0)
EXPECTED_ANGLES = {"-90", "-45", "0", "45", "90"}


@dataclass(frozen=True)
class SignalRecord:
    path: Path
    label: str
    label_id: int
    angle: str
    iteration: int


def find_extractor(custom_path: str | None = None) -> Path | None:
    candidates = [
        custom_path or "",
        shutil.which("7z"),
        shutil.which("WinRAR"),
        shutil.which("unrar"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files\NVIDIA Corporation\NVIDIA GeForce Experience\7z.exe",
        r"C:\Program Files\WinRAR\WinRAR.exe",
        r"C:\Program Files (x86)\WinRAR\WinRAR.exe",
    ]
    for item in candidates:
        if item and Path(item).is_file():
            return Path(item)
    return None


def extract_archives(data_root: Path, extracted_dir: Path, custom_extractor: str | None = None) -> None:
    archives = sorted(data_root.glob("*.rar"))
    if not archives:
        return

    extractor = find_extractor(custom_extractor)
    if extractor is None:
        if any(extracted_dir.rglob("*.mat")):
            print("No RAR extractor found; using already extracted .mat files.")
            return
        raise RuntimeError(
            "No RAR extractor found. Install 7-Zip/WinRAR or set --extractor to the executable path."
        )

    print(f"Using extractor: {extractor}")
    for archive in archives:
        label = archive.stem
        target_dir = extracted_dir / label
        if target_dir.exists() and any(target_dir.rglob("*.mat")):
            print(f"[{label}] already extracted")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{archive.name}] extracting...")
        name = extractor.name.lower()
        if "winrar" in name:
            cmd = [str(extractor), "x", "-ibck", "-y", str(archive), str(target_dir) + "\\"]
        elif "unrar" in name:
            cmd = [str(extractor), "x", "-y", str(archive), str(target_dir) + "\\"]
        else:
            cmd = [str(extractor), "x", "-y", f"-o{target_dir}", str(archive)]
        subprocess.run(cmd, check=True)


def parse_angle_and_iteration(path: Path) -> tuple[str, int]:
    # Expected pattern: 3m_0_file1.mat, 3m_-45_file12.mat, ...
    parts = path.stem.split("_")
    angle = parts[1] if len(parts) >= 2 else "unknown"
    iteration = -1
    for part in parts:
        if part.startswith("file"):
            try:
                iteration = int(part.replace("file", ""))
            except ValueError:
                iteration = -1
    return angle, iteration


def build_records(
    data_root: Path,
    extracted_dir: Path,
    *,
    auto_extract: bool = True,
    custom_extractor: str | None = None,
) -> tuple[list[SignalRecord], list[str]]:
    if auto_extract:
        extracted_dir.mkdir(parents=True, exist_ok=True)
        extract_archives(data_root, extracted_dir, custom_extractor)

    label_dirs = sorted([p for p in extracted_dir.iterdir() if p.is_dir()]) if extracted_dir.exists() else []
    if not label_dirs:
        raise RuntimeError(f"No extracted label folders found in {extracted_dir}")

    label_names = sorted(p.name for p in label_dirs if any(p.rglob("*.mat")))
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    records: list[SignalRecord] = []

    for label in label_names:
        for mat_path in sorted((extracted_dir / label).rglob("*.mat")):
            angle, iteration = parse_angle_and_iteration(mat_path)
            records.append(
                SignalRecord(
                    path=mat_path,
                    label=label,
                    label_id=label_to_id[label],
                    angle=angle,
                    iteration=iteration,
                )
            )

    if not records:
        raise RuntimeError(f"No .mat files found under {extracted_dir}")

    print_dataset_summary(records, label_names)
    return records, label_names


def print_dataset_summary(records: Sequence[SignalRecord], label_names: Sequence[str]) -> None:
    angles = sorted({r.angle for r in records})
    possible_noisy_samples = len(records) * len(DEFAULT_SNR_LEVELS)
    print("")
    print("Dataset summary")
    print(f"  activities/classes: {len(label_names)} -> {list(label_names)}")
    print(f"  angles found: {angles}")
    print(f"  clean .mat records: {len(records)}")
    print(f"  observed samples with 6 SNR levels: {possible_noisy_samples}")
    missing = sorted(EXPECTED_ANGLES - set(angles))
    if missing:
        print(f"  missing expected angles: {missing}")


def split_records(
    records: Sequence[SignalRecord],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[SignalRecord], list[SignalRecord], list[SignalRecord]]:
    labels = np.asarray([r.label_id for r in records])
    indices = np.arange(len(records))

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = labels[temp_idx]
    val_fraction_of_temp = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_fraction_of_temp,
        random_state=seed,
        stratify=temp_labels,
    )

    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


def load_complex_signal(path: Path) -> np.ndarray:
    data = loadmat(path)
    arrays = [
        value
        for key, value in data.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
    ]
    if not arrays:
        raise ValueError(f"No numeric array found in {path}")
    signal = np.asarray(arrays[0])
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    elif signal.ndim > 2:
        signal = signal.reshape(signal.shape[0], -1)
    return signal


def choose_sensor_indices(num_total: int, num_sensors: int | None) -> np.ndarray:
    if num_sensors is None or num_sensors <= 0 or num_sensors >= num_total:
        return np.arange(num_total)
    return np.linspace(0, num_total - 1, num_sensors).round().astype(np.int64)


def crop_or_pad(signal: np.ndarray, start: int, window_size: int) -> np.ndarray:
    total = signal.shape[1]
    if total >= window_size:
        start = min(max(start, 0), total - window_size)
        return signal[:, start : start + window_size]

    out = np.zeros((signal.shape[0], window_size), dtype=signal.dtype)
    out[:, :total] = signal
    return out


def complex_window_to_tensor(window: np.ndarray) -> torch.Tensor:
    stacked = np.concatenate([window.real, window.imag], axis=0)
    tensor = torch.from_numpy(stacked.astype(np.float32))
    rms = torch.sqrt(torch.mean(tensor.square()) + 1e-8)
    return tensor / rms


def add_awgn_torch(x: torch.Tensor, snr_db: float, generator: torch.Generator | None = None) -> tuple[torch.Tensor, float]:
    signal_power = torch.mean(x.square()).item()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = float(noise_power**0.5)
    if generator is None:
        noise = torch.randn_like(x) * noise_std
    else:
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator) * noise_std
    return x + noise, noise_std


class RadarWindowDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SignalRecord],
        *,
        window_size: int = 512,
        num_sensors: int | None = 32,
        windows_per_record: int = 1,
        random_crop: bool = True,
        seed: int = 42,
    ) -> None:
        self.records = list(records)
        self.window_size = window_size
        self.num_sensors = num_sensors
        self.windows_per_record = max(1, windows_per_record)
        self.random_crop = random_crop
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_record

    def __getitem__(self, index: int) -> dict[str, object]:
        record_index = index // self.windows_per_record
        window_index = index % self.windows_per_record
        record = self.records[record_index]
        signal = load_complex_signal(record.path)
        sensor_idx = choose_sensor_indices(signal.shape[0], self.num_sensors)
        signal = signal[sensor_idx]

        max_start = max(0, signal.shape[1] - self.window_size)
        if self.random_crop and max_start > 0:
            start = int(self.rng.integers(0, max_start + 1))
        elif max_start > 0:
            if self.windows_per_record == 1:
                start = max_start // 2
            else:
                start = int(round(max_start * window_index / max(1, self.windows_per_record - 1)))
        else:
            start = 0

        window = crop_or_pad(signal, start, self.window_size)
        x = complex_window_to_tensor(window)
        return {
            "x": x,
            "label": torch.tensor(record.label_id, dtype=torch.long),
            "path": str(record.path),
            "angle": record.angle,
            "iteration": record.iteration,
            "start": start,
        }


def save_label_map(path: Path, label_names: Sequence[str]) -> None:
    mapping = {name: idx for idx, name in enumerate(label_names)}
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def write_records_csv(path: Path, records: Sequence[SignalRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "label_id", "angle", "iteration"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": str(record.path),
                    "label": record.label,
                    "label_id": record.label_id,
                    "angle": record.angle,
                    "iteration": record.iteration,
                }
            )
