# %%
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
from tqdm.auto import tqdm

from data_utils import (
    SignalRecord,
    build_records,
    choose_sensor_indices,
    split_records,
)
from paper_micro_doppler import paper_micro_doppler_spectrogram

CONFIG = {
    "iq_cache_dir": "runs/dps_cache",
    "output_dir": "runs/spectrogram_cache",
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "auto_extract": True,
    "extractor": "",
    "splits": "train,val,test",
    "sources": "clean,noisy,dps",
    "snr_levels": "-15,-10,-5,0,5,10",
    "dtype": "float32",
    "seed": 42,
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
    parser = argparse.ArgumentParser(description="Build spectrogram cache from IQ cache.")
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


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def torch_load_cpu(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_raw_iq(path: Path) -> np.ndarray:
    from scipy.io import loadmat

    mat = loadmat(str(path))
    if "received_time_domain_signal" in mat:
        sig = mat["received_time_domain_signal"]
    else:
        arrays = [
            v for k, v in mat.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
        ]
        if not arrays:
            raise ValueError(f"No numeric array in {path}")
        sig = arrays[0]
    if sig.ndim != 2:
        sig = sig.reshape(sig.shape[0], -1)
    return np.asarray(sig)


def cache_tensor_to_complex_window(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] % 2 != 0:
        raise ValueError(f"Expected [2*C, T], got {x.shape}")
    sensors = x.shape[0] // 2
    return x[:sensors] + 1j * x[sensors:]


def model_window_rms(window: np.ndarray) -> float:
    stacked = np.concatenate([window.real, window.imag], axis=0).astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(stacked**2) + 1e-8))


def raw_to_spectrogram(sig: np.ndarray) -> np.ndarray:
    _, _, spec_db, _, _ = paper_micro_doppler_spectrogram(
        sig.real.astype(np.float64),
        sig.imag.astype(np.float64),
        **SPECTROGRAM_DEFAULTS,
    )
    return spec_db


def snr_match(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return round(float(a), 6) == round(float(b), 6)


def process_cache_file(
    entry: dict,
    iq_cache_dir: Path,
    num_sensors: int,
    window_size: int,
    out_dtype: torch.dtype,
) -> dict:
    payload = torch_load_cpu(iq_cache_dir / entry["path"])
    all_x = payload["x"].float().numpy()
    labels = payload["label"].long()
    paths = payload["paths"]
    starts = payload["starts"]
    n = all_x.shape[0]

    spectrograms = []
    for i in tqdm(range(n), desc=f"  spectrogram {entry['path']}", leave=False):
        raw_path = Path(paths[i])
        start = int(starts[i])
        raw = load_raw_iq(raw_path)
        sensor_idx = choose_sensor_indices(raw.shape[0], num_sensors)

        # Compute RMS from original raw window for denormalization
        raw_window = np.zeros((len(sensor_idx), window_size), dtype=raw.dtype)
        available = max(0, min(window_size, raw.shape[1] - start))
        if available > 0:
            raw_window[:, :available] = raw[sensor_idx, start : start + available]
        rms = model_window_rms(raw_window)

        # Convert cached IQ back to complex and denormalize
        cached_complex = cache_tensor_to_complex_window(all_x[i]) * rms
        if cached_complex.shape[0] != len(sensor_idx):
            raise ValueError(
                f"Cached {cached_complex.shape[0]} sensors != raw {len(sensor_idx)}"
            )

        # Insert cached window into raw
        merged = raw.copy()
        avail = max(0, min(cached_complex.shape[1], merged.shape[1] - start))
        if avail > 0:
            merged[sensor_idx, start : start + avail] = cached_complex[:, :avail]

        spec_db = raw_to_spectrogram(merged)
        spectrograms.append(torch.from_numpy(spec_db.copy()).to(out_dtype).unsqueeze(0))

    return {
        "x": torch.stack(spectrograms, dim=0),
        "label": labels,
        "split": entry.get("split"),
        "source": entry.get("source", "dps"),
        "snr_db": entry.get("snr_db"),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()

    iq_cache_dir = Path(args.iq_cache_dir)
    if not iq_cache_dir.is_absolute():
        iq_cache_dir = data_root / iq_cache_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = data_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = iq_cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"IQ cache manifest not found: {manifest_path}")
    iq_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    label_names = iq_manifest["label_names"]
    data_kwargs = iq_manifest.get("data_kwargs", {})
    num_sensors = int(data_kwargs.get("num_sensors", 32))
    window_size = int(data_kwargs.get("window_size", 512))

    splits = parse_list(args.splits)
    sources = parse_list(args.sources)
    snr_levels = parse_float_list(args.snr_levels)
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Filter IQ cache entries
    entries = []
    for item in iq_manifest["files"]:
        if item.get("split") not in splits:
            continue
        src = item.get("source", "dps")
        if src not in sources:
            continue
        if src != "clean" and not any(snr_match(item.get("snr_db"), s) for s in snr_levels):
            continue
        entries.append(item)

    if not entries:
        raise RuntimeError("No matching IQ cache entries found.")

    print()
    print("Building spectrogram cache from IQ cache")
    print(f"  IQ cache: {iq_cache_dir}")
    print(f"  output: {output_dir}")
    print(f"  splits: {splits}")
    print(f"  sources: {sources}")
    print(f"  SNR levels: {snr_levels}")
    print(f"  entries to process: {len(entries)}")

    out_manifest_path = output_dir / "manifest.json"
    if out_manifest_path.is_file():
        out_manifest = json.loads(out_manifest_path.read_text(encoding="utf-8"))
        out_files = list(out_manifest.get("files", []))
    else:
        out_files = []

    for entry in entries:
        src = entry.get("source", "dps")
        split = entry["split"]
        snr_db = entry.get("snr_db")
        snr_text = "clean" if snr_db is None else f"snr_{snr_db:g}".replace("-", "m").replace(".", "p")
        out_name = f"{split}_{src}_{snr_text}.pt"

        print(f"\nProcessing: {entry['path']} -> {out_name}")
        result = process_cache_file(entry, iq_cache_dir, num_sensors, window_size, out_dtype)

        out_path = output_dir / out_name
        torch.save(
            {"x": result["x"], "label": result["label"]},
            out_path,
        )
        spec_shape = list(result["x"].shape)
        print(f"  Saved {out_path} ({spec_shape[0]} samples, shape={spec_shape[1:]})")

        # Update manifest
        out_entry = {
            "path": out_name,
            "split": split,
            "source": src,
            "snr_db": snr_db,
            "samples": spec_shape[0],
            "shape": spec_shape,
        }
        out_files = [
            f for f in out_files
            if not (
                f.get("split") == split
                and f.get("source") == src
                and snr_match(f.get("snr_db"), snr_db)
            )
        ]
        out_files.append(out_entry)

        # Save manifest after each file (resume-friendly)
        out_manifest = {
            "label_names": label_names,
            "data_kwargs": data_kwargs,
            "spectrogram_config": SPECTROGRAM_DEFAULTS,
            "files": out_files,
        }
        out_manifest_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Manifest: {out_manifest_path}")


# %%
if __name__ == "__main__":
    main()
