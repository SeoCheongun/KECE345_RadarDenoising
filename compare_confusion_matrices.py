# %%
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from classifier_model import RadarActionClassifier2D
from data_utils import build_records, split_records
from train_classifier_dps import (
    CONFIG as DPS_CONFIG,
    PrecomputedDPSDataset,
    load_diffusion_model,
    parse_snr_levels,
    precompute_split,
)
from train_classifier_noisy import (
    CONFIG as NOISY_CONFIG,
    NoisySpectrogramDataset,
    collate_fn,
)


CONFIG = {
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "auto_extract": True,
    "extractor": "",
    "classifier_noisy_checkpoint": "runs/classifier_noisy/best.pt",
    "classifier_dps_checkpoint": "runs/classifier_dps/best.pt",
    "diffusion_checkpoint": "runs/diffusion_dps/best.pt",
    "dps_cache_dir": "runs/dps_spectrogram_cache",
    "output_dir": "runs/confusion_matrices",
    "split": "val",
    "snr_levels": "-15,-10,-5,0,5,10",
    "batch_size": 8,
    "max_items": 0,
    "normalize": True,
    "precompute_missing_dps": False,
    "dps_scale": 0.3,
    "start_from_measurement": False,
    "seed": 42,
    "show": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def _resolve_path(data_root: Path, path: str | Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else data_root / out


def load_classifier(checkpoint_path: Path, device: torch.device) -> tuple[RadarActionClassifier2D, list[str], dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    label_names = list(checkpoint["label_names"])
    classifier = RadarActionClassifier2D(**checkpoint["classifier_kwargs"]).to(device)
    classifier.load_state_dict(checkpoint["classifier_state"])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)
    return classifier, label_names, checkpoint


def predict_dataset(
    model: RadarActionClassifier2D,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, leave=True):
        x = batch["x"].to(device)
        labels = batch["label"].cpu().numpy()
        with torch.no_grad():
            pred = model(x).argmax(dim=1).detach().cpu().numpy()
        y_true.append(labels)
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for real, pred in zip(y_true, y_pred):
        cm[int(real), int(pred)] += 1
    return cm


def plot_confusion_matrix(
    ax: plt.Axes,
    cm: np.ndarray,
    label_names: list[str],
    *,
    title: str,
    normalize: bool,
) -> None:
    values = cm.astype(np.float64)
    if normalize:
        row_sum = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0)
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("predicted", fontsize=12)
    ax.set_ylabel("real", fontsize=12)
    ax.set_xticks(np.arange(len(label_names)), labels=label_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(label_names)), labels=label_names)

    threshold = float(values.max()) * 0.55 if values.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{values[i, j]:.2f}\n({cm[i, j]})" if normalize else str(cm[i, j])
            color = "white" if values[i, j] > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)


def split_seed(base_seed: int, split: str) -> int:
    return base_seed + {"train": 0, "val": 1000, "test": 2000}[split]


def maybe_precompute_dps_split(
    *,
    split: str,
    records,
    snr_levels: list[float],
    data_root: Path,
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    data_path = cache_dir / split / "data.pt"
    if data_path.is_file():
        return
    if not args.precompute_missing_dps:
        raise FileNotFoundError(
            f"DPS spectrogram cache is missing: {data_path}\n"
            f"For a quick comparison, use split='val'. To build this split, set precompute_missing_dps=True."
        )
    checkpoint_path = _resolve_path(data_root, args.diffusion_checkpoint)
    prior, diffusion, data_kwargs = load_diffusion_model(checkpoint_path, device)
    num_sensors = int(data_kwargs.get("num_sensors", 32))
    precompute_split(
        split,
        records,
        snr_levels,
        prior=prior,
        diffusion=diffusion,
        num_sensors=num_sensors,
        dps_scale=args.dps_scale,
        start_from_measurement=args.start_from_measurement,
        device=device,
        cache_dir=cache_dir,
        seed=split_seed(args.seed, split),
    )


def compare_noisy_dps_confusion(**overrides) -> dict:
    args = argparse.Namespace(**CONFIG)
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown option: {key}")
        setattr(args, key, value)

    if args.split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")

    device = torch.device(args.device)
    data_root = Path(args.data_root).resolve()
    output_dir = _resolve_path(data_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = _resolve_path(data_root, args.extracted_dir)
    records, build_label_names = build_records(
        data_root,
        extracted_dir,
        auto_extract=args.auto_extract,
        custom_extractor=args.extractor or None,
    )
    train_records, val_records, test_records = split_records(records, seed=args.seed)
    split_records_map = {"train": train_records, "val": val_records, "test": test_records}
    eval_records = split_records_map[args.split]
    snr_levels = parse_snr_levels(args.snr_levels)

    noisy_model, noisy_label_names, noisy_ckpt = load_classifier(
        _resolve_path(data_root, args.classifier_noisy_checkpoint),
        device,
    )
    dps_model, dps_label_names, dps_ckpt = load_classifier(
        _resolve_path(data_root, args.classifier_dps_checkpoint),
        device,
    )
    if noisy_label_names != dps_label_names:
        raise ValueError(f"Classifier label maps differ: {noisy_label_names} vs {dps_label_names}")
    if noisy_label_names != list(build_label_names):
        print("Warning: checkpoint label order differs from current dataset label order.")

    noisy_dataset = NoisySpectrogramDataset(eval_records, snr_levels, seed=split_seed(args.seed, args.split))
    dps_cache_dir = _resolve_path(data_root, args.dps_cache_dir)
    maybe_precompute_dps_split(
        split=args.split,
        records=eval_records,
        snr_levels=snr_levels,
        data_root=data_root,
        cache_dir=dps_cache_dir,
        args=args,
        device=device,
    )
    dps_dataset = PrecomputedDPSDataset(dps_cache_dir, args.split)
    if args.max_items and args.max_items > 0:
        limit = min(int(args.max_items), len(noisy_dataset), len(dps_dataset))
        indices = list(range(limit))
        noisy_dataset = Subset(noisy_dataset, indices)
        dps_dataset = Subset(dps_dataset, indices)

    y_true_noisy, y_pred_noisy = predict_dataset(
        noisy_model,
        noisy_dataset,
        device=device,
        batch_size=args.batch_size,
        desc=f"classifier_noisy {args.split}",
    )
    y_true_dps, y_pred_dps = predict_dataset(
        dps_model,
        dps_dataset,
        device=device,
        batch_size=args.batch_size,
        desc=f"classifier_dps {args.split}",
    )
    if not np.array_equal(y_true_noisy, y_true_dps):
        print("Warning: noisy and DPS datasets have different label ordering; plotting each with its own labels.")

    label_names = noisy_label_names
    num_classes = len(label_names)
    cm_noisy = confusion_matrix_np(y_true_noisy, y_pred_noisy, num_classes)
    cm_dps = confusion_matrix_np(y_true_dps, y_pred_dps, num_classes)
    acc_noisy = float(np.mean(y_true_noisy == y_pred_noisy))
    acc_dps = float(np.mean(y_true_dps == y_pred_dps))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    snr_text = ",".join(f"{x:g}" for x in snr_levels)
    plot_confusion_matrix(
        axes[0],
        cm_noisy,
        label_names,
        title=f"classifier_noisy, split={args.split}, acc={acc_noisy:.3f}",
        normalize=bool(args.normalize),
    )
    plot_confusion_matrix(
        axes[1],
        cm_dps,
        label_names,
        title=f"classifier_dps, split={args.split}, acc={acc_dps:.3f}",
        normalize=bool(args.normalize),
    )
    fig.suptitle(
        f"Confusion Matrix Compare, SNR=({snr_text}) dB, real=(rows), pred=(cols)",
        fontsize=18,
        fontweight="bold",
    )

    out_path = output_dir / f"confusion_compare_{args.split}.png"
    fig.savefig(out_path, dpi=180)
    print(f"classifier_noisy acc: {acc_noisy:.4f}")
    print(f"classifier_dps   acc: {acc_dps:.4f}")
    print(f"Saved figure: {out_path}")
    if args.show:
        plt.show()

    return {
        "args": args,
        "label_names": label_names,
        "cm_noisy": cm_noisy,
        "cm_dps": cm_dps,
        "acc_noisy": acc_noisy,
        "acc_dps": acc_dps,
        "y_true_noisy": y_true_noisy,
        "y_pred_noisy": y_pred_noisy,
        "y_true_dps": y_true_dps,
        "y_pred_dps": y_pred_dps,
        "figure": fig,
        "figure_path": out_path,
        "noisy_checkpoint": noisy_ckpt,
        "dps_checkpoint": dps_ckpt,
    }


if __name__ == "__main__":
    compare_noisy_dps_confusion()
