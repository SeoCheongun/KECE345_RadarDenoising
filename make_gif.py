# %%
"""Make a timestep GIF of DPS reconstruction as paper micro-Doppler spectrograms."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter
from tqdm.auto import tqdm

from data_utils import add_awgn_torch, build_records, choose_sensor_indices, split_records
from paper_micro_doppler import paper_micro_doppler_spectrogram, plot_micro_doppler_spectrogram
from train_classifier_dps import load_diffusion_model, load_raw_iq


CONFIG = {
    "checkpoint": "runs/diffusion_dps/best.pt",
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "output_dir": "runs/diffusion_dps/gifs",
    "auto_extract": True,
    "extractor": "",
    "split": "test",
    "sample_index": -1,
    "snr_db": -40.0,
    "dps_scale": 1.0,
    "start_from_measurement": False,
    "frame_mode": "xt",
    "first_frame_mode": "same",
    "frame_base": "zero",
    "final_frame_base": "raw",
    "frame_stride": 5,
    "fps": 6,
    "final_hold_frames": 30,
    "save_final_png": True,
    "seed": 270709,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

SPECTROGRAM_CONFIG = {
    "chirp_rate_hz": 1000.0,
    "fast_time_axis": 0,
    "target_range_bin": None,
    "range_bin_radius": 1,
    "combine": "magnitude_sum",
    "nperseg": 128,
    "overlap_fraction": 0.90,
    "nfft": 128,
    "window": "gaussian",
    "gaussian_std": None,
    "eps": 1e-8,
    "dynamic_range_db": 60.0,
    "cmap": "jet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a DPS timestep spectrogram GIF.")
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


def resolve_path(data_root: Path, path: str | Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else data_root / out


def tensor_to_selected_iq(x: torch.Tensor, num_sensors: int, rms: float) -> np.ndarray:
    arr = x.detach().cpu().squeeze(0).numpy()
    return (arr[:num_sensors] + 1j * arr[num_sensors:]) * rms


def tensor_to_full_iq(
    x: torch.Tensor,
    raw_iq: np.ndarray,
    sensor_idx: np.ndarray,
    rms: float,
    *,
    base: str,
) -> np.ndarray:
    selected_iq = tensor_to_selected_iq(x, len(sensor_idx), rms)
    if base == "zero":
        out = np.zeros_like(raw_iq)
    elif base == "raw":
        out = raw_iq.copy()
    else:
        raise ValueError("frame_base must be 'zero' or 'raw'.")
    out[sensor_idx] = selected_iq
    return out


def selected_rows_to_full_iq(raw_iq: np.ndarray, sensor_idx: np.ndarray, *, base: str) -> np.ndarray:
    if base == "zero":
        out = np.zeros_like(raw_iq)
    elif base == "raw":
        out = raw_iq.copy()
    else:
        raise ValueError("frame_base must be 'zero' or 'raw'.")
    out[sensor_idx] = raw_iq[sensor_idx]
    return out


def make_spectrogram(iq: np.ndarray, *, target_range_bin: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    freqs, times, spec_db, range_bin, _ = paper_micro_doppler_spectrogram(
        iq.real.astype(np.float64),
        iq.imag.astype(np.float64),
        chirp_rate_hz=float(SPECTROGRAM_CONFIG["chirp_rate_hz"]),
        fast_time_axis=int(SPECTROGRAM_CONFIG["fast_time_axis"]),
        target_range_bin=target_range_bin,
        range_bin_radius=int(SPECTROGRAM_CONFIG["range_bin_radius"]),
        combine=str(SPECTROGRAM_CONFIG["combine"]),
        nperseg=int(SPECTROGRAM_CONFIG["nperseg"]),
        overlap_fraction=float(SPECTROGRAM_CONFIG["overlap_fraction"]),
        nfft=int(SPECTROGRAM_CONFIG["nfft"]),
        window=str(SPECTROGRAM_CONFIG["window"]),
        gaussian_std=SPECTROGRAM_CONFIG["gaussian_std"],
        eps=float(SPECTROGRAM_CONFIG["eps"]),
    )
    return freqs, times, spec_db, int(range_bin)


def run_dps_with_spectrogram_frames(
    raw_iq: np.ndarray,
    sensor_idx: np.ndarray,
    *,
    prior,
    diffusion,
    snr_db: float,
    dps_scale: float,
    start_from_measurement: bool,
    frame_stride: int,
    frame_mode: str,
    first_frame_mode: str,
    frame_base: str,
    final_frame_base: str,
    seed: int,
    device: torch.device,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    signal = raw_iq[sensor_idx]
    num_sensors = signal.shape[0]
    stacked = np.concatenate([signal.real, signal.imag], axis=0).astype(np.float32)
    tensor = torch.from_numpy(stacked)
    rms = float(torch.sqrt(torch.mean(tensor.square()) + 1e-8).item())
    x0 = (tensor / rms).unsqueeze(0).to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    y_noisy, noise_std = add_awgn_torch(x0, snr_db, generator)
    clean_full_iq = selected_rows_to_full_iq(raw_iq, sensor_idx, base=frame_base)
    noisy_full_iq = tensor_to_full_iq(y_noisy, raw_iq, sensor_idx, rms, base=frame_base)

    sigma2 = max(float(noise_std) ** 2, 1e-8)
    xt = y_noisy.clone() if start_from_measurement else torch.randn_like(y_noisy)
    frame_stride = max(1, int(frame_stride))
    if frame_mode not in {"xt", "x0_hat"}:
        raise ValueError("frame_mode must be 'xt' or 'x0_hat'.")
    if first_frame_mode not in {"xt", "x0_hat", "same"}:
        raise ValueError("first_frame_mode must be 'xt', 'x0_hat', or 'same'.")
    frames: list[dict] = []

    model_was_training = prior.training
    prior.eval()
    last_x0_hat = None
    last_loss = 0.0
    steps = range(diffusion.timesteps - 1, -1, -1)
    for step in tqdm(steps, desc="DPS GIF reverse", leave=True):
        timesteps = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
        xt_req = xt.detach().requires_grad_(True)
        eps = prior(xt_req, timesteps)
        x0_hat = diffusion.predict_x0_from_eps(xt_req, timesteps, eps)
        last_x0_hat = x0_hat.detach()

        residual = x0_hat - y_noisy
        data_loss = residual.flatten(1).square().sum(dim=1).mean() / (2.0 * sigma2)
        last_loss = float(data_loss.detach().cpu())
        grad_loss = torch.autograd.grad(data_loss, xt_req)[0]

        should_capture = (step % frame_stride == 0) or step == diffusion.timesteps - 1 or step == 0
        if should_capture:
            current_frame_mode = frame_mode
            if step == diffusion.timesteps - 1 and first_frame_mode != "same":
                current_frame_mode = first_frame_mode
            frame_tensor = xt_req.detach() if current_frame_mode == "xt" else x0_hat.detach()
            frame_iq = tensor_to_full_iq(frame_tensor, raw_iq, sensor_idx, rms, base=frame_base)
            frames.append(
                {
                    "step": int(step),
                    "reverse_index": int(diffusion.timesteps - 1 - step),
                    "loss": float(data_loss.detach().cpu()),
                    "mode": current_frame_mode,
                    "iq": frame_iq,
                }
            )

        with torch.no_grad():
            mean = diffusion.q_posterior_mean(x0_hat.detach(), xt_req.detach(), timesteps)
            var = diffusion.posterior_variance.gather(0, timesteps).reshape(
                timesteps.shape[0],
                *((1,) * (len(xt.shape) - 1)),
            )
            guided_mean = mean - float(dps_scale) * var * grad_loss
            if step > 0:
                xt = guided_mean + torch.sqrt(var) * torch.randn_like(xt)
            else:
                xt = guided_mean

    if model_was_training:
        prior.train()
    if last_x0_hat is None:
        raise RuntimeError("DPS loop produced no x0_hat frames.")
    # Match check_one_sample_random_visualize.py for the final DPS display:
    # selected sensors are reconstructed, unselected sensors remain raw.
    final_tensor = xt.detach()
    dps_full_iq = tensor_to_full_iq(final_tensor, raw_iq, sensor_idx, rms, base=final_frame_base)
    frames.append(
        {
            "step": 0,
            "reverse_index": int(diffusion.timesteps),
            "loss": last_loss,
            "mode": "dps_final",
            "iq": dps_full_iq,
        }
    )
    return frames, clean_full_iq, noisy_full_iq, dps_full_iq


def save_spectrogram_gif(
    frames: list[dict],
    *,
    clean_iq: np.ndarray,
    noisy_iq: np.ndarray,
    output_path: Path,
    title_prefix: str,
    fps: int,
    final_hold_frames: int,
) -> None:
    clean_freqs, clean_times, clean_db, selected_range_bin = make_spectrogram(clean_iq)
    noisy_freqs, noisy_times, noisy_db, _ = make_spectrogram(noisy_iq, target_range_bin=selected_range_bin)

    specs = []
    for frame in tqdm(frames, desc="spectrogram frames", leave=True):
        freqs, times, spec_db, _ = make_spectrogram(frame["iq"], target_range_bin=selected_range_bin)
        specs.append((freqs, times, spec_db))

    all_values = np.concatenate([spec[2].ravel() for spec in specs])
    vmax = float(np.percentile(all_values, 98.0))
    vmin = max(float(np.percentile(all_values, 2.0)), vmax - float(SPECTROGRAM_CONFIG["dynamic_range_db"]))
    spec_max_hz = float(SPECTROGRAM_CONFIG["chirp_rate_hz"]) / 2.0

    fig, ax = plt.subplots(figsize=(10, 6))
    writer = PillowWriter(fps=max(1, int(fps)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output_path), dpi=120):
        frame_specs = list(zip(frames, specs))
        if frame_specs and final_hold_frames > 0:
            frame_specs.extend([frame_specs[-1]] * int(final_hold_frames))
        for frame, (freqs, times, spec_db) in tqdm(frame_specs, desc="write gif", leave=True):
            ax.clear()
            plot_micro_doppler_spectrogram(
                freqs,
                times,
                spec_db,
                ax=ax,
                title=(
                    f"{title_prefix}\n"
                    f"DPS {frame['mode']} step={frame['step']}, reverse={frame['reverse_index']}, "
                    f"loss={frame['loss']:.3g}, range_bin={selected_range_bin}"
                ),
                cmap=str(SPECTROGRAM_CONFIG["cmap"]),
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            ax.set_ylim(-spec_max_hz, spec_max_hz)
            ax.set_title(ax.get_title(), fontsize=13, fontweight="bold")
            writer.grab_frame()
    plt.close(fig)


def save_final_comparison_png(
    *,
    clean_iq: np.ndarray,
    noisy_iq: np.ndarray,
    dps_iq: np.ndarray,
    output_path: Path,
    title_prefix: str,
) -> None:
    clean_freqs, clean_times, clean_db, selected_range_bin = make_spectrogram(clean_iq)
    noisy_freqs, noisy_times, noisy_db, _ = make_spectrogram(noisy_iq, target_range_bin=selected_range_bin)
    dps_freqs, dps_times, dps_db, _ = make_spectrogram(dps_iq, target_range_bin=selected_range_bin)

    all_values = np.concatenate([clean_db.ravel(), noisy_db.ravel(), dps_db.ravel()])
    vmax = float(np.percentile(all_values, 98.0))
    vmin = max(float(np.percentile(all_values, 2.0)), vmax - float(SPECTROGRAM_CONFIG["dynamic_range_db"]))
    spec_max_hz = float(SPECTROGRAM_CONFIG["chirp_rate_hz"]) / 2.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    for ax, spec, title in (
        (axes[0], (clean_freqs, clean_times, clean_db), "clean full raw IQ"),
        (axes[1], (noisy_freqs, noisy_times, noisy_db), "noisy full raw IQ"),
        (axes[2], (dps_freqs, dps_times, dps_db), "final DPS full raw IQ"),
    ):
        freqs, times, spec_db = spec
        plot_micro_doppler_spectrogram(
            freqs,
            times,
            spec_db,
            ax=ax,
            title=f"{title_prefix}\n{title}, range_bin={selected_range_bin}",
            cmap=str(SPECTROGRAM_CONFIG["cmap"]),
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        ax.set_ylim(-spec_max_hz, spec_max_hz)
        ax.set_title(ax.get_title(), fontsize=13, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main(args: argparse.Namespace | None = None) -> dict:
    if args is None:
        args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root).resolve()
    output_dir = resolve_path(data_root, args.output_dir)
    checkpoint_path = resolve_path(data_root, args.checkpoint)
    extracted_dir = resolve_path(data_root, args.extracted_dir)
    device = torch.device(args.device)

    prior, diffusion, data_kwargs = load_diffusion_model(checkpoint_path, device)
    num_sensors = int(data_kwargs.get("num_sensors", 32))

    records, label_names = build_records(
        data_root,
        extracted_dir,
        auto_extract=args.auto_extract,
        custom_extractor=args.extractor or None,
    )
    train_records, val_records, test_records = split_records(records, seed=args.seed)
    split_map = {"train": train_records, "val": val_records, "test": test_records}
    if args.split not in split_map:
        raise ValueError("--split must be train, val, or test")
    split_records_list = split_map[args.split]

    index = int(args.sample_index) if int(args.sample_index) >= 0 else random.randrange(len(split_records_list))
    record = split_records_list[index]
    raw_iq = load_raw_iq(record.path)
    sensor_idx = choose_sensor_indices(raw_iq.shape[0], num_sensors)
    label_name = label_names[int(record.label_id)]

    frames, clean_iq, noisy_iq, dps_iq = run_dps_with_spectrogram_frames(
        raw_iq,
        sensor_idx,
        prior=prior,
        diffusion=diffusion,
        snr_db=float(args.snr_db),
        dps_scale=float(args.dps_scale),
        start_from_measurement=bool(args.start_from_measurement),
        frame_stride=int(args.frame_stride),
        frame_mode=str(args.frame_mode),
        first_frame_mode=str(args.first_frame_mode),
        frame_base=str(args.frame_base),
        final_frame_base=str(args.final_frame_base),
        seed=int(args.seed) + index,
        device=device,
    )

    out_path = output_dir / f"dps_steps_{args.split}_{index}_{args.snr_db:g}dB.gif"
    title = (
        f"real=({label_name}), split={args.split}, sample={index}, "
        f"SNR={args.snr_db:g} dB, dps_scale={args.dps_scale:g}, "
        f"base={args.frame_base}, final_base={args.final_frame_base}"
    )
    save_spectrogram_gif(
        frames,
        clean_iq=clean_iq,
        noisy_iq=noisy_iq,
        output_path=out_path,
        title_prefix=title,
        fps=int(args.fps),
        final_hold_frames=int(args.final_hold_frames),
    )
    final_png_path = output_dir / f"dps_steps_{args.split}_{index}_{args.snr_db:g}dB_final_compare.png"
    if bool(getattr(args, "save_final_png", True)):
        save_final_comparison_png(
            clean_iq=clean_iq,
            noisy_iq=noisy_iq,
            dps_iq=dps_iq,
            output_path=final_png_path,
            title_prefix=title,
        )

    print(f"Saved GIF: {out_path}")
    if bool(getattr(args, "save_final_png", True)):
        print(f"Saved final comparison PNG: {final_png_path}")
    print(f"sample path: {record.path}")
    print(f"label: {label_name}")
    print(f"raw shape: {raw_iq.shape}")
    print(f"frames: {len(frames)}")
    return {
        "gif_path": out_path,
        "final_png_path": final_png_path,
        "index": index,
        "record": record,
        "label_name": label_name,
        "frames": frames,
        "clean_iq": clean_iq,
        "noisy_iq": noisy_iq,
        "dps_iq": dps_iq,
    }


args = argparse.Namespace(**CONFIG)
args.split = "test"
args.sample_index =  5
args.snr_db = -40
args.dps_scale = 1.0
args.start_from_measurement = False
args.frame_mode = "xt"
args.first_frame_mode = "same"
args.frame_base = "zero"
args.final_frame_base = "raw"
args.frame_stride = 1
args.fps = 30
args.final_hold_frames = 60
args.save_final_png = True

GIF_RESULT = main(args)
GIF_RESULT["gif_path"]
