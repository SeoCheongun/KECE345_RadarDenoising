# %%
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, welch
from tqdm.auto import tqdm

from classifier_model import RadarActionClassifier2D
from data_utils import (
    RadarWindowDataset,
    add_awgn_torch,
    build_records,
    choose_sensor_indices,
    load_complex_signal,
    split_records,
)
from diffusion_model import EpsilonDenoiser1D
from diffusion_process import VPDiffusion
from paper_micro_doppler import (
    paper_micro_doppler_spectrogram,
    plot_micro_doppler_spectrogram as plot_paper_micro_doppler_spectrogram,
)


CONFIG = {
    "checkpoint": "runs/diffusion_dps/best.pt",
    "classifier_checkpoint": "runs/classifier_dps/best.pt",
    "baseline_classifier_checkpoint": "runs/classifier_noisy/best.pt",
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "output_dir": "runs/diffusion_dps/figures",
    "auto_extract": True,
    "extractor": "",
    "split": "test",
    "snr_db": -15,
    "inverse_task": "denoise",  # denoise or inpaint
    "observed_fraction": 0.5,
    "dps_scale": 1.0,
    "start_from_measurement": False,
    "sample_index": -1,
    "sensor_index": 0,
    "seed": 270709,
    "show": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

BPF_CONFIG = {
    # Set this to the real slow-time sampling rate if you know it.
    # The default is only a visualization setting for the current 512-point window.
    "fs_hz": 20.0,
    # Typical biological radar band covering respiration + heartbeat.
    # Respiration often sits around 0.1-0.5 Hz; heartbeat around 0.8-2.0 Hz.
    "low_hz": 0.1,
    "high_hz": 2.0,
    "order": 4,
    "use_last_result": True,
    "save": True,
    "show": True,
}

SPECTROGRAM_CONFIG = {
    # This is the slow-time sampling rate, i.e. chirp repetition rate.
    "chirp_rate_hz": 1000.0,
    # Current dataset tensors are stored as [fast-time/range-like rows, slow-time columns].
    # For raw ADC shaped [num_chirps, num_fast_time_samples], use fast_time_axis=1.
    "fast_time_axis": 0,
    "target_range_bin": None,
    # For this clean/noisy/DPS comparison plot, use the clean target bin by
    # default so all three panels look at the same physical bin.
    "range_bin_reference": "clean",
    "range_bin_radius": 1,
    "combine": "magnitude_sum",
    "max_hz": 500.0,
    "nperseg": 128,
    "overlap_fraction": 0.90,
    "nfft": 128,
    "window": "gaussian",
    "gaussian_std": None,
    "eps": 1e-8,
    "dynamic_range_db": 60.0,
    "cmap": "jet",
}

SNR_SWEEP_CONFIG = {
    "snr_db_list": [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0],
    "run_bpf_after_each": False,
    "show_each": True,
}

DPS_SCALE_SWEEP_CONFIG = {
    "dps_scale_list": [1.0, 0.5, 0.3, 0.1, 0.05],
    "show_each": True,
}

LAST_RESULT = None
OPEN_FIGURES = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one random DPS reconstruction sample.")
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


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[dict, EpsilonDenoiser1D, VPDiffusion]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EpsilonDenoiser1D(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint.get("ema_model_state", checkpoint["model_state"]))
    model.eval()
    diffusion = VPDiffusion(**checkpoint["diffusion_kwargs"]).to(device)
    return checkpoint, model, diffusion


def load_classifier(checkpoint_path: Path, device: torch.device) -> tuple[dict, RadarActionClassifier2D, list[str]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    label_names = list(checkpoint["label_names"])
    kwargs = dict(checkpoint.get("classifier_kwargs", {}))
    kwargs.setdefault("in_channels", 1)
    kwargs.setdefault("num_classes", len(label_names))
    classifier = RadarActionClassifier2D(**kwargs).to(device)
    classifier.load_state_dict(checkpoint["classifier_state"])
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)
    return checkpoint, classifier, label_names


def classify_dps_spectrogram(
    dps_full_iq: np.ndarray,
    classifier: RadarActionClassifier2D,
    label_names: list[str],
    device: torch.device,
    md_kwargs: dict,
) -> dict:
    _, _, spec_db, range_bin, _ = paper_micro_doppler_spectrogram(
        dps_full_iq.real.astype(np.float64),
        dps_full_iq.imag.astype(np.float64),
        **md_kwargs,
    )
    x = torch.from_numpy(spec_db.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    return {
        "pred_id": pred_id,
        "pred_name": label_names[pred_id],
        "confidence": confidence,
        "range_bin": int(range_bin),
        "probs": probs,
        "spec_shape": tuple(spec_db.shape),
    }


def channel_pair(sensor_index: int, total_channels: int) -> tuple[int, int]:
    sensors = total_channels // 2
    idx = min(max(sensor_index, 0), sensors - 1)
    return idx, idx + sensors


def to_numpy_signal(x: torch.Tensor, sensor_index: int) -> tuple:
    arr = x.detach().cpu().squeeze(0)
    real_idx, imag_idx = channel_pair(sensor_index, arr.shape[0])
    real = arr[real_idx].numpy()
    imag = arr[imag_idx].numpy()
    mag = (real**2 + imag**2) ** 0.5
    return real, imag, mag


def to_numpy_iq_matrix(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert a model tensor [1, 2*C, T] or [2*C, T] to real/imag matrices [C, T]."""
    arr = x.detach().cpu().squeeze(0).numpy()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D channel/time tensor after squeeze, got shape {arr.shape}.")
    if arr.shape[0] % 2 != 0:
        raise ValueError(f"Expected real/imag channels concatenated on axis 0, got {arr.shape[0]} channels.")
    half = arr.shape[0] // 2
    return arr[:half], arr[half:]


def add_awgn_numpy_complex(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.abs(iq) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = (noise_power / 2.0) ** 0.5
    noise = rng.normal(0.0, noise_std, size=iq.shape) + 1j * rng.normal(0.0, noise_std, size=iq.shape)
    return iq + noise


def model_window_rms(window: np.ndarray) -> float:
    stacked = np.concatenate([window.real, window.imag], axis=0).astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(stacked**2) + 1e-8))


def padded_raw_window(raw_iq: np.ndarray, sensor_idx: np.ndarray, start: int, window_size: int) -> np.ndarray:
    window = np.zeros((len(sensor_idx), window_size), dtype=raw_iq.dtype)
    available = max(0, min(window_size, raw_iq.shape[1] - start))
    if available > 0:
        window[:, :available] = raw_iq[sensor_idx, start : start + available]
    return window


def raw_with_model_window(
    raw_iq: np.ndarray,
    sensor_idx: np.ndarray,
    start: int,
    model_x: torch.Tensor,
    rms: float,
) -> np.ndarray:
    real, imag = to_numpy_iq_matrix(model_x)
    model_window = (real + 1j * imag) * rms
    out = raw_iq.copy()
    available = max(0, min(model_window.shape[1], out.shape[1] - start))
    if available > 0:
        out[sensor_idx, start : start + available] = model_window[:, :available]
    return out


def denoise_full_signal(
    raw_iq: np.ndarray,
    sensor_idx: np.ndarray,
    window_size: int,
    snr_db: float,
    model: EpsilonDenoiser1D,
    diffusion: VPDiffusion,
    dps_scale: float,
    device: torch.device,
    *,
    seed: int = 42,
    start_from_measurement: bool = False,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Add noise and run DPS denoising on the full signal at once.

    Steps:
      1. Extract selected sensors and normalize (RMS)
      2. Add AWGN at the given SNR in the normalized domain
      3. Run DPS denoising on the entire signal
      4. De-normalize

    Returns (noisy_full_iq, dps_full_iq, trace).
    """
    signal = raw_iq[sensor_idx]  # [num_sensors, total_time] complex
    num_sensors, total_time = signal.shape

    generator = torch.Generator(device=device).manual_seed(seed)

    # Normalize full signal (same logic as complex_window_to_tensor)
    stacked = np.concatenate([signal.real, signal.imag], axis=0).astype(np.float32)
    tensor = torch.from_numpy(stacked)
    rms = float(torch.sqrt(torch.mean(tensor.square()) + 1e-8).item())
    x0 = (tensor / rms).unsqueeze(0).to(device)

    # Add noise in normalized domain
    y_noisy, noise_std = add_awgn_torch(x0, snr_db, generator)

    # DPS denoising on full signal
    initial = y_noisy if start_from_measurement else None
    with torch.enable_grad():
        x_dps, trace = diffusion.dps_sample(
            model,
            y_noisy,
            noise_std=noise_std,
            dps_scale=dps_scale,
            initial=initial,
            return_trace=True,
            show_progress=show_progress,
        )

    # Convert noisy back to raw complex domain
    noisy_np = y_noisy.detach().cpu().squeeze(0).numpy()
    noisy_signal = (noisy_np[:num_sensors] + 1j * noisy_np[num_sensors:]) * rms

    # Convert DPS back to raw complex domain
    dps_np = x_dps.detach().cpu().squeeze(0).numpy()
    dps_signal = (dps_np[:num_sensors] + 1j * dps_np[num_sensors:]) * rms

    # Build full IQ arrays (replace selected sensors, keep rest unchanged)
    noisy_full_iq = raw_iq.copy()
    noisy_full_iq[sensor_idx] = noisy_signal

    dps_full_iq = raw_iq.copy()
    dps_full_iq[sensor_idx] = dps_signal

    return noisy_full_iq, dps_full_iq, trace


def shared_spectrogram_limits(*specs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([spec[2].ravel() for spec in specs])
    vmax = float(np.percentile(values, 98.0))
    vmin = max(float(np.percentile(values, 2.0)), vmax - float(SPECTROGRAM_CONFIG["dynamic_range_db"]))
    return vmin, vmax


def plot_spectrogram(
    ax: plt.Axes,
    spec: tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    *,
    vmin: float,
    vmax: float,
    ylim: tuple[float, float] | None = None,
) -> object:
    freqs, times, spec_db = spec
    _, _, image = plot_paper_micro_doppler_spectrogram(
        freqs,
        times,
        spec_db,
        ax=ax,
        title=title,
        cmap=str(SPECTROGRAM_CONFIG.get("cmap", "jet")),
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(*ylim)
    return image


def add_right_colorbar(fig: plt.Figure, mesh: object, label: str, rect: list[float]) -> None:
    cbar_ax = fig.add_axes(rect)
    fig.colorbar(mesh, cax=cbar_ax, label=label)


def show_figure(fig: plt.Figure) -> None:
    """Keep figures visible across repeated calls."""
    OPEN_FIGURES.append(fig)
    plt.figure(fig.number)
    plt.show(block=False)
    plt.pause(0.001)


def run_one_sample(args: argparse.Namespace, *, show_plot: bool = True, show_progress: bool = True) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = data_root / checkpoint_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = data_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint, model, diffusion = load_model(checkpoint_path, device)
    data_config = checkpoint["data_kwargs"]
    classifier_path = Path(args.classifier_checkpoint)
    if not classifier_path.is_absolute():
        classifier_path = data_root / classifier_path
    classifier_ckpt, classifier, classifier_label_names = load_classifier(classifier_path, device)
    baseline_classifier_path = Path(args.baseline_classifier_checkpoint)
    if not baseline_classifier_path.is_absolute():
        baseline_classifier_path = data_root / baseline_classifier_path
    baseline_classifier_ckpt, baseline_classifier, baseline_label_names = load_classifier(
        baseline_classifier_path,
        device,
    )

    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.is_absolute():
        extracted_dir = data_root / extracted_dir
    records, label_names = build_records(
        data_root,
        extracted_dir,
        auto_extract=args.auto_extract,
        custom_extractor=args.extractor or None,
    )
    train_records, val_records, test_records = split_records(records, seed=args.seed)
    split_map = {"train": train_records, "val": val_records, "test": test_records}
    dataset = RadarWindowDataset(
        split_map[args.split],
        window_size=data_config["window_size"],
        num_sensors=data_config["num_sensors"],
        windows_per_record=1,
        random_crop=False,
        seed=args.seed,
    )

    index = args.sample_index if args.sample_index >= 0 else random.randrange(len(dataset))
    item = dataset[index]
    x0 = item["x"].unsqueeze(0).to(device)
    label_id = int(item["label"])
    label_name = label_names[label_id]

    raw_iq = load_complex_signal(Path(item["path"]))
    sensor_idx = choose_sensor_indices(raw_iq.shape[0], data_config["num_sensors"])
    start = int(item["start"])
    window_size = int(data_config["window_size"])
    clean_full_iq = raw_iq

    if args.inverse_task == "denoise":
        observed_full_iq, dps_full_iq, trace = denoise_full_signal(
            raw_iq, sensor_idx, window_size, args.snr_db,
            model, diffusion, args.dps_scale, device,
            seed=args.seed + index,
            start_from_measurement=args.start_from_measurement,
            show_progress=show_progress,
        )
    elif args.inverse_task == "inpaint":
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        y_noisy, noise_std = add_awgn_torch(x0, args.snr_db, generator)
        mask = (torch.rand(x0.shape, device=device, generator=generator) < args.observed_fraction).float()
        measurement = mask * y_noisy
        operator = lambda z, mask=mask: mask * z
        initial = measurement if args.start_from_measurement else None
        with torch.enable_grad():
            x_dps_w, trace_t = diffusion.dps_sample(
                model, measurement, operator=operator,
                noise_std=noise_std, dps_scale=args.dps_scale,
                initial=initial, return_trace=True, show_progress=show_progress,
            )
        trace = list(trace_t)
        raw_window = padded_raw_window(raw_iq, sensor_idx, start, window_size)
        rms = model_window_rms(raw_window)
        observed_full_iq = raw_with_model_window(raw_iq, sensor_idx, start, measurement, rms)
        dps_full_iq = raw_with_model_window(raw_iq, sensor_idx, start, x_dps_w, rms)
    else:
        raise ValueError("--inverse_task must be denoise or inpaint")

    chirp_rate_hz = float(SPECTROGRAM_CONFIG["chirp_rate_hz"])
    spec_max_hz = min(float(SPECTROGRAM_CONFIG["max_hz"]), chirp_rate_hz / 2.0)
    target_range_bin = SPECTROGRAM_CONFIG.get("target_range_bin")
    target_range_bin = None if target_range_bin is None else int(target_range_bin)
    md_kwargs = {
        "chirp_rate_hz": chirp_rate_hz,
        "fast_time_axis": int(SPECTROGRAM_CONFIG["fast_time_axis"]),
        "range_bin_radius": int(SPECTROGRAM_CONFIG["range_bin_radius"]),
        "combine": str(SPECTROGRAM_CONFIG["combine"]),
        "nperseg": int(SPECTROGRAM_CONFIG["nperseg"]),
        "overlap_fraction": float(SPECTROGRAM_CONFIG["overlap_fraction"]),
        "nfft": int(SPECTROGRAM_CONFIG["nfft"]),
        "window": str(SPECTROGRAM_CONFIG["window"]),
        "gaussian_std": SPECTROGRAM_CONFIG["gaussian_std"],
        "eps": float(SPECTROGRAM_CONFIG["eps"]),
    }

    range_bin_reference = str(SPECTROGRAM_CONFIG.get("range_bin_reference", "clean")).lower()
    if target_range_bin is None and range_bin_reference == "observed":
        noisy_freqs, noisy_times, noisy_db, selected_range_bin, noisy_slow_iq = paper_micro_doppler_spectrogram(
            observed_full_iq.real,
            observed_full_iq.imag,
            target_range_bin=None,
            **md_kwargs,
        )
        clean_freqs, clean_times, clean_db, _, clean_slow_iq = paper_micro_doppler_spectrogram(
            clean_full_iq.real,
            clean_full_iq.imag,
            target_range_bin=selected_range_bin,
            **md_kwargs,
        )
    else:
        clean_freqs, clean_times, clean_db, selected_range_bin, clean_slow_iq = paper_micro_doppler_spectrogram(
            clean_full_iq.real,
            clean_full_iq.imag,
            target_range_bin=target_range_bin,
            **md_kwargs,
        )
        noisy_freqs, noisy_times, noisy_db, _, noisy_slow_iq = paper_micro_doppler_spectrogram(
            observed_full_iq.real,
            observed_full_iq.imag,
            target_range_bin=selected_range_bin,
            **md_kwargs,
        )

    dps_freqs, dps_times, dps_db, _, dps_slow_iq = paper_micro_doppler_spectrogram(
        dps_full_iq.real,
        dps_full_iq.imag,
        target_range_bin=selected_range_bin,
        **md_kwargs,
    )
    residual_freqs, residual_times, residual_db, _, _ = paper_micro_doppler_spectrogram(
        (dps_full_iq - clean_full_iq).real,
        (dps_full_iq - clean_full_iq).imag,
        target_range_bin=selected_range_bin,
        **md_kwargs,
    )
    baseline_classifier_result = classify_dps_spectrogram(
        observed_full_iq,
        baseline_classifier,
        baseline_label_names,
        device,
        md_kwargs,
    )
    dps_classifier_result = classify_dps_spectrogram(
        dps_full_iq,
        classifier,
        classifier_label_names,
        device,
        md_kwargs,
    )
    baseline_pred_name = baseline_classifier_result["pred_name"]
    baseline_pred_conf = baseline_classifier_result["confidence"]
    baseline_pred_range_bin = baseline_classifier_result["range_bin"]
    dps_pred_name = dps_classifier_result["pred_name"]
    dps_pred_conf = dps_classifier_result["confidence"]
    dps_pred_range_bin = dps_classifier_result["range_bin"]
    pred_title = (
        f"real=({label_name}), "
        f"baseline pred=({baseline_pred_name}, {baseline_pred_conf:.1%}), "
        f"dps pred=({dps_pred_name}, {dps_pred_conf:.1%})"
    )

    clean_real = np.real(clean_slow_iq)
    clean_imag = np.imag(clean_slow_iq)
    clean_mag = np.abs(clean_slow_iq)
    noisy_real = np.real(noisy_slow_iq)
    noisy_imag = np.imag(noisy_slow_iq)
    noisy_mag = np.abs(noisy_slow_iq)
    dps_real = np.real(dps_slow_iq)
    dps_imag = np.imag(dps_slow_iq)
    dps_mag = np.abs(dps_slow_iq)

    clean_signal = raw_iq[sensor_idx]
    noisy_signal = observed_full_iq[sensor_idx]
    dps_signal_sel = dps_full_iq[sensor_idx]
    clean_mse = float(np.mean(np.abs(noisy_signal - clean_signal) ** 2))
    dps_mse = float(np.mean(np.abs(dps_signal_sel - clean_signal) ** 2))
    time = np.arange(len(clean_real)) / chirp_rate_hz

    # --- I/Q comparison: 3 rows (clean / observed / DPS), each row has Real + Imag + Mag ---
    fig = plt.figure(figsize=(15, 22), constrained_layout=False)
    gs = fig.add_gridspec(8, 3, hspace=0.55, wspace=0.30,
                          left=0.06, right=0.89, top=0.96, bottom=0.03)

    # Row 0: clean
    ax_cr = fig.add_subplot(gs[0, 0])
    ax_ci = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[0, 2])
    ax_cr.plot(time, clean_real, color="tab:blue", linewidth=0.8)
    ax_cr.set_title("clean Real", fontsize=13)
    ax_ci.plot(time, clean_imag, color="tab:blue", linewidth=0.8)
    ax_ci.set_title("clean Imag", fontsize=13)
    ax_cm.plot(time, clean_mag, color="tab:blue", linewidth=0.8)
    ax_cm.set_title("clean Magnitude", fontsize=13)

    # Row 1: observed (noisy)
    ax_nr = fig.add_subplot(gs[1, 0], sharex=ax_cr, sharey=ax_cr)
    ax_ni = fig.add_subplot(gs[1, 1], sharex=ax_ci, sharey=ax_ci)
    ax_nm = fig.add_subplot(gs[1, 2], sharex=ax_cm, sharey=ax_cm)
    ax_nr.plot(time, noisy_real, color="tab:orange", linewidth=0.8)
    ax_nr.set_title(f"observed Real (SNR={args.snr_db:g} dB)", fontsize=13)
    ax_ni.plot(time, noisy_imag, color="tab:orange", linewidth=0.8)
    ax_ni.set_title(f"observed Imag (SNR={args.snr_db:g} dB)", fontsize=13)
    ax_nm.plot(time, noisy_mag, color="tab:orange", linewidth=0.8)
    ax_nm.set_title(f"observed Magnitude (SNR={args.snr_db:g} dB)", fontsize=13)

    # Row 2: DPS reconstructed
    ax_dr = fig.add_subplot(gs[2, 0], sharex=ax_cr, sharey=ax_cr)
    ax_di = fig.add_subplot(gs[2, 1], sharex=ax_ci, sharey=ax_ci)
    ax_dm = fig.add_subplot(gs[2, 2], sharex=ax_cm, sharey=ax_cm)
    ax_dr.plot(time, dps_real, color="tab:green", linewidth=0.8)
    ax_dr.set_title(f"DPS Real (scale={args.dps_scale:g})", fontsize=13)
    ax_di.plot(time, dps_imag, color="tab:green", linewidth=0.8)
    ax_di.set_title(f"DPS Imag (scale={args.dps_scale:g})", fontsize=13)
    ax_dm.plot(time, dps_mag, color="tab:green", linewidth=0.8)
    ax_dm.set_title(f"DPS Magnitude (scale={args.dps_scale:g})", fontsize=13)

    # Suptitle with metadata
    fig.suptitle(
        f"{args.inverse_task}, {pred_title}, SNR={args.snr_db:g} dB, "
        f"dps_scale={args.dps_scale:g}, range_bin={selected_range_bin}, raw_shape={raw_iq.shape}",
        fontsize=16, fontweight="bold", y=0.992,
    )

    # Row 3: data consistency loss trace (span full width)
    ax_loss = fig.add_subplot(gs[3, :])
    ax_loss.plot(trace)
    ax_loss.set_title(f"data consistency loss, observed MSE={clean_mse:.5f}, DPS MSE={dps_mse:.5f}", fontsize=13)
    ax_loss.set_xlabel("reverse diffusion step")
    ax_loss.set_ylabel("loss")

    # Axes list for spectrogram rows (rows 4-7 span full width, 4 panels)
    spec_axes = [fig.add_subplot(gs[r, :]) for r in range(4, 8)]

    clean_spec = (clean_freqs, clean_times, clean_db)
    noisy_spec = (noisy_freqs, noisy_times, noisy_db)
    dps_spec = (dps_freqs, dps_times, dps_db)
    residual_spec = (residual_freqs, residual_times, residual_db)
    spec_vmin, spec_vmax = shared_spectrogram_limits(clean_spec, noisy_spec, dps_spec)
    plot_spectrogram(
        spec_axes[0],
        clean_spec,
        f"clean raw paper micro-Doppler, real=({label_name}), range_bin={selected_range_bin}, CRR={chirp_rate_hz:g} Hz",
        vmin=spec_vmin,
        vmax=spec_vmax,
        ylim=(-spec_max_hz, spec_max_hz),
    )
    plot_spectrogram(
        spec_axes[1],
        noisy_spec,
        f"observed raw paper micro-Doppler, real=({label_name}), "
        f"baseline pred=({baseline_pred_name}, {baseline_pred_conf:.1%}), "
        f"range_bin={selected_range_bin}, clf_range_bin={baseline_pred_range_bin}, CRR={chirp_rate_hz:g} Hz",
        vmin=spec_vmin,
        vmax=spec_vmax,
        ylim=(-spec_max_hz, spec_max_hz),
    )
    spec_mesh = plot_spectrogram(
        spec_axes[2],
        dps_spec,
        f"DPS raw paper micro-Doppler, real=({label_name}), "
        f"dps pred=({dps_pred_name}, {dps_pred_conf:.1%}), "
        f"range_bin={selected_range_bin}, clf_range_bin={dps_pred_range_bin}, CRR={chirp_rate_hz:g} Hz",
        vmin=spec_vmin,
        vmax=spec_vmax,
        ylim=(-spec_max_hz, spec_max_hz),
    )
    residual_vmin, residual_vmax = shared_spectrogram_limits(residual_spec)
    residual_mesh = plot_spectrogram(
        spec_axes[3],
        residual_spec,
        f"DPS-clean residual raw paper micro-Doppler, real=({label_name}), range_bin={selected_range_bin}",
        vmin=residual_vmin,
        vmax=residual_vmax,
        ylim=(-spec_max_hz, spec_max_hz),
    )
    spec_axes[3].set_xlabel("time (s)")
    add_right_colorbar(fig, spec_mesh, "STFT magnitude (dB)", [0.91, 0.18, 0.015, 0.20])
    add_right_colorbar(fig, residual_mesh, "residual magnitude (dB)", [0.91, 0.05, 0.015, 0.10])
    out_path = output_dir / f"sample_{args.split}_{index}_{args.inverse_task}_{args.snr_db:g}dB.png"
    fig.savefig(out_path, dpi=160)
    print(f"Saved figure: {out_path}")
    print(f"sample path: {item['path']}")
    print(f"raw shape: {raw_iq.shape}")
    num_windows = max(1, (raw_iq.shape[1] + window_size - 1) // window_size)
    print(f"full-signal DPS: sensors={len(sensor_idx)}, total_time={raw_iq.shape[1]}, "
          f"windows={num_windows}, window_size={window_size}")
    print(
        "spectrogram mode: full raw paper pipeline, Range FFT -> target range-bin interval "
        "-> slow-time STFT -> dB"
    )
    print(f"selected range bin: {selected_range_bin}")
    print(f"baseline classifier checkpoint: {baseline_classifier_path}")
    print(f"DPS classifier checkpoint: {classifier_path}")
    print(
        f"baseline prediction: {baseline_pred_name} ({baseline_pred_conf:.2%}), "
        f"DPS prediction: {dps_pred_name} ({dps_pred_conf:.2%}), true label: {label_name}"
    )
    print(
        f"baseline classifier spectrogram shape: {baseline_classifier_result['spec_shape']}, "
        f"range bin: {baseline_pred_range_bin}"
    )
    print(
        f"DPS classifier spectrogram shape: {dps_classifier_result['spec_shape']}, "
        f"range bin: {dps_pred_range_bin}"
    )
    print(f"DPS scale: {args.dps_scale:g}")
    print(f"observed MSE: {clean_mse:.6f}")
    print(f"DPS MSE: {dps_mse:.6f}")
    result = {
        "args": args,
        "output_dir": output_dir,
        "index": index,
        "item": item,
        "label_name": label_name,
        "clean_full_iq": clean_full_iq,
        "noisy_full_iq": observed_full_iq,
        "dps_full_iq": dps_full_iq,
        "sensor_idx": sensor_idx,
        "trace": trace,
        "clean_mse": clean_mse,
        "dps_mse": dps_mse,
        "figure_path": out_path,
        "figure": fig,
        "raw_shape": raw_iq.shape,
        "selected_range_bin": selected_range_bin,
        "baseline_classifier_checkpoint": baseline_classifier_path,
        "classifier_checkpoint": classifier_path,
        "baseline_classifier_result": baseline_classifier_result,
        "dps_classifier_result": dps_classifier_result,
        "classifier_result": dps_classifier_result,
        "model_window_start": start,
        "model_window_size": window_size,
    }
    if args.show and show_plot:
        show_figure(fig)
    return result


def main() -> None:
    global LAST_RESULT
    args = parse_args()
    LAST_RESULT = run_one_sample(args, show_plot=True, show_progress=True)


def run_until_prediction_disagreement(
    args: argparse.Namespace | None = None,
    *,
    max_tries: int = 20,
    show_progress: bool = True,
    show_rejected: bool = False,
) -> dict:
    """Visualize the first random sample where baseline and DPS predictions differ."""
    global LAST_RESULT
    if args is None:
        args = argparse.Namespace(**CONFIG)
    base_args = argparse.Namespace(**vars(args))
    base_args.sample_index = -1

    for attempt in range(1, max_tries + 1):
        trial_args = argparse.Namespace(**vars(base_args))
        trial_args.seed = int(base_args.seed) + attempt - 1
        trial_args.show = bool(show_rejected)
        result = run_one_sample(trial_args, show_plot=show_rejected, show_progress=show_progress)

        baseline_pred = result["baseline_classifier_result"]["pred_name"]
        dps_pred = result["dps_classifier_result"]["pred_name"]
        real = result["label_name"]
        print(
            f"try {attempt}/{max_tries}: index={result['index']}, "
            f"real=({real}), baseline pred=({baseline_pred}), dps pred=({dps_pred})"
        )
        if baseline_pred != dps_pred:
            LAST_RESULT = result
            if args.show:
                show_figure(result["figure"])
            print("Found disagreement sample.")
            return result

        plt.close(result["figure"])

    raise RuntimeError(f"No baseline/DPS prediction disagreement found in {max_tries} random tries.")


def bandpass_filter_1d(signal: np.ndarray, fs_hz: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyquist = fs_hz / 2.0
    if low_hz <= 0:
        raise ValueError("low_hz must be positive.")
    if high_hz >= nyquist:
        raise ValueError(f"high_hz must be lower than Nyquist frequency ({nyquist:g} Hz).")
    if low_hz >= high_hz:
        raise ValueError("low_hz must be lower than high_hz.")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, signal)


def get_or_make_last_result() -> dict:
    global LAST_RESULT
    if LAST_RESULT is None:
        args = argparse.Namespace(**CONFIG)
        args.show = False
        LAST_RESULT = run_one_sample(args, show_plot=False, show_progress=True)
    return LAST_RESULT


def run_paper_micro_doppler_for_one_sample(
    sample_index: int | None = None,
    *,
    split: str | None = None,
    show: bool = True,
    save: bool = True,
) -> dict:
    """Run only the paper-style micro-Doppler preprocessing on one raw .mat file.

    This bypasses the diffusion model, DPS sampling, denoising, threshold
    masking, entropy range-bin search, and classifier. It loads the original
    complex FMCW matrix from the selected .mat file and applies:

    Range FFT -> fixed target range-bin interval -> slow-time STFT -> dB image.
    """
    random.seed(CONFIG["seed"])
    data_root = Path(CONFIG["data_root"]).resolve()
    extracted_dir = Path(CONFIG["extracted_dir"])
    if not extracted_dir.is_absolute():
        extracted_dir = data_root / extracted_dir
    output_dir = Path(CONFIG["output_dir"])
    if not output_dir.is_absolute():
        output_dir = data_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records, label_names = build_records(
        data_root,
        extracted_dir,
        auto_extract=bool(CONFIG["auto_extract"]),
        custom_extractor=CONFIG["extractor"] or None,
    )
    train_records, val_records, test_records = split_records(records, seed=int(CONFIG["seed"]))
    split_name = split or str(CONFIG["split"])
    split_map = {"train": train_records, "val": val_records, "test": test_records}
    if split_name not in split_map:
        raise ValueError(f"split must be train, val, or test, got {split_name!r}.")
    split_records_list = split_map[split_name]

    if sample_index is None:
        configured_index = int(CONFIG["sample_index"])
        index = configured_index if configured_index >= 0 else random.randrange(len(split_records_list))
    else:
        index = int(sample_index)
    if not 0 <= index < len(split_records_list):
        raise ValueError(f"sample_index={index} is outside [0, {len(split_records_list) - 1}].")

    record = split_records_list[index]
    raw = load_complex_signal(record.path)
    target_range_bin = SPECTROGRAM_CONFIG.get("target_range_bin")
    target_range_bin = None if target_range_bin is None else int(target_range_bin)
    freqs, times, spec_db, idxmax, slow_iq = paper_micro_doppler_spectrogram(
        raw.real,
        raw.imag,
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
    vmin, vmax = shared_spectrogram_limits((freqs, times, spec_db))
    label_name = label_names[record.label_id]
    fig, ax, image = plot_paper_micro_doppler_spectrogram(
        freqs,
        times,
        spec_db,
        title=f"paper micro-Doppler, label={label_name}, range_bin={idxmax}, raw shape={raw.shape}",
        cmap=str(SPECTROGRAM_CONFIG["cmap"]),
        vmin=vmin,
        vmax=vmax,
    )
    out_path = output_dir / f"paper_micro_doppler_{split_name}_{index}_rangebin_{idxmax}.png"
    if save:
        fig.savefig(out_path, dpi=160)
        print(f"Saved paper micro-Doppler figure: {out_path}")
    print(f"sample path: {record.path}")
    print(f"raw shape: {raw.shape}")
    print(
        "spectrogram mode: paper pipeline only, Range FFT -> selected range-bin interval "
        "-> slow-time STFT -> dB; no denoising"
    )
    print(f"selected range bin: {idxmax}")
    print(f"freq range: {freqs[0]:g} to {freqs[-1]:g} Hz")
    print(f"time range: {times[0]:g} to {times[-1]:g} s")
    if show:
        show_figure(fig)
    return {
        "figure": fig,
        "figure_path": out_path,
        "record": record,
        "label_name": label_name,
        "raw_shape": raw.shape,
        "freqs": freqs,
        "times": times,
        "spec_db": spec_db,
        "idxmax": idxmax,
        "slow_iq": slow_iq,
    }


def plot_bpf_for_last_result() -> Path:
    if BPF_CONFIG["use_last_result"]:
        result = get_or_make_last_result()
    else:
        args = argparse.Namespace(**CONFIG)
        args.show = False
        result = run_one_sample(args, show_plot=False, show_progress=True)

    args = result["args"]
    sensor_index = args.sensor_index
    fs_hz = float(BPF_CONFIG["fs_hz"])
    low_hz = float(BPF_CONFIG["low_hz"])
    high_hz = float(BPF_CONFIG["high_hz"])
    order = int(BPF_CONFIG["order"])

    si = result["sensor_idx"][sensor_index]
    clean_mag = np.abs(result["clean_full_iq"][si])
    observed_mag = np.abs(result["noisy_full_iq"][si])
    dps_mag = np.abs(result["dps_full_iq"][si])

    clean_bpf = bandpass_filter_1d(clean_mag, fs_hz, low_hz, high_hz, order)
    observed_bpf = bandpass_filter_1d(observed_mag, fs_hz, low_hz, high_hz, order)
    dps_bpf = bandpass_filter_1d(dps_mag, fs_hz, low_hz, high_hz, order)

    nperseg = min(256, len(clean_bpf))
    f_clean, p_clean = welch(clean_bpf, fs=fs_hz, nperseg=nperseg)
    f_obs, p_obs = welch(observed_bpf, fs=fs_hz, nperseg=nperseg)
    f_dps, p_dps = welch(dps_bpf, fs=fs_hz, nperseg=nperseg)

    t = np.arange(len(clean_mag)) / fs_hz
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=False)
    axes[0].plot(t, clean_mag, label="clean magnitude", linewidth=1.0)
    axes[0].plot(t, observed_mag, label="observed magnitude", alpha=0.65, linewidth=0.9)
    axes[0].plot(t, dps_mag, label="DPS magnitude", alpha=0.85, linewidth=0.9)
    axes[0].set_title(
        f"raw magnitude, label={result['label_name']}, sample={result['index']}, sensor={sensor_index}"
    )
    axes[0].set_ylabel("normalized amplitude")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, clean_bpf, label="clean BPF", linewidth=1.0)
    axes[1].plot(t, observed_bpf, label="observed BPF", alpha=0.65, linewidth=0.9)
    axes[1].plot(t, dps_bpf, label="DPS BPF", alpha=0.85, linewidth=0.9)
    axes[1].set_title(f"Butterworth BPF: {low_hz:g}-{high_hz:g} Hz, order={order}, fs={fs_hz:g} Hz")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("BPF amplitude")
    axes[1].legend(loc="upper right")

    axes[2].semilogy(f_clean, p_clean + 1e-12, label="clean BPF PSD")
    axes[2].semilogy(f_obs, p_obs + 1e-12, label="observed BPF PSD", alpha=0.75)
    axes[2].semilogy(f_dps, p_dps + 1e-12, label="DPS BPF PSD", alpha=0.85)
    axes[2].axvspan(low_hz, high_hz, color="gray", alpha=0.15)
    axes[2].set_xlim(0, min(fs_hz / 2.0, high_hz * 2.5))
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("PSD")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    out_path = result["output_dir"] / (
        f"sample_{args.split}_{result['index']}_{args.inverse_task}_{args.snr_db:g}dB_bpf.png"
    )
    if BPF_CONFIG["save"]:
        fig.savefig(out_path, dpi=160)
        print(f"Saved BPF figure: {out_path}")
    if BPF_CONFIG["show"]:
        show_figure(fig)
    return out_path


def run_snr_sweep_for_same_sample() -> list[dict]:
    global LAST_RESULT
    results = []
    base_args = argparse.Namespace(**CONFIG)

    if LAST_RESULT is not None:
        base_args.sample_index = LAST_RESULT["index"]
    elif base_args.sample_index < 0:
        base_args.sample_index = 0

    for snr_db in SNR_SWEEP_CONFIG["snr_db_list"]:
        args = argparse.Namespace(**vars(base_args))
        args.snr_db = float(snr_db)
        args.show = bool(SNR_SWEEP_CONFIG["show_each"])
        result = run_one_sample(args, show_plot=False, show_progress=True)
        LAST_RESULT = result
        results.append(result)
        if SNR_SWEEP_CONFIG["run_bpf_after_each"]:
            plot_bpf_for_last_result()
    if SNR_SWEEP_CONFIG["show_each"]:
        plt.show()
    return results


def run_dps_scale_sweep_for_same_sample() -> list[dict]:
    global LAST_RESULT
    results = []
    base_args = argparse.Namespace(**CONFIG)

    if LAST_RESULT is not None:
        base_args.sample_index = LAST_RESULT["index"]
        base_args.snr_db = LAST_RESULT["args"].snr_db
    elif base_args.sample_index < 0:
        base_args.sample_index = 0

    for dps_scale in DPS_SCALE_SWEEP_CONFIG["dps_scale_list"]:
        args = argparse.Namespace(**vars(base_args))
        args.dps_scale = float(dps_scale)
        args.show = bool(DPS_SCALE_SWEEP_CONFIG["show_each"])
        result = run_one_sample(args, show_plot=False, show_progress=True)
        LAST_RESULT = result
        results.append(result)

    if DPS_SCALE_SWEEP_CONFIG["show_each"]:
        plt.show()
    return results


import importlib
import check_one_sample_random_visualize as viz

importlib.reload(viz)

args = viz.argparse.Namespace(**viz.CONFIG)
args.sample_index = 5
args.split = "test"
args.snr_db = -40
args.dps_scale = 1.0
args.show = True

viz.LAST_RESULT = viz.run_one_sample(
    args,
    show_plot=True,
    show_progress=True,
)

print(viz.LAST_RESULT["baseline_classifier_result"])
print(viz.LAST_RESULT["dps_classifier_result"])


# %%
# VS Code/Jupyter Interactive Window:
# Run this cell after the main visualization cell to apply a biological-signal BPF
# to the same sample. Adjust BPF_CONFIG["fs_hz"] if the real sampling rate is known.
# plot_bpf_for_last_result()


#%%
# VS Code/Jupyter Interactive Window:
# Run this cell for the paper-style micro-Doppler spectrogram only.
# This does not run diffusion, DPS, denoising, threshold masking, or classification.
# PAPER_MD_RESULT = run_paper_micro_doppler_for_one_sample()


# %%
# VS Code/Jupyter Interactive Window:
# Run every configured SNR level for the same sample index.
# If a sample was already visualized, this reuses that index.
# SNR_SWEEP_RESULTS = run_snr_sweep_for_same_sample()


# %%
# Interactive example:
# import importlib
# import check_one_sample_random_visualize as viz
#
# importlib.reload(viz)
#
# args = viz.argparse.Namespace(**viz.CONFIG)
# args.sample_index = -1
# args.split = "test"
# args.snr_db = -15
# args.dps_scale = 1.0
# args.show = True
# args.classifier_checkpoint = "runs/classifier_dps/best.pt"
# args.baseline_classifier_checkpoint = "runs/classifier_noisy/best.pt"
#
# viz.LAST_RESULT = viz.run_one_sample(
#     args,
#     show_plot=True,
#     show_progress=True,
# )
#
# print(viz.LAST_RESULT["baseline_classifier_result"])
# print(viz.LAST_RESULT["dps_classifier_result"])
