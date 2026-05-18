#%%
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


RangeCombine = Literal["sum", "mean", "magnitude_sum", "magnitude_mean"]


def _normalize_axis(axis: int, ndim: int, name: str) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"{name}={axis} is invalid for ndim={ndim}.")
    return axis % ndim


def _stft_window(window: str, nperseg: int, gaussian_std: float | None) -> str | tuple[str, float]:
    if window.lower() == "gaussian":
        return ("gaussian", float(gaussian_std) if gaussian_std is not None else nperseg / 6.0)
    return window


def paper_micro_doppler_spectrogram(
    real: np.ndarray,
    imag: np.ndarray,
    *,
    chirp_rate_hz: float = 1000.0,
    fast_time_axis: int = 0,
    target_range_bin: int | None = None,
    range_bin_radius: int = 1,
    combine: RangeCombine = "magnitude_sum",
    nperseg: int = 128,
    overlap_fraction: float = 0.90,
    nfft: int | None = None,
    window: str = "gaussian",
    gaussian_std: float | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Generate a paper-style FMCW radar micro-Doppler spectrogram.

    This implements the non-denoising signal-processing path used before the
    classifier in PLOS ONE journal.pone.0308045:

    1. Build complex raw ADC data: ``iq = real + 1j * imag``.
    2. Apply Range FFT along the fast-time axis for every chirp.
    3. Find the strongest range-bin index for each chirp.
    4. Use the most frequently occurring strongest bin as ``idxmax`` unless
       ``target_range_bin`` is provided manually.
    5. Select the interval ``idxmax - range_bin_radius : idxmax + range_bin_radius``.
       The paper reports the best fixed interval as radius 1, i.e. 3 bins.
    6. Apply STFT along the slow-time/chirp axis on the selected range bins.
    7. Convert the result to dB magnitude.

    This function deliberately does not implement entropy-based optimal
    range-bin search, cut-threshold masking, denoising, or classifier training.

    Parameters
    ----------
    real, imag:
        Real and imaginary raw ADC matrices. Both must be 2D and have the same
        shape. The paper's convention is ``x[j, i]`` where each column ``i`` is
        one chirp and each row ``j`` is a fast-time sample, so the default
        ``fast_time_axis=0`` matches shape ``(num_fast_time_samples, num_chirps)``.
        If your data is ``(num_chirps, num_fast_time_samples)``, set
        ``fast_time_axis=1``.
    chirp_rate_hz:
        Slow-time sampling rate. The paper's chirp duration is 1 ms, so the
        default is 1000 Hz.
    target_range_bin:
        Optional manual range-bin index. If omitted, ``idxmax`` is found as the
        mode of the per-chirp max-magnitude range-bin indexes.
    range_bin_radius:
        Number of bins to include around ``idxmax``. The paper's selected
        interval of 3 bins corresponds to radius 1.
    combine:
        How selected range-bin STFTs are combined. ``"magnitude_sum"`` is the
        default for classifier spectrogram images because it preserves energy
        from adjacent bins without complex phase cancellation. ``"sum"`` and
        ``"mean"`` combine complex STFTs.
    nperseg, overlap_fraction, nfft:
        STFT settings. The paper uses window size 128 and 90% overlap.
    window, gaussian_std:
        STFT window. The paper specifies a Gaussian window for the denoising
        procedure; this implementation uses it by default for the same
        spectrogram construction. If ``window="hann"``, ``gaussian_std`` is ignored.
    eps:
        Stabilizer for ``20 * log10(magnitude + eps)``.

    Returns
    -------
    freqs, times, spec_db, idxmax, slow_iq:
        ``freqs`` are fftshifted Doppler frequencies in Hz,
        ``times`` are STFT frame times in seconds,
        ``spec_db`` is the micro-Doppler magnitude image in dB,
        ``idxmax`` is the selected center range bin,
        and ``slow_iq`` is the complex 1D slow-time signal obtained by summing
        the selected range bins before STFT. It is returned for debugging and
        plotting only; ``spec_db`` is generated from selected range-bin STFTs.

    Example
    -------
    ```python
    freqs, times, spec_db, idxmax, slow_iq = paper_micro_doppler_spectrogram(
        real_adc,
        imag_adc,
        chirp_rate_hz=1000.0,
        fast_time_axis=0,      # shape: [fast_time_samples, chirps]
        range_bin_radius=1,
    )
    fig, ax, image = plot_micro_doppler_spectrogram(freqs, times, spec_db)
    plt.show()
    ```
    """
    real_arr = np.asarray(real)
    imag_arr = np.asarray(imag)
    if real_arr.shape != imag_arr.shape:
        raise ValueError(f"real and imag must have the same shape, got {real_arr.shape} and {imag_arr.shape}.")
    if real_arr.ndim != 2:
        raise ValueError(f"real and imag must be 2D raw ADC matrices, got shape {real_arr.shape}.")
    if chirp_rate_hz <= 0:
        raise ValueError("chirp_rate_hz must be positive.")
    if range_bin_radius < 0:
        raise ValueError("range_bin_radius must be non-negative.")
    if combine not in {"sum", "mean", "magnitude_sum", "magnitude_mean"}:
        raise ValueError('combine must be "sum", "mean", "magnitude_sum", or "magnitude_mean".')
    if nperseg <= 0:
        raise ValueError("nperseg must be positive.")
    if nfft is not None and nfft <= 0:
        raise ValueError("nfft must be positive when provided.")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must satisfy 0 <= overlap_fraction < 1.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    fast_axis = _normalize_axis(fast_time_axis, real_arr.ndim, "fast_time_axis")
    slow_axis = 1 - fast_axis
    num_fast_time = real_arr.shape[fast_axis]
    num_chirps = real_arr.shape[slow_axis]
    if num_fast_time == 0 or num_chirps == 0:
        raise ValueError(f"Input axes must be non-empty, got shape {real_arr.shape}.")

    iq = real_arr.astype(np.float64, copy=False) + 1j * imag_arr.astype(np.float64, copy=False)
    range_fft = np.fft.fft(iq, axis=fast_axis)

    if target_range_bin is None:
        per_chirp_max_idx = np.argmax(np.abs(range_fft), axis=fast_axis).astype(np.int64, copy=False)
        counts = np.bincount(per_chirp_max_idx.ravel(), minlength=num_fast_time)
        idxmax = int(np.argmax(counts))
    else:
        idxmax = int(target_range_bin)
        if not 0 <= idxmax < num_fast_time:
            raise ValueError(f"target_range_bin={idxmax} is outside [0, {num_fast_time - 1}].")

    start = max(0, idxmax - int(range_bin_radius))
    stop = min(num_fast_time, idxmax + int(range_bin_radius) + 1)
    if fast_axis == 0:
        selected = range_fft[start:stop, :]
    else:
        selected = range_fft[:, start:stop].T
    if selected.ndim != 2 or selected.shape[1] != num_chirps:
        raise ValueError(f"Selected range bins should have shape [bins, chirps], got {selected.shape}.")

    slow_iq = selected.sum(axis=0)
    stft_nperseg = min(int(nperseg), num_chirps)
    hop = max(1, int(round(stft_nperseg * (1.0 - float(overlap_fraction)))))
    noverlap = max(0, stft_nperseg - hop)
    stft_nfft = max(int(nfft) if nfft is not None else stft_nperseg, stft_nperseg)
    stft_window = _stft_window(window, stft_nperseg, gaussian_std)

    zxx_list = []
    for range_bin_signal in selected:
        freqs, times, zxx = stft(
            range_bin_signal,
            fs=chirp_rate_hz,
            window=stft_window,
            nperseg=stft_nperseg,
            noverlap=noverlap,
            nfft=stft_nfft,
            detrend=False,
            return_onesided=False,
            boundary=None,
            padded=False,
        )
        zxx_list.append(np.fft.fftshift(zxx, axes=0))

    zxx_stack = np.stack(zxx_list, axis=0)
    if combine == "sum":
        magnitude = np.abs(zxx_stack.sum(axis=0))
    elif combine == "mean":
        magnitude = np.abs(zxx_stack.mean(axis=0))
    elif combine == "magnitude_mean":
        magnitude = np.abs(zxx_stack).mean(axis=0)
    else:
        magnitude = np.abs(zxx_stack).sum(axis=0)

    freqs = np.fft.fftshift(freqs)
    spec_db = 20.0 * np.log10(magnitude + eps)
    return freqs, times, spec_db, idxmax, slow_iq


def _imshow_extent(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Axis values must be a non-empty 1D array.")
    if arr.size == 1:
        center = float(arr[0])
        return center - 0.5, center + 0.5
    step = float(np.median(np.diff(arr)))
    return float(arr[0] - 0.5 * step), float(arr[-1] + 0.5 * step)


def plot_micro_doppler_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    spec_db: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "Micro-Doppler spectrogram",
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = True,
) -> tuple[plt.Figure, plt.Axes, object]:
    """Plot a micro-Doppler spectrogram.

    The plot uses ``origin="lower"`` and ``aspect="auto"``. The x-axis is
    time in seconds, the y-axis is Doppler frequency in Hz, and the colorbar is
    labeled ``Magnitude (dB)``.
    """
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64)
    spec_arr = np.asarray(spec_db, dtype=np.float64)
    if freqs_arr.ndim != 1 or times_arr.ndim != 1:
        raise ValueError("freqs and times must be 1D arrays.")
    expected_shape = (freqs_arr.size, times_arr.size)
    if spec_arr.shape != expected_shape:
        raise ValueError(f"spec_db must have shape {expected_shape}, got {spec_arr.shape}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    x0, x1 = _imshow_extent(times_arr)
    y0, y1 = _imshow_extent(freqs_arr)
    image = ax.imshow(
        spec_arr,
        origin="lower",
        aspect="auto",
        extent=(x0, x1, y0, y1),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Doppler frequency (Hz)")
    if add_colorbar:
        fig.colorbar(image, ax=ax, label="Magnitude (dB)")
    return fig, ax, image
