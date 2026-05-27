# %%
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


DEFAULT_SNRS = (-10.0, -5.0, 0.0, 5.0, 10.0)


# %%
def snr_tag(snr_db: float) -> str:
    value = int(snr_db) if float(snr_db).is_integer() else snr_db
    return f"snr_m{abs(value)}" if value < 0 else f"snr_{value}"


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    signal_power = float(np.mean(np.abs(signal) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    if np.iscomplexobj(signal):
        sigma = np.sqrt(noise_power / 2.0)
        noise = sigma * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    else:
        sigma = np.sqrt(noise_power)
        noise = sigma * rng.standard_normal(signal.shape)

    return signal + noise.astype(signal.dtype, copy=False), signal_power, noise_power


def load_signal(path: Path) -> np.ndarray:
    data = loadmat(path)
    if "received_time_domain_signal" not in data:
        keys = [key for key in data if not key.startswith("__")]
        raise KeyError(f"{path} does not contain received_time_domain_signal. Keys: {keys}")
    return np.asarray(data["received_time_domain_signal"])


def iter_mat_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.mat"))


# %%
def write_metadata_header(metadata_path: Path, append: bool) -> None:
    if append and metadata_path.exists():
        return
    with metadata_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "source_file",
                "output_file",
                "label",
                "snr_db",
                "repeat",
                "seed",
                "signal_power",
                "noise_power",
            ]
        )


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="Create AWGN-noisy .mat files from extracted_data.")
    parser.add_argument("--input-dir", type=Path, default=Path("extracted_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("extracted_data_awgn"))
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()

    files = iter_mat_files(args.input_dir)
    if not files:
        raise SystemExit(f"No .mat files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.csv"
    write_metadata_header(metadata_path, append=not args.overwrite)

    total = len(files) * len(args.snrs) * args.repeats
    done = 0
    skipped = 0

    with metadata_path.open("a", newline="", encoding="utf-8") as metadata_fp:
        writer = csv.writer(metadata_fp)
        for sample_index, source_path in enumerate(files):
            label = source_path.parts[-3]
            inner_label = source_path.parts[-2]
            signal = load_signal(source_path)

            for snr_index, snr_db in enumerate(args.snrs):
                tag = snr_tag(snr_db)
                for repeat in range(1, args.repeats + 1):
                    output_dir = args.output_dir / label / inner_label / tag
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{source_path.stem}_aug{repeat}.mat"

                    if output_path.exists() and not args.overwrite:
                        skipped += 1
                        done += 1
                        continue

                    seed_sequence = np.random.SeedSequence([args.seed, sample_index, snr_index, repeat])
                    rng = np.random.default_rng(seed_sequence)
                    noisy_signal, signal_power, noise_power = add_awgn(signal, snr_db, rng)

                    savemat(
                        output_path,
                        {
                            "received_time_domain_signal": noisy_signal,
                            "snr_db": np.array([[snr_db]], dtype=np.float64),
                            "awgn_repeat": np.array([[repeat]], dtype=np.int32),
                            "source_file": str(source_path),
                            "signal_power": np.array([[signal_power]], dtype=np.float64),
                            "noise_power": np.array([[noise_power]], dtype=np.float64),
                        },
                        do_compression=args.compress,
                    )
                    writer.writerow(
                        [
                            str(source_path),
                            str(output_path),
                            label,
                            snr_db,
                            repeat,
                            seed_sequence.entropy,
                            signal_power,
                            noise_power,
                        ]
                    )
                    done += 1

                    if done % 100 == 0 or done == total:
                        print(f"{done}/{total} files processed ({skipped} skipped)", flush=True)

    print(f"Finished: {done - skipped} written, {skipped} skipped, {total} total targets")


if __name__ == "__main__":
    main()


# %%
# Ctrl+Enter quick run in VS Code:
# main()
