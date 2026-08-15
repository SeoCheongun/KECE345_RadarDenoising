# KECE345 Radar Denoising

Simple radar signal denoising experiments for KECE345. The repository contains scripts for generating AWGN-corrupted radar measurements and applying a diffusion-prior/DPS denoising pipeline to complex time-domain radar data.

## Repository Structure

- `make_awgn_noisy_dataset.py`: creates noisy `.mat` files from clean radar signals at multiple SNR levels.
- `dps_awgn_denoiser.py`: trains a clean diffusion prior, checks DPS denoising performance, creates spectrogram visualizations, and exports denoised `.mat` files.

## Data Format

The scripts expect MATLAB `.mat` files containing:

```text
received_time_domain_signal
```

By default, clean data is read from:

```text
extracted_data/
```

Generated noisy data and experiment outputs are ignored by Git.

## Usage

Create AWGN-noisy data:

```bash
python make_awgn_noisy_dataset.py --input-dir extracted_data --output-dir extracted_data_awgn --overwrite
```

Run a small smoke test:

```bash
python dps_awgn_denoiser.py --mode smoke_test
```

Train the clean diffusion prior:

```bash
python dps_awgn_denoiser.py --mode train_prior --output_dir runs/awgn_dps_full102
```

Check DPS denoising performance:

```bash
python dps_awgn_denoiser.py --mode check --checkpoint runs/awgn_dps_full102/best.pt --output_dir runs/awgn_dps_full102/check
```

Export denoised `.mat` files:

```bash
python dps_awgn_denoiser.py --mode export --checkpoint runs/awgn_dps_full102/best.pt --export_dir denoised_data_dps_full102
```

## Outputs

Typical outputs include:

- `runs/.../best.pt`: best diffusion-prior checkpoint.
- `runs/.../history.json`: training history.
- `runs/.../check/summary_by_snr.csv`: denoising metrics grouped by SNR.
- `runs/.../check/figures/`: clean, noisy, and DPS-denoised spectrogram comparisons.
- `denoised_data_dps_full102/`: exported denoised `.mat` files.
