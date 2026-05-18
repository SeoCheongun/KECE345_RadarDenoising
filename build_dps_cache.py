# %%
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from data_utils import DEFAULT_SNR_LEVELS, RadarWindowDataset, add_awgn_torch, build_records, split_records
from diffusion_model import EpsilonDenoiser1D
from diffusion_process import VPDiffusion


CONFIG = {
    "diffusion_checkpoint": "runs/diffusion_dps/best.pt",
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "output_dir": "runs/dps_cache",
    "auto_extract": True,
    "extractor": "",
    "splits": "train,val,test",
    "snr_levels": ",".join(str(x) for x in DEFAULT_SNR_LEVELS),
    "sources": "clean,noisy,dps",
    "batch_size": 8,
    "num_workers": 0,
    "preload_clean": True,
    "dps_scale": 1.0,
    "sampler_noise_scale": 0.0,
    "start_from_measurement": False,
    "dtype": "float16",
    "max_items": 0,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute DPS-denoised radar windows for classifier training.")
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


def parse_float_list(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def parse_str_list(value: str) -> list[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one split.")
    return out


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_diffusion_prior(checkpoint: dict, device: torch.device) -> tuple[EpsilonDenoiser1D, VPDiffusion]:
    model = EpsilonDenoiser1D(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint.get("ema_model_state", checkpoint["model_state"]))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    diffusion = VPDiffusion(**checkpoint["diffusion_kwargs"]).to(device)
    return model, diffusion


def cache_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError("--dtype must be float16 or float32")


def dps_sample_compat(
    diffusion: VPDiffusion,
    model: EpsilonDenoiser1D,
    measurement: torch.Tensor,
    *,
    noise_std: float,
    dps_scale: float,
    sampler_noise_scale: float,
    initial: torch.Tensor | None,
) -> torch.Tensor:
    kwargs = {
        "noise_std": noise_std,
        "dps_scale": dps_scale,
        "initial": initial,
    }
    if "sampler_noise_scale" in inspect.signature(diffusion.dps_sample).parameters:
        kwargs["sampler_noise_scale"] = sampler_noise_scale
    elif sampler_noise_scale != 1.0:
        print("Warning: loaded VPDiffusion has no sampler_noise_scale; restart/reload diffusion_process.py.")
    return diffusion.dps_sample(model, measurement, **kwargs)


def snr_token(snr_db: float) -> str:
    return f"{snr_db:g}".replace("-", "m").replace(".", "p")


class PreloadedCacheInputDataset(Dataset):
    def __init__(self, source: Dataset, *, desc: str) -> None:
        xs = []
        labels = []
        paths = []
        starts = []
        for index in tqdm(range(len(source)), desc=desc, leave=True):
            item = source[index]
            xs.append(item["x"])
            labels.append(item["label"])
            paths.append(item["path"])
            starts.append(int(item["start"]))
        self.x = torch.stack(xs, dim=0).contiguous()
        self.label = torch.stack(labels, dim=0).long()
        self.paths = paths
        self.starts = starts

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "x": self.x[index],
            "label": self.label[index],
            "path": self.paths[index],
            "start": self.starts[index],
        }


def normalized_snr(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def add_manifest_entry(files: list[dict], entry: dict) -> None:
    entry_source = entry.get("source", "dps")
    entry_snr = normalized_snr(entry.get("snr_db"))
    files[:] = [
        item
        for item in files
        if not (
            item.get("split") == entry.get("split")
            and item.get("source", "dps") == entry_source
            and normalized_snr(item.get("snr_db")) == entry_snr
        )
    ]
    files.append(entry)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    torch.manual_seed(args.seed)
    sources = parse_str_list(args.sources)
    for source in sources:
        if source not in {"clean", "noisy", "dps"}:
            raise ValueError("--sources entries must be clean, noisy, or dps")

    device = torch.device(args.device)
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = data_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.diffusion_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = data_root / checkpoint_path
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = None
    diffusion = None
    if "dps" in sources:
        model, diffusion = build_diffusion_prior(checkpoint, device)
    data_config = checkpoint["data_kwargs"]

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
    record_map = {"train": train_records, "val": val_records, "test": test_records}
    snr_levels = parse_float_list(args.snr_levels)
    splits = parse_str_list(args.splits)
    out_dtype = cache_dtype(args.dtype)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_files = list(existing_manifest.get("files", []))
    else:
        manifest_files = []

    print("")
    print("Building classifier data cache")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output: {output_dir}")
    print(f"  splits: {splits}")
    print(f"  SNR levels: {snr_levels}")
    print(f"  sources: {sources}")
    print(f"  dps_scale: {args.dps_scale}, sampler_noise_scale: {args.sampler_noise_scale}")

    for split in splits:
        if split not in record_map:
            raise ValueError(f"Unknown split: {split}")
        dataset = RadarWindowDataset(
            record_map[split],
            window_size=data_config["window_size"],
            num_sensors=data_config["num_sensors"],
            windows_per_record=data_config.get("windows_per_record", 1) if split == "train" else 1,
            random_crop=split == "train",
            seed=args.seed,
        )
        if args.max_items > 0 and len(dataset) > args.max_items:
            dataset = Subset(dataset, list(range(args.max_items)))
        if args.preload_clean:
            dataset = PreloadedCacheInputDataset(dataset, desc=f"preload clean split={split}")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        if "clean" in sources:
            xs = []
            labels = []
            paths = []
            starts = []
            progress = tqdm(loader, desc=f"save clean split={split}", leave=True)
            for batch in progress:
                xs.append(batch["x"].cpu().to(out_dtype))
                labels.append(batch["label"].cpu())
                paths.extend(list(batch["path"]))
                starts.extend([int(x) for x in batch["start"]])

            cache_path = output_dir / f"{split}_clean.pt"
            payload = {
                "x": torch.cat(xs, dim=0),
                "label": torch.cat(labels, dim=0),
                "paths": paths,
                "starts": starts,
                "split": split,
                "source": "clean",
                "snr_db": None,
                "dtype": args.dtype,
            }
            torch.save(payload, cache_path)
            add_manifest_entry(
                manifest_files,
                {
                    "path": cache_path.name,
                    "split": split,
                    "source": "clean",
                    "snr_db": None,
                    "samples": int(payload["x"].shape[0]),
                    "shape": list(payload["x"].shape),
                },
            )
            print(f"Saved {cache_path} ({payload['x'].shape[0]} samples)")

        for snr_index, snr_db in enumerate(snr_levels):
            noisy_xs = []
            dps_xs = []
            labels = []
            paths = []
            starts = []
            generator = torch.Generator(device=device).manual_seed(args.seed + 1009 * (snr_index + 1))
            source_text = "+".join(source for source in ("noisy", "dps") if source in sources)
            progress = tqdm(
                loader,
                desc=f"save {source_text} split={split} SNR={snr_db:g}dB",
                leave=True,
            )
            for batch in progress:
                x0 = batch["x"].to(device, non_blocking=True)
                y_noisy, noise_std = add_awgn_torch(x0, snr_db, generator)
                if "noisy" in sources:
                    noisy_xs.append(y_noisy.detach().cpu().to(out_dtype))
                if "dps" in sources:
                    initial = y_noisy if args.start_from_measurement else None
                    with torch.enable_grad():
                        if model is None or diffusion is None:
                            raise RuntimeError("DPS source requested but diffusion model was not loaded.")
                        x_dps = dps_sample_compat(
                            diffusion,
                            model,
                            measurement=y_noisy,
                            noise_std=noise_std,
                            dps_scale=args.dps_scale,
                            sampler_noise_scale=args.sampler_noise_scale,
                            initial=initial,
                        )
                    dps_xs.append(x_dps.detach().cpu().to(out_dtype))
                labels.append(batch["label"].cpu())
                paths.extend(list(batch["path"]))
                starts.extend([int(x) for x in batch["start"]])

            for source, xs in (("noisy", noisy_xs), ("dps", dps_xs)):
                if source not in sources:
                    continue
                cache_path = output_dir / f"{split}_{source}_snr_{snr_token(snr_db)}.pt"
                payload = {
                    "x": torch.cat(xs, dim=0),
                    "label": torch.cat(labels, dim=0),
                    "paths": paths,
                    "starts": starts,
                    "split": split,
                    "source": source,
                    "snr_db": float(snr_db),
                    "dtype": args.dtype,
                }
                torch.save(payload, cache_path)
                add_manifest_entry(
                    manifest_files,
                    {
                        "path": cache_path.name,
                        "split": split,
                        "source": source,
                        "snr_db": float(snr_db),
                        "samples": int(payload["x"].shape[0]),
                        "shape": list(payload["x"].shape),
                    },
                )
                print(f"Saved {cache_path} ({payload['x'].shape[0]} samples)")

    manifest = {
        "label_names": label_names,
        "data_kwargs": data_config,
        "diffusion_checkpoint": str(checkpoint_path),
        "cache_config": vars(args),
        "files": manifest_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
#%%

if __name__ == "__main__":
    main()


# %%
# VS Code/Jupyter Interactive Window:
# Run this cell to create only noisy-signal cache from the raw clean signal.
# This does not run DPS and keeps existing DPS cache files/manifest entries.
if "get_ipython" in globals():
    NOISY_CACHE_ARGS = argparse.Namespace(**CONFIG)
    NOISY_CACHE_ARGS.sources = "noisy"
    NOISY_CACHE_ARGS.batch_size = 64
    main(NOISY_CACHE_ARGS)
