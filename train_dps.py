# %%
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_utils import (
    RadarWindowDataset,
    build_records,
    save_label_map,
    split_records,
    write_records_csv,
)
from diffusion_model import EpsilonDenoiser1D
from diffusion_process import VPDiffusion

l
CONFIG = {
    "data_root": ".",
    "extracted_dir": "extracted_data",
    "output_dir": "runs/diffusion_dps",
    "auto_extract": True,
    "extractor": "",
    "epochs": 120,
    "batch_size": 16,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "window_size": 512,
    "num_sensors": 32,
    "windows_per_record": 4,
    "timesteps": 200,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "base_channels": 96,
    "num_blocks": 6,
    "ema_decay": 0.999,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VP diffusion prior for radar inverse problems.")
    for key, value in CONFIG.items():
        arg_type = type(value)
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=value)
        elif value is None:
            parser.add_argument(f"--{key}", default=value)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value)
    args, _ = parser.parse_known_args()
    return args


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
    label_names: list[str],
    in_channels: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "label_names": label_names,
        "in_channels": in_channels,
        "model_kwargs": {
            "in_channels": in_channels,
            "base_channels": args.base_channels,
            "num_blocks": args.num_blocks,
        },
        "diffusion_kwargs": {
            "timesteps": args.timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
        },
        "data_kwargs": {
            "window_size": args.window_size,
            "num_sensors": args.num_sensors,
            "windows_per_record": args.windows_per_record,
            "seed": args.seed,
        },
        "config": vars(args),
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root).resolve()
    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.is_absolute():
        extracted_dir = data_root / extracted_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = data_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records, label_names = build_records(
        data_root,
        extracted_dir,
        auto_extract=args.auto_extract,
        custom_extractor=args.extractor or None,
    )
    train_records, val_records, test_records = split_records(records, seed=args.seed)
    save_label_map(output_dir / "label_map.json", label_names)
    write_records_csv(output_dir / "train_records.csv", train_records)
    write_records_csv(output_dir / "val_records.csv", val_records)
    write_records_csv(output_dir / "test_records.csv", test_records)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_dataset = RadarWindowDataset(
        train_records,
        window_size=args.window_size,
        num_sensors=args.num_sensors,
        windows_per_record=args.windows_per_record,
        random_crop=True,
        seed=args.seed,
    )
    val_dataset = RadarWindowDataset(
        val_records,
        window_size=args.window_size,
        num_sensors=args.num_sensors,
        windows_per_record=1,
        random_crop=False,
        seed=args.seed,
    )

    sample = train_dataset[0]["x"]
    in_channels = int(sample.shape[0])
    device = torch.device(args.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = EpsilonDenoiser1D(
        in_channels=in_channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    ema_model = EpsilonDenoiser1D(
        in_channels=in_channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()

    diffusion = VPDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")

    print("")
    print("Training diffusion prior")
    print(f"  device: {device}")
    print(f"  train windows: {len(train_dataset)}")
    print(f"  val windows: {len(val_dataset)}")
    print(f"  input shape: {tuple(sample.shape)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=True)
        for batch in progress:
            x0 = batch["x"].to(device, non_blocking=True)
            loss = diffusion.training_loss(model, x0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema_model, model, args.ema_decay)

            running_loss += float(loss.item()) * x0.shape[0]
            progress.set_postfix(loss=f"{loss.item():.5f}")

        train_loss = running_loss / max(1, len(train_dataset))

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="validation", leave=False):
                x0 = batch["x"].to(device, non_blocking=True)
                val_loss = diffusion.training_loss(ema_model, x0)
                val_loss_sum += float(val_loss.item()) * x0.shape[0]
        val_loss = val_loss_sum / max(1, len(val_dataset))

        print(f"epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        save_checkpoint(
            output_dir / "latest.pt",
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            args=args,
            label_names=label_names,
            in_channels=in_channels,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                args=args,
                label_names=label_names,
                in_channels=in_channels,
            )
            print(f"  saved best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
