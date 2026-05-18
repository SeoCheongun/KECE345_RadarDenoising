#%%
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_DIR = Path(__file__).resolve().parent
EXTRACTED_DIR = PROJECT_DIR / "extracted_data"
OUT_DIR = PROJECT_DIR / "python_dataset"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# If auto-detection fails, set this manually.
CUSTOM_EXTRACTOR = ""

# After python_dataset/*.npy and dataset.npz are saved, remove extracted .mat
# files under extracted_data/. Original .rar files are never deleted.
DELETE_EXTRACTED_MAT_AFTER_CONVERSION = True


def main() -> None:
    archives = sorted(PROJECT_DIR.glob("*.rar"))
    if not archives:
        raise FileNotFoundError(f"No .rar files found in {PROJECT_DIR}")

    EXTRACTED_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    extractor = find_extractor()
    extract_archives(archives, extractor)

    rows = collect_mat_files(archives)
    if not rows:
        raise RuntimeError(
            "No .mat files found. Check extraction or manually extract RAR files into extracted_data/<label>/."
        )

    label_names = sorted({row["label"] for row in rows})
    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    metadata: list[dict[str, str | int]] = []

    print(f"Found {len(rows)} .mat files across {len(label_names)} classes.")
    for idx, row in enumerate(rows, start=1):
        mat_path = Path(row["path"])
        label = str(row["label"])
        feature = mat_to_feature_vector(mat_path)

        X_list.append(feature.astype(np.float32))
        y_list.append(label_to_id[label])
        metadata.append(
            {
                "index": idx - 1,
                "file": str(mat_path),
                "label": label,
                "label_id": label_to_id[label],
            }
        )

        if idx % 25 == 0 or idx == len(rows):
            print(f"Converted {idx}/{len(rows)} files")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    train_idx, val_idx, test_idx = stratified_split(y)

    save_array("X.npy", X)
    save_array("y.npy", y)
    save_array("X_train.npy", X[train_idx])
    save_array("y_train.npy", y[train_idx])
    save_array("X_val.npy", X[val_idx])
    save_array("y_val.npy", y[val_idx])
    save_array("X_test.npy", X[test_idx])
    save_array("y_test.npy", y[test_idx])

    np.savez_compressed(
        OUT_DIR / "dataset.npz",
        X=X,
        y=y,
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        label_names=np.asarray(label_names),
    )

    write_json(OUT_DIR / "label_map.json", label_to_id)
    write_metadata(OUT_DIR / "metadata.csv", metadata)

    if DELETE_EXTRACTED_MAT_AFTER_CONVERSION:
        deleted_count = delete_extracted_mat_files()
        print(f"Deleted extracted .mat files: {deleted_count}")

    print("")
    print("Python dataset created:")
    print(f"  {OUT_DIR / 'dataset.npz'}")
    print(f"  X shape: {X.shape}")
    print(f"  train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"  labels: {label_to_id}")


def find_extractor() -> Path | None:
    candidates = [
        CUSTOM_EXTRACTOR,
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


def extract_archives(archives: list[Path], extractor: Path | None) -> None:
    if extractor is None:
        print("No extractor found. Using already-extracted files only.")
        return

    print(f"Using extractor: {extractor}")
    for archive in archives:
        label = archive.stem
        target_dir = EXTRACTED_DIR / label
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


def collect_mat_files(archives: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for archive in archives:
        label = archive.stem
        class_dir = EXTRACTED_DIR / label
        if not class_dir.exists():
            continue

        for mat_path in sorted(class_dir.rglob("*.mat")):
            rows.append({"path": str(mat_path), "label": label})
    return rows


def mat_to_feature_vector(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    arrays = [
        value
        for key, value in data.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
    ]
    if not arrays:
        raise ValueError(f"No numeric arrays in {mat_path}")

    features = [array_to_features(arr) for arr in arrays]
    return np.concatenate(features)


def array_to_features(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    mag = np.abs(x).astype(np.float64)
    real = np.real(x).astype(np.float64)
    imag = np.imag(x).astype(np.float64)

    feature_blocks = [
        mag.mean(axis=1),
        mag.std(axis=1),
        mag.max(axis=1),
        np.sqrt(np.mean(mag**2, axis=1)),
        real.mean(axis=1),
        real.std(axis=1),
        imag.mean(axis=1),
        imag.std(axis=1),
    ]

    return np.concatenate(feature_blocks)


def stratified_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    train_parts = []
    val_parts = []
    test_parts = []

    for label_id in np.unique(y):
        idx = np.flatnonzero(y == label_id)
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))

        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def save_array(name: str, array: np.ndarray) -> None:
    path = OUT_DIR / name
    np.save(path, array)
    print(f"Saved {path.name}: {array.shape}")


def write_json(path: Path, data: dict[str, int]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_metadata(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "file", "label", "label_id"])
        writer.writeheader()
        writer.writerows(rows)


def delete_extracted_mat_files() -> int:
    deleted = 0
    root = EXTRACTED_DIR.resolve()

    for mat_path in sorted(EXTRACTED_DIR.rglob("*.mat")):
        resolved = mat_path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"Refusing to delete outside extracted_data: {resolved}")
        resolved.unlink()
        deleted += 1

    for directory in sorted(EXTRACTED_DIR.rglob("*"), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass

    return deleted


if __name__ == "__main__":
    main()
