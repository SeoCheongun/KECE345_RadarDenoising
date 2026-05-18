# Radar DPS 기반 Micro-Doppler 동작 분류 프로젝트

이 프로젝트는 레이더 IQ 신호에서 사람 동작별 micro-Doppler 특징을 만들고, 잡음 환경에서 Diffusion Posterior Sampling(DPS) 기반 복원 결과가 분류 성능에 어떤 영향을 주는지 실험하기 위한 코드입니다.

주요 흐름은 다음과 같습니다.

1. 원본 `.rar` 데이터 압축 해제 및 `.mat` 파일 인덱싱
2. VP diffusion prior 학습
3. 잡음이 섞인 신호 또는 DPS로 복원한 신호에서 micro-Doppler spectrogram 생성
4. CNN 기반 동작 분류기 학습
5. MSE, SNR별 결과, confusion matrix, 샘플 시각화, GIF 생성

## 프로젝트 구조

```text
.
├── raw_data/                         # 원본 RAR 데이터 보관용 폴더
├── extracted_data/                   # 압축 해제된 클래스별 .mat 파일
├── python_dataset/                   # prepare_dataset.py가 만든 numpy 데이터셋
├── runs/                             # 학습 결과, 체크포인트, 캐시, 그림 저장 위치
├── data_utils.py                     # 데이터 로딩, split, AWGN, radar window dataset
├── diffusion_model.py                # 1D epsilon denoiser 모델
├── diffusion_process.py              # VP diffusion 및 DPS sampling
├── classifier_model.py               # 2D spectrogram 분류기
├── paper_micro_doppler.py            # micro-Doppler spectrogram 생성/시각화
├── train_dps.py                      # diffusion prior 학습
├── eval.py                           # DPS denoise/inpaint 평가
├── train_classifier_noisy.py         # noisy spectrogram 분류기 학습
├── train_classifier_dps.py           # DPS spectrogram 분류기 학습
├── compare_confusion_matrices.py     # noisy vs DPS confusion matrix 비교
├── check_one_sample_random_visualize.py
└── make_gif.py                       # DPS 복원 과정 GIF 생성
```

## 데이터

클래스 라벨은 원본 RAR 파일 이름 또는 `extracted_data/<label>/` 폴더 이름을 기준으로 정해집니다. 현재 사용되는 라벨 맵은 다음과 같습니다.

```json
{
  "B": 0,
  "FTW": 1,
  "G": 2,
  "K": 3,
  "P": 4,
  "SD": 5,
  "STW": 6,
  "SU": 7,
  "W": 8,
  "WTF": 9,
  "WTS": 10
}
```

라벨 약어의 의미는 공개 FMCW radar dataset 설명의 행동 정의를 기준으로 정리했습니다.

| 라벨 | 영문 의미 | 한국어 의미 |
| --- | --- | --- |
| `B` | Standing in a fixed position while rotating body | 제자리에서 몸통 회전 |
| `K` | Kicking | 발차기 |
| `P` | Punching | 주먹질 |
| `G` | Grabbing an object | 물체 잡기 |
| `W` | Walking back and forth in front of the radar | 레이더 앞에서 앞뒤로 걷기 |
| `SU` | Standing up from chair | 의자에서 일어서기 |
| `SD` | Sitting down on chair | 의자에 앉기 |
| `STW` | Stands up from chair to walk | 의자에서 일어나 걷기 |
| `WTS` | Walks to sit on chair | 걸어와서 의자에 앉기 |
| `WTF` | Walks to fall on the ground | 걷다가 바닥에 넘어지기 |
| `FTW` | Standing up from ground to walk | 바닥에서 일어나 걷기 |

압축 파일은 프로젝트 루트 또는 `raw_data/`에 보관할 수 있습니다. 대부분의 학습/평가 스크립트는 `extracted_data/`를 기본 입력으로 사용하며, `--auto_extract`가 켜져 있으면 사용 가능한 압축 해제 도구를 찾아 자동 압축 해제를 시도합니다.

Windows 환경에서는 7-Zip 또는 WinRAR 설치를 권장합니다.

## 환경 설정

Python 3.10 이상을 권장합니다. GPU가 있으면 PyTorch CUDA 버전을 설치하세요.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy scikit-learn matplotlib tqdm torch
```

CUDA 버전 PyTorch가 필요한 경우에는 본인 CUDA 환경에 맞는 설치 명령을 PyTorch 공식 안내에 따라 사용하세요.

## 빠른 실행 순서

### 1. 데이터셋 준비

기본 feature vector 기반 numpy 데이터셋을 만들려면 다음을 실행합니다.

```bash
python prepare_dataset.py
```

생성 파일:

```text
python_dataset/dataset.npz
python_dataset/X_train.npy
python_dataset/y_train.npy
python_dataset/X_val.npy
python_dataset/y_val.npy
python_dataset/X_test.npy
python_dataset/y_test.npy
python_dataset/label_map.json
python_dataset/metadata.csv
```

### 2. Diffusion prior 학습

```bash
python train_dps.py --epochs 120 --batch_size 16 --device cuda
```

CPU만 사용할 경우:

```bash
python train_dps.py --device cpu
```

주요 출력:

```text
runs/diffusion_dps/best.pt
runs/diffusion_dps/latest.pt
runs/diffusion_dps/config.json
runs/diffusion_dps/label_map.json
runs/diffusion_dps/train_records.csv
runs/diffusion_dps/val_records.csv
runs/diffusion_dps/test_records.csv
```

### 3. DPS 복원 성능 평가

```bash
python eval.py --checkpoint runs/diffusion_dps/best.pt --split test --snr_levels "-15,-10,-5,0,5,10"
```

출력:

```text
runs/diffusion_dps/eval/metrics.json
runs/diffusion_dps/eval/metrics.csv
```

`--inverse_task denoise`는 잡음 제거, `--inverse_task inpaint`는 일부 관측값만 있는 복원 문제를 평가합니다.

### 4. Noisy baseline 분류기 학습

DPS를 사용하지 않고 AWGN이 추가된 spectrogram으로 분류기를 학습합니다.

```bash
python train_classifier_noisy.py --epochs 200 --snr_levels "-15,-10,-5,0,5,10" --device cuda
```

출력:

```text
runs/classifier_noisy/best.pt
runs/classifier_noisy/latest.pt
runs/classifier_noisy/history.csv
runs/classifier_noisy/config.json
runs/classifier_noisy/label_map.json
```

### 5. DPS 기반 분류기 학습

Diffusion checkpoint를 사용해 noisy 신호를 DPS로 복원한 뒤 spectrogram을 만들고, 이를 이용해 분류기를 학습합니다.

```bash
python train_classifier_dps.py --diffusion_checkpoint runs/diffusion_dps/best.pt --epochs 200 --device cuda
```

첫 실행 시 `runs/dps_spectrogram_cache/`에 DPS spectrogram cache를 생성합니다. 캐시가 이미 있으면 복원 단계를 건너뛰고 바로 분류기를 학습합니다.

출력:

```text
runs/classifier_dps/best.pt
runs/classifier_dps/latest.pt
runs/classifier_dps/history.csv
runs/classifier_dps/config.json
runs/classifier_dps/label_map.json
runs/dps_spectrogram_cache/
```

### 6. Confusion matrix 비교

```bash
python compare_confusion_matrices.py --split val --snr_levels "-15,-10,-5,0,5,10" --device cuda
```

출력:

```text
runs/confusion_matrices/confusion_compare_val.png
```

`test` split의 DPS cache가 없을 경우 다음 옵션으로 필요한 cache를 함께 만들 수 있습니다.

```bash
python compare_confusion_matrices.py --split test --precompute_missing_dps --device cuda
```

### 7. 샘플 복원 시각화

```bash
python check_one_sample_random_visualize.py --snr_db -15 --sample_index -1 --device cuda
```

출력:

```text
runs/diffusion_dps/figures/
```

### 8. DPS 복원 과정 GIF 생성

```bash
python make_gif.py --snr_db -40 --frame_stride 5 --fps 6 --device cuda
```

출력:

```text
runs/diffusion_dps/gifs/
```

## 주요 옵션

자주 쓰는 옵션은 대부분의 스크립트에서 공통으로 지원됩니다.

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `--data_root` | 프로젝트/데이터 루트 | `.` |
| `--extracted_dir` | 압축 해제된 `.mat` 폴더 | `extracted_data` |
| `--output_dir` | 결과 저장 폴더 | 스크립트별 상이 |
| `--auto_extract` / `--no-auto_extract` | RAR 자동 압축 해제 여부 | `True` |
| `--extractor` | 7z/WinRAR/unrar 실행 파일 직접 지정 | 빈 문자열 |
| `--device` | `cuda` 또는 `cpu` | CUDA 가능 시 `cuda` |
| `--seed` | 난수 seed | `42` |
| `--snr_levels` | SNR 목록 | `-15,-10,-5,0,5,10` |
| `--dps_scale` | DPS measurement guidance 강도 | 스크립트별 상이 |
| `--start_from_measurement` | DPS 초기값을 측정값으로 시작 | `False` |

## 실험 결과 위치

현재 저장된 주요 결과는 다음 경로에서 확인할 수 있습니다.

```text
runs/diffusion_dps/eval/metrics.json
runs/classifier_noisy/history.csv
runs/classifier_dps/history.csv
runs/confusion_matrices/confusion_compare_val.png
runs/diffusion_dps/figures/
runs/diffusion_dps/gifs/
```

예를 들어 `runs/diffusion_dps/eval/metrics.json`에는 SNR별 noisy MSE와 DPS MSE가 저장됩니다.

## 참고 및 주의사항

- 학습과 DPS precompute는 GPU 사용을 강하게 권장합니다.
- DPS spectrogram cache 생성은 시간이 오래 걸릴 수 있습니다. 중간 결과는 `runs/dps_spectrogram_cache/` 또는 `runs/dps_cache/`에 저장됩니다.
- Windows PowerShell에서 프로필 실행 정책 경고가 보일 수 있지만, 명령 실행 자체가 성공하면 실험 결과에는 영향을 주지 않습니다.
- `KMP_DUPLICATE_LIB_OK=TRUE`는 일부 Windows/PyTorch/Matplotlib 조합에서 OpenMP 충돌을 피하기 위해 코드에서 기본 설정됩니다.
- `prepare_dataset.py`는 변환 후 `extracted_data/` 아래의 `.mat` 파일을 삭제하도록 설정되어 있습니다. 원본 `.rar` 파일은 삭제하지 않습니다.
