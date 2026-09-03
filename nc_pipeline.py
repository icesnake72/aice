"""GK2A AMI SW038 다음 프레임 예측 — 모델 공통 파이프라인.

데이터 적재/분할/정규화/학습/평가/기록을 모두 담는다. 모델 파일(`nc_predict_colab.py`,
`simvp_predict_colab.py`, `predrnn_v2_predict_colab.py`)은 `build_model()` 과 `MODEL_NAME` 만
정의하고 이 모듈의 `run()` / `main_for_model()` 을 그대로 쓴다. 모델 간 비교가 목적이므로
데이터 분할·손실·Persistence 베이스라인·평가 지표는 여기서만 정의한다.

파이프라인
  1. Drive 마운트 -> .nc 적재 -> 연속 구간(segment) 분리 -> 500->250 다운샘플 -> 캐시(npz)
  2. 정규화 -> Persistence 베이스라인
  3. 패치 슬라이딩 윈도우 데이터셋 -> 잔차 + SSIM 손실 학습
  4. 검증셋 평가 -> 250x250 full-frame 예측 -> 그림/가중치/metrics.json 을 저장

결과 레이아웃
  <out_dir>/cache/frames_sw038_t250.npz     모델 간 공유 캐시
  <out_dir>/<MODEL_NAME>/metrics.json       스키마 version 1 (SCHEMA_VERSION)
  <out_dir>/<MODEL_NAME>/samples.png hourly_mean.png history.png full_frame_prediction.png
  <out_dir>/<MODEL_NAME>/train_log.csv, *.weights.h5, pred_next.npy

Colab 사용법
  1) 로컬에서 데이터 다운로드 (AWS Open Data, 익명 접근)
       python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out netcdf
  2) 결과 폴더(또는 zip)를 Google Drive 에 업로드
       COLAB_DATA_DIR = MyDrive/netcdf 아래에 gk2a_ami_le1b_sw038_la020ge_202510170000.nc ...
       (zip 이면 MyDrive/netcdf.zip — COLAB_UNZIP_DIR 로 풀어서 쓴다)
  3) Colab 메뉴: 런타임 > 런타임 유형 변경 > T4 GPU
  4) 모델 스크립트를 Colab 에 올린 뒤 셀에서 실행
       %run nc_predict_colab.py                                   # 그림이 셀에 바로 표시된다
       %run nc_predict_colab.py --epochs 2 --hours 6 7 8 9 10 11  # 빠른 확인
     결과는 COLAB_OUT_DIR (MyDrive/nc_predict_output) 아래 모델 이름 폴더에 저장된다.

주의
  - Keras 3 의 ConvLSTM 은 입력 높이/너비에 None 을 허용하지 않는다.
    학습 모델은 PATCH 크기로 고정하고, 추론 시 250x250 모델을 새로 만들어 가중치를 옮긴다.
  - TARGET 은 원본 500 의 정수배 약수여야 한다 (300 이면 평균풀링이 아니라 좌상단 크롭이 된다).
  - 하루 720장 중 710장만 존재한다(위성 정기 점검 결측). find_segments 로 연속 구간을 나눠
    그 안에서만 윈도우를 만들어 '12분 점프'를 '2분 변화'로 학습하는 오류를 막는다.
  - T4 는 compute capability 7.5 라 mixed_float16 이 실제로 빨라진다. 끄려면 --no-mixed-precision.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("nc_pipeline")

# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------
COLAB_DRIVE_ROOT = Path("/content/drive")
COLAB_DATA_DIR = COLAB_DRIVE_ROOT / "MyDrive/netcdf"
COLAB_OUT_DIR = COLAB_DRIVE_ROOT / "MyDrive/nc_predict_output"
COLAB_UNZIP_DIR = Path("/content/gk2a_netcdf")
LOCAL_DATA_DIR = Path("resource/netcdf")
LOCAL_OUT_DIR = Path("results")

NC_GLOB = "gk2a*sw038*.nc"
STAMP_FMT = "%Y%m%d%H%M"
STAMP_RE = re.compile(r"(\d{12})\.nc$")   # 파일명 끝 12자리 = 관측 시각 YYYYMMDDHHMM
STEP_MINUTES = 2                          # GK2A LA 관측 주기
DAYTIME_END_HOUR = 6                      # 00~06 UTC = 09~15 KST 를 '주간'으로 본다
SSIM_CHUNK = 64                           # ssim 계산 시 GPU 로 한 번에 보내는 프레임 수
NANUM_FONT_PATH = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
KOREAN_FONT_CANDIDATES = ("NanumGothic", "AppleGothic", "Malgun Gothic")

SCHEMA_VERSION = 1
METRICS_NAME = "metrics.json"
TRAIN_LOG_NAME = "train_log.csv"
PRED_NEXT_NAME = "pred_next.npy"
# 모델 디렉터리가 이미 모델을 구분하지만 파일명만 봐도 알 수 있도록 이름을 넣는다.
WEIGHTS_NAME = "{model}.weights.h5"
CHECKPOINT_NAME = "checkpoint.weights.h5"
FIGURE_NAMES = {
  "samples": "samples.png",
  "hourly_mean": "hourly_mean.png",
  "history": "history.png",
  "full_frame": "full_frame_prediction.png",
}
HISTORY_KEYS = ("loss", "val_loss", "mae", "val_mae")   # lr 등 나머지는 기록하지 않는다


@dataclass
class Config:
  """실행 설정. CLI 인자(build_arg_parser/config_from_args)로 덮어쓴다."""

  data_dir: Path
  out_dir: Path
  data_zip: Path | None = None
  var: str = "image_pixel_values"
  in_frames: int = 4        # 입력 시퀀스 길이 (과거 N장)
  target: int | None = 250  # 다운샘플 해상도. None 이면 원본 500 유지
  patch: int = 96           # 패치 한 변 크기
  stride: int = 77          # 250 = 154 + 96 -> stride 77 이면 3x3 격자로 전 영역을 덮는다
  filters: int = 16         # 모델의 기본 hidden 폭 (의미는 모델마다 다르다)
  epochs: int = 4
  batch: int = 16
  lr: float = 1e-3
  hours: list[int] | None = None   # 사용할 UTC 시각. None 이면 24시간 전부
  seed: int = 42
  use_cache: bool = True
  mixed_precision: bool = True
  val_ratio: float = 0.8
  early_stop_patience: int = 8
  lr_patience: int = 4
  extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 런타임 준비 (Colab 감지 · Drive · 패키지 · 폰트 · GPU)
# --------------------------------------------------------------------------
def is_colab() -> bool:
  """google.colab 모듈이 import 가능하면 Colab 런타임으로 본다."""
  return importlib.util.find_spec("google.colab") is not None


def mount_drive(drive_root: Path = COLAB_DRIVE_ROOT) -> bool:
  """Colab 에서 Google Drive 를 마운트한다. 이미 마운트돼 있으면 건너뛴다.

  Returns:
    마운트 상태 (True=사용 가능)
  """
  if not is_colab():
    return False
  if (drive_root / "MyDrive").exists():
    logger.info("Drive 이미 마운트됨: %s", drive_root)
    return True
  try:
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(str(drive_root))
    logger.info("Drive 마운트 완료: %s", drive_root)
    return True
  except Exception as exc:  # noqa: BLE001 - 인증 취소 등 종류가 다양하다
    logger.error("Drive 마운트 실패: %s", exc)
    return False


def ensure_netcdf_backend() -> None:
  """xarray 가 NetCDF4(HDF5) 파일을 열 수 있는 엔진(netCDF4 또는 h5netcdf)을 보장한다.

  Colab 에 없으면 pip 로 netCDF4 를 설치한다. 로컬에서는 설치하지 않고 안내만 한다.
  """
  if any(importlib.util.find_spec(m) for m in ("netCDF4", "h5netcdf")):
    return
  if not is_colab():
    raise RuntimeError("netCDF4 또는 h5netcdf 가 필요하다: pip install netCDF4")
  logger.info("netCDF4 설치 중 ...")
  subprocess.run([sys.executable, "-m", "pip", "install", "-q", "netCDF4"], check=True)


def setup_korean_font() -> str:
  """matplotlib 한글 폰트를 설정한다.

  Colab 에는 한글 폰트가 없으므로 fonts-nanum 을 apt 로 설치한 뒤 등록한다.
  macOS 는 AppleGothic, Windows 는 Malgun Gothic 을 쓴다.

  Returns:
    적용된 font family 이름
  """
  import matplotlib.font_manager as fm
  import matplotlib.pyplot as plt

  if is_colab() and not NANUM_FONT_PATH.exists():
    logger.info("NanumGothic 설치 중 (apt-get) ...")
    subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-nanum"],
                   check=False, capture_output=True)
  if NANUM_FONT_PATH.exists():
    fm.fontManager.addfont(str(NANUM_FONT_PATH))   # 캐시 재생성 없이 즉시 등록

  available = {f.name for f in fm.fontManager.ttflist}
  family = next((c for c in KOREAN_FONT_CANDIDATES if c in available), None)
  if family is None:
    logger.warning("한글 폰트를 찾지 못했다. 그래프의 한글이 깨질 수 있다.")
    family = plt.rcParams["font.family"][0]
  plt.rcParams["font.family"] = family
  plt.rcParams["axes.unicode_minus"] = False
  return family


def setup_gpu(mixed_precision: bool) -> bool:
  """GPU 존재 여부를 확인하고 memory growth 와 mixed precision 정책을 설정한다.

  Args:
    mixed_precision: True 면 mixed_float16 (T4 이상에서 유효)
  Returns:
    GPU 사용 가능 여부
  """
  import tensorflow as tf
  from tensorflow import keras

  gpus = tf.config.list_physical_devices("GPU")
  if not gpus:
    logger.warning("GPU 없음 (CPU 실행). Colab 이면 런타임 > 런타임 유형 변경 > T4 GPU 로 바꿔라.")
  for gpu in gpus:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:   # 이미 초기화된 뒤에는 바꿀 수 없다
      logger.debug("memory growth 설정 생략: %s", exc)
  if gpus:
    details = tf.config.experimental.get_device_details(gpus[0])
    logger.info("GPU: %s (compute capability %s)",
                details.get("device_name", "?"), details.get("compute_capability", "?"))

  policy = "mixed_float16" if (mixed_precision and gpus) else "float32"
  keras.mixed_precision.set_global_policy(policy)
  logger.info("tf %s / precision policy: %s", tf.__version__, policy)
  return bool(gpus)


# --------------------------------------------------------------------------
# 데이터 적재
# --------------------------------------------------------------------------
def extract_zip(zip_path: Path, dest: Path) -> Path:
  """zip 안의 .nc 파일만 dest 에 평탄하게 푼다 (경로 순회 방지를 위해 basename 만 사용).

  Args:
    zip_path: Drive 위의 zip 경로
    dest: 로컬 디스크 추출 위치
  Returns:
    dest
  """
  if not zip_path.is_file():
    raise FileNotFoundError(f"zip 없음: {zip_path}")
  dest.mkdir(parents=True, exist_ok=True)
  count = 0
  with zipfile.ZipFile(zip_path) as zf:
    for member in zf.infolist():
      name = Path(member.filename).name
      if member.is_dir() or not name.endswith(".nc"):
        continue
      target = dest / name
      if target.exists() and target.stat().st_size == member.file_size:
        continue
      with zf.open(member) as src, open(target, "wb") as dst:
        dst.write(src.read())
      count += 1
  logger.info("zip 추출: 신규 %d개 -> %s", count, dest)
  return dest


def list_nc_files(data_dir: Path) -> list[str]:
  """디렉터리의 sw038 파일을 관측 시각 오름차순으로 반환.

  12자리 시각 문자열은 0 패딩이라 '사전순 == 시간순' 이 성립한다.
  형식이 안 맞는 파일은 제외한다.
  """
  files = glob.glob(os.path.join(str(data_dir), NC_GLOB))
  keyed = [(m.group(1), f) for f in files if (m := STAMP_RE.search(f))]
  return [f for _, f in sorted(keyed)]


def parse_stamp(path: str) -> datetime:
  """파일 경로 -> 관측 시각 datetime."""
  match = STAMP_RE.search(path)
  if not match:
    raise ValueError(f"관측 시각을 파싱할 수 없는 파일명: {path}")
  return datetime.strptime(match.group(1), STAMP_FMT)


def read_frames(files: list[str], var: str, log_every: int = 100) -> np.ndarray:
  """.nc 파일을 하나씩 열어 (T, H, W) float32 배열로 쌓는다.

  Drive 마운트는 파일 하나하나가 느리므로 진행 상황을 로그로 남긴다.
  """
  import xarray as xr

  arrays: list[np.ndarray] = []
  t0 = time.time()
  for i, f in enumerate(files, start=1):
    with xr.open_dataset(f) as ds:   # 파일 핸들 누수 방지
      arrays.append(ds[var].values.astype(np.float32))
    if i % log_every == 0 or i == len(files):
      logger.info("  적재 %d/%d (%.0fs)", i, len(files), time.time() - t0)
  return np.stack(arrays, axis=0)


def find_segments(ts: list[datetime], step_min: int = STEP_MINUTES) -> list[tuple[int, int]]:
  """[(start, end), ...] 형태의 연속 구간 목록. end 는 exclusive.

  간격이 step_min 이 아닌 지점(관측 결측)에서 구간을 끊는다.
  """
  segments: list[tuple[int, int]] = []
  start = 0
  for i in range(1, len(ts)):
    if ts[i] - ts[i - 1] != timedelta(minutes=step_min):
      segments.append((start, i))
      start = i
  segments.append((start, len(ts)))
  return segments


def downsample(arr: np.ndarray, target: int) -> np.ndarray:
  """(T, H, W) 를 (T, target, target) 으로 블록 평균 풀링한다.

  H % target != 0 이면 reshape 가 평균풀링이 아니라 좌상단 크롭으로 변질되므로 막는다.
  """
  T, H, W = arr.shape
  if H % target or W % target:
    raise ValueError(
      f"TARGET={target} 는 원본 {H} 의 정수배 약수가 아니다. "
      f"그대로 두면 평균풀링이 아니라 좌상단 {target}x{target} 크롭이 된다.")
  fy, fx = H // target, W // target
  # (T, target, fy, target, fx) 로 축을 쪼개면 [t, i, :, j, :] 가 (i, j) 번째 블록이다.
  return arr.reshape(T, target, fy, target, fx).mean(axis=(2, 4)).astype(np.float32)


def filter_hours(frames: np.ndarray, stamps: list[datetime],
                 hours: list[int] | None) -> tuple[np.ndarray, list[datetime]]:
  """특정 UTC 시각대만 남긴다. frames 와 stamps 를 같은 인덱스로 함께 거른다."""
  if hours is None:
    return frames, stamps
  allowed = set(hours)
  keep = [i for i, s in enumerate(stamps) if s.hour in allowed]
  if not keep:
    raise ValueError(f"HOURS={sorted(allowed)} 에 해당하는 프레임이 없다.")
  return frames[keep], [stamps[i] for i in keep]


def cache_path_for(cfg: Config) -> Path:
  """다운샘플 결과 캐시(npz) 경로. TARGET 별로 따로 둔다 (HOURS 는 캐시 후에 거른다).

  모델 디렉터리가 아니라 out_dir 바로 아래에 두어 세 모델이 같은 캐시를 공유한다.
  """
  return cfg.out_dir / "cache" / f"frames_sw038_t{cfg.target or 'full'}.npz"


def load_frames(cfg: Config) -> tuple[np.ndarray, list[datetime]]:
  """데이터 적재의 진입점. 캐시 -> zip -> .nc 순으로 시도한다.

  Returns:
    (frames (T, H, W) float32 다운샘플 완료·정규화 전, stamps)
  """
  cache = cache_path_for(cfg)
  data_dir = cfg.data_dir
  if cfg.data_zip is not None and not list_nc_files(data_dir):
    data_dir = extract_zip(cfg.data_zip, COLAB_UNZIP_DIR if is_colab() else cfg.out_dir / "netcdf")
  files = list_nc_files(data_dir)

  if cfg.use_cache and cache.is_file():
    with np.load(cache) as z:
      frames = z["frames"]
      stamps = [datetime.strptime(s, STAMP_FMT) for s in z["stamps"]]
    if files and len(files) != len(frames):
      logger.warning("캐시(%d장)와 .nc 개수(%d장)가 다르다 -> 캐시를 다시 만든다", len(frames), len(files))
    else:
      logger.info("캐시 적재: %s %s", cache, frames.shape)
      return frames, stamps

  if not files:
    raise FileNotFoundError(
      f"{data_dir} 에 {NC_GLOB} 가 없다. gk2a_download.py 로 받은 뒤 Drive 에 올려라.")
  stamps = [parse_stamp(f) for f in files]
  logger.info("파일 수: %d  기간: %s ~ %s UTC", len(files),
              f"{stamps[0]:%Y-%m-%d %H:%M}", f"{stamps[-1]:%H:%M}")
  frames = read_frames(files, cfg.var)
  logger.info("원본 frames: %s %s", frames.shape, frames.dtype)

  if cfg.target and cfg.target < frames.shape[1]:
    frames = downsample(frames, cfg.target)
    logger.info("다운샘플 후: %s (%.0f MB)", frames.shape, frames.nbytes / 1e6)

  if cfg.use_cache:
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, frames=frames, stamps=np.array([s.strftime(STAMP_FMT) for s in stamps]))
    logger.info("캐시 저장: %s", cache)
  return frames, stamps


# --------------------------------------------------------------------------
# 정규화 · 시각화 · 베이스라인
# --------------------------------------------------------------------------
def normalize(frames: np.ndarray) -> tuple[np.ndarray, float, float]:
  """전체 프레임 하나의 min/max 로 0~1 정규화한다.

  프레임별 정규화는 '이 시각이 저 시각보다 밝다'는 정보를 지우므로 전역 기준을 쓴다.
  Returns:
    (정규화 배열, gmin, gmax)
  """
  gmin, gmax = float(frames.min()), float(frames.max())
  if gmax <= gmin:
    raise ValueError(f"정규화 불가: min={gmin}, max={gmax}")
  return (frames - gmin) / (gmax - gmin), gmin, gmax


def denormalize(x: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
  """정규화 값을 원래 관측 계수로 되돌린다 (물리량 해석용 짝함수)."""
  return x * (gmax - gmin) + gmin


def save_and_show(fig, out_dir: Path, filename: str) -> Path:
  """그림을 out_dir/filename 에 저장하고 화면에도 표시한다 (%run 이면 셀에 인라인 표시).

  !python 처럼 headless(Agg) 로 돌 때는 show 가 의미 없고 경고만 나므로 저장만 한다.
  """
  import matplotlib
  import matplotlib.pyplot as plt

  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / filename
  fig.savefig(path, dpi=110, bbox_inches="tight")
  if matplotlib.get_backend().lower() != "agg":
    plt.show()
  plt.close(fig)
  return path


def plot_samples(frames_n: np.ndarray, stamps: list[datetime], model_dir: Path, n: int = 6) -> None:
  """하루를 균등 간격 n 장으로 뽑아 그린다. vmin/vmax 를 고정해야 시간대별 밝기 차이가 보인다."""
  import matplotlib.pyplot as plt

  idxs = np.linspace(0, len(frames_n) - 1, n).astype(int)
  fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
  for ax, i in zip(axes, idxs):
    ax.imshow(frames_n[i], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{stamps[i]:%H:%M}Z / {(stamps[i].hour + 9) % 24:02d}KST")
    ax.axis("off")
  fig.tight_layout()
  save_and_show(fig, model_dir, FIGURE_NAMES["samples"])


def plot_hourly_mean(frames_n: np.ndarray, stamps: list[datetime], model_dir: Path) -> None:
  """시간대별 평균으로 주야 전환(태양 반사)을 드러낸다."""
  import matplotlib.pyplot as plt

  hours = sorted({s.hour for s in stamps})
  hourly = [frames_n[[i for i, s in enumerate(stamps) if s.hour == h]].mean() for h in hours]
  fig = plt.figure(figsize=(9, 3))
  plt.plot(hours, hourly, marker="o")
  plt.xlabel("UTC hour")
  plt.ylabel("정규화 평균값")
  plt.grid(alpha=0.3)
  plt.title("시간대별 평균 — 주간(00~06Z)에 분포가 내려앉는다 (태양반사)")
  fig.tight_layout()
  save_and_show(fig, model_dir, FIGURE_NAMES["hourly_mean"])


def ssim_metric(a: np.ndarray, b: np.ndarray, chunk: int = SSIM_CHUNK) -> float:
  """두 이미지 묶음 (N, H, W) 의 평균 SSIM. 1 에 가까울수록 닮았다.

  tf.image.ssim 은 (N, H, W, C) 를 요구하므로 채널 축을 붙이고, chunk 단위로 나눠 계산한다.
  """
  import tensorflow as tf

  a4 = a[..., None].astype(np.float32)
  b4 = b[..., None].astype(np.float32)
  vals = [tf.image.ssim(a4[i:i + chunk], b4[i:i + chunk], max_val=1.0).numpy()
          for i in range(0, len(a4), chunk)]
  return float(np.concatenate(vals).mean())


def persistence_baseline(frames_n: np.ndarray, stamps: list[datetime],
                         segments: list[tuple[int, int]]) -> tuple[float, float]:
  """Persistence(다음 = 현재) 베이스라인. 세그먼트 경계를 넘는 쌍은 제외한다.

  잔차 구조 모델에서 Δ=0 이면 출력이 곧 Persistence 이므로 이 값이 학습의 출발점이다.
  Returns:
    (MAE, SSIM)
  """
  pairs = [(t, t + 1) for s, e in segments for t in range(s, e - 1)]
  if not pairs:
    raise ValueError("연속 쌍이 없다. 세그먼트가 너무 짧다.")
  idx_i = np.array([i for i, _ in pairs])
  idx_j = np.array([j for _, j in pairs])
  mae_arr = np.abs(frames_n[idx_j] - frames_n[idx_i]).mean(axis=(1, 2))
  ssim = ssim_metric(frames_n[idx_i], frames_n[idx_j])
  mae = float(mae_arr.mean())

  total = len(frames_n) - 1
  logger.info("연속 쌍 %d개 (전체 %d쌍 중 경계 %d쌍 제외)", len(pairs), total, total - len(pairs))
  logger.info("Persistence  MAE = %.5f   SSIM = %.4f", mae, ssim)

  day = np.array([stamps[i].hour < DAYTIME_END_HOUR for i in idx_i])
  if day.any() and (~day).any():   # HOURS 로 한쪽만 남기면 비교가 불가능하다
    d, n = mae_arr[day].mean(), mae_arr[~day].mean()
    logger.info("  주간(00~06Z) MAE = %.5f / 야간 MAE = %.5f -> 주간이 %.1f배 어렵다", d, n, d / n)
  return mae, ssim


# --------------------------------------------------------------------------
# 데이터셋
# --------------------------------------------------------------------------
def window_starts(segments: list[tuple[int, int]], in_frames: int) -> list[int]:
  """세그먼트 경계를 넘지 않는 윈도우 시작 인덱스 목록 (시간순).

  정답 프레임 w + in_frames 가 구간 안(e-1 이하)이어야 하므로 range 끝은 e - in_frames.
  """
  starts: list[int] = []
  for s, e in segments:
    starts.extend(range(s, e - in_frames))
  return starts


def split_starts(starts: list[int], in_frames: int,
                 ratio: float) -> tuple[list[int], list[int]]:
  """시간 순서를 유지한 train/val 분할 (셔플 금지).

  val 은 cut 에서 in_frames 만큼 띄운다. 그렇지 않으면 val 첫 윈도우의 입력이
  train 마지막 윈도우의 타깃을 포함해 누수가 된다.
  """
  cut = int(len(starts) * ratio)
  train, val = starts[:cut], starts[cut + in_frames:]
  if not train or not val:
    raise ValueError(f"윈도우 {len(starts)}개로는 분할할 수 없다 (train={len(train)}, val={len(val)}).")
  return train, val


def patch_grid(size: int, patch: int, stride: int) -> list[int]:
  """한 축의 패치 시작 좌표. size - patch + 1 의 '+1' 이 있어야 오른쪽 끝 패치가 포함된다.

  250, 96, 77 -> [0, 77, 154] (154 + 96 = 250, 커버리지 100%)
  """
  if patch > size:
    raise ValueError(f"PATCH={patch} 가 프레임 크기 {size} 보다 크다.")
  return list(range(0, size - patch + 1, stride))


def build_dataset(frames_n: np.ndarray, starts: list[int], in_frames: int,
                  patch: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
  """윈도우 x 패치 조합으로 (X, Y) 를 만든다.

  X: (N, in_frames, patch, patch, 1)  Y: (N, patch, patch, 1)
  list.append 후 np.array 는 피크 메모리가 2배라 미리 할당해 바로 채운다.
  """
  _, H, W = frames_n.shape
  ys, xs = patch_grid(H, patch, stride), patch_grid(W, patch, stride)
  n = len(starts) * len(ys) * len(xs)
  X = np.empty((n, in_frames, patch, patch, 1), dtype=np.float32)
  Y = np.empty((n, patch, patch, 1), dtype=np.float32)
  k = 0
  for w in starts:
    seq = frames_n[w:w + in_frames]
    tgt = frames_n[w + in_frames]
    for y in ys:
      for x in xs:
        X[k, ..., 0] = seq[:, y:y + patch, x:x + patch]
        Y[k, ..., 0] = tgt[y:y + patch, x:x + patch]
        k += 1
  return X, Y


def log_coverage(H: int, W: int, patch: int, stride: int) -> float:
  """패치 격자가 프레임을 얼마나 덮는지 로그로 남긴다. 1.0 이면 빠짐없이 덮은 것."""
  ys, xs = patch_grid(H, patch, stride), patch_grid(W, patch, stride)
  cov = ((max(ys) + patch) * (max(xs) + patch)) / (H * W)
  logger.info("패치 격자 %dx%d = %d개/프레임, 커버리지 %.0f%%", len(ys), len(xs), len(ys) * len(xs), cov * 100)
  return cov


# --------------------------------------------------------------------------
# 손실 · 잔차 head · 컴파일
# --------------------------------------------------------------------------
def ssim_mae_loss(y_true, y_pred):
  """손실 = 0.5*MAE + 0.5*(1-SSIM). 엣지/구조 보존을 유도해 흐릿함을 줄인다."""
  import tensorflow as tf

  mae = tf.reduce_mean(tf.abs(y_true - y_pred))
  ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
  return 0.5 * mae + 0.5 * (1.0 - ssim)


def make_take_last_frame_layer():
  """TakeLastFrame 레이어 클래스를 지연 생성한다 (모듈 import 시 tensorflow 를 강제하지 않기 위해)."""
  from tensorflow import keras

  class TakeLastFrame(keras.layers.Layer):
    """시퀀스에서 마지막 시점만 뽑는다. (B, T, H, W, C) -> (B, H, W, C)

    Lambda 는 바이트코드로 저장돼 다른 환경에서 불러오기 어렵고,
    compute_output_shape 를 직접 주면 Keras 2/3 양쪽에서 안전하다.
    """

    def call(self, x):
      """시간 축의 마지막 원소를 고른다."""
      return x[:, -1]

    def compute_output_shape(self, input_shape):
      """시간 축(index 1)을 뺀 shape."""
      return (input_shape[0],) + tuple(input_shape[2:])

  return TakeLastFrame


def residual_head(inp, delta):
  """출력 = 입력 마지막 프레임 + Δ. 세 모델이 공유하는 잔차 head.

  mixed_float16 정책에서도 Δ 와 합은 float32 로 유지한다 (손실 수치 안정성).
  Args:
    inp: (B, T, H, W, 1) 모델 입력 텐서
    delta: (B, H, W, 1) float32 예측 변화량
  Returns:
    (B, H, W, 1) float32 출력 텐서
  """
  from tensorflow.keras import layers

  TakeLastFrame = make_take_last_frame_layer()
  return layers.Add(dtype="float32")([TakeLastFrame(dtype="float32")(inp), delta])


def make_optimizer(lr: float):
  """Keras 2 는 legacy.Adam(Apple Silicon 에서 빠름), Keras 3 는 표준 Adam.

  Keras 3 에도 keras.optimizers.legacy 네임스페이스는 남아 있지만
  Adam 을 만드는 순간 ImportError 를 던지므로 getattr 검사만으로는 부족하다.
  """
  from tensorflow import keras

  try:
    return keras.optimizers.legacy.Adam(lr)
  except (AttributeError, ImportError):
    return keras.optimizers.Adam(lr)


def compile_model(model, lr: float):
  """세 모델 공통 컴파일 (같은 optimizer·손실·지표로 맞춰야 비교가 성립한다).

  Returns:
    컴파일된 model (체이닝용으로 그대로 반환)
  """
  model.compile(optimizer=make_optimizer(lr), loss=ssim_mae_loss, metrics=["mae"])
  return model


# --------------------------------------------------------------------------
# 학습 · 평가
# --------------------------------------------------------------------------
def train_model(model, X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray, cfg: Config,
                model_dir: Path, model_name: str) -> tuple[Any, float]:
  """EarlyStopping / ReduceLROnPlateau / 체크포인트 / CSV 로그와 함께 학습한다.

  Colab 세션이 끊겨도 체크포인트가 Drive 에 남도록 매 epoch best 가중치를 저장한다.
  Returns:
    (keras History, model.fit 벽시계 초)
  """
  from tensorflow import keras

  model_dir.mkdir(parents=True, exist_ok=True)
  weights_path = model_dir / WEIGHTS_NAME.format(model=model_name.lower())
  callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience,
                                  restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=cfg.lr_patience, factor=0.5),
    keras.callbacks.ModelCheckpoint(str(model_dir / CHECKPOINT_NAME), monitor="val_loss",
                                    save_best_only=True, save_weights_only=True),
    keras.callbacks.CSVLogger(str(model_dir / TRAIN_LOG_NAME)),
  ]
  steps = int(np.ceil(len(X_train) / cfg.batch))
  logger.info("%d샘플 / batch %d = %d step/epoch, 최대 %d epoch", len(X_train), cfg.batch, steps, cfg.epochs)

  t0 = time.time()
  history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                      epochs=cfg.epochs, batch_size=cfg.batch, callbacks=callbacks, verbose=1)
  seconds = time.time() - t0
  ran = len(history.history["loss"])
  logger.info("학습 시간: %.1f분 (%d epoch, epoch 당 %.0fs)", seconds / 60, ran, seconds / ran)
  model.save_weights(str(weights_path))
  logger.info("가중치 저장: %s", weights_path)
  return history, seconds


def plot_history(history, model_dir: Path) -> None:
  """학습 곡선을 그린다."""
  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(7, 4))
  plt.plot(history.history["loss"], label="train")
  plt.plot(history.history["val_loss"], label="val")
  plt.xlabel("epoch")
  plt.ylabel("loss = 0.5*MAE + 0.5*(1-SSIM)")
  plt.legend()
  plt.grid(alpha=0.3)
  plt.title("Training history")
  save_and_show(fig, model_dir, FIGURE_NAMES["history"])


def evaluate_patches(model, X_val: np.ndarray, Y_val: np.ndarray, batch: int,
                     model_name: str = "model") -> dict[str, float]:
  """검증 패치에서 모델 vs Persistence 를 비교한다 (MAE 낮을수록 / SSIM 높을수록 좋다)."""
  pred = np.clip(model.predict(X_val, batch_size=batch, verbose=0), 0, 1)
  result = {
    "model_mae": float(np.mean(np.abs(pred - Y_val))),
    "pers_mae": float(np.mean(np.abs(X_val[:, -1] - Y_val))),
    "model_ssim": ssim_metric(pred[..., 0], Y_val[..., 0]),
    "pers_ssim": ssim_metric(X_val[:, -1, ..., 0], Y_val[..., 0]),
  }
  logger.info("[Val] %-11s MAE = %.5f   SSIM = %.4f", model_name, result["model_mae"], result["model_ssim"])
  logger.info("[Val] Persistence MAE = %.5f   SSIM = %.4f", result["pers_mae"], result["pers_ssim"])
  gain = (result["pers_mae"] - result["model_mae"]) / result["pers_mae"] * 100
  logger.info("-> MAE 기준 모델이 베이스라인보다 %s (%+.1f%%)", "우수" if gain > 0 else "열등", gain)
  return result


def predict_full_frame(model, frames_n: np.ndarray, stamps: list[datetime],
                       segments: list[tuple[int, int]], cfg: Config,
                       build_model_fn: Callable[..., Any], model_dir: Path,
                       model_name: str = "model") -> dict[str, Any]:
  """마지막 연속 구간의 끝 시점을 250x250 전체 크기로 예측하고 그림·지표를 남긴다.

  학습은 PATCH 크기로 했지만 Conv 계열 가중치는 크기와 무관하므로
  build_model_fn 으로 전체 크기 모델을 새로 만들어 가중치를 옮긴다.
  Returns:
    MAE/SSIM 4개와 예측 시각 문자열(t_pred, inputs)
  """
  import matplotlib.pyplot as plt

  seg_s, seg_e = segments[-1]
  if seg_e - seg_s < cfg.in_frames + 1:
    raise ValueError("마지막 연속 구간이 너무 짧다. 다른 세그먼트를 골라라.")
  t_pred = seg_e - 1
  seq = frames_n[t_pred - cfg.in_frames:t_pred][None, ..., None]
  true_next = frames_n[t_pred]

  H, W = frames_n.shape[1:]
  infer_model = build_model_fn(cfg.in_frames, cfg.filters, H, W, cfg.lr)
  infer_model.set_weights(model.get_weights())
  logger.info("추론용 모델 재구성: %dx%d, 가중치 %s개 이전 완료", H, W, f"{infer_model.count_params():,}")

  pred_next = np.clip(infer_model.predict(seq, verbose=0)[0, ..., 0], 0, 1)
  inputs_label = f"{stamps[t_pred - cfg.in_frames]:%H:%M}~{stamps[t_pred - 1]:%H:%M}"
  t_pred_label = f"{stamps[t_pred]:%H:%M}"
  logger.info("입력 %s -> 예측 %s UTC", inputs_label, t_pred_label)

  fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  axes[0].imshow(frames_n[t_pred - 1], cmap="gray", vmin=0, vmax=1)
  axes[0].set_title("last input (t-1)")
  axes[1].imshow(true_next, cmap="gray", vmin=0, vmax=1)
  axes[1].set_title("ground truth (t)")
  axes[2].imshow(pred_next, cmap="gray", vmin=0, vmax=1)
  axes[2].set_title(f"prediction (t) — {model_name}")
  im = axes[3].imshow(np.abs(pred_next - true_next), cmap="inferno", vmin=0, vmax=0.05)
  axes[3].set_title("|error|")
  fig.colorbar(im, ax=axes[3], fraction=0.046)
  for ax in axes:
    ax.axis("off")
  fig.tight_layout()
  save_and_show(fig, model_dir, FIGURE_NAMES["full_frame"])
  np.save(model_dir / PRED_NEXT_NAME, pred_next)

  result: dict[str, Any] = {
    "model_mae": float(np.mean(np.abs(pred_next - true_next))),
    "model_ssim": ssim_metric(pred_next[None], true_next[None]),
    "pers_mae": float(np.mean(np.abs(frames_n[t_pred - 1] - true_next))),
    "pers_ssim": ssim_metric(frames_n[t_pred - 1][None], true_next[None]),
    "t_pred": t_pred_label,
    "inputs": inputs_label,
  }
  logger.info("full-frame  %-11s MAE = %.5f  SSIM = %.4f", model_name,
              result["model_mae"], result["model_ssim"])
  logger.info("full-frame  Persistence MAE = %.5f  SSIM = %.4f", result["pers_mae"], result["pers_ssim"])
  return result


# --------------------------------------------------------------------------
# 실행 기록 (metrics.json)
# --------------------------------------------------------------------------
def environment_info(colab: bool, precision_policy: str) -> dict[str, Any]:
  """metrics.json 의 env 절. 모델 비교 시 실행 환경 차이를 구분하기 위해 남긴다.

  Args:
    colab: Colab 런타임 여부
    precision_policy: keras.mixed_precision.global_policy().name
  Returns:
    colab/platform/python/tensorflow/keras/gpu/precision_policy 7개 키
  """
  import tensorflow as tf
  from tensorflow import keras

  keras_version = getattr(keras, "__version__", None)
  if keras_version is None:      # Keras 2 의 tf.keras 에는 __version__ 이 없다
    import keras as keras_pkg

    keras_version = keras_pkg.__version__

  gpus = tf.config.list_physical_devices("GPU")
  gpu = "none"
  if gpus:
    gpu = str(tf.config.experimental.get_device_details(gpus[0]).get("device_name") or "unknown")

  return {
    "colab": bool(colab),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "tensorflow": str(tf.__version__),
    "keras": str(keras_version),
    "gpu": gpu,
    "precision_policy": str(precision_policy),
  }


def _jsonable(value: Any) -> Any:
  """numpy 스칼라/배열을 python 기본형으로 바꾼다 (json.dump 는 numpy 타입을 못 쓴다)."""
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  if isinstance(value, np.ndarray):
    return _jsonable(value.tolist())
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  return value


def write_metrics(path: Path, payload: dict) -> None:
  """metrics.json 을 쓴다. numpy 타입은 float/int 로 변환한다."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
  logger.info("metrics 저장: %s", path)


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------
def build_arg_parser(description: str) -> argparse.ArgumentParser:
  """모델 스크립트 공통 CLI 파서. 경로 기본값은 Colab/로컬 여부에 따라 다르다."""
  colab = is_colab()
  p = argparse.ArgumentParser(description=description)
  p.add_argument("--data-dir", type=Path, default=COLAB_DATA_DIR if colab else LOCAL_DATA_DIR,
                 help=".nc 디렉터리 (Drive 경로)")
  p.add_argument("--data-zip", type=Path, default=None,
                 help=".nc 를 담은 zip (Drive 경로). data-dir 이 비어 있으면 로컬 디스크에 푼다")
  p.add_argument("--out-dir", type=Path, default=COLAB_OUT_DIR if colab else LOCAL_OUT_DIR,
                 help="모델별 결과 디렉터리의 부모. 그림/가중치/캐시가 저장된다")
  p.add_argument("--epochs", type=int, default=4)
  p.add_argument("--batch", type=int, default=16)
  p.add_argument("--filters", type=int, default=16)
  p.add_argument("--target", type=int, default=250, help="다운샘플 해상도. 0 이면 원본 유지")
  p.add_argument("--hours", nargs="*", type=int, default=None,
                 help="사용할 UTC 시각 목록. 예: --hours 6 7 8 (주간 태양반사 제외 시 6~23)")
  p.add_argument("--no-cache", action="store_true", help="npz 캐시를 쓰지 않고 .nc 를 다시 읽는다")
  p.add_argument("--no-mixed-precision", action="store_true")
  p.add_argument("--verbose", action="store_true")
  return p


def config_from_args(a: argparse.Namespace) -> Config:
  """파싱된 인자를 Config 로 변환한다."""
  hours = sorted({h for h in a.hours if 0 <= h <= 23}) if a.hours else None
  cfg = Config(
    data_dir=a.data_dir, out_dir=a.out_dir, data_zip=a.data_zip,
    epochs=a.epochs, batch=a.batch, filters=a.filters,
    target=a.target or None, hours=hours,
    use_cache=not a.no_cache, mixed_precision=not a.no_mixed_precision,
    extra={"verbose": a.verbose},
  )
  if cfg.epochs < 1 or cfg.batch < 1:
    raise ValueError("epochs 와 batch 는 1 이상이어야 한다.")
  return cfg


def run(cfg: Config, build_model_fn: Callable[..., Any], model_name: str) -> dict[str, Any]:
  """전체 파이프라인을 실행하고 <out_dir>/<model_name>/metrics.json 을 남긴다.

  Args:
    cfg: 실행 설정
    build_model_fn: build_model(in_frames, filters, h, w, lr) -> compile 완료 keras.Model
    model_name: 결과 디렉터리 이름 (ConvLSTM | SimVP | PredRNN_V2)
  Returns:
    저장한 metrics payload
  """
  from tensorflow import keras

  mount_drive()
  ensure_netcdf_backend()
  logger.info("font: %s", setup_korean_font())
  setup_gpu(cfg.mixed_precision)
  keras.utils.set_random_seed(cfg.seed)
  model_dir = cfg.out_dir / model_name
  model_dir.mkdir(parents=True, exist_ok=True)

  # 1) 적재 · 세그먼트
  frames, stamps = load_frames(cfg)
  frames, stamps = filter_hours(frames, stamps, cfg.hours)
  segments = find_segments(stamps)
  logger.info("연속 구간 %d개 (결측으로 끊긴 지점):", len(segments))
  for s, e in segments:
    logger.info("  [%3d:%3d] %s ~ %s  (%d장)", s, e, f"{stamps[s]:%H:%M}", f"{stamps[e - 1]:%H:%M}", e - s)

  # 2) 정규화 · 시각화 · 베이스라인
  frames_n, gmin, gmax = normalize(frames)
  del frames
  logger.info("global min=%s, max=%s", gmin, gmax)
  plot_samples(frames_n, stamps, model_dir)
  plot_hourly_mean(frames_n, stamps, model_dir)
  base_mae, base_ssim = persistence_baseline(frames_n, stamps, segments)

  # 3) 데이터셋
  log_coverage(frames_n.shape[1], frames_n.shape[2], cfg.patch, cfg.stride)
  starts = window_starts(segments, cfg.in_frames)
  train_starts, val_starts = split_starts(starts, cfg.in_frames, cfg.val_ratio)
  logger.info("윈도우 %d개 -> train %d / val %d (경계 %d개는 누수 방지로 버림)",
              len(starts), len(train_starts), len(val_starts), cfg.in_frames)
  train_period = f"{stamps[train_starts[0]]:%H:%M} ~ {stamps[train_starts[-1] + cfg.in_frames]:%H:%M}"
  val_period = f"{stamps[val_starts[0]]:%H:%M} ~ {stamps[val_starts[-1] + cfg.in_frames]:%H:%M}"
  logger.info("train 기간: %s", train_period)
  logger.info("val   기간: %s", val_period)
  X_train, Y_train = build_dataset(frames_n, train_starts, cfg.in_frames, cfg.patch, cfg.stride)
  X_val, Y_val = build_dataset(frames_n, val_starts, cfg.in_frames, cfg.patch, cfg.stride)
  logger.info("train: %s %s (%.2f GB)", X_train.shape, Y_train.shape, (X_train.nbytes + Y_train.nbytes) / 1e9)
  logger.info("val  : %s %s (%.2f GB)", X_val.shape, Y_val.shape, (X_val.nbytes + Y_val.nbytes) / 1e9)

  # 4) 모델 · 학습 · 평가
  model = build_model_fn(cfg.in_frames, cfg.filters, cfg.patch, cfg.patch, cfg.lr)
  model.summary(print_fn=logger.info)
  history, seconds = train_model(model, X_train, Y_train, X_val, Y_val, cfg, model_dir, model_name)
  plot_history(history, model_dir)
  val_result = evaluate_patches(model, X_val, Y_val, cfg.batch, model_name)
  full_result = predict_full_frame(model, frames_n, stamps, segments, cfg,
                                   build_model_fn, model_dir, model_name)

  # 5) 기록
  epochs_run = len(history.history["loss"])
  payload: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "model": model_name,
    "params": int(model.count_params()),
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "env": environment_info(is_colab(), keras.mixed_precision.global_policy().name),
    "config": {
      "in_frames": cfg.in_frames, "target": cfg.target, "patch": cfg.patch, "stride": cfg.stride,
      "filters": cfg.filters, "epochs": cfg.epochs, "batch": cfg.batch, "lr": cfg.lr,
      "hours": cfg.hours, "seed": cfg.seed,
    },
    "data": {
      "n_frames": int(len(frames_n)),
      "period": f"{stamps[0]:%Y-%m-%d %H:%M} ~ {stamps[-1]:%H:%M} UTC",
      "segments": [[s, e] for s, e in segments],
      "gmin": gmin, "gmax": gmax,
      "n_train": int(len(X_train)), "n_val": int(len(X_val)),
      "train_period": train_period, "val_period": val_period,
    },
    "baseline": {"mae": base_mae, "ssim": base_ssim},
    "train": {
      "epochs_run": epochs_run,
      "seconds": round(seconds, 1),
      "sec_per_epoch": round(seconds / epochs_run, 1),
      "history": {k: [float(v) for v in history.history[k]]
                  for k in HISTORY_KEYS if k in history.history},
    },
    "val": {
      **{k: val_result[k] for k in ("model_mae", "model_ssim", "pers_mae", "pers_ssim")},
      "mae_gain_pct": (val_result["pers_mae"] - val_result["model_mae"]) / val_result["pers_mae"] * 100,
    },
    "full_frame": full_result,
    "figures": dict(FIGURE_NAMES),
  }
  write_metrics(model_dir / METRICS_NAME, payload)
  logger.info("결과 저장 위치: %s", model_dir)
  return payload


def main_for_model(build_model_fn: Callable[..., Any], model_name: str,
                   description: str, argv: list[str] | None = None) -> int:
  """모델 스크립트 공통 CLI 진입점. 인자 파싱 + logging 설정 + run()."""
  try:
    cfg = config_from_args(build_arg_parser(description).parse_args(argv))
  except ValueError as exc:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    logger.error("%s", exc)
    return 1
  logging.basicConfig(
    level=logging.DEBUG if cfg.extra.get("verbose") else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    stream=sys.stdout, force=True,
  )
  try:
    run(cfg, build_model_fn, model_name)
    return 0
  except (FileNotFoundError, ValueError, RuntimeError) as exc:
    logger.error("%s", exc)
    return 1
