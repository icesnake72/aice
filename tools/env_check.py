"""GK2A 위성영상 예측 파이프라인 — 실행 환경 점검 스크립트.

Windows(WSL2)·Mac·Colab 등 어떤 인터프리터로 돌리든 TensorFlow/GPU/한글 폰트/데이터가
파이프라인을 돌릴 준비가 됐는지 한 번에 확인한다. 표준 라이브러리만으로 실행되고,
tensorflow/xarray/netCDF4/matplotlib 는 있으면 쓰고 없으면 "미설치"로 보고할 뿐 죽지 않는다
(단, tensorflow 자체가 없으면 파이프라인을 아예 못 돌리므로 종료 코드 1을 낸다).

실행:
  python tools/env_check.py
  python tools/env_check.py --data-dir resource/netcdf --no-gpu-bench
  python tools/env_check.py --json > env_report.json

종료 코드:
  0: tensorflow 설치됨 + .nc 데이터 1개 이상 (GPU 없음은 WARN 일 뿐 실패가 아니다)
  1: tensorflow 미설치 또는 --data-dir 에 .nc 파일이 0개
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# nc_pipeline.py 의 계약과 맞춘 상수 (표준 라이브러리만 쓰기 위해 여기서 다시 선언한다).
DEFAULT_DATA_DIR = Path("resource/netcdf")
CACHE_PATH = Path("results/cache/frames_sw038_t250.npz")
KOREAN_FONT_CANDIDATES: tuple[str, ...] = ("NanumGothic", "AppleGothic", "Malgun Gothic")
MATMUL_SIZE = 4096
MATMUL_REPS = 5


def get_python_platform_info() -> dict[str, str]:
  """Python 버전과 OS/아키텍처 문자열을 돌려준다."""
  return {
    "python_version": platform.python_version(),
    "platform": platform.platform(),
  }


def import_tensorflow() -> Any | None:
  """tensorflow 모듈을 임포트한다. 실패하면 None (미설치로 취급, 예외를 올리지 않는다)."""
  try:
    import tensorflow as tf

    return tf
  except Exception:
    return None


def get_keras_version(tf_module: Any | None) -> str | None:
  """Keras 버전을 얻는다.

  우선 `import keras`로 최상위 keras 패키지(Keras 3)를 시도하고,
  없으면 `tf.keras.__version__`(Keras 2, tf.keras 안에만 있음)로 대체하며,
  그마저 없으면 "2.x"로 표시한다. tensorflow 자체가 없으면 None.
  """
  if tf_module is None:
    return None
  try:
    import keras

    return str(keras.__version__)
  except Exception:
    pass
  version = getattr(getattr(tf_module, "keras", None), "__version__", None)
  return str(version) if version else "2.x"


def list_gpu_devices(tf_module: Any | None) -> list[dict[str, str]]:
  """GPU 목록을 device_name/compute_capability 와 함께 돌려준다 (없으면 빈 리스트)."""
  if tf_module is None:
    return []
  try:
    gpus = tf_module.config.list_physical_devices("GPU")
  except Exception:
    return []
  result: list[dict[str, str]] = []
  for gpu in gpus:
    try:
      details = tf_module.config.experimental.get_device_details(gpu)
    except Exception:
      details = {}
    result.append({
      "device_name": str(details.get("device_name", gpu.name)),
      "compute_capability": str(details.get("compute_capability", "?")),
    })
  return result


def check_mixed_precision(tf_module: Any | None) -> bool | None:
  """`mixed_float16` 정책 적용 성공 여부. tensorflow 가 없으면 None.

  검사 뒤에는 finally 에서 float32 로 되돌려, 다음 검사(GPU 벤치 등)에 영향을 주지 않는다.
  """
  if tf_module is None:
    return None
  try:
    from tensorflow import keras

    keras.mixed_precision.set_global_policy("mixed_float16")
    ok = keras.mixed_precision.global_policy().name == "mixed_float16"
  except Exception:
    ok = False
  finally:
    try:
      from tensorflow import keras as _keras

      _keras.mixed_precision.set_global_policy("float32")
    except Exception:
      pass
  return ok


def run_gpu_matmul_benchmark(
  tf_module: Any | None, size: int = MATMUL_SIZE, reps: int = MATMUL_REPS
) -> float | None:
  """`size`x`size` float16 행렬곱을 `reps`번 돌려 평균 소요 시간(ms)을 돌려준다.

  GPU 가 없거나 tensorflow 가 없으면 None (CPU 로는 측정 의미가 크지 않아 생략한다).
  """
  if tf_module is None:
    return None
  try:
    gpus = tf_module.config.list_physical_devices("GPU")
    if not gpus:
      return None
    with tf_module.device("/GPU:0"):
      a = tf_module.random.normal((size, size), dtype=tf_module.float16)
      b = tf_module.random.normal((size, size), dtype=tf_module.float16)
      _ = tf_module.matmul(a, b).numpy()   # 워밍업 (커널 컴파일 비용 제외)
      start = time.perf_counter()
      for _ in range(reps):
        c = tf_module.matmul(a, b)
      _ = c.numpy()                        # GPU 큐를 비워 실제 완료 시각을 잰다
      elapsed = time.perf_counter() - start
    return elapsed / reps * 1000.0
  except Exception:
    return None


def check_xarray_netcdf() -> tuple[bool, bool]:
  """xarray, (netCDF4 또는 h5netcdf) 임포트 가능 여부를 돌려준다."""
  try:
    import xarray  # noqa: F401

    xarray_ok = True
  except Exception:
    xarray_ok = False

  netcdf_ok = False
  for backend in ("netCDF4", "h5netcdf"):
    try:
      __import__(backend)
      netcdf_ok = True
      break
    except Exception:
      continue
  return xarray_ok, netcdf_ok


def count_nc_files(data_dir: Path) -> int:
  """`data_dir` 아래 `*.nc` 파일 개수. 디렉터리가 없으면 0."""
  if not data_dir.is_dir():
    return 0
  return sum(1 for _ in data_dir.glob("*.nc"))


def check_korean_font() -> str | None:
  """matplotlib 에 등록된 한글 폰트 후보(NanumGothic/AppleGothic/Malgun Gothic) 중 하나를 찾는다."""
  try:
    import matplotlib.font_manager as fm
  except Exception:
    return None
  try:
    available = {f.name for f in fm.fontManager.ttflist}
  except Exception:
    return None
  for candidate in KOREAN_FONT_CANDIDATES:
    if candidate in available:
      return candidate
  return None


def get_free_disk_gb(root: Path) -> float:
  """`root` 가 속한 드라이브의 여유 공간(GB)."""
  usage = shutil.disk_usage(root if root.exists() else root.parent)
  return usage.free / (1024 ** 3)


def build_report(data_dir: Path, run_gpu_bench: bool = True) -> dict[str, Any]:
  """환경 점검 결과를 JSON 직렬화 가능한 dict 로 만든다.

  Args:
    data_dir: `.nc` 파일을 셀 디렉터리 (보통 resource/netcdf)
    run_gpu_bench: True 면 GPU 가 있을 때 행렬곱 벤치마크를 돌린다
  Returns:
    python_version/platform/tensorflow_version/keras_version/gpus/gpu_count/
    mixed_precision_ok/gpu_bench_ms/xarray_ok/netcdf_ok/data_dir/nc_file_count/
    cache_path/cache_exists/korean_font/disk_free_gb/warnings 키를 담은 dict
  """
  warnings: list[str] = []

  tf_module = import_tensorflow()
  tensorflow_version = str(tf_module.__version__) if tf_module is not None else None
  if tensorflow_version is None:
    warnings.append("tensorflow 가 설치되어 있지 않다. 파이프라인을 실행할 수 없다.")

  keras_version = get_keras_version(tf_module)
  gpus = list_gpu_devices(tf_module)
  if tf_module is not None and not gpus:
    warnings.append("GPU 를 찾지 못했다 (CPU 로 실행됨). WSL2 라면 nvidia-smi 로 드라이버 인식을 확인한다.")

  mixed_precision_ok = check_mixed_precision(tf_module)
  gpu_bench_ms = run_gpu_matmul_benchmark(tf_module) if (run_gpu_bench and gpus) else None

  xarray_ok, netcdf_ok = check_xarray_netcdf()
  if not xarray_ok:
    warnings.append("xarray 를 임포트할 수 없다.")
  if not netcdf_ok:
    warnings.append("netCDF4/h5netcdf 를 임포트할 수 없다 (.nc 파일을 읽지 못한다).")

  nc_file_count = count_nc_files(data_dir)
  if nc_file_count == 0:
    warnings.append(f"{data_dir} 에 .nc 파일이 없다. gk2a_download.py 로 받거나 --data-dir 로 지정한다.")

  cache_exists = CACHE_PATH.is_file()

  korean_font = check_korean_font()
  if korean_font is None:
    warnings.append("한글 폰트(NanumGothic/AppleGothic/Malgun Gothic)를 찾지 못했다. 그래프 한글이 깨질 수 있다.")

  info = get_python_platform_info()

  return {
    "python_version": info["python_version"],
    "platform": info["platform"],
    "tensorflow_version": tensorflow_version,
    "keras_version": keras_version,
    "gpus": gpus,
    "gpu_count": len(gpus),
    "mixed_precision_ok": mixed_precision_ok,
    "gpu_bench_ms": gpu_bench_ms,
    "xarray_ok": xarray_ok,
    "netcdf_ok": netcdf_ok,
    "data_dir": str(data_dir),
    "nc_file_count": nc_file_count,
    "cache_path": str(CACHE_PATH),
    "cache_exists": cache_exists,
    "korean_font": korean_font,
    "disk_free_gb": round(get_free_disk_gb(Path.cwd()), 1),
    "warnings": warnings,
  }


def determine_exit_code(report: dict[str, Any]) -> int:
  """tensorflow 미설치 또는 .nc 파일 0개일 때만 1, 그 외(GPU 없음 포함)는 0."""
  if report["tensorflow_version"] is None:
    return 1
  if report["nc_file_count"] == 0:
    return 1
  return 0


def _fmt(value: Any) -> str:
  """표 출력용 값 서식화 (None/bool/float 를 사람이 읽기 좋게 바꾼다)."""
  if value is None:
    return "확인 불가"
  if isinstance(value, bool):
    return "가능" if value else "불가"
  if isinstance(value, float):
    return f"{value:.1f}"
  return str(value)


def format_table(report: dict[str, Any]) -> str:
  """report 를 사람이 읽는 표 문자열로 만든다."""
  gpu_line = "없음 (CPU)"
  if report["gpus"]:
    gpu_line = "; ".join(
      f"{g['device_name']} (compute {g['compute_capability']})" for g in report["gpus"]
    )
  bench_line = f"{report['gpu_bench_ms']:.2f} ms/회" if report["gpu_bench_ms"] is not None else "건너뜀/GPU 없음"

  rows: list[tuple[str, str]] = [
    ("Python", report["python_version"]),
    ("Platform", report["platform"]),
    ("TensorFlow", report["tensorflow_version"] or "미설치"),
    ("Keras", report["keras_version"] or "미설치"),
    ("GPU", gpu_line),
    (f"행렬곱 벤치({MATMUL_SIZE}x{MATMUL_SIZE} fp16)", bench_line),
    ("mixed_float16 정책 적용", _fmt(report["mixed_precision_ok"])),
    ("xarray", _fmt(report["xarray_ok"])),
    ("netCDF4/h5netcdf", _fmt(report["netcdf_ok"])),
    ("데이터 디렉터리", report["data_dir"]),
    (".nc 파일 수", str(report["nc_file_count"])),
    ("프레임 캐시", f"{report['cache_path']} ({'있음' if report['cache_exists'] else '없음'})"),
    ("한글 폰트", report["korean_font"] or "없음"),
    ("여유 디스크 공간", f"{report['disk_free_gb']:.1f} GB"),
  ]
  width = max(len(label) for label, _ in rows)
  lines = ["=== GK2A 파이프라인 환경 점검 ===", ""]
  lines += [f"{label.ljust(width)} : {value}" for label, value in rows]
  if report["warnings"]:
    lines.append("")
    lines.append("경고:")
    lines += [f"  - {w}" for w in report["warnings"]]
  return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI 파서. Windows/WSL2·Mac·Colab 어디서든 같은 옵션으로 쓴다."""
  parser = argparse.ArgumentParser(description="GK2A 예측 파이프라인 실행 환경을 점검한다.")
  parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                      help=f".nc 디렉터리 (기본: {DEFAULT_DATA_DIR})")
  parser.add_argument("--no-gpu-bench", action="store_true",
                      help="GPU 행렬곱 벤치마크를 건너뛴다")
  parser.add_argument("--json", action="store_true",
                      help="표 대신 JSON 한 덩어리로 출력한다 (스크립트 연동용)")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """CLI 진입점: 점검 결과를 출력하고 종료 코드를 돌려준다."""
  args = build_arg_parser().parse_args(argv)
  report = build_report(args.data_dir, run_gpu_bench=not args.no_gpu_bench)

  if args.json:
    print(json.dumps(report, ensure_ascii=False, indent=2))
  else:
    print(format_table(report))

  return determine_exit_code(report)


if __name__ == "__main__":
  raise SystemExit(main())
