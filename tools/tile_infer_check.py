"""해상도 전이 가설 검증 — 96x96 타일 추론 vs 250x250 직접 추론 (학습 없음).

네 모델은 전부 PATCH(96) 크기로 학습한 뒤, `nc_pipeline.predict_full_frame()` 에서
250x250 모델을 새로 만들어 `set_weights` 로 가중치를 옮겨 full-frame 을 예측한다.
Conv 계열은 평행이동 등변성 덕에 이 전이가 자연스럽지만, VideoTransformer 의 공간
attention 은 토큰 수가 144(12x12) -> 1024(32x32) 로 7.1배 늘어나 동작이 달라진다.

정확한 메커니즘은 아직 특정하지 못했다. val 136장 실측(2026-09-18)으로 확인한 것은
  - attention logit 을 log(N) 비율로 키우는 보정: 오히려 나빠진다 (배수를 키울수록 단조 악화)
  - 위치 인코딩 좌표를 학습 격자 범위로 보간: 3.3% 개선에 그친다
  - 학습 때와 같은 96x96 으로 잘라 추론: 20.0% 개선
앞의 두 가설이 설명하지 못하는 몫이 대부분이라, 파이프라인은 측정된 쪽(타일 추론)을 쓴다.

이 스크립트는 학습된 가중치를 그대로 쓰고 추론 방식만 바꿔 그 차이를 잰다.

  direct : 250x250 모델 1회 추론 (현재 파이프라인과 동일)
  tiled  : 96x96 모델로 3x3=9 타일을 각각 추론한 뒤 겹치는 픽셀을 평균 (학습 분포와 동일)

두 방식의 차이가 VideoTransformer 에서만 크게 나오면 전이 가설이 확정된다.
Conv 계열 세 모델이 대조군이다.

읽기 전용이다 — results/ 아래 어떤 파일도 고치지 않고 표만 출력한다.

실행:
  python3 tools/tile_infer_check.py
  python3 tools/tile_infer_check.py --models VideoTransformer SimVP
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import numpy as np

# tf.image.ssim 의 Gaussian filter 는 conv 라서 Ampere(compute 8.6)에서 기본값인 TF32 로
# 돌면 mantissa 10비트만 써서 SSIM 이 약 0.005 낮게 나온다 (MAE 는 영향 없다). 이 값을 끄면
# results/<Model>/metrics.json 의 full_frame SSIM 이 비트 단위로 재현된다.
# tensorflow import 보다 먼저 설정해야 효과가 있다.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nc_pipeline as P  # noqa: E402  (sys.path 조정 후에 import 해야 한다)

logger = logging.getLogger("tile_check")

# 결과 디렉터리 이름 -> build_model 을 가진 모듈.
MODEL_MODULES: dict[str, str] = {
  "ConvLSTM": "nc_predict_colab",
  "SimVP": "simvp_predict_colab",
  "PredRNN_V2": "predrnn_v2_predict_colab",
  "VideoTransformer": "videotf_predict_colab",
}


def score(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
  """(MAE, SSIM). 파이프라인과 같은 방식으로 잰다."""
  return float(np.mean(np.abs(pred - true))), P.ssim_metric(pred[None], true[None])


def check_model(name: str, seq: np.ndarray, true_next: np.ndarray,
                cfg: P.Config) -> dict[str, float] | None:
  """한 모델에 대해 direct/tiled 추론을 모두 돌리고 지표를 낸다. 가중치가 없으면 None."""
  from tensorflow import keras

  weights = cfg.out_dir / name / P.WEIGHTS_NAME.format(model=name.lower())
  if not weights.exists():
    logger.warning("[%s] 가중치 없음: %s", name, weights)
    return None

  build_model = importlib.import_module(MODEL_MODULES[name]).build_model
  height, width = true_next.shape

  model_patch = build_model(cfg.in_frames, cfg.filters, cfg.patch, cfg.patch, cfg.lr)
  model_patch.load_weights(weights)

  model_full = build_model(cfg.in_frames, cfg.filters, height, width, cfg.lr)
  model_full.set_weights(model_patch.get_weights())

  direct = np.clip(model_full.predict(seq, verbose=0)[0, ..., 0], 0, 1)
  tiled = np.clip(P.predict_tiled(model_patch, seq, cfg.patch, cfg.stride), 0, 1)

  d_mae, d_ssim = score(direct, true_next)
  t_mae, t_ssim = score(tiled, true_next)
  logger.info("[%s] direct MAE=%.5f SSIM=%.4f | tiled MAE=%.5f SSIM=%.4f",
              name, d_mae, d_ssim, t_mae, t_ssim)

  del model_patch, model_full
  keras.backend.clear_session()
  return {"direct_mae": d_mae, "direct_ssim": d_ssim, "tiled_mae": t_mae, "tiled_ssim": t_ssim}


def main(argv: list[str] | None = None) -> int:
  """CLI 진입점. 캐시된 프레임을 읽어 마지막 시점을 두 방식으로 예측하고 비교표를 낸다."""
  parser = argparse.ArgumentParser(description="96x96 타일 추론 vs 250x250 직접 추론 비교")
  parser.add_argument("--models", nargs="*", default=list(MODEL_MODULES),
                      choices=list(MODEL_MODULES))
  parser.add_argument("--data-dir", type=Path, default=P.LOCAL_DATA_DIR)
  parser.add_argument("--out-dir", type=Path, default=P.LOCAL_OUT_DIR)
  parser.add_argument("--target", type=int, default=250)
  parser.add_argument("--filters", type=int, default=16)
  args = parser.parse_args(argv)

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                      datefmt="%H:%M:%S")

  cfg = P.Config(data_dir=args.data_dir, out_dir=args.out_dir,
                 target=args.target, filters=args.filters)
  P.ensure_netcdf_backend()
  P.setup_gpu(cfg.mixed_precision)

  frames, stamps = P.load_frames(cfg)
  segments = P.find_segments(stamps)
  frames_n, _, _ = P.normalize(frames)
  del frames

  # predict_full_frame() 과 같은 시점을 고른다: 마지막 연속 구간의 끝.
  _, seg_e = segments[-1]
  t_pred = seg_e - 1
  seq = frames_n[t_pred - cfg.in_frames:t_pred][None, ..., None]
  true_next = frames_n[t_pred]
  pers_mae, pers_ssim = score(frames_n[t_pred - 1], true_next)
  logger.info("입력 %s~%s -> 예측 %s UTC (%dx%d)",
              f"{stamps[t_pred - cfg.in_frames]:%H:%M}", f"{stamps[t_pred - 1]:%H:%M}",
              f"{stamps[t_pred]:%H:%M}", *true_next.shape)
  logger.info("타일: %d개 (patch %d, stride %d)",
              len(P.patch_grid(true_next.shape[0], cfg.patch, cfg.stride)) ** 2,
              cfg.patch, cfg.stride)

  results = {name: r for name in args.models
             if (r := check_model(name, seq, true_next, cfg)) is not None}

  print()
  print(f"{'MODEL':<18}{'direct MAE':>12}{'tiled MAE':>12}{'개선':>9}"
        f"{'direct SSIM':>13}{'tiled SSIM':>12}")
  print("-" * 76)
  for name in args.models:
    r = results.get(name)
    if r is None:
      continue
    gain = 100 * (r["direct_mae"] - r["tiled_mae"]) / r["direct_mae"]
    print(f"{name:<18}{r['direct_mae']:>12.5f}{r['tiled_mae']:>12.5f}{gain:>8.1f}%"
          f"{r['direct_ssim']:>13.4f}{r['tiled_ssim']:>12.4f}")
  print("-" * 76)
  print(f"{'Persistence':<18}{pers_mae:>12.5f}{'—':>12}{'—':>9}{pers_ssim:>13.4f}{'—':>12}")
  print()
  print("개선 = direct 대비 tiled 의 MAE 감소율. Conv 계열(대조군)에서 0 근처인데")
  print("VideoTransformer 에서만 크면 96->250 해상도 전이가 원인이다.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
