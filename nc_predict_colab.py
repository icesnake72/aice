"""GK2A SW038 다음 프레임 예측 — ConvLSTM 엔트리 (공통 파이프라인: nc_pipeline.py).

데이터 적재·분할·손실·평가·기록은 전부 `nc_pipeline` 에 있고, 이 파일은 모델 구조만 정의한다.
SimVP · PredRNN-V2 와 같은 데이터·손실·지표로 비교하기 위한 구성이다.

Colab 사용법
  1) 로컬에서 데이터 다운로드 (AWS Open Data, 익명 접근)
       python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out netcdf
  2) 결과 폴더(또는 zip)를 Google Drive 에 업로드
       nc_pipeline.COLAB_DATA_DIR (MyDrive/netcdf) 아래에 .nc 를 두거나
       zip 하나로 올린 뒤 --data-zip 으로 가리킨다 (COLAB_UNZIP_DIR 에 풀린다)
  3) Colab 메뉴: 런타임 > 런타임 유형 변경 > T4 GPU
  4) nc_pipeline.py 와 이 파일을 Colab 에 올린 뒤 셀에서 실행
       %run nc_predict_colab.py                                   # 그림이 셀에 바로 표시된다
       %run nc_predict_colab.py --epochs 2 --hours 6 7 8 9 10 11  # 빠른 확인
     노트북으로 쓰려면 `python3 tools/build_colab_notebook.py --model convlstm --profile colab`.

  결과는 nc_pipeline.COLAB_OUT_DIR / "ConvLSTM" (로컬은 LOCAL_OUT_DIR / "ConvLSTM") 에
  metrics.json, 그림 4장, 가중치, train_log.csv 로 저장된다.
"""

from __future__ import annotations

import sys

from nc_pipeline import (
  compile_model,
  delta_readout,
  main_for_model,
  residual_head,
)

MODEL_NAME = "ConvLSTM"


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------
def build_model(in_frames: int, filters: int, h: int, w: int, lr: float = 1e-3):
  """ConvLSTM2D x2 -> Conv2D(Δ) -> 마지막 입력 프레임 + Δ 잔차 모델 (Functional API).

  Keras 3 의 ConvLSTM 은 공간 크기에 None 을 허용하지 않아 h, w 를 고정한다.
  가중치는 입력 크기와 무관하므로 추론 시 다른 크기로 다시 만들어 set_weights 하면 된다.
  Metal 에서 5D BatchNorm 이 비호환이라 BN 은 쓰지 않는다.
  """
  from tensorflow import keras
  from tensorflow.keras import layers

  inp = keras.Input(shape=(in_frames, h, w, 1))
  x = layers.ConvLSTM2D(filters, (3, 3), padding="same",
                        return_sequences=True, activation="tanh")(inp)
  x = layers.ConvLSTM2D(filters, (3, 3), padding="same",
                        return_sequences=False, activation="tanh")(x)
  # Δ 는 0 초기화 readout 이라 학습 시작 시 출력 = 입력 마지막 프레임(Persistence)이다.
  delta = delta_readout(x, kernel_size=3)
  return compile_model(keras.Model(inputs=inp, outputs=residual_head(inp, delta)), lr)


def main(argv: list[str] | None = None) -> int:
  """CLI 진입점."""
  return main_for_model(build_model, MODEL_NAME,
                        "GK2A SW038 next-frame prediction (ConvLSTM)", argv)


if __name__ == "__main__":
  sys.exit(main())
