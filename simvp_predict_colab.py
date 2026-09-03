"""GK2A SW038 다음 프레임 예측 — SimVP 엔트리 (공통 파이프라인: nc_pipeline.py).

데이터 적재·분할·손실·평가·기록은 전부 `nc_pipeline` 에 있고, 이 파일은 모델 구조만 정의한다.
ConvLSTM · PredRNN-V2 와 같은 데이터·손실·지표로 비교하기 위한 구성이다.

SimVP (Gao et al., CVPR 2022, arXiv:2206.05099, 공식 구현 OpenSTL, Apache-2.0) 는
RNN 없이 CNN 만으로 영상을 예측한다. Encoder(공간 압축) → Translator(시간을 채널로 접어
Inception 으로 섞음) → Decoder(복원) 3단 구조이고, 여기서는 다음 1장만 예측하도록
`aft_seq_length = 1` 로 줄이고 다운샘플을 1회(N_S=2)만 해서 96 과 250 을 모두 지원한다.

Colab 사용법
  1) 로컬에서 데이터 다운로드 (AWS Open Data, 익명 접근)
       python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out netcdf
  2) 결과 폴더(또는 zip)를 Google Drive 에 업로드
       nc_pipeline.COLAB_DATA_DIR (MyDrive/netcdf) 아래에 .nc 를 두거나
       zip 하나로 올린 뒤 --data-zip 으로 가리킨다 (COLAB_UNZIP_DIR 에 풀린다)
  3) Colab 메뉴: 런타임 > 런타임 유형 변경 > T4 GPU
  4) nc_pipeline.py 와 이 파일을 Colab 에 올린 뒤 셀에서 실행
       %run simvp_predict_colab.py                                   # 그림이 셀에 바로 표시된다
       %run simvp_predict_colab.py --epochs 2 --hours 6 7 8 9 10 11  # 빠른 확인
     노트북으로 쓰려면 `python3 tools/build_colab_notebook.py --model simvp --profile colab`.

  결과는 nc_pipeline.COLAB_OUT_DIR / "SimVP" (로컬은 LOCAL_OUT_DIR / "SimVP") 에
  metrics.json, 그림 4장, 가중치, train_log.csv 로 저장된다.
"""

from __future__ import annotations

import sys

from nc_pipeline import (
  compile_model,
  main_for_model,
  make_take_last_frame_layer,
  residual_head,
)

MODEL_NAME = "SimVP"

# N_S 는 인코더/디코더 ConvSC 개수다. 2(= 다운샘플 1회) 로 고정해 96->48, 250->125 를 맞췄고
# build_model 이 두 블록을 직접 쓴다. 값을 바꾸려면 build_model 의 인코더·디코더도 함께 고쳐야 한다.
N_S = 2
N_T = 2                     # Translator(Mid_Xnet) 의 enc/dec Inception 블록 개수 (translator 가 사용)
INCEP_KER = (3, 5, 7)       # Inception 병렬 커널. 공식 기본값 [3,5,7,11] 중 11 은 250 에 과하다
GN_GROUPS_S = 2             # ConvSC 의 GroupNormalization 그룹 수 (공식 구현과 동일)
GN_GROUPS_T = 8             # Inception 의 GroupNormalization 그룹 수 (공식 구현과 동일)
DELTA_LAYER_NAME = "delta"  # 잔차 Δ 를 내는 readout Conv2D. 테스트가 이름으로 찾는다


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------
def norm_groups(channels: int, preferred: int) -> int:
  """channels 를 나누어떨어지는 가장 큰 그룹 수(<= preferred)를 고른다.

  filters 가 작으면(테스트의 filters=2) 기본 그룹 수로 채널을 나눌 수 없어 GN 생성이 실패한다.
  공식 구현도 나누어떨어지지 않으면 groups 를 낮추므로 같은 취지다.
  """
  return next((g for g in range(min(preferred, channels), 0, -1) if channels % g == 0), 1)


def conv_sc(c_out: int, stride: int, transpose: bool, name: str):
  """SimVP 의 ConvSC 블록: Conv(3x3) -> GroupNormalization(2) -> LeakyReLU(0.2).

  Sequential 로 감싸는 이유는 TimeDistributed 로 프레임마다 같은 가중치를 적용하기 위해서다.
  업샘플은 원논문 구현대로 Conv2DTranspose(stride 2) 를 쓴다
  (OpenSTL 최신판의 PixelShuffle 은 같은 폭에서 파라미터가 4배가 되어 비교 조건이 어긋난다).
  Returns:
    keras.Sequential 블록 (4D 텐서에 직접 호출하거나 TimeDistributed 로 감싼다)
  """
  from tensorflow import keras
  from tensorflow.keras import layers

  conv = layers.Conv2DTranspose if transpose else layers.Conv2D
  return keras.Sequential([
    conv(c_out, 3, strides=stride, padding="same"),
    layers.GroupNormalization(groups=norm_groups(c_out, GN_GROUPS_S)),
    layers.LeakyReLU(0.2),
  ], name=name)


def inception_block(x, c_hid: int, c_out: int, name: str):
  """SimVP 의 gInception_ST: 1x1 conv 로 채널을 줄인 뒤 3/5/7 커널 병렬 conv 를 더한다.

  커널을 여러 개 쓰는 이유는 구름의 이동 거리(작은 이동~큰 이동)를 한 층에서 함께 보기 위해서다.
  공식 구현은 grouped conv(groups=8) 를 쓰지만 여기서는 일반 conv 를 쓰고
  GroupNormalization 의 그룹 수로만 8 을 유지한다 (설계 0.4 절).
  Args:
    x: (B, H, W, C_in) 입력 텐서
    c_hid: 1x1 conv 로 줄일 채널 수 (보통 hid_T // 2)
    c_out: 각 병렬 가지의 출력 채널 수 = 블록 출력 채널 수
  Returns:
    (B, H, W, c_out) 텐서
  """
  from tensorflow.keras import layers

  z = layers.Conv2D(c_hid, 1, padding="same", name=f"{name}_reduce")(x)
  groups = norm_groups(c_out, GN_GROUPS_T)
  branches = []
  for k in INCEP_KER:
    b = layers.Conv2D(c_out, k, padding="same", name=f"{name}_k{k}")(z)
    b = layers.GroupNormalization(groups=groups, name=f"{name}_k{k}_gn")(b)
    branches.append(layers.LeakyReLU(0.2, name=f"{name}_k{k}_act")(b))
  return layers.Add(name=f"{name}_sum")(branches)


def translator(z, hid_S: int, hid_T: int):
  """Mid_Xnet: Inception 블록 N_T 개로 인코딩하고 N_T 개로 디코딩한다 (U-Net 형 skip).

  첫 dec 블록은 skip 없이 마지막 enc 출력을 받고, 이후 dec 블록은 enc 출력을 역순으로 concat 한다.
  마지막 dec 블록의 출력 채널을 hid_S 로 두는 것이 우리 적응(T_out = 1, 다음 1장만 예측)이다.
  Args:
    z: (B, H/2, W/2, in_frames * hid_S) 시간을 채널로 접은 텐서
  Returns:
    (B, H/2, W/2, hid_S) 텐서
  """
  from tensorflow.keras import layers

  skips = []
  for i in range(N_T):
    z = inception_block(z, hid_T // 2, hid_T, f"mid_enc{i + 1}")
    if i < N_T - 1:
      skips.append(z)

  z = inception_block(z, hid_T // 2, hid_T, "mid_dec1")
  for i in range(1, N_T):
    z = layers.Concatenate(name=f"mid_dec{i + 1}_skip")([z, skips[-i]])
    c_out = hid_S if i == N_T - 1 else hid_T
    z = inception_block(z, hid_T // 2, c_out, f"mid_dec{i + 1}")
  return z


def build_model(in_frames: int, filters: int, h: int, w: int, lr: float = 1e-3):
  """SimVP (Encoder -> Translator -> Decoder) 잔차 모델 (Functional API).

  filters 는 공간 hidden 폭 hid_S 이고 시간 hidden 폭 hid_T 는 그 4배다 (공식 비율 64:256).
  Reshape 로 시간 축을 채널로 접기 때문에 h, w 를 고정해야 하지만, 가중치는 전부 conv 라
  크기와 무관하다. 추론 시 다른 크기로 다시 만들어 set_weights 하면 된다.
  """
  from tensorflow import keras
  from tensorflow.keras import layers

  hid_S, hid_T = filters, 4 * filters
  h2, w2 = (h + 1) // 2, (w + 1) // 2   # stride 2 conv 의 'same' 출력 = ceil(h / 2)

  inp = keras.Input(shape=(in_frames, h, w, 1))
  # Encoder: 프레임마다 같은 ConvSC 를 적용한다. 첫 블록 출력은 디코더 skip 으로 남긴다.
  enc1 = layers.TimeDistributed(conv_sc(hid_S, 1, False, "enc_sc1"), name="td_enc1")(inp)
  latent = layers.TimeDistributed(conv_sc(hid_S, 2, False, "enc_sc2"), name="td_enc2")(enc1)

  # Translator: (B, T, H2, W2, C) -> (B, H2, W2, T*C). 시간을 채널로 접어 2D conv 로 섞는다.
  z = layers.Permute((2, 3, 1, 4), name="to_hwtc")(latent)
  z = layers.Reshape((h2, w2, in_frames * hid_S), name="fold_time")(z)
  z = translator(z, hid_S, hid_T)

  # Decoder: 업샘플 후 마지막 입력 프레임의 enc1 을 붙여 고해상도 정보를 되살린다.
  hid = conv_sc(hid_S, 2, True, "dec_sc1")(z)
  # h 가 홀수면 transpose conv 가 h+1 을 내므로 잘라 맞춘다 (짝수면 파라미터 없는 no-op).
  hid = layers.Cropping2D(((0, h2 * 2 - h), (0, w2 * 2 - w)), name="dec_crop")(hid)
  take_last_frame = make_take_last_frame_layer()
  hid = layers.Concatenate(name="dec_skip")([hid, take_last_frame(name="enc1_last")(enc1)])
  hid = conv_sc(hid_S, 1, False, "dec_sc2")(hid)

  # mixed precision 에서도 Δ 와 잔차 합은 float32 로 유지한다 (손실 수치 안정성)
  delta = layers.Conv2D(1, 1, padding="same", activation=None, dtype="float32",
                        name=DELTA_LAYER_NAME)(hid)
  return compile_model(keras.Model(inputs=inp, outputs=residual_head(inp, delta)), lr)


def main(argv: list[str] | None = None) -> int:
  """CLI 진입점."""
  return main_for_model(build_model, MODEL_NAME,
                        "GK2A SW038 next-frame prediction (SimVP)", argv)


if __name__ == "__main__":
  sys.exit(main())
