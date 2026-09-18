"""GK2A SW038 다음 프레임 예측 — VideoTransformer 엔트리 (공통 파이프라인: nc_pipeline.py).

데이터 적재·분할·손실·평가·기록은 전부 `nc_pipeline` 에 있고, 이 파일은 모델 구조만 정의한다.
ConvLSTM · SimVP · PredRNN-V2 와 같은 데이터·손실·지표로 비교하기 위한 구성이다.

모델: PredFormer 계열의 factorized space-time attention (Tang et al., 2024,
arXiv:2410.04733, 공식 구현 github.com/yyyujintang/PredFormer — README 에 라이선스 표기
없음, 2026-09-04 확인). RNN·CNN 없이 순수 transformer 로 영상을 예측하고, 3D attention 을
공간 축과 시간 축으로 쪼개(factorize) 비용을 줄인다.
우리 적응
  - 공식 Gated Transformer Block 의 게이팅(SwiGLU 계열) 대신 평범한 GELU MLP 를 쓴다.
  - 9가지 factorization 변형 중 BinaryST(블록마다 공간 -> 시간) 하나만 쓴다.
  - 학습형 위치 임베딩 대신 고정 sinusoidal 인코딩을 쓴다. 학습 파라미터가 없어야
    96x96 으로 학습한 가중치를 250x250 모델로 그대로 옮길 수 있기 때문이다.
  - 다음 1 프레임만 예측하므로 디코더 없이 마지막 시점 토큰만 읽어 Δ 를 만든다.
    readout 은 픽셀당 READOUT_CH(8) 채널을 낸다 (1채널이면 Δ conv 가 스칼라 gain 하나가 된다).
  - readout 은 hybrid conv head 다: transformer 특징에 마지막 입력 프레임의 conv 특징을 붙여 섞는다.
    8x8 패치 토큰의 선형 투영만으로는 픽셀 단위 국소 Δ 를 표현할 수 없어 Δ 가 입력 구조와
    정렬되지 않고 학습이 Δ=0 에 머물렀기 때문이다 (2026-09-04 실측, 계획 0.2 절 5번).
  - dim 128 / depth 4 / head 4 로 작게 잡았다 (공식은 과제별로 훨씬 크다).
  - 정규화·최적화는 논문 Table 8 의 WeatherBench 설정을 따른다: dropout 0.1, stochastic
    depth 0.25, AdamW(weight decay 1e-2), warmup 1 epoch 후 cosine, peak LR 5e-4.
    2026-09-18 실측에서 이것들이 전부 꺼진 채로 20 epoch 을 돌렸더니 epoch 13 에서
    train_loss 가 val_loss 아래로 내려갔다 (val 구간이 더 쉬운데도 역전 = 과적합).
    논문도 "작은 데이터셋에서 ViT 를 scratch 학습할 때는 dropout 과 stochastic depth 를
    함께 쓸 때 최고"라고 못박는다.

Colab 사용법
  1) 로컬에서 데이터 다운로드 (AWS Open Data, 익명 접근)
       python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out netcdf
  2) 결과 폴더(또는 zip)를 Google Drive 에 업로드
       nc_pipeline.COLAB_DATA_DIR (MyDrive/netcdf) 아래에 .nc 를 두거나
       zip 하나로 올린 뒤 --data-zip 으로 가리킨다 (COLAB_UNZIP_DIR 에 풀린다)
  3) Colab 메뉴: 런타임 > 런타임 유형 변경 > T4 GPU
  4) nc_pipeline.py 와 이 파일을 Colab 에 올린 뒤 셀에서 실행
       %run videotf_predict_colab.py                    # 그림이 셀에 바로 표시된다
       %run videotf_predict_colab.py --epochs 2 --hours 6 7 8 9 10 11
     노트북으로 쓰려면 `python3 tools/build_colab_notebook.py --model videotf --profile colab`.

  결과는 nc_pipeline.COLAB_OUT_DIR / "VideoTransformer" (로컬은 LOCAL_OUT_DIR / ...) 에
  metrics.json, 그림 4장, 가중치, train_log.csv 로 저장된다.
"""

from __future__ import annotations

import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

from nc_pipeline import (
  compile_model,
  delta_readout,
  main_for_model,
  make_take_last_frame_layer,
  residual_head,
)

MODEL_NAME = "VideoTransformer"

# 0.2 절 스펙 상수. filters(=폭) 만 CLI 로 바뀌고 DIM = 8 * filters 로 연결된다.
PATCH = 8            # 패치 한 변의 픽셀 수. 입력은 이 배수로 zero-pad 된다 (96->96, 250->256)
HEADS = 4            # MultiHeadAttention head 수 (DIM 을 나누어야 한다)
DEPTH = 4            # Transformer 블록 수
MLP_RATIO = 4        # MLP 은닉 폭 배수
DIM_PER_FILTER = 8   # DIM = DIM_PER_FILTER * filters (filters=16 -> DIM 128)

# 정규화·최적화 (논문 Table 8 의 WeatherBench 열. 기상 필드 예측이라 우리 과제와 가장 가깝다).
# 논문 결론: "dropout 과 stochastic depth 는 각각도 정규화 없는 경우보다 낫지만,
# 둘을 함께 쓸 때 최고의 결과를 준다" — 작은 데이터셋에서 ViT 를 scratch 학습할 때의 이야기다.
# 우리는 이 셋이 전부 꺼진 채로 20 epoch 을 돌렸고 epoch 13 에서 train_loss < val_loss 로
# 역전됐다 (val 구간이 더 쉬운데도 역전 = 과적합). 2026-09-18 실측.
DROPOUT = 0.1        # attention / MLP dropout
DROP_PATH = 0.25     # stochastic depth 최대 비율. 블록마다 0 -> 이 값까지 선형으로 올린다
PEAK_LR = 5e-4       # 논문 WeatherBench 학습률. warmup 1 epoch 뒤 cosine 으로 감쇠한다
WEIGHT_DECAY = 1e-2  # 논문: AdamW weight decay 1e-2
# Δ readout 이 보는 특징 채널 수. 1채널이면 zero-init delta_readout 이 스칼라 gain 하나가 되어
# gradient 부호가 배치마다 뒤집히고 본체로 전달되지 않는다 (2026-09-04 실측: 4 epoch loss 평평).
# ConvLSTM 16 · SimVP 16 · PredRNN-V2 4 채널과 같은 취지로 다채널을 준다.
READOUT_CH = 8
HEAD_CH = 16         # hybrid head 의 혼합 conv 폭 (transformer 특징 + 입력 conv 특징을 섞는다)

DELTA_LAYER_NAME = "delta"    # 잔차 Δ 를 내는 readout Conv2D. 테스트가 이름으로 찾는다
POS_LAYER_NAME = "posenc"     # 고정 위치 인코딩 레이어. 테스트가 파라미터 0 을 확인한다
TOKEN_LAYER_NAME = "tokens"   # (B, T, N, DIM) 토큰 레이어. 테스트가 N 을 확인한다


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------
def sinusoidal_1d(positions: np.ndarray, dim: int) -> np.ndarray:
  """1축 sin/cos 위치 인코딩 (P,) -> (P, dim).

  앞 절반이 sin, 뒤 절반이 cos 인 concat 형태다 (ViT/MAE 구현과 같은 배치).
  주파수는 1 / 10000^(i / (dim/2)) 로 낮은 축부터 지수적으로 촘촘해진다.
  Args:
    positions: 정수 위치 배열 (P,)
    dim: 출력 채널 수 (홀수면 마지막 한 칸은 0 으로 채운다)
  Returns:
    (P, dim) float32 배열
  """
  half = dim // 2
  if half == 0:   # dim <= 1 이면 인코딩할 주파수가 없다
    return np.zeros((len(positions), dim), dtype=np.float32)
  omega = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float64) / float(half)))
  angles = positions.reshape(-1, 1).astype(np.float64) * omega.reshape(1, -1)
  enc = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
  if enc.shape[1] < dim:
    enc = np.pad(enc, ((0, 0), (0, dim - enc.shape[1])))
  return enc.astype(np.float32)


def sinusoidal_2d(grid_h: int, grid_w: int, dim: int) -> np.ndarray:
  """2D 공간 위치 인코딩 (grid_h*grid_w, dim). 채널 절반씩 y 축·x 축에 쓴다.

  토큰 순서는 행 우선(row-major)이라 `Reshape((T, N, DIM))` 이 패치 격자를 펴는 순서와 같다.
  """
  ys, xs = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
  half = dim // 2
  return np.concatenate([sinusoidal_1d(ys.reshape(-1), half),
                         sinusoidal_1d(xs.reshape(-1), dim - half)], axis=1)


class SinusoidalPositionEncoding(keras.layers.Layer):
  """(B, T, N, D) 토큰에 고정 2D 공간 + 1D 시간 sinusoidal 인코딩을 더한다.

  상수는 `build()` 에서 static shape 로 만들고 `add_weight` 를 쓰지 않는다. 학습
  파라미터가 없어야 토큰 수가 다른 모델(96 -> 144, 250 -> 1024) 사이에서 `set_weights`
  가 성립한다. Lambda 는 바이트코드로 저장돼 이식이 어려우므로 Layer 로 감싼다.
  """

  def __init__(self, grid_h: int, grid_w: int, **kwargs) -> None:
    """grid_h, grid_w 는 패치 격자 크기 (N = grid_h * grid_w)."""
    super().__init__(**kwargs)
    self.grid_h = int(grid_h)
    self.grid_w = int(grid_w)

  def build(self, input_shape) -> None:
    """(1, 1, N, D) 공간 상수와 (1, T, 1, D) 시간 상수를 미리 더해 하나로 만든다."""
    _, steps, tokens, dim = input_shape
    if tokens != self.grid_h * self.grid_w:
      raise ValueError(f"토큰 수 {tokens} != grid {self.grid_h}x{self.grid_w}")
    space = sinusoidal_2d(self.grid_h, self.grid_w, int(dim))[None, None]
    time = sinusoidal_1d(np.arange(int(steps)), int(dim))[None, :, None]
    self.encoding = tf.constant(space + time, dtype=tf.float32)
    super().build(input_shape)

  @tf.autograph.experimental.do_not_convert
  def call(self, x, training=None):
    """상수 인코딩을 입력 dtype 으로 캐스팅해 더한다 (mixed_float16 에서는 float16)."""
    del training
    return x + tf.cast(self.encoding, x.dtype)

  def compute_output_shape(self, input_shape):
    """덧셈이라 shape 이 바뀌지 않는다."""
    return tuple(input_shape)

  def get_config(self) -> dict:
    """격자 크기를 직렬화한다."""
    return {**super().get_config(), "grid_h": self.grid_h, "grid_w": self.grid_w}


class FoldSpace(keras.layers.Layer):
  """(B, T, N, D) -> (B*T, N, D). 프레임마다 독립인 공간 attention 을 걸기 위한 접기."""

  @tf.autograph.experimental.do_not_convert
  def call(self, x, training=None):
    """시간 축을 배치로 밀어 넣는다. N, D 는 static 이라 -1 로 배치를 맡긴다."""
    del training
    return tf.reshape(x, [-1, x.shape[2], x.shape[3]])

  def compute_output_shape(self, input_shape):
    """배치와 시간이 합쳐지므로 첫 축은 알 수 없다."""
    return (None, input_shape[2], input_shape[3])


class UnfoldSpace(keras.layers.Layer):
  """(B*T, N, D) -> (B, T, N, D). FoldSpace 의 역연산."""

  def __init__(self, steps: int, **kwargs) -> None:
    """steps 는 시간 축 길이 T (static)."""
    super().__init__(**kwargs)
    self.steps = int(steps)

  @tf.autograph.experimental.do_not_convert
  def call(self, x, training=None):
    """T, N, D 가 static 이라 reshape 만으로 배치 축이 복원된다."""
    del training
    return tf.reshape(x, [-1, self.steps, x.shape[1], x.shape[2]])

  def compute_output_shape(self, input_shape):
    """시간 축을 되살린다."""
    return (None, self.steps, input_shape[1], input_shape[2])

  def get_config(self) -> dict:
    """steps 를 직렬화한다."""
    return {**super().get_config(), "steps": self.steps}


class FoldTime(keras.layers.Layer):
  """(B, T, N, D) -> (B*N, T, D). 토큰 위치마다 독립인 시간 attention 을 걸기 위한 접기."""

  @tf.autograph.experimental.do_not_convert
  def call(self, x, training=None):
    """(B, T, N, D) -> (B, N, T, D) 로 축을 바꾼 뒤 토큰 축을 배치로 민다."""
    del training
    return tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, x.shape[1], x.shape[3]])

  def compute_output_shape(self, input_shape):
    """남는 축은 (T, D)."""
    return (None, input_shape[1], input_shape[3])


class UnfoldTime(keras.layers.Layer):
  """(B*N, T, D) -> (B, T, N, D). FoldTime 의 역연산."""

  def __init__(self, tokens: int, **kwargs) -> None:
    """tokens 는 토큰 수 N (static)."""
    super().__init__(**kwargs)
    self.tokens = int(tokens)

  @tf.autograph.experimental.do_not_convert
  def call(self, x, training=None):
    """(B, N, T, D) 로 되돌린 뒤 다시 (B, T, N, D) 로 축을 바꾼다."""
    del training
    unfolded = tf.reshape(x, [-1, self.tokens, x.shape[1], x.shape[2]])
    return tf.transpose(unfolded, [0, 2, 1, 3])

  def compute_output_shape(self, input_shape):
    """토큰 축을 되살리고 시간 축을 앞으로 되돌린다."""
    return (None, input_shape[1], self.tokens, input_shape[2])

  def get_config(self) -> dict:
    """tokens 를 직렬화한다."""
    return {**super().get_config(), "tokens": self.tokens}


class DepthToSpace(keras.layers.Layer):
  """(B, H, W, C*b*b) -> (B, H*b, W*b, C). 패치 채널을 다시 픽셀 격자로 편다."""

  def __init__(self, block_size: int = PATCH, **kwargs) -> None:
    """block_size 는 패치 배수 (입력 채널이 block_size^2 로 나누어떨어져야 한다)."""
    super().__init__(**kwargs)
    self.block_size = int(block_size)

  @tf.autograph.experimental.do_not_convert
  def call(self, x, training=None):
    """채널을 공간 블록으로 편다 (제어 흐름이 없어 AutoGraph 변환이 필요 없다)."""
    del training
    return tf.nn.depth_to_space(x, self.block_size)

  def compute_output_shape(self, input_shape):
    """H, W 는 block_size 배가 되고 채널은 block_size^2 로 나뉜다."""
    batch, height, width, channels = input_shape
    b = self.block_size
    return (batch,
            None if height is None else height * b,
            None if width is None else width * b,
            None if channels is None else channels // (b * b))

  def get_config(self) -> dict:
    """block_size 를 직렬화한다."""
    return {**super().get_config(), "block_size": self.block_size}


class DropPath(keras.layers.Layer):
  """샘플 단위 stochastic depth. 학습 때만 잔차 가지를 통째로 확률 rate 로 끈다.

  Dropout 이 원소 하나하나를 끄는 것과 달리 배치의 샘플마다 가지 전체를 끄고,
  살아남은 샘플은 1/(1-rate) 로 키워 기댓값을 맞춘다. 깊은 residual 스택에서
  Dropout 보다 강한 정규화가 되고, 추론 때는 항등 함수다.
  학습 파라미터가 없어 96x96 -> 250x250 가중치 이전에 영향을 주지 않는다.
  """

  def __init__(self, rate: float, **kwargs) -> None:
    """rate 는 가지를 끌 확률 (0 이면 항등)."""
    super().__init__(**kwargs)
    self.rate = float(rate)

  @tf.autograph.experimental.do_not_convert
  def call(self, x, training=None):
    """배치 축만 랜덤인 0/1 마스크를 곱한다 (나머지 축은 1 로 브로드캐스트)."""
    if not training or self.rate <= 0.0:
      return x
    keep = 1.0 - self.rate
    shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
    mask = tf.floor(keep + tf.random.uniform(shape, dtype=x.dtype))
    return x / keep * mask

  def compute_output_shape(self, input_shape):
    """마스크 곱이라 shape 이 바뀌지 않는다."""
    return tuple(input_shape)

  def get_config(self) -> dict:
    """rate 를 직렬화한다."""
    return {**super().get_config(), "rate": self.rate}


def transformer_block(x, steps: int, tokens: int, dim: int, index: int,
                      drop_path: float = 0.0):
  """pre-LN factorized space-time 블록 하나 (공간 attention -> 시간 attention -> MLP).

  PredFormer 의 BinaryST 배열이다. 한 블록 안에서 공간을 먼저 섞고 시간을 섞으면
  3D attention (N*T)^2 대신 N^2 + T^2 비용으로 같은 수용영역을 얻는다.
  Args:
    x: (B, T, N, D) 토큰 텐서
    steps: 시간 축 길이 T (static)
    tokens: 토큰 수 N (static)
    dim: 채널 폭 D
    index: 블록 번호 (레이어 이름에 쓴다, 1부터)
    drop_path: 이 블록의 stochastic depth 비율 (세 잔차 가지에 모두 건다)
  Returns:
    (B, T, N, D) 텐서
  """
  from tensorflow.keras import layers

  key_dim = max(1, dim // HEADS)
  name = f"blk{index}"

  # 공간 attention: 프레임 안의 패치끼리 섞는다 (배치 = B*T)
  y = layers.LayerNormalization(name=f"{name}_ln_s")(x)
  y = FoldSpace(name=f"{name}_fold_s")(y)
  y = layers.MultiHeadAttention(HEADS, key_dim, dropout=DROPOUT,
                                name=f"{name}_attn_s")(y, y)
  y = UnfoldSpace(steps, name=f"{name}_unfold_s")(y)
  y = DropPath(drop_path, name=f"{name}_dp_s")(y)
  x = layers.Add(name=f"{name}_add_s")([x, y])

  # 시간 attention: 같은 위치의 패치를 시간축으로 섞는다 (배치 = B*N)
  y = layers.LayerNormalization(name=f"{name}_ln_t")(x)
  y = FoldTime(name=f"{name}_fold_t")(y)
  y = layers.MultiHeadAttention(HEADS, key_dim, dropout=DROPOUT,
                                name=f"{name}_attn_t")(y, y)
  y = UnfoldTime(tokens, name=f"{name}_unfold_t")(y)
  y = DropPath(drop_path, name=f"{name}_dp_t")(y)
  x = layers.Add(name=f"{name}_add_t")([x, y])

  # MLP: 채널 축만 섞는다 (Dense 는 마지막 축에 걸리므로 접을 필요가 없다)
  y = layers.LayerNormalization(name=f"{name}_ln_m")(x)
  y = layers.Dense(dim * MLP_RATIO, activation="gelu", name=f"{name}_mlp1")(y)
  y = layers.Dropout(DROPOUT, name=f"{name}_mlp_drop")(y)
  y = layers.Dense(dim, name=f"{name}_mlp2")(y)
  y = DropPath(drop_path, name=f"{name}_dp_m")(y)
  return layers.Add(name=f"{name}_add_m")([x, y])


def build_model(in_frames: int, filters: int, h: int, w: int,
                lr: float = 1e-3) -> keras.Model:
  """패치 임베딩 -> 고정 위치 인코딩 -> factorized attention 블록 -> hybrid conv head -> 잔차 모델.

  가중치는 토큰 수와 무관하므로(위치 인코딩이 상수) 96x96 으로 학습한 뒤 250x250
  모델에 set_weights 하면 된다. h, w 가 PATCH 의 배수가 아니면 zero-pad 했다가 마지막에
  잘라 원래 크기로 되돌린다 (250 -> 256 -> 250).
  """
  from tensorflow.keras import layers

  dim = DIM_PER_FILTER * filters
  pad_h, pad_w = -h % PATCH, -w % PATCH            # PATCH 배수까지 모자란 픽셀 수
  grid_h, grid_w = (h + pad_h) // PATCH, (w + pad_w) // PATCH
  tokens = grid_h * grid_w

  inp = keras.Input(shape=(in_frames, h, w, 1))
  # 패치 격자에 맞추려면 오른쪽·아래에만 0 을 덧댄다 (배수면 크기가 그대로다).
  x = layers.TimeDistributed(layers.ZeroPadding2D(((0, pad_h), (0, pad_w))), name="pad")(inp)
  x = layers.TimeDistributed(
    layers.Conv2D(dim, PATCH, strides=PATCH, padding="valid"), name="patch_embed")(x)
  x = layers.Reshape((in_frames, tokens, dim), name=TOKEN_LAYER_NAME)(x)
  x = SinusoidalPositionEncoding(grid_h, grid_w, name=POS_LAYER_NAME)(x)

  # stochastic depth 는 블록마다 0 -> DROP_PATH 로 선형 증가시킨다 (timm/ViT 관행).
  # 앞쪽 블록은 저수준 특징이라 통째로 끄면 손해가 크고, 뒤로 갈수록 여유가 있다.
  for i in range(1, DEPTH + 1):
    x = transformer_block(x, in_frames, tokens, dim, i,
                          DROP_PATH * (i - 1) / max(1, DEPTH - 1))

  # Readout (hybrid conv head): transformer 경로와 픽셀 해상도 입력 경로를 합친다.
  # transformer 경로 — 마지막 시점 토큰만 읽어 패치를 픽셀로 되돌린다 (다음 1 프레임만 예측한다).
  take_last_frame = make_take_last_frame_layer()
  y = take_last_frame(name="last_token")(x)
  y = layers.LayerNormalization(name="head_ln")(y)
  # 패치당 PATCH*PATCH 픽셀 × READOUT_CH 채널을 내고 depth_to_space 로 픽셀 격자를 편다
  # (depth_to_space 는 채널이 PATCH^2 의 배수여야 하므로 이 곱이 항상 조건을 만족한다).
  y = layers.Dense(PATCH * PATCH * READOUT_CH, name="head_proj")(y)
  y = layers.Reshape((grid_h, grid_w, PATCH * PATCH * READOUT_CH), name="head_grid")(y)
  y = DepthToSpace(PATCH, name="from_patch")(y)
  feat_tf = layers.Cropping2D(((0, pad_h), (0, pad_w)), name="head_crop")(y)

  # 입력 skip 경로 — 한 패치의 픽셀은 모두 같은 토큰의 선형 투영이라 픽셀 단위 국소 Δ 를
  # 표현하지 못한다. 마지막 입력 프레임에서 직접 뽑은 conv 특징을 붙여 Δ 가 입력의 국소
  # 구조와 정렬되게 한다 (다른 세 모델의 readout 입력이 conv 특징인 것과 같은 성질).
  feat_in = layers.Conv2D(READOUT_CH, 3, padding="same", activation="gelu",
                          name="skip_conv")(take_last_frame(name="last_frame")(inp))

  head = layers.Concatenate(name="head_concat")([feat_tf, feat_in])
  head = layers.Conv2D(HEAD_CH, 3, padding="same", activation="gelu", name="head_mix")(head)

  # Δ 는 0 초기화 readout 이라 학습 시작 시 출력 = 입력 마지막 프레임(Persistence)이다.
  delta = delta_readout(head, kernel_size=1, name=DELTA_LAYER_NAME)
  model = keras.Model(inputs=inp, outputs=residual_head(inp, delta))
  return compile_model(model, lr, optimizer=make_optimizer_adamw())


def make_optimizer_adamw():
  """논문 레시피의 AdamW. 다른 세 모델이 쓰는 공통 Adam 대신 이 모델에만 건다.

  학습률은 인자로 받은 `lr`(공통 기본값 1e-3) 대신 논문 WeatherBench 값 PEAK_LR 을 쓴다.
  실제 학습에서는 `nc_pipeline.make_warmup_cosine_callback` 이 이 값을 정점으로 삼아
  warmup -> cosine 으로 매 step 덮어쓴다 (여기 값은 정점 지정이 목적이다).
  """
  return keras.optimizers.AdamW(learning_rate=PEAK_LR, weight_decay=WEIGHT_DECAY)


def main(argv: list[str] | None = None) -> int:
  """CLI 진입점."""
  return main_for_model(build_model, MODEL_NAME,
                        "GK2A SW038 next-frame prediction (VideoTransformer)", argv,
                        extra={"lr_schedule": "warmup_cosine", "warmup_epochs": 1.0})


if __name__ == "__main__":
  sys.exit(main())
