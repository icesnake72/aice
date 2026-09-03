"""GK2A SW038 다음 프레임 예측 — PredRNN-V2 엔트리 (공통 파이프라인: nc_pipeline.py).

데이터 적재·분할·손실·평가·기록은 전부 `nc_pipeline` 에 있고, 이 파일은 모델 구조만 정의한다.
ConvLSTM · SimVP 와 같은 데이터·손실·지표로 비교하기 위한 구성이다.

모델: Wang et al., "PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive
Learning" (TPAMI 2022, arXiv:2103.09504). 공식 구현 github.com/thuml/predrnn-pytorch (MIT).
우리 적응
  - 다음 1 프레임만 예측하므로 reverse scheduled sampling 은 쓰지 않는다.
  - filter_size 는 공식 5 대신 3 (ConvLSTM 과 같은 수용영역 조건으로 맞추기 위해).
  - patch_size 2 (space-to-depth 로 96->48, 250->125, 채널 4) 로 시간 축 비용을 줄인다.
  - LayerNorm 은 쓰지 않는다 (Metal 에서의 호환·속도 문제). 대신 conv bias 를 살린다.

Colab 사용법
  1) 로컬에서 데이터 다운로드 (AWS Open Data, 익명 접근)
       python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out netcdf
  2) 결과 폴더(또는 zip)를 Google Drive 에 업로드
       nc_pipeline.COLAB_DATA_DIR (MyDrive/netcdf) 아래에 .nc 를 두거나
       zip 하나로 올린 뒤 --data-zip 으로 가리킨다 (COLAB_UNZIP_DIR 에 풀린다)
  3) Colab 메뉴: 런타임 > 런타임 유형 변경 > T4 GPU
  4) nc_pipeline.py 와 이 파일을 Colab 에 올린 뒤 셀에서 실행
       %run predrnn_v2_predict_colab.py                    # 그림이 셀에 바로 표시된다
       %run predrnn_v2_predict_colab.py --epochs 2 --hours 6 7 8 9 10 11
     노트북으로 쓰려면 `python3 tools/build_colab_notebook.py --model predrnn_v2 --profile colab`.

  결과는 nc_pipeline.COLAB_OUT_DIR / "PredRNN_V2" (로컬은 LOCAL_OUT_DIR / "PredRNN_V2") 에
  metrics.json, 그림 4장, 가중치, train_log.csv 로 저장된다.
"""

from __future__ import annotations

import sys

import tensorflow as tf
from tensorflow import keras

from nc_pipeline import (
  compile_model,
  delta_readout,
  main_for_model,
  residual_head,
)

MODEL_NAME = "PredRNN_V2"

# 0.4 절 스펙 상수. filters(=num_hidden) 만 CLI 로 바뀐다.
NUM_LAYERS = 2        # ST-LSTM 층 수
FILTER_SIZE = 3       # 셀 내부 conv 커널 (공식은 5, ConvLSTM 과 조건을 맞추려 3)
PATCH_SIZE = 2        # space-to-depth 배수. 96->48, 250->125, 채널 1 -> 4
DECOUPLE_BETA = 0.1   # decoupling loss 가중치 (공식 기본값)
FORGET_BIAS = 1.0     # f 게이트를 초기에 열어 두어 장기 기억을 살린다 (공식과 동일)


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------
class SpaceToDepth(keras.layers.Layer):
  """(B, H, W, C) -> (B, H/b, W/b, C*b*b). 해상도를 낮춰 순환 계산량을 줄인다.

  PredRNN 공식 구현의 `reshape_patch` 와 같은 역할이며, 정보 손실이 없어
  DepthToSpace 로 정확히 되돌릴 수 있다. Lambda 는 바이트코드로 저장돼 환경 간
  이식이 어려우므로 Layer 로 감싼다.
  """

  def __init__(self, block_size: int = PATCH_SIZE, **kwargs) -> None:
    """block_size 는 patch 배수 (입력 H, W 가 이 값으로 나누어떨어져야 한다)."""
    super().__init__(**kwargs)
    self.block_size = int(block_size)

  def call(self, x):
    """공간 블록을 채널로 접는다."""
    return tf.nn.space_to_depth(x, self.block_size)

  def compute_output_shape(self, input_shape):
    """H, W 는 block_size 로 나누고 채널은 block_size^2 배가 된다."""
    batch, height, width, channels = input_shape
    b = self.block_size
    return (batch,
            None if height is None else height // b,
            None if width is None else width // b,
            None if channels is None else channels * b * b)

  def get_config(self) -> dict:
    """block_size 를 직렬화한다."""
    return {**super().get_config(), "block_size": self.block_size}


class DepthToSpace(keras.layers.Layer):
  """(B, H, W, C*b*b) -> (B, H*b, W*b, C). SpaceToDepth 의 역연산."""

  def __init__(self, block_size: int = PATCH_SIZE, **kwargs) -> None:
    """block_size 는 patch 배수 (입력 채널이 block_size^2 로 나누어떨어져야 한다)."""
    super().__init__(**kwargs)
    self.block_size = int(block_size)

  def call(self, x):
    """채널을 다시 공간 블록으로 편다."""
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


class STLSTMCell(keras.layers.Layer):
  """PredRNN-V2 의 Spatiotemporal LSTM 셀 (공식 SpatioTemporalLSTMCell_v2 이식).

  일반 LSTM 의 시간 상태 c 와 층을 가로지르는 공간 기억 m 을 함께 갱신하고,
  두 상태의 증가분 Δc, Δm 을 함께 돌려준다 (V2 의 memory decoupling 손실용).

  서브레이어는 `__init__` 에서 만든다. Keras 3 는 `call` 안에서 처음 만든 레이어의
  가중치를 추적하지 못할 수 있어, Keras 2/3 양쪽에서 같은 순서로 가중치가 생기도록 한다.
  """

  def __init__(self, num_hidden: int, filter_size: int = FILTER_SIZE, **kwargs) -> None:
    """num_hidden 은 상태 채널 수, filter_size 는 내부 conv 커널 크기."""
    super().__init__(**kwargs)
    self.num_hidden = int(num_hidden)
    self.filter_size = int(filter_size)
    kernel = (self.filter_size, self.filter_size)
    hid = self.num_hidden
    # 공식은 LayerNorm 이 shift 를 담당해 bias=False 지만, 우리는 LayerNorm 을 빼므로
    # conv bias 를 남겨 게이트가 0 이 아닌 지점에서 시작할 수 있게 한다.
    self.conv_x = keras.layers.Conv2D(7 * hid, kernel, padding="same", name="conv_x")
    self.conv_h = keras.layers.Conv2D(4 * hid, kernel, padding="same", name="conv_h")
    self.conv_m = keras.layers.Conv2D(3 * hid, kernel, padding="same", name="conv_m")
    self.conv_o = keras.layers.Conv2D(hid, kernel, padding="same", name="conv_o")
    self.conv_last = keras.layers.Conv2D(hid, (1, 1), padding="same", name="conv_last")

  def build(self, input_shape) -> None:
    """서브레이어를 미리 build 한다 (가중치 생성 순서를 두 Keras 버전에서 고정)."""
    spatial = tuple(input_shape[:-1])
    hidden_shape = spatial + (self.num_hidden,)
    memory_shape = spatial + (2 * self.num_hidden,)
    self.conv_x.build(tuple(input_shape))
    self.conv_h.build(hidden_shape)
    self.conv_m.build(hidden_shape)
    self.conv_o.build(memory_shape)
    self.conv_last.build(memory_shape)
    super().build(input_shape)

  def call(self, x, h, c, m):
    """한 스텝 갱신. Returns: (h', c', m', Δc, Δm) 모두 (B, H, W, num_hidden)."""
    i_x, f_x, g_x, i_xm, f_xm, g_xm, o_x = tf.split(self.conv_x(x), 7, axis=-1)
    i_h, f_h, g_h, o_h = tf.split(self.conv_h(h), 4, axis=-1)
    i_m, f_m, g_m = tf.split(self.conv_m(m), 3, axis=-1)

    i_t = tf.sigmoid(i_x + i_h)
    f_t = tf.sigmoid(f_x + f_h + FORGET_BIAS)
    delta_c = i_t * tf.tanh(g_x + g_h)
    c_new = f_t * c + delta_c

    i_t2 = tf.sigmoid(i_xm + i_m)
    f_t2 = tf.sigmoid(f_xm + f_m + FORGET_BIAS)
    delta_m = i_t2 * tf.tanh(g_xm + g_m)
    m_new = f_t2 * m + delta_m

    mem = tf.concat([c_new, m_new], axis=-1)
    o_t = tf.sigmoid(o_x + o_h + self.conv_o(mem))
    h_new = o_t * tf.tanh(self.conv_last(mem))
    return h_new, c_new, m_new, delta_c, delta_m

  def compute_output_shape(self, input_shape):
    """5개 출력 모두 (B, H, W, num_hidden)."""
    state_shape = tuple(input_shape[:-1]) + (self.num_hidden,)
    return (state_shape,) * 5

  def get_config(self) -> dict:
    """num_hidden 과 filter_size 를 직렬화한다."""
    return {**super().get_config(),
            "num_hidden": self.num_hidden, "filter_size": self.filter_size}


class PredRNNV2Core(keras.layers.Layer):
  """ST-LSTM 층을 쌓아 시퀀스를 unroll 하고 마지막 스텝·마지막 층의 h 를 낸다.

  공간 기억 m 은 zigzag 로 흐른다: 한 스텝 안에서 층 0 -> 1 -> ... 로 올라가고,
  마지막 층이 갱신한 m 이 다음 스텝의 첫 층 입력이 된다.

  V2 의 memory decoupling: 매 스텝·매 층에서 Δc 와 Δm 을 공유 adapter(1x1 conv)로
  투영한 뒤 공간축으로 L2 정규화해 코사인 유사도를 재고, 그 절대값 평균을
  `add_loss` 로 학습 손실에 더한다. c 와 m 이 같은 변화를 중복 학습하지 않게 만드는 항이다.
  시간 축 unroll 길이는 static 이라 Python 루프로 편다 (in_frames 는 4로 짧다).
  """

  def __init__(self, num_hidden: int, num_layers: int = NUM_LAYERS,
               filter_size: int = FILTER_SIZE, decouple_beta: float = DECOUPLE_BETA,
               **kwargs) -> None:
    """셀 num_layers 개와 모든 층·Δc/Δm 이 공유하는 adapter 하나를 만든다."""
    super().__init__(**kwargs)
    self.num_hidden = int(num_hidden)
    self.num_layers = int(num_layers)
    self.filter_size = int(filter_size)
    self.decouple_beta = float(decouple_beta)
    self.cells = [STLSTMCell(self.num_hidden, self.filter_size, name=f"stlstm_cell_{i}")
                  for i in range(self.num_layers)]
    # 공식과 같이 adapter 는 층·Δc/Δm 을 통틀어 하나만 쓴다 (bias 없음).
    self.adapter = keras.layers.Conv2D(self.num_hidden, (1, 1), use_bias=False, name="adapter")

  def build(self, input_shape) -> None:
    """(B, T, H, W, C) 를 받아 셀 0 은 입력 채널로, 나머지 셀과 adapter 는 hidden 폭으로 build."""
    batch, _, height, width, channels = input_shape
    frame_shape = (batch, height, width, channels)
    hidden_shape = (batch, height, width, self.num_hidden)
    self.cells[0].build(frame_shape)
    for cell in self.cells[1:]:
      cell.build(hidden_shape)
    self.adapter.build(hidden_shape)
    super().build(input_shape)

  def _decouple_term(self, delta_c, delta_m):
    """Δc 와 Δm 의 채널별 코사인 유사도 절대값 평균 (float32 스칼라).

    mixed_float16 에서 L2 정규화가 언더플로하지 않도록 adapter 출력을 float32 로 올린다.
    """
    shape = tf.shape(delta_c)
    flat = tf.stack([shape[0], shape[1] * shape[2], self.num_hidden])
    d_c = tf.reshape(tf.cast(self.adapter(delta_c), tf.float32), flat)
    d_m = tf.reshape(tf.cast(self.adapter(delta_m), tf.float32), flat)
    cos = tf.reduce_sum(tf.math.l2_normalize(d_c, axis=1) * tf.math.l2_normalize(d_m, axis=1),
                        axis=1)
    return tf.reduce_mean(tf.abs(cos))

  def call(self, x):
    """(B, T, H, W, C) -> (B, H, W, num_hidden). decoupling loss 를 add_loss 로 등록한다."""
    steps, height, width = x.shape[1], x.shape[2], x.shape[3]
    # 상태는 0 초기화. batch 는 실행 시점에 정해지므로 tf.shape 로 읽는다.
    zero = tf.zeros(tf.stack([tf.shape(x)[0], height, width, self.num_hidden]), dtype=x.dtype)
    hidden = [zero] * self.num_layers
    cell_state = [zero] * self.num_layers
    memory = zero

    terms = []
    for t in range(steps):
      frame = x[:, t]
      for i, cell in enumerate(self.cells):
        cell_input = frame if i == 0 else hidden[i - 1]
        hidden[i], cell_state[i], memory, delta_c, delta_m = cell(
          cell_input, hidden[i], cell_state[i], memory)
        terms.append(self._decouple_term(delta_c, delta_m))
    self.add_loss(self.decouple_beta * tf.add_n(terms) / float(len(terms)))
    return hidden[-1]

  def compute_output_shape(self, input_shape):
    """시간 축을 소비하고 채널을 num_hidden 으로 바꾼다."""
    batch, _, height, width, _ = input_shape
    return (batch, height, width, self.num_hidden)

  def get_config(self) -> dict:
    """구조 상수를 직렬화한다."""
    return {**super().get_config(),
            "num_hidden": self.num_hidden, "num_layers": self.num_layers,
            "filter_size": self.filter_size, "decouple_beta": self.decouple_beta}


def build_model(in_frames: int, filters: int, h: int, w: int,
                lr: float = 1e-3) -> keras.Model:
  """SpaceToDepth -> PredRNN-V2 core -> DepthToSpace -> Conv2D(Δ) -> 잔차 모델.

  가중치는 입력 크기와 무관하므로 96x96 으로 학습한 뒤 250x250 모델에 set_weights 하면 된다.
  h, w 는 PATCH_SIZE 로 나누어떨어져야 한다 (96 -> 48, 250 -> 125).
  """
  from tensorflow.keras import layers

  inp = keras.Input(shape=(in_frames, h, w, 1))
  packed = layers.TimeDistributed(SpaceToDepth(PATCH_SIZE), name="to_patch")(inp)
  feat = PredRNNV2Core(filters, NUM_LAYERS, FILTER_SIZE, DECOUPLE_BETA, name="core")(packed)
  # patch 채널(=PATCH_SIZE^2)로 되돌린 뒤 원해상도로 펴고 마지막에 1채널 Δ 를 만든다.
  y = layers.Conv2D(PATCH_SIZE * PATCH_SIZE, (1, 1), padding="same", name="to_pixel")(feat)
  y = DepthToSpace(PATCH_SIZE, name="from_patch")(y)
  # Δ 는 0 초기화 readout 이라 학습 시작 시 출력 = 입력 마지막 프레임(Persistence)이다.
  delta = delta_readout(y, kernel_size=1, name="delta")
  return compile_model(keras.Model(inputs=inp, outputs=residual_head(inp, delta)), lr)


def main(argv: list[str] | None = None) -> int:
  """CLI 진입점."""
  return main_for_model(build_model, MODEL_NAME,
                        "GK2A SW038 next-frame prediction (PredRNN-V2)", argv)


if __name__ == "__main__":
  sys.exit(main())
