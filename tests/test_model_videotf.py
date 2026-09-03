"""videotf_predict_colab.py (VideoTransformer 엔트리) 모델 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_model_videotf.py -q
학습(96x96)과 추론(250x250) 모델의 파라미터 수가 같고 가중치를 옮길 수 있는지,
250 -> 256 -> 250 pad/crop 이 정확히 되돌아오는지, 위치 인코딩에 학습 파라미터가 없는지,
잔차 head 와 mixed precision 출력 dtype 이 규약(0.4 절)대로인지를 본다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotf_predict_colab import (  # noqa: E402
  DELTA_LAYER_NAME,
  MODEL_NAME,
  PATCH,
  POS_LAYER_NAME,
  TOKEN_LAYER_NAME,
  FoldSpace,
  FoldTime,
  UnfoldSpace,
  UnfoldTime,
  build_model,
)


def _dtype_name(dtype) -> str:
  """dtype 이름을 문자열로 정규화한다 (Keras 2 는 tf.DType, Keras 3 는 str)."""
  return getattr(dtype, "name", None) or str(dtype)


def _token_count(model) -> int:
  """토큰 레이어 출력 (B, T, N, D) 의 N (= Hp * Wp)."""
  return int(model.get_layer(TOKEN_LAYER_NAME).output.shape[2])


class VideoTransformerModelTest(unittest.TestCase):
  """TensorFlow 가 필요한 모델 테스트 (CPU, 소형)."""

  def test_model_name(self) -> None:
    """모듈 상수가 결과 디렉터리 이름과 같다."""
    self.assertEqual(MODEL_NAME, "VideoTransformer")

  def test_output_shape_and_param_transfer(self) -> None:
    """입력 크기가 달라도 파라미터 수가 같고 가중치를 그대로 옮길 수 있다.

    위치 인코딩이 학습 파라미터가 아니라 상수라서 토큰 수(16 vs 36)가 달라도
    가중치 목록이 같다. 96x96 으로 학습해 250x250 으로 추론하는 전제가 이 성질이다.
    """
    small = build_model(4, 4, 32, 32)
    large = build_model(4, 4, 48, 48)
    self.assertEqual(small.count_params(), large.count_params())

    large.set_weights(small.get_weights())
    out = large.predict(np.zeros((1, 4, 48, 48, 1), np.float32), verbose=0)
    self.assertEqual(out.shape, (1, 48, 48, 1))
    self.assertEqual(out.dtype, np.float32)
    self.assertTrue(np.isfinite(out).all())

  def test_odd_size_250_roundtrip(self) -> None:
    """250 -> 256(pad) -> 250(crop). PATCH 의 배수가 아닌 크기도 원래 크기로 돌아온다."""
    model = build_model(4, 2, 250, 250)
    self.assertEqual(model.output_shape, (None, 250, 250, 1))

    # target 은 500 의 약수라 125 처럼 홀수도 올 수 있다 (125 -> 128 -> 125)
    odd = build_model(4, 2, 125, 125)
    self.assertEqual(odd.output_shape, (None, 125, 125, 1))
    self.assertEqual(odd.count_params(), model.count_params())

  def test_residual_identity_when_delta_zero(self) -> None:
    """Δ 를 만드는 readout 을 0 으로 두면 출력은 입력 마지막 프레임 그대로다."""
    model = build_model(2, 2, 16, 16)
    readout = model.get_layer(DELTA_LAYER_NAME)
    readout.set_weights([np.zeros_like(w) for w in readout.get_weights()])

    x = np.random.default_rng(0).random((1, 2, 16, 16, 1)).astype(np.float32)
    out = model.predict(x, verbose=0)
    np.testing.assert_allclose(out, x[:, -1], atol=1e-6)

  def test_initial_output_is_persistence(self) -> None:
    """가중치를 건드리지 않은 초기 모델의 출력 = 입력 마지막 프레임 (Δ readout 0 초기화)."""
    model = build_model(2, 2, 16, 16)
    x = np.random.default_rng(1).random((2, 2, 16, 16, 1)).astype(np.float32)
    pred = model.predict(x, verbose=0)
    np.testing.assert_allclose(pred, x[:, -1], atol=1e-6)

  def test_mixed_precision_output_float32(self) -> None:
    """mixed_float16 정책에서도 Δ 와 최종 출력은 float32 로 유지된다.

    출력 dtype 만 보면 부족하다. readout 의 dtype="float32" 를 빼도 residual_head 의
    Add(dtype="float32") 가 마지막에 캐스팅해 model.output 은 float32 로 남기 때문에,
    Δ 가 float16 으로 계산되는 회귀를 놓친다. 은닉층(patch embed · attention · MLP)을 함께 본다.
    """
    from tensorflow import keras

    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
      model = build_model(2, 2, 16, 16)

      # 정책이 실제로 걸렸는지 먼저 본다. 은닉층이 float32 면 이 테스트는 공허해진다.
      for name in ("patch_embed", "blk1_attn_s", "blk1_attn_t", "blk1_mlp1", "head_proj"):
        self.assertEqual(_dtype_name(model.get_layer(name).compute_dtype), "float16", name)

      # Δ(readout Conv2D) -> 잔차 합 -> 모델 출력은 전부 float32 여야 한다
      readout = model.get_layer(DELTA_LAYER_NAME)
      self.assertEqual(_dtype_name(readout.compute_dtype), "float32")
      self.assertEqual(_dtype_name(readout.output.dtype), "float32")
      self.assertEqual(_dtype_name(model.output.dtype), "float32")

      pred = model.predict(np.zeros((1, 2, 16, 16, 1), np.float32), verbose=0)
      self.assertEqual(pred.dtype, np.float32)
    finally:   # 전역 정책을 되돌리지 않으면 뒤따르는 테스트가 오염된다
      keras.mixed_precision.set_global_policy("float32")

  def test_fit_one_step(self) -> None:
    """공통 손실(ssim_mae_loss)로 1 epoch 학습이 유한한 loss 를 낸다."""
    rng = np.random.default_rng(42)
    x = rng.random((8, 4, 32, 32, 1)).astype(np.float32)
    y = rng.random((8, 32, 32, 1)).astype(np.float32)

    model = build_model(4, 2, 32, 32)
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
    self.assertTrue(np.isfinite(history.history["loss"][0]))

  def test_model_is_compiled_with_shared_loss(self) -> None:
    """공통 compile_model 로 네 모델이 같은 손실을 쓴다."""
    import nc_pipeline

    model = build_model(2, 2, 16, 16)
    self.assertIsNotNone(model.optimizer)
    self.assertIs(model.loss, nc_pipeline.ssim_mae_loss)

  def test_positional_encoding_has_no_weights(self) -> None:
    """위치 인코딩은 고정 상수라 학습 파라미터가 0 이다.

    학습 가능한 위치 임베딩이면 토큰 수가 달라질 때 shape 이 어긋나
    96x96 학습 -> 250x250 추론 가중치 이전이 불가능해진다.
    """
    model = build_model(4, 2, 96, 96)
    pos = model.get_layer(POS_LAYER_NAME)
    self.assertEqual(pos.count_params(), 0)
    self.assertEqual(pos.weights, [])

  def test_fold_unfold_roundtrip(self) -> None:
    """접기/펼치기 왕복이 항등이다.

    transpose 순서를 틀려도 shape 은 그대로 유효해서 다른 테스트가 잡지 못한다
    (Δ readout 이 zero-init 이라 persistence 계열은 본체를 지나가고, 학습 테스트는
    loss 의 유한성만 본다). 값으로 잠근다.
    """
    import tensorflow as tf

    b, t, n, d = 2, 3, 5, 4
    x = np.random.default_rng(7).random((b, t, n, d)).astype(np.float32)
    tensor = tf.constant(x)

    np.testing.assert_array_equal(UnfoldSpace(t)(FoldSpace()(tensor)).numpy(), x)
    np.testing.assert_array_equal(UnfoldTime(n)(FoldTime()(tensor)).numpy(), x)

  def test_fold_index_semantics(self) -> None:
    """접힌 배치 인덱스가 올바른 (b, t) / (b, n) 조합을 가리킨다.

    왕복만 보면 두 번 틀린 전치가 상쇄돼 통과할 수 있다. 접힌 상태에서
    공간 fold 는 b*T+t 가 원본 [b, t] 와, 시간 fold 는 b*N+n 이 [b, :, n] 과
    같아야 attention 이 "같은 프레임의 패치끼리" / "같은 위치의 시간열끼리" 섞는다.
    """
    import tensorflow as tf

    b, t, n, d = 2, 3, 5, 4
    x = np.random.default_rng(11).random((b, t, n, d)).astype(np.float32)
    tensor = tf.constant(x)

    folded_space = FoldSpace()(tensor).numpy()        # (B*T, N, D)
    self.assertEqual(folded_space.shape, (b * t, n, d))
    for bi in range(b):
      for ti in range(t):
        np.testing.assert_array_equal(folded_space[bi * t + ti], x[bi, ti],
                                      err_msg=f"space fold b={bi} t={ti}")

    folded_time = FoldTime()(tensor).numpy()          # (B*N, T, D)
    self.assertEqual(folded_time.shape, (b * n, t, d))
    for bi in range(b):
      for ni in range(n):
        np.testing.assert_array_equal(folded_time[bi * n + ni], x[bi, :, ni, :],
                                      err_msg=f"time fold b={bi} n={ni}")

  def test_pad_multiple_of_patch(self) -> None:
    """PATCH 배수로 pad 한 뒤의 토큰 수: 96 -> 12x12=144, 250 -> 32x32=1024."""
    self.assertEqual(PATCH, 8)
    self.assertEqual(_token_count(build_model(4, 2, 96, 96)), (96 // PATCH) ** 2)
    self.assertEqual(_token_count(build_model(4, 2, 250, 250)), (256 // PATCH) ** 2)


if __name__ == "__main__":
  unittest.main()
