"""simvp_predict_colab.py (SimVP 엔트리) 모델 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_model_simvp.py -q
학습(96x96)과 추론(250x250) 모델의 파라미터 수가 같고 가중치를 옮길 수 있는지,
250 -> 125 -> 250 다운/업샘플이 정확히 되돌아오는지, 잔차 head 와 mixed precision
출력 dtype 이 규약(0.4 절)대로인지를 본다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simvp_predict_colab import DELTA_LAYER_NAME, MODEL_NAME, build_model  # noqa: E402


def _dtype_name(dtype) -> str:
  """dtype 이름을 문자열로 정규화한다 (Keras 2 는 tf.DType, Keras 3 는 str)."""
  return getattr(dtype, "name", None) or str(dtype)


class SimVPModelTest(unittest.TestCase):
  """TensorFlow 가 필요한 모델 테스트 (CPU, 소형)."""

  def test_model_name(self) -> None:
    """모듈 상수가 결과 디렉터리 이름과 같다."""
    self.assertEqual(MODEL_NAME, "SimVP")

  def test_output_shape_and_param_transfer(self) -> None:
    """입력 크기가 달라도 파라미터 수가 같고 가중치를 그대로 옮길 수 있다."""
    small = build_model(4, 4, 32, 32)
    large = build_model(4, 4, 48, 48)
    self.assertEqual(small.count_params(), large.count_params())

    large.set_weights(small.get_weights())
    out = large.predict(np.zeros((1, 4, 48, 48, 1), np.float32), verbose=0)
    self.assertEqual(out.shape, (1, 48, 48, 1))
    self.assertEqual(out.dtype, np.float32)
    self.assertTrue(np.isfinite(out).all())

  def test_odd_size_250_roundtrip(self) -> None:
    """250 -> 125 -> 250. 다운샘플이 홀수 크기를 만들어도 원래 크기로 돌아온다."""
    model = build_model(4, 2, 250, 250)
    self.assertEqual(model.output_shape, (None, 250, 250, 1))

    # target 은 500 의 약수라 125 처럼 홀수도 올 수 있다 (63 -> 126 을 잘라 125 로 되돌린다)
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
    Δ 가 float16 으로 계산되는 회귀를 놓친다. 은닉층·readout 을 함께 본다.
    """
    from tensorflow import keras

    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
      model = build_model(2, 2, 16, 16)

      # 정책이 실제로 걸렸는지 먼저 본다. 은닉층이 float32 면 이 테스트는 공허해진다.
      for name in ("td_enc1", "mid_enc1_k3", "dec_sc2"):
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
    """공통 compile_model 로 세 모델이 같은 손실을 쓴다."""
    import nc_pipeline

    model = build_model(2, 2, 16, 16)
    self.assertIsNotNone(model.optimizer)
    self.assertIs(model.loss, nc_pipeline.ssim_mae_loss)


if __name__ == "__main__":
  unittest.main()
