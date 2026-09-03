"""predrnn_v2_predict_colab.py (PredRNN-V2 엔트리) 모델 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_model_predrnn_v2.py -q
학습(96x96)과 추론(250x250) 모델의 파라미터 수가 같고 가중치를 옮길 수 있는지,
잔차 head 로 출력이 (B, H, W, 1) float32 인지, decoupling loss 가 `add_loss` 로
등록돼 실제 학습 손실에 더해지는지를 본다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predrnn_v2_predict_colab import (  # noqa: E402
  DECOUPLE_BETA,
  MODEL_NAME,
  DepthToSpace,
  PredRNNV2Core,
  STLSTMCell,
  SpaceToDepth,
  build_model,
)


def _dtype_name(dtype) -> str:
  """dtype 이름을 문자열로 정규화한다 (Keras 2 는 tf.DType, Keras 3 는 str)."""
  return getattr(dtype, "name", None) or str(dtype)


def _synthetic(n: int, in_frames: int, size: int) -> tuple[np.ndarray, np.ndarray]:
  """재현 가능한 합성 (X, Y) 를 만든다. 값은 정규화 후 범위인 [0, 1] 안에 둔다."""
  rng = np.random.default_rng(0)
  x = rng.random((n, in_frames, size, size, 1), dtype=np.float32)
  y = rng.random((n, size, size, 1), dtype=np.float32)
  return x, y


class PredRNNV2ModelTest(unittest.TestCase):
  """TensorFlow 가 필요한 모델 테스트 (CPU, 소형)."""

  def test_model_name(self) -> None:
    """모듈 상수가 결과 디렉터리 이름과 같다."""
    self.assertEqual(MODEL_NAME, "PredRNN_V2")

  def test_stlstm_cell_shapes(self) -> None:
    """셀은 (h', c', m', Δc, Δm) 5개를 모두 (B, H, W, hid) 로 돌려준다.

    입력을 0 으로 주면 bias 가 zeros 라 출력도 정확히 0 이 되어 finite 단언이 공허해진다.
    난수를 넣어 gate 조합·decoupling 항까지 실제 값으로 지나가게 한다.
    """
    import tensorflow as tf

    rng = np.random.default_rng(0)
    cell = STLSTMCell(8, 3)
    x = tf.constant(rng.standard_normal((2, 16, 16, 4), dtype=np.float32))
    h = tf.constant(rng.standard_normal((2, 16, 16, 8), dtype=np.float32))
    c = tf.constant(rng.standard_normal((2, 16, 16, 8), dtype=np.float32))
    m = tf.constant(rng.standard_normal((2, 16, 16, 8), dtype=np.float32))
    outs = cell(x, h, c, m)
    self.assertGreater(float(tf.reduce_max(tf.abs(outs[0]))), 0.0)
    self.assertEqual(len(outs), 5)
    for tensor in outs:
      self.assertEqual(tuple(tensor.shape), (2, 16, 16, 8))
      self.assertTrue(np.isfinite(tensor.numpy()).all())

  def test_space_depth_roundtrip(self) -> None:
    """DepthToSpace(SpaceToDepth(x)) 가 원본과 같다 (patch 앞뒤 처리의 무손실 확인)."""
    import tensorflow as tf

    x = tf.constant(np.arange(2 * 8 * 8 * 1, dtype=np.float32).reshape(2, 8, 8, 1))
    packed = SpaceToDepth(2)(x)
    self.assertEqual(tuple(packed.shape), (2, 4, 4, 4))
    back = DepthToSpace(2)(packed)
    np.testing.assert_allclose(back.numpy(), x.numpy())

  def test_output_shape_and_param_transfer(self) -> None:
    """96 학습 모델과 250 추론 모델처럼 크기가 달라도 가중치가 호환된다."""
    small = build_model(4, 4, 32, 32)
    large = build_model(4, 4, 48, 48)
    self.assertEqual(small.count_params(), large.count_params())
    large.set_weights(small.get_weights())
    out = large.predict(np.zeros((1, 4, 48, 48, 1), np.float32), verbose=0)
    self.assertEqual(out.shape, (1, 48, 48, 1))
    self.assertEqual(out.dtype, np.float32)
    self.assertTrue(np.isfinite(out).all())

  def test_odd_size_250_roundtrip(self) -> None:
    """추론 해상도 250 이 patch 2 로 125 를 거쳐 250 으로 정확히 돌아온다."""
    model = build_model(4, 2, 250, 250)
    self.assertEqual(tuple(model.output_shape), (None, 250, 250, 1))

  def test_residual_identity_when_delta_zero(self) -> None:
    """readout 이 0 이면 출력 = 입력 마지막 프레임 (잔차 head 가 붙어 있다)."""
    model = build_model(4, 2, 32, 32)
    readout = model.get_layer("delta")
    readout.set_weights([np.zeros_like(w) for w in readout.get_weights()])
    x, _ = _synthetic(2, 4, 32)
    pred = model.predict(x, verbose=0)
    np.testing.assert_allclose(pred, x[:, -1], atol=1e-6)

  def test_initial_output_is_persistence(self) -> None:
    """가중치를 건드리지 않은 초기 모델의 출력 = 입력 마지막 프레임 (Δ readout 0 초기화)."""
    model = build_model(4, 2, 32, 32)
    x, _ = _synthetic(2, 4, 32)
    pred = model.predict(x, verbose=0)
    np.testing.assert_allclose(pred, x[:, -1], atol=1e-6)

  def test_mixed_precision_output_float32(self) -> None:
    """mixed_float16 정책에서도 Δ·출력·decoupling loss 는 float32 로 유지된다.

    출력 dtype 만 보면 잔차 head 의 Add(dtype="float32") 가 마지막에 캐스팅해 주기 때문에
    readout 이 float16 으로 계산되는 회귀를 놓친다. 그래서 은닉층이 float16 인지부터 본다.
    """
    from tensorflow import keras

    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
      model = build_model(4, 2, 32, 32)
      core = model.get_layer("core")
      readout = model.get_layer("delta")

      # 정책이 실제로 걸렸는지 먼저 본다. 은닉층이 float32 면 이 테스트는 공허해진다.
      self.assertEqual(_dtype_name(core.compute_dtype), "float16")
      self.assertEqual(_dtype_name(core.cells[0].conv_x.compute_dtype), "float16")
      self.assertEqual(_dtype_name(model.get_layer("to_pixel").compute_dtype), "float16")

      # Δ(readout Conv2D) -> 잔차 합 -> 모델 출력은 전부 float32 여야 한다
      self.assertEqual(_dtype_name(readout.compute_dtype), "float32")
      self.assertEqual(_dtype_name(readout.output.dtype), "float32")
      self.assertEqual(_dtype_name(model.output.dtype), "float32")

      x, _ = _synthetic(2, 4, 32)
      model(x)
      self.assertTrue(model.losses)
      for loss in model.losses:   # decoupling 항은 float16 언더플로를 피해 float32 로 잰다
        self.assertEqual(_dtype_name(loss.dtype), "float32")
      self.assertEqual(model.predict(x, verbose=0).dtype, np.float32)
    finally:   # 전역 정책을 되돌리지 않으면 뒤따르는 테스트가 오염된다
      keras.mixed_precision.set_global_policy("float32")

  def test_decouple_loss_registered(self) -> None:
    """모델을 한 번 호출하면 decoupling loss 가 model.losses 에 잡힌다."""
    model = build_model(4, 4, 32, 32)
    x, _ = _synthetic(2, 4, 32)
    model(x)
    self.assertTrue(model.losses)
    values = [float(v) for v in model.losses]
    for value in values:
      self.assertTrue(np.isfinite(value))
      self.assertGreaterEqual(value, 0.0)
    # cos 의 절대값 평균 <= 1 이므로 항 하나의 상한은 decouple_beta 다.
    self.assertLessEqual(sum(values), DECOUPLE_BETA + 1e-6)

  def test_decouple_loss_added_to_fit_loss(self) -> None:
    """학습 손실 = ssim_mae_loss + decoupling loss 로, 단독 손실보다 크다.

    lr=0 으로 가중치를 고정해 fit 전후의 예측이 같도록 만든 뒤 두 값을 비교한다.
    """
    import nc_pipeline

    model = build_model(4, 4, 32, 32, lr=0.0)
    x, y = _synthetic(8, 4, 32)
    fit_loss = model.fit(x, y, epochs=1, batch_size=8, verbose=0).history["loss"][0]
    base = float(nc_pipeline.ssim_mae_loss(y, model.predict(x, verbose=0)))
    self.assertTrue(np.isfinite(fit_loss))
    self.assertGreater(fit_loss, base + 1e-6)
    self.assertLess(fit_loss, base + DECOUPLE_BETA + 1e-6)

  def test_fit_one_step(self) -> None:
    """합성 데이터 1 epoch 학습이 finite loss 로 끝난다."""
    model = build_model(4, 4, 32, 32)
    x, y = _synthetic(8, 4, 32)
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
    self.assertTrue(np.isfinite(history.history["loss"][0]))

  def test_core_is_compiled_with_shared_loss(self) -> None:
    """공통 compile_model 로 같은 손실이 걸리고 core 설정이 스펙과 맞는다."""
    import nc_pipeline

    model = build_model(4, 4, 32, 32)
    self.assertIsNotNone(model.optimizer)
    self.assertIs(model.loss, nc_pipeline.ssim_mae_loss)
    core = next(layer for layer in model.layers if isinstance(layer, PredRNNV2Core))
    self.assertEqual(len(core.cells), 2)
    self.assertEqual(core.num_hidden, 4)
    self.assertEqual(core.filter_size, 3)


if __name__ == "__main__":
  unittest.main()
