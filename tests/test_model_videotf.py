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
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videotf_predict_colab import (  # noqa: E402
  DELTA_LAYER_NAME,
  MODEL_NAME,
  PATCH,
  POS_LAYER_NAME,
  READOUT_CH,
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


def _reset_keras2_seed_generator() -> None:
  """`keras.utils.set_random_seed` 가 남긴 Keras 2 전역 시드 상태를 되돌린다.

  Keras 2 는 전역 시드가 걸리면 레이어 초기화 시드를 Python `random` 으로 뽑고,
  그 경로가 `random.randint(1, 1e9)` 를 호출해 Python 3.11 이 변수마다
  DeprecationWarning 을 낸다. 되돌리지 않으면 이 테스트 이후에 만드는 모든 모델까지
  경고를 쏟아낸다 (mixed precision 테스트가 정책을 finally 로 복구하는 것과 같은 이유).
  Keras 3 에는 이 내부 경로가 없어 조용히 넘어간다.
  """
  try:
    import keras.src.backend as keras_backend
  except ImportError:
    return
  holder = getattr(keras_backend, "_SEED_GENERATOR", None)
  try:
    del holder.generator
  except AttributeError:   # Keras 3 이거나 애초에 설정되지 않았다
    pass


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

  def test_initial_output_is_near_persistence(self) -> None:
    """초기 모델의 출력이 입력 마지막 프레임 "근처" 에서 출발한다.

    다른 세 모델과 달리 Δ readout 커널을 작은 난수(DELTA_INIT_STD)로 시작하므로
    출력이 Persistence 와 정확히 같지는 않다. bias 는 그대로 0 이라 편향은 없고,
    Δ 가 작은 범위에 머물러야 학습이 Persistence 근처에서 출발한다는 계약이 유지된다.
    """
    model = build_model(2, 2, 16, 16)
    x = np.random.default_rng(1).random((2, 2, 16, 16, 1)).astype(np.float32)
    delta = model.predict(x, verbose=0) - x[:, -1]
    self.assertLess(float(np.abs(delta).mean()), 0.02)
    self.assertLess(float(np.abs(delta).max()), 0.2)

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

  def test_model_actually_learns(self) -> None:
    """합성 과제로 loss 가 실제로 내려간다 (readout 채널 수 회귀 방지).

    2026-09-04 실측: readout 입력이 1채널이면 zero-init `delta_readout` 이 스칼라
    gain 하나가 되어 gradient 부호가 배치마다 뒤집히고, gain 이 0 근처를 맴돌아
    본체 110개 변수의 gradient 가 정확히 0 이 된다. 실데이터 4 epoch 동안 loss 가
    평평했다(val MAE = Persistence). 다른 테스트는 이 결함을 잡지 못한다 —
    Δ 가 zero-init 이라 persistence 계열은 통과하고, `test_fit_one_step` 은
    loss 의 유한성만 본다.

    타깃은 "마지막 프레임 + 고정 공간 패턴" 이라 위치 인코딩을 가진 본체가
    풀 수 있는 과제다. 40 step (32샘플 / batch 8 x 10 epoch) 이면 충분하다.
    """
    from tensorflow import keras

    self.assertGreater(READOUT_CH, 1, "readout 입력이 1채널이면 학습이 시작되지 않는다")
    # 40 step 은 초기값에 민감해 전역 시드를 고정한다. step 수를 늘리면 1채널 모델도
    # 결국 이 합성 과제를 풀어버려 회귀 검출력이 사라진다 (80 step 실측: 1채널도 35~70% 감소).
    keras.utils.set_random_seed(0)
    try:
      rng = np.random.default_rng(3)
      x = rng.random((32, 4, 16, 16, 1)).astype(np.float32)
      grid_y, grid_x = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
      pattern = (0.1 * np.sin(grid_y / 3.0) * np.cos(grid_x / 4.0)).astype(np.float32)
      y = np.clip(x[:, -1] + pattern[None, :, :, None], 0.0, 1.0)

      with warnings.catch_warnings():
        # 위 시드 때문에 Keras 2 초기화가 변수마다 DeprecationWarning 을 낸다 (Keras 내부).
        warnings.simplefilter("ignore", DeprecationWarning)
        model = build_model(4, 2, 16, 16)
        body_before = model.get_layer("blk1_mlp1").get_weights()[0].copy()
        losses = model.fit(x, y, epochs=10, batch_size=8, verbose=0).history["loss"]
    finally:   # 전역 시드를 되돌리지 않으면 뒤따르는 테스트가 경고로 덮인다
      _reset_keras2_seed_generator()

    # 1) 손실이 유의미하게 내려간다. 이것이 1채널 회귀를 가르는 단언이다
    #    (실측: 8채널 Keras 2 58.7% / Keras 3 49.4%, 1채널 5.3% / 10.7%).
    drop = 1.0 - losses[-1] / losses[0]
    self.assertGreater(drop, 0.20, f"loss 가 거의 안 내려갔다: {losses}")

    # 2) Δ readout 이 0 에서 벗어났고 본체 가중치도 갱신됐다 (sanity check)
    delta_kernel = model.get_layer(DELTA_LAYER_NAME).get_weights()[0]
    self.assertGreater(float(np.linalg.norm(delta_kernel)), 1e-3)
    body_after = model.get_layer("blk1_mlp1").get_weights()[0]
    self.assertGreater(float(np.abs(body_after - body_before).max()), 1e-4)

  def test_readout_feature_channels(self) -> None:
    """Δ conv 직전 특징 맵이 READOUT_CH 채널이다 (depth_to_space 채널 계산 포함)."""
    model = build_model(4, 2, 32, 32)
    self.assertEqual(model.get_layer("head_crop").output.shape[-1], READOUT_CH)
    self.assertEqual(model.get_layer("head_proj").output.shape[-1], PATCH * PATCH * READOUT_CH)

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
