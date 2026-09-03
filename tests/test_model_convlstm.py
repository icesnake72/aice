"""nc_predict_colab.py (ConvLSTM 엔트리) 모델 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_model_convlstm.py -q
학습(96x96)과 추론(250x250) 모델의 파라미터 수가 같고 가중치를 옮길 수 있는지,
잔차 head 가 붙어 출력 shape 이 (B, H, W, 1) 인지를 본다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nc_predict_colab import MODEL_NAME, build_model  # noqa: E402


class ConvLSTMModelTest(unittest.TestCase):
  """TensorFlow 가 필요한 모델 테스트 (CPU, 소형)."""

  def test_model_name(self) -> None:
    """모듈 상수가 결과 디렉터리 이름과 같다."""
    self.assertEqual(MODEL_NAME, "ConvLSTM")

  def test_weights_transfer_between_sizes(self) -> None:
    """96 학습 모델과 250 추론 모델의 가중치가 호환된다."""
    small = build_model(in_frames=2, filters=2, h=16, w=16)
    large = build_model(in_frames=2, filters=2, h=24, w=24)
    self.assertEqual(small.count_params(), large.count_params())
    large.set_weights(small.get_weights())
    out = large.predict(np.zeros((1, 2, 24, 24, 1), np.float32), verbose=0)
    self.assertEqual(out.shape, (1, 24, 24, 1))
    # 잔차 구조: 입력이 0 이면 출력은 Δ 와 같다 (마지막 프레임 0 + Δ)
    x = np.full((1, 2, 16, 16, 1), 0.5, np.float32)
    pred = small.predict(x, verbose=0)
    self.assertEqual(pred.shape, (1, 16, 16, 1))
    self.assertTrue(np.isfinite(pred).all())

  def test_model_is_compiled_with_shared_loss(self) -> None:
    """공통 compile_model 로 같은 손실이 걸린다."""
    import nc_pipeline

    model = build_model(in_frames=2, filters=2, h=16, w=16)
    self.assertIsNotNone(model.optimizer)
    self.assertIs(model.loss, nc_pipeline.ssim_mae_loss)


if __name__ == "__main__":
  unittest.main()
