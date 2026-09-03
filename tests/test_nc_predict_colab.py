"""nc_predict_colab.py 단위 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_nc_predict_colab.py -q
  /usr/local/bin/python3 -m unittest tests.test_nc_predict_colab -v
데이터 파일 없이 합성 배열만으로 순수 함수와 모델 구성을 검증한다.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nc_predict_colab as m  # noqa: E402


def _stamps(minutes: list[int]) -> list[datetime]:
  """2025-10-17 00:00 UTC 기준 분 오프셋 목록을 datetime 목록으로 만든다."""
  base = datetime(2025, 10, 17, 0, 0)
  return [base + timedelta(minutes=x) for x in minutes]


class FileListingTest(unittest.TestCase):
  """파일 목록·시각 파싱."""

  def test_list_nc_files_sorted_by_stamp_and_skips_bad_names(self) -> None:
    with tempfile.TemporaryDirectory() as d:
      names = [
        "gk2a_ami_le1b_sw038_la020ge_202510171000.nc",
        "gk2a_ami_le1b_sw038_la020ge_202510170958.nc",
        "gk2a_ami_le1b_sw038_la020ge_202510170800.nc",
        "gk2a_ami_le1b_sw038_la020ge_bad.nc",          # 시각 없음 -> 제외
        "gk2a_ami_le1b_ir105_la020ge_202510170000.nc",  # 다른 채널 -> glob 제외
      ]
      for n in names:
        (Path(d) / n).touch()
      files = m.list_nc_files(Path(d))
      self.assertEqual([Path(f).name for f in files], [names[2], names[1], names[0]])

  def test_parse_stamp(self) -> None:
    self.assertEqual(m.parse_stamp("x/gk2a_sw038_202510171500.nc"), datetime(2025, 10, 17, 15, 0))
    with self.assertRaises(ValueError):
      m.parse_stamp("x/gk2a_sw038_bad.nc")


class SegmentTest(unittest.TestCase):
  """결측 구간 분리·시간대 필터."""

  def test_find_segments_splits_on_gap(self) -> None:
    ts = _stamps([0, 2, 4, 16, 18, 20, 22])   # 4 -> 16 은 12분 점프
    self.assertEqual(m.find_segments(ts), [(0, 3), (3, 7)])

  def test_find_segments_single_frame(self) -> None:
    self.assertEqual(m.find_segments(_stamps([0])), [(0, 1)])

  def test_filter_hours_keeps_pairs_aligned(self) -> None:
    ts = _stamps([0, 2, 60, 62, 120])
    frames = np.arange(5, dtype=np.float32)[:, None, None] * np.ones((5, 2, 2), np.float32)
    f2, s2 = m.filter_hours(frames, ts, [1])
    self.assertEqual(len(s2), 2)
    self.assertTrue(all(s.hour == 1 for s in s2))
    np.testing.assert_array_equal(f2[:, 0, 0], [2.0, 3.0])
    self.assertIs(m.filter_hours(frames, ts, None)[0], frames)
    with self.assertRaises(ValueError):
      m.filter_hours(frames, ts, [23])


class DownsampleTest(unittest.TestCase):
  """블록 평균 풀링."""

  def test_block_mean(self) -> None:
    arr = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    out = m.downsample(arr, 2)
    self.assertEqual(out.shape, (2, 2, 2))
    self.assertAlmostEqual(float(out[0, 0, 0]), float(arr[0, :2, :2].mean()))
    self.assertAlmostEqual(float(out[1, 1, 1]), float(arr[1, 2:, 2:].mean()))

  def test_rejects_non_divisor(self) -> None:
    with self.assertRaises(ValueError):
      m.downsample(np.zeros((1, 500, 500), np.float32), 300)


class DatasetTest(unittest.TestCase):
  """윈도우·분할·패치 데이터셋."""

  def test_window_starts_respects_boundaries(self) -> None:
    self.assertEqual(m.window_starts([(0, 10), (10, 15), (15, 17)], 4),
                     list(range(0, 6)) + [10])

  def test_split_starts_leaves_gap_against_leakage(self) -> None:
    train, val = m.split_starts(list(range(100)), in_frames=4, ratio=0.8)
    self.assertEqual(len(train), 80)
    self.assertEqual(val[0], 84)
    # train 마지막 타깃(79+4=83) 이 val 첫 입력(84~87) 에 포함되지 않는다
    self.assertLess(train[-1] + 4, val[0])
    with self.assertRaises(ValueError):
      m.split_starts(list(range(5)), in_frames=4, ratio=0.8)

  def test_patch_grid_covers_right_edge(self) -> None:
    self.assertEqual(m.patch_grid(250, 96, 77), [0, 77, 154])
    self.assertEqual(m.patch_grid(250, 96, 96), [0, 96])
    with self.assertRaises(ValueError):
      m.patch_grid(50, 96, 77)

  def test_build_dataset_shapes_and_values(self) -> None:
    T, H, W, patch, stride, in_frames = 8, 20, 20, 8, 6, 4
    frames = np.random.default_rng(0).random((T, H, W), dtype=np.float32)
    X, Y = m.build_dataset(frames, [0, 1], in_frames, patch, stride)
    self.assertEqual(X.shape, (18, in_frames, patch, patch, 1))
    self.assertEqual(Y.shape, (18, patch, patch, 1))
    np.testing.assert_array_equal(X[0, ..., 0], frames[0:4, 0:8, 0:8])
    np.testing.assert_array_equal(Y[0, ..., 0], frames[4, 0:8, 0:8])
    # 두 번째 윈도우(w=1), 격자 마지막 칸(y=12, x=12)
    np.testing.assert_array_equal(X[17, ..., 0], frames[1:5, 12:20, 12:20])
    np.testing.assert_array_equal(Y[17, ..., 0], frames[5, 12:20, 12:20])

  def test_log_coverage_full(self) -> None:
    self.assertAlmostEqual(m.log_coverage(250, 250, 96, 77), 1.0)
    self.assertLess(m.log_coverage(250, 250, 96, 96), 0.6)


class NormalizeTest(unittest.TestCase):
  """전역 정규화."""

  def test_roundtrip(self) -> None:
    frames = np.array([[[14848.5, 16361.5]], [[15000.0, 15500.0]]], np.float32)
    n, gmin, gmax = m.normalize(frames)
    self.assertEqual((gmin, gmax), (14848.5, 16361.5))
    self.assertEqual(float(n.min()), 0.0)
    self.assertEqual(float(n.max()), 1.0)
    np.testing.assert_allclose(m.denormalize(n, gmin, gmax), frames, rtol=1e-6)
    with self.assertRaises(ValueError):
      m.normalize(np.ones((2, 2, 2), np.float32))


class ZipTest(unittest.TestCase):
  """Drive zip 추출."""

  def test_extract_only_nc_flattened(self) -> None:
    with tempfile.TemporaryDirectory() as d:
      zpath = Path(d) / "netcdf.zip"
      with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("netcdf/gk2a_ami_le1b_sw038_la020ge_202510170000.nc", b"abc")
        zf.writestr("netcdf/readme.txt", b"skip")
        zf.writestr("../evil_202510170002.nc", b"x")   # 경로 순회 시도 -> basename 만 사용
      dest = m.extract_zip(zpath, Path(d) / "out")
      names = sorted(p.name for p in dest.iterdir())
      self.assertEqual(names, ["evil_202510170002.nc", "gk2a_ami_le1b_sw038_la020ge_202510170000.nc"])
      self.assertFalse((Path(d) / "evil_202510170002.nc").exists())
    with self.assertRaises(FileNotFoundError):
      m.extract_zip(Path("/nonexistent.zip"), Path(d))


class ArgsTest(unittest.TestCase):
  """CLI 인자 파싱."""

  def test_defaults_and_overrides(self) -> None:
    cfg = m.parse_args(["--hours", "23", "5", "5", "99", "--target", "0", "--epochs", "2",
                        "--no-cache", "--no-mixed-precision"])
    self.assertEqual(cfg.hours, [5, 23])
    self.assertIsNone(cfg.target)
    self.assertEqual(cfg.epochs, 2)
    self.assertFalse(cfg.use_cache)
    self.assertFalse(cfg.mixed_precision)
    self.assertEqual(m.parse_args([]).hours, None)
    self.assertEqual(m.parse_args([]).target, 250)
    with self.assertRaises(ValueError):
      m.parse_args(["--epochs", "0"])


class ModelTest(unittest.TestCase):
  """TensorFlow 가 필요한 모델·지표 테스트 (CPU, 소형)."""

  def test_ssim_metric_identical_is_one(self) -> None:
    a = np.random.default_rng(1).random((3, 16, 16), dtype=np.float32)
    self.assertAlmostEqual(m.ssim_metric(a, a, chunk=2), 1.0, places=4)

  def test_loss_zero_for_identical(self) -> None:
    import tensorflow as tf

    y = tf.constant(np.random.default_rng(2).random((2, 16, 16, 1), dtype=np.float32))
    self.assertAlmostEqual(float(m.ssim_mae_loss(y, y)), 0.0, places=5)

  def test_weights_transfer_between_sizes(self) -> None:
    small = m.build_model(in_frames=2, filters=2, h=16, w=16)
    large = m.build_model(in_frames=2, filters=2, h=24, w=24)
    self.assertEqual(small.count_params(), large.count_params())
    large.set_weights(small.get_weights())
    out = large.predict(np.zeros((1, 2, 24, 24, 1), np.float32), verbose=0)
    self.assertEqual(out.shape, (1, 24, 24, 1))
    # 잔차 구조: 입력이 0 이면 출력은 Δ 와 같다 (마지막 프레임 0 + Δ)
    x = np.full((1, 2, 16, 16, 1), 0.5, np.float32)
    pred = small.predict(x, verbose=0)
    self.assertEqual(pred.shape, (1, 16, 16, 1))
    self.assertTrue(np.isfinite(pred).all())

  def test_persistence_baseline_on_synthetic(self) -> None:
    frames = np.stack([np.full((16, 16), 0.1 * t, np.float32) for t in range(5)])
    mae, ssim = m.persistence_baseline(frames, _stamps([0, 2, 4, 6, 8]), [(0, 5)])
    self.assertAlmostEqual(mae, 0.1, places=5)
    self.assertTrue(0.0 <= ssim <= 1.0)


if __name__ == "__main__":
  unittest.main()
