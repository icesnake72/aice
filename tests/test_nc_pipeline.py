"""nc_pipeline.py (모델 공통 파이프라인) 단위 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_nc_pipeline.py -q
  /usr/local/bin/python3 -m unittest tests.test_nc_pipeline -v
데이터 파일 없이 합성 배열만으로 순수 함수·잔차 head·metrics 기록을 검증한다.
모델 구조 자체는 tests/test_model_*.py 에서 따로 본다.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nc_pipeline as m  # noqa: E402


def _stamps(minutes: list[int]) -> list[datetime]:
  """2025-10-17 00:00 UTC 기준 분 오프셋 목록을 datetime 목록으로 만든다."""
  base = datetime(2025, 10, 17, 0, 0)
  return [base + timedelta(minutes=x) for x in minutes]


class FileListingTest(unittest.TestCase):
  """파일 목록·시각 파싱."""

  def test_list_nc_files_sorted_by_stamp_and_skips_bad_names(self) -> None:
    """관측 시각 오름차순 정렬 + 형식 불일치 파일 제외."""
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
    """파일명 끝 12자리를 datetime 으로 파싱한다."""
    self.assertEqual(m.parse_stamp("x/gk2a_sw038_202510171500.nc"), datetime(2025, 10, 17, 15, 0))
    with self.assertRaises(ValueError):
      m.parse_stamp("x/gk2a_sw038_bad.nc")


class SegmentTest(unittest.TestCase):
  """결측 구간 분리·시간대 필터."""

  def test_find_segments_splits_on_gap(self) -> None:
    """2분 간격이 깨지는 지점에서 구간을 끊는다."""
    ts = _stamps([0, 2, 4, 16, 18, 20, 22])   # 4 -> 16 은 12분 점프
    self.assertEqual(m.find_segments(ts), [(0, 3), (3, 7)])

  def test_find_segments_single_frame(self) -> None:
    """프레임 1장이면 구간도 1개다."""
    self.assertEqual(m.find_segments(_stamps([0])), [(0, 1)])

  def test_filter_hours_keeps_pairs_aligned(self) -> None:
    """frames 와 stamps 가 같은 인덱스로 걸러진다."""
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
    """블록 평균 풀링 값이 원본 블록 평균과 같다."""
    arr = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    out = m.downsample(arr, 2)
    self.assertEqual(out.shape, (2, 2, 2))
    self.assertAlmostEqual(float(out[0, 0, 0]), float(arr[0, :2, :2].mean()))
    self.assertAlmostEqual(float(out[1, 1, 1]), float(arr[1, 2:, 2:].mean()))

  def test_rejects_non_divisor(self) -> None:
    """약수가 아닌 target 은 크롭이 되므로 거부한다."""
    with self.assertRaises(ValueError):
      m.downsample(np.zeros((1, 500, 500), np.float32), 300)


class DatasetTest(unittest.TestCase):
  """윈도우·분할·패치 데이터셋."""

  def test_window_starts_respects_boundaries(self) -> None:
    """윈도우가 세그먼트 경계를 넘지 않는다."""
    self.assertEqual(m.window_starts([(0, 10), (10, 15), (15, 17)], 4),
                     list(range(0, 6)) + [10])

  def test_split_starts_leaves_gap_against_leakage(self) -> None:
    """val 첫 입력이 train 마지막 타깃을 포함하지 않는다."""
    train, val = m.split_starts(list(range(100)), in_frames=4, ratio=0.8)
    self.assertEqual(len(train), 80)
    self.assertEqual(val[0], 84)
    # train 마지막 타깃(79+4=83) 이 val 첫 입력(84~87) 에 포함되지 않는다
    self.assertLess(train[-1] + 4, val[0])
    with self.assertRaises(ValueError):
      m.split_starts(list(range(5)), in_frames=4, ratio=0.8)

  def test_patch_grid_covers_right_edge(self) -> None:
    """격자 마지막 패치가 오른쪽 끝을 덮는다."""
    self.assertEqual(m.patch_grid(250, 96, 77), [0, 77, 154])
    self.assertEqual(m.patch_grid(250, 96, 96), [0, 96])
    with self.assertRaises(ValueError):
      m.patch_grid(50, 96, 77)

  def test_build_dataset_shapes_and_values(self) -> None:
    """(X, Y) shape 와 실제 슬라이스 값이 맞는지 본다."""
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
    """stride 77 이면 커버리지 100%, 96 이면 60% 미만이다."""
    self.assertAlmostEqual(m.log_coverage(250, 250, 96, 77), 1.0)
    self.assertLess(m.log_coverage(250, 250, 96, 96), 0.6)


class NormalizeTest(unittest.TestCase):
  """전역 정규화."""

  def test_roundtrip(self) -> None:
    """정규화 -> 역정규화가 원본을 복원한다."""
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
    """.nc 만 basename 으로 평탄하게 푼다 (경로 순회 차단)."""
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
  """CLI 인자 파싱 (build_arg_parser + config_from_args)."""

  def _cfg(self, argv: list[str]) -> m.Config:
    """argv -> Config 지름길."""
    return m.config_from_args(m.build_arg_parser("test").parse_args(argv))

  def test_config_from_args_defaults(self) -> None:
    """기본값·중복/범위 밖 hours 정리·검증 실패를 확인한다."""
    cfg = self._cfg(["--hours", "23", "5", "5", "99", "--target", "0", "--epochs", "2",
                     "--no-cache", "--no-mixed-precision"])
    self.assertEqual(cfg.hours, [5, 23])
    self.assertIsNone(cfg.target)
    self.assertEqual(cfg.epochs, 2)
    self.assertFalse(cfg.use_cache)
    self.assertFalse(cfg.mixed_precision)
    default = self._cfg([])
    self.assertIsNone(default.hours)
    self.assertEqual(default.target, 250)
    self.assertEqual(default.out_dir, m.LOCAL_OUT_DIR)
    self.assertEqual(default.data_dir, m.LOCAL_DATA_DIR)
    with self.assertRaises(ValueError):
      self._cfg(["--epochs", "0"])


class MetricsTest(unittest.TestCase):
  """metrics.json 기록 · 환경 정보."""

  def test_write_metrics_converts_numpy(self) -> None:
    """numpy 스칼라/배열이 json 기본형으로 저장된다."""
    payload = {"a": np.float32(1.5), "b": np.int64(3), "c": np.array([1, 2]),
               "nested": {"d": [np.float64(0.25)]}, "path": Path("results/ConvLSTM")}
    with tempfile.TemporaryDirectory() as d:
      out = Path(d) / "sub" / m.METRICS_NAME   # 부모 디렉터리도 만들어야 한다
      m.write_metrics(out, payload)
      loaded = json.loads(out.read_text(encoding="utf-8"))
    self.assertEqual(loaded["a"], 1.5)
    self.assertEqual(loaded["b"], 3)
    self.assertIsInstance(loaded["b"], int)
    self.assertEqual(loaded["c"], [1, 2])
    self.assertEqual(loaded["nested"]["d"], [0.25])
    self.assertEqual(loaded["path"], "results/ConvLSTM")

  def test_environment_info_keys(self) -> None:
    """env 절의 키 집합과 값 타입이 스키마와 맞는다."""
    env = m.environment_info(colab=False, precision_policy="float32")
    self.assertEqual(set(env), {"colab", "platform", "python", "tensorflow",
                                "keras", "gpu", "precision_policy"})
    self.assertIs(env["colab"], False)
    self.assertEqual(env["precision_policy"], "float32")
    for key, value in env.items():
      self.assertIsInstance(value, bool if key == "colab" else str, key)
    self.assertTrue(env["gpu"])   # GPU 가 없으면 "none"


class FigureNamesTest(unittest.TestCase):
  """metrics.json 의 figures 절과 실제 저장 파일명이 어긋나지 않는지 본다."""

  def test_figure_names_are_png_filenames(self) -> None:
    """FIGURE_NAMES 키와 확장자가 스키마와 맞는다."""
    self.assertEqual(set(m.FIGURE_NAMES), {"samples", "hourly_mean", "history", "full_frame"})
    self.assertTrue(all(v.endswith(".png") for v in m.FIGURE_NAMES.values()))

  def test_report_constants_match_pipeline_contract(self) -> None:
    """tools/build_report.py 가 다시 선언한 상수가 파이프라인 계약과 같다.

    리포트는 표준 라이브러리만 쓰려고 metrics 파일명과 figures 키를 따로 갖는다.
    파이프라인에서 이름을 바꾸면 리포트가 조용히 빈 페이지를 만드므로 여기서 잠근다.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import build_report as br

    self.assertEqual(br.METRICS_NAME, m.METRICS_NAME)
    self.assertEqual(set(br.FIGURE_KEYS), set(m.FIGURE_NAMES))

  def test_save_and_show_uses_given_filename(self) -> None:
    """전달한 파일명 그대로 저장한다 (확장자 중복 없음)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as d:
      fig = plt.figure()
      path = m.save_and_show(fig, Path(d) / "ConvLSTM", m.FIGURE_NAMES["samples"])
      self.assertEqual(path.name, "samples.png")
      self.assertTrue(path.is_file())


class ModelTest(unittest.TestCase):
  """TensorFlow 가 필요한 모델·지표 테스트 (CPU, 소형)."""

  def test_ssim_metric_identical_is_one(self) -> None:
    """같은 이미지의 SSIM 은 1 이다."""
    a = np.random.default_rng(1).random((3, 16, 16), dtype=np.float32)
    self.assertAlmostEqual(m.ssim_metric(a, a, chunk=2), 1.0, places=4)

  def test_loss_zero_for_identical(self) -> None:
    """같은 텐서면 손실이 0 이다."""
    import tensorflow as tf

    y = tf.constant(np.random.default_rng(2).random((2, 16, 16, 1), dtype=np.float32))
    self.assertAlmostEqual(float(m.ssim_mae_loss(y, y)), 0.0, places=5)

  def test_residual_head_adds_last_frame(self) -> None:
    """Δ=0 이면 출력이 입력 마지막 프레임과 같다."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(2, 8, 8, 1))
    last = m.make_take_last_frame_layer()(dtype="float32")(inp)
    # 커널·바이어스를 0 으로 두면 Δ = 0 이라 출력은 마지막 입력 프레임과 정확히 같아야 한다
    delta = layers.Conv2D(1, 1, kernel_initializer="zeros", dtype="float32")(last)
    model = keras.Model(inputs=inp, outputs=m.residual_head(inp, delta))
    x = np.random.default_rng(3).random((2, 2, 8, 8, 1)).astype(np.float32)
    out = model.predict(x, verbose=0)
    self.assertEqual(out.shape, (2, 8, 8, 1))
    np.testing.assert_allclose(out, x[:, -1], atol=1e-6)

  def test_persistence_baseline_on_synthetic(self) -> None:
    """일정 증가 시퀀스의 Persistence MAE 를 확인한다."""
    frames = np.stack([np.full((16, 16), 0.1 * t, np.float32) for t in range(5)])
    mae, ssim = m.persistence_baseline(frames, _stamps([0, 2, 4, 6, 8]), [(0, 5)])
    self.assertAlmostEqual(mae, 0.1, places=5)
    self.assertTrue(0.0 <= ssim <= 1.0)


if __name__ == "__main__":
  unittest.main()
