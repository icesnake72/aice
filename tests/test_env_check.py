"""tools/env_check.py (실행 환경 점검 스크립트) 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_env_check.py -q -p no:cacheprovider

GPU 없이도 통과해야 하므로 tensorflow 를 모듈 최상단에서 임포트하지 않는다.
env_check.py 자체도 tensorflow 를 각 함수 안에서만 지연 임포트하므로,
`tools.env_check` 를 여기서 임포트해도 tensorflow 가 곧바로 로드되지는 않는다
(tensorflow 가 설치돼 있으면 build_report() 호출 시점에 비로소 쓰인다).
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import env_check as ec  # noqa: E402

EXPECTED_KEYS = {
  "python_version", "platform", "tensorflow_version", "keras_version",
  "gpus", "gpu_count", "mixed_precision_ok", "gpu_bench_ms",
  "xarray_ok", "netcdf_ok", "data_dir", "nc_file_count",
  "cache_path", "cache_exists", "korean_font", "disk_free_gb", "warnings",
}


class BuildReportTest(unittest.TestCase):
  """build_report() 가 돌려주는 dict 의 키/타입."""

  def test_returns_expected_keys(self) -> None:
    """모든 필드가 빠짐없이 채워진다."""
    with tempfile.TemporaryDirectory() as d:
      data_dir = Path(d)
      (data_dir / "gk2a_ami_le1b_sw038_la020ge_202510170000.nc").write_bytes(b"")
      report = ec.build_report(data_dir, run_gpu_bench=False)
    self.assertEqual(set(report.keys()), EXPECTED_KEYS)
    self.assertIsInstance(report["gpus"], list)
    self.assertIsInstance(report["warnings"], list)
    self.assertEqual(report["nc_file_count"], 1)

  def test_missing_data_dir_counts_zero(self) -> None:
    """존재하지 않는 데이터 디렉터리는 0개로 집계되고 경고가 붙는다."""
    report = ec.build_report(Path("이런_디렉터리는_없다"), run_gpu_bench=False)
    self.assertEqual(report["nc_file_count"], 0)
    self.assertTrue(any("nc" in w or ".nc" in w for w in report["warnings"]))

  def test_no_gpu_bench_flag_skips_benchmark(self) -> None:
    """run_gpu_bench=False 면 GPU 유무와 무관하게 gpu_bench_ms 가 None 이다."""
    with tempfile.TemporaryDirectory() as d:
      report = ec.build_report(Path(d), run_gpu_bench=False)
    self.assertIsNone(report["gpu_bench_ms"])


class ExitCodeTest(unittest.TestCase):
  """determine_exit_code() 의 조건: tensorflow 미설치 또는 .nc 0개일 때만 1."""

  def test_tensorflow_missing_is_failure(self) -> None:
    report = {"tensorflow_version": None, "nc_file_count": 5}
    self.assertEqual(ec.determine_exit_code(report), 1)

  def test_no_nc_files_is_failure(self) -> None:
    report = {"tensorflow_version": "2.15.0", "nc_file_count": 0}
    self.assertEqual(ec.determine_exit_code(report), 1)

  def test_gpu_absent_is_not_failure(self) -> None:
    """GPU 가 없어도(빈 리스트) tensorflow 와 데이터만 있으면 실패가 아니다."""
    report = {"tensorflow_version": "2.15.0", "nc_file_count": 3}
    self.assertEqual(ec.determine_exit_code(report), 0)


class CliTest(unittest.TestCase):
  """main() 의 --json / --no-gpu-bench / 종료 코드."""

  def test_json_output_parses(self) -> None:
    """--json 출력이 유효한 JSON 이고 build_report() 와 같은 키를 담는다."""
    with tempfile.TemporaryDirectory() as d:
      (Path(d) / "sample.nc").write_bytes(b"")
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        code = ec.main(["--data-dir", d, "--no-gpu-bench", "--json"])
      parsed = json.loads(buf.getvalue())
    self.assertEqual(set(parsed.keys()), EXPECTED_KEYS)
    self.assertEqual(code, 0)

  def test_no_gpu_bench_cli_flag(self) -> None:
    """CLI 에서도 --no-gpu-bench 를 주면 벤치마크를 건너뛴다."""
    with tempfile.TemporaryDirectory() as d:
      (Path(d) / "sample.nc").write_bytes(b"")
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        ec.main(["--data-dir", d, "--no-gpu-bench", "--json"])
      parsed = json.loads(buf.getvalue())
    self.assertIsNone(parsed["gpu_bench_ms"])

  def test_missing_data_dir_nonzero_exit_but_prints_table(self) -> None:
    """데이터 디렉터리가 없으면 종료 코드는 0이 아니지만 표는 그대로 출력된다."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      code = ec.main(["--data-dir", "이런_디렉터리는_없다", "--no-gpu-bench"])
    output = buf.getvalue()
    self.assertNotEqual(code, 0)
    self.assertIn("GK2A", output)
    self.assertIn(".nc 파일 수", output)


if __name__ == "__main__":
  unittest.main()
