"""tools/build_colab_notebook.py (공통 파이프라인 + 모델 파일 -> 노트북) 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_build_colab_notebook.py -q
TensorFlow 없이 순수 텍스트 변환만 검증한다 (셀은 compile 까지만 하고 실행하지 않는다).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_colab_notebook as g  # noqa: E402

PIPELINE_SRC = ROOT / "nc_pipeline.py"
CONVLSTM_SRC = ROOT / "nc_predict_colab.py"


def _code_cells(nb: dict) -> list[str]:
  """노트북의 코드 셀 본문 목록."""
  return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _fake_root(tmp: Path, models: list[str]) -> Path:
  """임시 소스 루트를 만든다.

  `--all` 결과가 저장소에 어떤 모델 스크립트가 있는지에 좌우되면 Task 2/3 가
  파일을 추가하는 순간 테스트가 깨진다. 그래서 소스 루트를 통째로 격리한다.
  모델 파일은 ConvLSTM 엔트리를 복사해 MODEL_NAME 만 바꾼 것으로 충분하다.
  """
  root = tmp / "src"
  root.mkdir(parents=True, exist_ok=True)
  shutil.copy2(PIPELINE_SRC, root / "nc_pipeline.py")
  base = CONVLSTM_SRC.read_text(encoding="utf-8")
  for model in models:
    src_name, display = g.MODEL_SPECS[model]
    (root / src_name).write_text(
      base.replace('MODEL_NAME = "ConvLSTM"', f'MODEL_NAME = "{display}"'), encoding="utf-8")
  return root


def _built(out_dir: Path) -> list[str]:
  """out_dir 에 생긴 노트북 파일명 (정렬)."""
  return sorted(p.name for p in out_dir.glob("*.ipynb"))


class StripImportTest(unittest.TestCase):
  """모델 파일의 `from nc_pipeline import (...)` 제거."""

  def test_strip_pipeline_import_multiline(self) -> None:
    """여러 줄 괄호 import 를 통째로 지운다."""
    text = "from nc_pipeline import (\n  a,\n  b,\n)\nX = 1\n"
    out = g.strip_pipeline_import(text)
    self.assertIn("X = 1\n", out)
    self.assertNotIn("nc_pipeline", out)

  def test_strip_pipeline_import_single_line(self) -> None:
    """한 줄 import 만 지우고 나머지는 남긴다."""
    text = "from __future__ import annotations\nfrom nc_pipeline import run\nX = 1\n"
    out = g.strip_pipeline_import(text)
    self.assertIn("from __future__ import annotations", out)
    self.assertIn("X = 1", out)
    self.assertNotIn("nc_pipeline", out)

  def test_leaves_other_imports_alone(self) -> None:
    """nc_pipeline 이 아닌 import 는 건드리지 않는다."""
    text = "import numpy as np\nY = 2\n"
    self.assertEqual(g.strip_pipeline_import(text), text)


class BuildNotebookTest(unittest.TestCase):
  """실제 저장소 파일로 노트북을 만든다."""

  def test_build_notebook_convlstm_colab(self) -> None:
    """colab 프로필 노트북의 메타데이터·셀 내용·문법을 검증한다."""
    nb = g.build_notebook(PIPELINE_SRC, CONVLSTM_SRC, "ConvLSTM", "colab")
    self.assertEqual(nb["metadata"]["colab"]["gpuType"], "T4")
    self.assertEqual(nb["metadata"]["accelerator"], "GPU")
    cells = _code_cells(nb)
    joined = "".join(cells)
    self.assertIn("def run(", joined)
    self.assertIn("def build_model(", joined)
    self.assertIn("MODEL_NAME", joined)
    self.assertNotIn("from nc_pipeline", joined)
    self.assertNotIn('if __name__ == "__main__"', joined)
    self.assertNotIn("def main(", joined)
    self.assertIn("run(cfg, build_model, MODEL_NAME)", cells[-1])
    self.assertIn("COLAB_DATA_DIR", cells[-1])
    for i, code in enumerate(cells):   # 모든 코드 셀이 문법적으로 유효해야 한다
      compile(code, f"<cell {i}>", "exec")

  def test_build_notebook_local_profile(self) -> None:
    """local 프로필은 가속기 메타데이터 없이 로컬 경로를 쓴다."""
    nb = g.build_notebook(PIPELINE_SRC, CONVLSTM_SRC, "ConvLSTM", "local")
    self.assertNotIn("accelerator", nb["metadata"])
    self.assertNotIn("gpuType", nb["metadata"].get("colab", {}))
    cells = _code_cells(nb)
    self.assertIn("LOCAL_DATA_DIR", cells[-1])
    self.assertIn("LOCAL_OUT_DIR", cells[-1])
    self.assertIn("run(cfg, build_model, MODEL_NAME)", cells[-1])

  def test_pipeline_definitions_precede_model_cell(self) -> None:
    """모델 셀보다 파이프라인 정의 셀이 앞선다."""
    nb = g.build_notebook(PIPELINE_SRC, CONVLSTM_SRC, "ConvLSTM", "colab")
    cells = _code_cells(nb)
    compile_idx = next(i for i, c in enumerate(cells) if "def compile_model(" in c)
    model_idx = next(i for i, c in enumerate(cells) if "def build_model(" in c)
    self.assertLess(compile_idx, model_idx)

  def test_rejects_source_without_sections(self) -> None:
    """섹션 머리글이 없으면 ValueError 다."""
    with tempfile.TemporaryDirectory() as d:
      bad = Path(d) / "bad.py"
      bad.write_text('"""doc."""\nX = 1\n', encoding="utf-8")
      with self.assertRaises(ValueError):
        g.build_notebook(bad, CONVLSTM_SRC, "ConvLSTM", "colab")


class NotebookPathTest(unittest.TestCase):
  """생성 노트북 경로 규칙."""

  def test_notebook_path_refuses_handwritten(self) -> None:
    """수작업 ConvLSTM_prediction.ipynb 는 덮어쓰지 않는다."""
    root = Path("/tmp/repo")
    # ConvLSTM_prediction.ipynb 는 수작업 노트북이라 덮어쓰면 안 된다
    with self.assertRaises(ValueError):
      g.notebook_path("ConvLSTM", "local", root)

  def test_out_cannot_overwrite_handwritten_notebook(self) -> None:
    """--out 으로 수작업 노트북을 가리켜도 쓰지 않고 1 을 돌려준다."""
    with tempfile.TemporaryDirectory() as d:
      out = Path(d) / "ConvLSTM_prediction.ipynb"
      out.write_text("수작업 노트북", encoding="utf-8")
      code = g.main(["--model", "convlstm", "--profile", "colab", "--out", str(out)])
      self.assertEqual(code, 1)
      self.assertEqual(out.read_text(encoding="utf-8"), "수작업 노트북")

  def test_write_refuses_handwritten_name(self) -> None:
    """_write 자체가 수작업 노트북 이름을 거부한다 (경로를 어디서 받든 동일)."""
    with tempfile.TemporaryDirectory() as d:
      with self.assertRaises(ValueError):
        g._write({"cells": []}, Path(d) / "ConvLSTM_prediction.ipynb")

  def test_notebook_path_by_profile(self) -> None:
    """profile 에 따라 _colab 접미사가 붙는다."""
    root = Path("/tmp/repo")
    self.assertEqual(g.notebook_path("ConvLSTM", "colab", root),
                     root / "ConvLSTM_prediction_colab.ipynb")
    self.assertEqual(g.notebook_path("SimVP", "local", root),
                     root / "SimVP_prediction.ipynb")
    self.assertEqual(g.notebook_path("PredRNN_V2", "colab", root),
                     root / "PredRNN_V2_prediction_colab.ipynb")

  def test_model_specs(self) -> None:
    """모델 키 -> (스크립트, 표시 이름) 매핑이 계획과 같다."""
    self.assertEqual(g.MODEL_SPECS["convlstm"], ("nc_predict_colab.py", "ConvLSTM"))
    self.assertEqual(g.MODEL_SPECS["simvp"], ("simvp_predict_colab.py", "SimVP"))
    self.assertEqual(g.MODEL_SPECS["predrnn_v2"], ("predrnn_v2_predict_colab.py", "PredRNN_V2"))


class CliTest(unittest.TestCase):
  """CLI 동작 (--model / --all)."""

  def test_main_writes_notebook(self) -> None:
    """--model/--out 으로 노트북 파일이 만들어진다."""
    with tempfile.TemporaryDirectory() as d:
      out = Path(d) / "ConvLSTM_prediction_colab.ipynb"
      rc = g.main(["--model", "convlstm", "--profile", "colab", "--out", str(out)])
      self.assertEqual(rc, 0)
      self.assertTrue(out.is_file())

  def test_main_all_skips_missing_model_files(self) -> None:
    """--all 은 없는 모델 스크립트를 건너뛰고 0 을 반환한다."""
    with tempfile.TemporaryDirectory() as d:
      root = _fake_root(Path(d), ["convlstm"])
      out = Path(d) / "out"
      rc = g.main(["--all", "--root", str(root), "--out-dir", str(out)])
      self.assertEqual(rc, 0)
      self.assertEqual(_built(out), ["ConvLSTM_prediction_colab.ipynb"])

  def test_main_all_builds_every_target_when_sources_exist(self) -> None:
    """세 모델 스크립트가 모두 있으면 --all 이 5개 조합을 만든다."""
    with tempfile.TemporaryDirectory() as d:
      root = _fake_root(Path(d), ["convlstm", "simvp", "predrnn_v2"])
      out = Path(d) / "out"
      rc = g.main(["--all", "--root", str(root), "--out-dir", str(out)])
      self.assertEqual(rc, 0)
      self.assertEqual(_built(out), [
        "ConvLSTM_prediction_colab.ipynb",
        "PredRNN_V2_prediction.ipynb", "PredRNN_V2_prediction_colab.ipynb",
        "SimVP_prediction.ipynb", "SimVP_prediction_colab.ipynb",
      ])
      # ConvLSTM 의 local 은 수작업 노트북이 있어 --all 대상이 아니다
      self.assertFalse((out / "ConvLSTM_prediction.ipynb").exists())

  def test_all_warns_once_per_missing_model(self) -> None:
    """없는 모델 경고는 profile 조합마다가 아니라 모델당 한 번만 찍힌다."""
    with tempfile.TemporaryDirectory() as d:
      root = _fake_root(Path(d), ["convlstm"])
      with self.assertLogs(g.logger, level="WARNING") as caught:
        g.main(["--all", "--root", str(root), "--out-dir", str(Path(d) / "out")])
      missing = [line for line in caught.output if "모델 스크립트 없음" in line]
      self.assertEqual(len(missing), 2)   # simvp, predrnn_v2 각각 1번

  def test_main_uses_root_for_out_dir_by_default(self) -> None:
    """--out-dir 을 안 주면 --root 아래에 만든다 (저장소 루트를 건드리지 않는다)."""
    with tempfile.TemporaryDirectory() as d:
      root = _fake_root(Path(d), ["simvp"])
      rc = g.main(["--model", "simvp", "--profile", "local", "--root", str(root)])
      self.assertEqual(rc, 0)
      self.assertEqual(_built(root), ["SimVP_prediction.ipynb"])

  def test_main_reports_failure_for_missing_model_script(self) -> None:
    """--model 로 지정한 스크립트가 없으면 1 을 반환한다."""
    with tempfile.TemporaryDirectory() as d:
      root = _fake_root(Path(d), ["convlstm"])
      self.assertEqual(g.main(["--model", "simvp", "--root", str(root)]), 1)


if __name__ == "__main__":
  unittest.main()
