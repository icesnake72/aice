"""tools/build_colab_notebook.py (공통 파이프라인 + 모델 파일 -> 노트북) 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_build_colab_notebook.py -q
TensorFlow 없이 순수 텍스트 변환만 검증한다 (셀은 compile 까지만 하고 실행하지 않는다).
"""

from __future__ import annotations

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
      rc = g.main(["--all", "--out-dir", d])
      self.assertEqual(rc, 0)
      names = sorted(p.name for p in Path(d).glob("*.ipynb"))
      # simvp / predrnn_v2 스크립트는 아직 없으므로 건너뛴다
      self.assertEqual(names, ["ConvLSTM_prediction_colab.ipynb"])


if __name__ == "__main__":
  unittest.main()
