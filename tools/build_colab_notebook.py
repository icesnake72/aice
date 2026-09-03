"""공통 파이프라인(nc_pipeline.py) + 모델 스크립트 -> Colab/로컬 노트북 생성기.

Colab 의 '파일 > 노트 업로드' 는 .ipynb 만 받으므로, 두 스크립트를 섹션 단위 셀로 잘라
하나의 노트북으로 합친다. `.py` 를 고친 뒤 다시 실행하면 노트북이 갱신된다.

실행:
  python3 tools/build_colab_notebook.py --model convlstm --profile colab
  python3 tools/build_colab_notebook.py --model simvp --profile local
  python3 tools/build_colab_notebook.py --all      # 모델 스크립트가 있는 것만 생성

경로
  --root     nc_pipeline.py 와 모델 스크립트를 찾을 디렉터리 (기본: 저장소 루트)
  --out-dir  노트북을 만들 디렉터리 (기본: --root 와 같은 곳)

profile
  colab: 경로 기본값이 Drive (COLAB_DATA_DIR/COLAB_OUT_DIR), T4 GPU 메타데이터 포함
  local: 경로 기본값이 저장소 상대경로 (LOCAL_DATA_DIR/LOCAL_OUT_DIR), 가속기 메타데이터 없음

주의: `ConvLSTM_prediction.ipynb` 는 수작업 노트북이라 생성 대상에서 제외한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger("build_colab_notebook")

SECTION_RULE = re.compile(r"^# -{20,}$")
MAIN_DEF = "def main("
DUNDER_MAIN = "if __name__"
# `from nc_pipeline import (...)` (여러 줄) / 한 줄 / `import nc_pipeline` 을 모두 지운다.
PIPELINE_IMPORT_RE = re.compile(
  r"^from nc_pipeline import \([^)]*\)\n"
  r"|^from nc_pipeline import .*\n"
  r"|^import nc_pipeline.*\n",
  re.MULTILINE,
)

PROFILES = ("local", "colab")
MODEL_SPECS: dict[str, tuple[str, str]] = {
  "convlstm": ("nc_predict_colab.py", "ConvLSTM"),
  "simvp": ("simvp_predict_colab.py", "SimVP"),
  "predrnn_v2": ("predrnn_v2_predict_colab.py", "PredRNN_V2"),
}
# 수작업으로 만든 노트북. 생성기가 덮어쓰면 안 된다.
HANDWRITTEN_NOTEBOOKS = frozenset({"ConvLSTM_prediction.ipynb"})
# --all 로 만드는 조합 (ConvLSTM 의 local 은 수작업 노트북이 이미 있다)
ALL_TARGETS: tuple[tuple[str, str], ...] = (
  ("convlstm", "colab"),
  ("simvp", "local"), ("simvp", "colab"),
  ("predrnn_v2", "local"), ("predrnn_v2", "colab"),
)

PIPELINE_MODULE = "nc_pipeline.py"

INTRO_MD = """\
# GK2A AMI SW038 다음 프레임 예측 — {display} ({profile_label})

`{pipeline}` (공통 파이프라인) 과 `{model}` (모델 정의) 를 셀 단위로 합친 노트북이다. **이 파일은 `tools/build_colab_notebook.py` 로 자동 생성되므로 직접 고치지 말고 .py 를 고친 뒤 다시 생성한다.**

## 준비
{prep}

## 결과
`{out_hint}/{display}` 에 `metrics.json`, 그림 4장(`samples.png`, `hourly_mean.png`, `history.png`, `full_frame_prediction.png`), 가중치, `train_log.csv` 가 저장된다.
프레임 캐시(`{out_hint}/cache/*.npz`)는 모델끼리 공유하므로 두 번째 실행부터는 .nc 를 다시 읽지 않는다.

## 설정
마지막 셀의 `Config(...)` 값만 바꾸면 된다 (`epochs`, `hours`, `data_zip` 등).
"""

COLAB_PREP = """\
1. 로컬에서 데이터 다운로드: `python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out netcdf`
2. Google Drive 에 업로드
   - `MyDrive/netcdf/gk2a_ami_le1b_sw038_la020ge_202510170000.nc ...`
   - 또는 zip 하나로 `MyDrive/netcdf.zip` (마지막 셀에서 `data_zip` 지정)
3. 메뉴 **런타임 > 런타임 유형 변경 > T4 GPU** 확인 (노트북 메타데이터에 T4 가 지정돼 있어 보통 자동 선택된다)
4. **런타임 > 모두 실행**. Drive 마운트 승인 창이 뜨면 허용한다."""

LOCAL_PREP = """\
1. 데이터 다운로드: `python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out netcdf`
2. `resource/netcdf/` 에 .nc 가 있는지 확인한다 (저장소 루트에서 노트북을 연다)
3. `pip install tensorflow xarray netCDF4 matplotlib` 로 의존성을 갖춘다
4. 위에서부터 순서대로 실행한다."""

RUN_MD = """\
## 설정 · 실행
필요한 값만 바꾸고 실행한다. `hours=list(range(6, 24))` 로 두면 주간 태양반사 구간(00~06 UTC)을 뺀다.
"""

RUN_CODE = """\
cfg = Config(
{data_line}
{out_line}
{zip_line}
  epochs=4,
  batch=16,
  filters=16,                   # 모델의 기본 hidden 폭
  target=250,                   # 원본 500 의 정수배 약수. None 이면 원본 유지
  hours=None,                   # 예: list(range(6, 24))
  use_cache=True,               # False 면 .nc 를 다시 읽는다
  mixed_precision=True,         # GPU 가 있을 때만 적용된다. 문제가 있으면 False
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout, force=True)
results = run(cfg, build_model, MODEL_NAME)
results["val"]
"""

COMMENT_COL = 32   # Config(...) 안의 주석 시작 열

PROFILE_DIRS = {
  "colab": ("COLAB_DATA_DIR", "COLAB_OUT_DIR", "MyDrive/nc_predict_output",
            'COLAB_DRIVE_ROOT / "MyDrive/netcdf.zip"'),
  "local": ("LOCAL_DATA_DIR", "LOCAL_OUT_DIR", "results",
            'Path("resource/netcdf.zip")'),
}


def _cell(kind: str, source: str) -> dict:
  """nbformat 4 셀 하나를 만든다. source 는 줄 단위 리스트로 저장한다."""
  lines = source.rstrip("\n").splitlines(keepends=True)
  cell: dict = {"cell_type": kind, "metadata": {}, "source": lines}
  if kind == "code":
    cell.update({"execution_count": None, "outputs": []})
  return cell


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
  """스크립트를 (docstring 이후 머리부, [(섹션 제목, 섹션 코드), ...]) 로 나눈다.

  섹션 경계는 `# ----` / `# 제목` / `# ----` 3줄짜리 머리글이다.
  """
  lines = text.splitlines(keepends=True)
  if not lines or not lines[0].startswith('"""'):
    raise ValueError("모듈 docstring 으로 시작해야 한다.")
  first = lines[0].rstrip()
  if len(first) > 3 and first.endswith('"""'):   # 한 줄짜리 docstring
    doc_end = 0
  else:
    doc_end = next((i for i in range(1, len(lines)) if lines[i].rstrip().endswith('"""')), -1)
  if doc_end < 0:
    raise ValueError("모듈 docstring 이 닫히지 않았다.")

  starts = [i for i in range(doc_end + 1, len(lines) - 2)
            if SECTION_RULE.match(lines[i].rstrip()) and SECTION_RULE.match(lines[i + 2].rstrip())]
  if not starts:
    raise ValueError("섹션 머리글(# ----- / # 제목 / # -----)을 찾지 못했다.")

  head = "".join(lines[doc_end + 1:starts[0]]).strip("\n") + "\n"
  sections: list[tuple[str, str]] = []
  for idx, s in enumerate(starts):
    e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
    title = lines[s + 1].strip()
    sections.append((title, "".join(lines[s:e]).strip("\n") + "\n"))
  return head, sections


def strip_pipeline_import(text: str) -> str:
  """모델 파일의 `from nc_pipeline import ...` 문장을 지운다.

  노트북에는 파이프라인 코드가 이미 앞 셀로 들어가 있어 import 가 실패한다.
  여러 줄 괄호 형태와 한 줄 형태를 모두 지우고, 남은 빈 줄을 정리한다.
  """
  out = PIPELINE_IMPORT_RE.sub("", text)
  return re.sub(r"\n{3,}", "\n\n", out)


def strip_entry_points(code: str) -> str:
  """셀에 불필요한 최상위 `def main(...)` / `if __name__` 블록 이후를 잘라낸다.

  `def main_for_model(` 처럼 이름이 더 긴 함수는 건드리지 않는다.
  """
  lines = code.splitlines(keepends=True)
  cut = next((i for i, ln in enumerate(lines)
              if ln.startswith(MAIN_DEF) or ln.startswith(DUNDER_MAIN)), None)
  if cut is None:
    return code
  return "".join(lines[:cut]).rstrip("\n") + "\n"


def reject_handwritten(name: str) -> None:
  """파일 이름이 수작업 노트북이면 ValueError 를 던진다 (덮어쓰기 방지)."""
  if name in HANDWRITTEN_NOTEBOOKS:
    raise ValueError(f"{name} 은 수작업 노트북이라 생성기가 덮어쓰지 않는다.")


def notebook_path(display: str, profile: str, root: Path) -> Path:
  """생성 노트북 경로. 수작업 노트북 이름과 겹치면 거부한다."""
  if profile not in PROFILES:
    raise ValueError(f"profile 은 {PROFILES} 중 하나여야 한다: {profile}")
  name = f"{display}_prediction{'_colab' if profile == 'colab' else ''}.ipynb"
  reject_handwritten(name)
  return root / name


def _config_line(field: str, value: str, comment: str) -> str:
  """`  field=value,` 뒤 주석을 COMMENT_COL 열에 맞춰 붙인다."""
  code = f"  {field}={value},"
  return f"{code}{' ' * max(1, COMMENT_COL - len(code))}# {comment}"


def _run_cell(profile: str) -> str:
  """profile 에 맞는 마지막 실행 셀 코드."""
  data_const, out_const, _, zip_hint = PROFILE_DIRS[profile]
  return RUN_CODE.format(
    data_line=_config_line("data_dir", data_const, ".nc 디렉터리"),
    out_line=_config_line("out_dir", out_const, "결과는 out_dir / MODEL_NAME 아래에 쌓인다"),
    zip_line=_config_line("data_zip", "None", f"예: {zip_hint}"),
  )


def _intro_cell(display: str, model_src: Path, profile: str) -> str:
  """profile 에 맞는 안내 markdown."""
  _, _, out_hint, _ = PROFILE_DIRS[profile]
  return INTRO_MD.format(
    display=display,
    profile_label="Colab T4" if profile == "colab" else "로컬",
    pipeline=PIPELINE_MODULE,
    model=model_src.name,
    prep=COLAB_PREP if profile == "colab" else LOCAL_PREP,
    out_hint=out_hint,
  )


def build_notebook(pipeline_src: Path, model_src: Path, display: str, profile: str) -> dict:
  """공통 파이프라인 + 모델 파일을 합친 nbformat 4 딕셔너리를 만든다.

  셀 순서: 안내 md -> 파이프라인 머리부 -> 파이프라인 섹션들 -> 모델 머리부(MODEL_NAME)
           -> 모델 섹션들 -> 설정 md -> 실행 셀.
  """
  if profile not in PROFILES:
    raise ValueError(f"profile 은 {PROFILES} 중 하나여야 한다: {profile}")

  pipe_head, pipe_sections = split_sections(pipeline_src.read_text(encoding="utf-8"))
  model_head, model_sections = split_sections(model_src.read_text(encoding="utf-8"))

  cells = [_cell("markdown", _intro_cell(display, model_src, profile)), _cell("code", pipe_head)]
  for _, code in pipe_sections:
    cells.append(_cell("code", strip_entry_points(code)))

  cells.append(_cell("code", strip_pipeline_import(model_head)))
  for _, code in model_sections:
    cells.append(_cell("code", strip_entry_points(strip_pipeline_import(code))))

  cells.append(_cell("markdown", RUN_MD))
  cells.append(_cell("code", _run_cell(profile)))

  for i, c in enumerate(cells):   # 셀 단위로 문법 검사
    if c["cell_type"] == "code":
      compile("".join(c["source"]), f"<cell {i}>", "exec")

  metadata: dict = {
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
  }
  if profile == "colab":
    metadata["accelerator"] = "GPU"
    metadata["colab"] = {"provenance": [], "gpuType": "T4",
                         "name": f"{display}_prediction_colab.ipynb"}
  return {"nbformat": 4, "nbformat_minor": 0, "metadata": metadata, "cells": cells}


def _write(nb: dict, out: Path) -> None:
  """노트북 JSON 을 파일로 쓴다. `--out` 으로 수작업 노트북을 가리켜도 덮어쓰지 않는다."""
  reject_handwritten(out.name)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
  n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
  logger.info("생성: %s (셀 %d개, 코드 셀 %d개)", out, len(nb["cells"]), n_code)


def _build_one(model: str, profile: str, root: Path, out_dir: Path, out: Path | None) -> bool:
  """모델 하나를 생성한다. 모델 스크립트가 없으면 경고 후 False."""
  src_name, display = MODEL_SPECS[model]
  model_src = root / src_name
  if not model_src.is_file():
    logger.warning("모델 스크립트 없음 -> 건너뜀: %s", model_src)
    return False
  nb = build_notebook(root / PIPELINE_MODULE, model_src, display, profile)
  _write(nb, out if out is not None else notebook_path(display, profile, out_dir))
  return True


def available_targets(root: Path) -> list[tuple[str, str]]:
  """--all 대상 중 모델 스크립트가 실제로 있는 조합만 고른다.

  경고는 (모델, profile) 조합마다가 아니라 모델당 한 번만 찍는다.
  """
  missing = {m for m, _ in ALL_TARGETS if not (root / MODEL_SPECS[m][0]).is_file()}
  for m in sorted(missing):
    logger.warning("모델 스크립트 없음 -> 건너뜀: %s", root / MODEL_SPECS[m][0])
  return [(m, prof) for m, prof in ALL_TARGETS if m not in missing]


def main(argv: list[str] | None = None) -> int:
  """CLI 진입점."""
  repo_root = Path(__file__).resolve().parents[1]
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--model", choices=sorted(MODEL_SPECS), default="convlstm")
  p.add_argument("--profile", choices=PROFILES, default="colab")
  p.add_argument("--all", action="store_true", help="모델 스크립트가 있는 조합을 전부 생성한다")
  p.add_argument("--root", type=Path, default=repo_root,
                 help="nc_pipeline.py 와 모델 스크립트를 찾을 디렉터리 (기본: 저장소 루트)")
  p.add_argument("--out", type=Path, default=None, help="출력 .ipynb 경로 (--model 일 때만)")
  p.add_argument("--out-dir", type=Path, default=None,
                 help="노트북을 만들 디렉터리 (기본: --root 와 같은 곳)")
  a = p.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  out_dir = a.out_dir if a.out_dir is not None else a.root

  try:
    if a.all:
      if a.out is not None:
        raise ValueError("--all 과 --out 은 함께 쓸 수 없다.")
      targets = available_targets(a.root)
      made = sum(_build_one(m, prof, a.root, out_dir, None) for m, prof in targets)
      logger.info("총 %d개 생성 (%d개 조합 중)", made, len(ALL_TARGETS))
      return 0
    if not _build_one(a.model, a.profile, a.root, out_dir, a.out):
      return 1
    return 0
  except (OSError, ValueError, SyntaxError) as exc:
    logger.error("%s", exc)
    return 1


if __name__ == "__main__":
  sys.exit(main())
