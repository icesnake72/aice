"""nc_predict_colab.py -> ConvLSTM_prediction_colab.ipynb 변환기.

Colab 의 '파일 > 노트 업로드' 는 .ipynb 만 받으므로, 스크립트를 섹션 단위 셀로 잘라
Colab 노트북(T4 GPU 런타임 메타데이터 포함)을 만든다. .py 를 고친 뒤 다시 실행하면 된다.

실행:
  python3 tools/build_colab_notebook.py
  python3 tools/build_colab_notebook.py --src nc_predict_colab.py --out ConvLSTM_prediction_colab.ipynb
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
ENTRY_SECTION_TITLE = "# 진입점"
RUN_DEF = "def run("
MAIN_DEF = "def main("

INTRO_MD = """\
# GK2A AMI SW038 다음 프레임 예측 (ConvLSTM) — Colab T4

`nc_predict_colab.py` 를 셀 단위로 나눈 Colab 노트북이다. **이 파일은 `tools/build_colab_notebook.py` 로 자동 생성되므로 직접 고치지 말고 .py 를 고친 뒤 다시 생성한다.**

## 준비
1. 로컬에서 데이터 다운로드: `python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out netcdf`
2. Google Drive 에 업로드
   - `MyDrive/netcdf/gk2a_ami_le1b_sw038_la020ge_202510170000.nc ...`
   - 또는 zip 하나로 `MyDrive/netcdf.zip` (마지막 셀에서 `data_zip` 지정)
3. 메뉴 **런타임 > 런타임 유형 변경 > T4 GPU** 확인 (노트북 메타데이터에 T4 가 지정돼 있어 보통 자동 선택된다)
4. **런타임 > 모두 실행**. Drive 마운트 승인 창이 뜨면 허용한다.

## 결과
`MyDrive/nc_predict_output` 에 그림 4장, 가중치(`convlstm.weights.h5`), 매 epoch 체크포인트, `train_log.csv`, 프레임 캐시(`cache/*.npz`)가 저장된다.
두 번째 실행부터는 캐시를 읽으므로 .nc 를 다시 읽지 않는다.

## 설정
마지막 셀의 `Config(...)` 값만 바꾸면 된다 (`epochs`, `hours`, `data_zip` 등).
"""

RUN_MD = """\
## 설정 · 실행
필요한 값만 바꾸고 실행한다. `hours=list(range(6, 24))` 로 두면 주간 태양반사 구간(00~06 UTC)을 뺀다.
"""

RUN_CODE = """\
cfg = Config(
  data_dir=COLAB_DATA_DIR,      # MyDrive/netcdf
  out_dir=COLAB_OUT_DIR,        # MyDrive/nc_predict_output
  data_zip=None,                # 예: COLAB_DRIVE_ROOT / "MyDrive/netcdf.zip"
  epochs=4,
  batch=16,
  filters=16,
  target=250,                   # 원본 500 의 정수배 약수. None 이면 원본 유지
  hours=None,                   # 예: list(range(6, 24))
  use_cache=True,               # False 면 .nc 를 다시 읽는다
  mixed_precision=True,         # T4 에서 유효. 문제가 있으면 False
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout, force=True)
results = run(cfg)
results
"""


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
  doc_end = next(i for i in range(1, len(lines)) if lines[i].rstrip().endswith('"""'))

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


def extract_run(entry_code: str) -> str:
  """진입점 섹션에서 run() 함수만 떼어낸다 (parse_args/main/__main__ 은 노트북에 불필요)."""
  s = entry_code.find(RUN_DEF)
  e = entry_code.find(MAIN_DEF, s)
  if s < 0 or e < 0:
    raise ValueError("진입점 섹션에서 run()/main() 을 찾지 못했다.")
  return entry_code[s:e].rstrip("\n") + "\n"


def build_notebook(src: Path) -> dict:
  """스크립트를 읽어 nbformat 4 딕셔너리를 만든다."""
  head, sections = split_sections(src.read_text(encoding="utf-8"))
  cells = [_cell("markdown", INTRO_MD), _cell("code", head)]
  for title, code in sections:
    if title == ENTRY_SECTION_TITLE:
      cells.append(_cell("code", extract_run(code)))
    else:
      cells.append(_cell("code", code))
  cells.append(_cell("markdown", RUN_MD))
  cells.append(_cell("code", RUN_CODE))

  for i, c in enumerate(cells):   # 셀 단위로 문법 검사
    if c["cell_type"] == "code":
      compile("".join(c["source"]), f"<cell {i}>", "exec")

  return {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
      "accelerator": "GPU",
      "colab": {"provenance": [], "gpuType": "T4", "name": "ConvLSTM_prediction_colab.ipynb"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"},
    },
    "cells": cells,
  }


def main(argv: list[str] | None = None) -> int:
  """CLI 진입점."""
  root = Path(__file__).resolve().parents[1]
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--src", type=Path, default=root / "nc_predict_colab.py")
  p.add_argument("--out", type=Path, default=root / "ConvLSTM_prediction_colab.ipynb")
  a = p.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  try:
    nb = build_notebook(a.src)
    a.out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
    logger.info("생성: %s (셀 %d개, 코드 셀 %d개)", a.out, len(nb["cells"]), n_code)
    return 0
  except (OSError, ValueError, SyntaxError) as exc:
    logger.error("%s", exc)
    return 1


if __name__ == "__main__":
  sys.exit(main())
