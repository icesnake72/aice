"""tools/build_report.py 리포트 빌더 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_build_report.py -q -p no:cacheprovider
tensorflow 없이 표준 라이브러리만으로 도는 테스트다.
"""

from __future__ import annotations

import base64
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_report as br  # noqa: E402

# PIL 없이 쓰는 1x1 투명 PNG 바이트 상수.
PNG_1X1 = base64.b64decode(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

# HTML void element (닫는 태그가 없다).
VOID_TAGS = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"})


def _metrics(model: str, params: int, model_mae: float, model_ssim: float) -> dict[str, Any]:
  """0.3 스키마를 따르는 metrics.json 내용을 만든다."""
  return {
    "schema_version": 1,
    "model": model,
    "params": params,
    "created_at": "2026-09-03T14:05:00+09:00",
    "env": {
      "colab": False,
      "platform": "macOS-15.2-arm64",
      "python": "3.11.4",
      "tensorflow": "2.15.0",
      "keras": "2.15.0",
      "gpu": "METAL",
      "precision_policy": "mixed_float16",
    },
    "config": {
      "in_frames": 4, "target": 250, "patch": 96, "stride": 77, "filters": 16,
      "epochs": 4, "batch": 16, "lr": 0.001, "hours": None, "seed": 42,
    },
    "data": {
      "n_frames": 710,
      "period": "2025-10-17 00:00 ~ 23:58 UTC",
      "segments": [[0, 180], [180, 630], [630, 710]],
      "gmin": 14848.5, "gmax": 16361.5, "n_train": 5022, "n_val": 1224,
      "train_period": "00:00 ~ 19:00", "val_period": "19:02 ~ 23:58",
    },
    "baseline": {"mae": 0.00801, "ssim": 0.9560},
    "train": {
      "epochs_run": 2, "seconds": 906.2, "sec_per_epoch": 453.1,
      "history": {
        "loss": [0.0187, 0.0111], "val_loss": [0.0101, 0.0092],
        "mae": [0.0082, 0.0061], "val_mae": [0.0046, 0.0044],
      },
    },
    "val": {
      "model_mae": model_mae, "model_ssim": model_ssim,
      "pers_mae": 0.00623, "pers_ssim": 0.9595, "mae_gain_pct": 38.1,
    },
    "full_frame": {
      "model_mae": 0.00662, "model_ssim": 0.9728,
      "pers_mae": 0.01091, "pers_ssim": 0.9204,
      "t_pred": "23:58", "inputs": "23:50~23:56",
    },
    "figures": {
      "samples": "samples.png", "hourly_mean": "hourly_mean.png",
      "history": "history.png", "full_frame": "full_frame_prediction.png",
    },
  }


def _write_fixture(root: Path, model: str, params: int, mae: float, ssim: float,
                   figures: tuple[str, ...]) -> Path:
  """results/<Model>/ 디렉터리에 metrics.json 과 그림 파일을 만든다."""
  d = root / model
  d.mkdir(parents=True, exist_ok=True)
  (d / "metrics.json").write_text(json.dumps(_metrics(model, params, mae, ssim)), encoding="utf-8")
  for name in figures:
    (d / name).write_bytes(PNG_1X1)
  return d


def make_results(root: Path) -> Path:
  """ConvLSTM(그림 4장), PredRNN_V2(history.png 누락) fixture 를 만든다."""
  all_figs = ("samples.png", "hourly_mean.png", "history.png", "full_frame_prediction.png")
  _write_fixture(root, "ConvLSTM", 28497, 0.00386, 0.9866, all_figs)
  _write_fixture(root, "PredRNN_V2", 51233, 0.00351, 0.9881,
                 tuple(f for f in all_figs if f != "history.png"))
  return root


class TagBalance(HTMLParser):
  """열고 닫는 태그가 짝이 맞는지 확인하는 최소 파서."""

  def __init__(self) -> None:
    """스택과 오류 목록을 초기화한다."""
    super().__init__(convert_charrefs=True)
    self.stack: list[str] = []
    self.errors: list[str] = []

  def handle_starttag(self, tag: str, attrs: list[Any]) -> None:
    """void 가 아닌 태그를 스택에 넣는다."""
    if tag not in VOID_TAGS:
      self.stack.append(tag)

  def handle_endtag(self, tag: str) -> None:
    """스택 최상단과 비교해 짝을 확인한다."""
    if tag in VOID_TAGS:
      return
    if not self.stack:
      self.errors.append(f"닫는 태그가 남음: </{tag}>")
    elif self.stack[-1] != tag:
      self.errors.append(f"짝이 안 맞음: <{self.stack[-1]}> vs </{tag}>")
      self.stack.pop()
    else:
      self.stack.pop()


def test_load_results_order(tmp_path: Path) -> None:
  """표시 순서는 ConvLSTM, SimVP, PredRNN_V2 이고 없는 모델은 빠진다."""
  results = br.load_results(make_results(tmp_path))
  assert [r["model"] for r in results] == ["ConvLSTM", "PredRNN_V2"]
  assert all(isinstance(r["_dir"], Path) for r in results)
  assert results[0]["_dir"].name == "ConvLSTM"


def test_load_results_missing_dir(tmp_path: Path) -> None:
  """results 디렉터리가 없거나 비어 있으면 빈 리스트."""
  assert br.load_results(tmp_path / "nope") == []
  assert br.load_results(tmp_path) == []


def test_load_results_sorts_unknown_models_by_name(tmp_path: Path) -> None:
  """알려지지 않은 모델은 알려진 모델 뒤에 이름순으로 붙는다."""
  make_results(tmp_path)
  _write_fixture(tmp_path, "Zeta", 10, 0.01, 0.9, ())
  _write_fixture(tmp_path, "Alpha", 10, 0.01, 0.9, ())
  assert [r["model"] for r in br.load_results(tmp_path)] == \
    ["ConvLSTM", "PredRNN_V2", "Alpha", "Zeta"]


def test_load_results_skips_broken_json(tmp_path: Path) -> None:
  """깨진 metrics.json 은 건너뛰고 나머지를 읽는다."""
  make_results(tmp_path)
  broken = tmp_path / "SimVP"
  broken.mkdir()
  (broken / "metrics.json").write_text("{not json", encoding="utf-8")
  assert [r["model"] for r in br.load_results(tmp_path)] == ["ConvLSTM", "PredRNN_V2"]


def test_encode_image(tmp_path: Path) -> None:
  """있는 파일은 data URI, 없는 파일은 None."""
  p = tmp_path / "x.png"
  p.write_bytes(PNG_1X1)
  uri = br.encode_image(p)
  assert uri is not None and uri.startswith("data:image/png;base64,")
  assert base64.b64decode(uri.split(",", 1)[1]) == PNG_1X1
  assert br.encode_image(tmp_path / "missing.png") is None
  assert br.encode_image(tmp_path) is None


def test_render_contains_table_and_missing_placeholder(tmp_path: Path) -> None:
  """두 모델이 표에 있고 SimVP 는 '결과 없음', 그림은 base64, 외부 리소스는 0."""
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-03T12:00:00+09:00")
  assert "ConvLSTM" in html_text
  assert "PredRNN" in html_text
  assert "SimVP" in html_text
  assert "결과 없음" in html_text
  assert "data:image/png;base64," in html_text
  assert "<script" not in html_text
  assert "http://" not in html_text
  assert "https://" not in html_text
  assert "2026-09-03T12:00:00+09:00" in html_text


def test_render_is_well_formed_and_self_contained(tmp_path: Path) -> None:
  """태그 짝이 맞고 lang/charset/차트/표가 들어 있다."""
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-03T12:00:00+09:00")
  parser = TagBalance()
  parser.feed(html_text)
  parser.close()
  assert parser.errors == [], parser.errors
  assert parser.stack == [], parser.stack
  assert '<html lang="ko">' in html_text
  assert '<meta charset="utf-8"' in html_text
  assert "<svg" in html_text
  assert "<table" in html_text
  assert "prefers-color-scheme" in html_text
  # 외부 파일 참조가 없어야 한다 (src/href 는 data: 또는 fragment 만).
  for token in ('src="', 'href="'):
    for chunk in html_text.split(token)[1:]:
      value = chunk.split('"', 1)[0]
      assert value.startswith(("data:", "#")), value


def test_render_shows_numbers_and_reference_line(tmp_path: Path) -> None:
  """비교표에 지정 서식의 값과 Persistence 기준선이 나온다."""
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-03T12:00:00+09:00")
  assert "0.00386" in html_text        # val MAE 5자리
  assert "0.9866" in html_text         # val SSIM 4자리
  assert "38.1" in html_text           # 개선율 1자리
  assert "28,497" in html_text         # params 천 단위 구분
  assert "453" in html_text            # sec/epoch 0자리
  assert "Persistence" in html_text
  assert "METAL" in html_text


def test_render_handles_no_results() -> None:
  """결과가 하나도 없어도 페이지는 만들어지고 모든 모델이 '결과 없음'."""
  html_text = br.render_html([], "2026-09-03T12:00:00+09:00")
  assert html_text.count("결과 없음") >= 3
  for model in ("ConvLSTM", "SimVP", "PredRNN"):
    assert model in html_text
  parser = TagBalance()
  parser.feed(html_text)
  parser.close()
  assert parser.errors == [] and parser.stack == []
  assert "data:image/png;base64," not in html_text


def test_render_missing_figure_placeholder(tmp_path: Path) -> None:
  """history.png 가 없는 모델은 그림 자리에 안내 문구가 들어간다."""
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-03T12:00:00+09:00")
  assert "그림 없음" in html_text


def test_render_rejects_figure_path_outside_model_dir(tmp_path: Path) -> None:
  """figures 값이 파일명이 아니면(경로 이탈) 읽지 않고 '그림 없음' 으로 둔다."""
  make_results(tmp_path)
  secret = tmp_path / "secret.png"
  secret.write_bytes(PNG_1X1)
  path = tmp_path / "ConvLSTM" / "metrics.json"
  data = json.loads(path.read_text(encoding="utf-8"))
  data["figures"]["full_frame"] = "../secret.png"
  path.write_text(json.dumps(data), encoding="utf-8")
  html_text = br.render_html(br.load_results(tmp_path), "2026-09-03T12:00:00+09:00")
  assert "그림 없음" in html_text
  assert base64.b64encode(PNG_1X1).decode() in html_text  # 정상 그림은 그대로 들어간다
  assert html_text.count("data:image/png;base64,") == 4  # 5장 중 이탈 1장 제외


def test_cli_writes_file(tmp_path: Path) -> None:
  """main() 이 파일을 만들고 0 을 반환한다."""
  results_dir = make_results(tmp_path / "results")
  out = tmp_path / "site" / "index.html"
  code = br.main(["--results-dir", str(results_dir), "--out", str(out)])
  assert code == 0
  assert out.is_file()
  text = out.read_text(encoding="utf-8")
  assert text.startswith("<!doctype html>")
  assert "ConvLSTM" in text


def test_cli_writes_file_without_results(tmp_path: Path) -> None:
  """results 디렉터리가 없어도 '결과 없음' 페이지를 만든다."""
  out = tmp_path / "site" / "index.html"
  assert br.main(["--results-dir", str(tmp_path / "none"), "--out", str(out)]) == 0
  assert "결과 없음" in out.read_text(encoding="utf-8")
