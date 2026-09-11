"""tools/build_report.py 리포트 빌더 테스트.

실행:
  /usr/local/bin/python3 -m pytest tests/test_build_report.py -q -p no:cacheprovider
tensorflow 없이 표준 라이브러리만으로 도는 테스트다.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
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
  """표시 순서는 ConvLSTM, SimVP, PredRNN_V2, VideoTransformer 이고 없는 모델은 빠진다."""
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
  """두 모델이 표에 있고 SimVP·VideoTransformer 는 '결과 없음', 그림은 base64, 외부 리소스는 0."""
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-03T12:00:00+09:00")
  assert "ConvLSTM" in html_text
  assert "PredRNN" in html_text
  assert "SimVP" in html_text
  assert "VideoTransformer" in html_text
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


def test_title_and_h1_list_every_model() -> None:
  """<title> 과 h1 이 MODEL_ORDER 의 네 모델을 모두 나열한다.

  제목을 하드코딩하면 모델을 추가할 때 조용히 낡는다 (VideoTransformer 를 넣기 전까지
  실제로 그랬다). MODEL_ORDER 에서 만들어지는지 확인한다.
  """
  html_text = br.render_html([], "2026-09-04T12:00:00+09:00")
  head_title = html_text.split("</head>", 1)[0].split("<title>", 1)[1].split("</title>", 1)[0]
  h1 = html_text.split("<h1>", 1)[1].split("</h1>", 1)[0]

  for name in ("ConvLSTM", "SimVP", "PredRNN-V2", "VideoTransformer"):
    assert name in head_title, name
    assert name in h1, name
  assert head_title == h1 == br.PAGE_TITLE
  assert br.PAGE_TITLE == " · ".join(br._label(m) for m in br.MODEL_ORDER) + " 비교"
  # 방법론 각주·데이터 요약 부제의 모델 수도 MODEL_ORDER 를 따른다
  assert br.MODEL_COUNT_PHRASE == "4개 모델"
  assert "세 모델" not in html_text


def test_fourth_series_uses_own_color_not_fallback(tmp_path: Path) -> None:
  """모델 4개를 그릴 때 네 번째 시리즈는 중립 fallback 이 아니라 slot 4 색을 쓴다.

  SERIES_TOKENS 가 3개뿐이면 VideoTransformer 가 회색(--series-other)으로 떨어져
  4모델 비교 페이지에서 한 모델만 다르게 보인다. dataviz 스킬 팔레트의 slot 4
  (라이트 #eda100 / 다크 #c98500)를 쓰는지 토큰과 렌더 결과 양쪽에서 확인한다.
  """
  assert br._series_token(3) == "--series-4"
  assert br._series_token(3) != br.SERIES_FALLBACK
  assert len(br.SERIES_TOKENS) >= len(br.MODEL_ORDER)

  all_figs = ("samples.png", "hourly_mean.png", "history.png", "full_frame_prediction.png")
  for i, model in enumerate(br.MODEL_ORDER):
    _write_fixture(tmp_path, model, 10000 + i, 0.004 + i * 0.0001, 0.98, all_figs)
  html_text = br.render_html(br.load_results(tmp_path), "2026-09-04T12:00:00+09:00")

  assert "var(--series-4)" in html_text
  assert "var(--series-other)" not in html_text
  # 라이트·다크 두 블록 모두에 토큰 값이 정의돼 있어야 테마 전환 시 색이 남는다
  assert "--series-4: #eda100;" in html_text
  assert "--series-4: #c98500;" in html_text


def test_render_handles_no_results() -> None:
  """결과가 하나도 없어도 페이지는 만들어지고 모든 모델이 '결과 없음'."""
  html_text = br.render_html([], "2026-09-03T12:00:00+09:00")
  assert html_text.count("결과 없음") >= 4
  for model in ("ConvLSTM", "SimVP", "PredRNN", "VideoTransformer"):
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


# 배너·각주 판별에 쓰는 표시 문구 (구현과 같은 문자열이어야 한다).
BANNER_MARK = "실행 조건이 모델마다 다르다."
NOTE_SAME = f"{br.MODEL_COUNT_PHRASE} 모두 같은 프레임 캐시"   # 모델 수를 따라 바뀐다
NOTE_DIFF = "실행 조건이 모델마다 다르다 (metrics.json 의 config·data 절이 어긋난다)"


class ConditionMismatchTest(unittest.TestCase):
  """모델 간 실행 조건 일치 검사와 경고 배너 (I-2).

  문서가 안내하는 'Colab 결과만 results/<Model>/ 로 복사' 절차는 모델마다 다른 조건을
  섞을 수 있다. 그때 페이지가 조용히 같은 조건인 척하지 않는지 본다.
  """

  def setUp(self) -> None:
    """임시 results 디렉터리를 만든다."""
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.root = Path(tmp.name)
    make_results(self.root)

  def _set_config(self, model: str, key: str, value: Any) -> None:
    """fixture 의 config 값 하나를 바꾼다."""
    path = self.root / model / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["config"][key] = value
    path.write_text(json.dumps(data), encoding="utf-8")

  def test_identical_conditions_have_no_banner(self) -> None:
    """조건이 같으면 배너가 없고 각주는 '같은 조건' 문장을 쓴다."""
    results = br.load_results(self.root)
    self.assertEqual(br.find_condition_mismatches(results), [])
    html_text = br.render_html(results, "2026-09-03T12:00:00+09:00")
    self.assertNotIn(BANNER_MARK, html_text)
    self.assertNotIn('class="banner"', html_text)
    self.assertIn(NOTE_SAME, html_text)
    self.assertNotIn(NOTE_DIFF, html_text)

  def test_differing_hours_shows_banner_and_warns(self) -> None:
    """config.hours 가 다르면 모델·키를 짚는 배너와 경고 로그가 나온다."""
    self._set_config("PredRNN_V2", "hours", [6, 7, 8])
    results = br.load_results(self.root)
    self.assertEqual(br.find_condition_mismatches(results), [("PredRNN_V2", ["hours"])])
    with self.assertLogs("build_report", level="WARNING") as captured:
      html_text = br.render_html(results, "2026-09-03T12:00:00+09:00")
    joined = " ".join(captured.output)
    self.assertIn("PredRNN_V2", joined)
    self.assertIn("hours", joined)
    self.assertIn(BANNER_MARK, html_text)
    banner = html_text.split('class="banner"', 1)[1].split("</div>", 1)[0]
    self.assertIn("PredRNN-V2", banner)
    self.assertIn("hours", banner)
    self.assertIn(NOTE_DIFF, html_text)
    self.assertNotIn(NOTE_SAME, html_text)
    parser = TagBalance()
    parser.feed(html_text)
    parser.close()
    self.assertEqual((parser.errors, parser.stack), ([], []))

  def test_differing_data_key_is_detected(self) -> None:
    """data 절(정규화 범위)이 달라도 잡는다."""
    path = self.root / "PredRNN_V2" / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["data"]["gmax"] = 20000.0
    path.write_text(json.dumps(data), encoding="utf-8")
    self.assertEqual(br.find_condition_mismatches(br.load_results(self.root)),
                     [("PredRNN_V2", ["gmax"])])

  def test_single_result_has_nothing_to_compare(self) -> None:
    """결과가 0~1개면 비교 대상이 없어 빈 목록이다."""
    self.assertEqual(br.find_condition_mismatches([]), [])
    self.assertEqual(br.find_condition_mismatches(br.load_results(self.root)[:1]), [])

  def test_epochs_difference_is_not_a_mismatch(self) -> None:
    """epochs·batch·lr 는 모델마다 달라도 되는 값이라 배너를 띄우지 않는다."""
    self._set_config("PredRNN_V2", "epochs", 99)
    self.assertEqual(br.find_condition_mismatches(br.load_results(self.root)), [])


def test_cli_returns_1_when_output_is_unwritable(tmp_path: Path) -> None:
  """출력 경로의 부모가 일반 파일이면 예외를 삼키고 1 을 돌려준다."""
  blocker = tmp_path / "blocker"
  blocker.write_text("not a directory", encoding="utf-8")
  out = blocker / "site" / "index.html"
  assert br.main(["--results-dir", str(make_results(tmp_path / "results")), "--out", str(out)]) == 1
  assert blocker.read_text(encoding="utf-8") == "not a directory"


# ---------------------------------------------------------------------------
# 모바일 대응 (좁은 화면 레이아웃·차트·네비)
# ---------------------------------------------------------------------------

class CompareTableParser(HTMLParser):
  """비교표의 열 제목과 행별 <td> data-label 을 모은다."""

  def __init__(self) -> None:
    """상태를 초기화한다."""
    super().__init__(convert_charrefs=True)
    self.in_table = False
    self.headers: list[str] = []
    self.rows: list[list[tuple[str | None, bool]]] = []
    self._reading_header = False

  def handle_starttag(self, tag: str, attrs: list[Any]) -> None:
    """표 안의 열 제목과 데이터 셀을 기록한다."""
    attr = dict(attrs)
    if tag == "table" and "stack-table" in (attr.get("class") or ""):
      self.in_table = True
      return
    if not self.in_table:
      return
    if tag == "th" and attr.get("scope") == "col":
      self._reading_header = True
      self.headers.append("")
    elif tag == "tr":
      self.rows.append([])
    elif tag == "td":
      self.rows[-1].append((attr.get("data-label"), "colspan" in attr))

  def handle_data(self, data: str) -> None:
    """열 제목 텍스트를 모은다."""
    if self._reading_header:
      self.headers[-1] += data

  def handle_endtag(self, tag: str) -> None:
    """열 제목·표의 끝을 표시한다."""
    if tag == "th":
      self._reading_header = False
    elif tag == "table":
      self.in_table = False


def _full_results(root: Path) -> Path:
  """MODEL_ORDER 네 모델 모두 결과가 있는 fixture 를 만든다."""
  all_figs = ("samples.png", "hourly_mean.png", "history.png", "full_frame_prediction.png")
  for i, model in enumerate(br.MODEL_ORDER):
    _write_fixture(root, model, 10000 + i, 0.004 + i * 0.0003, 0.98 - i * 0.002, all_figs)
  return root


def test_viewport_meta_is_present(tmp_path: Path) -> None:
  """휴대폰에서 축소되지 않도록 viewport meta 가 있어야 한다."""
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-11T12:00:00+09:00")
  assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html_text


def test_comparison_table_cells_carry_matching_data_label(tmp_path: Path) -> None:
  """비교표의 모든 <td> 가 자기 열 제목을 data-label 로 들고 있다.

  좁은 화면에서 표를 카드로 접을 때 CSS 가 이 값을 라벨로 되살리므로, 열 제목과
  어긋나면 값이 엉뚱한 이름표를 달게 된다.
  """
  html_text = br.render_html(br.load_results(_full_results(tmp_path)), "2026-09-11T12:00:00+09:00")
  parser = CompareTableParser()
  parser.feed(html_text)
  parser.close()

  assert parser.headers == list(br.COMPARE_HEADERS)
  data_rows = [r for r in parser.rows if r]
  assert len(data_rows) == len(br.MODEL_ORDER)
  for row in data_rows:
    assert all(label for label, _ in row), row
    assert [label for label, _ in row] == list(br.COMPARE_HEADERS[1:]), row


def test_missing_model_row_cell_also_has_data_label(tmp_path: Path) -> None:
  """'결과 없음' colspan 셀도 data-label 을 들고 있다 (CSS 가 감춘다)."""
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-11T12:00:00+09:00")
  parser = CompareTableParser()
  parser.feed(html_text)
  parser.close()
  spanned = [cell for row in parser.rows for cell in row if cell[1]]
  assert spanned, "결과 없음 행이 없다"
  assert all(label for label, _ in spanned)


def test_both_chart_variants_are_emitted(tmp_path: Path) -> None:
  """막대 차트 2개와 꺾은선 차트 1개가 넓은 화면용·좁은 화면용 두 벌로 나온다."""
  html_text = br.render_html(br.load_results(_full_results(tmp_path)), "2026-09-11T12:00:00+09:00")
  assert html_text.count('class="chart-scroll chart-wide"') == 3
  assert html_text.count('class="chart-narrow"') == 3
  # 같은 제목이 두 벌 모두에 있어야 같은 차트의 두 판이다
  for title in ("모델별 검증 MAE", "모델별 검증 SSIM", "모델별 epoch 손실 곡선"):
    assert html_text.count(f"<title>{title}</title>") == 2, title


def test_narrow_bar_chart_has_one_rect_per_model_and_reference_line() -> None:
  """좁은 화면 막대 차트는 모델당 <rect> 하나와 Persistence 점선 하나를 그린다."""
  labels = [br._label(m) for m in br.MODEL_ORDER]
  values = [0.0040, 0.0044, 0.0052, 0.0048]
  svg = br.svg_bar_chart_narrow(labels, values, list(range(4)), 0.00623,
                                "Persistence 기준선", 5, "검증 MAE", "모델별 검증 MAE")
  assert svg.count("<rect") == len(br.MODEL_ORDER)
  assert svg.count('stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 4"') == 2  # 기준선 + 범례
  assert 'viewBox="0 0 340' in svg
  assert "font-size=\"11\"" not in svg and "font-size=\"10\"" not in svg  # 12px 미만 글자 금지


def test_narrow_bar_chart_skips_bar_for_missing_value() -> None:
  """값이 없는 모델은 막대 없이 '결과 없음' 으로 남는다."""
  svg = br.svg_bar_chart_narrow(["A", "B"], [0.004, None], [0, 1], None,
                                "Persistence 기준선", 5, "검증 MAE", "t")
  assert svg.count("<rect") == 1
  assert br.NO_RESULT in svg


def test_chart_variants_share_the_same_series_tokens() -> None:
  """두 판이 같은 카테고리 색 토큰을 쓴다 (좁은 화면에서만 색이 달라지면 안 된다)."""
  labels = [br._label(m) for m in br.MODEL_ORDER]
  values = [0.004, 0.0044, 0.0052, 0.0048]
  wide = br.svg_bar_chart(labels, values, list(range(4)), 0.006, "ref", 5, "x", "t")
  narrow = br.svg_bar_chart_narrow(labels, values, list(range(4)), 0.006, "ref", 5, "x", "t")
  for token in br.SERIES_TOKENS:
    assert f"var({token})" in wide, token
    assert f"var({token})" in narrow, token
  assert "var(--series-other)" not in narrow


def test_narrow_line_chart_is_narrow_and_legible() -> None:
  """좁은 화면 꺾은선은 viewBox 가 좁고 글자가 12px 이상이며 범례가 아래에 있다."""
  series = [{"name": "ConvLSTM", "slot": 0, "loss": [0.02, 0.011], "val_loss": [0.010, 0.009]},
            {"name": "SimVP", "slot": 1, "loss": [0.03, 0.021], "val_loss": [0.020, 0.019]}]
  svg = br.svg_line_chart_narrow(series, "epoch", "손실", "모델별 epoch 손실 곡선")
  assert 'viewBox="0 0 340' in svg
  assert "font-size=\"11\"" not in svg
  assert "rotate(-90" not in svg  # 세로 회전 축 제목을 쓰지 않는다
  assert "ConvLSTM" in svg and "SimVP" in svg
  assert br.svg_line_chart_narrow([], "epoch", "손실", "t").count(br.NO_RESULT) == 1


def test_section_nav_links_every_section(tmp_path: Path) -> None:
  """네비가 페이지에 있는 모든 섹션 id 를 하나씩 가리킨다."""
  import re
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-11T12:00:00+09:00")
  ids = re.findall(r'<section id="([^"]+)"', html_text)
  nav = re.search(r'<nav class="section-nav".*?</nav>', html_text, re.S)
  assert nav is not None
  hrefs = re.findall(r'href="#([^"]+)"', nav.group(0))
  assert ids == list(br.SECTION_IDS)
  assert hrefs == ids
  assert 'aria-label="섹션 바로가기"' in nav.group(0)


def test_mobile_css_rules_exist(tmp_path: Path) -> None:
  """좁은 화면 규칙(640px 분기, data-label 라벨, 그림 패닝)이 CSS 에 있다."""
  html_text = br.render_html(br.load_results(make_results(tmp_path)), "2026-09-11T12:00:00+09:00")
  assert "@media (max-width: 640px)" in html_text
  assert "content: attr(data-label)" in html_text
  assert ".figure-pan img { min-width: 720px; }" in html_text
  assert br.PAN_HINT in html_text
  assert "position: sticky" in html_text
  assert "min-height: 44px" in html_text  # 탭 영역
  # 새 색은 라이트·다크 양쪽에 정의돼야 한다
  for token in ("--nav-bg", "--chip-bg"):
    assert html_text.count(f"{token}:") == 2, token


def test_page_stays_self_contained_after_mobile_changes(tmp_path: Path) -> None:
  """모바일 대응 뒤에도 스크립트·외부 리소스가 없다."""
  html_text = br.render_html(br.load_results(_full_results(tmp_path)), "2026-09-11T12:00:00+09:00")
  assert "<script" not in html_text
  assert "http://" not in html_text
  assert "https://" not in html_text
  parser = TagBalance()
  parser.feed(html_text)
  parser.close()
  assert parser.errors == [] and parser.stack == []
