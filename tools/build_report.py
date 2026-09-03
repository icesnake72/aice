"""results/<Model>/metrics.json + png -> 자기완결 HTML 리포트 생성기.

`site/index.html` 한 장에 CSS(inline), 그림(base64 data URI), 차트(inline SVG)를 모두 넣어
외부 리소스가 0 인 파일을 만든다. Netlify 는 `netlify.toml` 의 `publish = "site"` 로 이 파일을
그대로 서빙한다.

실행:
  python3 tools/build_report.py
  python3 tools/build_report.py --results-dir results --out site/index.html

표준 라이브러리만 쓴다 (numpy/matplotlib/tensorflow 불필요).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("build_report")

# 표시 순서 (그 외 모델은 이름순으로 뒤에 붙는다).
MODEL_ORDER: tuple[str, ...] = ("ConvLSTM", "SimVP", "PredRNN_V2")
# 디렉터리 이름 -> 화면 표기.
MODEL_LABELS: dict[str, str] = {
  "ConvLSTM": "ConvLSTM",
  "SimVP": "SimVP",
  "PredRNN_V2": "PredRNN-V2",
}
# nc_pipeline 의 계약 상수를 리포트 쪽에서 다시 선언한다 (표준 라이브러리만 쓰기 위해).
# 두 정의가 어긋나면 리포트가 조용히 비므로 tests/test_build_report.py 가 일치를 잠근다.
METRICS_NAME = "metrics.json"
FIGURE_KEYS: tuple[str, ...] = ("samples", "hourly_mean", "history", "full_frame")
# 모델 간 실행 조건 일치 검사 대상 (다르면 배너 + 경고).
CONDITION_CONFIG_KEYS: tuple[str, ...] = (
  "in_frames", "target", "patch", "stride", "filters", "hours", "seed")
CONDITION_DATA_KEYS: tuple[str, ...] = (
  "n_frames", "period", "segments", "gmin", "gmax", "train_period", "val_period")
NO_RESULT = "결과 없음"
NO_FIGURE = "그림 없음"
DASH = "—"
# dataviz 스킬 카테고리 팔레트 slot 1~3 (검증 완료). 4번째부터는 중립색으로 접는다.
SERIES_TOKENS: tuple[str, ...] = ("--series-1", "--series-2", "--series-3")
SERIES_FALLBACK = "--series-other"
MIME_BY_SUFFIX: dict[str, str] = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
}

PAGE_TITLE = "ConvLSTM · SimVP · PredRNN-V2 비교"

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-other: #898781;
  --ref: #52514e;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-other: #898781;
    --ref: #c3c2b7;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 20px 64px;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Apple SD Gothic Neo", "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.6;
}
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 26px; line-height: 1.3; margin: 0 0 8px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 0 0 4px; letter-spacing: -0.01em; }
h3 { font-size: 16px; margin: 0 0 4px; }
p { margin: 0 0 12px; }
.meta { color: var(--text-secondary); font-size: 13px; margin: 0; }
.lead { color: var(--text-secondary); }
.banner {
  margin: 16px 0 0;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--series-2);
  border-radius: 8px;
  background: var(--surface-1);
  font-size: 14px;
  line-height: 1.55;
}
.banner strong { color: var(--series-2); }
.banner .detail { display: block; margin-top: 6px; color: var(--text-secondary); font-size: 13px; }
section { margin-top: 32px; }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
}
.section-head { margin-bottom: 12px; }
.section-head p { margin: 4px 0 0; color: var(--text-secondary); font-size: 13px; }
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px 20px;
  margin: 16px 0 0;
}
.summary div { border-top: 1px solid var(--border); padding-top: 8px; }
.summary dt { color: var(--text-secondary); font-size: 12px; margin: 0; }
.summary dd { margin: 2px 0 0; font-size: 15px; }
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; min-width: 720px; font-size: 14px; }
caption { caption-side: bottom; text-align: left; color: var(--muted); font-size: 12px; padding-top: 8px; }
th, td { text-align: right; padding: 9px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; font-size: 12px; }
tbody th { color: var(--text-primary); font-size: 14px; font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
tbody td { font-variant-numeric: tabular-nums; }
tbody tr:last-child td, tbody tr:last-child th { border-bottom: none; }
.name { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; }
.swatch { width: 10px; height: 10px; border-radius: 3px; flex: 0 0 auto; }
.empty { color: var(--muted); font-style: normal; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
.chart { margin: 0; }
.chart figcaption { color: var(--text-secondary); font-size: 13px; margin: 0 0 8px; }
.chart-scroll { overflow-x: auto; }
.chart-scroll svg { width: 100%; min-width: 460px; height: auto; display: block; margin: 0 auto; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
.card-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.card-head .tag { color: var(--text-secondary); font-size: 12px; }
figure { margin: 0 0 16px; }
figure img { width: 100%; height: auto; display: block; border-radius: 8px; border: 1px solid var(--border); background: var(--surface-1); }
figure figcaption { color: var(--text-secondary); font-size: 12px; margin-top: 6px; }
.placeholder {
  border: 1px dashed var(--axis);
  border-radius: 8px;
  padding: 28px 12px;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
}
.kv { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 13px; margin: 0; }
.kv dt { color: var(--text-secondary); }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; text-align: right; }
.kv-title { color: var(--text-secondary); font-size: 12px; font-weight: 600; margin: 12px 0 6px; }
.notes { margin: 0; padding-left: 20px; color: var(--text-secondary); font-size: 14px; }
.notes li { margin-bottom: 6px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; text-align: center; }
"""


# ---------------------------------------------------------------------------
# 값 읽기 · 서식
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
  """어떤 값이든 문자열로 바꾼 뒤 HTML escape 한다."""
  return html.escape(str(value), quote=True)


def _as_number(value: Any) -> float | None:
  """숫자면 float, 아니면 None (bool 은 숫자로 보지 않는다)."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  return float(value)


def _get(data: Any, *keys: str) -> Any:
  """중첩 dict 에서 경로를 따라 값을 읽고 없으면 None 을 준다."""
  cur = data
  for key in keys:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur


def fmt_float(value: Any, digits: int, suffix: str = "") -> str:
  """소수 자릿수를 고정해 서식화한다 (숫자가 아니면 '—')."""
  num = _as_number(value)
  return DASH if num is None else f"{num:.{digits}f}{suffix}"


def fmt_int(value: Any) -> str:
  """천 단위 구분자를 넣은 정수 서식 (숫자가 아니면 '—')."""
  num = _as_number(value)
  return DASH if num is None else f"{int(round(num)):,}"


def fmt_mae(value: Any) -> str:
  """MAE 는 소수 5자리."""
  return fmt_float(value, 5)


def fmt_ssim(value: Any) -> str:
  """SSIM 은 소수 4자리."""
  return fmt_float(value, 4)


def fmt_pct(value: Any) -> str:
  """퍼센트는 소수 1자리."""
  return fmt_float(value, 1, "%")


def fmt_seconds(value: Any) -> str:
  """초는 정수로 반올림."""
  return fmt_float(value, 0)


def fmt_scalar(value: Any) -> str:
  """env/config 의 임의 스칼라 값을 사람이 읽는 문자열로 바꾼다."""
  if value is None:
    return DASH
  if isinstance(value, bool):
    return "예" if value else "아니오"
  if isinstance(value, int):
    return f"{value:,}"
  if isinstance(value, float):
    return f"{value:g}"
  if isinstance(value, (list, tuple)):
    return ", ".join(fmt_scalar(v) for v in value) or DASH
  return str(value)


# ---------------------------------------------------------------------------
# 입력 적재
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> list[dict[str, Any]]:
  """`<results_dir>/<Model>/metrics.json` 을 모두 읽어 표시 순서로 정렬해 돌려준다.

  각 dict 에는 그림 경로 계산용 `_dir: Path` 가 추가된다. 디렉터리가 없거나 비어 있으면
  빈 리스트를 준다. 읽기·파싱에 실패한 항목은 경고를 남기고 건너뛴다.
  """
  results: list[dict[str, Any]] = []
  if not results_dir.is_dir():
    logger.warning("results 디렉터리가 없다: %s", results_dir)
    return results
  for child in sorted(results_dir.iterdir()):
    metrics_path = child / METRICS_NAME
    if not child.is_dir() or not metrics_path.is_file():
      continue
    try:
      data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
      logger.warning("metrics.json 을 읽지 못했다 (%s): %s", metrics_path, exc)
      continue
    if not isinstance(data, dict):
      logger.warning("metrics.json 형식이 dict 가 아니다: %s", metrics_path)
      continue
    data["_dir"] = child
    if not isinstance(data.get("model"), str):
      data["model"] = child.name
    results.append(data)

  def sort_key(item: dict[str, Any]) -> tuple[int, str]:
    """알려진 모델은 고정 순서, 나머지는 그 뒤 이름순."""
    name = str(item["model"])
    return (MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER), name)

  results.sort(key=sort_key)
  return results


def encode_image(path: Path) -> str | None:
  """그림 파일을 base64 data URI 로 바꾼다. 파일이 없거나 읽지 못하면 None."""
  if not path.is_file():
    return None
  try:
    raw = path.read_bytes()
  except OSError as exc:
    logger.warning("그림을 읽지 못했다 (%s): %s", path, exc)
    return None
  mime = MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
  return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# SVG 차트 (외부 라이브러리·스크립트 없음)
# ---------------------------------------------------------------------------

def _nice_ticks(vmin: float, vmax: float, count: int = 4) -> list[float]:
  """[vmin, vmax] 를 모두 덮는 보기 좋은 눈금 값 목록을 만든다."""
  if not vmax > vmin:
    pad = abs(vmax) * 0.5 or 1.0
    vmin, vmax = vmin - pad, vmax + pad
  raw = (vmax - vmin) / max(count, 1)
  magnitude = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
  step = magnitude
  for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
    step = mult * magnitude
    if raw <= step:
      break
  ticks: list[float] = []
  value = math.floor(vmin / step) * step
  while value <= vmax + step * 1e-9:
    ticks.append(round(value, 12))
    value += step
  if not ticks:
    ticks = [round(vmin, 12)]
  if ticks[-1] < vmax:
    ticks.append(round(ticks[-1] + step, 12))
  return ticks


def _tick_digits(step: float) -> int:
  """눈금 간격에 맞는 소수 자릿수를 고른다."""
  if step <= 0:
    return 0
  return max(0, min(6, int(math.ceil(-math.log10(step))) + 1))


def _fmt_tick(value: float, digits: int) -> str:
  """눈금 라벨 서식 (0 은 항상 '0')."""
  if abs(value) < 1e-12:
    return "0"
  return f"{value:.{digits}f}"


def _bar_path(x: float, y: float, width: float, height: float, radius: float = 4.0) -> str:
  """기준선 쪽은 각지고 값 끝만 둥근 가로 막대 path 를 만든다."""
  r = max(0.0, min(radius, width))
  return (
    f"M {x:.2f} {y:.2f} H {x + width - r:.2f} "
    f"A {r:.2f} {r:.2f} 0 0 1 {x + width:.2f} {y + r:.2f} "
    f"V {y + height - r:.2f} "
    f"A {r:.2f} {r:.2f} 0 0 1 {x + width - r:.2f} {y + height:.2f} "
    f"H {x:.2f} Z"
  )


def _series_token(slot: int) -> str:
  """모델 순번을 카테고리 색 토큰 이름으로 바꾼다."""
  return SERIES_TOKENS[slot] if 0 <= slot < len(SERIES_TOKENS) else SERIES_FALLBACK


def svg_bar_chart(
  labels: Sequence[str],
  values: Sequence[Any],
  slots: Sequence[int],
  ref_value: Any,
  ref_label: str,
  digits: int,
  axis_label: str,
  title: str,
) -> str:
  """모델별 가로 막대 차트를 inline SVG 문자열로 만든다.

  값이 없는 모델은 막대 없이 '결과 없음' 으로 표시하고, `ref_value` 가 있으면
  Persistence 기준선을 점선으로 겹쳐 그린다.
  """
  width, left, right, top = 560.0, 110.0, 76.0, 14.0
  row_h, bar_h, axis_h, legend_h = 46.0, 24.0, 38.0, 26.0
  plot_w = width - left - right
  rows = max(len(labels), 1)
  plot_h = row_h * rows
  height = top + plot_h + axis_h + legend_h

  numbers = [_as_number(v) for v in values]
  ref = _as_number(ref_value)
  candidates = [v for v in numbers if v is not None]
  if ref is not None:
    candidates.append(ref)
  vmax = max(candidates) if candidates else 1.0
  if vmax <= 0:
    vmax = 1.0
  # 눈금은 실제 최대값 기준으로 잡고, 값 라벨이 들어갈 여백만 축 끝에 더한다.
  ticks = _nice_ticks(0.0, vmax, 4)
  scale_max = max(ticks[-1], vmax * 1.06)
  tick_digits = _tick_digits(ticks[1] - ticks[0] if len(ticks) > 1 else scale_max)

  def sx(value: float) -> float:
    """값을 x 좌표로 바꾼다."""
    return left + (value / scale_max) * plot_w

  parts: list[str] = [
    f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
    f'preserveAspectRatio="xMinYMin meet" style="max-width: {width:.0f}px">',
    f"<title>{esc(title)}</title>",
  ]
  for tick in ticks:
    x = sx(tick)
    parts.append(
      f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{top + plot_h:.2f}" '
      f'stroke="var(--grid)" stroke-width="1" />'
    )
    parts.append(
      f'<text x="{x:.2f}" y="{top + plot_h + 17:.2f}" text-anchor="middle" '
      f'font-size="11" fill="var(--muted)">{esc(_fmt_tick(tick, tick_digits))}</text>'
    )
  parts.append(
    f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + plot_h:.2f}" '
    f'stroke="var(--axis)" stroke-width="1" />'
  )
  parts.append(
    f'<text x="{left + plot_w / 2:.2f}" y="{top + plot_h + 34:.2f}" text-anchor="middle" '
    f'font-size="12" fill="var(--text-secondary)">{esc(axis_label)}</text>'
  )

  for index in range(rows):
    label = labels[index] if index < len(labels) else ""
    number = numbers[index] if index < len(numbers) else None
    slot = slots[index] if index < len(slots) else index
    y = top + index * row_h + (row_h - bar_h) / 2
    parts.append(
      f'<text x="{left - 12:.2f}" y="{y + bar_h / 2:.2f}" text-anchor="end" '
      f'dominant-baseline="central" font-size="13" fill="var(--text-primary)">{esc(label)}</text>'
    )
    if number is None:
      parts.append(
        f'<text x="{left + 10:.2f}" y="{y + bar_h / 2:.2f}" dominant-baseline="central" '
        f'font-size="12" fill="var(--muted)">{NO_RESULT}</text>'
      )
      continue
    bar_w = max(sx(number) - left, 1.0)
    parts.append(
      f'<path d="{_bar_path(left, y, bar_w, bar_h)}" fill="var({_series_token(slot)})" />'
    )
    parts.append(
      f'<text x="{left + bar_w + 8:.2f}" y="{y + bar_h / 2:.2f}" dominant-baseline="central" '
      f'font-size="12" fill="var(--text-secondary)">{esc(fmt_float(number, digits))}</text>'
    )

  legend_y = height - 8.0
  if ref is not None:
    x = sx(ref)
    parts.append(
      f'<line x1="{x:.2f}" y1="{top - 4:.2f}" x2="{x:.2f}" y2="{top + plot_h + 4:.2f}" '
      f'stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 4" />'
    )
    parts.append(
      f'<line x1="{left:.2f}" y1="{legend_y - 4:.2f}" x2="{left + 22:.2f}" y2="{legend_y - 4:.2f}" '
      f'stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 4" />'
    )
    parts.append(
      f'<text x="{left + 30:.2f}" y="{legend_y:.2f}" font-size="12" fill="var(--text-secondary)">'
      f'{esc(ref_label)} {esc(fmt_float(ref, digits))}</text>'
    )
  else:
    parts.append(
      f'<text x="{left:.2f}" y="{legend_y:.2f}" font-size="12" fill="var(--muted)">'
      f'{esc(ref_label)} {DASH}</text>'
    )
  parts.append("</svg>")
  return "".join(parts)


def svg_line_chart(series: Sequence[dict[str, Any]], x_label: str, y_label: str, title: str) -> str:
  """모델별 학습/검증 손실 꺾은선 차트를 inline SVG 문자열로 만든다.

  `series` 각 항목은 `{"name": str, "slot": int, "loss": list, "val_loss": list}` 이다.
  실선이 val_loss, 점선이 loss 이며 축은 하나만 쓴다.
  """
  width, left, right, top = 900.0, 78.0, 22.0, 16.0
  plot_h, axis_h, legend_h = 260.0, 34.0, 48.0
  plot_w = width - left - right
  height = top + plot_h + axis_h + legend_h

  clean: list[dict[str, Any]] = []
  for item in series:
    loss = [v for v in (_as_number(x) for x in item.get("loss") or []) if v is not None]
    val_loss = [v for v in (_as_number(x) for x in item.get("val_loss") or []) if v is not None]
    if loss or val_loss:
      clean.append({"name": item.get("name", ""), "slot": int(item.get("slot", 0)),
                    "loss": loss, "val_loss": val_loss})

  if not clean:
    # 그릴 게 없으면 빈 공간을 크게 남기지 않고 낮은 안내용 SVG 를 준다.
    return (
      f'<svg viewBox="0 0 {width:.0f} 96" role="img" preserveAspectRatio="xMinYMin meet" '
      f'style="max-width: {width:.0f}px"><title>{esc(title)}</title>'
      f'<text x="{width / 2:.2f}" y="52" text-anchor="middle" font-size="13" '
      f'fill="var(--muted)">{NO_RESULT}</text></svg>'
    )

  parts: list[str] = [
    f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
    f'preserveAspectRatio="xMinYMin meet" style="max-width: {width:.0f}px">',
    f"<title>{esc(title)}</title>",
  ]

  n_epochs = max(max(len(s["loss"]), len(s["val_loss"])) for s in clean)
  all_values = [v for s in clean for v in s["loss"] + s["val_loss"]]
  ticks = _nice_ticks(min(all_values), max(all_values), 4)
  ymin, ymax = ticks[0], ticks[-1]
  if ymax <= ymin:
    ymax = ymin + 1.0
  tick_digits = _tick_digits(ticks[1] - ticks[0] if len(ticks) > 1 else ymax)

  def sx(epoch: int) -> float:
    """1-based epoch 을 x 좌표로 바꾼다."""
    if n_epochs <= 1:
      return left + plot_w / 2
    return left + (epoch - 1) / (n_epochs - 1) * plot_w

  def sy(value: float) -> float:
    """손실 값을 y 좌표로 바꾼다."""
    return top + plot_h - (value - ymin) / (ymax - ymin) * plot_h

  for tick in ticks:
    y = sy(tick)
    parts.append(
      f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + plot_w:.2f}" y2="{y:.2f}" '
      f'stroke="var(--grid)" stroke-width="1" />'
    )
    parts.append(
      f'<text x="{left - 10:.2f}" y="{y:.2f}" text-anchor="end" dominant-baseline="central" '
      f'font-size="11" fill="var(--muted)">{esc(_fmt_tick(tick, tick_digits))}</text>'
    )
  parts.append(
    f'<line x1="{left:.2f}" y1="{top + plot_h:.2f}" x2="{left + plot_w:.2f}" '
    f'y2="{top + plot_h:.2f}" stroke="var(--axis)" stroke-width="1" />'
  )

  every = max(1, math.ceil(n_epochs / 10))
  for epoch in range(1, n_epochs + 1):
    if epoch != 1 and epoch != n_epochs and epoch % every:
      continue
    parts.append(
      f'<text x="{sx(epoch):.2f}" y="{top + plot_h + 18:.2f}" text-anchor="middle" '
      f'font-size="11" fill="var(--muted)">{epoch}</text>'
    )
  parts.append(
    f'<text x="{left + plot_w / 2:.2f}" y="{top + plot_h + 34:.2f}" text-anchor="middle" '
    f'font-size="12" fill="var(--text-secondary)">{esc(x_label)}</text>'
  )
  cy = top + plot_h / 2
  parts.append(
    f'<text x="16" y="{cy:.2f}" text-anchor="middle" font-size="12" '
    f'fill="var(--text-secondary)" transform="rotate(-90 16 {cy:.2f})">{esc(y_label)}</text>'
  )

  for item in clean:
    color = f"var({_series_token(item['slot'])})"
    for key, dash in (("loss", ' stroke-dasharray="6 5"'), ("val_loss", "")):
      points = item[key]
      if not points:
        continue
      coords = " ".join(f"{sx(i + 1):.2f},{sy(v):.2f}" for i, v in enumerate(points))
      parts.append(
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"{dash} />'
      )
      if len(points) == 1 or key == "val_loss":
        parts.append(
          f'<circle cx="{sx(len(points)):.2f}" cy="{sy(points[-1]):.2f}" r="4" fill="{color}" '
          f'stroke="var(--surface-1)" stroke-width="2" />'
        )

  key_y = top + plot_h + axis_h + 16.0
  step = min(170.0, plot_w / max(len(clean), 1))
  for index, item in enumerate(clean):
    x = left + index * step
    color = f"var({_series_token(item['slot'])})"
    parts.append(
      f'<line x1="{x:.2f}" y1="{key_y - 4:.2f}" x2="{x + 22:.2f}" y2="{key_y - 4:.2f}" '
      f'stroke="{color}" stroke-width="2" stroke-linecap="round" />'
    )
    parts.append(
      f'<text x="{x + 30:.2f}" y="{key_y:.2f}" font-size="12" fill="var(--text-primary)">'
      f'{esc(item["name"])}</text>'
    )
  parts.append(
    f'<text x="{left:.2f}" y="{key_y + 20:.2f}" font-size="11" fill="var(--text-secondary)">'
    f'실선 = 검증 손실(val_loss) · 점선 = 학습 손실(loss)</text>'
  )
  parts.append("</svg>")
  return "".join(parts)


# ---------------------------------------------------------------------------
# HTML 조립
# ---------------------------------------------------------------------------

def _model_names(results: Sequence[dict[str, Any]]) -> list[str]:
  """표시할 모델 이름 목록 (기본 3종 + 결과에만 있는 모델을 이름순으로)."""
  extras = sorted({str(r["model"]) for r in results} - set(MODEL_ORDER))
  return list(MODEL_ORDER) + extras


def _label(name: str) -> str:
  """디렉터리 이름을 화면 표기로 바꾼다."""
  return MODEL_LABELS.get(name, name.replace("_", "-"))


def _swatch(slot: int) -> str:
  """모델 색 스와치 span."""
  return f'<span class="swatch" style="background: var({_series_token(slot)})"></span>'


def _is_plain_filename(name: Any) -> bool:
  """스키마상 figures 값은 모델 디렉터리 기준 파일명 하나다 (경로 이탈 차단)."""
  return (
    isinstance(name, str)
    and bool(name)
    and name not in (".", "..")
    and "/" not in name
    and "\\" not in name
  )


def _figure(result: dict[str, Any] | None, key: str, caption: str) -> str:
  """metrics.figures[key] 그림을 base64 로 넣거나 자리 표시를 만든다.

  `key` 는 FIGURE_KEYS 중 하나여야 한다 (파이프라인의 figures 계약과 같은 집합).
  """
  uri = None
  if key not in FIGURE_KEYS:
    logger.warning("알 수 없는 그림 키라 무시한다: %r", key)
    result = None
  if result is not None:
    name = _get(result, "figures", key)
    if _is_plain_filename(name):
      uri = encode_image(Path(result["_dir"]) / str(name))
    elif name is not None:
      logger.warning("figures.%s 값이 파일명이 아니라 무시한다: %r", key, name)
  if uri is None:
    return (
      f'<figure><div class="placeholder">{NO_FIGURE}</div>'
      f'<figcaption>{esc(caption)}</figcaption></figure>'
    )
  return (
    f'<figure><img src="{uri}" alt="{esc(caption)}" />'
    f'<figcaption>{esc(caption)}</figcaption></figure>'
  )


def _canonical(value: Any) -> Any:
  """비교용 정규화: tuple 을 list 로 바꿔 JSON 값과 같은 모양으로 만든다."""
  if isinstance(value, (list, tuple)):
    return [_canonical(v) for v in value]
  if isinstance(value, dict):
    return {k: _canonical(v) for k, v in sorted(value.items())}
  return value


def find_condition_mismatches(results: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
  """첫 결과를 기준으로 실행 조건이 다른 모델과 그 키 이름을 찾는다.

  `config` 는 CONDITION_CONFIG_KEYS, `data` 는 CONDITION_DATA_KEYS 만 본다 (epochs·batch·lr
  처럼 모델별로 달라도 되는 값은 제외). 반환값은 (모델 이름, 다른 키 목록) 목록이며 모든
  결과가 같은 조건이면 빈 리스트다. 결과가 2개 미만이면 비교 대상이 없으므로 빈 리스트다.
  """
  if len(results) < 2:
    return []
  base = results[0]
  mismatches: list[tuple[str, list[str]]] = []
  for result in results[1:]:
    diff = [key for key in CONDITION_CONFIG_KEYS
            if _canonical(_get(base, "config", key)) != _canonical(_get(result, "config", key))]
    diff += [key for key in CONDITION_DATA_KEYS
             if _canonical(_get(base, "data", key)) != _canonical(_get(result, "data", key))]
    if diff:
      mismatches.append((str(result.get("model", "?")), diff))
  return mismatches


def _mismatch_banner(base_name: str, mismatches: Sequence[tuple[str, list[str]]]) -> str:
  """실행 조건 불일치 경고 배너 (일치하면 빈 문자열)."""
  if not mismatches:
    return ""
  detail = " / ".join(
    f"{_label(name)}: {', '.join(keys)}" for name, keys in mismatches
  )
  return (
    '<div class="banner" role="note">'
    "<strong>실행 조건이 모델마다 다르다.</strong> "
    f"아래 표와 차트의 값은 서로 다른 조건에서 나온 것이므로 모델끼리 직접 비교할 수 없고 "
    f"모델별로 따로 읽어야 한다. 1절 데이터 요약은 {esc(_label(base_name))} 기준이다."
    f'<span class="detail">{esc(_label(base_name))} 과(와) 다른 항목 — {esc(detail)}</span>'
    "</div>"
  )


def _summary_section(results: Sequence[dict[str, Any]]) -> str:
  """데이터 요약(기간, 프레임 수, 분할) 정의 목록."""
  data = _get(results[0], "data") if results else None
  if not isinstance(data, dict):
    return f'<p class="lead">{NO_RESULT} — 데이터 요약을 만들 metrics.json 이 없다.</p>'
  segments = data.get("segments")
  seg_text = DASH
  if isinstance(segments, list) and segments:
    seg_text = ", ".join(
      f"{fmt_scalar(seg[0])}–{fmt_scalar(seg[1])}"
      for seg in segments if isinstance(seg, (list, tuple)) and len(seg) >= 2
    ) or DASH
  items = [
    ("관측 기간", fmt_scalar(data.get("period"))),
    ("프레임 수", fmt_int(data.get("n_frames"))),
    ("연속 세그먼트", f"{seg_text} (총 {len(segments) if isinstance(segments, list) else 0}개)"),
    ("학습 구간", f"{fmt_scalar(data.get('train_period'))} · {fmt_int(data.get('n_train'))} 샘플"),
    ("검증 구간", f"{fmt_scalar(data.get('val_period'))} · {fmt_int(data.get('n_val'))} 샘플"),
    ("정규화 범위", f"{fmt_scalar(data.get('gmin'))} ~ {fmt_scalar(data.get('gmax'))}"),
  ]
  cells = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in items)
  return f'<dl class="summary">{cells}</dl>'


def _comparison_table(names: Sequence[str], by_model: dict[str, dict[str, Any]]) -> str:
  """핵심 비교표 (없는 모델은 '결과 없음' 행)."""
  headers = [
    "모델", "파라미터", "학습 epoch", "초/epoch", "val MAE", "val SSIM",
    "전체 프레임 MAE", "전체 프레임 SSIM", "Persistence 대비 MAE 개선", "GPU",
  ]
  head = "".join(f"<th scope=\"col\">{esc(h)}</th>" for h in headers)
  rows: list[str] = []
  for slot, name in enumerate(names):
    result = by_model.get(name)
    cell_name = f'<th scope="row"><span class="name">{_swatch(slot)}{esc(_label(name))}</span></th>'
    if result is None:
      rows.append(
        f'<tr>{cell_name}<td class="empty" colspan="{len(headers) - 1}">{NO_RESULT}</td></tr>'
      )
      continue
    values = [
      fmt_int(result.get("params")),
      fmt_int(_get(result, "train", "epochs_run")),
      fmt_seconds(_get(result, "train", "sec_per_epoch")),
      fmt_mae(_get(result, "val", "model_mae")),
      fmt_ssim(_get(result, "val", "model_ssim")),
      fmt_mae(_get(result, "full_frame", "model_mae")),
      fmt_ssim(_get(result, "full_frame", "model_ssim")),
      fmt_pct(_get(result, "val", "mae_gain_pct")),
      fmt_scalar(_get(result, "env", "gpu")),
    ]
    rows.append(f"<tr>{cell_name}" + "".join(f"<td>{esc(v)}</td>" for v in values) + "</tr>")
  return (
    '<div class="table-wrap"><table>'
    f"<caption>MAE 는 낮을수록, SSIM 은 높을수록 좋다. 값은 정규화된 밝기 기준이며 "
    f"'전체 프레임' 은 250×250 원본 해상도 1스텝 예측이다.</caption>"
    f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
  )


def _charts_section(names: Sequence[str], by_model: dict[str, dict[str, Any]]) -> str:
  """막대 차트 2개(val MAE, val SSIM)를 만든다."""
  labels = [_label(n) for n in names]
  slots = list(range(len(names)))
  mae_values = [_get(by_model.get(n) or {}, "val", "model_mae") for n in names]
  ssim_values = [_get(by_model.get(n) or {}, "val", "model_ssim") for n in names]
  pers_mae = next((v for v in (_get(by_model.get(n) or {}, "val", "pers_mae") for n in names)
                   if _as_number(v) is not None), None)
  pers_ssim = next((v for v in (_get(by_model.get(n) or {}, "val", "pers_ssim") for n in names)
                    if _as_number(v) is not None), None)
  mae_chart = svg_bar_chart(labels, mae_values, slots, pers_mae, "Persistence 기준선",
                            5, "검증 MAE (낮을수록 좋다)", "모델별 검증 MAE")
  ssim_chart = svg_bar_chart(labels, ssim_values, slots, pers_ssim, "Persistence 기준선",
                             4, "검증 SSIM (높을수록 좋다)", "모델별 검증 SSIM")
  return (
    '<div class="chart-grid">'
    f'<figure class="chart"><figcaption>검증 MAE — 낮을수록 좋다</figcaption>'
    f'<div class="chart-scroll">{mae_chart}</div></figure>'
    f'<figure class="chart"><figcaption>검증 SSIM — 높을수록 좋다</figcaption>'
    f'<div class="chart-scroll">{ssim_chart}</div></figure>'
    "</div>"
  )


def _history_section(names: Sequence[str], by_model: dict[str, dict[str, Any]]) -> str:
  """epoch 별 손실 꺾은선 차트를 만든다."""
  series: list[dict[str, Any]] = []
  for slot, name in enumerate(names):
    result = by_model.get(name)
    if result is None:
      continue
    history = _get(result, "train", "history") or {}
    series.append({
      "name": _label(name),
      "slot": slot,
      "loss": history.get("loss") or [],
      "val_loss": history.get("val_loss") or [],
    })
  chart = svg_line_chart(series, "epoch", "손실", "모델별 epoch 손실 곡선")
  return f'<figure class="chart"><div class="chart-scroll">{chart}</div></figure>'


def _kv_table(title: str, data: Any, keys: Sequence[str]) -> str:
  """env/config dict 를 키-값 목록으로 렌더한다."""
  source = data if isinstance(data, dict) else {}
  rows = "".join(
    f"<dt>{esc(k)}</dt><dd>{esc(fmt_scalar(source.get(k)))}</dd>" for k in keys
  )
  return f'<p class="kv-title">{esc(title)}</p><dl class="kv">{rows}</dl>'


def _model_cards(names: Sequence[str], by_model: dict[str, dict[str, Any]]) -> str:
  """모델별 카드 (전체 프레임 예측 그림, 학습 곡선 그림, env·config)."""
  env_keys = ("colab", "platform", "python", "tensorflow", "keras", "gpu", "precision_policy")
  cfg_keys = ("in_frames", "target", "patch", "stride", "filters", "epochs", "batch", "lr",
              "hours", "seed")
  cards: list[str] = []
  for slot, name in enumerate(names):
    result = by_model.get(name)
    head = (
      f'<div class="card-head"><h3><span class="name">{_swatch(slot)}{esc(_label(name))}</span></h3>'
    )
    if result is None:
      cards.append(
        f'<article class="card">{head}<span class="tag">{NO_RESULT}</span></div>'
        f'<p class="lead">{NO_RESULT} — `results/{esc(name)}/metrics.json` 이 없다. '
        f'해당 모델 스크립트를 실행한 뒤 리포트를 다시 만들면 채워진다.</p></article>'
      )
      continue
    created = fmt_scalar(result.get("created_at"))
    body = [
      head + f'<span class="tag">{esc(created)}</span></div>',
      _figure(result, "full_frame", f"{_label(name)} 전체 프레임(250×250) 예측"),
      _figure(result, "history", f"{_label(name)} 학습 곡선"),
      _kv_table("실행 환경", result.get("env"), env_keys),
      _kv_table("하이퍼파라미터", result.get("config"), cfg_keys),
    ]
    cards.append(f'<article class="card">{"".join(body)}</article>')
  return f'<div class="cards">{"".join(cards)}</div>'


def _data_figures(results: Sequence[dict[str, Any]]) -> str:
  """samples.png, hourly_mean.png 를 첫 모델 것으로 한 번만 보여준다."""
  first = results[0] if results else None
  source = f" (출처: {_label(str(first['model']))} 실행)" if first is not None else ""
  return (
    '<div class="chart-grid">'
    + _figure(first, "samples", f"입력 시퀀스와 정답 샘플{source}")
    + _figure(first, "hourly_mean", f"시간대별 평균 밝기{source}")
    + "</div>"
  )


def _notes_section(consistent: bool = True) -> str:
  """방법론 각주. `consistent` 가 False 면 첫 항목을 조건 불일치 문장으로 바꾼다."""
  first = (
    "세 모델 모두 같은 프레임 캐시·같은 시간 분할(앞 구간 학습, 뒤 구간 검증)·같은 정규화 "
    "범위(gmin~gmax)를 쓴다. 분할과 정규화는 공통 모듈에서만 정의한다."
    if consistent else
    "실행 조건이 모델마다 다르다 (metrics.json 의 config·data 절이 어긋난다). 프레임 캐시·"
    "시간 분할·정규화 범위가 같다는 전제가 깨졌으므로 표와 차트는 모델별로 따로 읽어야 하고 "
    "모델끼리 직접 비교하면 안 된다."
  )
  items = [
    first,
    "출력은 residual head 다: 예측 = 마지막 입력 프레임 + Δ. 모델은 Δ 만 학습한다.",
    "손실은 세 모델 공통으로 0.5·MAE + 0.5·(1 − SSIM) 이며, 혼합 정밀도에서도 Δ 와 출력은 "
    "float32 로 계산한다.",
    "Persistence 베이스라인은 '다음 프레임 = 마지막 입력 프레임' 이다. 같은 검증 표본에서 "
    "계산하므로 모델 값과 직접 비교할 수 있고, 차트의 점선 기준선이 그 값이다.",
    "MAE 개선율 = (Persistence MAE − 모델 MAE) / Persistence MAE × 100.",
    "MAE·SSIM 은 정규화된 밝기 기준이며, 전체 프레임 지표는 250×250 원본 해상도로 옮긴 "
    "가중치로 1스텝 예측해 계산한다.",
  ]
  return '<ul class="notes">' + "".join(f"<li>{esc(t)}</li>" for t in items) + "</ul>"


def _section(title: str, subtitle: str, body: str) -> str:
  """제목·부제·본문을 가진 섹션 하나를 만든다."""
  return (
    f'<section><div class="section-head"><h2>{esc(title)}</h2>'
    f"<p>{esc(subtitle)}</p></div>{body}</section>"
  )


def render_html(results: list[dict[str, Any]], generated_at: str) -> str:
  """metrics 목록으로 자기완결 HTML 문서 한 장을 만든다.

  결과가 없어도 항상 문서를 만들고, 없는 모델은 '결과 없음' 으로 표시한다.
  외부 리소스(스크립트/스타일시트/이미지 URL)는 쓰지 않는다.
  """
  mismatches = find_condition_mismatches(results)
  if mismatches:
    for name, keys in mismatches:
      logger.warning("실행 조건이 기준 모델(%s)과 다르다 — %s: %s",
                     str(results[0].get("model", "?")), name, ", ".join(keys))
  banner = _mismatch_banner(str(results[0].get("model", "?")), mismatches) if results else ""
  by_model = {str(r["model"]): r for r in results}
  names = _model_names(results)
  done = [n for n in names if n in by_model]
  subtitle = (
    f"결과 {len(done)}/{len(names)}개 · " + (", ".join(_label(n) for n in done) if done else NO_RESULT)
  )
  body = "".join([
    "<header>",
    f"<h1>{esc(PAGE_TITLE)}</h1>",
    f'<p class="meta">GK2A AMI sw038 다음 프레임 예측 · {esc(subtitle)} · '
    f"생성 시각 {esc(generated_at)}</p>",
    "</header>",
    banner,
    _section("1. 데이터 요약", "세 모델이 공유하는 입력 데이터와 분할", _summary_section(results)),
    _section("2. 핵심 비교", "학습 비용과 검증·전체 프레임 성능", _comparison_table(names, by_model)),
    _section("3. 검증 지표 비교", "점선은 Persistence 베이스라인 기준선이다",
             _charts_section(names, by_model)),
    _section("4. 학습 곡선", "epoch 별 학습·검증 손실 (축 하나)", _history_section(names, by_model)),
    _section("5. 모델별 결과", "전체 프레임 예측, 학습 곡선, 실행 환경", _model_cards(names, by_model)),
    _section("6. 데이터 그림", "모델과 무관한 공통 데이터 그림", _data_figures(results)),
    _section("7. 방법론", "비교가 공정한 이유", _notes_section(not mismatches)),
    f'<footer>외부 리소스 없이 생성된 단일 HTML · tools/build_report.py · {esc(generated_at)}</footer>',
  ])
  return (
    "<!doctype html>\n"
    '<html lang="ko">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    f"<title>{esc(PAGE_TITLE)}</title>\n"
    f"<style>{CSS}</style>\n"
    "</head>\n<body>\n"
    f"<main>{body}</main>\n"
    "</body>\n</html>\n"
  )


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
  """CLI 진입점: results 를 읽어 자기완결 HTML 을 쓰고 0 을 돌려준다."""
  parser = argparse.ArgumentParser(
    description="results/<Model>/metrics.json 을 모아 site/index.html 을 만든다.")
  parser.add_argument("--results-dir", type=Path, default=Path("results"),
                      help="모델별 결과 디렉터리 (기본: results)")
  parser.add_argument("--out", type=Path, default=Path("site/index.html"),
                      help="생성할 HTML 경로 (기본: site/index.html)")
  args = parser.parse_args(argv)

  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  results = load_results(args.results_dir)
  found = ", ".join(str(r["model"]) for r in results) or NO_RESULT
  logger.info("결과 %d개: %s", len(results), found)

  generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
  page = render_html(results, generated_at)
  try:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
  except OSError as exc:
    logger.error("HTML 을 쓰지 못했다 (%s): %s", args.out, exc)
    return 1
  logger.info("생성 완료: %s (%.1f KB)", args.out, len(page.encode("utf-8")) / 1024)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
