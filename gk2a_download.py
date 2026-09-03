"""GK2A AMI L1B NetCDF downloader (AWS Open Data, anonymous access).

Bucket : s3://noaa-gk2a-pds  (us-east-1, no credentials required)
Layout : AMI/L1B/{FD|LA}/{YYYYMM}/{DD}/{HH}/gk2a_ami_le1b_{ch}_{area}_{YYYYMMDDHHMM}.nc

Example:
  python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out ~/Documents/practice/data/NetCDF_1day
  python3 gk2a_download.py --date 2025-10-17 --hours 15 16 17 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

BUCKET_URL = "https://noaa-gk2a-pds.s3.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
CHANNELS = (
  "vi004", "vi005", "vi006", "vi008", "nr013", "nr016", "sw038",
  "wv063", "wv069", "wv073", "ir087", "ir096", "ir105", "ir112", "ir123", "ir133",
)
AREAS = ("LA", "FD")
TIMEOUT = 60
MAX_RETRY = 3

logger = logging.getLogger("gk2a")


@dataclass(frozen=True)
class S3Object:
  key: str
  size: int

  @property
  def name(self) -> str:
    return self.key.rsplit("/", 1)[-1]


def _http_get(url: str) -> bytes:
  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRY + 1):
    try:
      with urllib.request.urlopen(url, timeout=TIMEOUT) as res:
        return res.read()
    except Exception as exc:  # noqa: BLE001 - network errors are heterogeneous
      last_error = exc
      logger.debug("retry %d/%d for %s (%s)", attempt, MAX_RETRY, url, exc)
  raise RuntimeError(f"GET failed after {MAX_RETRY} retries: {url}") from last_error


def list_objects(prefix: str) -> list[S3Object]:
  """List every object under prefix, following S3 pagination."""
  objects: list[S3Object] = []
  token: str | None = None
  while True:
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    if token:
      params["continuation-token"] = token
    body = _http_get(f"{BUCKET_URL}/?{urllib.parse.urlencode(params)}")
    root = ET.fromstring(body)
    for node in root.findall(f"{S3_NS}Contents"):
      key = node.findtext(f"{S3_NS}Key", default="")
      size = int(node.findtext(f"{S3_NS}Size", default="0"))
      if key.endswith(".nc"):
        objects.append(S3Object(key=key, size=size))
    if root.findtext(f"{S3_NS}IsTruncated") != "true":
      return objects
    token = root.findtext(f"{S3_NS}NextContinuationToken")
    if not token:
      return objects


def collect(target: date, area: str, channel: str, hours: list[int],
            interval: int) -> list[S3Object]:
  """Gather objects for the given day/hours, filtered by channel and time interval."""
  base = f"AMI/L1B/{area}/{target:%Y%m}/{target:%d}"
  found: list[S3Object] = []
  with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(list_objects, f"{base}/{h:02d}/"): h for h in hours}
    for future in as_completed(futures):
      hour = futures[future]
      try:
        found.extend(future.result())
      except RuntimeError as exc:
        logger.warning("hour %02d listing failed: %s", hour, exc)

  pattern = re.compile(rf"_{channel}_.*_(\d{{12}})\.nc$")
  selected: list[S3Object] = []
  for obj in found:
    match = pattern.search(obj.name)
    if not match:
      continue
    stamp = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    if stamp.minute % interval == 0:
      selected.append(obj)
  return sorted(selected, key=lambda o: o.name)


def download(obj: S3Object, out_dir: Path) -> tuple[str, bool]:
  """Download one object. Returns (name, downloaded) — False means skipped."""
  dest = out_dir / obj.name
  if dest.exists() and dest.stat().st_size == obj.size:
    return obj.name, False
  tmp = dest.with_suffix(".nc.part")
  data = _http_get(f"{BUCKET_URL}/{urllib.parse.quote(obj.key)}")
  tmp.write_bytes(data)
  tmp.replace(dest)
  return obj.name, True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="GK2A AMI L1B downloader")
  parser.add_argument("--date", required=True, help="관측 날짜 (YYYY-MM-DD)")
  parser.add_argument("--channel", default="sw038", choices=CHANNELS)
  parser.add_argument("--area", default="LA", choices=AREAS,
                      help="LA=한반도 500x500, FD=전구")
  parser.add_argument("--hours", nargs="*", type=int, default=list(range(24)),
                      help="UTC 시각 목록 (기본 0~23)")
  parser.add_argument("--interval", type=int, default=2,
                      help="분 간격 (2=전량, 10=1/5 샘플링)")
  parser.add_argument("--out", type=Path, required=False,
                      default=Path.home() / "Documents/practice/data/NetCDF_bulk")
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument("--dry-run", action="store_true", help="목록/용량만 출력")
  parser.add_argument("--verbose", action="store_true")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(levelname)s %(message)s",
  )

  try:
    target = datetime.strptime(args.date, "%Y-%m-%d").date()
  except ValueError:
    logger.error("날짜 형식 오류: %s (YYYY-MM-DD 필요)", args.date)
    return 2
  if not 1 <= args.interval <= 60:
    logger.error("interval 은 1~60 분이어야 한다: %d", args.interval)
    return 2

  hours = sorted({h for h in args.hours if 0 <= h <= 23})
  if not hours:
    logger.error("유효한 --hours 가 없다")
    return 2

  logger.info("조회 중: %s %s %s (%d시간)", target, args.area, args.channel, len(hours))
  objects = collect(target, args.area, args.channel, hours, args.interval)
  if not objects:
    logger.error("해당 조건의 파일이 없다 (날짜가 2023-02 이전이거나 미래인지 확인)")
    return 1

  total_mb = sum(o.size for o in objects) / 1e6
  logger.info("파일 %d개, 총 %.1f MB", len(objects), total_mb)
  logger.info("처음: %s / 마지막: %s", objects[0].name, objects[-1].name)
  if args.dry_run:
    return 0

  args.out.mkdir(parents=True, exist_ok=True)
  done, skipped, failed = 0, 0, 0
  with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(download, o, args.out): o for o in objects}
    for i, future in enumerate(as_completed(futures), start=1):
      obj = futures[future]
      try:
        _, fetched = future.result()
        done += fetched
        skipped += not fetched
      except RuntimeError as exc:
        failed += 1
        logger.warning("실패 %s: %s", obj.name, exc)
      if i % 50 == 0 or i == len(objects):
        logger.info("진행 %d/%d", i, len(objects))

  logger.info("완료: 신규 %d / 스킵 %d / 실패 %d → %s", done, skipped, failed, args.out)
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
