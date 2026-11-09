"""노트북 검증: nbformat 스키마, stderr/error 출력, 그림 수, 지정 키워드가 포함된 셀 출력."""
import json
import sys

import nbformat


def main(path: str, keys: list[str]) -> int:
  nb = json.load(open(path, encoding="utf-8"))
  nbformat.validate(nbformat.read(path, as_version=4))
  images = errors = 0
  for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
      continue
    text = ""
    for out in cell["outputs"]:
      if "image/png" in out.get("data", {}):
        images += 1
      if out.get("name") == "stderr" or out.get("output_type") == "error":
        errors += 1
        print("STDERR cell", i, "".join(out.get("text", ""))[:250])
      text += "".join(out.get("text", "")) if "text" in out else "".join(out.get("data", {}).get("text/plain", ""))
    if any(k in text for k in keys):
      print("----- cell", i)
      print(text.strip()[:700])
  print(f"{path} | validate OK | images: {images} | stderr/error: {errors}")
  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1], sys.argv[2:]))
