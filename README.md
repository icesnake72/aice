# GK2A 위성영상 다음 프레임 예측 — 모델 비교

> 이 저장소에는 AICE 학습용 노트북도 함께 있다. 아래 내용은 위성영상 예측 파이프라인에만 해당한다.

GK2A(천리안위성 2A) AMI `sw038` 위성영상으로 **2분 뒤 프레임 한 장**을 예측하고,
ConvLSTM · SimVP · PredRNN-V2 세 모델을 같은 데이터·손실·평가로 비교해
자기완결 HTML 리포트 한 장으로 만들어 Netlify 에 배포한다.

---

## 1. 빠른 시작

```bash
# 데이터 준비 (2025-10-17 하루치, 710장)
python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out resource/netcdf

# 세 모델 학습 (TensorFlow 가 설치된 인터프리터로 실행). 결과는 results/<Model>/
/usr/local/bin/python3 nc_predict_colab.py         --epochs 4
/usr/local/bin/python3 simvp_predict_colab.py      --epochs 4
/usr/local/bin/python3 predrnn_v2_predict_colab.py --epochs 4

# 리포트 생성 -> site/index.html
python3 tools/build_report.py

# 테스트 (데이터 없이 실행 가능)
/usr/local/bin/python3 -m pytest tests -q
```

빠른 확인은 `--hours 23 --epochs 1` 을 붙이고, `--out-dir` 을 임시 디렉터리로 주면 `results/` 를 건드리지 않는다.

---

## 2. 파일 구조

| 경로 | 역할 |
| --- | --- |
| `nc_pipeline.py` | 공통 파이프라인: 데이터 적재·세그먼트·정규화·데이터셋·손실·residual head·학습·평가·`metrics.json` |
| `nc_predict_colab.py` | ConvLSTM 엔트리 (`MODEL_NAME` + `build_model`) |
| `simvp_predict_colab.py` | SimVP 엔트리 |
| `predrnn_v2_predict_colab.py` | PredRNN-V2 엔트리 |
| `tools/build_colab_notebook.py` | `.py` 를 셀로 잘라 `.ipynb` 생성 (`--model`, `--profile`, `--all`, `--root`) |
| `tools/build_report.py` | `results/*/metrics.json` + png → `site/index.html` |
| `tests/` | 파이프라인·모델 3종·노트북 생성기·리포트 테스트 |
| `*_prediction.ipynb`, `*_prediction_colab.ipynb` | 실행 노트북. `ConvLSTM_prediction.ipynb` 외에는 생성물이라 직접 고치지 않는다 |
| `results/<Model>/` | 실행 결과: `metrics.json`, 그림 4장, `train_log.csv`, 가중치 (가중치·캐시는 `.gitignore`) |
| `site/index.html` | 자기완결 비교 리포트 (외부 JS/CSS 없음). Netlify 가 이 디렉터리를 서빙한다 |
| `doc/` | 설계·원리·실행 문서 |

---

## 3. 결과와 배포

로컬 결과는 `results/<Model>/`, Colab 결과는 Drive `MyDrive/nc_predict_output/<Model>/` 에 쌓인다.
Colab 결과는 그 폴더를 `results/<Model>/` 로 복사한 뒤 `python3 tools/build_report.py` 를 돌린다.

`netlify.toml` 이 `publish = "site"` 를 지정하므로, Netlify 에서 Add new site > Import an existing project 로
이 저장소를 연결하면 build command 없이 `site/` 가 그대로 배포되고 이후 push 마다 자동 재배포된다.

---

## 4. 더 읽을 문서

| 알고 싶은 것 | 문서 |
| --- | --- |
| 세 모델 구조·적응 사항·실행·배포 절차 | [doc/model_comparison.md](doc/model_comparison.md) |
| 데이터 전처리·분할·평가가 왜 그렇게 되어 있는지 | [doc/nc_predict_pipeline.md](doc/nc_predict_pipeline.md) |
| ConvLSTM 자체의 원리 | [doc/convlstm_principles.md](doc/convlstm_principles.md) |
| SSIM 지표의 의미 | [doc/ssim_explained.md](doc/ssim_explained.md) |
