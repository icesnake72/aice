# ConvLSTM · SimVP · PredRNN-V2 성능 비교 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GK2A sw038 위성영상 다음 프레임 예측을 ConvLSTM, SimVP, PredRNN-V2 세 모델로 **동일한 데이터·손실·평가** 기준으로 학습하고, 결과를 하나의 HTML 리포트로 만들어 GitHub repo 에서 Netlify 로 바로 배포할 수 있게 한다.

**Architecture:** 기존 `nc_predict_colab.py` 의 데이터/학습/평가 코드를 공통 모듈 `nc_pipeline.py` 로 분리하고, 모델 파일 3개는 `build_model()` 만 정의하는 얇은 엔트리로 둔다. 각 실행은 `results/<Model>/metrics.json` 과 그림을 남기고, `tools/build_report.py` 가 이를 모아 자기완결(self-contained) HTML 한 장을 `site/index.html` 로 만든다. Colab 노트북은 `tools/build_colab_notebook.py` 가 공통 모듈 + 모델 파일을 셀로 합쳐 생성한다.

**Tech Stack:** Python 3.11, TensorFlow/Keras (로컬 TF 2.15/Keras 2 + Colab TF 2.21/Keras 3 양쪽 호환), numpy, xarray, matplotlib, pytest. 리포트는 외부 JS/CSS 없이 inline CSS + inline SVG.

**Spec:** 이 문서의 "0. 설계" 절이 spec 이다. 기존 구현의 근거는 `doc/nc_predict_pipeline.md`, `nc_predict_colab.py`(현재 HEAD) 를 본다.

## Global Constraints

- Python: 2-space indent, tab 금지, type hint 필수, 모든 함수에 docstring, 빈 except 금지, 민감정보 하드코딩 금지.
- 두 환경 모두에서 테스트와 smoke 가 통과해야 한다:
  - Keras 2: `/usr/local/bin/python3` (TF 2.15.0, Keras 2.15, Apple Metal GPU)
  - Keras 3: `/private/tmp/claude-501/-Users-eunbumkim-Desktop-02---------aice-test/91f49e57-8096-45a5-bcac-aa8a54edfa48/scratchpad/k3venv/bin/python` (TF 2.21.0, Keras 3.15.1, CPU)
- 모델 인터페이스는 정확히 `build_model(in_frames: int, filters: int, h: int, w: int, lr: float = 1e-3) -> keras.Model` (compile 완료 상태로 반환) 이고 모듈 상수 `MODEL_NAME: str` 을 가진다.
- 모든 모델의 입력은 `(B, in_frames, h, w, 1)` float32, 출력은 `(B, h, w, 1)` float32 이며 **출력 = 입력 마지막 프레임 + Δ** (잔차 head, `nc_pipeline.residual_head`) 이고 손실은 `nc_pipeline.ssim_mae_loss` (0.5·MAE + 0.5·(1−SSIM)) 로 통일한다. mixed_float16 정책에서도 Δ 와 출력은 float32 여야 한다.
- 학습 96×96 패치 모델과 추론 250×250 모델은 파라미터 수가 같아야 하고 `set_weights` 로 가중치를 옮길 수 있어야 한다.
- 실행 결과 디렉터리 레이아웃과 `metrics.json` 스키마는 "0.3" 절과 정확히 일치해야 한다.
- 데이터 분할·정규화·Persistence 베이스라인·평가 함수는 모델 간에 공유하며 모델 파일에서 재정의하지 않는다.
- smoke 명령: `MPLBACKEND=Agg <python> <model>_predict_colab.py --hours 23 --epochs 1 --out-dir <scratch 디렉터리>` 가 두 환경에서 exit 0. 데이터는 `--data-dir /Users/eunbumkim/Desktop/02_프로젝트_코드/aice_test/resource/netcdf` (worktree 에서 작업할 때 절대경로 필요). 테스트·smoke 는 절대 repo 의 `results/` 에 쓰지 않는다.
- `ConvLSTM_prediction.ipynb` (수작업 노트북) 는 수정하지 않는다.
- 커밋: `git add <파일 목록>` 만 사용 (`git add -A` 금지). 커밋 메시지 끝에 두 줄 trailer:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` 와
  `Claude-Session: https://claude.ai/code/session_01JL1YEWpmMHNyzY9KEaPJcJ`
- 설명·주석·문서는 한국어, 코드 식별자는 영어. md 문서는 `md-docs` 스킬 규칙(번호 H2, `---` 구분, 결론 우선 표).

---

## 0. 설계 (Spec)

### 0.1 파일 구조

| 파일 | 책임 | 작업 |
| --- | --- | --- |
| `nc_pipeline.py` | 공통: Config, 런타임 준비, 데이터 적재/캐시, 세그먼트, 정규화, 시각화, Persistence, 데이터셋, 손실, residual head, 학습/평가/전체 프레임 예측, metrics.json 저장, `run()` | Task 1 |
| `nc_predict_colab.py` | ConvLSTM 엔트리: `MODEL_NAME = "ConvLSTM"`, `build_model`, `main` | Task 1 (리팩토링) |
| `simvp_predict_colab.py` | SimVP 엔트리 | Task 2 |
| `predrnn_v2_predict_colab.py` | PredRNN-V2 엔트리 | Task 3 |
| `tools/build_colab_notebook.py` | `--model`, `--profile local\|colab`, `--all`: 공통 모듈 + 모델 파일 → 노트북 | Task 1 (확장) |
| `tools/build_report.py` | `results/*/metrics.json` + png → `site/index.html` | Task 4 |
| `netlify.toml` | `publish = "site"`, build command 없음 | Task 4 |
| `tests/test_nc_pipeline.py` | 기존 `tests/test_nc_predict_colab.py` 이관 + metrics 테스트 | Task 1 |
| `tests/test_build_colab_notebook.py` | 생성기 테스트 | Task 1 |
| `tests/test_model_convlstm.py`, `tests/test_model_simvp.py`, `tests/test_model_predrnn_v2.py` | 모델별 shape/param/가중치 이전/mixed precision 테스트 | Task 1 / 2 / 3 |
| `tests/test_build_report.py` | 리포트 생성 테스트 | Task 4 |
| `SimVP_prediction.ipynb`, `SimVP_prediction_colab.ipynb`, `PredRNN_V2_prediction.ipynb`, `PredRNN_V2_prediction_colab.ipynb`, `ConvLSTM_prediction_colab.ipynb` | 생성 노트북 (직접 수정 금지) | Task 1 / 2 / 3 |
| `doc/model_comparison.md`, `README.md` | 모델 소개·근거·실행·결과 수집·Netlify 배포 가이드 | Task 5 |

### 0.2 실행 결과 레이아웃

```
<out_dir>/                       로컬 기본 results/ , Colab 기본 /content/drive/MyDrive/nc_predict_output/
  cache/frames_sw038_t250.npz    모델 간 공유 캐시 (git 제외)
  <MODEL_NAME>/                  ConvLSTM | SimVP | PredRNN_V2
    metrics.json
    samples.png  hourly_mean.png  history.png  full_frame_prediction.png
    train_log.csv
    convlstm.weights.h5 등 가중치, checkpoint.weights.h5, pred_next.npy   (git 제외)
```

`.gitignore` 에 추가: `results/cache/`, `results/**/*.npz`, `results/**/*.h5`, `results/**/*.npy`, `results/**/*.part`.

### 0.3 `metrics.json` 스키마 (schema_version 1)

```json
{
  "schema_version": 1,
  "model": "SimVP",
  "params": 123456,
  "created_at": "2026-09-03T14:05:00+09:00",
  "env": {"colab": false, "platform": "macOS-15.2-arm64", "python": "3.11.4",
          "tensorflow": "2.15.0", "keras": "2.15.0", "gpu": "METAL", "precision_policy": "mixed_float16"},
  "config": {"in_frames": 4, "target": 250, "patch": 96, "stride": 77, "filters": 16,
             "epochs": 4, "batch": 16, "lr": 0.001, "hours": null, "seed": 42},
  "data": {"n_frames": 710, "period": "2025-10-17 00:00 ~ 23:58 UTC", "segments": [[0, 180], [180, 630], [630, 710]],
           "gmin": 14848.5, "gmax": 16361.5, "n_train": 5022, "n_val": 1224,
           "train_period": "00:00 ~ 19:00", "val_period": "19:02 ~ 23:58"},
  "baseline": {"mae": 0.00801, "ssim": 0.9560},
  "train": {"epochs_run": 4, "seconds": 906.2, "sec_per_epoch": 226.6,
            "history": {"loss": [0.0187, 0.0111], "val_loss": [0.0101, 0.0092], "mae": [0.0082, 0.0061], "val_mae": [0.0046, 0.0044]}},
  "val": {"model_mae": 0.00386, "model_ssim": 0.9866, "pers_mae": 0.00623, "pers_ssim": 0.9595, "mae_gain_pct": 38.1},
  "full_frame": {"model_mae": 0.00662, "model_ssim": 0.9728, "pers_mae": 0.01091, "pers_ssim": 0.9204,
                 "t_pred": "23:58", "inputs": "23:50~23:56"},
  "figures": {"samples": "samples.png", "hourly_mean": "hourly_mean.png", "history": "history.png", "full_frame": "full_frame_prediction.png"}
}
```

- `env.gpu`: `tf.config.experimental.get_device_details(gpus[0]).get("device_name")`, GPU 없으면 `"none"`. `env.keras`: `keras.__version__` 이 없으면(Keras 2 의 tf.keras) `import keras; keras.__version__`.
- `train.history` 의 키는 Keras 가 준 것 중 `loss, val_loss, mae, val_mae` 만 저장 (lr/learning_rate 제외). 값은 float 리스트.
- `val.mae_gain_pct = (pers_mae - model_mae) / pers_mae * 100`.
- `figures` 는 model 디렉터리 기준 상대 파일명. 파일이 실제로 있어야 한다.
- 숫자는 python float/int (numpy 타입 금지 — `json.dump` 실패 방지).

### 0.4 모델 스펙

공통: `filters` 는 "기본 hidden 폭"이며 모델마다 의미가 다르다(아래). 세 모델 모두 `nc_pipeline.residual_head(inp, delta)` 로 출력을 만들고 `nc_pipeline.compile_model(model, lr)` 로 컴파일한다. 파라미터 수는 `model.count_params()` 로 metrics 에 기록된다.

**ConvLSTM (기존 유지)**: ConvLSTM2D(filters, 3×3, return_sequences=True) → ConvLSTM2D(filters, 3×3) → Conv2D(1, 3×3, linear, float32) = Δ. 28,497 params (filters=16).

**SimVP** (Gao et al., CVPR 2022, arXiv:2206.05099. 공식 구현 github.com/A4Bear/SimVP 및 OpenSTL, Apache-2.0). 우리 적응: `aft_seq_length = 1` (다음 1장), 다운샘플 1회(N_S=2) 로 250 과 96 모두 호환, grouped conv 대신 일반 conv.
- `hid_S = filters` (16), `hid_T = 4 * filters` (64), `N_S = 2`, `N_T = 2`, `incep_ker = (3, 5, 7)`, `groups = 1`.
- Encoder (프레임별, `layers.TimeDistributed`): ConvSC ×2 = Conv2D(hid_S, 3, strides s, 'same') → GroupNormalization(groups=2) → LeakyReLU(0.2), strides `[1, 2]`. 첫 블록 출력을 `enc1` 로 보관(skip).
- Translator 입력: `(B, T, H/2, W/2, hid_S)` → Permute → Reshape `(H/2, W/2, T*hid_S)`.
- Inception 블록: Conv2D(hid_T//2, 1×1) → 병렬 [Conv2D(hid_T, k, 'same') → GroupNormalization(groups=8) → LeakyReLU(0.2)] for k in incep_ker → 합(sum). 번역기(Mid_Xnet)는 enc N_T개(첫 블록 입력 `T*hid_S`), dec N_T개(skip concat, 첫 dec 는 concat 없음, 마지막 dec 출력 채널 = `hid_S`, 즉 T_out=1).
- Decoder: ConvSC 역순 strides `[2, 1]`: 첫 블록 Conv2DTranspose(hid_S, 3, strides 2, 'same') → GN(2) → LeakyReLU; 둘째 블록 입력은 `concat(hid, enc1[:, -1])` (마지막 입력 프레임의 enc1) → Conv2D(hid_S, 3) → GN → LeakyReLU. readout Conv2D(1, 1×1, linear, float32) = Δ.
- 96 → 48 → 96, 250 → 125 → 250 이 정확히 맞는지 테스트로 확인.

**PredRNN-V2** (Wang et al., TPAMI 2022, arXiv:2103.09504. 공식 구현 github.com/thuml/predrnn-pytorch, MIT). 우리 적응: 단일 스텝 예측이라 reverse scheduled sampling 은 적용하지 않음(문서에 명시), `filter_size = 3` (공식 5; ConvLSTM 과 동일 조건), `patch_size = 2` (space_to_depth 로 96→48, 250→125, 채널 4), `num_layers = 2`, `num_hidden = filters` (16), `decouple_beta = 0.1`, LayerNorm 미사용.
- `STLSTMCell(num_hidden, filter_size)` 커스텀 Layer. 서브레이어: `conv_x` (7·hid), `conv_h` (4·hid), `conv_m` (3·hid), `conv_o` (hid, 입력 concat(c, m)), `conv_last` (1×1, hid). forget bias 1.0.
  ```
  i_x, f_x, g_x, i_x', f_x', g_x', o_x = split(conv_x(x), 7)
  i_h, f_h, g_h, o_h = split(conv_h(h), 4)
  i_m, f_m, g_m = split(conv_m(m), 3)
  i = σ(i_x + i_h); f = σ(f_x + f_h + 1); g = tanh(g_x + g_h); Δc = i·g; c' = f·c + Δc
  i' = σ(i_x' + i_m); f' = σ(f_x' + f_m + 1); g' = tanh(g_x' + g_m); Δm = i'·g'; m' = f'·m + Δm
  mem = concat(c', m'); o = σ(o_x + o_h + conv_o(mem)); h' = o · tanh(conv_last(mem))
  return h', c', m', Δc, Δm
  ```
- `PredRNNV2Core` Layer: 셀 `num_layers` 개 + 공유 adapter Conv2D(hid, 1×1, bias 없음). `call(x)` 에서 t = 0..T−1 을 Python 루프로 unroll (T 는 static). 상태 0 초기화 (`tf.shape(x)[0]` 로 batch). 메모리 M 은 zigzag: t 의 마지막 층 출력 m 이 t+1 의 첫 층 입력. 매 스텝·매 층에서 decoupling: `dc = normalize(adapter(Δc) reshaped (B, H·W, hid), axis=1)`, `dm` 동일, `cos = sum(dc·dm, axis=1)` → `mean(|cos|)`. 전체 평균 × `decouple_beta` 를 `self.add_loss(...)` 로 추가 (float32 로 cast). 출력은 마지막 스텝 마지막 층의 h.
- 앞뒤: `SpaceToDepth(2)` → core → Conv2D(4, 1×1) → `DepthToSpace(2)` → Conv2D(1, 1×1, linear, float32) = Δ. (`tf.nn.space_to_depth`/`depth_to_space` 를 감싼 커스텀 Layer. Lambda 금지.)
- `add_loss` 가 두 환경 모두에서 `model.fit` 손실에 반영되는지(`model.losses` 비어 있지 않음, 학습 loss 가 ssim_mae_loss 단독값보다 큼) 테스트로 확인.

### 0.5 리포트 (`site/index.html`)

- 입력: `results/<Model>/metrics.json` 전부. 표시 순서 `ConvLSTM, SimVP, PredRNN_V2`, 그 외는 이름순. 없는 모델은 "결과 없음" 행으로 표시하고 페이지는 항상 생성된다.
- 자기완결 단일 파일: CSS inline, 그림은 base64 `data:image/png`, 차트는 inline SVG. 외부 리소스 0.
- 구성: (1) 제목·생성 시각·데이터 요약(기간, 프레임 수, 분할) (2) 핵심 비교표: params, epochs_run, sec/epoch, val MAE·SSIM, full-frame MAE·SSIM, Persistence 대비 MAE 개선율, GPU (3) SVG 막대 차트: val MAE (Persistence 기준선 표시), val SSIM (4) SVG 꺾은선: 모델별 val_loss (실선) 와 loss (점선) per epoch, 하나의 축 (5) 모델별 카드: full_frame_prediction.png, history.png, env·config 표 (6) 데이터 절: samples.png, hourly_mean.png 를 첫 모델 것으로 한 번만 (7) 방법론 각주: 동일 분할·residual head·손실·Persistence 정의.
- 라이트/다크: `prefers-color-scheme` 로 색 토큰 전환, 모든 색은 CSS 변수. 한국어 UI. 반응형(표는 `overflow-x:auto` 컨테이너).
- `netlify.toml`: `[build]` `publish = "site"`, command 없음. Netlify 에서 "Import from Git" 으로 repo 를 고르면 그대로 배포된다.

---

### Task 1: 공통 파이프라인 분리 + metrics.json + 생성기 확장

**Files:**
- Create: `nc_pipeline.py`, `tests/test_nc_pipeline.py` (기존 `tests/test_nc_predict_colab.py` 를 `git mv` 후 수정), `tests/test_model_convlstm.py`, `tests/test_build_colab_notebook.py`
- Modify: `nc_predict_colab.py` (얇은 엔트리로), `tools/build_colab_notebook.py`, `.gitignore`
- Regenerate: `ConvLSTM_prediction_colab.ipynb`

**Interfaces:**
- Produces (`nc_pipeline.py`):
  - 상수: `COLAB_DRIVE_ROOT, COLAB_DATA_DIR, COLAB_OUT_DIR, COLAB_UNZIP_DIR, LOCAL_DATA_DIR = Path("resource/netcdf"), LOCAL_OUT_DIR = Path("results"), METRICS_NAME = "metrics.json", FIGURE_NAMES = {"samples": "samples.png", "hourly_mean": "hourly_mean.png", "history": "history.png", "full_frame": "full_frame_prediction.png"}, WEIGHTS_NAME, CHECKPOINT_NAME, SCHEMA_VERSION = 1`
  - `Config` dataclass (기존 필드 유지: data_dir, out_dir, data_zip, var, in_frames, target, patch, stride, filters, epochs, batch, lr, hours, seed, use_cache, mixed_precision, val_ratio, early_stop_patience, lr_patience, extra)
  - 기존 함수 전부 (`is_colab, mount_drive, ensure_netcdf_backend, setup_korean_font, setup_gpu, extract_zip, list_nc_files, parse_stamp, read_frames, find_segments, downsample, filter_hours, cache_path_for, load_frames, normalize, denormalize, save_and_show, plot_samples, plot_hourly_mean, ssim_metric, persistence_baseline, window_starts, split_starts, patch_grid, build_dataset, log_coverage, ssim_mae_loss, make_optimizer, train_model, plot_history, evaluate_patches, predict_full_frame`)
  - 신규:
    - `make_take_last_frame_layer()` (기존 `_take_last_frame_layer` 개명, public)
    - `residual_head(inp, delta)` → `layers.Add(dtype="float32")([TakeLastFrame(dtype="float32")(inp), delta])`. `delta` 는 float32 `(B, h, w, 1)`.
    - `compile_model(model, lr: float)` → `model.compile(optimizer=make_optimizer(lr), loss=ssim_mae_loss, metrics=["mae"])`, model 반환.
    - `environment_info(colab: bool, precision_policy: str) -> dict` (스키마 0.3 `env`).
    - `write_metrics(path: Path, payload: dict) -> None` (numpy 타입을 float/int 로 변환 후 `json.dump(ensure_ascii=False, indent=2)`).
    - `build_arg_parser(description: str) -> argparse.ArgumentParser` (기존 `parse_args` 의 인자 정의), `config_from_args(a) -> Config`.
    - `run(cfg: Config, build_model_fn, model_name: str) -> dict`: 기존 `run` 의 흐름 + `model_dir = cfg.out_dir / model_name` 아래에 그림/가중치/로그 저장, 마지막에 `metrics.json` 저장, 반환값은 저장한 payload.
    - `main_for_model(build_model_fn, model_name: str, description: str, argv=None) -> int`: 인자 파싱 + logging 설정 + `run`, 예외는 로그 후 1.
- Produces (`nc_predict_colab.py`): `MODEL_NAME = "ConvLSTM"`, `build_model(in_frames, filters, h, w, lr=1e-3)`, `main(argv=None) -> int`. 모듈 상단 import 는 반드시 한 문장 `from nc_pipeline import (...)` 로 묶는다(생성기가 이 문장을 제거한다).
- Produces (`tools/build_colab_notebook.py`): CLI `--model {convlstm,simvp,predrnn_v2}` `--profile {local,colab}` `--all`; `MODEL_SPECS = {"convlstm": ("nc_predict_colab.py", "ConvLSTM"), "simvp": ("simvp_predict_colab.py", "SimVP"), "predrnn_v2": ("predrnn_v2_predict_colab.py", "PredRNN_V2")}`; 함수 `build_notebook(pipeline_src: Path, model_src: Path, display: str, profile: str) -> dict`, `notebook_path(display, profile, root) -> Path`, `strip_pipeline_import(text: str) -> str`.

- [ ] **Step 1: 테스트 파일 이관**
  `git mv tests/test_nc_predict_colab.py tests/test_nc_pipeline.py`; import 를 `import nc_pipeline as m` 로 바꾸고, `ModelTest.test_weights_transfer_between_sizes` 는 `tests/test_model_convlstm.py` 로 옮긴다(`from nc_predict_colab import build_model, MODEL_NAME`; `MODEL_NAME == "ConvLSTM"` 확인 추가). 실행: `/usr/local/bin/python3 -m pytest tests -q -p no:cacheprovider` → import 실패로 FAIL 확인.

- [ ] **Step 2: `nc_pipeline.py` 작성**
  `nc_predict_colab.py` 의 내용 중 `build_model`, `main`, `if __name__` 을 뺀 전부를 옮긴다. 섹션 머리글(`# ----` / `# 제목` / `# ----`) 은 유지한다(생성기가 셀 경계로 쓴다). `run` 은 `build_model_fn` 과 `model_name` 을 받고, `train_model/plot_*/predict_full_frame` 은 `model_dir: Path` 를 인자로 받아 그 아래 저장한다. 그림 파일명은 `FIGURE_NAMES` 를 쓴다(기존 `01_samples.png` 등에서 변경). `predict_full_frame` 은 `build_model_fn` 을 받아 250×250 모델을 만든다. `run` 마지막에 스키마 0.3 대로 payload 를 만들고 `write_metrics(model_dir / METRICS_NAME, payload)`.
  `data.segments` 는 `[[s, e], ...]`, `data.period` 는 `f"{stamps[0]:%Y-%m-%d %H:%M} ~ {stamps[-1]:%H:%M} UTC"`, `train_period` 는 `f"{stamps[train_starts[0]]:%H:%M} ~ {stamps[train_starts[-1] + in_frames]:%H:%M}"` (val 동일).
  `train.seconds` 는 `model.fit` 벽시계 초, `sec_per_epoch = seconds / epochs_run`.

- [ ] **Step 3: `nc_predict_colab.py` 를 얇은 엔트리로**
  ```python
  """GK2A SW038 다음 프레임 예측 — ConvLSTM (공통 파이프라인: nc_pipeline.py)."""
  from __future__ import annotations
  import sys
  from nc_pipeline import (compile_model, make_take_last_frame_layer, residual_head)

  MODEL_NAME = "ConvLSTM"

  # --------------------------------------------------------------------------
  # 모델
  # --------------------------------------------------------------------------
  def build_model(in_frames: int, filters: int, h: int, w: int, lr: float = 1e-3):
    """ConvLSTM2D x2 -> Conv2D(Δ) -> 마지막 입력 프레임 + Δ."""
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input(shape=(in_frames, h, w, 1))
    x = layers.ConvLSTM2D(filters, (3, 3), padding="same", return_sequences=True, activation="tanh")(inp)
    x = layers.ConvLSTM2D(filters, (3, 3), padding="same", return_sequences=False, activation="tanh")(x)
    delta = layers.Conv2D(1, (3, 3), padding="same", activation=None, dtype="float32")(x)
    return compile_model(keras.Model(inp, residual_head(inp, delta)), lr)

  def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    from nc_pipeline import main_for_model
    return main_for_model(build_model, MODEL_NAME, "GK2A SW038 next-frame prediction (ConvLSTM)", argv)

  if __name__ == "__main__":
    sys.exit(main())
  ```
  (docstring 에 기존 Colab 사용법 요약을 유지하되 경로는 `nc_pipeline` 상수를 가리킨다.)

- [ ] **Step 4: 테스트 추가**
  `tests/test_nc_pipeline.py` 에 추가:
  - `test_write_metrics_converts_numpy`: `np.float32(1.5)`, `np.int64(3)`, `np.array([1,2])` 가 들어간 dict 를 저장 후 `json.load` 로 `1.5, 3, [1, 2]` 확인.
  - `test_environment_info_keys`: 키 집합이 `{"colab","platform","python","tensorflow","keras","gpu","precision_policy"}` 이고 모두 str/bool.
  - `test_residual_head_adds_last_frame`: `inp = keras.Input((2, 8, 8, 1))`, `delta = layers.Conv2D(1, 1, dtype="float32")(inp[:, -1])` 대신 상수 Δ 를 만들기 어려우므로 `keras.Model(inp, residual_head(inp, layers.Conv2D(1, 1, kernel_initializer="zeros", dtype="float32")(make_take_last_frame_layer()(dtype="float32")(inp))))` 로 Δ=0 모델을 만들어 출력이 입력 마지막 프레임과 같음을 확인.
  - `test_config_from_args_defaults`: 기존 `ArgsTest` 를 `build_arg_parser/config_from_args` 로 옮긴다.
  `tests/test_build_colab_notebook.py`:
  - `test_strip_pipeline_import_multiline`: `"from nc_pipeline import (\n  a,\n  b,\n)\nX = 1\n"` → `"X = 1\n"` 을 포함하고 `nc_pipeline` 문자열 없음.
  - `test_build_notebook_convlstm_colab`: 실제 `nc_pipeline.py` + `nc_predict_colab.py` 로 `build_notebook(..., "ConvLSTM", "colab")` 결과가 `metadata.colab.gpuType == "T4"`, 코드 셀에 `def run(`, `def build_model(`, `MODEL_NAME` 가 있고 `from nc_pipeline` 문자열이 없으며 마지막 셀에 `run(cfg, build_model, MODEL_NAME)` 가 있다. 모든 코드 셀 `compile()` 통과.
  - `test_build_notebook_local_profile`: profile local 이면 `metadata` 에 `accelerator` 키가 없고 마지막 셀에 `LOCAL_DATA_DIR` 가 있다.
  - `test_notebook_path_refuses_handwritten`: `notebook_path("ConvLSTM", "local", root)` 가 `ValueError`.

- [ ] **Step 5: 생성기 확장**
  섹션 분리는 `nc_pipeline.py` 와 모델 파일 각각에 `split_sections` 를 적용하고, 모델 파일 head 는 `strip_pipeline_import` 로 import 를 지운다. 셀 순서: intro md → pipeline head → pipeline 섹션들 (진입점 섹션은 `run`·`main_for_model` 등 정의만 남기고 `if __name__` 블록 제거) → 모델 head(`MODEL_NAME` 포함) → 모델 섹션들(`main`/`if __name__` 제거) → 설정 md → 실행 셀. 실행 셀은 profile 별로 `data_dir=COLAB_DATA_DIR / LOCAL_DATA_DIR`, `out_dir=COLAB_OUT_DIR / LOCAL_OUT_DIR`, `results = run(cfg, build_model, MODEL_NAME)`. colab profile 만 `accelerator/colab.gpuType` 메타데이터. `--all` 은 convlstm colab, simvp local+colab, predrnn_v2 local+colab 을 만들되 모델 파일이 없으면 경고 로그 후 건너뛴다.

- [ ] **Step 6: `.gitignore` 추가** (0.2 절 패턴).

- [ ] **Step 7: 검증**
  ```bash
  /usr/local/bin/python3 -m pytest tests -q -p no:cacheprovider
  <k3venv python> -m pytest tests -q -p no:cacheprovider
  /usr/local/bin/python3 tools/build_colab_notebook.py --model convlstm --profile colab
  MPLBACKEND=Agg /usr/local/bin/python3 nc_predict_colab.py --hours 23 --epochs 1 --out-dir <scratch>/smoke_k2
  MPLBACKEND=Agg <k3venv python> nc_predict_colab.py --hours 23 --epochs 1 --out-dir <scratch>/smoke_k3
  ```
  두 smoke 모두 `<scratch>/smoke_*/ConvLSTM/metrics.json` 이 스키마 0.3 의 키를 전부 갖는지 `python -c` 로 확인하고 결과를 리포트에 붙인다.

- [ ] **Step 8: 커밋** (`git add nc_pipeline.py nc_predict_colab.py tools/build_colab_notebook.py tests .gitignore ConvLSTM_prediction_colab.ipynb`).

---

### Task 2: SimVP 구현

**Files:**
- Create: `simvp_predict_colab.py`, `tests/test_model_simvp.py`
- Generate: `SimVP_prediction.ipynb`, `SimVP_prediction_colab.ipynb` (`python3 tools/build_colab_notebook.py --model simvp --profile local` 과 `--profile colab`)

**Interfaces:**
- Consumes: `nc_pipeline.residual_head, compile_model, main_for_model` (Task 1).
- Produces: `MODEL_NAME = "SimVP"`, `build_model(in_frames, filters, h, w, lr=1e-3)`, `main`.

- [ ] **Step 1: 논문·공식 구현 확인** — WebFetch 로 arXiv:2206.05099 abstract 와 OpenSTL/SimVP 의 `simvp_model.py` 를 확인해 0.4 절의 블록 구조(ConvSC, Inception, Mid_Xnet skip)가 맞는지 대조하고 차이가 있으면 리포트에 적는다 (구현은 0.4 절을 따른다).
- [ ] **Step 2: 실패하는 테스트 작성** (`tests/test_model_simvp.py`):
  - `test_model_name`: `MODEL_NAME == "SimVP"`.
  - `test_output_shape_and_param_transfer`: `build_model(4, 4, 32, 32)` 와 `build_model(4, 4, 48, 48)` 의 `count_params()` 가 같고 `set_weights` 후 `predict(zeros (1,4,48,48,1))` shape `(1,48,48,1)`, float32, finite.
  - `test_odd_size_250_roundtrip`: `build_model(4, 2, 250, 250)` 출력 shape `(None, 250, 250, 1)` (`model.output_shape`).
  - `test_residual_identity_when_delta_zero`: readout Conv2D 의 kernel/bias 를 0 으로 `set_weights` 한 뒤 출력이 입력 마지막 프레임과 같다(atol 1e-6).
  - `test_mixed_precision_output_float32`: `keras.mixed_precision.set_global_policy("mixed_float16")` 후 build → `model.output.dtype == "float32"` 확인, 테스트 끝에 정책 `float32` 복원.
  - `test_fit_one_step`: 합성 데이터 `(8, 4, 32, 32, 1)` 로 `fit(epochs=1, batch_size=4, verbose=0)` 가 finite loss.
- [ ] **Step 3: 구현** — 0.4 절 SimVP 스펙. 모델 파일 구조는 `nc_predict_colab.py` 와 같은 섹션 머리글(`# 모델`) 을 쓴다. `from nc_pipeline import (...)` 한 문장.
- [ ] **Step 4: 두 환경 pytest + smoke** (Global Constraints 의 smoke 명령, `<scratch>/smoke_simvp_k2`, `_k3`). metrics.json 생성 확인. 1 epoch 의 step 당 시간을 리포트에 적는다.
- [ ] **Step 5: 노트북 생성 + 커밋** (`git add simvp_predict_colab.py tests/test_model_simvp.py SimVP_prediction.ipynb SimVP_prediction_colab.ipynb`).

---

### Task 3: PredRNN-V2 구현

**Files:**
- Create: `predrnn_v2_predict_colab.py`, `tests/test_model_predrnn_v2.py`
- Generate: `PredRNN_V2_prediction.ipynb`, `PredRNN_V2_prediction_colab.ipynb`

**Interfaces:** Task 2 와 동일 형태, `MODEL_NAME = "PredRNN_V2"`.

- [ ] **Step 1: 공식 구현 확인** — WebFetch 로 github.com/thuml/predrnn-pytorch 의 `core/layers/SpatioTemporalLSTMCell_v2.py` 와 `core/models/predrnn_v2.py` 를 확인해 0.4 절 셀 수식·decoupling 계산·adapter 공유·zigzag 메모리 흐름을 대조한다. 차이가 있으면 0.4 절(스펙)을 따르고 리포트에 차이를 적는다.
- [ ] **Step 2: `add_loss` 사전 검증** — 두 환경에서 `keras.Input` → 커스텀 Layer(`self.add_loss(tf.reduce_mean(x) * 0.1)`) → functional model 을 만들고 `model.losses` 가 1개, `fit` 1 step 의 loss 가 반영되는지 확인하는 10줄짜리 스크립트를 scratch 에서 실행한다. 실패하면 BLOCKED 로 보고한다.
- [ ] **Step 3: 실패하는 테스트 작성** (`tests/test_model_predrnn_v2.py`): Task 2 의 여섯 테스트와 같은 항목(이름은 PredRNN_V2) + 추가:
  - `test_stlstm_cell_shapes`: `STLSTMCell(8, 3)` 에 `x (2,16,16,4), h,c,m (2,16,16,8)` 를 넣으면 5개 출력 모두 `(2,16,16,8)`.
  - `test_decouple_loss_registered`: `build_model(4, 4, 32, 32)` 를 합성 입력으로 한 번 호출한 뒤 `model.losses` 가 비어 있지 않고 값이 finite ≥ 0.
  - `test_space_depth_roundtrip`: `DepthToSpace(2)(SpaceToDepth(2)(x)) == x`.
- [ ] **Step 4: 구현** — 0.4 절 PredRNN-V2 스펙.
- [ ] **Step 5: 두 환경 pytest + smoke** (`<scratch>/smoke_predrnn_k2`, `_k3`). step 당 시간 기록.
- [ ] **Step 6: 노트북 생성 + 커밋**.

---

### Task 4: 리포트 빌더 + Netlify 설정

**Files:**
- Create: `tools/build_report.py`, `tests/test_build_report.py`, `netlify.toml`, `site/index.html` (생성물)

**Interfaces:**
- Consumes: 0.3 스키마의 `results/<Model>/metrics.json` 과 그림 파일.
- Produces: CLI `python3 tools/build_report.py [--results-dir results] [--out site/index.html]`; 함수 `load_results(results_dir: Path) -> list[dict]` (정렬 규칙 적용, 각 dict 에 `_dir: Path` 추가), `render_html(results: list[dict], generated_at: str) -> str`, `encode_image(path: Path) -> str | None` (없으면 None), `svg_bar_chart(...)`, `svg_line_chart(...)`.

- [ ] **Step 1: `dataviz` 스킬을 Skill 도구로 불러 읽고**(차트 색·형식 규칙), 그 규칙대로 색 토큰을 정한다. 외부 라이브러리는 쓰지 않는다.
- [ ] **Step 2: 실패하는 테스트 작성** (`tests/test_build_report.py`): tmp 디렉터리에 0.3 스키마 fixture 두 개(`ConvLSTM`, `PredRNN_V2`; 그림은 1×1 PNG 를 PIL 없이 바이트 상수로 기록, `PredRNN_V2` 는 `history.png` 를 일부러 빠뜨린다) 를 만들고:
  - `test_load_results_order`: 순서가 `["ConvLSTM", "PredRNN_V2"]` 이고 `SimVP` 는 없음.
  - `test_render_contains_table_and_missing_placeholder`: HTML 에 두 모델명, `SimVP` 행에 "결과 없음", `data:image/png;base64,` 가 존재하고, `<script` 와 `http://`, `https://` 가 없다(자기완결).
  - `test_render_handles_no_results`: 빈 리스트로도 HTML 생성(모든 모델이 "결과 없음").
  - `test_cli_writes_file`: `main(["--results-dir", tmp, "--out", tmp/"site/index.html"])` → 파일 생성, 0 반환.
- [ ] **Step 3: 구현** — 0.5 절. HTML 은 f-string 조립 시 `html.escape` 로 값 escape. SVG 차트는 폭 100% (viewBox), 축 라벨·범례 포함.
- [ ] **Step 4: `netlify.toml`** 작성, `python3 tools/build_report.py` 로 `site/index.html` 생성(현재 results 가 비어 있으면 "결과 없음" 페이지).
- [ ] **Step 5: 검증 + 커밋** (`git add tools/build_report.py tests/test_build_report.py netlify.toml site/index.html`).

---

### Task 5: 문서

**Files:**
- Create: `doc/model_comparison.md`, `README.md`
- Modify: `doc/nc_predict_pipeline.md` (7.5 절 파일 표에 새 파일·`results/` 레이아웃 반영), `coding_history.md` (항목 추가, 20줄 이내)

- [ ] **Step 1: `md-docs` 스킬 규칙으로 `doc/model_comparison.md`** — 1) 핵심 요약: 세 모델 비교표(핵심 아이디어, 논문·공식 구현 링크·라이선스, 우리 적응 사항, params) 2) 공통 실험 조건(데이터·분할·손실·head·평가) 3) 모델별 구조 설명(mermaid flowchart 1개씩) 4) 실행 방법(로컬 CLI/노트북, Colab 노트북) 5) 결과 수집: Drive `nc_predict_output/<Model>` → repo `results/<Model>` 복사, `python3 tools/build_report.py`, 커밋·푸시 6) Netlify 배포 절차(Import from Git → repo 선택 → `netlify.toml` 이 publish=site 지정 → Deploy; 이후 push 마다 자동 재배포) 7) 실무 선택 기준.
- [ ] **Step 2: `README.md`** — 프로젝트 한 줄 소개, 파일 구조 표, 빠른 시작 명령, `doc/model_comparison.md` 링크.
- [ ] **Step 3: 커밋**.

---

## 실행 순서와 병렬화

1. Task 1 (단독, 작업 트리) — 다른 작업이 의존하는 인터페이스.
2. Task 2, Task 3, Task 4 병렬 (각각 git worktree 브랜치) → controller 가 `feature/model-comparison` 에 merge.
3. Task 5 (문서) — 2·3·4 의 실제 파라미터 수·적응 사항을 반영.
4. Controller: 세 모델 로컬 실행(동일 epochs) → `results/` 커밋 → `tools/build_report.py` → `site/index.html` 커밋 → main merge → push.
