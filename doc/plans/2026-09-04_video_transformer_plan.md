# Video Transformer (factorized space-time ViT) 4번째 모델 추가 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 비교 파이프라인(ConvLSTM · SimVP · PredRNN-V2)에 네 번째 모델 **VideoTransformer**(공간·시간을 분리한 factorized space-time attention, PredFormer 계열)를 같은 인터페이스로 추가해 로컬(macOS)·Colab(T4) 양쪽에서 돌리고 리포트에 포함한다.

**Architecture:** 모델 파일 `videotf_predict_colab.py` 하나가 `MODEL_NAME`, `build_model` 을 정의하고 `nc_pipeline` 의 `residual_head`, `delta_readout`, `compile_model`, `main_for_model` 을 그대로 쓴다. 생성기 `MODEL_SPECS`/`ALL_TARGETS` 와 리포트 `MODEL_ORDER` 에 등록만 하면 노트북 생성·결과 수집·리포트가 기존 흐름을 탄다.

**Tech Stack:** TensorFlow/Keras (Keras 2 = TF 2.15 로컬, Keras 3 = TF 2.21 Colab), `layers.MultiHeadAttention`, `layers.LayerNormalization`.

**Spec:** 이 문서 "0. 설계" 절. 기존 계약은 `doc/plans/2026-09-03_model_comparison_plan.md` 0.1~0.5 (파일 구조, results 레이아웃, metrics.json 스키마, 공통 head/손실, 리포트).

## Global Constraints

- 2026-09-03 계획의 Global Constraints 전부 (2-space, type hint, docstring, 두 환경 테스트·smoke 통과, `build_model(in_frames, filters, h, w, lr=1e-3)` 인터페이스, 출력 = 마지막 프레임 + Δ, `delta_readout` zero-init, `ssim_mae_loss`, 96↔250 가중치 이전, `Lambda` 금지, 커밋 trailer 2줄, `git add` 명시 파일).
- Keras 2 python: `/usr/local/bin/python3`. Keras 3 python: `/private/tmp/claude-501/-Users-eunbumkim-Desktop-02---------aice-test/91f49e57-8096-45a5-bcac-aa8a54edfa48/scratchpad/k3venv/bin/python`.
- 데이터: `resource/netcdf`. smoke: `MPLBACKEND=Agg <python> videotf_predict_colab.py --hours 23 --epochs 1 --out-dir <scratch>` exit 0. 테스트·smoke 는 `results/` 에 쓰지 않는다.
- `ConvLSTM_prediction.ipynb` 수정 금지. `results/` 의 기존 세 모델 결과 수정 금지.

---

## 0. 설계 (Spec)

### 0.1 이름과 등록

| 항목 | 값 |
| --- | --- |
| 모델 파일 | `videotf_predict_colab.py` |
| `MODEL_NAME` | `"VideoTransformer"` |
| 생성기 `MODEL_SPECS` 키 | `"videotf": ("videotf_predict_colab.py", "VideoTransformer")` |
| 생성기 `ALL_TARGETS` | `("videotf", "local")`, `("videotf", "colab")` 추가 → 총 7개 |
| 생성 노트북 | `VideoTransformer_prediction.ipynb`, `VideoTransformer_prediction_colab.ipynb` |
| 리포트 `MODEL_ORDER` | `("ConvLSTM", "SimVP", "PredRNN_V2", "VideoTransformer")` |
| 결과 디렉터리 | `results/VideoTransformer/` (로컬), `MyDrive/nc_predict_output/VideoTransformer/` (Colab) |
| 테스트 | `tests/test_model_videotf.py` |

### 0.2 모델 구조 (PredFormer 계열 factorized space-time attention)

근거 논문: PredFormer — "Transformers Are Effective Spatial-Temporal Predictive Learners" (Tang et al., 2024, arXiv:2410.04733; 공식 구현 github.com/yyyujintang/PredFormer). 구현 전에 WebFetch 로 abstract 와 저장소 README 를 확인해 블록 구성(Gated Transformer Block, factorized TS/ST 배열)을 대조하고 차이를 리포트에 적는다. 아래 스펙이 binding 이다(우리 적응: 단일 스텝 예측, 소형 dim, 고정 위치 인코딩).

모듈 상수: `PATCH = 8`, `DIM = 128`, `HEADS = 4`, `DEPTH = 4`, `MLP_RATIO = 4`, `DROPOUT = 0.0`. `filters` 인자는 `DIM = 8 * filters` 로 연결한다(기본 filters=16 → 128). `DEPTH`, `HEADS`, `PATCH` 는 상수.

1. **Pad**: 입력 `(B, T, h, w, 1)` 을 프레임별로 `h_p = ceil(h/PATCH)*PATCH`, `w_p` 로 zero-pad (96→96, 250→256). 크기가 이미 배수면 pad 없음.
2. **Patch embedding**: `TimeDistributed(Conv2D(DIM, PATCH, strides=PATCH, padding="valid"))` → `(B, T, Hp, Wp, DIM)` → 토큰 `(B, T, N, DIM)`, `N = Hp*Wp` (96: 144, 256: 1024).
3. **위치 인코딩 (고정, 학습 파라미터 없음)**: 2D sinusoidal 공간 인코딩 `(1, 1, N, DIM)` (DIM 의 절반씩 y, x 축) + sinusoidal 시간 인코딩 `(1, T, 1, DIM)`. 커스텀 Layer 가 `build()` 에서 static shape 로 상수를 만들어 더한다. 가중치가 없으므로 96 학습 → 256 추론 가중치 이전이 성립한다.
4. **Block × DEPTH** (pre-LN, residual):
   - 공간 attention: `(B, T, N, D)` → `(B*T, N, D)` 로 접어 `MultiHeadAttention(HEADS, key_dim=DIM//HEADS)` self-attention → 되돌림.
   - 시간 attention: `(B, T, N, D)` → `(B*N, T, D)` 로 접어 self-attention → 되돌림.
   - MLP: `Dense(DIM*MLP_RATIO, gelu) → Dense(DIM)`.
   - 각 서브레이어 앞에 `LayerNormalization`, 뒤에 residual add. 접기/펼치기는 `tf.reshape` + `tf.shape(x)[0]` 를 쓰는 커스텀 Layer(`compute_output_shape` 포함)로 만든다. `Lambda` 금지.
5. **Readout**: 마지막 시점 토큰 `(B, N, DIM)` → `LayerNormalization` → `Dense(PATCH*PATCH)` → `(B, Hp, Wp, PATCH*PATCH)` → `DepthToSpace(PATCH)` (tf.nn.depth_to_space 래퍼) → `(B, h_p, w_p, 1)` → `Cropping2D` 로 `(B, h, w, 1)` → `delta_readout(kernel_size=1)` (float32, zero-init) → `residual_head`.
6. `compile_model(model, lr)`.

파라미터 규모 추정: 블록당 약 4·DIM²(공간) + 4·DIM²(시간) + 8·DIM²(MLP) ≈ 262k → DEPTH 4 ≈ 1.05M + embedding. 리포트에 실측 `count_params()` 를 적는다.

### 0.3 테스트 (`tests/test_model_videotf.py`)

기존 `tests/test_model_simvp.py` 의 7개 항목과 같은 이름·형태로:
- `test_model_name`, `test_output_shape_and_param_transfer` (32→48), `test_odd_size_250_roundtrip` (`build_model(4, 2, 250, 250).output_shape == (None, 250, 250, 1)`), `test_residual_identity_when_delta_zero`, `test_initial_output_is_persistence`, `test_mixed_precision_output_float32` (은닉 attention/Dense 의 `compute_dtype == "float16"`, readout `delta` 의 `compute_dtype == "float32"`, `model.output.dtype == "float32"`, `finally` 로 정책 복구), `test_fit_one_step`.
- 추가: `test_positional_encoding_has_no_weights` (위치 인코딩 레이어 `count_params() == 0`), `test_pad_multiple_of_patch` (h=250 일 때 내부 토큰 수가 `(256/8)**2 = 1024`, h=96 일 때 144 — 레이어 출력 shape 로 확인).

### 0.4 문서

- `doc/model_comparison.md`: 1절 비교표에 VideoTransformer 행(핵심 아이디어, arXiv 링크, 공식 구현 링크·라이선스(WebFetch 로 확인한 값만), 적응 사항, params), 3절에 mermaid flowchart 1개 + 적응 표, 4절 실행 명령에 `videotf_predict_colab.py` 와 노트북 2개, 7.2 표에 행 추가(값은 controller 가 실행 후 채움: `(실행 후 갱신)`), 8절 선택 기준에 한 줄.
- `README.md` 파일 구조 표에 추가. `coding_history.md` 항목 ≤ 8줄.
- `doc/nc_predict_pipeline.md` 7.5 파일 표에 한 줄 추가.

---

### Task 1: VideoTransformer 모델 + 등록 + 테스트 + 노트북

**Files:**
- Create: `videotf_predict_colab.py`, `tests/test_model_videotf.py`
- Modify: `tools/build_colab_notebook.py` (`MODEL_SPECS`, `ALL_TARGETS`), `tests/test_build_colab_notebook.py` (`--all` 기대 목록 5→7, fake root 에 videotf 사본 추가), `tools/build_report.py` (`MODEL_ORDER`), `tests/test_build_report.py` (순서·"결과 없음" 행 기대값에 VideoTransformer 반영)
- Generate: `VideoTransformer_prediction.ipynb`, `VideoTransformer_prediction_colab.ipynb`

- [ ] Step 1: 논문·공식 구현 확인 (WebFetch). 실패해도 0.2 스펙대로 진행하고 리포트에 적는다.
- [ ] Step 2: 실패하는 테스트 작성 → 실행해 실패 확인.
- [ ] Step 3: 모델 구현 (0.2). 파일 구조는 `simvp_predict_colab.py` 를 그대로 따른다(모듈 docstring, `from nc_pipeline import (...)` 한 문장, `MODEL_NAME`, `# 모델` 섹션 머리글, `build_model`, `main`, `if __name__`).
- [ ] Step 4: 생성기·리포트 등록 + 기존 테스트 기대값 갱신.
- [ ] Step 5: 두 환경 전체 pytest + 두 환경 smoke (`<scratch>/smoke_videotf_k2`, `_k3`). Keras 2 의 ms/step 기록.
- [ ] Step 6: `python3 tools/build_colab_notebook.py --all` 로 노트북 생성(총 7개 중 새 2개 추가, 기존 5개는 변경 없음 확인). 노트북 마지막 셀이 `run(cfg, build_model, MODEL_NAME)` 인지 확인.
- [ ] Step 7: 커밋 (명시 파일).

### Task 2: 문서 (실행 결과 이후)

0.4 절대로 갱신. 수치는 controller 가 전달한 `results/VideoTransformer/metrics.json` 값만 쓴다.

---

## 실행 순서

1. Task 1 (implementer → review → fix loop).
2. Controller: 로컬 4 epoch 실행 → `results/VideoTransformer/` → `tools/build_report.py` → `site/index.html`.
3. Task 2 문서 → review.
4. 최종 브랜치 리뷰 → main merge → push.
