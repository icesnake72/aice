# Coding History

## 2026-09-03 문서 갱신 (Colab 버전 반영)
- `doc/nc_predict_pipeline.md`: 7.5 절 "Google Colab (T4) 실행" 추가 (파일 구성, Drive 경로, 실행 순서, 로컬 노트북과 차이), 참고 링크 추가
- `doc/nc_predict_pipeline.md`: 7.2 `optimizers.legacy` 행을 실측대로 수정(ImportError), 7.3 `EPOCHS` 기본값 20 -> 4
- `nc_model_architecture.md`: `Lambda` -> `TakeLastFrame`, 추론 크기 300 -> 250, 공간 크기 None 설명을 Keras 3 방식으로 수정
- `doc/convlstm_principles.md` 6.3: Keras 3 의 공간 크기 None 불허와 가중치 이전 방식 주의문 추가
- `nc_model_arch.png` 그림은 `Lambda` 표기 그대로 (재생성 필요 시 `keras.utils.plot_model`)
- `.gitignore` 에 `.obsidian/`, `.omc/` 추가 (편집기·에이전트 상태 파일 커밋 방지)

## 2026-09-03 노트북 이름 변경
- `nc_predict.ipynb` -> `ConvLSTM_prediction.ipynb` (git mv, 이력 유지)
- `nc_predict_colab.ipynb` -> `ConvLSTM_prediction_colab.ipynb`
- 생성기 `tools/build_colab_notebook.py` 기본 출력 파일명·Colab 메타데이터 이름 변경 후 재생성
- `nc_predict_colab.py` docstring, `nc_model_architecture.md`, `doc/*.md` 의 노트북 참조·상대 링크 갱신
- `.py` 파일명(`nc_predict_colab.py`)과 `doc/nc_predict_pipeline.md` 파일명은 요청 범위 밖이라 유지

## 2026-09-03 nc_predict_colab.ipynb 추가 (Colab 업로드용)
- Colab '파일 > 노트 업로드' 는 .ipynb 만 받으므로 `.py` 를 셀 단위로 나눈 노트북을 자동 생성
- 생성기 `tools/build_colab_notebook.py`: `# ----` 섹션 머리글 기준으로 코드 셀 분리, `run()` 만 남기고 `parse_args/main` 제외
- 마지막 셀에 `Config(...)` 설정과 `run(cfg)` 호출을 두어 값만 바꿔 실행하도록 구성
- 노트북 메타데이터에 `accelerator: GPU`, `gpuType: T4` 지정 (열면 T4 런타임 자동 선택)
- 셀 단위 compile 검사 내장. `.py` 수정 후 `python3 tools/build_colab_notebook.py` 로 재생성
- 검증: 코드 셀 8개를 순차 exec 후 smoke `run()` 실행, Keras 2 / Keras 3 모두 정상
- Drive 기본 경로 변경: 데이터 `MyDrive/netcdf`, 결과 `MyDrive/nc_predict_output` (.py 상수·docstring·노트북 재생성)

## 2026-09-03 nc_predict_colab.py 신규 작성
- `nc_predict.ipynb`(ConvLSTM 다음 프레임 예측)를 Google Colab T4 실행용 스크립트로 변환
- 데이터는 Google Drive(`MyDrive/gk2a/netcdf`, 또는 `--data-zip`)에서 읽고, 결과는 `MyDrive/gk2a/output` 에 저장
- Colab 자동 준비: Drive 마운트, netCDF4 설치 확인, NanumGothic 설치·등록, GPU memory growth, mixed_float16
- 로컬(TF 2.15/Keras 2)과 Colab(Keras 3) 양쪽 호환: 고정 입력 크기 모델 + 추론 시 250x250 재구성/가중치 이전
- Drive 의 느린 파일 I/O 대비: 다운샘플 프레임을 `output/cache/frames_sw038_t250.npz` 로 캐시 (`--no-cache` 로 재생성)
- 세션 끊김 대비: 매 epoch best 가중치 `checkpoint.weights.h5` + `train_log.csv` 를 Drive 에 저장
- CLI: `--data-dir --data-zip --out-dir --epochs --batch --filters --target --hours --no-cache --no-mixed-precision`
- 노트북 로직 유지: 세그먼트 분리, TARGET 약수 검사, stride 77 3x3 격자, 누수 방지 gap, Persistence 베이스라인
- 개선: SSIM 을 chunk 단위 batch 계산, `xr.open_dataset` 컨텍스트 매니저로 핸들 누수 방지, headless 시 `plt.show()` 생략
- 테스트: `tests/test_nc_predict_colab.py` 19건 (합성 데이터로 순수 함수·모델 가중치 이전 검증) 통과
- 검증: Keras 2(TF 2.15, M1 GPU) 와 Keras 3(TF 2.21, CPU venv) 양쪽에서 smoke 실행(`--hours 23 --epochs 1`) 정상 완료
- 버그 수정: Keras 3 는 `keras.optimizers.legacy.Adam` 생성 시 ImportError 를 던지므로 getattr 대신 try/except 로 처리
