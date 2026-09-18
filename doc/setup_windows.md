# Windows(WSL2)에서 GK2A 예측 파이프라인 실행

> 대상 하드웨어: Windows 11, i7-12700H, NVIDIA RTX 3070 Laptop GPU (VRAM 8GB, 시스템 RAM 공유), 시스템 RAM 64GB.
> 이 문서의 TensorFlow/CUDA 관련 수치는 tensorflow.org 설치 가이드와 PyPI 의 tensorflow 패키지 메타데이터에서
> 확인한 값만 쓴다 (근거는 각 절에 표기). 실측이 없는 항목은 "실측 없음"이라고 그대로 적는다.

네 모델(ConvLSTM · SimVP · PredRNN-V2 · VideoTransformer)은 지금 macOS(TF 2.15 + Metal)와
Google Colab(T4, TF 2.20)에서만 실행이 검증돼 있다. 이 문서는 같은 코드를 Windows 노트북에서
GPU 가속으로 돌리기 위한 절차를 정리한다.

---

## 1. 핵심 요약

| 경로 | 권장 여부 | 이유 |
| --- | --- | --- |
| **WSL2(Ubuntu 22.04) + `tensorflow[and-cuda]==2.20.*`** | 권장 | `tensorflow.org/install/pip`: WSL2 GPU 는 Windows 10 19044(21H2)+ 에서 지원되고 Windows 11 은 이를 만족한다. `[and-cuda]` 익스트라가 PyPI 상 CUDA/cuDNN 을 pip wheel(`nvidia-cudnn-cu12` 등)로 함께 설치하므로 CUDA Toolkit 을 따로 설치할 필요가 없다 — Windows NVIDIA 드라이버만 있으면 된다 |
| 네이티브 Windows TensorFlow | 불가 | `tensorflow.org/install/pip`: "TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows." 이 프로젝트는 `simvp_predict_colab.py` 가 `layers.GroupNormalization`(TF>=2.11 필요)을 쓰므로 TF 2.10 으로는 애초에 코드가 안 돈다 |
| TensorFlow-DirectML-Plugin | 비권장 | 같은 공식 문서가 네이티브 GPU 의 대안으로만 언급하는 우회 경로다. WSL2 라는 정식 GPU 경로가 이 하드웨어에서 그대로 되므로, 별도로 유지되는 플러그인의 연산 커버리지·버전 지연 리스크를 감수할 이유가 없다 |

> 중요: `tensorflow.org/install/pip` 는 "2.21 부터 Python 3.9 지원 종료, 3.10-3.13 사용 권장"이라고 명시한다.
> `requirements-wsl-cuda.txt` 가 `tensorflow==2.20.*` 를 쓰는 이유이기도 하다 (Colab 실측과도 버전이 맞는다).

---

## 2. 사전 준비

| 항목 | 내용 |
| --- | --- |
| OS | Windows 11 (WSL2 GPU 지원 최소 조건인 Windows 10 19044 를 넉넉히 상회) |
| NVIDIA 드라이버 | Windows 쪽에 최신 GeForce 드라이버 설치. `tensorflow.org/install/pip` 가 명시한 WSL GPU 최소 버전은 `>= 528.33` — 그보다 낮으면 GPU 가 WSL2 커널에 노출되지 않는다 |
| WSL2 배포판 | PowerShell(관리자)에서 `wsl --install -d Ubuntu-22.04` |

`.wslconfig` 로 WSL2 에 배정할 메모리를 제한한다 (시스템 RAM 64GB 중 일부를 Windows 호스트용으로 남긴다).
`%UserProfile%\.wslconfig` 에 다음 내용을 쓰고 `wsl --shutdown` 후 다시 연다.

```ini
[wsl2]
memory=48GB
swap=16GB
```

> 주의: 프로젝트는 **WSL 파일시스템**(`~/repo` 처럼 `/home/<user>/...`)에 clone 한다. `/mnt/c/...` 는
> Windows 드라이브를 WSL 이 경유해서 접근하는 경로라 파일 I/O 가 크게 느리다 — 710개 `.nc` 파일을
> 반복해서 여는 이 파이프라인에서는 체감 차이가 크다 (8절 참고).

```bash
git clone <repo-url> ~/nc_predict
cd ~/nc_predict
```

---

## 3. 환경 구성 명령

```bash
sudo apt update
sudo apt install -y python3-venv fonts-nanum git

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-wsl-cuda.txt
```

`fonts-nanum` 을 여기서 apt 로 설치하는 이유: `nc_pipeline.setup_korean_font()` 의 자동 apt 설치 분기는
`is_colab()`(= `google.colab` 임포트 가능 여부)가 참일 때만 동작한다. WSL2 는 Colab 이 아니므로 그 분기가
실행되지 않고, 폰트 파일이 `/usr/share/fonts/truetype/nanum/NanumGothic.ttf` 에 이미 있을 때만
(apt 설치 여부와 무관하게) `fm.fontManager.addfont()` 로 등록한다. 즉 WSL2 에서는 이 apt 설치를 직접 해 둬야
그래프 제목의 한글이 깨지지 않는다.

---

## 4. 검증

```bash
python tools/env_check.py
nvidia-smi
python -m pytest tests -q -p no:cacheprovider
```

`tools/env_check.py` 는 이 저장소 전용으로 새로 만든 점검 스크립트다. 실제 표 형식은 이 문서를 쓰면서
macOS 두 인터프리터(TF 2.15/Metal, TF 2.21/CPU)에서 검증했고, WSL2 + RTX 3070 에서는 GPU 줄이 아래
형식으로 나올 것으로 예상한다(초당 ms 실측값은 첫 실행 로그로 직접 확인한다. compute capability 8.6 은
RTX 3070(Ampere GA104)의 고정된 하드웨어 사양이라 실행 전에도 알 수 있다):

```text
=== GK2A 파이프라인 환경 점검 ===

Python                 : 3.10.x
Platform               : Linux-...-x86_64-with-glibc2.35
TensorFlow             : 2.20.0
Keras                  : 3.x.x
GPU                    : NVIDIA GeForce RTX 3070 Laptop GPU (compute 8.6)
행렬곱 벤치(4096x4096 fp16) : <실측 ms>/회
mixed_float16 정책 적용    : 가능
...
```

`nvidia-smi` 는 Windows 드라이버가 WSL2 에 GPU 를 제대로 노출했는지 보는 별도 확인이다(둘 다 GPU 를
보여줘야 정상). `pytest` 는 데이터 없이 도는 테스트만 모아 둔 것이라 데이터 이동(5절) 전에 먼저 돌려도 된다.

---

## 5. 데이터 이동

`resource/netcdf/` 는 하루(2025-10-17) 710개 파일, 약 287MB (`doc/nc_predict_pipeline.md` 2.1절:
"약 0.40MB/장, 하루 710장 = 287MB")다. 둘 중 하나로 준비한다.

```bash
# 방법 A: Mac 에서 이미 받아 둔 데이터를 복사 (scp, rsync 등으로 WSL 쪽 ~/nc_predict/resource/netcdf 에)
rsync -av mac-host:~/nc_predict/resource/netcdf/ ~/nc_predict/resource/netcdf/

# 방법 B: WSL2 안에서 AWS Open Data 에서 다시 받기 (익명 접근, 크레덴셜 불필요)
python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out resource/netcdf
```

`results/cache/frames_sw038_t250.npz`(다운샘플 프레임 캐시, 약 178MB — `doc/nc_predict_pipeline.md` 7.5절)는
있으면 두 번째 실행부터 `.nc` 재파싱을 건너뛰므로 함께 복사하면 좋지만, 없어도 첫 실행 때 새로 만들어진다.
캐시는 `.gitignore` 대상이라 git 으로는 옮겨지지 않는다.

---

## 6. 실행

모델 네 개는 같은 `nc_pipeline.build_arg_parser()` 옵션을 공유한다. VideoTransformer 부터 실행한다.

```bash
export MPLBACKEND=Agg   # 헤드리스 WSL2 에 디스플레이가 없을 때 matplotlib GUI 백엔드 시도를 막는다

python3 videotf_predict_colab.py    --epochs 4
python3 nc_predict_colab.py         --epochs 4
python3 simvp_predict_colab.py      --epochs 4
python3 predrnn_v2_predict_colab.py --epochs 4
```

> 주의: `MPLBACKEND=Agg` 는 bash 문법이다. WSL2 밖의 Windows PowerShell 에서 같은 값을 설정하려면
> `$env:MPLBACKEND="Agg"` 를 쓴다 (두 셸의 환경변수 문법이 다르다). `nc_pipeline.save_and_show()` 는
> 백엔드가 `agg` 가 아니면 `plt.show()` 를 시도하므로, 디스플레이 없는 WSL2 세션에서 이 값을 비워 두면
> 불필요한 경고가 난다.

### 6.1 한 번에 돌리기 — `tools/run_all.sh`

위 네 줄을 순서대로 실행하고 리포트 생성(7절)까지 이어 주는 스크립트다. `MPLBACKEND=Agg` 설정과
`.venv/bin/python3` 선택도 스크립트 안에서 한다(activate 여부에 의존하지 않는다).

```bash
./tools/run_all.sh                  # 네 모델, 파서 기본 옵션
./tools/run_all.sh --epochs 10      # 추가 인자는 네 모델에 그대로 전달된다
RUN_REPORT=0 ./tools/run_all.sh     # build_report.py 는 건너뛴다

# Windows 쪽에서 파일을 만들어 실행 비트가 없다면 (core.filemode=false)
chmod +x tools/run_all.sh           # WSL 에서 한 번만
git add --chmod=+x tools/run_all.sh # 커밋할 때 실행 비트를 같이 기록한다
```

동작 규칙은 다음과 같다.

| 규칙 | 이유 |
| --- | --- |
| 모델을 **순차** 실행한다 | VRAM 8GB 에 네 프로세스를 동시에 올리면 OOM 이다. 프로세스가 끝나면 GPU 메모리는 드라이버가 회수하므로 순차면 서로 간섭이 없다 |
| 한 모델이 실패해도 나머지를 계속 돌린다 (`set -e` 미사용) | 세 모델만이라도 비교표가 나와야 한다. 모델별 exit code 는 마지막 요약 표에 찍고, 하나라도 실패하면 스크립트 종료 코드는 1 이다 |
| 로그에 `ResourceExhaustedError` 가 보이면 `--batch 8` 로 1회 자동 재시도 | 8절의 OOM 조치와 같다. `batch` 는 리포트의 조건 일치 검사 대상이 아니라 모델마다 달라져도 "실행 조건 불일치" 배너가 뜨지 않는다 |
| 실패한 모델이 있으면 리포트 생성을 건너뛴다 | `metrics.json` 이 없는 모델은 리포트에 "결과 없음"으로 박힌다. 그대로 `site/index.html` 을 덮어쓰면 배포본이 조용히 퇴보한다 |
| 데이터도 캐시도 없으면 시작 전에 멈춘다 | 네 번 연속 실패를 기다릴 이유가 없다 |

실행 로그는 `results/logs/<타임스탬프>_<Model>.log` 에 남는다(`.gitignore` 대상). 첫 모델이 `.nc` 710개를
파싱해 `results/cache` 에 npz 를 만들고 나머지 세 모델이 그 캐시를 재사용하므로, 데이터 적재 비용은 한 번만 든다.

VRAM 8GB 에서 기본 `--batch 16` 이 OOM 나면 절반으로 줄인다: `--batch 8`. `nc_pipeline.setup_gpu()` 가
GPU 마다 `set_memory_growth(gpu, True)` 를 이미 걸어 두므로(한꺼번에 VRAM 을 선점하지 않는다) 이 옵션을
따로 켤 필요는 없다.

예상 학습 시간은 실측이 있는 범위 안에서만 말한다.

| 모델 | 실측 | 근거 |
| --- | --- | --- |
| SimVP | T4(Colab) 29.5 s/epoch | `results/SimVP/metrics.json` (`train.sec_per_epoch`) |
| PredRNN-V2 | T4(Colab) 24.6 s/epoch | `results/PredRNN_V2/metrics.json` |
| ConvLSTM | M1 Pro/Metal 177.6 s/epoch (T4 실측 없음) | `results/ConvLSTM/metrics.json` |
| VideoTransformer | 실측 없음 | 아직 로컬/Colab 실행 결과가 없다(`videotf_predict_colab.py` 헤더: 학습 정체 이슈 미해결) |

RTX 3070 은 compute capability 8.6(Ampere)이라 `mixed_float16` 이 유효한 하드웨어다 — TF 자체가
"compute capability 7.0 미만이면 mixed_float16 이 느릴 수 있다"고 경고하는데(Mac 의 METAL 실행 시
이 경고가 실제로 뜬다) 8.6 은 그 기준을 넘는다. 그래서 SimVP·PredRNN-V2 는 T4 실측(30 s·25 s급)과
비슷하거나 더 빠른 급으로 예상하지만, 정확한 수치는 실행 로그(`sec_per_epoch`, `metrics.json`)로 확인한다.

---

## 7. 결과 반영과 배포

```bash
python3 tools/build_report.py
git add results/VideoTransformer results/ConvLSTM results/SimVP results/PredRNN_V2 site/index.html
git commit -m "WSL2(RTX 3070) 실행 결과 반영"
git push
```

`.gitignore` 가 `results/cache/`, `results/**/*.npz`, `*.h5`, `*.npy`, `*.part` 를 빼므로
`git add results/<Model>` 는 실제로 `metrics.json`·그림 4장·`train_log.csv` 만 추적한다(가중치·캐시는 제외).
Netlify 는 이 저장소에 `netlify.toml`(`publish = "site"`)로 이미 연결돼 있다는 전제이므로, push 하면
`site/index.html` 이 자동 재배포된다(`doc/model_comparison.md` 6절). Windows 에서 git 을 함께 쓴다면
이번에 추가한 `.gitattributes`(`* text=auto eol=lf`)가 CRLF 로 커밋되는 것을 막아 준다.

---

## 8. 문제 해결

| 증상 | 확인 | 조치 |
| --- | --- | --- |
| GPU 미인식 (`tf.config.list_physical_devices('GPU')` 가 빈 리스트) | Windows 에서 `nvidia-smi`, WSL2 안에서도 `nvidia-smi` (드라이버가 WSL 커널까지 노출됐는지) | 드라이버를 `>=528.33` 으로 갱신, `wsl --update`, `.wslconfig` 저장 후 `wsl --shutdown` 으로 재시작 |
| cuDNN/ptxas 관련 경고 로그 | `tf.config.list_physical_devices('GPU')` 가 그래도 GPU 를 나열하는지 | 나열되면 대개 초기화 단계의 정보성 경고다. `python tools/env_check.py` 의 행렬곱 벤치가 정상적으로 ms 값을 내면 실제 가속은 되고 있는 것이다 |
| OOM (`ResourceExhaustedError`) | 기본 `--batch` 16, `--target` 250(96x96 패치) | `--batch 8` 로 줄인다. `setup_gpu()` 의 memory growth 는 이미 켜져 있어 추가 설정은 불필요하다 |
| 그래프 제목 한글이 □ 로 깨짐 | `fc-list \| grep -i nanum` (설치 여부) | `sudo apt install fonts-nanum` 후 재실행. WSL2 는 `is_colab()` 이 False 라 자동 설치가 안 된다(3절) |
| `.nc` 읽기·캐시 쓰기가 느림 | 프로젝트가 `/mnt/c/...` 에 있는지 | WSL 파일시스템(`~/...`)으로 clone (2절). `/mnt/c` 경유 I/O 는 WSL2 의 알려진 병목이다 |

---

## 9. 실무 선택 기준

| 상황 | 선택 | 이유 |
| --- | --- | --- |
| 이 Windows 노트북에서 여러 epoch·여러 모델을 반복해서 돌린다 | WSL2(RTX 3070) | GPU 8GB 로 네 모델 모두 `--batch 8~16` 에서 돌아가고, Colab 처럼 세션 제한·GPU 대기열이 없다 |
| 설정 없이 한 번만 빠르게 결과를 보고 싶다 | Colab(T4) | 브라우저에서 바로 실행되지만 세션 끊김과 GPU 배정 대기가 있다 |
| 이미 Mac 에 TF 2.15 + Metal 환경이 있고 노트북 하나만 확인하면 된다 | Mac | 추가 설치가 없지만, `mixed_float16` 가속 효과는 compute capability 가 없는 METAL 특성상 T4/RTX 대비 제한적이다(`doc/model_comparison.md` 7.2절 측정 조건) |
| 네 모델을 한 번에 비교해 `site/index.html` 을 갱신해야 한다 | WSL2 또는 Mac | 로컬 파일시스템에 바로 `results/` 를 쓸 수 있다. Colab 은 Drive → `results/` 복사 단계가 하나 더 필요하다(`doc/model_comparison.md` 4.2절) |
