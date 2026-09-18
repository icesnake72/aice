#!/usr/bin/env bash
#
# doc/setup_windows.md 6~7절(네 모델 실행 -> 리포트 생성)을 한 번에 돌린다.
# 대상 환경: WSL2 + NVIDIA GPU. bash 전용 문법을 쓰므로 sh 로 실행하면 안 된다.
#
# 사용법
#   ./tools/run_all.sh                      # 네 모델 --epochs 4 (파서 기본값)
#   ./tools/run_all.sh --epochs 10          # 추가 인자는 네 모델에 그대로 전달된다
#   ./tools/run_all.sh --batch 8 --hours 6 7 8
#   RUN_REPORT=0 ./tools/run_all.sh         # build_report.py 는 건너뛴다
#
# 설계 근거
#   - set -e 를 쓰지 않는다. 한 모델이 죽어도 나머지는 돌아야 비교표가 나온다.
#     대신 모델별 exit code 를 모아 마지막에 요약하고, 하나라도 실패하면 1 로 끝낸다.
#   - 모델은 순차 실행한다. VRAM 8GB 에 네 프로세스를 동시에 올리면 OOM 이고,
#     프로세스가 끝나면 GPU 메모리는 드라이버가 회수하므로 순차면 서로 간섭이 없다.
#   - OOM 이면 --batch 8 로 한 번만 자동 재시도한다 (setup_windows.md 8절 조치).
#     batch 는 build_report.py 의 조건 일치 검사 대상(CONDITION_CONFIG_KEYS)이 아니라
#     모델마다 달라져도 리포트에 "실행 조건 불일치" 배너가 뜨지 않는다.
#   - 첫 모델이 .nc 710개를 파싱해 results/cache 에 npz 를 만들고,
#     나머지 세 모델은 그 캐시를 재사용한다 (순차 실행이라 캐시 경합이 없다).

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export MPLBACKEND=Agg   # 헤드리스 WSL2: matplotlib 이 GUI 백엔드를 찾지 않게 한다

# venv 가 있으면 activate 없이 인터프리터를 직접 쓴다 (셸 상태에 의존하지 않는다).
PY=python3
[[ -x .venv/bin/python3 ]] && PY=.venv/bin/python3

# 실행 순서는 setup_windows.md 6절과 같다 ("표시이름:스크립트" 형식).
MODELS=(
  "VideoTransformer:videotf_predict_colab.py"
  "ConvLSTM:nc_predict_colab.py"
  "SimVP:simvp_predict_colab.py"
  "PredRNN_V2:predrnn_v2_predict_colab.py"
)

ARGS=("$@")
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=results/logs
mkdir -p "$LOG_DIR"

# 데이터도 캐시도 없으면 네 번 연속 실패할 게 뻔하므로 미리 끊는다.
# --data-dir / --data-zip 를 직접 준 경우는 경로를 알 수 없으므로 검사를 건너뛴다.
if [[ "${ARGS[*]-}" != *--data-dir* && "${ARGS[*]-}" != *--data-zip* ]]; then
  if ! compgen -G "resource/netcdf/*.nc" >/dev/null && ! compgen -G "results/cache/*.npz" >/dev/null; then
    echo "오류: resource/netcdf 에 .nc 도, results/cache 에 npz 캐시도 없다." >&2
    echo "      python3 gk2a_download.py --date 2025-10-17 --channel sw038 --out resource/netcdf" >&2
    exit 1
  fi
fi

echo "인터프리터: $PY"
echo "공통 인자  : ${ARGS[*]-(없음, 파서 기본값)}"
echo "로그       : $LOG_DIR/${STAMP}_<Model>.log"
echo

names=()
codes=()
secs=()
failed=0

for entry in "${MODELS[@]}"; do
  name=${entry%%:*}
  script=${entry#*:}
  log="$LOG_DIR/${STAMP}_${name}.log"

  echo "=========================================================="
  echo "[$name] $script ${ARGS[*]-}"
  echo "=========================================================="
  start=$SECONDS
  "$PY" "$script" ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}

  # OOM 이면 batch 를 절반으로 줄여 1회 재시도한다 (argparse 는 뒤에 온 값이 이긴다).
  if [[ $status -ne 0 ]] && grep -qE "ResourceExhaustedError|OOM when allocating" "$log"; then
    echo "[$name] OOM 감지 -> --batch 8 로 1회 재시도"
    "$PY" "$script" ${ARGS[@]+"${ARGS[@]}"} --batch 8 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
  fi

  elapsed=$((SECONDS - start))
  names+=("$name")
  codes+=("$status")
  secs+=("$elapsed")
  [[ $status -ne 0 ]] && failed=1
  echo "[$name] exit=$status  (${elapsed}s)"
  echo
done

if [[ ${RUN_REPORT:-1} -eq 1 && $failed -eq 0 ]]; then
  echo "=========================================================="
  echo "리포트 생성: tools/build_report.py"
  echo "=========================================================="
  "$PY" tools/build_report.py 2>&1 | tee "$LOG_DIR/${STAMP}_report.log"
elif [[ ${RUN_REPORT:-1} -eq 1 ]]; then
  # 실패한 모델은 metrics.json 이 없어 리포트에 "결과 없음"으로 박힌다.
  # 그 상태로 site/index.html 을 덮어쓰면 배포본이 조용히 퇴보하므로 막는다.
  echo "실패한 모델이 있어 리포트 생성을 건너뛴다. 수동 실행: $PY tools/build_report.py"
fi

echo "=========================================================="
printf '%-18s %6s %8s\n' "MODEL" "EXIT" "SEC"
for i in "${!names[@]}"; do
  printf '%-18s %6s %8s\n' "${names[$i]}" "${codes[$i]}" "${secs[$i]}"
done
echo "=========================================================="

exit $failed
