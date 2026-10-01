"""1~8회차 빌더가 공유하는 셀 텍스트."""

FONT_CELL = r'''
# 한글 폰트 설정 (1회차와 동일). Colab 은 나눔폰트 설치 후 런타임 재시작 필요.
#   !apt-get -qq install fonts-nanum
#   !rm -rf ~/.cache/matplotlib
from matplotlib import font_manager


def set_korean_font() -> str:
  candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]
  installed = {f.name for f in font_manager.fontManager.ttflist}
  for name in candidates:
    if name in installed:
      plt.rcParams["font.family"] = name
      plt.rcParams["axes.unicode_minus"] = False
      return name
  plt.rcParams["axes.unicode_minus"] = False
  return "(한글 폰트 없음 - 그래프의 한글이 깨질 수 있음)"


print("적용된 폰트:", set_korean_font())
'''

ENSURE_DATA_CELL = r'''
# ===== 1회차 실습 데이터가 없을 때만 재생성 (실행만 하면 됩니다) =====
def ensure_session1_data(data_dir: str = DATA_DIR) -> None:
  required = ["customers.csv", "sales_2024.csv"]
  if all(os.path.exists(f"{data_dir}/{f}") for f in required):
    print("1회차 데이터 확인 완료:", required)
    return

  os.makedirs(data_dir, exist_ok=True)
  rng = np.random.default_rng(42)
  n = 300
  regions = ["서울", "경기", "부산", "대구", "기타"]
  plans = ["5G", "LTE", "3G"]

  # 데이터 사용량: 일반 사용자(90%)는 평균 12GB·표준편차 4GB 의 종 모양(정규분포)으로,
  #               헤비 유저(10%)는 30~70GB 사이에서 고르게(균등분포) 만든다.
  #               -> 대부분은 10GB 안팎이고 소수만 큰 값을 가지는, 오른쪽 꼬리가 긴 분포가 된다.
  is_heavy = rng.random(n) < 0.10                                    # 행마다 10% 확률로 헤비 유저
  usage = np.where(is_heavy, rng.uniform(30, 70, size=n), rng.normal(12, 4, size=n))
  usage = np.clip(usage, 0.5, None)                                  # 음수 방지: 최소 0.5GB
  customers = pd.DataFrame({
    "고객ID": [f"C{i:04d}" for i in range(1, n + 1)],
    "성별": rng.choice(["M", "F"], size=n),
    "나이": rng.integers(19, 70, size=n).astype(float),
    "지역": rng.choice(regions, size=n, p=[0.35, 0.3, 0.15, 0.1, 0.1]),
    "우편번호": [f"{z:05d}" for z in rng.integers(1000, 63999, size=n)],
    "요금제": rng.choice(plans, size=n, p=[0.5, 0.4, 0.1]),
    "월요금": rng.choice([29000, 35000, 45000, 55000, 65000, 79000, 89000], size=n).astype(float),
    "데이터사용량": np.round(usage, 1),
    "가입일": pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 2000, size=n), unit="D"),
  })
  customers["가입개월수"] = ((pd.to_datetime("2024-12-31") - customers["가입일"]).dt.days // 30)
  churn_prob = 0.15 + 0.25 * (customers["가입개월수"] < 12) + 0.1 * (customers["요금제"] == "3G")
  customers["이탈여부"] = np.where(rng.random(n) < churn_prob, "Yes", "No")
  customers.loc[rng.choice(n, 12, replace=False), "나이"] = np.nan
  customers.loc[rng.choice(n, 8, replace=False), "데이터사용량"] = np.nan
  customers.to_csv(f"{data_dir}/customers.csv", index=False)

  plan_info = pd.DataFrame({"요금제": plans, "속도Mbps": [1000, 150, 10], "출시연도": [2019, 2011, 2006]})
  with pd.ExcelWriter(f"{data_dir}/customers.xlsx") as writer:
    customers.to_excel(writer, sheet_name="customers", index=False)
    plan_info.to_excel(writer, sheet_name="plans", index=False)

  dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
  sales = pd.DataFrame({
    "날짜": np.repeat(dates, 3),
    "매장": np.tile(["강남점", "홍대점", "부산점"], len(dates)),
    "판매량": rng.poisson(20, size=len(dates) * 3),
  })
  sales["매출액"] = sales["판매량"] * rng.choice([12000, 15000, 18000], size=len(sales))
  sales.to_csv(f"{data_dir}/sales_2024.csv", index=False)
  print("1회차 데이터를 재생성했습니다:", required)


ensure_session1_data()
'''

CURRICULUM_TABLE = r'''
### 전체 커리큘럼

| 회차 | 주제 | 핵심 키워드 |
|:---:|------|------------|
| 1 | AI/ML/DL 개요, 데이터 획득하기 | AI ⊃ ML ⊃ DL, 지도/비지도, `read_csv`, `read_excel`, `to_csv` |
| 2 | 데이터 구조 확인하기, 기초 데이터 다루기 | `info`, `describe`, `loc/iloc`, 필터링, 정렬, `groupby`, `merge` |
| 3 | 데이터 이해하기 (EDA) | 분포, 상관관계, `matplotlib`, `seaborn`, 가설 검증 |
| 4 | 데이터 전처리하기 | 결측치, 이상치, 구간화, 인코딩, 스케일링, `train_test_split` |
| 5 | AI 모델링 필수 개념, 지도학습 I | 과적합, 평가지표, 선형회귀, 로지스틱 회귀 |
| 6 | 지도학습 II | 의사결정나무, 앙상블, 랜덤포레스트, 그라디언트부스팅 |
| 7 | 인공신경망, 심층신경망, 딥러닝 프레임워크 | 퍼셉트론, 활성화함수, Keras `Sequential`, `EarlyStopping` |
| 8 | 비지도학습, 모델 성능 향상시키기 | K-Means, PCA, 교차검증, 하이퍼파라미터 튜닝, 모의고사 |
'''


def curriculum(session: int) -> str:
  """해당 회차 행만 볼드로 강조한 커리큘럼 표."""
  lines = []
  for line in CURRICULUM_TABLE.strip("\n").split("\n"):
    if line.startswith(f"| {session} |"):
      cells = line.split("|")
      cells[1] = f" **{session}** "
      cells[2] = f" **{cells[2].strip()}** "
      line = "|".join(cells)
    lines.append(line)
  return "\n".join(lines)


# 5회차 이후 공용: 캘리포니아 주택·타이타닉 로드 (영어 컬럼 -> 한글)
LOAD_DATA_CELL = r'''
# ===== 실습 데이터: 캘리포니아 주택 가격(회귀), 타이타닉 생존(분류) =====
# data/ 폴더에 아래 파일이 있어야 합니다. (Colab 이면 왼쪽 파일 탭에서 data/ 폴더를 만들고 업로드)
#   california_housing_train.csv, california_housing_test.csv, titanic_train.csv, titanic_test.csv
# 원본 컬럼명은 영어라서 읽은 뒤 한글로 바꿉니다. (영어 = 한글 대응표는 아래 사전 참고)

HOUSING_COLS = {
  "longitude": "경도",                 # 서쪽일수록 작은 값 (-124 ~ -114)
  "latitude": "위도",                  # 북쪽일수록 큰 값 (32 ~ 42)
  "housing_median_age": "주택연식",    # 그 구역 주택 나이의 중앙값 (년)
  "total_rooms": "총방수",             # 구역 내 전체 방 수
  "total_bedrooms": "총침실수",        # 구역 내 전체 침실 수
  "population": "인구",                # 구역 인구
  "households": "가구수",              # 구역 가구 수
  "median_income": "소득중앙값",       # 가구 소득 중앙값 (단위: 만 달러)
  "median_house_value": "주택가격",    # 구역 주택 가격의 중앙값 (달러) <- 회귀 타깃
}
TITANIC_COLS = {
  "PassengerId": "승객ID",
  "Survived": "생존",                  # 1 = 생존, 0 = 사망 <- 분류 타깃
  "Pclass": "객실등급",                # 1등석 / 2등석 / 3등석
  "Name": "이름",
  "Sex": "성별",                       # male / female
  "Age": "나이",
  "SibSp": "동반형제배우자",           # 함께 탄 형제·배우자 수
  "Parch": "동반부모자녀",             # 함께 탄 부모·자녀 수
  "Ticket": "티켓번호",
  "Fare": "운임",                      # 지불한 요금 (파운드)
  "Cabin": "객실번호",
  "Embarked": "탑승항구",              # C = Cherbourg, Q = Queenstown, S = Southampton
}

for name in ["california_housing_train.csv", "california_housing_test.csv", "titanic_train.csv", "titanic_test.csv"]:
  if not os.path.exists(f"{DATA_DIR}/{name}"):
    raise FileNotFoundError(f"{DATA_DIR}/{name} 가 없습니다. data 폴더에 실습 파일을 넣어 주세요.")

housing = pd.read_csv(f"{DATA_DIR}/california_housing_train.csv").rename(columns=HOUSING_COLS)
housing_test = pd.read_csv(f"{DATA_DIR}/california_housing_test.csv").rename(columns=HOUSING_COLS)
titanic = pd.read_csv(f"{DATA_DIR}/titanic_train.csv").rename(columns=TITANIC_COLS)
titanic_test = pd.read_csv(f"{DATA_DIR}/titanic_test.csv").rename(columns=TITANIC_COLS)

print("housing:", housing.shape, "| housing_test:", housing_test.shape)
print("titanic:", titanic.shape, "| titanic_test:", titanic_test.shape)
'''
