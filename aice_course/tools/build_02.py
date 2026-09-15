"""2회차 노트북 빌더: 데이터 구조 확인하기 + 기초 데이터 다루기."""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "02_데이터구조확인_기초데이터다루기.ipynb"
cells = []


def md(src: str) -> None:
  cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
  cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- 표지
md(r"""
# AICE Associate 대비 실습 과정 — 2회차
## 데이터 구조 확인하기, 기초 데이터 다루기

> **과정 구성**: 총 8회 × 3시간, 실습 위주  
> **선수 학습**: 1회차 (데이터 획득). 1회차에서 만든 `data/customers.csv`, `data/sales_2024.csv`, `data/customers.xlsx` 를 사용합니다.

### 전체 커리큘럼

| 회차 | 주제 | 핵심 키워드 |
|:---:|------|------------|
| 1 | AI/ML/DL 개요, 데이터 획득하기 | AI ⊃ ML ⊃ DL, 지도/비지도, `read_csv`, `read_excel`, `to_csv` |
| **2** | **데이터 구조 확인하기, 기초 데이터 다루기** | `info`, `describe`, `loc/iloc`, 필터링, 정렬, `groupby`, `merge` |
| 3 | 데이터 이해하기 (EDA) | 분포, 상관관계, `matplotlib`, `seaborn`, 가설 검증 |
| 4 | 데이터 전처리하기 | 결측치, 이상치, 구간화, 인코딩, 스케일링, `train_test_split` |
| 5 | AI 모델링 필수 개념, 지도학습 I | 과적합, 평가지표, 선형회귀, 로지스틱 회귀 |
| 6 | 지도학습 II | 의사결정나무, 앙상블, 랜덤포레스트, 그라디언트부스팅 |
| 7 | 인공신경망, 심층신경망, 딥러닝 프레임워크 | 퍼셉트론, 활성화함수, Keras `Sequential`, `EarlyStopping` |
| 8 | 비지도학습, 모델 성능 향상시키기 | K-Means, PCA, 교차검증, 하이퍼파라미터 튜닝, 모의고사 |

### 오늘의 학습 목표

1. `head`, `info`, `describe`, `value_counts` 로 처음 보는 데이터의 **크기·자료형·결측·분포** 를 5분 안에 파악할 수 있다.
2. `loc` 와 `iloc` 의 차이를 설명하고, 행·열을 자유롭게 선택할 수 있다.
3. 조건식(`&`, `|`, `~`, `isin`, `between`, `str.contains`)으로 원하는 행만 골라낼 수 있다.
4. 컬럼을 추가·수정·삭제·이름변경하고, `apply`/`map`/`np.where` 로 파생 변수를 만들 수 있다.
5. `groupby`, `pivot_table` 로 그룹별 통계를 내고, `merge`/`concat` 으로 데이터를 결합할 수 있다.
6. 날짜형 컬럼에서 연·월·요일을 추출해 시간 단위로 집계할 수 있다.

### 시간 계획 (180분)

| 시간 | 내용 |
|------|------|
| 00:00 ~ 00:10 | 0. 환경 준비, 데이터 불러오기 |
| 00:10 ~ 01:00 | 1. 데이터 구조 확인하기 |
| 01:00 ~ 01:10 | 휴식 |
| 01:10 ~ 02:30 | 2. 기초 데이터 다루기 |
| 02:30 ~ 02:40 | 휴식 |
| 02:40 ~ 03:00 | 3. 종합 실습 + 정리 |
""")

# ---------------------------------------------------------------- 0. 환경
md(r"""
---
## 0. 환경 준비, 데이터 불러오기
""")
code(r"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 출력 옵션: 컬럼이 많아도 생략하지 않고, 소수점은 2자리까지
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:,.2f}".format)

DATA_DIR = "data"
print("pandas", pd.__version__)
""")
code(r"""
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
""")
md(r"""
1회차에서 만든 파일이 없다면 아래 셀이 **동일한 데이터를 다시 생성** 합니다. (1회차와 같은 seed 를 쓰므로 내용이 같습니다.)
""")
code(r"""
# ===== 1회차 실습 데이터가 없을 때만 재생성 (실행만 하면 됩니다) =====
def ensure_session1_data(data_dir: str = DATA_DIR) -> None:
  required = ["customers.csv", "customers.xlsx", "sales_2024.csv"]
  if all(os.path.exists(f"{data_dir}/{f}") for f in required):
    print("1회차 데이터 확인 완료:", required)
    return

  os.makedirs(data_dir, exist_ok=True)
  rng = np.random.default_rng(42)
  n = 300
  regions = ["서울", "경기", "부산", "대구", "기타"]
  plans = ["5G", "LTE", "3G"]
  customers = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
    "gender": rng.choice(["M", "F"], size=n),
    "age": rng.integers(19, 70, size=n).astype(float),
    "region": rng.choice(regions, size=n, p=[0.35, 0.3, 0.15, 0.1, 0.1]),
    "postal_code": [f"{z:05d}" for z in rng.integers(1000, 63999, size=n)],
    "plan": rng.choice(plans, size=n, p=[0.5, 0.4, 0.1]),
    "monthly_fee": rng.choice([29000, 35000, 45000, 55000, 65000, 79000, 89000], size=n).astype(float),
    "data_usage_gb": np.round(rng.gamma(2.0, 8.0, size=n), 1),
    "join_date": pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 2000, size=n), unit="D"),
  })
  customers["tenure_months"] = ((pd.to_datetime("2024-12-31") - customers["join_date"]).dt.days // 30)
  churn_prob = 0.15 + 0.25 * (customers["tenure_months"] < 12) + 0.1 * (customers["plan"] == "3G")
  customers["churn"] = np.where(rng.random(n) < churn_prob, "Yes", "No")
  customers.loc[rng.choice(n, 12, replace=False), "age"] = np.nan
  customers.loc[rng.choice(n, 8, replace=False), "data_usage_gb"] = np.nan
  customers.to_csv(f"{data_dir}/customers.csv", index=False)

  plan_info = pd.DataFrame({"plan": plans, "speed_mbps": [1000, 150, 10], "launch_year": [2019, 2011, 2006]})
  with pd.ExcelWriter(f"{data_dir}/customers.xlsx") as writer:
    customers.to_excel(writer, sheet_name="customers", index=False)
    plan_info.to_excel(writer, sheet_name="plans", index=False)

  dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
  sales = pd.DataFrame({
    "date": np.repeat(dates, 3),
    "store": np.tile(["강남점", "홍대점", "부산점"], len(dates)),
    "qty": rng.poisson(20, size=len(dates) * 3),
  })
  sales["amount"] = sales["qty"] * rng.choice([12000, 15000, 18000], size=len(sales))
  sales.to_csv(f"{data_dir}/sales_2024.csv", index=False)
  print("1회차 데이터를 재생성했습니다:", required)


ensure_session1_data()
""")
code(r"""
# 1회차에서 배운 옵션을 그대로 사용해 읽는다.
df = pd.read_csv(
  f"{DATA_DIR}/customers.csv",
  dtype={"postal_code": str},
  parse_dates=["join_date"],
)
sales = pd.read_csv(f"{DATA_DIR}/sales_2024.csv", parse_dates=["date"])
plans = pd.read_excel(f"{DATA_DIR}/customers.xlsx", sheet_name="plans")

print("customers:", df.shape, "| sales:", sales.shape, "| plans:", plans.shape)
""")

# ---------------------------------------------------------------- 1. 구조 확인
md(r"""
---
## 1. 데이터 구조 확인하기

### 한 줄 정의
모델링 전에 **"이 데이터가 몇 행 몇 열이고, 각 컬럼이 무슨 타입이며, 어디가 비어 있고, 값이 어떻게 분포하는지"** 를 파악하는 단계.

### 직관적 설명
병원에서 진료 전에 키·몸무게·혈압을 재는 것과 같습니다. 이 단계를 건너뛰면 "문자열 컬럼에 평균을 내려다 에러", "결측치 때문에 학습 실패" 같은 문제를 나중에 훨씬 비싸게 치릅니다.

#### 처음 보는 데이터를 만났을 때의 5단계 루틴

| 순서 | 질문 | 코드 |
|:---:|------|------|
| 1 | 어떻게 생겼나? | `df.head()`, `df.tail()`, `df.sample(5)` |
| 2 | 얼마나 크고 어떤 컬럼이 있나? | `df.shape`, `df.columns`, `df.dtypes` |
| 3 | 타입과 결측은? | `df.info()`, `df.isnull().sum()` |
| 4 | 숫자는 어떤 범위인가? | `df.describe()` |
| 5 | 범주는 어떤 값이 몇 개인가? | `df["col"].value_counts()`, `df["col"].nunique()` |
""")

md(r"""
### 1.1 첫인상: `head` / `tail` / `sample` / `shape` / `columns`
""")
code(r"""
df.head()        # 앞 5행 (괄호 안에 숫자를 넣으면 그만큼)
""")
code(r"""
df.tail(3)       # 뒤 3행: 파일 끝에 합계 행이나 이상한 행이 붙어 있는지 확인
""")
code(r"""
df.sample(5, random_state=1)   # 무작위 5행: 앞부분만 보고 착각하는 것을 방지
""")
code(r"""
print("shape   :", df.shape)            # (행 수, 열 수) 튜플
print("행 수    :", df.shape[0], "=", len(df))
print("열 수    :", df.shape[1])
print("columns :", df.columns.tolist())  # 컬럼명 리스트 (복사해서 쓰기 편함)
print("index   :", df.index)            # 기본은 RangeIndex (0, 1, 2, ...)
""")

md(r"""
### 1.2 `info()`: 자료형과 결측을 한 번에

`info()` 출력을 읽는 법:

```
RangeIndex: 300 entries, 0 to 299      ← 전체 행 수 = 300
 #   Column         Non-Null Count  Dtype
 2   age            288 non-null    float64   ← 300 - 288 = 12개 결측!
 8   join_date      300 non-null    datetime64[ns]
```

- **Non-Null Count 가 전체 행 수보다 작으면 결측치가 있다.**
- **Dtype 이 `object` 면 문자열(또는 섞인 타입)** 이다. 숫자여야 할 컬럼이 `object` 면 정제가 필요하다.
""")
code(r"""
df.info()
""")
code(r"""
# dtypes 만 따로: 타입별로 컬럼을 나눠 볼 때 유용
print(df.dtypes, "\n")

num_cols = df.select_dtypes(include="number").columns.tolist()
obj_cols = df.select_dtypes(include="object").columns.tolist()
print("숫자형 컬럼 :", num_cols)
print("문자형 컬럼 :", obj_cols)
""")

md(r"""
### 1.3 `describe()`: 숫자의 범위와 분포

| 통계량 | 의미 | 확인 포인트 |
|------|------|------|
| `count` | 결측 아닌 개수 | 행 수보다 작으면 결측 |
| `mean` / `50%` | 평균 / 중앙값 | 둘이 크게 다르면 **치우친 분포** 또는 이상치 |
| `std` | 표준편차 | 0 이면 값이 하나뿐 (쓸모없는 컬럼) |
| `min` / `max` | 최소 / 최대 | 나이 -1, 요금 0 처럼 **말이 안 되는 값** 찾기 |
| `25%` / `75%` | 사분위수 | 4회차 이상치(IQR) 계산에 사용 |
""")
code(r"""
df.describe()          # 숫자형 컬럼만 (datetime 은 제외됨)
""")
code(r"""
df.describe().T        # 전치(.T)하면 컬럼이 많을 때 읽기 쉽다
""")
code(r"""
# 문자형(범주형) 컬럼의 요약: 고유값 수, 최빈값, 최빈값 빈도
df.describe(include="object")
""")
code(r"""
# 개별 통계량 직접 계산 (시험에서 "평균을 구하시오" 형태로 출제)
print("나이 평균   :", round(df["age"].mean(), 2))
print("나이 중앙값 :", df["age"].median())
print("나이 최대   :", df["age"].max())
print("요금 표준편차:", round(df["monthly_fee"].std(), 2))
print("사용량 90% 분위수:", df["data_usage_gb"].quantile(0.9))
print("\n여러 컬럼 한 번에:")
print(df[["age", "monthly_fee", "data_usage_gb"]].mean())
""")

md(r"""
### 1.4 `value_counts()`: 범주형 값의 분포

- 기본은 **빈도 내림차순**. `normalize=True` 면 비율.
- 결측치(NaN)는 기본으로 **세지 않는다**. 세려면 `dropna=False`.
- 분류 문제에서 **타깃(y)의 클래스 비율** 을 확인하는 데 반드시 씁니다. (불균형 데이터 판단)
""")
code(r"""
print(df["plan"].value_counts(), "\n")
print(df["plan"].value_counts(normalize=True).round(3), "\n")   # 비율
print("고유값 목록:", df["region"].unique())
print("고유값 개수:", df["region"].nunique())
""")
code(r"""
# 타깃 클래스 비율 확인: 이탈(Yes) 이 약 25% -> 불균형 데이터 (5회차 평가지표 선택에 영향)
df["churn"].value_counts(normalize=True)
""")
code(r"""
# 숫자형 컬럼에도 쓸 수 있다. bins 로 구간 나눠 세기
df["age"].value_counts(bins=5, sort=False)
""")

md(r"""
### 1.5 결측치 개수와 비율

4회차에서 처리 방법을 배우고, 오늘은 **얼마나 비어 있는지 세는 것** 까지만 합니다.
""")
code(r"""
missing = df.isnull().sum()                     # 컬럼별 결측 개수 (isna() 와 동일)
missing_ratio = (df.isnull().mean() * 100).round(1)   # 비율(%) : mean() 은 True 의 비율

missing_table = pd.DataFrame({"결측수": missing, "결측비율(%)": missing_ratio})
missing_table[missing_table["결측수"] > 0]
""")
code(r"""
print("전체 결측치 수      :", df.isnull().sum().sum())
print("결측이 하나라도 있는 행:", df.isnull().any(axis=1).sum())
""")

md(r"""
### 1.6 자료형 변환: `astype`, `to_numeric`, `to_datetime`

| 상황 | 코드 |
|------|------|
| 실수 → 정수 | `df["col"].astype(int)` (결측이 있으면 에러 → 먼저 채우거나 `"Int64"` 사용) |
| 문자 → 숫자 (오류값 포함) | `pd.to_numeric(df["col"], errors="coerce")` : 변환 불가는 NaN |
| 문자 → 날짜 | `pd.to_datetime(df["col"])` |
| 문자 → 범주형 | `df["col"].astype("category")` : 메모리 절약, 순서 지정 가능 |
| 숫자 → 문자 | `df["col"].astype(str)` |
""")
code(r"""
# monthly_fee 는 소수점이 필요 없으니 정수로 (결측이 없어서 바로 가능)
df["monthly_fee"] = df["monthly_fee"].astype(int)

# age 는 결측이 있어 int 로 바꾸면 에러 -> nullable 정수형 "Int64" 사용
df["age"] = df["age"].astype("Int64")

print(df[["monthly_fee", "age"]].dtypes)
df[["monthly_fee", "age"]].head(3)
""")
md(r"""
> ⚠️ `Int64` 의 결측은 `NaN` 이 아니라 `pd.NA` 입니다. `np.where` 같은 numpy 함수에 넣으면 `TypeError: boolean value of NA is ambiguous` 가 납니다.  
> 시험에서는 결측이 있는 숫자 컬럼을 **float 그대로 두고 4회차 방식으로 결측을 채운 뒤** 정수로 바꾸는 편이 안전합니다. 이후 실습을 위해 float 로 되돌립니다.
""")
code(r"""
df["age"] = df["age"].astype(float)
print(df["age"].dtype)
""")
code(r"""
# to_numeric(errors="coerce"): 숫자로 바꿀 수 없는 값은 NaN 처리 (지저분한 데이터에서 매우 자주 사용)
dirty = pd.Series(["100", "200", "삼백", "400", None])
print(pd.to_numeric(dirty, errors="coerce"))
""")
code(r"""
# category 타입: 값 종류가 적은 문자열 컬럼에 사용. 순서를 줄 수도 있다.
df["plan"] = pd.Categorical(df["plan"], categories=["3G", "LTE", "5G"], ordered=True)
print(df["plan"].dtype)
print("정렬하면 지정한 순서를 따른다:", df["plan"].sort_values().unique().tolist())
""")

md(r"""
### 📝 시험 출제 포인트 (1장)

- "데이터의 행과 열 개수를 출력하시오" → `df.shape`
- "각 컬럼의 자료형과 결측치를 확인하시오" → `df.info()`
- "수치형 컬럼의 기초 통계량을 출력하시오" → `df.describe()`
- "`churn` 컬럼의 값별 개수(비율)를 출력하시오" → `df["churn"].value_counts(normalize=True)`
- "결측치 개수를 컬럼별로 출력하시오" → `df.isnull().sum()`
- "`age` 컬럼을 정수형으로 변환하시오" → `astype(int)` (결측이 있으면 먼저 처리)

### ⚠️ 자주 하는 실수 (1장)

- **`df.shape()` 처럼 괄호를 붙임**: `shape`, `columns`, `dtypes`, `index` 는 **속성** 이라 괄호가 없습니다. `head()`, `info()`, `describe()` 는 **메서드** 라 괄호가 있습니다.
- **`describe()` 에 숫자형만 나온다고 당황**: 기본 동작입니다. 문자형은 `include="object"`, 전부는 `include="all"`.
- **`value_counts()` 가 결측을 안 셈**: `dropna=False` 를 줘야 NaN 도 셉니다.
- **`info()` 결과를 변수에 담으려 함**: `info()` 는 화면에 출력만 하고 `None` 을 반환합니다. 결측 개수를 "값"으로 쓰려면 `isnull().sum()` 을 씁니다.
""")

# ---------------------------------------------------------------- 2. 기초 데이터 다루기
md(r"""
---
## 2. 기초 데이터 다루기

### 2.1 컬럼 선택: `df["col"]` vs `df[["col"]]`

| 코드 | 반환 타입 | 모양 |
|------|------|------|
| `df["age"]` | **Series** (1차원) | 값 목록 |
| `df[["age"]]` | **DataFrame** (2차원) | 열이 1개인 표 |
| `df[["age", "plan"]]` | DataFrame | 열이 2개인 표 |

> scikit-learn 의 `X` 는 2차원이어야 하므로 **특징이 하나여도 `df[["col"]]`** 로 뽑습니다. (1회차 복습)
""")
code(r"""
s = df["age"]
d = df[["age"]]
print(type(s).__name__, s.shape)
print(type(d).__name__, d.shape)

df[["customer_id", "age", "plan"]].head(3)
""")

md(r"""
### 2.2 행 선택: `loc` (이름) vs `iloc` (위치)

#### 한 줄 정의
`loc` 는 **인덱스 이름(label)** 으로, `iloc` 는 **정수 위치(0부터)** 로 행·열을 고른다.

| 항목 | `loc` | `iloc` |
|------|------|------|
| 기준 | 인덱스/컬럼 **이름** | 정수 **위치** |
| 슬라이스 끝 | **포함** (`loc[0:3]` → 0,1,2,3) | **미포함** (`iloc[0:3]` → 0,1,2) |
| 열 지정 | `loc[행, "컬럼명"]` | `iloc[행, 열번호]` |
| 조건식 | `loc[df["age"] > 30, "plan"]` 가능 | 불가 (불리언 배열은 가능) |

기본 RangeIndex(0, 1, 2 …) 에서는 이름과 위치가 같아 헷갈리지만, **인덱스를 바꾸면 완전히 달라집니다.**
""")
code(r"""
# 기본 인덱스에서: 이름 == 위치 이지만 슬라이스 끝 포함 여부가 다르다
print("loc[0:2]  ->", len(df.loc[0:2]), "행 (끝 포함)")
print("iloc[0:2] ->", len(df.iloc[0:2]), "행 (끝 미포함)")
""")
code(r"""
# 행 + 열 동시에 지정
print(df.loc[0, "age"])                       # 0번 행의 age
print(df.iloc[0, 2])                          # 0번 행, 2번 열 (age)
df.loc[0:2, ["customer_id", "age", "plan"]]   # 행 범위 + 열 이름 목록
""")
code(r"""
# 인덱스를 customer_id 로 바꾸면 loc 와 iloc 의 차이가 분명해진다
dfc = df.set_index("customer_id")
print(dfc.loc["C0003", ["age", "plan"]].to_dict())   # 이름으로
print(dfc.iloc[2, [1, 4]].to_dict())                 # 위치로 (같은 행)
dfc.loc["C0002":"C0004", "age":"plan"]               # 이름 슬라이스 (양 끝 포함)
""")
code(r"""
# iloc 의 음수 위치, 마지막 행/열
print("마지막 행:", df.iloc[-1]["customer_id"])
print("마지막 열 이름:", df.columns[-1], "| 값 3개:", df.iloc[:3, -1].tolist())
""")

md(r"""
### 2.3 조건 필터링 (Boolean Indexing)

#### 직관적 설명
`df["age"] > 40` 은 각 행마다 True/False 가 적힌 **체크리스트(Series)** 를 만듭니다. 그 체크리스트를 `df[...]` 에 넣으면 True 인 행만 남습니다.

| 조건 | 코드 | 주의 |
|------|------|------|
| AND | `(조건1) & (조건2)` | `and` 아님! 각 조건을 **괄호** 로 감싼다 |
| OR | `(조건1) \| (조건2)` | `or` 아님 |
| NOT | `~(조건)` | `not` 아님 |
| 목록 포함 | `df["region"].isin(["서울", "경기"])` | `==` 여러 개 대신 |
| 범위 | `df["age"].between(30, 39)` | 양 끝 포함 |
| 문자열 포함 | `df["customer_id"].str.contains("00")` | `.str` 접근자 |
| 결측 여부 | `df["age"].isna()` / `.notna()` | `== np.nan` 은 항상 False |
| SQL 스타일 | `df.query("age > 40 and plan == '5G'")` | 문자열 안에 조건 |
""")
code(r"""
cond = df["age"] > 60
print(type(cond).__name__, "| True 개수:", cond.sum())   # True == 1 이므로 sum 이 개수
df[cond].head(3)
""")
code(r"""
# AND / OR / NOT : 각 조건에 괄호 필수
senior_5g = df[(df["age"] >= 60) & (df["plan"] == "5G")]
print("60세 이상 & 5G :", len(senior_5g))

young_or_3g = df[(df["age"] < 25) | (df["plan"] == "3G")]
print("25세 미만 | 3G :", len(young_or_3g))

not_seoul = df[~(df["region"] == "서울")]
print("서울이 아닌 고객:", len(not_seoul))
""")
code(r"""
# isin / between / str.contains / isna
print("수도권(서울,경기)   :", df[df["region"].isin(["서울", "경기"])].shape[0])
print("30대              :", df[df["age"].between(30, 39)].shape[0])
print("ID 에 '00' 포함    :", df[df["customer_id"].str.contains("00")].shape[0])
print("나이 결측          :", df[df["age"].isna()].shape[0])
""")
code(r"""
# query(): 조건이 길 때 읽기 좋다. 외부 변수는 @ 로 참조
min_age = 50
df.query("age >= @min_age and plan == '5G' and churn == 'Yes'")[["customer_id", "age", "plan", "churn"]].head()
""")
code(r"""
# 조건 + 열 선택은 loc 로 한 번에 (시험 단골)
df.loc[df["churn"] == "Yes", ["customer_id", "tenure_months", "monthly_fee"]].head()
""")

md(r"""
### 2.4 정렬: `sort_values`, `sort_index`, `nlargest`

- `sort_values(by=..., ascending=...)` : 값 기준. 여러 컬럼이면 리스트로, 방향도 리스트로.
- 결측치는 기본으로 **맨 뒤** (`na_position="first"` 로 변경 가능).
- 정렬은 **새 DataFrame 을 반환** 하며 원본은 그대로. 원본을 바꾸려면 다시 대입하거나 `inplace=True`.
""")
code(r"""
df.sort_values("monthly_fee", ascending=False).head(3)
""")
code(r"""
# 다중 정렬: 요금제 오름차순, 같은 요금제 안에서는 사용량 내림차순
df.sort_values(["plan", "data_usage_gb"], ascending=[True, False])[["customer_id", "plan", "data_usage_gb"]].head(6)
""")
code(r"""
# 상위/하위 N 개: nlargest / nsmallest (정렬 + head 를 한 번에)
print(df.nlargest(3, "data_usage_gb")[["customer_id", "data_usage_gb"]])
print(df.nsmallest(3, "tenure_months")[["customer_id", "tenure_months"]])
""")
code(r"""
# 정렬 후 인덱스가 뒤섞인다 -> reset_index(drop=True) 로 0부터 다시
top = df.sort_values("age", ascending=False).head(3)
print("정렬 직후 인덱스     :", top.index.tolist())
print("reset_index 후 인덱스:", top.reset_index(drop=True).index.tolist())
""")

md(r"""
### 2.5 컬럼 추가 · 수정 · 삭제 · 이름 변경

| 작업 | 코드 |
|------|------|
| 연산으로 추가 | `df["yearly_fee"] = df["monthly_fee"] * 12` |
| 조건으로 추가 (2갈래) | `df["senior"] = np.where(df["age"] >= 60, "Y", "N")` |
| 조건으로 추가 (여러 갈래) | `np.select([...], [...], default=...)` 또는 `pd.cut` (4회차) |
| 값 매핑 | `df["gender_kr"] = df["gender"].map({"M": "남", "F": "여"})` |
| 함수 적용 | `df["col"].apply(함수)` / `apply(lambda x: ...)` |
| 삭제 | `df.drop(columns=["col1", "col2"])` 또는 `df.drop("col", axis=1)` |
| 이름 변경 | `df.rename(columns={"old": "new"})` |
| 전체 이름 변경 | `df.columns = [...]` (개수가 정확히 맞아야 함) |
""")
code(r"""
df["yearly_fee"] = df["monthly_fee"] * 12                                   # 연산
df["fee_per_gb"] = (df["monthly_fee"] / df["data_usage_gb"]).round(0)       # 결측이 있으면 결과도 NaN
df["is_senior"] = np.where(df["age"] >= 60, "Y", "N")                       # 2갈래 조건
df["gender_kr"] = df["gender"].map({"M": "남", "F": "여"})                  # 매핑

df[["customer_id", "monthly_fee", "yearly_fee", "data_usage_gb", "fee_per_gb", "age", "is_senior", "gender_kr"]].head()
""")
code(r"""
# 여러 갈래 조건: np.select (조건 리스트, 값 리스트, 기본값)
conditions = [
  df["data_usage_gb"] < 10,
  df["data_usage_gb"] < 30,
]
labels = ["low", "mid"]
df["usage_level"] = np.select(conditions, labels, default="high")
df["usage_level"].value_counts()
""")
code(r"""
# apply + lambda: 한 값씩 함수를 적용 (map 으로 안 되는 복잡한 규칙에)
def fee_grade(fee: int) -> str:
  if fee >= 79000:
    return "프리미엄"
  if fee >= 45000:
    return "스탠다드"
  return "베이직"


df["fee_grade"] = df["monthly_fee"].apply(fee_grade)
df["id_number"] = df["customer_id"].apply(lambda x: int(x[1:]))    # "C0007" -> 7
df[["customer_id", "id_number", "monthly_fee", "fee_grade"]].head(3)
""")
code(r"""
# 행 단위 apply (axis=1): 여러 컬럼을 동시에 보는 규칙
df["risk"] = df.apply(
  lambda row: "high" if (row["tenure_months"] < 12 and row["plan"] == "3G") else "normal",
  axis=1,
)
df["risk"].value_counts()
""")
code(r"""
# 삭제와 이름 변경 (원본을 바꾸려면 다시 대입!)
df = df.drop(columns=["id_number", "risk"])
df = df.rename(columns={"gender_kr": "성별", "is_senior": "시니어여부"})
print(df.columns.tolist())
""")
code(r"""
# 컬럼 순서 바꾸기: 원하는 순서의 리스트로 다시 선택
front = ["customer_id", "churn"]
df = df[front + [c for c in df.columns if c not in front]]
df.head(2)
""")

md(r"""
### 2.6 행 다루기: 삭제, 중복, 인덱스

- 행 삭제: `df.drop(index=[0, 1])` 또는 조건 필터링으로 "남길 행" 을 고르는 편이 안전.
- 중복: `df.duplicated()` 로 확인, `df.drop_duplicates()` 로 제거. `subset=` 으로 기준 컬럼 지정.
- 인덱스: `set_index("col")` ↔ `reset_index()`.
""")
code(r"""
# 중복 행 만들어서 확인해 보기
dup = pd.concat([df.head(3), df.head(2)], ignore_index=True)
print("중복 여부:", dup.duplicated().tolist())
print("중복 개수:", dup.duplicated().sum())
print("제거 후 행 수:", len(dup.drop_duplicates()))
print("plan 기준 중복 제거(첫 값 유지):", len(dup.drop_duplicates(subset=["plan"], keep="first")))
""")
code(r"""
# set_index / reset_index 왕복
dfi = df.set_index("customer_id")
print("인덱스 이름:", dfi.index.name, "| 컬럼 수:", dfi.shape[1])
dfr = dfi.reset_index()
print("reset 후 컬럼 수:", dfr.shape[1], "| 첫 컬럼:", dfr.columns[0])
""")

md(r"""
### 2.7 `groupby`: 그룹별 통계

#### 한 줄 정의
"~별 평균/합계/개수" 를 구하는 도구. **분할(split) → 적용(apply) → 결합(combine)** 순서로 동작한다.

#### 직관적 설명
엑셀의 피벗 테이블입니다. `df.groupby("plan")["monthly_fee"].mean()` 은 "요금제별로 묶어서 → 각 묶음의 월요금 → 평균" 입니다.

| 형태 | 코드 | 결과 |
|------|------|------|
| 단일 키, 단일 통계 | `df.groupby("plan")["monthly_fee"].mean()` | Series |
| 다중 키 | `df.groupby(["plan", "gender"])["monthly_fee"].mean()` | MultiIndex Series |
| 여러 통계 | `.agg(["mean", "max", "count"])` | DataFrame |
| 컬럼별 다른 통계 | `.agg({"age": "mean", "monthly_fee": "sum"})` | DataFrame |
| 결과 컬럼 이름 지정 | `.agg(avg_fee=("monthly_fee", "mean"))` | DataFrame |
| 그룹 크기 | `.size()` (결측 포함) vs `.count()` (결측 제외) | |
""")
code(r"""
df.groupby("plan", observed=True)["monthly_fee"].mean()
""")
code(r"""
# 다중 키 + 여러 통계 -> reset_index() 로 평평한 표로
summary = (
  df.groupby(["plan", "gender"], observed=True)["monthly_fee"]
    .agg(["mean", "max", "count"])
    .round(0)
    .reset_index()
)
summary
""")
code(r"""
# 컬럼별로 다른 통계 + 결과 이름 지정 (named aggregation)
df.groupby("region").agg(
  고객수=("customer_id", "count"),
  평균나이=("age", "mean"),
  총요금=("monthly_fee", "sum"),
  이탈률=("churn", lambda s: (s == "Yes").mean()),
).round(2).sort_values("이탈률", ascending=False)
""")
code(r"""
# size vs count: 결측이 있는 컬럼에서 차이가 난다
print("size  (결측 포함):", df.groupby("plan", observed=True).size().to_dict())
print("count (age 결측 제외):", df.groupby("plan", observed=True)["age"].count().to_dict())
""")
code(r"""
# transform: 그룹 통계를 원래 행 길이로 되돌려 붙인다 (그룹 평균 대비 얼마나 큰가?)
df["plan_avg_fee"] = df.groupby("plan", observed=True)["monthly_fee"].transform("mean")
df["fee_vs_plan_avg"] = (df["monthly_fee"] - df["plan_avg_fee"]).round(0)
df[["customer_id", "plan", "monthly_fee", "plan_avg_fee", "fee_vs_plan_avg"]].head()
""")

md(r"""
### 2.8 `pivot_table` 과 `crosstab`

- `pivot_table(index=행, columns=열, values=값, aggfunc=통계)` : groupby 결과를 **2차원 표** 로.
- `pd.crosstab(행, 열)` : 두 범주형 변수의 **빈도표**. `normalize="index"` 로 행 기준 비율.
""")
code(r"""
pd.pivot_table(df, index="region", columns="plan", values="monthly_fee", aggfunc="mean", observed=True).round(0)
""")
code(r"""
# 빈도표 + 합계
pd.crosstab(df["region"], df["churn"], margins=True)
""")
code(r"""
# 행 기준 비율: 지역별 이탈률을 바로 읽을 수 있다
pd.crosstab(df["region"], df["churn"], normalize="index").round(3)
""")

md(r"""
### 2.9 데이터 결합: `merge` 와 `concat`

| 함수 | 방향 | 기준 | 비유 |
|------|------|------|------|
| `pd.merge(left, right, on="key", how=...)` | 옆으로 (열 추가) | **공통 키 값** | SQL JOIN, VLOOKUP |
| `pd.concat([a, b], axis=0)` | 아래로 (행 추가) | 컬럼명 | 파일 이어 붙이기 |
| `pd.concat([a, b], axis=1)` | 옆으로 (열 추가) | **인덱스** | 열 붙이기 |

`how` 옵션:

| how | 남는 행 |
|------|------|
| `inner` (기본) | 양쪽에 모두 있는 키 |
| `left` | 왼쪽 전부 + 오른쪽은 매칭될 때만 (없으면 NaN) |
| `right` | 오른쪽 전부 |
| `outer` | 양쪽 전부 |
""")
code(r"""
print(plans)   # 요금제 정보 (1회차 Excel 의 plans 시트)
""")
code(r"""
# 고객 데이터에 요금제 정보를 붙인다 (plan 컬럼이 공통 키)
df["plan"] = df["plan"].astype(str)          # category 와 object 키 타입을 맞춘다
merged = pd.merge(df, plans, on="plan", how="left")
print("merge 전:", df.shape, "-> 후:", merged.shape)
merged[["customer_id", "plan", "speed_mbps", "launch_year"]].head()
""")
code(r"""
# how 에 따른 행 수 차이 실험: 오른쪽 표에 5G 가 없고, 왼쪽에 없는 "6G" 가 있다면?
plans_partial = pd.DataFrame({"plan": ["LTE", "3G", "6G"], "speed_mbps": [150, 10, 5000]})
for how in ["inner", "left", "right", "outer"]:
  m = pd.merge(df[["customer_id", "plan"]], plans_partial, on="plan", how=how)
  print(f"{how:<6}: {len(m):>4} 행 | speed 결측 {m['speed_mbps'].isna().sum():>3}")
""")
code(r"""
# 키 이름이 다를 때: left_on / right_on
plans_renamed = plans.rename(columns={"plan": "plan_name"})
pd.merge(df[["customer_id", "plan"]], plans_renamed, left_on="plan", right_on="plan_name", how="left").head(3)
""")
code(r"""
# concat: 행 방향으로 이어 붙이기 (예: 월별 파일 합치기)
part1 = df.iloc[:100]
part2 = df.iloc[100:]
stacked = pd.concat([part1, part2], axis=0, ignore_index=True)
print("행 방향 concat:", part1.shape, "+", part2.shape, "->", stacked.shape)

# 열 방향: 인덱스가 같아야 올바르게 붙는다
left_cols = df[["customer_id", "age"]]
right_cols = df[["plan", "churn"]]
side = pd.concat([left_cols, right_cols], axis=1)
print("열 방향 concat:", side.shape)
""")

md(r"""
### 2.10 날짜 다루기: `.dt` 접근자와 시간 단위 집계

`parse_dates` 로 읽은 datetime 컬럼은 `.dt` 로 연·월·일·요일을 꺼낼 수 있습니다.

| 속성 | 의미 |
|------|------|
| `.dt.year` / `.dt.month` / `.dt.day` | 연 / 월 / 일 |
| `.dt.dayofweek` | 요일 (월=0 … 일=6) |
| `.dt.day_name()` | 요일 이름 |
| `.dt.to_period("M")` | 연-월 (예: 2024-03) |
| `.dt.quarter` | 분기 |
""")
code(r"""
sales["year"] = sales["date"].dt.year
sales["month"] = sales["date"].dt.month
sales["dayofweek"] = sales["date"].dt.dayofweek
sales["weekday"] = sales["date"].dt.day_name()
sales["is_weekend"] = sales["dayofweek"] >= 5
sales.head()
""")
code(r"""
# 월별 매장별 매출 합계 (pivot_table)
monthly = pd.pivot_table(sales, index="month", columns="store", values="amount", aggfunc="sum")
monthly.head()
""")
code(r"""
# 요일별 평균 판매량: 주말 효과가 있나?
sales.groupby("dayofweek")["qty"].mean().round(2)
""")
code(r"""
# 월별 총매출 추이 (간단한 시각화 맛보기, 자세한 시각화는 3회차)
monthly_total = sales.groupby("month")["amount"].sum() / 1_000_000

plt.figure(figsize=(8, 3.5))
monthly_total.plot(kind="bar", color="tab:blue")
plt.title("2024년 월별 총매출 (백만원)")
plt.xlabel("월")
plt.ylabel("매출 (백만원)")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.show()
""")
code(r"""
# 날짜 범위 필터링: 문자열 비교가 그대로 된다
q2 = sales[(sales["date"] >= "2024-04-01") & (sales["date"] <= "2024-06-30")]
print("2분기 행 수:", len(q2), "| 총매출:", f"{q2['amount'].sum():,}")

# 연-월 단위 그룹: to_period
sales.groupby(sales["date"].dt.to_period("M"))["amount"].sum().head(3)
""")

md(r"""
### 📝 시험 출제 포인트 (2장)

- "`age` 가 40 이상이고 `plan` 이 `5G` 인 고객 수" → `len(df[(df["age"] >= 40) & (df["plan"] == "5G")])`
- "`monthly_fee` 기준 내림차순 상위 10개" → `df.sort_values("monthly_fee", ascending=False).head(10)` 또는 `nlargest`
- "`region` 별 `monthly_fee` 평균" → `df.groupby("region")["monthly_fee"].mean()`
- "불필요한 컬럼 `customer_id` 삭제" → `df = df.drop(columns=["customer_id"])` (또는 `axis=1`). **모델링 전 ID 컬럼 삭제** 는 거의 매번 출제됩니다.
- "두 데이터를 `plan` 기준으로 결합" → `pd.merge(df1, df2, on="plan", how="left")`
- "컬럼명 변경" → `df.rename(columns={...})`
- 파생 변수 생성 → `np.where`, `apply`, `map` 중 하나

### ⚠️ 자주 하는 실수 (2장)

- **`and` / `or` 사용**: pandas 조건은 `&` / `|` 이고, 각 조건에 **괄호** 를 씌워야 합니다. 안 씌우면 우선순위 때문에 에러(`ValueError: The truth value of a Series is ambiguous`)가 납니다.
- **`drop`, `rename`, `sort_values` 결과를 대입하지 않음**: 이 메서드들은 새 객체를 반환합니다. `df = df.drop(...)` 처럼 다시 대입하거나 `inplace=True` 를 씁니다. 둘 다 하면(`df = df.drop(..., inplace=True)`) `df` 가 `None` 이 됩니다.
- **`loc[0:3]` 과 `iloc[0:3]` 의 행 수가 다름**: `loc` 는 끝을 포함(4행), `iloc` 는 미포함(3행).
- **`groupby` 결과에서 컬럼을 못 찾음**: 그룹 키가 **인덱스** 로 가 있으므로 `reset_index()` 를 하거나 `as_index=False` 를 씁니다.
- **`merge` 후 행이 늘어남**: 오른쪽 표에 키가 중복이면 행이 곱해집니다. 결합 전 `right["key"].is_unique` 를 확인합니다.
- **`SettingWithCopyWarning`**: 필터링한 결과에 바로 값을 대입할 때 발생. `sub = df[cond].copy()` 로 복사본을 만든 뒤 수정합니다.
""")

# ---------------------------------------------------------------- 3. 종합 실습
md(r"""
---
## 3. 종합 실습

`data/customers.csv` 를 다시 읽어 시작합니다. 각 문제의 빈 셀에 코드를 작성하세요. 정답은 접힌 영역에 있습니다.
""")
code(r"""
df = pd.read_csv(f"{DATA_DIR}/customers.csv", dtype={"postal_code": str}, parse_dates=["join_date"])
sales = pd.read_csv(f"{DATA_DIR}/sales_2024.csv", parse_dates=["date"])
plans = pd.read_excel(f"{DATA_DIR}/customers.xlsx", sheet_name="plans")
print(df.shape, sales.shape, plans.shape)
""")
md(r"""
### 문제 1. 구조 파악

`df` 의 행·열 개수, 컬럼별 결측치 개수, `churn` 의 값별 비율을 각각 출력하시오.
""")
code(r"""
# 여기에 코드를 작성하세요
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
print(df.shape)
print(df.isnull().sum())
print(df["churn"].value_counts(normalize=True))
```

</details>

### 문제 2. 조건 필터링

`region` 이 `"서울"` 또는 `"경기"` 이고, `data_usage_gb` 가 30 이상인 고객을 `q2` 에 저장하고 행 수를 출력하시오.
""")
code(r"""
# 여기에 코드를 작성하세요
q2 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
q2 = df[df["region"].isin(["서울", "경기"]) & (df["data_usage_gb"] >= 30)]
print(len(q2))
```

</details>

### 문제 3. 정렬과 선택

`tenure_months` 가 가장 긴 고객 5명의 `customer_id`, `tenure_months`, `monthly_fee` 를 출력하시오. (인덱스는 0부터 다시 매길 것)
""")
code(r"""
# 여기에 코드를 작성하세요
q3 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
q3 = (
  df.sort_values("tenure_months", ascending=False)
    .head(5)[["customer_id", "tenure_months", "monthly_fee"]]
    .reset_index(drop=True)
)
q3
```

</details>

### 문제 4. 파생 변수

`age` 를 기준으로 `age_group` 컬럼을 만드시오. 30 미만 `"20대이하"`, 30 이상 50 미만 `"30-40대"`, 50 이상 `"50대이상"`, 결측은 `"미상"`. 그리고 `age_group` 별 고객 수를 출력하시오.

> **힌트**: `np.select` 를 쓰고, 결측 조건을 **가장 먼저** 두어야 `NaN` 비교(항상 False) 문제를 피할 수 있다.
""")
code(r"""
# 여기에 코드를 작성하세요
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
conditions = [
  df["age"].isna(),
  df["age"] < 30,
  df["age"] < 50,
]
labels = ["미상", "20대이하", "30-40대"]
df["age_group"] = np.select(conditions, labels, default="50대이상")
print(df["age_group"].value_counts())
```

</details>

### 문제 5. 그룹 집계

`plan` 별로 고객 수, 평균 `monthly_fee`, 이탈률(`churn == "Yes"` 비율)을 구해 `q5` 에 저장하시오. 컬럼명은 `고객수`, `평균요금`, `이탈률` 로 하고, 이탈률 내림차순으로 정렬하시오.
""")
code(r"""
# 여기에 코드를 작성하세요
q5 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
q5 = (
  df.groupby("plan")
    .agg(
      고객수=("customer_id", "count"),
      평균요금=("monthly_fee", "mean"),
      이탈률=("churn", lambda s: (s == "Yes").mean()),
    )
    .sort_values("이탈률", ascending=False)
)
q5
```

</details>

### 문제 6. 결합

`df` 와 `plans` 를 `plan` 기준으로 왼쪽 결합(left join)하여 `q6` 에 저장하고, `speed_mbps` 별 평균 `data_usage_gb` 를 출력하시오.
""")
code(r"""
# 여기에 코드를 작성하세요
q6 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
q6 = pd.merge(df, plans, on="plan", how="left")
print(q6.groupby("speed_mbps")["data_usage_gb"].mean().round(2))
```

</details>

### 문제 7. 날짜 집계

`sales` 에서 **분기별(quarter) · 매장별** 매출(`amount`) 합계를 pivot_table 로 만들어 `q7` 에 저장하고 출력하시오. 행은 분기, 열은 매장.
""")
code(r"""
# 여기에 코드를 작성하세요
q7 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
sales["quarter"] = sales["date"].dt.quarter
q7 = pd.pivot_table(sales, index="quarter", columns="store", values="amount", aggfunc="sum")
q7
```

</details>

### 문제 8. 모델링 준비 (5회차 예습)

`df` 에서 `customer_id`, `postal_code`, `join_date` 컬럼을 삭제하고, `churn` 을 `y`, 나머지를 `X` 로 분리하시오. `X.shape`, `y.shape` 를 출력하시오.
""")
code(r"""
# 여기에 코드를 작성하세요
X, y = None, None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
df_model = df.drop(columns=["customer_id", "postal_code", "join_date"])
y = df_model["churn"]
X = df_model.drop(columns=["churn"])
print(X.shape, y.shape)
```

</details>
""")

# ---------------------------------------------------------------- 4. 정리
md(r"""
---
## 4. 오늘의 정리

### 핵심 요약

| 주제 | 기억할 것 |
|------|-----------|
| 구조 확인 루틴 | `head` → `shape` → `info` → `describe` → `value_counts` → `isnull().sum()` |
| 속성 vs 메서드 | `shape`, `columns`, `dtypes` 는 괄호 없음. `head()`, `info()`, `describe()` 는 괄호 있음 |
| 컬럼 선택 | `df["c"]` 는 Series, `df[["c"]]` 는 DataFrame |
| `loc` / `iloc` | 이름 / 위치. `loc` 슬라이스는 끝 포함 |
| 조건 필터링 | `&`, `\|`, `~` + 괄호. `isin`, `between`, `str.contains`, `isna` |
| 파생 변수 | `np.where`(2갈래), `np.select`(여러 갈래), `map`(사전), `apply`(함수) |
| 수정 결과 대입 | `drop`, `rename`, `sort_values` 는 새 객체 반환 → `df = df.xxx(...)` |
| `groupby` | `groupby(키)[값].통계`, `agg`, named aggregation, `reset_index` |
| 결합 | `merge(on, how)` 는 키 기준, `concat(axis)` 는 이어 붙이기 |
| 날짜 | `.dt.year/month/dayofweek`, `to_period("M")` |

### 자기 점검 체크리스트

- [ ] 처음 보는 CSV 를 5분 안에 크기·타입·결측·분포까지 파악할 수 있다.
- [ ] `loc[0:3]` 과 `iloc[0:3]` 의 행 수 차이를 설명할 수 있다.
- [ ] 두 조건 이상을 `&`, `|` 로 결합해 필터링할 수 있다.
- [ ] `np.where` 와 `np.select` 로 조건 파생 변수를 만들 수 있다.
- [ ] `groupby().agg()` 로 그룹별 여러 통계를 한 표로 만들 수 있다.
- [ ] `merge` 의 `how` 네 가지 차이를 행 수로 설명할 수 있다.
- [ ] 날짜 컬럼에서 월을 꺼내 월별 합계를 낼 수 있다.

### 다음 회차 예고 — 3회차: 데이터 이해하기 (EDA)

- 한 변수의 분포 보기: 히스토그램, 박스플롯, `countplot`
- 두 변수의 관계 보기: 산점도, 상관계수(`corr`), `heatmap`
- 타깃(`churn`)과 각 변수의 관계를 그림으로 확인하고 가설 세우기
- `matplotlib` 와 `seaborn` 의 역할 분담, 시험에 나오는 그래프 유형
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11"},
}
nbf.write(nb, OUT)
print("written:", OUT, "cells:", len(cells))
