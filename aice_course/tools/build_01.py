"""1회차 노트북 빌더: AI/ML/DL 개요 + 데이터 획득하기."""
import nbformat as nbf
from pathlib import Path

OUT = Path("/Users/eunbumkim/Desktop/02_프로젝트_코드/aice_test/aice_course/01_AI개요_데이터획득하기.ipynb")
cells = []


def md(src: str) -> None:
  cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
  cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- 표지
md(r"""
# AICE Associate 대비 실습 과정 — 1회차
## AI / ML / DL 개요와 데이터 획득하기

> **과정 구성**: 총 8회 × 3시간, 실습 위주  
> **대상**: Python 기초 문법을 아는 비전공자 ~ 준전공자  
> **환경**: Google Colab 또는 로컬 Jupyter (Python 3.9+, pandas, scikit-learn)

### 전체 커리큘럼

| 회차 | 주제 | 핵심 키워드 |
|:---:|------|------------|
| **1** | **AI/ML/DL 개요, 데이터 획득하기** | AI ⊃ ML ⊃ DL, 지도/비지도, `read_csv`, `read_excel`, `to_csv` |
| 2 | 데이터 구조 확인하기, 기초 데이터 다루기 | `info`, `describe`, `loc/iloc`, 필터링, 정렬, `groupby`, `merge` |
| 3 | 데이터 이해하기 (EDA) | 분포, 상관관계, `matplotlib`, `seaborn`, 가설 검증 |
| 4 | 데이터 전처리하기 | 결측치, 이상치, 구간화, 인코딩, 스케일링, `train_test_split` |
| 5 | AI 모델링 필수 개념, 지도학습 I | 과적합, 평가지표, 선형회귀, 로지스틱 회귀 |
| 6 | 지도학습 II | 의사결정나무, 앙상블, 랜덤포레스트, 그라디언트부스팅 |
| 7 | 인공신경망, 심층신경망, 딥러닝 프레임워크 | 퍼셉트론, 활성화함수, Keras `Sequential`, `EarlyStopping` |
| 8 | 비지도학습, 모델 성능 향상시키기 | K-Means, PCA, 교차검증, 하이퍼파라미터 튜닝, 모의고사 |

### 오늘의 학습 목표

1. AI, 머신러닝, 딥러닝의 관계와 차이를 **한 문장으로** 설명할 수 있다.
2. 지도학습 / 비지도학습 / 강화학습을 구분하고, 회귀와 분류 문제를 구별할 수 있다.
3. AI 모델링의 전체 흐름(데이터 획득 → 전처리 → 모델링 → 평가)을 그릴 수 있다.
4. CSV / Excel / JSON / URL / 내장 데이터셋에서 `pandas` 로 데이터를 읽어올 수 있다.
5. 읽기 옵션(`sep`, `encoding`, `header`, `index_col`, `usecols`, `na_values` …)을 상황에 맞게 쓸 수 있다.
6. 데이터를 파일로 저장하고, 파일 입출력에서 자주 나는 오류를 스스로 해결할 수 있다.

### 시간 계획 (180분)

| 시간 | 내용 |
|------|------|
| 00:00 ~ 00:15 | 0. 실습 환경 준비 |
| 00:15 ~ 00:55 | 1. AI / ML / DL 정의 |
| 00:55 ~ 01:05 | 휴식 |
| 01:05 ~ 02:25 | 2. 데이터 획득하기 |
| 02:25 ~ 02:35 | 휴식 |
| 02:35 ~ 03:00 | 3. 종합 실습 + 정리 |
""")

# ---------------------------------------------------------------- 0. 환경
md(r"""
---
## 0. 실습 환경 준비

### 0.1 Colab vs 로컬

| 항목 | Google Colab | 로컬 Jupyter |
|------|--------------|--------------|
| 설치 | 불필요 (브라우저, 모두 기본 설치됨) | Python + `pip install numpy pandas matplotlib seaborn scikit-learn tensorflow openpyxl` |
| 파일 위치 | `/content/` (세션 종료 시 삭제) | 노트북이 있는 폴더 |
| 한글 폰트 | 나눔폰트 설치 + 런타임 재시작 필요 | Mac: AppleGothic, Windows: Malgun Gothic |
| AICE 시험 환경 | 시험도 **웹 브라우저 기반 Jupyter** 로 진행 | — |

#### 시험에서 사용하는 라이브러리

| 라이브러리 | 용도 | 시험 출제 영역 |
|------|------|------|
| `numpy` | 수치 배열 연산 | 전 영역 기반 |
| `pandas` | 표 데이터 읽기·가공 | 데이터 획득, 구조 확인, 전처리 |
| `matplotlib` | 기본 시각화 | 데이터 이해(EDA) |
| `seaborn` | 통계 시각화 (`histplot`, `boxplot`, `heatmap`, `countplot`) | 데이터 이해(EDA) |
| `scikit-learn` | 전처리, 머신러닝 모델, 평가지표 | 전처리, 모델링, 평가 |
| `tensorflow` / `keras` | 딥러닝(DNN) 모델 | 모델링(딥러닝 문항) |

> 💡 AICE Associate 시험은 인터넷 검색이 허용되지 않지만, **Jupyter 의 자동완성(Tab)과 `?` 도움말은 사용 가능**합니다. 이번 과정에서는 도움말을 적극 활용하는 습관을 들입니다.
""")
code(r"""
# 실습에 사용하는 라이브러리를 불러오고 버전을 확인한다.
# 버전이 달라도 이 과정의 코드는 대부분 그대로 동작한다.
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

print(f"Python       : {sys.version.split()[0]}")
print(f"numpy        : {np.__version__}")
print(f"pandas       : {pd.__version__}")
print(f"matplotlib   : {matplotlib.__version__}")
print(f"seaborn      : {sns.__version__}")
print(f"scikit-learn : {sklearn.__version__}")
""")
code(r"""
# TensorFlow 는 크기가 커서 import 에 수 초가 걸린다. 설치가 안 되어 있으면 안내만 출력한다.
# 로컬 설치: pip install tensorflow   (Apple Silicon Mac: pip install tensorflow-macos)
try:
  import tensorflow as tf
  from tensorflow import keras
  print(f"tensorflow   : {tf.__version__}")
  print(f"keras        : {getattr(keras, '__version__', '(tensorflow 내장 keras)')}")
  print(f"GPU 사용 가능 : {len(tf.config.list_physical_devices('GPU')) > 0}")
except ImportError:
  print("tensorflow 미설치 - 7회차(딥러닝) 전까지 설치하면 됩니다.")
""")
code(r"""
# 한글 폰트 설정: 환경에 맞는 폰트를 자동으로 고른다.
# Colab 이라면 아래 주석의 명령을 먼저 실행하고 런타임을 재시작해야 한다.
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
code(r"""
# 실습 파일을 저장할 폴더를 만든다. 이미 있으면 그대로 사용한다.
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

print("현재 작업 폴더:", os.getcwd())
print("data 폴더 존재 여부:", os.path.isdir(DATA_DIR))
""")

# ---------------------------------------------------------------- 1. AI/ML/DL
md(r"""
---
## 1. AI / ML / DL 정의

### 1.1 세 용어의 관계

#### 한 줄 정의

| 용어 | 한 줄 정의 |
|------|-----------|
| **AI (인공지능)** | 사람처럼 판단·예측·행동하는 컴퓨터 프로그램 전체 |
| **ML (머신러닝)** | 규칙을 사람이 코딩하지 않고, **데이터에서 규칙을 스스로 찾게** 하는 AI 의 한 방법 |
| **DL (딥러닝)** | 여러 층의 **인공신경망**으로 규칙을 찾는 머신러닝의 한 방법 |

#### 직관적 설명

```
┌──────────────────────────────── AI ────────────────────────────────┐
│  규칙 기반 시스템(if-else 전문가 시스템), 탐색 알고리즘 …             │
│   ┌────────────────────────── ML ───────────────────────────┐      │
│   │  선형회귀, 로지스틱회귀, 의사결정나무, 랜덤포레스트, SVM … │      │
│   │   ┌───────────────────── DL ─────────────────────┐       │      │
│   │   │  DNN, CNN(이미지), RNN/LSTM(시계열), Transformer │       │      │
│   │   └──────────────────────────────────────────────┘       │      │
│   └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

- **AI 가 가장 큰 개념**이고, ML 은 그 안의 "데이터로 학습하는" 부분, DL 은 ML 안에서 "신경망을 깊게 쌓은" 부분입니다.
- 딥러닝도 머신러닝이므로, 이 과정에서 배우는 **전처리 · 평가 방법은 딥러닝에도 그대로** 적용됩니다.
""")

md(r"""
### 1.2 전통 프로그래밍 vs 머신러닝

| 항목 | 전통 프로그래밍 | 머신러닝 |
|------|----------------|---------|
| 입력 | **규칙 + 데이터** | **데이터 + 정답** |
| 출력 | 결과 | **규칙(모델)** |
| 예시 | 섭씨 → 화씨 공식을 코드로 작성 | 섭씨·화씨 쌍을 보여주고 공식을 찾게 함 |
| 언제 유리 | 규칙이 명확하고 단순할 때 | 규칙이 복잡하거나 사람이 설명하기 어려울 때 (이미지, 언어, 고객 이탈…) |

아래 실습에서 **같은 문제를 두 방식으로** 풀어 봅니다.
""")
code(r"""
# [전통 프로그래밍] 사람이 규칙(공식)을 직접 코딩한다.
def celsius_to_fahrenheit(c: float) -> float:
  return c * 1.8 + 32


for c in [0, 10, 25, 100]:
  print(f"{c:>4}°C -> {celsius_to_fahrenheit(c):.1f}°F")
""")
code(r"""
# [머신러닝] 규칙은 모른 채, 데이터(입력 X, 정답 y)만 주고 규칙을 찾게 한다.
from sklearn.linear_model import LinearRegression

X = np.array([[-10], [0], [10], [20], [30], [40]])   # 입력: 섭씨 (2차원 배열이어야 함!)
y = np.array([14, 32, 50, 68, 86, 104])              # 정답: 화씨

model = LinearRegression()
model.fit(X, y)                                       # 학습: 데이터에서 규칙 찾기

print(f"찾아낸 규칙: 화씨 = {model.coef_[0]:.2f} × 섭씨 + {model.intercept_:.2f}")
print(f"25°C 예측  : {model.predict([[25]])[0]:.1f}°F  (정답 77.0)")
""")
code(r"""
# 학습된 규칙을 그림으로 확인한다.
x_line = np.linspace(-20, 50, 100).reshape(-1, 1)

plt.figure(figsize=(6, 4))
plt.scatter(X, y, color="tab:red", label="학습 데이터 (섭씨, 화씨)")
plt.plot(x_line, model.predict(x_line), label="모델이 찾은 규칙")
plt.xlabel("섭씨 (°C)")
plt.ylabel("화씨 (°F)")
plt.title("머신러닝: 데이터에서 규칙을 찾는다")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
""")
md(r"""
> **📌 핵심**: 머신러닝의 `fit()` 은 "데이터를 보고 규칙(파라미터)을 찾아라", `predict()` 는 "찾은 규칙으로 새 입력의 답을 내라" 입니다.  
> 이 두 메서드 이름은 scikit-learn 의 **모든 모델에서 동일**합니다. (5회차에서 자세히)
""")

md(r"""
### 1.3 머신러닝의 세 가지 종류

| 종류 | 정답(label) 유무 | 목적 | 대표 알고리즘 | 예시 |
|------|:---:|------|------|------|
| **지도학습**<br>(Supervised) | **있음** | 정답을 맞히는 규칙 학습 | 선형회귀, 로지스틱회귀, 의사결정나무, 랜덤포레스트, 신경망 | 집값 예측, 이탈 여부 분류 |
| **비지도학습**<br>(Unsupervised) | **없음** | 데이터의 숨은 구조 발견 | K-Means, 계층 군집, PCA | 고객 세분화, 차원 축소 |
| **강화학습**<br>(Reinforcement) | 보상(reward) | 시행착오로 최적 행동 학습 | Q-Learning, DQN | 게임 AI, 로봇 제어 |

#### 지도학습은 다시 두 가지로 나뉜다

| 구분 | **회귀 (Regression)** | **분류 (Classification)** |
|------|------|------|
| 정답 y 의 형태 | **연속된 숫자** (집값, 온도, 매출) | **범주** (Yes/No, 고양이/개, 등급 A/B/C) |
| 질문 형태 | "얼마?" "몇 개?" | "어느 쪽?" "무엇?" |
| 평가 지표 | MAE, MSE, RMSE, R² | 정확도, 정밀도, 재현율, F1 |
| 대표 모델 | `LinearRegression` | `LogisticRegression` (이름에 회귀가 있지만 **분류** 모델!) |

> **시험 팁**: 문제에서 예측 대상 컬럼(y)의 값이 **숫자면 회귀, 카테고리면 분류**. 이것만 정확히 판단해도 모델·평가지표 선택이 결정됩니다.
""")
code(r"""
# 같은 고객 데이터에서 "회귀 문제"와 "분류 문제"를 정의해 본다.
# y 의 형태만 다를 뿐, X(입력)는 동일할 수 있다.
customers = pd.DataFrame({
  "age": [25, 34, 45, 52, 23, 38],
  "tenure_months": [3, 24, 60, 84, 1, 36],
  "monthly_fee": [35000, 55000, 89000, 79000, 29000, 65000],   # 회귀 타깃 후보
  "churn": ["Yes", "No", "No", "No", "Yes", "No"],              # 분류 타깃 후보
})
customers
""")
code(r"""
# 회귀 문제: 나이·가입기간으로 월 요금(연속값)을 예측
X = customers[["age", "tenure_months"]]
y_reg = customers["monthly_fee"]
print("회귀 타깃 dtype :", y_reg.dtype, "-> 숫자이므로 회귀")

# 분류 문제: 나이·가입기간으로 이탈 여부(범주)를 예측
y_clf = customers["churn"]
print("분류 타깃 dtype :", y_clf.dtype, "-> 문자열(범주)이므로 분류")
print("분류 타깃 종류  :", y_clf.unique(), "-> 클래스가 2개이므로 이진 분류")
""")

md(r"""
### 1.4 딥러닝은 언제 쓰는가

| 항목 | 머신러닝 (전통 ML) | 딥러닝 |
|------|------|------|
| 특징(feature) 추출 | **사람이** 컬럼을 설계 | **모델이** 원시 데이터에서 스스로 추출 |
| 잘 맞는 데이터 | 표(테이블) 형태, 수천 ~ 수십만 행 | 이미지, 음성, 텍스트, 대용량 |
| 필요한 데이터 양 | 비교적 적음 | 많음 |
| 학습 시간 / 자원 | 짧음, CPU 로 충분 | 김, GPU 권장 |
| 해석 가능성 | 높음 (트리, 회귀 계수) | 낮음 (블랙박스) |
| 라이브러리 | scikit-learn | TensorFlow/Keras, PyTorch |

> AICE Associate 시험은 **테이블 데이터** 를 다루므로 scikit-learn 모델이 중심이고, 마지막에 Keras 로 간단한 신경망(DNN)을 만드는 문항이 나옵니다. (7회차)
""")

md(r"""
### 1.5 AI 모델링의 전체 흐름

AICE 시험 문항은 아래 순서 그대로 출제됩니다. 이 과정의 회차도 이 순서를 따릅니다.

```
① 데이터 획득  →  ② 구조 확인  →  ③ 탐색(EDA)  →  ④ 전처리  →  ⑤ 모델링  →  ⑥ 평가  →  ⑦ 성능 개선
   read_csv       info/describe   시각화/상관      결측/인코딩    fit          predict/score   튜닝/앙상블
   (1회차)        (2회차)          (3회차)          (4회차)       (5~7회차)    (5~7회차)       (8회차)
```

아래는 **전체 흐름을 5분 만에 훑는 맛보기**입니다. 지금은 코드를 이해하지 못해도 괜찮습니다. 흐름만 눈에 익히세요.
""")
code(r"""
# ===== AI 모델링 전체 흐름 맛보기 (붓꽃 품종 분류) =====
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ① 데이터 획득
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# ② 구조 확인
print("데이터 크기:", df.shape)
display(df.head(3))

# ③ 탐색 (간단히 클래스 분포만)
print("품종별 개수:\n", df["species"].value_counts().to_dict())

# ④ 전처리: 입력 X 와 정답 y 분리, 학습용/평가용 분할
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⑤ 모델링: 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ⑥ 평가: 예측 후 정확도
y_pred = model.predict(X_test)
print(f"\n테스트 정확도: {accuracy_score(y_test, y_pred):.3f}")
""")

md(r"""
### 📝 시험 출제 포인트 (1장)

- AICE Associate 는 **실기 시험** 이라 "AI 의 정의" 를 직접 묻지는 않지만, **문제 지문을 읽고 회귀인지 분류인지 판단** 하는 능력이 모든 모델링 문항의 출발점입니다.
- `fit()` → `predict()` 흐름과, 학습용/평가용 데이터를 나누는 이유(**과적합 확인**)는 매 시험 출제됩니다. (4~5회차)
- `LogisticRegression` 은 이름과 달리 **분류** 모델입니다.

### ⚠️ 자주 하는 실수 (1장)

- **"y 가 0/1 숫자니까 회귀"**: 0/1 은 범주를 숫자로 표현한 것이므로 **분류** 입니다. 값의 의미가 "양(quantity)" 인지 "종류(category)" 인지로 판단합니다.
- **입력 X 를 1차원으로 넣음**: scikit-learn 의 `fit(X, y)` 에서 X 는 항상 **2차원** (행=샘플, 열=특징) 이어야 합니다. 특징이 1개여도 `X.reshape(-1, 1)` 또는 `df[["col"]]` (대괄호 두 개) 로 2차원을 유지합니다.
- **딥러닝이 항상 좋다는 생각**: 테이블 데이터에서는 랜덤포레스트·그라디언트부스팅이 신경망보다 좋은 경우가 매우 흔합니다.
""")

# ---------------------------------------------------------------- 2. 데이터 획득
md(r"""
---
## 2. 데이터 획득하기

### 2.1 데이터는 어디서 오는가

| 출처 | 형태 | pandas 함수 | 실무/시험 빈도 |
|------|------|-------------|:---:|
| 파일 | **CSV** (`.csv`, `.txt`) | `pd.read_csv()` | ★★★★★ |
| 파일 | Excel (`.xlsx`) | `pd.read_excel()` | ★★★ |
| 파일 | JSON (`.json`) | `pd.read_json()` | ★★ |
| 웹 | URL 의 CSV | `pd.read_csv("https://…")` | ★★ |
| 라이브러리 내장 | scikit-learn, seaborn 샘플 | `load_iris()`, `sns.load_dataset()` | ★★★ (연습용) |
| 데이터베이스 | SQL | `pd.read_sql()` | ★ (시험 미출제) |
| 직접 생성 | dict / list / numpy | `pd.DataFrame()` | ★★★ |

> AICE 시험에서는 **문제에서 지정한 파일명과 변수명을 정확히 사용** 해야 합니다. 예: `df = pd.read_csv("data.csv")` 처럼 변수명이 `df` 로 지정되면 이후 문항 채점이 이 변수를 기준으로 이뤄집니다.
""")

md(r"""
### 2.2 실습용 파일 만들기

읽기 연습을 하려면 먼저 파일이 있어야 합니다. 아래 셀은 **가상의 통신사 고객 데이터** 를 만들어 여러 형식으로 저장합니다.  
(생성 코드 자체는 지금 이해하지 않아도 됩니다. 실행만 하세요.)
""")
code(r"""
# ===== 실습용 가상 데이터 생성 (실행만 하면 됩니다) =====
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

# 결측치를 일부러 심는다 (4회차 전처리에서 다룰 예정)
customers.loc[rng.choice(n, 12, replace=False), "age"] = np.nan
customers.loc[rng.choice(n, 8, replace=False), "data_usage_gb"] = np.nan

print("생성된 데이터 크기:", customers.shape)
customers.head()
""")
code(r"""
# 여러 형식으로 저장 (2.6 절에서 저장 함수를 자세히 배운다)
customers.to_csv(f"{DATA_DIR}/customers.csv", index=False)                              # 기본 CSV (UTF-8)
customers.to_csv(f"{DATA_DIR}/customers_cp949.csv", index=False, encoding="cp949")      # 윈도우 한글 인코딩
customers.to_csv(f"{DATA_DIR}/customers_semicolon.txt", index=False, sep=";")          # 구분자가 세미콜론
customers.to_csv(f"{DATA_DIR}/customers_noheader.csv", index=False, header=False)     # 헤더 없음
customers.to_json(f"{DATA_DIR}/customers.json", orient="records", force_ascii=False, indent=2)

# 결측치가 "-" 로 표기되고, 맨 위에 설명 2줄이 붙은 "지저분한" 파일
with open(f"{DATA_DIR}/customers_raw.csv", "w", encoding="utf-8") as f:
  f.write("# 통신사 고객 데이터 (2024-12 기준)\n")
  f.write("# 결측값은 - 로 표기\n")
  customers.to_csv(f, index=False, na_rep="-")

# Excel: 시트 2개 (customers, plans)
plan_info = pd.DataFrame({
  "plan": plans,
  "speed_mbps": [1000, 150, 10],
  "launch_year": [2019, 2011, 2006],
})
with pd.ExcelWriter(f"{DATA_DIR}/customers.xlsx") as writer:
  customers.to_excel(writer, sheet_name="customers", index=False)
  plan_info.to_excel(writer, sheet_name="plans", index=False)

# 날짜가 있는 매출 데이터
dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
sales = pd.DataFrame({
  "date": np.repeat(dates, 3),
  "store": np.tile(["강남점", "홍대점", "부산점"], len(dates)),
  "qty": rng.poisson(20, size=len(dates) * 3),
})
sales["amount"] = sales["qty"] * rng.choice([12000, 15000, 18000], size=len(sales))
sales.to_csv(f"{DATA_DIR}/sales_2024.csv", index=False)

print("생성된 파일 목록:")
for name in sorted(os.listdir(DATA_DIR)):
  size_kb = os.path.getsize(f"{DATA_DIR}/{name}") / 1024
  print(f"  {name:<28} {size_kb:7.1f} KB")
""")

md(r"""
### 2.3 CSV 읽기: `pd.read_csv()`

#### 한 줄 정의
쉼표(`,`)로 구분된 텍스트 파일을 읽어 **DataFrame** 으로 만드는 함수. 데이터 분석의 90% 는 여기서 시작합니다.

#### 직관적 설명
CSV 는 "엑셀에서 서식을 다 빼고 값만 쉼표로 이어 붙인 텍스트" 입니다. 그래서 어떤 프로그램에서도 열리고 용량이 작습니다.  
`read_csv` 는 그 텍스트를 **표(행 × 열)** 로 되돌리는 번역기입니다. 번역기이므로 "어떤 문자로 구분했는지, 어떤 인코딩인지, 첫 줄이 제목인지" 를 알려 줘야 할 때가 있습니다.

#### 자주 쓰는 파라미터

| 파라미터 | 의미 | 언제 쓰나 |
|------|------|------|
| `filepath_or_buffer` | 파일 경로 또는 URL | 항상 (첫 번째 인자) |
| `sep` / `delimiter` | 구분 문자 (기본 `,`) | 탭(`\t`), 세미콜론(`;`), 파이프(`\|`) 파일 |
| `encoding` | 문자 인코딩 (기본 `utf-8`) | 한글 깨짐 → `cp949` 또는 `euc-kr` |
| `header` | 컬럼명이 있는 행 번호 (기본 0) | 컬럼명이 없으면 `header=None` |
| `names` | 컬럼명 직접 지정 | `header=None` 과 함께 |
| `index_col` | 인덱스로 쓸 컬럼 | `id` 컬럼을 인덱스로 |
| `usecols` | 읽을 컬럼만 선택 | 큰 파일에서 일부 컬럼만 |
| `nrows` | 앞에서 n 행만 읽기 | 큰 파일 미리보기 |
| `skiprows` | 건너뛸 행 | 파일 앞의 설명 줄 제거 |
| `na_values` | 결측치로 취급할 문자열 | `"-"`, `"?"`, `"없음"` 등 |
| `dtype` | 컬럼 자료형 지정 | 우편번호·전화번호의 앞자리 0 보존 |
| `parse_dates` | 날짜로 변환할 컬럼 | 날짜 연산이 필요할 때 |
""")
code(r"""
# [기본] 파일 경로만 주면 된다. 첫 줄은 자동으로 컬럼명이 된다.
df = pd.read_csv(f"{DATA_DIR}/customers.csv")

print("크기 (행, 열):", df.shape)
df.head()
""")
code(r"""
# 읽은 직후에는 반드시 자료형을 확인하는 습관을 들인다.
# postal_code 가 숫자(int64)로 읽혀 앞자리 0 이 사라졌고, join_date 는 문자열(object)이다.
df.dtypes
""")
code(r"""
# [dtype / parse_dates] 자료형을 읽는 시점에 바로잡는다.
df = pd.read_csv(
  f"{DATA_DIR}/customers.csv",
  dtype={"postal_code": str},     # 앞자리 0 보존
  parse_dates=["join_date"],      # 문자열 -> datetime64
)

print(df.dtypes, "\n")
print("postal_code 예시:", df["postal_code"].head(3).tolist())
print("join_date 연도  :", df["join_date"].dt.year.head(3).tolist())   # 날짜형이라 .dt 사용 가능
""")
code(r"""
# [sep] 구분자가 쉼표가 아니면 컬럼이 1개로 뭉쳐 읽힌다. -> sep 지정
wrong = pd.read_csv(f"{DATA_DIR}/customers_semicolon.txt")
print("sep 미지정 -> 컬럼 수:", wrong.shape[1], "| 컬럼명:", wrong.columns.tolist()[:1])

right = pd.read_csv(f"{DATA_DIR}/customers_semicolon.txt", sep=";")
print("sep=';'    -> 컬럼 수:", right.shape[1])
right.head(3)
""")
code(r"""
# [encoding] 윈도우에서 만든 한글 CSV 는 cp949 인 경우가 많다.
# 기본(utf-8)으로 읽으면 UnicodeDecodeError 가 난다. 직접 확인해 보자.
try:
  pd.read_csv(f"{DATA_DIR}/customers_cp949.csv")
except UnicodeDecodeError as e:
  print("UnicodeDecodeError 발생:", str(e)[:60], "...")

df_kr = pd.read_csv(f"{DATA_DIR}/customers_cp949.csv", encoding="cp949")
print("\nencoding='cp949' 로 성공. region 값:", df_kr["region"].unique())
""")
code(r"""
# [header / names] 컬럼명 행이 없는 파일
no_header_wrong = pd.read_csv(f"{DATA_DIR}/customers_noheader.csv")
print("header 미지정 -> 첫 데이터 행이 컬럼명이 되어 버림:", no_header_wrong.columns.tolist()[:3])

col_names = ["customer_id", "gender", "age", "region", "postal_code", "plan",
             "monthly_fee", "data_usage_gb", "join_date", "tenure_months", "churn"]
no_header = pd.read_csv(f"{DATA_DIR}/customers_noheader.csv", header=None, names=col_names)
print("header=None, names=... ->", no_header.columns.tolist()[:3])
no_header.head(3)
""")
code(r"""
# [index_col / usecols / nrows] 필요한 부분만 골라 읽기
df_part = pd.read_csv(
  f"{DATA_DIR}/customers.csv",
  index_col="customer_id",                        # customer_id 를 행 인덱스로
  usecols=["customer_id", "age", "plan", "churn"],  # 4개 컬럼만
  nrows=5,                                        # 앞 5행만
)
df_part
""")
code(r"""
# [skiprows / na_values] 지저분한 파일 처리
# 파일의 앞 3줄을 직접 들여다본다. (문제가 생기면 항상 원본 텍스트를 먼저 본다!)
with open(f"{DATA_DIR}/customers_raw.csv", encoding="utf-8") as f:
  for _ in range(4):
    print(repr(f.readline()))
""")
code(r"""
# 설명 2줄은 건너뛰고, "-" 는 결측치로 읽는다.
df_raw = pd.read_csv(
  f"{DATA_DIR}/customers_raw.csv",
  skiprows=2,          # 앞 2줄 무시 (comment="#" 으로도 가능)
  na_values=["-"],     # "-" 를 NaN 으로
)
print("age 결측치 개수:", df_raw["age"].isna().sum())
print("age dtype      :", df_raw["age"].dtype, "  <- na_values 를 안 주면 '-' 때문에 object 가 된다")
""")
code(r"""
# 비교: na_values 를 안 주면 age 가 문자열(object) 컬럼이 되어 평균 계산이 불가능하다.
df_bad = pd.read_csv(f"{DATA_DIR}/customers_raw.csv", skiprows=2)
print("na_values 미지정 -> age dtype:", df_bad["age"].dtype)
print("고유값 예시:", df_bad["age"].unique()[:6])
""")
md(r"""
> **💡 도움말 활용**: 파라미터가 기억나지 않으면 셀에서 `pd.read_csv?` 를 실행하거나 `pd.read_csv(` 입력 후 `Shift+Tab` 을 누르세요. 시험장에서도 사용할 수 있습니다.
""")
code(r"""
# 도움말 보기 (출력이 길어서 앞부분만 표시)
doc = pd.read_csv.__doc__
print(doc[:600])
""")

md(r"""
### 2.4 Excel, JSON, URL, 내장 데이터셋 읽기

#### Excel: `pd.read_excel()`

- `openpyxl` 패키지가 필요합니다 (`pip install openpyxl`).
- `sheet_name` 으로 시트를 선택합니다. 이름(문자열) 또는 순서(0, 1, …).
- `sheet_name=None` 이면 **모든 시트를 dict** 로 읽습니다.
""")
code(r"""
# 시트 이름으로 읽기
df_xl = pd.read_excel(f"{DATA_DIR}/customers.xlsx", sheet_name="customers")
print("customers 시트:", df_xl.shape)

plans_xl = pd.read_excel(f"{DATA_DIR}/customers.xlsx", sheet_name="plans")
print("plans 시트    :", plans_xl.shape)
plans_xl
""")
code(r"""
# 모든 시트를 한 번에: {시트명: DataFrame} 형태의 dict
sheets = pd.read_excel(f"{DATA_DIR}/customers.xlsx", sheet_name=None)
print("시트 목록:", list(sheets.keys()))
print("타입     :", type(sheets["plans"]))
""")
md(r"""
#### JSON: `pd.read_json()`

- JSON 은 웹 API 응답의 표준 형식입니다.
- `[{"a": 1, "b": 2}, {"a": 3, "b": 4}]` 처럼 **레코드 목록(records)** 형태가 가장 흔하고, `read_json` 이 바로 표로 바꿔 줍니다.
""")
code(r"""
# 파일 앞부분을 먼저 본다.
with open(f"{DATA_DIR}/customers.json", encoding="utf-8") as f:
  print(f.read(220), "...")
""")
code(r"""
df_json = pd.read_json(f"{DATA_DIR}/customers.json")
print(df_json.shape)
df_json.head(3)
""")
code(r"""
# 중첩(nested) JSON 은 json.normalize 로 펼친다. (API 응답 처리에 자주 필요)
nested = [
  {"id": 1, "name": "김철수", "contact": {"email": "kim@example.com", "phone": "010-1111-2222"}},
  {"id": 2, "name": "이영희", "contact": {"email": "lee@example.com", "phone": "010-3333-4444"}},
]
pd.json_normalize(nested)
""")
md(r"""
#### URL 에서 바로 읽기

`read_csv` 의 첫 인자에 **http(s) 주소** 를 주면 다운로드 없이 바로 읽습니다. 인터넷이 없는 환경(시험장)에서는 동작하지 않으므로 `try / except` 로 감쌌습니다.
""")
code(r"""
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
try:
  tips = pd.read_csv(url)
  print("URL 읽기 성공:", tips.shape)
  display(tips.head(3))
except Exception as e:
  print("인터넷 연결이 없거나 URL 접근 실패:", type(e).__name__)
""")
md(r"""
#### 라이브러리 내장 데이터셋

연습·시험용으로 자주 등장합니다. scikit-learn 의 `load_*` 함수는 **Bunch** 라는 dict 비슷한 객체를 돌려주므로 **DataFrame 으로 변환하는 방법** 을 반드시 알아 두세요.
""")
code(r"""
from sklearn.datasets import load_iris, load_diabetes, load_wine

iris = load_iris()
print("반환 타입 :", type(iris).__name__)
print("키 목록   :", list(iris.keys()))
print("특징 이름 :", iris.feature_names)
print("정답 이름 :", iris.target_names)
""")
code(r"""
# 방법 1: 직접 조립 (가장 범용적)
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target
iris_df["species"] = iris_df["target"].map(dict(enumerate(iris.target_names)))
iris_df.head()
""")
code(r"""
# 방법 2: as_frame=True (scikit-learn 0.23+) -> .frame 속성에 완성된 DataFrame
wine_df = load_wine(as_frame=True).frame
print(wine_df.shape)
wine_df.head(3)
""")
code(r"""
# seaborn 내장 데이터셋 (인터넷에서 다운로드하므로 오프라인이면 실패할 수 있음)
import seaborn as sns

try:
  titanic = sns.load_dataset("titanic")
  print("titanic:", titanic.shape)
  display(titanic.head(3))
except Exception as e:
  print("seaborn 데이터셋 다운로드 실패 (오프라인):", type(e).__name__)
""")

md(r"""
### 2.5 데이터 직접 만들기: `pd.DataFrame()`

시험에서 작은 예시 데이터를 만들거나, 예측 결과를 표로 정리할 때 씁니다. **dict → DataFrame** 이 가장 많이 쓰입니다.

| 입력 형태 | 코드 | 특징 |
|------|------|------|
| 컬럼 단위 dict | `pd.DataFrame({"a": [1, 2], "b": [3, 4]})` | 가장 흔함, 키가 컬럼명 |
| 행 단위 list of dict | `pd.DataFrame([{"a": 1, "b": 3}, {"a": 2, "b": 4}])` | JSON 레코드와 같은 모양 |
| 2차원 리스트/numpy | `pd.DataFrame(arr, columns=[...])` | 컬럼명은 따로 지정 |
| Series 여러 개 | `pd.concat([s1, s2], axis=1)` | 2회차 |
""")
code(r"""
# 컬럼 단위 dict (열 이름 -> 값 리스트)
df1 = pd.DataFrame({
  "name": ["김철수", "이영희", "박민수"],
  "score": [85, 92, 78],
  "passed": [True, True, False],
})
print(df1, "\n")

# 행 단위 list of dict
df2 = pd.DataFrame([
  {"name": "김철수", "score": 85},
  {"name": "이영희", "score": 92},
])
print(df2, "\n")

# numpy 배열 + 컬럼명
arr = np.arange(12).reshape(4, 3)
df3 = pd.DataFrame(arr, columns=["x", "y", "z"], index=["r1", "r2", "r3", "r4"])
print(df3)
""")
code(r"""
# 날짜 인덱스 만들기: pd.date_range (시계열 데이터 생성/검증에 자주 사용)
ts = pd.DataFrame({
  "date": pd.date_range("2024-01-01", periods=7, freq="D"),
  "value": np.round(np.random.default_rng(0).normal(100, 10, 7), 1),
})
ts
""")

md(r"""
### 2.6 데이터 저장하기: `to_csv()`, `to_excel()`, `to_json()`

#### 한 줄 정의
DataFrame 을 파일로 내보내는 메서드. **`index=False` 를 빼먹는 것이 가장 흔한 실수** 입니다.

| 메서드 | 핵심 옵션 | 메모 |
|------|------|------|
| `df.to_csv(path, index=False)` | `index`, `encoding`, `sep`, `na_rep` | `encoding="utf-8-sig"` 로 저장하면 윈도우 엑셀에서 한글이 안 깨짐 |
| `df.to_excel(path, index=False, sheet_name=...)` | `sheet_name`, `ExcelWriter` 로 다중 시트 | openpyxl 필요 |
| `df.to_json(path, orient="records", force_ascii=False)` | `orient`, `force_ascii`, `indent` | 한글은 `force_ascii=False` |
""")
code(r"""
# index=False 의 중요성: 빼먹으면 저장 때마다 "Unnamed: 0" 컬럼이 늘어난다.
sample = df1.copy()

sample.to_csv(f"{DATA_DIR}/tmp_with_index.csv")                 # index 저장 (기본값)
sample.to_csv(f"{DATA_DIR}/tmp_without_index.csv", index=False)  # index 미저장 (권장)

print("index 저장   :", pd.read_csv(f"{DATA_DIR}/tmp_with_index.csv").columns.tolist())
print("index=False  :", pd.read_csv(f"{DATA_DIR}/tmp_without_index.csv").columns.tolist())
""")
code(r"""
# 윈도우 엑셀 호환 한글 CSV: utf-8-sig (BOM 포함)
sample.to_csv(f"{DATA_DIR}/tmp_excel_friendly.csv", index=False, encoding="utf-8-sig")

# 저장 후 다시 읽어 검증하는 습관 (round-trip check)
back = pd.read_csv(f"{DATA_DIR}/tmp_excel_friendly.csv")
print("원본과 동일한가?", sample.equals(back))
""")
code(r"""
# Excel / JSON 저장
sample.to_excel(f"{DATA_DIR}/tmp_sample.xlsx", index=False, sheet_name="scores")
sample.to_json(f"{DATA_DIR}/tmp_sample.json", orient="records", force_ascii=False, indent=2)

with open(f"{DATA_DIR}/tmp_sample.json", encoding="utf-8") as f:
  print(f.read())
""")
code(r"""
# 임시 파일 정리 (tmp_ 로 시작하는 실습 파일만 삭제)
for name in os.listdir(DATA_DIR):
  if name.startswith("tmp_"):
    os.remove(f"{DATA_DIR}/{name}")
print("남은 파일:", sorted(os.listdir(DATA_DIR)))
""")

md(r"""
### 2.7 파일 읽기에서 자주 나는 오류와 해결법

| 증상 | 원인 | 해결 |
|------|------|------|
| `FileNotFoundError` | 경로가 틀림, 현재 작업 폴더가 다름 | `os.getcwd()`, `os.listdir()` 로 위치 확인 후 상대/절대 경로 수정 |
| `UnicodeDecodeError` | 인코딩 불일치 | `encoding="cp949"` 또는 `"euc-kr"`, `"latin1"` 시도 |
| 컬럼이 1개로 뭉침 | 구분자가 쉼표가 아님 | `sep="\t"`, `sep=";"`, 원본 텍스트를 열어 확인 |
| `Unnamed: 0` 컬럼 생김 | 저장 시 `index=False` 누락 | 저장을 고치거나 `index_col=0` 으로 읽기 |
| 숫자 컬럼이 `object` | `"-"`, `"?"`, `","` 등이 섞임 | `na_values=[...]`, `thousands=","`, `pd.to_numeric(errors="coerce")` |
| 우편번호 앞 0 사라짐 | 자동 정수 변환 | `dtype={"col": str}` |
| `ParserError: Error tokenizing data` | 행마다 컬럼 수가 다름 | `on_bad_lines="skip"`, 원본 확인 |
""")
code(r"""
# FileNotFoundError 진단 절차: 어디에 있는지, 무엇이 있는지부터 본다.
try:
  pd.read_csv("customers.csv")     # data/ 를 빠뜨렸다!
except FileNotFoundError as e:
  print("에러:", e)
  print("\n현재 폴더:", os.getcwd())
  print("현재 폴더의 csv:", [f for f in os.listdir(".") if f.endswith(".csv")])
  print("data 폴더의 csv:", [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
""")
code(r"""
# 인코딩을 모를 때: 후보를 차례로 시도하는 작은 도우미 함수
def read_csv_any_encoding(path: str, **kwargs) -> pd.DataFrame:
  for enc in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]:
    try:
      result = pd.read_csv(path, encoding=enc, **kwargs)
      print(f"성공한 인코딩: {enc}")
      return result
    except UnicodeDecodeError:
      continue
  raise ValueError(f"지원하는 인코딩으로 읽을 수 없음: {path}")


df_auto = read_csv_any_encoding(f"{DATA_DIR}/customers_cp949.csv")
df_auto.head(2)
""")
code(r"""
# 숫자에 쉼표(천 단위)가 섞인 파일: thousands 옵션
with open(f"{DATA_DIR}/tmp_thousands.csv", "w", encoding="utf-8") as f:
  f.write('item,price\n노트북,"1,250,000"\n마우스,"35,000"\n')

bad = pd.read_csv(f"{DATA_DIR}/tmp_thousands.csv")
good = pd.read_csv(f"{DATA_DIR}/tmp_thousands.csv", thousands=",")
print("thousands 미지정 dtype:", bad["price"].dtype)
print("thousands=',' dtype  :", good["price"].dtype, "| 합계:", good["price"].sum())
os.remove(f"{DATA_DIR}/tmp_thousands.csv")
""")

md(r"""
### 📝 시험 출제 포인트 (2장)

- **1번 문항은 거의 항상 `pd.read_csv()`** 입니다. "`data.csv` 파일을 읽어 `df` 변수에 저장하시오" 형태이며, 지정된 **변수명을 정확히** 써야 이후 문항이 채점됩니다.
- 시험 파일은 대부분 UTF-8 이지만, 한글 깨짐이 보이면 `encoding="cp949"` 를 시도합니다.
- `sep` 옵션 문항: "탭으로 구분된 파일" → `sep="\t"`.
- 내장 데이터셋 문항: `load_iris()` 등을 **DataFrame 으로 변환** 하는 코드가 나올 수 있습니다.
- 저장 문항: "`result.csv` 로 저장하되 인덱스는 제외" → `df.to_csv("result.csv", index=False)`.

### ⚠️ 자주 하는 실수 (2장)

- **`index=False` 누락**: 저장 파일에 `Unnamed: 0` 이 생깁니다. 저장은 항상 `index=False` 를 기본으로 생각하세요.
- **경로 문자열의 백슬래시**: 윈도우 경로 `"C:\new\data.csv"` 는 `\n` 이 줄바꿈으로 해석됩니다. `r"C:\new\data.csv"` 또는 `/` 를 씁니다.
- **읽고 나서 확인 안 함**: `read_csv` 는 에러 없이 **잘못** 읽을 수 있습니다 (컬럼 1개로 뭉침, 앞자리 0 소실). 읽은 직후 `df.head()`, `df.shape`, `df.dtypes` 를 보는 것을 습관화하세요.
- **`sheet_name` 미지정**: `read_excel` 은 기본으로 **첫 번째 시트만** 읽습니다.
- **DataFrame 을 `print()` 로만 확인**: Jupyter 에서는 셀 마지막 줄에 변수만 두거나 `display(df)` 가 표 형태로 보여 줍니다.
""")

# ---------------------------------------------------------------- 3. 종합 실습
md(r"""
---
## 3. 종합 실습

각 문제의 **빈 코드 셀** 에 직접 작성해 보세요. 정답은 각 문제 아래 접힌 영역에 있습니다.  
AICE 시험과 같은 형식으로 **변수명을 지정** 합니다.

### 문제 1. 세미콜론 구분 파일 읽기

`data/customers_semicolon.txt` 파일을 읽어 `q1` 변수에 저장하고, 행과 열의 개수를 출력하시오.

> **조건**: 구분자는 `;` 이다. `postal_code` 컬럼은 문자열로 읽는다.
""")
code(r"""
# 여기에 코드를 작성하세요
q1 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
q1 = pd.read_csv(f"{DATA_DIR}/customers_semicolon.txt", sep=";", dtype={"postal_code": str})
print(q1.shape)
```

</details>

### 문제 2. 일부 컬럼만 읽기

`data/customers.csv` 에서 `customer_id`, `plan`, `monthly_fee`, `churn` 네 컬럼만 읽고, `customer_id` 를 인덱스로 설정하여 `q2` 에 저장하시오. 상위 5개 행을 출력하시오.
""")
code(r"""
# 여기에 코드를 작성하세요
q2 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
q2 = pd.read_csv(
  f"{DATA_DIR}/customers.csv",
  usecols=["customer_id", "plan", "monthly_fee", "churn"],
  index_col="customer_id",
)
q2.head()
```

</details>

### 문제 3. Excel 두 번째 시트 읽기

`data/customers.xlsx` 의 `plans` 시트를 읽어 `q3` 에 저장하고 출력하시오.
""")
code(r"""
# 여기에 코드를 작성하세요
q3 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
q3 = pd.read_excel(f"{DATA_DIR}/customers.xlsx", sheet_name="plans")
q3
```

</details>

### 문제 4. 내장 데이터셋 → DataFrame

scikit-learn 의 `load_diabetes()` 를 불러와 특징 컬럼과 `target` 컬럼을 가진 DataFrame `q4` 를 만드시오. `q4.shape` 와 `q4.head(3)` 를 출력하시오.

> **힌트**: `load_diabetes()` 의 `.data`, `.feature_names`, `.target` 을 사용한다.
""")
code(r"""
# 여기에 코드를 작성하세요
q4 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
q4 = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
q4["target"] = diabetes.target
print(q4.shape)
q4.head(3)
```

</details>

### 문제 5. 날짜 파싱 후 저장

`data/sales_2024.csv` 를 `date` 컬럼을 날짜형으로 읽어 `q5` 에 저장하시오.  
그리고 `date` 가 2024년 3월인 행만 골라 `data/sales_2024_03.csv` 로 **인덱스 없이** 저장한 뒤, 저장한 파일을 다시 읽어 행 수를 출력하시오.

> **힌트**: 날짜형 컬럼은 `q5["date"].dt.month` 로 월을 꺼낼 수 있다. 조건 필터링 `df[조건]` 은 2회차에서 자세히 다루지만, 정답 코드를 보고 따라 해 보자.
""")
code(r"""
# 여기에 코드를 작성하세요
q5 = None
""")
md(r"""
<details>
<summary>정답 보기</summary>

```python
q5 = pd.read_csv(f"{DATA_DIR}/sales_2024.csv", parse_dates=["date"])
march = q5[q5["date"].dt.month == 3]
march.to_csv(f"{DATA_DIR}/sales_2024_03.csv", index=False)

check = pd.read_csv(f"{DATA_DIR}/sales_2024_03.csv")
print("3월 데이터 행 수:", len(check))   # 31일 × 3개 매장 = 93
```

</details>

### 문제 6 (도전). 문제 지문 읽고 회귀 / 분류 판단하기

아래 각 상황이 **회귀** 인지 **분류** 인지 적고, 예측 대상(y) 컬럼의 자료형이 무엇일지 말해 보시오. (코드 없이 마크다운 셀에 답을 적으세요.)

1. 고객의 나이·요금제·사용량으로 **다음 달 데이터 사용량(GB)** 을 예측한다.
2. 고객의 나이·요금제·사용량으로 **이탈 여부(Yes/No)** 를 예측한다.
3. 아파트 면적·층·연식으로 **매매가** 를 예측한다.
4. 이메일 본문으로 **스팸 / 정상 / 광고** 를 구분한다.
5. 환자 정보로 **당뇨 진행 정도(수치)** 를 예측한다.
""")
md(r"""
_(여기에 답을 적어 보세요)_

1.  
2.  
3.  
4.  
5.  
""")
md(r"""
<details>
<summary>정답 보기</summary>

| 번호 | 정답 | 이유 |
|:---:|------|------|
| 1 | 회귀 | GB 는 연속 숫자 (`float`) |
| 2 | 분류 (이진) | Yes/No 두 범주 (`object` 또는 0/1) |
| 3 | 회귀 | 가격은 연속 숫자 |
| 4 | 분류 (다중) | 3개 범주 |
| 5 | 회귀 | 진행 정도가 연속 수치 (`load_diabetes` 의 target) |

</details>
""")

# ---------------------------------------------------------------- 4. 정리
md(r"""
---
## 4. 오늘의 정리

### 핵심 요약

| 주제 | 기억할 것 |
|------|-----------|
| AI ⊃ ML ⊃ DL | ML 은 데이터에서 규칙을 찾고, DL 은 그 방법으로 깊은 신경망을 쓴다 |
| 지도 / 비지도 / 강화 | 정답이 있으면 지도학습. 회귀(숫자) vs 분류(범주)는 **y 의 형태** 로 판단 |
| 모델링 흐름 | 획득 → 구조확인 → 탐색 → 전처리 → 학습(`fit`) → 예측(`predict`) → 평가 → 개선 |
| `read_csv` | `sep`, `encoding`, `header/names`, `index_col`, `usecols`, `na_values`, `dtype`, `parse_dates` |
| 기타 읽기 | `read_excel(sheet_name=)`, `read_json`, URL, `load_iris()` → DataFrame 변환 |
| 저장 | `to_csv(index=False)` 를 기본으로, 한글 엑셀용은 `encoding="utf-8-sig"` |
| 습관 | 읽은 직후 `head()`, `shape`, `dtypes` 확인. 문제가 생기면 원본 텍스트를 연다 |

### 자기 점검 체크리스트

- [ ] 머신러닝과 전통 프로그래밍의 차이를 입력·출력 관점에서 설명할 수 있다.
- [ ] 문제 지문을 읽고 회귀인지 분류인지 3초 안에 판단할 수 있다.
- [ ] 구분자·인코딩이 다른 CSV 를 옵션을 바꿔 읽을 수 있다.
- [ ] `header=None` + `names=` 조합을 쓸 수 있다.
- [ ] Excel 의 특정 시트를 읽을 수 있다.
- [ ] scikit-learn 내장 데이터셋을 DataFrame 으로 바꿀 수 있다.
- [ ] `index=False` 로 저장하고, 다시 읽어 검증할 수 있다.

### 다음 회차 예고 — 2회차: 데이터 구조 확인하기, 기초 데이터 다루기

- `info()`, `describe()`, `value_counts()` 로 데이터의 "건강 상태" 진단하기
- `loc` / `iloc` 인덱싱, 조건 필터링, 정렬, 컬럼 추가·삭제·이름 변경
- `groupby`, `pivot_table`, `merge`, `concat` 으로 데이터 요약·결합하기
- 오늘 만든 `data/customers.csv`, `data/sales_2024.csv` 를 계속 사용합니다. **파일을 지우지 마세요.**
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("written:", OUT, "cells:", len(cells))
