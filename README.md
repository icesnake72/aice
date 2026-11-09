# AICE Associate 시험 대비 교육 자료

AICE(AI Certificate for Everyone) Associate 시험 대비용 강의·실습·모의시험 노트북 모음입니다.
GK2A 위성영상 예측 프로젝트는 2026-11-09에 `../gk2a_nowcast/` 로 분리했습니다.

## 폴더 구조

| 폴더 | 내용 |
| --- | --- |
| `course/` | 정규 8회차 실습 과정 (01~08회차 노트북, `data/`, 노트북 빌더 `tools/`, `ml_models_easy_guide.md`) |
| `lectures/` | 주제별 강의·실습 노트북 22개. 번호 순서가 학습 순서 (pandas → 전처리 → 회귀 → 분류 → 딥러닝) |
| `exams/` | AICE Associate 공식 샘플문항 2개, 모의시험 2개 |
| `practice/` | 자유 실습 노트북 |
| `reference/` | 보조 자료 (ML/DL 함수 모음, TensorFlow history 요약, 강의 이미지) |
| `data/` | `lectures/`, `exams/`, `practice/` 노트북이 공유하는 CSV 데이터. 노트북에서는 `../data/<파일>` 로 참조 |

`course/` 노트북은 자체 `course/data/` 를 `DATA_DIR = "data"` 로 참조하므로 `course/` 를 작업 디렉터리로 열어야 합니다.

## 실행 환경

- Python 3.11 (`/usr/local/bin/python3`) + TensorFlow 2.15 + tensorflow-metal, 또는 `~/.venv/ml` (Python 3.13, TF 2.21 CPU)
- Jupyter 에서 노트북이 있는 폴더를 기준으로 상대경로가 동작합니다

## lectures/ 학습 순서

| 번호 | 노트북 | 데이터 |
| --- | --- | --- |
| 01 | pandas 데이터 조작 | computer_prices_all, 안동시 상하수도, 국세청 사업자현황 |
| 02 | pandas groupby | 안동시 상하수도 |
| 03 | 시각화 (matplotlib, seaborn) | 여러 데이터 |
| 04~08 | 결측치 · 이상치 · 구간화 · 범주형 정제 · 스케일링 | california_housing_test, computer_prices_all |
| 09 | 선형회귀 강의 | california_housing_test, computer_prices_all |
| 10 | 선형회귀 실습 (ML + DL) | computer_prices_all |
| 11 | 로지스틱회귀 기초 (시그모이드, log loss) | 없음 |
| 12 | 로지스틱회귀 강의 | Student_Performance, heart, mobile_price |
| 13 | 로지스틱회귀 실습 | titanic_train |
| 14 | 트리 모델 (DT · RF · GB) 실습 | titanic_train |
| 15 | 지도학습 3교시 종합 수업 | titanic_train |
| 16~20 | 딥러닝 실습 (타이타닉, 당뇨병, 컴퓨터가격, 타이타닉2, 캘리포니아주택) | 각 CSV |
| 21~22 | ML + DL 종합 (주택가격, 심부전) | house_prices, heart |

## 2026-11-09 정리 전 파일명 대응표

| 이전 (루트) | 현재 |
| --- | --- |
| `pandas_데이터.ipynb` | `lectures/01_pandas_데이터조작.ipynb` |
| `groupby_pandas.ipynb` | `lectures/02_pandas_groupby.ipynb` |
| `01_시각화.ipynb` | `lectures/03_시각화_matplotlib_seaborn.ipynb` |
| `02_결측치.ipynb` ~ `06_스케일링.ipynb` | `lectures/04_결측치_처리.ipynb` ~ `lectures/08_스케일링.ipynb` |
| `10_linear_regression_lecture.ipynb` | `lectures/09_선형회귀_강의.ipynb` |
| `07_선형회귀.ipynb` | `lectures/10_선형회귀_실습_컴퓨터가격.ipynb` |
| `11_logistic_기초.ipynb` | `lectures/11_로지스틱회귀_기초_시그모이드.ipynb` |
| `08_logistic_regression_lecture.ipynb` | `lectures/12_로지스틱회귀_강의.ipynb` |
| `08_로지스틱회귀.ipynb` | `lectures/13_로지스틱회귀_실습_타이타닉.ipynb` |
| `09_tree_models.ipynb` | `lectures/14_트리모델_실습_타이타닉.ipynb` |
| `지도학습.ipynb` | `lectures/15_지도학습_3교시_종합수업.ipynb` |
| `ex04_deeplearning.ipynb` ~ `ex04_deeplearning4.ipynb` | `lectures/16_딥러닝_이진분류_타이타닉.ipynb` ~ `lectures/19_딥러닝_이진분류_타이타닉2.ipynb` |
| `ex05_deep_learning.ipynb` | `lectures/20_딥러닝_회귀_캘리포니아주택.ipynb` |
| `ex04_AI.ipynb` | `lectures/21_ML_DL종합_주택가격.ipynb` |
| `ex05_ai_total_test.ipynb` | `lectures/22_ML_DL종합_심부전예측.ipynb` |
| `AICE Associaet 샘플문항_회귀.ipynb` | `exams/AICE_Associate_샘플문항_회귀.ipynb` (`signal_data.csv` 는 시험 제공 데이터로 저장소에 없음) |
| `Asso_샘플문항_분류_정답_v1.ipynb` | `exams/AICE_Associate_샘플문항_분류_정답.ipynb` |
| `exam01.ipynb`, `exam02.ipynb` | `exams/모의시험01_자동차데이터_ML.ipynb`, `exams/모의시험02_모바일가격분류.ipynb` |
| `california_housing_analysis.ipynb` | `practice/캘리포니아주택_이상치분석.ipynb` |
| `10_flight_price_lr.ipynb` | `practice/항공권가격_선형회귀.ipynb` |
| `ex01.ipynb` | `practice/컴퓨터가격_EDA_이상치.ipynb` |
| `ML_DL에 사용하는 함수들.ipynb` | `reference/ML_DL_함수모음.ipynb` |
| `resource/` | `reference/` |
| `pandas/data/` | `data/` |
| `aice_course/` | `course/` |

백업으로 이동한 파일(`../aice_backup_20261109/`): `exam01_q.ipynb`(exam01 과 동일), `09_decision_tree.ipynb`(14번에 통합), `test03.ipynb`, `test05.ipynb`(03 시각화 초안), `titanic.ipynb`, `Untitled.ipynb`, `unet_synthetic_lecture.ipynb`(AICE 범위 외), `Equipping agents ... .md`
