# EDA 출제 패턴

AICE Associate 탐색적 데이터 분석(EDA) 영역 출제 유형 목록.
문항 생성 시 아래 유형에서 선택하여 조건을 구성한다.

---

## 자주 출제되는 유형 (빈도 높은 순)

### 1. 데이터 로딩 및 기본 확인 (난이도: 하)

**출제 포인트**
- CSV 파일 읽기: `pd.read_csv('파일명.csv')`
- 상위 N행 확인: `df.head(N)`
- 데이터 크기 확인: `df.shape`
- 컬럼 목록 확인: `df.columns`
- 데이터 타입 확인: `df.dtypes` 또는 `df.info()`

**문항 조건 예시**
```
- 파일명: 'customer_data.csv'
- 데이터프레임 변수명: df
- 상위 5행 출력
- 데이터 크기(행, 열) 출력
- 결과를 변수 df에 저장
```

**정답 코드 예시**
```python
import pandas as pd

df = pd.read_csv('customer_data.csv')
print(df.head(5))
print(df.shape)
```

---

### 2. 기초 통계량 분석 (난이도: 하~중)

**출제 포인트**
- 수치형 통계: `df.describe()`
- 특정 컬럼 통계: `df['컬럼'].mean()`, `.median()`, `.std()`
- 최대/최솟값 인덱스: `df['컬럼'].idxmax()`, `.idxmin()`
- 범주형 빈도: `df['컬럼'].value_counts()`
- 고유값 수: `df['컬럼'].nunique()`

**문항 조건 예시**
```
- 대상 데이터프레임: df
- 수치형 컬럼 전체의 기초 통계량 출력
- 'age' 컬럼의 평균, 중앙값, 표준편차를 각각 출력
- 'gender' 컬럼의 고유값 종류와 빈도 출력
```

---

### 3. 결측치 확인 (난이도: 하)

**출제 포인트**
- 결측치 총계: `df.isnull().sum()`
- 결측치 비율: `df.isnull().mean() * 100`
- 결측치 있는 컬럼만: `df.isnull().sum()[df.isnull().sum() > 0]`
- 전체 결측치 수: `df.isnull().sum().sum()`

**문항 조건 예시**
```
- 대상 데이터프레임: df
- 컬럼별 결측치 수를 출력
- 결측치 비율(%)도 함께 출력
- 결과를 변수 null_info에 저장
```

**정답 코드 예시**
```python
null_info = pd.DataFrame({
    '결측치 수': df.isnull().sum(),
    '결측치 비율(%)': df.isnull().mean() * 100
})
print(null_info)
```

---

### 4. 데이터 시각화 (난이도: 중)

**출제 포인트**
- 히스토그램: `df['컬럼'].hist()` 또는 `plt.hist(df['컬럼'])`
- 박스플롯: `df.boxplot(column='컬럼')` 또는 `sns.boxplot()`
- 상관관계 히트맵: `sns.heatmap(df.corr(), annot=True)`
- 산점도: `plt.scatter(df['x'], df['y'])`
- 막대그래프: `df['컬럼'].value_counts().plot(kind='bar')`

**문항 조건 예시**
```
- 대상 데이터프레임: df
- 'income' 컬럼의 분포를 히스토그램으로 시각화
- 수치형 컬럼 간 상관관계를 히트맵으로 출력 (annot=True)
- matplotlib.pyplot as plt, seaborn as sns 사용
```

---

### 5. 이상치 시각화 확인 (난이도: 중)

**출제 포인트**
- 박스플롯으로 이상치 위치 확인
- IQR 계산으로 이상치 기준 산출
  ```python
  Q1 = df['컬럼'].quantile(0.25)
  Q3 = df['컬럼'].quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  ```
- jointplot (seaborn): `sns.jointplot(data=df, x='x컬럼', y='y컬럼')`

**시험 함정**: jointplot 그래프를 보고 이상치 임계값을 직접 눈으로 판단해서 코드 작성

---

### 6. 상관관계 분석 (난이도: 중)

**출제 포인트**
- 상관계수 행렬: `df.corr()`
- 특정 컬럼과의 상관관계: `df.corr()['타겟컬럼'].sort_values(ascending=False)`
- 피어슨 상관계수: 기본값
- 스피어만 상관계수: `df.corr(method='spearman')`

---

## EDA 문항 출제 시 주의사항

1. **변수명 일관성**: df → df_result 처럼 명확한 단계별 변수명 지정
2. **출력 명시**: `print()` 또는 `display()` 중 어느 것을 쓸지 조건에 명시
3. **라이브러리 지정**: 시각화 시 plt/sns 중 어느 것을 쓸지 조건에 포함
4. **저장 여부**: 분석 결과를 변수에 저장하는지 출력만 하는지 명확히 구분
