# 데이터 전처리 출제 패턴

AICE Associate 데이터 전처리 영역 출제 유형 목록.
전처리는 전체 시험 문항의 30~40%를 차지하는 핵심 영역이다.

---

## 자주 출제되는 유형 (빈도 높은 순)

### 1. 결측치 처리 (난이도: 하~중) ★★★ 최빈출

**출제 포인트**
```python
# 결측치 확인
df.isnull().sum()

# 행 삭제
df_clean = df.dropna()
df_clean = df.dropna(subset=['특정컬럼'])  # 특정 컬럼 기준

# 평균/중앙값/최빈값 대체
df['컬럼'] = df['컬럼'].fillna(df['컬럼'].mean())
df['컬럼'] = df['컬럼'].fillna(df['컬럼'].median())
df['컬럼'] = df['컬럼'].fillna(df['컬럼'].mode()[0])  # mode()는 Series 반환

# 특정 값으로 대체
df['컬럼'] = df['컬럼'].fillna(0)
df['컬럼'] = df['컬럼'].fillna('Unknown')
```

**시험 함정**
- `mode()[0]` — mode()는 Series를 반환하므로 [0] 인덱싱 필수
- `inplace=True` vs 새 변수 저장 — 문항 조건 확인 필수
- 특정 컬럼만 처리할 때 `subset` 파라미터 활용

**문항 조건 예시 (중)**
```
- 대상 데이터프레임: df
- 결측치 비율이 30% 이상인 컬럼은 삭제
- 나머지 수치형 컬럼의 결측치는 해당 컬럼의 중앙값으로 대체
- object 타입 컬럼의 결측치는 최빈값으로 대체
- 결과를 변수 df_clean에 저장
```

**정답 코드 예시**
```python
# 30% 이상 결측치 컬럼 삭제
threshold = len(df) * 0.3
df_clean = df.dropna(thresh=threshold, axis=1)

# 수치형: 중앙값 대체
num_cols = df_clean.select_dtypes(include='number').columns
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# object형: 최빈값 대체
obj_cols = df_clean.select_dtypes(include='object').columns
for col in obj_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
```

---

### 2. 이상치 제거 (난이도: 중) ★★

**출제 포인트**
```python
# IQR 방식
Q1 = df['컬럼'].quantile(0.25)
Q3 = df['컬럼'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_clean = df[(df['컬럼'] >= lower) & (df['컬럼'] <= upper)]

# 직접 임계값 지정 (jointplot 확인 후)
df_clean = df[df['Speed'] < 300]
```

---

### 3. 원-핫 인코딩 (난이도: 중) ★★★ 최빈출

**출제 포인트**
```python
# 특정 컬럼 인코딩
df_encoded = pd.get_dummies(df, columns=['sex', 'region'])

# drop_first=True (가변수 함정 방지)
df_encoded = pd.get_dummies(df, columns=['sex', 'region'], drop_first=True)

# object 타입 전체 인코딩
obj_cols = df.select_dtypes('object').columns
df_encoded = pd.get_dummies(df, columns=obj_cols)

# dtype 명시
df_encoded = pd.get_dummies(df, columns=['sex'], dtype=int)
```

**시험 함정**
- `drop_first=True` 조건이 붙으면 반드시 포함
- 인코딩 후 변수명이 `df_preset`처럼 바뀌는 경우 주의
- `dtype=int`로 True/False 대신 1/0으로 출력 요구하는 경우 있음

---

### 4. 데이터 분할 (train_test_split) (난이도: 하~중) ★★★ 최빈출

**출제 포인트**
```python
from sklearn.model_selection import train_test_split

# 기본 분할
X = df.drop('target', axis=1)
y = df['target']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,      # 시험 기본값
    random_state=42,    # 시험 기본값
    stratify=y          # 분류 문제 시 추가
)
```

**시험에서 자주 지정하는 고정값**
- `test_size=0.2` (80:20 분할)
- `random_state=42`
- 분류 문제: `stratify=y` 추가 요구하는 경우 있음

---

### 5. 스케일링 (난이도: 중) ★★

**출제 포인트**
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()   # 또는 StandardScaler()

# 핵심: fit은 train에만, transform은 train+valid 양쪽
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)   # fit 없이 transform만
```

**시험 함정 ★★★**
- `fit_transform(X_valid)` 사용 → 데이터 누수 발생 → 오답
- 반드시 `scaler.fit(X_train)` 후 `scaler.transform()` 구분

---

### 6. 컬럼 삭제 & 타입 변환 (난이도: 하)

**출제 포인트**
```python
# 컬럼 삭제
df_del = df.drop(['id', 'name'], axis=1)
df_del = df.drop(columns=['id', 'name'])  # 동일

# 타입 변환
df['age'] = df['age'].astype(int)
df['price'] = df['price'].astype(float)
df = df.astype({'age': int, 'price': float})  # 여러 컬럼 동시

# object → category
df['grade'] = df['grade'].astype('category')
```

---

### 7. 컬럼 내용 변경 (난이도: 중)

**출제 포인트**
```python
# 특정 값 교체
df['gender'] = df['gender'].replace({'M': 0, 'F': 1})
df['gender'] = df['gender'].map({'M': 0, 'F': 1})

# 조건부 변경
df['age_group'] = df['age'].apply(lambda x: '청년' if x < 30 else '중년')

# 문자열 처리
df['name'] = df['name'].str.strip()
df['phone'] = df['phone'].str.replace('-', '')
```

---

## 전처리 문항 연결 순서 (시험 실제 흐름)

```
① 결측치 확인 (EDA) → df
② 불필요 컬럼 삭제  → df_del
③ 이상치 제거       → df_temp
④ 결측치 처리       → df_na
⑤ 원-핫 인코딩      → df_preset
⑥ train_test_split  → X_train, X_valid, y_train, y_valid
⑦ 스케일링          → X_train_scaled, X_valid_scaled
```

각 단계의 변수명이 다음 단계의 입력이 되도록 문항 구성 시 연결 필수.
