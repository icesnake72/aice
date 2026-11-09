# 머신러닝 모델링 출제 패턴

AICE Associate 머신러닝/딥러닝 모델링 영역 출제 유형 목록.

---

## 지도학습 — 분류 (Classification)

### 1. Logistic Regression (난이도: 하)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
```

---

### 2. Decision Tree (난이도: 하~중) ★★

```python
from sklearn.tree import DecisionTreeClassifier   # 분류
from sklearn.tree import DecisionTreeRegressor    # 회귀

model = DecisionTreeClassifier(
    max_depth=5,            # 시험에서 자주 지정
    min_samples_split=3,    # 시험에서 자주 지정
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
```

---

### 3. Random Forest (난이도: 중) ★★★ 최빈출

```python
from sklearn.ensemble import RandomForestClassifier   # 분류
from sklearn.ensemble import RandomForestRegressor    # 회귀

model = RandomForestClassifier(
    n_estimators=100,   # 트리 수, 시험에서 자주 지정
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
```

---

### 4. Gradient Boosting / XGBoost (난이도: 중~상)

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
```

---

### 5. K-Nearest Neighbors (난이도: 중)

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
```

---

### 6. SVM (난이도: 중)

```python
from sklearn.svm import SVC   # 분류
from sklearn.svm import SVR   # 회귀

model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)
```

---

## 지도학습 — 회귀 (Regression)

### Linear Regression (난이도: 하)

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
```

---

## 비지도학습

### K-Means 클러스터링 (난이도: 중)

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
labels = kmeans.labels_
```

### PCA 차원 축소 (난이도: 중~상)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)
```

---

## 딥러닝 (Keras) (난이도: 중~상)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # 이진 분류
    # Dense(n, activation='softmax')  # 다중 분류
    # Dense(1)                         # 회귀
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',   # 이진 분류
    # loss='categorical_crossentropy',  # 다중 분류
    # loss='mse',                       # 회귀
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    verbose=1
)
```

---

## 모델링 문항 출제 시 고정 파라미터 (시험에서 자주 지정)

| 파라미터 | 자주 쓰이는 값 |
|----------|--------------|
| random_state | 42, 120 |
| test_size | 0.2 |
| max_depth | 3, 5 |
| n_estimators | 100 |
| n_neighbors | 5 |
| epochs | 50, 100 |
| batch_size | 32 |

---

## 모델링 문항 연결 흐름

```
전처리 결과 (X_train, X_valid, y_train, y_valid)
        ↓
모델 생성 (model = ...)
        ↓
모델 학습 (model.fit(X_train, y_train))
        ↓
예측     (y_pred = model.predict(X_valid))
        ↓
성능 평가 (다음 영역으로 연결)
```
