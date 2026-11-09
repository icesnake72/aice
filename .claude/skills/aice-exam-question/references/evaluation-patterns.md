# 모델 성능 평가 출제 패턴

AICE Associate 모델 성능 평가 영역 출제 유형 목록.

---

## 분류(Classification) 평가 지표

### 1. 기본 평가 지표 (난이도: 하~중) ★★★ 최빈출

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 각 지표 개별 계산
acc  = accuracy_score(y_valid, y_pred)
prec = precision_score(y_valid, y_pred)          # 이진 분류 기본
rec  = recall_score(y_valid, y_pred)
f1   = f1_score(y_valid, y_pred)

# 다중 분류 시 average 파라미터 필요
prec = precision_score(y_valid, y_pred, average='macro')
f1   = f1_score(y_valid, y_pred, average='weighted')

# 한 번에 출력
print(classification_report(y_valid, y_pred))
```

**시험 함정**
- 이진 분류: `average` 파라미터 생략 가능 (default='binary')
- 다중 분류: `average='macro'` 또는 `'weighted'` 반드시 지정 → 없으면 오류

---

### 2. 혼동 행렬 (Confusion Matrix) (난이도: 중)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_valid, y_pred)
print(cm)

# 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

---

### 3. ROC-AUC (난이도: 중~상)

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# AUC 점수
auc = roc_auc_score(y_valid, y_pred_proba)  # predict_proba 결과 사용

# 확률 예측
y_pred_proba = model.predict_proba(X_valid)[:, 1]  # 양성 클래스 확률

# ROC 커브 시각화
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
```

---

## 회귀(Regression) 평가 지표

### 기본 회귀 지표 (난이도: 하~중) ★★

```python
from sklearn.metrics import (
    mean_absolute_error,     # MAE
    mean_squared_error,      # MSE
    r2_score                 # R²
)
import numpy as np

mae  = mean_absolute_error(y_valid, y_pred)
mse  = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)           # RMSE = MSE의 제곱근
r2   = r2_score(y_valid, y_pred)

print(f'MAE:  {mae:.4f}')
print(f'MSE:  {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²:   {r2:.4f}')
```

**각 지표 의미**
- MAE: 평균 절대 오차 (이상치에 덜 민감)
- MSE: 평균 제곱 오차 (이상치에 민감)
- RMSE: MSE의 제곱근 (원래 단위와 동일)
- R²: 설명력 (1에 가까울수록 좋음, 음수 가능)

---

## 딥러닝 학습 결과 평가

### 학습 곡선 시각화 (난이도: 중)

```python
import matplotlib.pyplot as plt

# model.fit() 결과가 history에 저장된 경우
plt.figure(figsize=(12, 4))

# 손실 곡선
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.title('Loss')
plt.legend()

# 정확도 곡선
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 모델 간 성능 비교 (난이도: 중~상)

```python
# 여러 모델 성능 비교 딕셔너리/DataFrame으로 정리
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    results[name] = {
        'accuracy': accuracy_score(y_valid, y_pred),
        'f1': f1_score(y_valid, y_pred, average='weighted')
    }

result_df = pd.DataFrame(results).T
print(result_df.sort_values('f1', ascending=False))
```

---

## 교차 검증 (난이도: 상)

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f'평균 정확도: {scores.mean():.4f}')
print(f'표준 편차: {scores.std():.4f}')
```

---

## 평가 문항에서 자주 나오는 조건 패턴

```
- y_valid와 y_pred를 사용하여 {지표명}를 계산하세요
- 결과를 변수 {변수명}에 저장하세요
- 소수점 4째 자리까지 출력하세요  → f'{값:.4f}'
- 분류 리포트 전체를 출력하세요   → classification_report()
- 과적합 여부를 학습 곡선으로 확인하세요 → train/val loss 비교
```
