# TensorFlow `model.fit()` 반환값 `history` 요약

## 1. `history` 객체란?

`model.fit()` 호출 시 반환되는 `keras.callbacks.History` 객체로, **에포크(epoch)마다 기록된 손실과 평가지표**를 담고 있습니다.

```python
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=50, batch_size=32)
```

## 2. 주요 속성

| 속성 | 설명 |
|---|---|
| `history.history` | 에포크별 loss/metrics 값 dict (**핵심**) |
| `history.epoch` | 학습된 에포크 번호 리스트 |
| `history.params` | batch_size, epochs, steps 등 학습 파라미터 |
| `history.model` | 학습에 사용된 모델 참조 |

## 3. `history.history` 구조

`compile()`에 지정한 loss/metrics가 키가 되며, 검증 지표는 **`val_` 접두사**가 붙습니다.

```python
{
    'loss':         [0.69, 0.55, 0.42, ...],   # 훈련 손실
    'accuracy':     [0.55, 0.68, 0.78, ...],   # 훈련 정확도
    'val_loss':     [0.70, 0.58, 0.48, ...],   # 검증 손실
    'val_accuracy': [0.54, 0.66, 0.74, ...]    # 검증 정확도
}
```

> 리스트 길이 = 실제 학습된 에포크 수 (EarlyStopping 시 더 짧아짐)

## 4. 활용 방법

### 4-1. 학습 곡선 시각화

```python
import matplotlib.pyplot as plt

hist = history.history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(hist['loss'], label='train_loss')
axes[0].plot(hist['val_loss'], label='val_loss')
axes[0].set_title('Loss'); axes[0].legend()

axes[1].plot(hist['accuracy'], label='train_acc')
axes[1].plot(hist['val_accuracy'], label='val_acc')
axes[1].set_title('Accuracy'); axes[1].legend()

plt.show()
```

### 4-2. 과적합/과소적합 진단

- **정상**: train/val loss가 함께 감소 후 수렴
- **과적합**: train_loss는 ↓, val_loss는 ↑ (gap 벌어짐)
- **과소적합**: 둘 다 높은 값에서 정체

### 4-3. 최적 에포크 찾기

```python
import numpy as np
best_epoch = np.argmin(hist['val_loss'])
print(f"최적 에포크: {best_epoch+1}")
```

### 4-4. 실험 로그 저장 및 비교

```python
import pandas as pd
df = pd.DataFrame(history.history)
df.to_csv('train_log.csv', index=False)
```

### 4-5. 그 외 활용
- 콜백(`ReduceLROnPlateau` 등) 효과 검증
- 여러 모델 후보의 학습 안정성 비교
- EarlyStopping `patience` 설정 근거

## 5. AICE 시험 핵심 포인트

1. `history.history`는 **dict**, 키는 compile 시 지정한 loss/metrics
2. 검증 지표는 **`val_` 접두사**
3. 학습 곡선으로 **과적합/과소적합 판단**
4. `np.argmin(history.history['val_loss'])`로 최적 에포크 탐색
5. EarlyStopping 시 history 길이 < epochs

## 6. 실전 팁

- 회귀 문제: 키가 `mae`, `mse` 등으로 달라짐
- 다중 출력: 출력 레이어 이름이 키에 포함됨 (`output_1_loss`)
- `pd.DataFrame(history.history)`로 변환하면 분석/저장이 편리

---

**요약**: `history`는 학습 과정의 기록장입니다. 모델 가중치는 `model.save()`로 저장하지만, **학습 진단·디버깅·실험 비교의 모든 근거**가 이 객체에 들어 있습니다.
