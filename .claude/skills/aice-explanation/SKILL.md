---
name: aice-explanation
description: |
  AICE Associate 시험 대비 개념 설명, 이론 교안, 함수 사용법 정리 생성 스킬.
  머신러닝·딥러닝·데이터분석 개념을 비전공자 기준으로 직관적으로 설명하되
  반드시 "시험에서는 이렇게 나온다" 섹션을 포함한다.
  "설명해줘", "교안 만들어", "개념 정리", "이론 정리", "왜 쓰는지",
  "차이점", "언제 쓰는지", "함수 설명", "어떻게 동작해", "정리해줘" 키워드 +
  AICE/머신러닝/데이터분석/pandas/sklearn/keras 맥락에서 반드시 참조.
---

# AICE Associate 개념 설명 스킬

## 설명 원칙

- **비전공자 기준**: 수식보다 직관적 비유 우선
- **실용 중심**: "왜 쓰는가"를 먼저, 이론은 그 다음
- **코드 증명**: 모든 개념은 5~15줄 실제 코드로 확인
- **시험 직결**: 설명 후 반드시 "📝 시험 출제 포인트" 섹션 추가
- **오해 방지**: "⚠️ 자주 하는 실수" 섹션으로 오답 유형 예방

## 출력 구조 (항상 이 순서로)

```markdown
## {개념명}

### 한 줄 정의
{가장 단순한 정의 1문장}

### 직관적 설명
{비유 또는 일상 언어 설명. 수식 최소화}

### 코드 예시
```python
{실제 동작 확인 가능한 코드 5~15줄}
```

### 📝 시험 출제 포인트
- {시험에서 이 개념이 나오는 방식}
- {자주 요구하는 파라미터}
- {출력 형태}

### ⚠️ 자주 하는 실수
- {오답 패턴 1}: {왜 틀리는지}
- {오답 패턴 2}: {올바른 방법}

### 관련 개념
- {연결되는 다음 개념}
```

## 영역별 핵심 개념 목록

### 데이터 분석 (pandas/numpy)
- DataFrame, Series 구조
- 인덱싱 (.loc, .iloc)
- groupby, pivot_table
- merge, concat
- apply, lambda

### 전처리
- 결측치: isnull, fillna, dropna
- 인코딩: get_dummies, LabelEncoder
- 스케일링: MinMaxScaler vs StandardScaler (차이점)
- train_test_split: stratify 파라미터 의미

### 머신러닝 개념
- 과적합(Overfitting) vs 과소적합(Underfitting)
- 편향-분산 트레이드오프
- 교차검증(Cross-Validation)
- 앙상블: 배깅 vs 부스팅

### 평가 지표
- 분류: Accuracy vs Precision vs Recall vs F1
- 혼동행렬 해석 (TP, FP, FN, TN)
- ROC-AUC 의미
- 회귀: MAE vs MSE vs RMSE vs R²

### 딥러닝
- 활성화 함수: relu, sigmoid, softmax 언제 쓰는지
- 손실 함수: binary_crossentropy vs categorical_crossentropy vs mse
- Dropout 역할
- 학습 곡선으로 과적합 확인하는 방법

## 비교 설명 요청 시 포맷

두 개 이상 개념 비교 요청 시 표 형식 추가:

```markdown
| 항목 | {개념A} | {개념B} |
|------|---------|---------|
| 용도 | ... | ... |
| 특징 | ... | ... |
| 시험에서 | ... | ... |
| 코드 | ... | ... |
```

## 난이도 조정

- "쉽게", "초보자용", "비전공자" 키워드 → 비유·그림 중심, 코드 최소화
- "심화", "자세히", "전공자" 키워드 → 수학적 배경, 파라미터 상세 설명
- 기본값: 준전공자 수준 (코드+직관 균형)
