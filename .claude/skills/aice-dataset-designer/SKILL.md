---
name: aice-dataset-designer
description: |
  AICE Associate 시험 실습 및 문항 출제용 CSV 데이터셋을 설계하고
  pandas 코드로 샘플 데이터를 생성하는 스킬.
  "데이터셋 만들어", "실습 데이터 필요해", "CSV 구조 설계", "샘플 데이터 생성",
  "어떤 데이터로 연습하면 좋을지", "타이타닉 말고 다른 데이터",
  "문항용 데이터", "연습용 데이터프레임" 키워드에 반드시 참조.
  결측치·이상치·범주형 컬럼을 내장한 현실감 있는 데이터 생성.
---

# AICE Associate 실습 데이터셋 설계 스킬

## 데이터셋 설계 원칙

- **현실 시나리오**: 실제 비즈니스 맥락 반영 (통신·금융·의료·물류·커머스)
- **전처리 포인트 내장**: 결측치 15~25%, 이상치 2~5%, object 컬럼 3개 이상
- **적정 규모**: 행 500~2000개, 컬럼 10~15개 (90분 내 처리 가능)
- **타겟 명확화**: 분류 → binary(0/1) or 다중, 회귀 → 연속형 수치
- **재현성**: `random_state=42` 고정으로 항상 같은 데이터 생성

## 출력 구조 (항상 이 순서로)

### 1. 데이터셋 시나리오
```
- 배경: {비즈니스 상황 1~2문장}
- 목표: {예측/분류/분석 목표}
- 타겟 컬럼: {컬럼명} ({타입: binary/multiclass/regression})
```

### 2. 컬럼 정의 테이블
| 컬럼명 | 타입 | 설명 | 결측치 | 이상치 |
|--------|------|------|--------|--------|
| ... | int/float/object | ... | 있음/없음 | 있음/없음 |

### 3. 데이터 생성 코드 (50~100행 샘플)
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000  # 행 수

df = pd.DataFrame({
    ...
})

# 결측치 삽입
# 이상치 삽입

df.to_csv('{파일명}.csv', index=False)
print(df.shape)
print(df.head())
```

### 4. 이 데이터로 출제 가능한 문항 목록
```
EDA:
  - {문항 예시}
전처리:
  - {문항 예시}
모델링:
  - {문항 예시}
평가:
  - {문항 예시}
```

---

## 추천 시나리오 목록 (요청 없으면 이 중 선택)

### A. 통신 고객 이탈 예측 (분류) ★ 가장 빈출
```
컬럼: customer_id, age, gender, contract_type, monthly_charge,
       total_charges, tenure_months, num_complaints, internet_service,
       phone_service, payment_method, churn (0/1)
특징: contract_type/gender/payment_method → 원-핫 인코딩 대상
      monthly_charge 이상치, total_charges 결측치 포함
```

### B. 의료 당뇨병 예측 (분류)
```
컬럼: patient_id, age, bmi, glucose, blood_pressure, insulin,
       skin_thickness, pregnancies, diabetes_pedigree, outcome (0/1)
특징: insulin/skin_thickness 결측치 多, bmi 이상치
```

### C. 주택 가격 예측 (회귀)
```
컬럼: house_id, area_sqm, num_rooms, num_bathrooms, floor,
       district, building_age, parking, school_nearby,
       renovation_year, price (타겟)
특징: renovation_year 결측치, price 이상치, district → 인코딩 대상
```

### D. 온라인 쇼핑몰 구매 예측 (분류)
```
컬럼: user_id, age, gender, visit_duration, page_views,
       bounce_rate, traffic_source, device_type, region,
       prev_purchases, purchase (0/1)
특징: visit_duration 이상치, prev_purchases 결측치
```

### E. 물류 배송 지연 예측 (분류)
```
컬럼: order_id, weight_kg, distance_km, carrier, warehouse_region,
       order_hour, is_weekend, fragile, priority_level,
       delivery_delay (0/1)
특징: weight_kg 이상치 (300kg+), carrier → 인코딩 대상
```

---

## 결측치·이상치 삽입 코드 패턴

```python
# 결측치 삽입 (약 15~20%)
def insert_missing(df, col, rate=0.15):
    idx = np.random.choice(df.index, size=int(len(df) * rate), replace=False)
    df.loc[idx, col] = np.nan
    return df

# 이상치 삽입 (상위 2~3%에 극단값)
def insert_outliers(df, col, factor=5):
    idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[idx, col] = df[col].mean() + factor * df[col].std()
    return df
```

---

## 시나리오 미지정 시 기본 동작

1. 아래 질문으로 방향 확인:
   - 분류/회귀 중 어느 쪽?
   - 어떤 산업/도메인 선호?
   - 문항 몇 개용 데이터가 필요한가? (모의고사 전체 vs 전처리 연습만)
2. 응답 없으면 → **A. 통신 고객 이탈 예측** 기본 사용
