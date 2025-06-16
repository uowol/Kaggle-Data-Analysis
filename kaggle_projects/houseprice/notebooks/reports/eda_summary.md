
# 📊 EDA Summary

## ✅ 기본 정보
- 총 샘플 수: 1460개
- 총 컬럼 수: 81개
- 타겟 변수: `SalePrice` (연속형)
- 식별자 컬럼: `Id` (모델 학습에 불필요)

## 🧩 컬럼 타입
- 수치형(int/float): 38개
- 범주형(object): 43개

## 🚨 결측치 분석
- 총 19개 컬럼에 결측 존재
- 상위 결측 컬럼:
  - `PoolQC`: 1453개 → 수영장 없음 (MAR)
  - `MiscFeature`: 1406개 → 특이 시설 없음 (MAR)
  - `Alley`: 1369개 → 골목 없음 (MAR)
  - `Fence`: 1179개 → 울타리 없음 (MAR)
  - `FireplaceQu`: 690개 → 벽난로 없음 (MAR)
- `Electrical`: 1개 → 입력 누락 (MCAR)

## 📌 결측 유형 분류
- 대부분 **MAR (Missing At Random)** → 다른 변수로부터 예측 가능
- `Electrical`: **MCAR** → 최빈값 대체
- `LotFrontage`: 지역(Neighborhood) 기반 평균 대체 고려

## 🔎 향후 계획
- `SalePrice` 분포 시각화 및 로그 변환 여부 확인
- 수치형 변수 상관관계 분석
- 범주형 변수별 평균 가격 시각화

## 🧩 전체 결측 컬럼 분석 (결측 수 기준 정렬)

### 🔹 PoolQC (1453개 결측)
- **유형**: MAR
- 수영장이 없는 경우 결측 발생
- 관련 변수: `PoolArea` (거의 대부분 0)

### 🔹 MiscFeature (1406개 결측)
- **유형**: MAR
- 특이 시설(테니스코트, 엘리베이터 등)이 없는 경우 결측

### 🔹 Alley (1369개 결측)
- **유형**: MAR
- 골목이 존재하지 않는 경우 결측

### 🔹 Fence (1179개 결측)
- **유형**: MAR
- 울타리가 없는 경우 결측

### 🔹 FireplaceQu (690개 결측)
- **유형**: MAR
- 벽난로가 없는 경우(`Fireplaces`=0) 결측

### 🔹 LotFrontage (259개 결측)
- **유형**: MAR
- 인접 도로와의 거리
- 관련 변수: `Neighborhood` (지역별로 평균값 다름)

### 🔹 GarageCond (81개 결측)
- **유형**: MAR
- 차고가 없는 경우 (`GarageCars`=0)

### 🔹 GarageQual (81개 결측)
- **유형**: MAR
- 차고가 없는 경우

### 🔹 GarageFinish (81개 결측)
- **유형**: MAR
- 차고가 없는 경우

### 🔹 GarageYrBlt (81개 결측)
- **유형**: MAR
- 차고가 없는 경우

### 🔹 GarageType (81개 결측)
- **유형**: MAR
- 차고가 없는 경우

### 🔹 BsmtExposure (38개 결측)
- **유형**: MAR
- 지하실이 없는 경우 (`TotalBsmtSF`=0)

### 🔹 BsmtFinType2 (38개 결측)
- **유형**: MAR
- 지하실이 없는 경우

### 🔹 BsmtFinType1 (37개 결측)
- **유형**: MAR
- 지하실이 없는 경우

### 🔹 BsmtCond (37개 결측)
- **유형**: MAR
- 지하실이 없는 경우

### 🔹 BsmtQual (37개 결측)
- **유형**: MAR
- 지하실이 없는 경우

### 🔹 MasVnrArea (8개 결측)
- **유형**: MAR
- 석조 벽 마감이 없는 경우

### 🔹 MasVnrType (8개 결측)
- **유형**: MAR
- 석조 벽 마감이 없는 경우

### 🔹 Electrical (1개 결측)
- **유형**: MCAR
- 단순 누락 또는 오입력으로 추정

## 🎯 SalePrice 분포 분석

### 원본 분포
- `SalePrice`는 오른쪽으로 긴 꼬리를 가진 비대칭 분포 (양의 왜도)
- **Skewness (왜도)**: 1.88
- **Kurtosis (첨도)**: 6.54
- 이상치 및 고가 주택이 존재함

![SalePrice 분포](../figures/saleprice_distribution.png)

### 로그 변환
- `np.log1p(SalePrice)`로 로그 변환 시 정규분포에 가까워짐
- 모델링 전 로그 변환 고려할 가치 있음

![SalePrice 로그 변환 분포](../figures/saleprice_log_distribution.png)
