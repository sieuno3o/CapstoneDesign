# CapstoneDesign

주가 예측 성능 비교 프로젝트

- 고전적 모델
- AI 기반 모델
- 심리 지수만 이용한 모델
- AI 기반 + 심리 지수 결합 최종 모델

## 1. 프로젝트 개요

본 프로젝트는 주가 예측에서 과거 가격 데이터와 기술적 지표뿐 아니라
뉴스, 기사 댓글, 주식 커뮤니티 반응 등을 반영한 투자 심리 지수를 함께 활용하여
예측 성능을 비교하는 것을 목표로 한다.

기존 주가 예측은 주로 과거 가격 데이터와 기술적 지표 중심으로 이루어졌으나,
본 프로젝트는 투자자 심리를 반영한 지표를 추가하여 예측 정확도 향상 가능성을 검토한다.

## 2. 예측 목표

- 기본 목표: 다음 날 시가 예측
- 추가 목표: 상승/하락 방향성 예측
- 비교 대상: 아래 4개 모델

## 3. 모델 구성

### 1) 고전적 모델

과거 주가 데이터와 기술적 지표만 활용하는 전통적 모델

예시:

- ARIMA / ARMA
- 선형회귀
- 로지스틱 회귀

### 2) AI 기반 모델

과거 주가 데이터와 기술적 지표를 입력으로 사용하는 머신러닝 / AI 모델

예시:

- Random Forest
- XGBoost
- LSTM

### 3) 심리 지수만 이용한 모델

뉴스, 댓글, 커뮤니티 글, 거래량 변화율, 포지션 변화율 등으로 만든
투자 심리 지수만을 입력으로 사용하는 모델

### 4) AI 기반 + 심리 지수 결합 최종 모델

과거 주가 데이터, 기술적 지표, 투자 심리 지수를 모두 결합한 최종 모델

## 4. 입력 데이터

### 가격 데이터

- Date
- Open
- High
- Low
- Close
- Volume

### 기술적 지표

- 이동평균선
- 수익률
- 로그수익률
- 변동성
- 최근 주가 흐름
- 거래량 변화율

### 심리 데이터

- 뉴스 기사 제목/본문
- 기사 댓글
- 주식 커뮤니티 게시글
- 감성 분석 점수
- 군중심리지수
- 포지션 변화율
- 콜/풋 계약수
- 선물계약수

## 5. 프로젝트 폴더 구조

```text
CapstoneDesign/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ main.py
├─ configs/
│  └─ settings.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ sentiment/
├─ notebooks/
├─ results/
│  ├─ metrics/
│  ├─ predictions/
│  └─ figures/
└─ src/
   ├─ __init__.py
   ├─ data_loader.py
   ├─ preprocess.py
   ├─ feature_engineering.py
   ├─ sentiment_index.py
   ├─ split.py
   ├─ classical_model.py
   ├─ ai_model.py
   ├─ sentiment_only_model.py
   ├─ hybrid_model.py
   ├─ train.py
   ├─ evaluate.py
   └─ utils.py
```
