# Customer Churn AI — 알고리즘 & 성능지표 완전정리

`train_model.py`에서 사용하는 개념들을, 머신러닝을 처음 배우는 사람도
따라올 수 있도록 비유와 예시 위주로 정리한 문서입니다.

이 프로젝트는 고객의 여러 정보(계약 형태, 요금, 이용 기간 등)를 보고
그 고객이 **이탈(Churn)할지, 잔류(Stay)할지**를 예측하는
**이진 분류(Binary Classification)** 문제이며, **랜덤포레스트 분류
(Random Forest Classifier)** 알고리즘을 사용합니다.

---

## 목차

- [Customer Churn AI — 알고리즘 \& 성능지표 완전정리](#customer-churn-ai--알고리즘--성능지표-완전정리)
  - [목차](#목차)
  - [1. 분류(Classification) 문제란](#1-분류classification-문제란)
  - [2. 데이터 전처리 — 왜 특정 컬럼을 제거할까? (Data Leakage)](#2-데이터-전처리--왜-특정-컬럼을-제거할까-data-leakage)
  - [3. One-Hot Encoding (문자를 숫자로 바꾸기)](#3-one-hot-encoding-문자를-숫자로-바꾸기)
  - [4. Train / Test 분리](#4-train--test-분리)
  - [5. 랜덤포레스트 분류(Random Forest Classifier)](#5-랜덤포레스트-분류random-forest-classifier)
  - [6. 클래스 불균형(Class Imbalance)이란](#6-클래스-불균형class-imbalance이란)
  - [7. 혼동행렬(Confusion Matrix)과 성능 지표](#7-혼동행렬confusion-matrix과-성능-지표)
    - [Accuracy (정확도)](#accuracy-정확도)
    - [Precision (정밀도)](#precision-정밀도)
    - [Recall (재현율)](#recall-재현율)
    - [F1 Score](#f1-score)
    - [왜 Recall이 특히 중요할 수 있을까? (비즈니스 관점)](#왜-recall이-특히-중요할-수-있을까-비즈니스-관점)
  - [8. 이 프로젝트의 실제 결과 해석](#8-이-프로젝트의-실제-결과-해석)
  - [9. 한눈에 보는 용어 정리표](#9-한눈에-보는-용어-정리표)

---

## 1. 분류(Classification) 문제란

정답(Target)이 "숫자 값"이 아니라 "카테고리(범주)"인 문제를
**분류(Classification)** 라고 부릅니다.

- **Feature (X, 입력값)**: `Gender`, `Tenure in Months`, `Contract`,
  `Monthly Charge` 등 고객에 대한 여러 정보
- **Target (y, 정답)**: `Churn Label` — 고객이 이탈했는지(`Yes`) /
  안 했는지(`No`)를 나타내는 값

정답이 두 가지 값(이탈/잔류)만 가지므로 이를 **이진 분류(Binary
Classification)** 라고 부릅니다. (참고: 정답이 3개 이상이면
"다중 분류(Multi-class Classification)"라고 부릅니다.)

```python
df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
```

문자 `'Yes'`/`'No'`는 모델이 계산할 수 없으므로, 미리 숫자
`1`(이탈)과 `0`(잔류)으로 바꿔줍니다. 이제부터 모델의 목표는
"주어진 고객 정보(X)를 보고, `Churn Label`이 0인지 1인지를
맞히는 것"이 됩니다.

---

## 2. 데이터 전처리 — 왜 특정 컬럼을 제거할까? (Data Leakage)

```python
drop_columns = [
    "Customer ID",
    "Customer Status",
    "Churn Score",
    "Churn Category",
    "Churn Reason"
]
df = df.drop(columns=drop_columns)
```

- **`Customer ID`**: 고객을 구분하는 고유 번호일 뿐, 이탈 여부와
  아무 관련이 없는 정보이므로 제거합니다. (이런 정보를 학습에
  포함하면 모델이 "무의미한 우연의 규칙"을 외워버릴 수 있습니다.)
- **`Customer Status`, `Churn Score`, `Churn Category`,
  `Churn Reason`**: 이 컬럼들은 사실 **"이미 이탈이 일어난 뒤에
  기록된 결과"** 에 해당합니다. 예를 들어 `Customer Status`가
  `Churned`라면, 그것 자체가 곧 `Churn Label = Yes`라는 뜻입니다.

  이렇게 **"미래에만 알 수 있는(정답과 사실상 같은) 정보"** 를
  실수로 입력값(X)에 포함시키는 것을 **데이터 누수(Data Leakage)**
  라고 부릅니다. 데이터 누수가 있으면 모델은 정답을 "예측"하는 게
  아니라 "커닝"하게 되어, 학습/테스트에서는 정확도가 비정상적으로
  높게 나오지만 실제 서비스(아직 이탈 여부를 모르는 신규 고객
  예측)에서는 전혀 쓸모가 없어집니다. 그래서 이런 컬럼들은 학습
  전에 반드시 제거해야 합니다.

---

## 3. One-Hot Encoding (문자를 숫자로 바꾸기)

```python
X = pd.get_dummies(X)
```

머신러닝 모델은 숫자만 계산할 수 있습니다. `Contract` 컬럼처럼
`"Month-to-Month"`, `"One Year"`, `"Two Year"` 같은 문자(카테고리)
값은 그대로 넣을 수 없습니다.

**One-Hot Encoding**은 카테고리 컬럼 하나를, 카테고리 개수만큼의
"있다/없다(1/0)" 컬럼으로 쪼개는 방법입니다.

| Contract (원본) |
| --------------- |
| Month-to-Month  |
| One Year        |
| Two Year        |

⬇ One-Hot Encoding 적용 후

| Contract_Month-to-Month | Contract_One Year | Contract_Two Year |
| ----------------------- | ----------------- | ----------------- |
| 1                       | 0                 | 0                 |
| 0                       | 1                 | 0                 |
| 0                       | 0                 | 1                 |

이 프로젝트는 `Gender`, `Contract`, `Payment Method` 등 문자형
컬럼이 많아서, One-Hot Encoding 이후 최종 Feature 개수가
**1,177개**까지 늘어납니다. (원본 컬럼은 40여 개였지만, 카테고리
값 하나하나가 새로운 0/1 컬럼이 되기 때문입니다.)

---

## 4. Train / Test 분리

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

House Price AI 프로젝트와 동일한 이유로, 전체 고객의 80%(Train)로
모델을 학습시키고 나머지 20%(Test)는 "한 번도 보여주지 않은 고객"
으로 남겨두어 채점에만 사용합니다. 이렇게 해야 모델이 실제로
새로운 고객을 만났을 때도 잘 예측하는지 확인할 수 있습니다.
(`random_state=42`는 매번 같은 방식으로 나뉘도록 고정하는
재현성 장치입니다.)

---

## 5. 랜덤포레스트 분류(Random Forest Classifier)

기본 원리는 House Price AI에서 사용한 `RandomForestRegressor`와
같은 "숲(Forest)" 구조입니다. 서로 다른 데이터·Feature 샘플로
수백 개의 결정 트리(Decision Tree)를 만들고, 그 결과를 모아
최종 예측을 만듭니다. (자세한 결정 트리/배깅 원리는 House Price
AI 프로젝트의 `ALGORITHM_GUIDE.md`를 참고하세요.)

다만 **회귀(숫자 예측)** 와 **분류(카테고리 예측)** 는 트리들의
결과를 모으는 방식이 다릅니다.

| 구분                          | 모으는 방식                                                                            |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| RandomForestRegressor (회귀)  | 수백 개 트리의 예측값을 **평균**                                                       |
| RandomForestClassifier (분류) | 수백 개 트리가 각각 "이탈/잔류"에 **투표(Voting)**, 다수결(또는 확률 평균)로 최종 결정 |

즉, 트리 100개 중 78개가 "이탈(1)"이라고 예측하고 22개가
"잔류(0)"라고 예측하면, 다수결에 따라 최종 예측은 "이탈"이 됩니다.
(scikit-learn은 실제로는 각 트리가 낸 확률을 평균 내는 방식을
사용하지만, 개념적으로는 "여러 전문가의 투표"라고 이해하면
충분합니다.)

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

---

## 6. 클래스 불균형(Class Imbalance)이란

이 데이터셋의 실제 정답 분포를 보면:

```
Churn Label
No (잔류, 0)  : 5,174명 (약 73.4%)
Yes (이탈, 1) : 1,869명 (약 26.6%)
```

이렇게 한쪽 클래스(잔류)가 다른 쪽(이탈)보다 훨씬 많은 상태를
**클래스 불균형(Class Imbalance)** 이라고 부릅니다.

**왜 중요할까요?** 만약 아무 생각 없이 "모든 고객은 잔류할
것이다"라고만 찍는 매우 게으른 모델을 만들어도, 정확도(Accuracy)는
약 73.4%가 나옵니다! 즉, 클래스가 불균형한 데이터에서는
**정확도(Accuracy) 하나만으로는 모델이 정말 똑똑한지 판단할 수
없습니다.** 그래서 아래에서 설명할 Precision, Recall, F1 Score를
함께 봐야 합니다.

---

## 7. 혼동행렬(Confusion Matrix)과 성능 지표

분류 모델을 채점할 때는 예측과 실제를 아래처럼 4가지 경우로
나누어 표로 정리합니다. 이를 **혼동행렬(Confusion Matrix)**
이라고 부릅니다. (이 프로젝트에서는 "이탈(1)"을 맞히는 것이
목표이므로, 이탈을 Positive(양성)로 봅니다.)

|                   | 실제: 이탈(1)                                           | 실제: 잔류(0)                                             |
| ----------------- | ------------------------------------------------------- | --------------------------------------------------------- |
| **예측: 이탈(1)** | TP (True Positive)<br>이탈을 이탈로 맞게 예측           | FP (False Positive)<br>잔류인데 이탈로 잘못 예측 (헛다리) |
| **예측: 잔류(0)** | FN (False Negative)<br>이탈인데 잔류로 잘못 예측 (놓침) | TN (True Negative)<br>잔류를 잔류로 맞게 예측             |

이 4가지 숫자를 조합해서 아래 지표들을 계산합니다.

### Accuracy (정확도)

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

전체 고객 중, 이탈이든 잔류든 **맞춘 비율**입니다. (비유: 화살이
전반적으로 과녁 중심에 얼마나 가까이 모였는가)

### Precision (정밀도)

```
Precision = TP / (TP + FP)
```

모델이 **"이탈할 것"이라고 예측한 고객들 중에서, 실제로 이탈한
비율**입니다. Precision이 낮다는 것은 "이탈 경고"를 남발해서
헛다리(FP)를 많이 짚었다는 뜻입니다. (비유: 화살들이 서로 얼마나
가깝게 뭉쳐 있는가 — 과녁 중심이 아니어도 뭉쳐 있으면 Precision은
높을 수 있음)

### Recall (재현율)

```
Recall = TP / (TP + FN)
```

**실제로 이탈한 고객들 중에서, 모델이 놓치지 않고 잡아낸
비율**입니다. Recall이 낮다는 것은 실제 이탈 고객을 "잔류할
것"이라고 잘못 판단해(FN) 놓쳤다는 뜻입니다.

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Precision과 Recall은 보통 한쪽을 올리면 다른 쪽이 떨어지는
**트레이드오프(Trade-off)** 관계에 있습니다. F1 Score는 이 둘의
**조화평균(harmonic mean)** 으로, 두 지표의 균형을 하나의
숫자로 요약해서 보여줍니다.

### 왜 Recall이 특히 중요할 수 있을까? (비즈니스 관점)

고객 이탈 예측에서는 보통 다음 두 실수의 "비용"이 다릅니다.

- **FN (이탈 고객을 놓침)**: 실제로 떠날 고객에게 아무 조치도
  취하지 못해 그대로 고객을 잃습니다. → 손해가 큼
- **FP (잔류 고객을 이탈로 오판)**: 사실 떠나지 않을 고객에게
  괜히 할인 쿠폰 등 유지 혜택을 하나 더 줍니다. → 비용은 들지만
  손해는 상대적으로 작음

이런 이유로 이탈 예측 서비스에서는 Precision보다 **Recall을
더 중요하게 보는 경우가 많습니다.** ("혹시 놓치는 이탈 고객이
없도록, 조금 과하게 경고하더라도 최대한 잡아내자"는 전략)

---

## 8. 이 프로젝트의 실제 결과 해석

`train_model.py`를 실행하면 대략 아래와 같은 결과가 출력됩니다.

```
Accuracy  : 0.9312
Precision : 0.9605
Recall    : 0.7900
F1 Score  : 0.8669
```

이 숫자들을 풀어서 설명하면:

- **Accuracy = 93.12%** → 전체 테스트 고객 1,409명 중 약 93%를
  이탈/잔류 어느 쪽이든 맞게 예측했습니다. 앞서 설명했듯 "무조건
  잔류로 찍기"만 해도 73.4%가 나오는 데이터이므로, 93.12%는
  실제로 유의미하게 학습이 되었다는 뜻입니다.
- **Precision = 96.05%** → 모델이 "이탈할 것"이라고 예측한
  고객들 중, 실제로 96%가 진짜 이탈했습니다. 즉 이탈 경고를
  거의 헛다리 짚지 않습니다(FP가 적음).
- **Recall = 79.00%** → 실제로 이탈한 고객 중 79%를 모델이
  잡아냈습니다. 바꿔 말하면, 실제 이탈 고객의 약 **21%는
  모델이 놓쳤다(FN)** 는 뜻입니다.
- **F1 Score = 86.69%** → Precision(96%)과 Recall(79%)의 균형을
  하나로 요약한 값입니다. Precision보다 Recall이 상대적으로
  낮으므로, 이 모델은 "이탈 경고를 내면 믿을 만하지만(Precision
  높음), 일부 이탈 고객은 놓치는 편(Recall 상대적으로 낮음)"
  이라고 해석할 수 있습니다. 위에서 설명한 비즈니스 관점에서는,
  실전에 투입하기 전 Recall을 더 끌어올리는 방향(예: 임계값
  조정, 클래스 가중치 부여 등)을 고려해볼 수 있습니다.

---

## 9. 한눈에 보는 용어 정리표

| 용어                              | 한 줄 설명                                                    |
| --------------------------------- | ------------------------------------------------------------- |
| Classification (분류)             | 정답이 카테고리(예: 이탈/잔류)인 문제                         |
| Binary Classification (이진 분류) | 정답이 두 가지 값 중 하나인 분류 문제                         |
| Data Leakage (데이터 누수)        | 정답과 사실상 동일한 정보가 실수로 입력값(X)에 포함되는 문제  |
| One-Hot Encoding                  | 카테고리 컬럼을 "있다/없다(1/0)" 컬럼 여러 개로 변환하는 방법 |
| Class Imbalance (클래스 불균형)   | 정답 클래스의 비율이 한쪽으로 치우친 상태                     |
| Confusion Matrix (혼동행렬)       | 예측/실제 결과를 TP·FP·FN·TN 4가지로 정리한 표                |
| TP (True Positive)                | 이탈을 이탈로 맞게 예측                                       |
| FP (False Positive)               | 잔류인데 이탈로 잘못 예측 (헛다리)                            |
| FN (False Negative)               | 이탈인데 잔류로 잘못 예측 (놓침)                              |
| TN (True Negative)                | 잔류를 잔류로 맞게 예측                                       |
| Accuracy (정확도)                 | 전체 중 맞춘 비율                                             |
| Precision (정밀도)                | "이탈"이라고 예측한 것 중 실제로 맞은 비율                    |
| Recall (재현율)                   | 실제 이탈 중 모델이 잡아낸 비율                               |
| F1 Score                          | Precision과 Recall의 조화평균                                 |
