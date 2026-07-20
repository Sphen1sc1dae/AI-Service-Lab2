# main_simple.py
# ---------------------------------------------------------------
#  Customer Churn AI - FastAPI 서버 (축소판 Feature Engineering 모델용)
#  ST_Customer_Churn_v2.ipynb 에서 학습하고 저장한
#  models/customer_churn_simple_model.pkl (Logistic Regression Pipeline) 을
#  그대로 불러와 예측하는 API 입니다.
#
#  기존 main.py(원본 48개 Feature, RandomForest)와는 입력 스키마가 다르므로
#  별도 파일로 분리했습니다. 두 서버를 동시에 띄워 비교할 수도 있습니다.
# ---------------------------------------------------------------

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# ---------------------------------------------------------------
#   1. FastAPI 앱 생성
# ---------------------------------------------------------------
app = FastAPI(
    title="Customer Churn AI API (Simple / Feature Engineering)",
    description=(
        "ST_Customer_Churn_v2.ipynb 에서 EDA -> Feature Engineering -> "
        "Feature Selection -> Model Comparison 을 거쳐 선정한 "
        "Logistic Regression 모델로 이탈(Churn) 여부를 예측하는 API 입니다."
    ),
    version="1.0.0",
)

# React 개발 서버(Vite 기본 포트)에서 호출할 수 있도록 CORS를 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
#   2. 학습된 모델(Pipeline) 불러오기
#   - 전처리기(Imputer/Encoder/Scaler)와 모델이 하나의 Pipeline으로 저장되어
#     있으므로, 이 파일 하나만 불러오면 원본 형태의 입력을 그대로 넣을 수 있습니다.
# ---------------------------------------------------------------
MODEL_PATH = "models/customer_churn_simple_model.pkl"
model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------
#   3. Notebook의 Feature Engineering 로직을 동일하게 재현하기 위한 상수
#   (ST_Customer_Churn_v2.ipynb Chapter 4, Chapter 5 참고)
# ---------------------------------------------------------------

# Notebook의 Step 5(Final Feature Selection Report)에서 최종 선정된 22개 Feature.
# 이 중 TenureGroup, ContractRisk 2개는 원본 컬럼이 아니라 파생변수이므로,
# 아래에서 매 요청마다 동일한 공식으로 다시 계산합니다.
FINAL_FEATURE_COLUMNS = [
    "Satisfaction Score", "Contract", "ContractRisk", "Tenure in Months",
    "Number of Referrals", "Monthly Charge", "Total Revenue", "Total Charges",
    "Age", "TenureGroup", "Total Long Distance Charges", "Internet Type",
    "CLTV", "Avg Monthly GB Download", "Avg Monthly Long Distance Charges",
    "Payment Method", "Dependents", "Senior Citizen", "Online Security",
    "Internet Service", "Paperless Billing", "Offer",
]

TENURE_BINS = [0, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]
NEW_CUSTOMER_TENURE_LIMIT = 6


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """학습 때와 동일한 방식으로 파생변수를 계산해서 추가합니다.

    - Offer, Internet Type의 결측치는 "해당 사항 없음"을 의미하므로
      평균/최빈값이 아니라 의미가 명확한 문자열로 채웁니다. (2-4, Chapter 6-0 참고)
    - TenureGroup, ContractRisk는 학습에 사용한 공식과 반드시 동일해야 합니다.
    """
    df = df.copy()

    df["Offer"] = df["Offer"].fillna("No Offer")
    df["Internet Type"] = df["Internet Type"].fillna("No Internet")

    df["TenureGroup"] = pd.cut(
        df["Tenure in Months"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
    ).astype(str)

    is_month_to_month = (df["Contract"] == "Month-to-Month").astype(int)
    is_new_customer = (df["Tenure in Months"] <= NEW_CUSTOMER_TENURE_LIMIT).astype(int)
    df["ContractRisk"] = is_month_to_month + is_new_customer

    return df


# ---------------------------------------------------------------
#   4. 입력 데이터 형태 정의 (Pydantic BaseModel)
#   - TenureGroup, ContractRisk는 파생변수이므로 사용자가 직접 입력하지 않고
#     서버에서 자동으로 계산합니다. (main.py의 위도/경도 자동 변환과 같은 방식)
#   - Offer, Internet Type은 결측치가 의미를 가지는 컬럼이므로 Optional 로
#     받고, 비어 있으면 "No Offer"/"No Internet"으로 처리합니다.
# ---------------------------------------------------------------
class CustomerFeaturesSimple(BaseModel):
    satisfaction_score: int = Field(alias="Satisfaction Score", ge=1, le=5)
    contract: str = Field(alias="Contract")
    tenure_in_months: int = Field(alias="Tenure in Months", ge=0)
    number_of_referrals: int = Field(alias="Number of Referrals", ge=0)
    monthly_charge: float = Field(alias="Monthly Charge", gt=0)
    total_revenue: float = Field(alias="Total Revenue", ge=0)
    total_charges: float = Field(alias="Total Charges", ge=0)
    age: int = Field(alias="Age", ge=0)
    total_long_distance_charges: float = Field(alias="Total Long Distance Charges", ge=0)
    internet_type: Optional[str] = Field(default=None, alias="Internet Type")
    cltv: int = Field(alias="CLTV")
    avg_monthly_gb_download: int = Field(alias="Avg Monthly GB Download", ge=0)
    avg_monthly_long_distance_charges: float = Field(alias="Avg Monthly Long Distance Charges", ge=0)
    payment_method: str = Field(alias="Payment Method")
    dependents: str = Field(alias="Dependents")
    senior_citizen: str = Field(alias="Senior Citizen")
    online_security: str = Field(alias="Online Security")
    internet_service: str = Field(alias="Internet Service")
    paperless_billing: str = Field(alias="Paperless Billing")
    offer: Optional[str] = Field(default=None, alias="Offer")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Satisfaction Score": 3,
                "Contract": "Month-to-Month",
                "Tenure in Months": 8,
                "Number of Referrals": 1,
                "Monthly Charge": 80.65,
                "Total Revenue": 1024.1,
                "Total Charges": 633.3,
                "Age": 74,
                "Total Long Distance Charges": 390.8,
                "Internet Type": "Fiber Optic",
                "CLTV": 5302,
                "Avg Monthly GB Download": 17,
                "Avg Monthly Long Distance Charges": 48.85,
                "Payment Method": "Credit Card",
                "Dependents": "Yes",
                "Senior Citizen": "Yes",
                "Online Security": "No",
                "Internet Service": "Yes",
                "Paperless Billing": "Yes",
                "Offer": "Offer E",
            }
        },
    }


# ---------------------------------------------------------------
#   5. GET / : 서버가 살아있는지 확인하는 기본 경로
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Customer Churn AI API (Simple / Feature Engineering) is running",
        "status": "success",
    }


# ---------------------------------------------------------------
#   6. GET /health : 서버 상태 점검용(헬스 체크) 경로
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "OK",
    }


# ---------------------------------------------------------------
#   7. POST /predict : 고객 정보를 받아 이탈 여부를 예측
# ---------------------------------------------------------------
@app.post("/predict")
def predict(features: CustomerFeaturesSimple):
    # (1) 입력값(JSON)을 원본 컬럼명(alias) 기준의 딕셔너리로 변환합니다.
    data = features.model_dump(by_alias=True)

    # (2) 한 줄짜리 표(DataFrame)로 만듭니다.
    input_df = pd.DataFrame([data])

    # (3) 학습 때와 동일한 Feature Engineering을 적용합니다.
    #     (Offer/Internet Type 결측치 처리, TenureGroup, ContractRisk 계산)
    input_df = apply_feature_engineering(input_df)

    # (4) 모델이 학습한 22개 Feature만, 학습 때와 동일한 컬럼 이름으로 전달합니다.
    #     Pipeline 내부의 ColumnTransformer가 결측치 처리 / One-Hot Encoding /
    #     스케일링을 모두 알아서 처리합니다.
    input_df = input_df[FINAL_FEATURE_COLUMNS]

    # (5) 예측을 수행합니다. 결과는 0(잔류) 또는 1(이탈)이고,
    #     확률(이탈할 가능성)도 함께 계산합니다.
    prediction = int(model.predict(input_df)[0])
    churn_probability = float(model.predict_proba(input_df)[0][1])

    # (6) 숫자 예측값을 사람이 읽기 쉬운 문자로 변환합니다.
    result = "Churn" if prediction == 1 else "Stay"

    return {
        "prediction": prediction,
        "result": result,
        "churn_probability": round(churn_probability, 4),
    }