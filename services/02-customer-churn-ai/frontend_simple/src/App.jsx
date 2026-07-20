import { useState } from "react";
import "./App.css";

// FastAPI 서버 주소
// 개발 중에는 vite.config.js의 proxy 설정을 통해
// backend-simple-api(main_simple.py) 컨테이너로 전달됩니다.
const API_URL = "/predict";

// -----------------------------------------------------------
// main_simple.py(CustomerFeaturesSimple)가 요구하는 입력 항목 목록
// TenureGroup, ContractRisk는 서버가 Tenure in Months/Contract 값으로
// 자동 계산하는 파생변수이므로 화면에서는 입력받지 않습니다.
// key: 서버로 보낼 데이터의 이름(FastAPI가 요구하는 컬럼명과 동일)
// label: 화면에 보여줄 이름
// type: "select"(선택형) 또는 "number"(숫자 입력)
// -----------------------------------------------------------
const FIELDS = [
  { key: "Satisfaction Score", label: "만족도 점수 (1~5)", type: "number", min: 1, max: 5 },
  {
    key: "Contract",
    label: "계약 형태",
    type: "select",
    options: ["Month-to-Month", "One Year", "Two Year"],
  },
  { key: "Tenure in Months", label: "이용 기간 (개월)", type: "number", min: 0 },
  { key: "Number of Referrals", label: "추천 횟수", type: "number", min: 0 },
  { key: "Monthly Charge", label: "월 요금 ($)", type: "number" },
  { key: "Total Revenue", label: "총 수익 ($)", type: "number", min: 0 },
  { key: "Total Charges", label: "누적 요금 ($)", type: "number", min: 0 },
  { key: "Age", label: "나이", type: "number", min: 0 },
  { key: "Total Long Distance Charges", label: "총 장거리 통화 요금 ($)", type: "number", min: 0 },
  {
    key: "Internet Type",
    label: "인터넷 종류",
    type: "select",
    options: ["DSL", "Fiber Optic", "Cable", "No Internet"],
  },
  { key: "CLTV", label: "고객 생애 가치 (CLTV)", type: "number" },
  { key: "Avg Monthly GB Download", label: "월 평균 다운로드 (GB)", type: "number", min: 0 },
  {
    key: "Avg Monthly Long Distance Charges",
    label: "월 평균 장거리 통화 요금 ($)",
    type: "number",
    min: 0,
  },
  {
    key: "Payment Method",
    label: "결제 방법",
    type: "select",
    options: ["Bank Withdrawal", "Credit Card", "Mailed Check"],
  },
  { key: "Dependents", label: "부양가족 여부", type: "select", options: ["No", "Yes"] },
  { key: "Senior Citizen", label: "고령자 여부", type: "select", options: ["No", "Yes"] },
  { key: "Online Security", label: "온라인 보안 서비스 이용", type: "select", options: ["No", "Yes"] },
  { key: "Internet Service", label: "인터넷 서비스 이용", type: "select", options: ["Yes", "No"] },
  { key: "Paperless Billing", label: "전자 청구서 이용", type: "select", options: ["Yes", "No"] },
  {
    key: "Offer",
    label: "프로모션 오퍼",
    type: "select",
    options: ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"],
  },
];

// 화면에 보이는 입력폼의 초기값 (선택형은 첫 번째 옵션, 숫자형은 빈 문자열)
const INITIAL_FORM = FIELDS.reduce((acc, field) => {
  acc[field.key] = field.type === "select" ? field.options[0] : "";
  return acc;
}, {});

export default function App() {
  // 입력폼 값들을 저장하는 state
  const [form, setForm] = useState(INITIAL_FORM);

  // 예측 결과를 저장하는 state ({ prediction, result, churn_probability } 형태)
  const [prediction, setPrediction] = useState(null);

  // 에러 메시지를 저장하는 state
  const [error, setError] = useState("");

  // API 요청 중인지 여부를 저장하는 state
  const [loading, setLoading] = useState(false);

  // 입력값이 바뀔 때마다 실행되는 함수
  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  // "예측하기" 버튼을 눌렀을 때 실행되는 함수
  async function handleSubmit(e) {
    e.preventDefault();

    // 이전 결과와 에러 메시지 초기화
    setPrediction(null);
    setError("");
    setLoading(true);

    try {
      // 화면의 입력값을 FastAPI가 원하는 타입(숫자/문자)으로 변환
      const requestBody = {};
      FIELDS.forEach((field) => {
        const value = form[field.key];
        requestBody[field.key] = field.type === "number" ? Number(value) : value;
      });

      // FastAPI 서버로 예측 요청 보내기
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      // 서버가 에러 응답을 준 경우
      if (!response.ok) {
        throw new Error("서버 응답 오류");
      }

      const data = await response.json();
      setPrediction(data);
    } catch {
      // fetch 자체가 실패하거나(서버가 꺼져있는 경우) 응답이 실패한 경우
      setError("FastAPI 서버에 연결할 수 없습니다.");
    } finally {
      setLoading(false);
    }
  }

  // prediction 값(0/1)에 따라 결과 카드에 다른 스타일 클래스를 적용
  const isChurn = prediction?.prediction === 1;

  return (
    <main className="dashboard">
      <section className="hero-card">
        {/* Header 영역 */}
        <p className="badge">AI Service Blueprint · Simple</p>
        <h1>Customer Churn AI (Simple)</h1>
        <p className="description">
          React + FastAPI 기반
          <br />
          Feature Engineering 모델(main_simple.py)을 이용한 고객 이탈 예측 서비스
        </p>

        {/* 입력폼 영역 */}
        <form className="predict-form" onSubmit={handleSubmit}>
          <div className="input-grid">
            {FIELDS.map((field) => (
              <label key={field.key} className="input-item">
                <span>{field.label}</span>
                {field.type === "select" ? (
                  <select name={field.key} value={form[field.key]} onChange={handleChange}>
                    {field.options.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step="any"
                    min={field.min}
                    max={field.max}
                    name={field.key}
                    value={form[field.key]}
                    onChange={handleChange}
                    required
                  />
                )}
              </label>
            ))}
          </div>

          <button type="submit" className="primary-button" disabled={loading}>
            {loading ? "예측 중..." : "예측하기"}
          </button>
        </form>

        {/* 에러 메시지 영역 */}
        {error && <p className="error-message">{error}</p>}

        {/* 예측 결과 카드 */}
        {prediction && (
          <div className={`result-card ${isChurn ? "result-churn" : "result-stay"}`}>
            <p className="result-label">예측 결과</p>
            <p className="result-value">{prediction.result}</p>
            <p className="result-sub">
              prediction: {prediction.prediction} · churn_probability:{" "}
              {prediction.churn_probability}
            </p>
          </div>
        )}
      </section>
    </main>
  );
}
