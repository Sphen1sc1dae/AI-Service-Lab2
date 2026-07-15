#!/bin/bash

echo "모델 파일 확인 중..."

if [ ! -f models/bike_demand_model.pkl ]; then
    echo "모델 파일이 없습니다. 모델 학습을 시작합니다."
    python train_model.py
else
    echo "모델 파일이 이미 존재합니다."
fi

echo "FastAPI 서버를 시작합니다."
uvicorn main:app --host 0.0.0.0 --port 8000