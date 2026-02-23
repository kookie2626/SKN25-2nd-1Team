import joblib
import torch
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# [경로 수정] image_aaf659.png 구조 반영
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "results"  # 모델이 results 폴더에 있음

@st.cache_resource
def get_resources():
    try:
        # 1. XGBoost 로드 (.pkl)
        xgb = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        
        # 2. ResNet 로드 (.pth) - CPU 환경 최적화
        # 모델 구조 선언이 필요할 수 있으나, 전체 저장 방식(torch.save) 기준으로 로드
        resnet = torch.load(MODELS_DIR / "resnet_model.pth", map_location='cpu')
        resnet.eval()
        
        # 3. 스케일러 로드
        scaler = joblib.load(MODELS_DIR / "resnet_scaler.pkl")
        
        # 피처 이름은 학습 데이터셋에서 직접 추출하거나 고정 (XGBoost 객체에서 추출 권장)
        feature_names = xgb.get_booster().feature_names
        
        return xgb, resnet, scaler, feature_names
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

def predict_churn(data_dict):
    res = get_resources()
    if not res: return 0.0, 0.0, 0.0
    xgb, resnet, scaler, feature_names = res
    
    # 데이터프레임 생성 및 정렬
    df = pd.DataFrame([data_dict]).reindex(columns=feature_names, fill_value=0)
    
    # XGBoost 예측
    p_xgb = xgb.predict_proba(df)[0][1]
    
    # ResNet 예측 (0% 에러 방지용 Sigmoid 처리)
    scaled_df = scaler.transform(df)
    input_tensor = torch.tensor(scaled_df, dtype=torch.float32)
    with torch.no_grad():
        output = resnet(input_tensor)
        # 로짓(음수 포함)으로 나올 경우를 대비해 반드시 Sigmoid 적용
        p_resnet = torch.sigmoid(output).flatten()[0].item()
    
    # 하이브리드 결과 (비중 조절 가능)
    final_score = (p_xgb * 0.6) + (p_resnet * 0.4)
    return p_xgb, p_resnet, final_score