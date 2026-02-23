import pandas as pd
import numpy as np

def preprocess_for_modeling(df):
    """
    XGBoost/모델링을 위한 최종 전처리.
    - 파생 변수 생성 (Feature Engineering)
    - object/category 타입 처리
    - X와 y 분리
    """
    if "is_churn" not in df.columns:
        raise ValueError("데이터프레임에서 'is_churn' 컬럼을 찾을 수 없습니다.")
        
    df = df.copy()
    
    # --- 파생 변수 생성 (Feature Engineering) ---
    print("파생 변수 생성 중...")
    
    # 1. 활동성 품질 (완독률): 100% 재생 비율
    all_num_cols = ["num_25_sum", "num_50_sum", "num_75_sum", "num_985_sum", "num_100_sum"]
    df["play_total_count"] = df[all_num_cols].sum(axis=1)
    df["play_100_ratio"] = df["num_100_sum"] / (df["play_total_count"] + 1e-5)
    
    # 2. 이용 효율 (초당 유니크 곡 수)
    df["unq_per_sec"] = df["num_unq_mean"] / (df["total_secs_mean"] + 1e-5)
    
    # 3. 결제 합리성 (일평균 결제액 대비 활동량) 
    df["paid_per_sec"] = df["total_paid"] / (df["total_secs_sum"] + 1e-5)
    
    # 4. 자동 결제 및 취소 상호작용
    df["auto_cancel_inter"] = df["auto_renew_rate"] * (1 - df["cancel_rate"])
    
    # 5. 활동성 부재 여부 보정 (no_log_flag가 1이면 모든 로그 관련 변수는 0)
    # (이미 데이터에 반영되어 있을 확률이 높지만 명시적으로 처리 가능)
    
    # ------------------------------------------

    y = df["is_churn"]
    X = df.drop(columns=["is_churn", "msno", "play_total_count"], errors='ignore')
    
    # XGBoost를 위해 컬럼명을 문자열로 변환
    X.columns = X.columns.map(str)
    
    # 범주형(categorical) 컬럼 식별
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    
    if len(cat_cols) > 0:
        print(f"범주형 컬럼 인코딩 중: {list(cat_cols)}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    return X, y
