"""
predict.py - 저장된 XGBoost / ResNet 모델을 불러와서 이탈 예측만 실행합니다.
재학습 불필요 - main.py / dl_main.py 실행 후 생성된 파일을 사용합니다.

사용법:
    python predict.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_data
from src.preprocessing import preprocess_for_modeling
from src.dl_model import ChurnResNet, get_device


RESULTS_DIR  = "results"
XGB_MODEL    = os.path.join(RESULTS_DIR, "xgboost_model.pkl")
RESNET_MODEL = os.path.join(RESULTS_DIR, "resnet_model.pth")
RESNET_SCALER= os.path.join(RESULTS_DIR, "resnet_scaler.pkl")


def predict_xgboost(X, threshold=0.6):
    """저장된 XGBoost 모델로 이탈 예측 (임계값 0.6 확정)"""
    if not os.path.exists(XGB_MODEL):
        raise FileNotFoundError(f"XGBoost 모델 없음: {XGB_MODEL}\n→ 먼저 'python main.py'를 실행하세요.")

    with open(XGB_MODEL, "rb") as f:
        model = pickle.load(f)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    print(f"\n[XGBoost] 임계값 {threshold} 적용")
    print(f"  예측 이탈자 수: {preds.sum():,} / {len(preds):,}")
    return proba, preds


def predict_resnet(X, device=None):
    """저장된 ResNet 모델로 이탈 예측 (임계값 0.8 확정)"""
    if not os.path.exists(RESNET_MODEL):
        raise FileNotFoundError(f"ResNet 모델 없음: {RESNET_MODEL}\n→ 먼저 'python dl_main.py'를 실행하세요.")
    if not os.path.exists(RESNET_SCALER):
        raise FileNotFoundError(f"스케일러 없음: {RESNET_SCALER}")

    # 스케일러 로드 & 변환
    with open(RESNET_SCALER, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)

    # 모델 구조 및 가중치 복원
    checkpoint = torch.load(RESNET_MODEL, map_location="cpu")
    threshold  = checkpoint['threshold']
    if device is None:
        device = get_device()

    model = ChurnResNet(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_blocks=checkpoint['num_blocks'],
        dropout=checkpoint['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X_tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        proba = model(X_tensor).cpu().numpy().flatten()

    preds = (proba >= threshold).astype(int)
    print(f"\n[ResNet] 임계값 {threshold} 적용 (Val AP: {checkpoint['best_val_ap']:.4f})")
    print(f"  예측 이탈자 수: {preds.sum():,} / {len(preds):,}")
    return proba, preds


def main():
    print("="*50)
    print("KKBox 이탈 예측 - 저장 모델 호출 (재학습 없음)")
    print("="*50)

    # 데이터 로드 & 전처리
    data_path = "data/kkbox_v3.parquet"
    if not os.path.exists(data_path):
        data_path = "kkbox_v3.parquet"
    df = load_data(data_path)
    X, y = preprocess_for_modeling(df)

    # XGBoost 예측
    print("\n--- XGBoost 예측 ---")
    xgb_proba, xgb_preds = predict_xgboost(X)

    # ResNet 예측
    print("\n--- ResNet 예측 ---")
    rn_proba, rn_preds = predict_resnet(X)

    # 두 모델 동의율
    agree = (xgb_preds == rn_preds).mean() * 100
    both_churn = ((xgb_preds == 1) & (rn_preds == 1)).sum()
    print(f"\n[앙상블 참고]")
    print(f"  두 모델 동의율:         {agree:.1f}%")
    print(f"  두 모델 모두 이탈 예측: {both_churn:,}명")

    print("\n" + "="*50)
    print("예측 완료.")
    print("="*50)


def predict_churn(data_dict):
    """
    단일 샘플 예측용 함수 (스트림릿용)
    """
    # 모델 리소스 로드
    try:
        # XGBoost 로드
        with open(XGB_MODEL, "rb") as f:
            xgb = pickle.load(f)
        
        # ResNet 로드
        checkpoint = torch.load(RESNET_MODEL, map_location='cpu')
        from src.dl_model import ChurnResNet
        resnet = ChurnResNet(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_blocks=checkpoint['num_blocks'],
            dropout=checkpoint['dropout']
        )
        resnet.load_state_dict(checkpoint['model_state_dict'])
        resnet.eval()
        
        # 스케일러 로드
        with open(RESNET_SCALER, "rb") as f:
            scaler = pickle.load(f)
        
        # 피처 이름 로드 (XGBoost에서 가져옴)
        feature_names = xgb.get_booster().feature_names
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return 0.0, 0.0, 0.0
    
    # 데이터 프레임 생성 및 정렬
    df = pd.DataFrame([data_dict]).reindex(columns=feature_names, fill_value=0)
    
    # XGBoost 예측
    p_xgb_raw = float(xgb.predict_proba(df)[0][1])
    
    # XGBoost scale_pos_weight 보정
    if p_xgb_raw > 0.5:
        import math
        logit = math.log(p_xgb_raw / (1 - p_xgb_raw))
        scale_pos_weight = 10.12
        logit_adjusted = logit / scale_pos_weight
        p_xgb = 1 / (1 + math.exp(-logit_adjusted))
    else:
        p_xgb = p_xgb_raw
    
    # ResNet 예측
    scaled_data = scaler.transform(df)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    
    try:
        with torch.no_grad():
            raw_output = resnet(input_tensor)
            val = raw_output.flatten()[0].item()
            if 0 <= val <= 1:
                p_resnet = val
            else:
                p_resnet = torch.sigmoid(raw_output).flatten()[0].item()
    except Exception:
        p_resnet = p_xgb
    
    # 최종 결과
    final_score = p_resnet
    
    return float(p_xgb), float(p_resnet), float(final_score)


if __name__ == "__main__":
    main()
