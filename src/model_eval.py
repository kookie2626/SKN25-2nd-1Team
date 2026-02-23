from sklearn.metrics import average_precision_score, confusion_matrix, classification_report, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import os

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    다양한 임계값을 테스트하여 특정 지표(기본값 F1-score)를 최대로 만드는 임계값을 찾습니다.
    """
    thresholds = np.arange(0.1, 0.91, 0.05)
    results = []
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        # 클래스 1에 대한 Precision/Recall 추출
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        
        results.append({
            'Threshold': round(thr, 2),
            'F1-Score': round(f1, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4)
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n[임계값별 성능 분석]")
    print(results_df.to_string(index=False))
    
    best_idx = results_df['F1-Score'].idxmax()
    best_thr = results_df.loc[best_idx, 'Threshold']
    best_f1 = results_df.loc[best_idx, 'F1-Score']
    
    print(f"\n최적의 임계값: {best_thr} (F1-Score: {best_f1})")
    
    return best_thr

def evaluate_model(model, X_va, y_va, results_dir="results", va_proba=None):
    """
    XGBoost 모델을 평가합니다.
    [확정 설정] 임계값 0.6 고정
    - 실측 기준: Precision ≈ 0.8319, Recall ≈ 0.9452 (임계값 0.6 적용 시)
    """
    if va_proba is None:
        va_proba = model.predict_proba(X_va)[:, 1]
    
    # ★ 확정 임계값: 0.6
    best_thr = 0.6
    va_pred = (va_proba >= best_thr).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    ap  = average_precision_score(y_va, va_proba)
    p   = precision_score(y_va, va_pred)
    r   = recall_score(y_va, va_pred)
    f1  = f1_score(y_va, va_pred)
    cm  = confusion_matrix(y_va, va_pred)
    
    print(f"\n{'='*50}")
    print(f"[XGBoost 최종 평가 결과 - 임계값 {best_thr}]")
    print(f"{'='*50}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Precision:              {p:.4f}")
    print(f"Recall:                 {r:.4f}")
    print(f"F1-Score:               {f1:.4f}")
    print("\n혼동 행렬(Confusion Matrix):")
    print(cm)
    print("\n분류 보고서(Classification Report):")
    print(classification_report(y_va, va_pred))
    
    # 혼동 행렬 시각화 저장
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'XGBoost Confusion Matrix (Threshold: {best_thr})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    return {'ap': ap, 'precision': p, 'recall': r, 'f1': f1}

def plot_shap_values(model, X_tr, X_va, results_dir="results"):
    """
    SHAP 요약 플롯 및 중요도 플롯을 생성합니다.
    """
    # 데이터가 너무 크면 2000개만 샘플링하여 계산
    if len(X_va) > 2000:
        print(f"SHAP 분석을 위해 데이터를 2000개로 샘플링합니다. (전체: {len(X_va)}개)")
        X_va_sample = X_va.sample(2000, random_state=42)
    else:
        X_va_sample = X_va

    print("SHAP 값 계산 중 (시간이 소요될 수 있습니다)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_va_sample)
    
    # SHAP 요약 플롯
    plt.figure()
    shap.summary_plot(shap_values, X_va_sample, show=False)
    plt.savefig(os.path.join(results_dir, "shap_summary.png"))
    plt.close()
    print(f"SHAP 플롯이 {results_dir} 폴더에 저장되었습니다.")
