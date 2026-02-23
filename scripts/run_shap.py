import pandas as pd
import numpy as np
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.abspath('src'))
from preprocessing import preprocess_for_modeling
from data_loader import load_data

print("데이터 로딩 중...")
df = load_data('data/kkbox_v3.parquet')
X, y = preprocess_for_modeling(df)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("모델 학습 중...")
model = LGBMClassifier(n_estimators=100, random_state=42, is_unbalance=True)
model.fit(X_tr, y_tr)

print("SHAP 값 계산 중...")
explainer = shap.TreeExplainer(model)
X_sample = X_va.sample(1000, random_state=42)
shap_vals = explainer.shap_values(X_sample)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

print('\nSHAP 기반 상위 10개 피처:')
importance = np.abs(shap_vals).mean(axis=0)
feat_importance = pd.DataFrame(list(zip(X.columns, importance)), columns=['feature', 'importance']).sort_values('importance', ascending=False)
print(feat_importance.head(10))
