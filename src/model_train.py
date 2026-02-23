from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import optuna
import numpy as np

def train_model(X, y, use_tuning=False):
    """
    XGBoost 분류기를 학습합니다. 
    - use_tuning=True일 경우 Optuna를 사용해 최적의 하이퍼파라미터를 찾습니다.
    """
    # 데이터 분리 (8:2)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    
    if use_tuning:
        print("\n[Optuna] 하이퍼파라미터 튜닝 시작 (10회 시도)...")
        
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'logloss',
                'random_state': 42
            }
            model = XGBClassifier(**param, early_stopping_rounds=20)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds_proba = model.predict_proba(X_va)[:, 1]
            return average_precision_score(y_va, preds_proba)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        
        print("\n최적의 파라미터:", study.best_params)
        best_params = study.best_params
        best_params.update({'scale_pos_weight': scale_pos_weight, 'eval_metric': 'logloss', 'random_state': 42})
    else:
        best_params = {
            'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.05,
            'scale_pos_weight': scale_pos_weight, 'eval_metric': 'logloss', 'random_state': 42
        }

    print(f"최종 모델 학습 시작... (파생 변수 포함, scale_pos_weight: {scale_pos_weight:.2f})")
    model = XGBClassifier(**best_params, early_stopping_rounds=20)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    va_proba = model.predict_proba(X_va)[:, 1]
    print("모델 학습 완료.")
    
    return model, X_va, va_proba, y_va
