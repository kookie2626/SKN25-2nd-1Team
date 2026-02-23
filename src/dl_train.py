import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import optuna

def evaluate_dl_model(model, val_loader, device='cpu', threshold=None):
    """
    딥러닝 모델을 평가합니다.
    threshold=None 이면 F1-Score 기준 최적 임계값을 자동 탐색합니다.
    """
    model.eval()
    all_preds_proba = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds_proba.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            
    all_preds_proba = np.array(all_preds_proba).flatten()
    all_labels = np.array(all_labels).flatten()
    
    ap = average_precision_score(all_labels, all_preds_proba)
    
    if threshold is not None:
        # 확정 임계값 사용 - 탐색 없이 바로 평가
        best_thr = threshold
    else:
        # 최적 임계값 자동 탐색
        thresholds = np.arange(0.1, 0.95, 0.05)
        print(f"\n[임계값별 성능 비교]")
        print(f"{'임계값':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        print("-" * 40)
        
        best_thr, best_f1 = 0.5, 0.0
        for thr in thresholds:
            preds = (all_preds_proba >= thr).astype(int)
            p = precision_score(all_labels, preds, zero_division=0)
            r = recall_score(all_labels, preds, zero_division=0)
            f1 = f1_score(all_labels, preds, zero_division=0)
            marker = " <-- 최적" if f1 > best_f1 else ""
            print(f"{thr:>8.2f} {f1:>8.4f} {p:>10.4f} {r:>8.4f}{marker}")
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

    # 확정 임계값으로 최종 평가
    best_preds = (all_preds_proba >= best_thr).astype(int)
    p_best = precision_score(all_labels, best_preds)
    r_best = recall_score(all_labels, best_preds)
    f1_best = f1_score(all_labels, best_preds)
    
    print(f"\n{'='*50}")
    label = f"확정 임계값: {best_thr:.2f}" if threshold is not None else f"최적 임계값: {best_thr:.2f}"
    print(f"[ResNet 최종 평가 결과 - {label}]")
    print(f"{'='*50}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Precision:              {p_best:.4f}")
    print(f"Recall:                 {r_best:.4f}")
    print(f"F1-Score:               {f1_best:.4f}")
    print(f"\n혼동 행렬(Confusion Matrix):")
    print(confusion_matrix(all_labels, best_preds))
    
    return {'ap': ap, 'precision': p_best, 'recall': r_best, 'f1': f1_best, 'threshold': best_thr}


def train_dl_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu', verbose=True):
    """
    딥러닝 모델을 학습합니다.
    (Early Stopping, LR Scheduler, 메모리 내 최적 가중치 복원)
    
    [과적합 방지 핵심]
    - 이전 학습과 완전히 독립: 파일이 아닌 메모리(copy.deepcopy)에 best 가중치 저장
    - 학습 완료 후 best epoch 가중치를 모델에 복원하여 평가에 사용
    """
    import copy
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_ap = 0.0
    best_state_dict = None  # 파일 저장 NO, 메모리에만 보관 (이전 실행 영향 완전 차단)
    patience = 5
    counter = 0
    history = {'val_ap': []}
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # 검증
        model.eval()
        all_preds_proba = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                all_preds_proba.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_ap = average_precision_score(all_labels, all_preds_proba)
        history['val_ap'].append(val_ap)
        
        if verbose:
            print(f"Epoch {epoch+1}: Val AP: {val_ap:.4f} (LR: {optimizer.param_groups[0]['lr']:.6f})")
        
        scheduler.step(val_ap)
        
        # best 가중치를 메모리에 복사 (파일 I/O 없음 → 이전 실행과 완전 독립)
        if val_ap > best_ap:
            best_ap = val_ap
            best_state_dict = copy.deepcopy(model.state_dict())
            counter = 0
            if verbose:
                print(f"  ★ Best 갱신 (Val AP: {best_ap:.4f})")
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"  조기 종료 - Epoch {epoch+1} (patience={patience} 도달)")
                break
    
    # 학습 종료 후 best epoch 가중치 복원 (과적합 방지의 핵심!)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if verbose:
            print(f"\n✓ Best 가중치 복원 완료 (Val AP: {best_ap:.4f})")
            
    return best_ap, history

def tune_dl_lr(input_dim, train_loader, val_loader, device='cpu', n_trials=10, model_type='resnet'):
    """
    Optuna를 사용하여 최적의 Learning Rate를 탐색합니다.
    """
    from src.dl_model import ChurnResNet, ChurnLSTM
    print(f"\n[Optuna] {model_type.upper()} 최적의 Learning Rate 탐색 시작 ({n_trials}회 시도)...")
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        if model_type == 'resnet':
            model = ChurnResNet(input_dim=input_dim).to(device)
        elif model_type == 'lstm':
            model = ChurnLSTM(input_dim=input_dim).to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        # 탐색 정밀도를 위해 7 에폭 학습
        ap, _ = train_dl_model(model, train_loader, val_loader, epochs=7, lr=lr, device=device, verbose=False)
        return ap

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"탐색 완료! 최적의 Learning Rate: {study.best_params['lr']:.6f}")
    return study.best_params['lr']

def finetune_resnet(input_dim, train_loader, val_loader, device='cpu', n_trials=20):
    """
    ResNet의 모든 핵심 하이퍼파라미터를 동시에 Optuna로 최적화합니다. (Full Fine-tuning)
    
    탐색 대상:
    - lr         : 학습률 (1e-4 ~ 3e-2)
    - hidden_dim : 레이어 너비 (128 / 256 / 512)
    - num_blocks : Residual Block 개수 (3 / 4 / 5 / 6)
    - dropout    : 드롭아웃 비율 (0.1 ~ 0.4)
    """
    import copy
    from src.dl_model import ChurnResNet
    print(f"\n[Full Fine-tuning] ResNet 전체 구조 최적화 시작 ({n_trials}회 시도)...")
    print("탐색 파라미터: lr, hidden_dim, num_blocks, dropout")
    
    def objective(trial):
        lr         = trial.suggest_float("lr",         1e-4, 3e-2,  log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        num_blocks = trial.suggest_int("num_blocks",   3, 6)
        dropout    = trial.suggest_float("dropout",    0.1, 0.4)
        
        # 매 trial마다 완전히 새로운 모델 (이전 학습 영향 없음)
        model = ChurnResNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        ).to(device)
        
        # 탐색 당 10 에폭 (정확도와 속도의 균형)
        ap, _ = train_dl_model(
            model, train_loader, val_loader,
            epochs=10, lr=lr, device=device, verbose=False
        )
        return ap

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best = study.best_params
    print(f"\n[Fine-tuning 완료] 최적 하이퍼파라미터:")
    print(f"  lr         = {best['lr']:.6f}")
    print(f"  hidden_dim = {best['hidden_dim']}")
    print(f"  num_blocks = {best['num_blocks']}")
    print(f"  dropout    = {best['dropout']:.2f}")
    print(f"  Best Val AP (10 epoch)  = {study.best_value:.4f}")
    
    return best
