import os
from src.data_loader import load_data
from src.preprocessing import preprocess_for_modeling
from src.dl_preprocessing import prepare_dl_data
from src.dl_model import ChurnResNet, get_device
from src.dl_train import train_dl_model, evaluate_dl_model, finetune_resnet

def main():
    print("="*50)
    print("KKBox 이탈 예측 파이프라인 (Deep Learning - ResNet Fine-tuned)")
    print("[확정 모델] 임계값 0.8 고정")
    print("="*50)

    # 1. 데이터 로드
    data_path = "data/kkbox_v3.parquet"
    if not os.path.exists(data_path):
        data_path = "kkbox_v3.parquet"
    df = load_data(data_path)

    # 2. 기초 전처리
    print("\n[Step 1] 기초 전처리 중...")
    X, y = preprocess_for_modeling(df)

    # 3. 딥러닝용 데이터 준비 (ResNet: WeightedRandomSampler 미사용)
    print("\n[Step 2] 딥러닝용 데이터 준비 중...")
    train_loader, val_loader, scaler, input_dim = prepare_dl_data(X, y, use_weighted_sampler=False)

    # 4. 장치 설정
    device = get_device()

    # 5. 확정 하이퍼파라미터 (Fine-tuning Trial 5 Best, Val AP: 0.9378)
    #    lr=0.01121, hidden_dim=256, num_blocks=5, dropout=0.1669
    BEST_LR         = 0.01121
    BEST_HIDDEN_DIM = 256
    BEST_NUM_BLOCKS = 5
    BEST_DROPOUT    = 0.1669

    # --- [Optuna 재탐색 - 필요 시 주석 해제] ---
    # print("\n[Step 3] ResNet Full Fine-tuning 시작 (10회 구조 탐색 중)...")
    # best_params = finetune_resnet(input_dim, train_loader, val_loader, device=device, n_trials=10)
    # BEST_LR         = best_params['lr']
    # BEST_HIDDEN_DIM = best_params['hidden_dim']
    # BEST_NUM_BLOCKS = best_params['num_blocks']
    # BEST_DROPOUT    = best_params['dropout']
    # -------------------------------------------

    print(f"\n[Step 3] 확정 하이퍼파라미터:")
    print(f"  lr={BEST_LR}, hidden_dim={BEST_HIDDEN_DIM}, num_blocks={BEST_NUM_BLOCKS}, dropout={BEST_DROPOUT}")

    # 6. 모델 학습
    print(f"\n[Step 4] 모델 학습 시작 (Max Epochs: 50)...")
    model = ChurnResNet(
        input_dim=input_dim,
        hidden_dim=BEST_HIDDEN_DIM,
        num_blocks=BEST_NUM_BLOCKS,
        dropout=BEST_DROPOUT
    ).to(device)
    best_ap, history = train_dl_model(
        model, train_loader, val_loader,
        epochs=50, lr=BEST_LR, device=device
    )

    # 7. 평가 (임계값 0.8 확정)
    print("\n[Step 5] 모델 평가 중... (확정 임계값: 0.8)")
    evaluate_dl_model(model, val_loader, device=device, threshold=0.8)

    # 8. 모델 및 스케일러 저장 (나중에 재학습 없이 바로 호출 가능)
    import pickle, torch
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim':        input_dim,
        'hidden_dim':       BEST_HIDDEN_DIM,
        'num_blocks':       BEST_NUM_BLOCKS,
        'dropout':          BEST_DROPOUT,
        'threshold':        0.8,
        'best_val_ap':      best_ap
    }, os.path.join(results_dir, "resnet_model.pth"))
    with open(os.path.join(results_dir, "resnet_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n모델 저장 완료: {results_dir}/resnet_model.pth")

    print("\n" + "="*50)
    print("ResNet Fine-tuned 파이프라인 실행 완료.")
    print(f"Best Val AP: {best_ap:.4f} | 확정 임계값: 0.8")
    print("="*50)

if __name__ == "__main__":
    main()
