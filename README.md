# KKBox 음악 스트리밍 서비스 이탈 예측 프로젝트

KKBox 구독자 이탈(Churn) 여부를 예측하는 머신러닝 / 딥러닝 파이프라인입니다.  
**XGBoost (ML)** 와 **ResNet Fine-tuned (DL)** 두 모델을 확정 사용합니다.

---

## 📁 디렉토리 구조

```
kkbox_0222/
│
├── main.py                  # XGBoost 파이프라인 실행 진입점
├── dl_main.py               # ResNet 딥러닝 파이프라인 실행 진입점
├── predict.py               # 저장된 모델 호출 후 예측 실행 (재학습 불필요)
├── run_shap.py              # SHAP 분석 단독 실행 스크립트
│
├── src/
│   ├── data_loader.py       # 데이터 로드 (parquet / pkl)
│   ├── preprocessing.py     # 공통 전처리 & 파생변수 생성
│   │
│   ├── model_train.py       # XGBoost 학습 (Optuna 튜닝 포함)
│   ├── model_eval.py        # XGBoost 평가 & SHAP 시각화
│   │
│   ├── dl_preprocessing.py  # 딥러닝용 Dataset / DataLoader 생성
│   ├── dl_model.py          # ResNet 모델 클래스 정의
│   └── dl_train.py          # 딥러닝 학습 / 평가 / Fine-tuning 함수
│
├── data/
│   ├── kkbox_v3.parquet     # 파이프라인 입력 데이터 (전처리 완료본, 97만 건)
│   ├── kkbox_v3.pkl         # parquet 백업본
│   └── raw/                 # KKBox 원본 CSV 파일
│       ├── members_v3.csv
│       ├── train_v2.csv
│       ├── transactions_v2.csv
│       └── user_logs_v2.csv
│
├── results/                 # 학습 후 자동 생성되는 파일들
│   ├── xgboost_model.pkl    # XGBoost 모델 영구 보관 (main.py 실행 후)
│   ├── resnet_model.pth     # ResNet 모델 영구 보관 (dl_main.py 실행 후)
│   ├── resnet_scaler.pkl    # ResNet 스케일러 (예측 시 필수)
│   ├── confusion_matrix.png # 혼동 행렬 이미지
│   └── shap_summary.png     # SHAP 중요도 이미지
│
├── EDA.ipynb                # 탐색적 데이터 분석 노트북 (참고용 보관)
├── requirements.txt         # 의존성 패키지 목록
└── README.md
```

---

## 🚀 실행 순서

### 환경 설정
```bash
conda activate tp
pip install -r requirements.txt
```

### 데이터 준비
파이프라인 실행 전 아래 경로에 데이터 파일이 있어야 합니다:
```
data/kkbox_v3.parquet     ← 반드시 필요 (파이프라인 입력)
data/raw/*.csv            ← 원본 CSV (kkbox_v3.parquet 생성에 사용)
```

### 1. XGBoost 파이프라인
```bash
python main.py
```
- Optuna 10회 하이퍼파라미터 탐색 → 최적 모델 학습
- **확정 임계값: 0.6** (Precision ≈ 0.83, Recall ≈ 0.95)
- 학습 후 모델 저장: `results/xgboost_model.pkl`
- 시각화 저장: `results/confusion_matrix.png`, `results/shap_summary.png`

### 2. ResNet 딥러닝 파이프라인
```bash
python dl_main.py
```
- 확정 하이퍼파라미터로 ResNet 학습 (Optuna 재탐색 없음, 빠름)
- **확정 임계값: 0.8** (Precision ≈ 0.95, Recall ≈ 0.70)
- Early Stopping + LR Scheduler + Gradient Clipping 적용
- 학습 완료 후 영구 보관: `results/resnet_model.pth`, `results/resnet_scaler.pkl`
- 재탐색 시: `dl_main.py`의 Optuna 주석 블록 해제 후 실행

### 3. 저장된 모델로 예측만 실행 (재학습 불필요)
```bash
python predict.py
```
- `main.py`, `dl_main.py` 실행 후 생성된 파일을 불러와서 예측만 수행
- 두 모델의 예측 결과와 동의율(앙상블 참고)도 출력

### 4. SHAP 분석 (선택)
```bash
python run_shap.py
```

---

## 🏆 확정 모델 성능 비교

| 지표 | XGBoost | ResNet Fine-tuned |
| :--- | :---: | :---: |
| **임계값** | 0.6 | 0.8 |
| **AP Score** | 0.9522 | 0.9450 |
| **Precision** | 0.8319 | 0.9514 |
| **Recall** | 0.9452 | 0.7011 |
| **F1-Score** | 0.8847 | 0.8073 |

### 모델 선택 가이드
- **XGBoost 추천**: 이탈자를 최대한 많이 잡고 싶을 때 (Recall 우선)
- **ResNet 추천**: 정확한 이탈자만 타겟팅할 때 (Precision 우선, 마케팅 비용 최소화)

---

## 🔧 작업 이력 요약

### 데이터
- 원본: KKBox 구독자 로그 약 **97만 건**, 24개 변수
- 파생 변수 생성: 구독 기간, 자동 갱신 여부, 나이 그룹 등
- 클래스 불균형: 이탈(1) 약 7만 건, 정상(0) 약 71만 건 (약 10:1)

### ML 파이프라인 (XGBoost)
- Optuna 기반 하이퍼파라미터 자동 탐색 (10회)
- `scale_pos_weight`로 클래스 불균형 대응
- `find_optimal_threshold()`로 F1 기준 최적 임계값 탐색 후 0.6 확정

### DL 파이프라인 (ResNet)
- **모델**: `ChurnResNet` (Residual Block 5개, hidden_dim=256)
- **Fine-tuning**: Optuna로 lr + hidden_dim + num_blocks + dropout 동시 탐색
- **과적합 방지**:
  - `copy.deepcopy`로 best epoch 메모리 보존 후 복원 (파일 저장 없음)
  - Early Stopping (patience=5), Gradient Clipping (max_norm=1.0)
  - LR Scheduler (`ReduceLROnPlateau`)
- **확정 파라미터**: lr=0.01121, hidden_dim=256, num_blocks=5, dropout=0.1669
- 임계값 0.8 고정 평가

### 서브 실험 (LSTM - 참고용)
- Bidirectional LSTM + Attention 구조 실험
- WeightedRandomSampler로 클래스 불균형 대응
- AP Score: 0.9363 / 임계값 0.9 기준 F1: 0.8720
- 현재 코드에는 미포함 (필요 시 `dl_model.py`의 `ChurnLSTM` 클래스 활용)

---

## 📦 주요 의존성

| 패키지 | 버전 | 용도 |
| :--- | :---: | :--- |
| `torch` | 2.10.0 | ResNet 딥러닝 |
| `xgboost` | 3.2.0 | XGBoost 모델 |
| `optuna` | 4.7.0 | 하이퍼파라미터 튜닝 |
| `scikit-learn` | 1.8.0 | 전처리 / 평가 |
| `shap` | 0.50.0 | 모델 해석 |
| `pandas` | 2.3.3 | 데이터 처리 |
