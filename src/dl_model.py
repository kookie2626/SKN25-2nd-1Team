import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    ResNet의 핵심인 Skip Connection이 적용된 블록
    """
    def __init__(self, dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # 입력 x를 결과에 직접 더함 (Skip Connection)
        return self.relu(x + self.block(x))

class ChurnResNet(nn.Module):
    # ... (기존 ResNet 코드 유지)
    def __init__(self, input_dim, hidden_dim=256, num_blocks=5, dropout=0.2):
        super(ChurnResNet, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.first_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

class ChurnLSTM(nn.Module):
    """
    Bidirectional LSTM + Attention 기반 이탈 예측 모델
    - 양방향(Bidirectional): 시퀀스를 앞뒤로 모두 읽어 더 풍부한 패턴 학습
    - Attention: 어떤 타임스텝이 중요한지 스스로 가중치를 부여
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(ChurnLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM: 출력 차원은 hidden_dim * 2 (앞방향 + 뒤방향)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        
        # Attention: 각 타임스텝의 중요도 계산
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 최종 분류기
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 2차원 입력 -> 3차원 시퀀스로 변환 (seq_len=1)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Attention 가중치 계산: (batch, seq_len, 1) -> softmax
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # 가중합으로 컨텍스트 벡터 생성: (batch, hidden_dim * 2)
        context = (attn_weights * lstm_out).sum(dim=1)
        
        return self.fc(context)

def get_device():
    """
    사용 가능한 장치(CUDA 또는 CPU)를 반환합니다.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
