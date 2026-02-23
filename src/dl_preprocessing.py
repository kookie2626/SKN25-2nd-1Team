import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class KKBoxDataset(Dataset):
    """
    KKBox 데이터를 PyTorch Dataset으로 변환합니다.
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values).view(-1, 1) if isinstance(y, (pd.Series, pd.DataFrame)) else torch.FloatTensor(y).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_dl_data(X, y, batch_size=1024, use_weighted_sampler=True):
    """
    데이터를 스케일링하고 DataLoader로 변환합니다.
    WeightedRandomSampler로 클래스 불균형(이탈 10:1)을 해결합니다.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    train_dataset = KKBoxDataset(X_train_scaled, y_train)
    val_dataset = KKBoxDataset(X_val_scaled, y_val)
    
    if use_weighted_sampler:
        # 이탈(1) 클래스를 더 자주 샘플링하여 클래스 불균형 해소
        y_train_np = np.array(y_train)
        class_counts = np.bincount(y_train_np.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train_np.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.FloatTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        print(f"WeightedRandomSampler 적용 (이탈:{class_counts[1]}, 정상:{class_counts[0]})")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler, X_train.shape[1]
