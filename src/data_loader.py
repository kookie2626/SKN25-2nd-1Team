import pandas as pd
import os

def load_data(filepath):
    """
    CSV, PKL, 또는 Parquet 파일로부터 데이터를 로드합니다.
    """
    if not os.path.exists(filepath):
        # 파일이 없으면 data/ 폴더에서 다시 확인 (fallback)
        alt_path = os.path.join("data", os.path.basename(filepath))
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    
    print(f"{filepath}에서 데이터를 로드하는 중...")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.pkl'):
        df = pd.read_pickle(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError("지원되지 않는 형식입니다. .csv, .pkl, 또는 .parquet을 사용하세요.")
        
    print(f"데이터 로드 완료: {df.shape}")
    return df
