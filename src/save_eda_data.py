import pandas as pd
from pathlib import Path
import numpy as np


def prepare_eda_data():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    DATA_PATH = ROOT_DIR / "data" / "preprocessed" / "kkbox_data.pkl"
    SAVE_PATH = ROOT_DIR / "data" / "preprocessed"
    
    print("데이터 로딩 중...")
    df = pd.read_pickle(DATA_PATH)

    # 0. 요약 지표 데이터
    summary_stats = {
        'total_users': len(df),
        'churn_rate': df['is_churn'].mean() * 100,
        'avg_secs': df['total_secs_mean'].mean()
    }
    pd.to_pickle(summary_stats, SAVE_PATH / "eda_summary.pkl")


    # 2. Tab 2용: 박스 플롯용 데이터 
    df_sample = df[['is_churn', 'total_secs_mean']]
    df_sample.to_pickle(SAVE_PATH / "eda_box_plot.pkl")

    # 3. Tab 3용: 자동결제별 이탈률 (Groupby 결과만 저장)
    bins = np.arange(0, 1.01, 0.2)

    df["auto_renew_bin"] = pd.cut(df["auto_renew_rate"], bins=bins, include_lowest=True)

    churn_dist = (
        df.groupby("auto_renew_bin")["is_churn"]
        .sum()                      # 이탈자 수
        .reset_index(name="churners")
    )

    churn_dist["auto_renew_bin"] = churn_dist["auto_renew_bin"].astype(str)
    churn_dist.to_pickle(SAVE_PATH / "eda_churn_auto.pkl")

    print("EDA 전용 요약 데이터 저장 완료!")

if __name__ == "__main__":
    prepare_eda_data()