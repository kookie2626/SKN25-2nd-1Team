import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True

TARGET = "is_churn"


# =========================
# Plot 함수들 (Streamlit용)
# =========================
def compute_churn_by_category(df: pd.DataFrame, col: str, min_n: int = 100):
    g = (df.groupby(col)[TARGET]
           .agg(churn_rate="mean", n="size")
           .reset_index())
    g["churn_rate"] *= 100
    g = g[g["n"] >= min_n].copy()
    return g


def plot_churn_by_category_st(df: pd.DataFrame, col: str, top_n: int = 20, min_n: int = 100, sort_by: str = "churn"):
    g = compute_churn_by_category(df, col, min_n=min_n)

    if g.empty:
        st.warning(f"'{col}'에서 min_n={min_n} 조건을 만족하는 카테고리가 없어요.")
        return None, g

    if sort_by == "churn":
        g = g.sort_values("churn_rate", ascending=False)
    else:
        g = g.sort_values("n", ascending=False)

    g_plot = g.head(top_n).copy()

    fig, ax = plt.subplots()
    ax.bar(g_plot[col].astype(str), g_plot["churn_rate"])
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(f"Churn rate by {col} (top {top_n}, n≥{min_n})")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    return fig, g


def compute_churn_by_bins_equal_width(df: pd.DataFrame, col: str, width: float = 0.2):
    """
    auto_renew_rate 같은 '0~1 비율' 변수에 추천:
    0.2 단위 등폭 bin
    """
    tmp = df[[col, TARGET]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)

    # 등폭 bin 만들기 (0~1 가정)
    start = 0.0
    end = 1.0
    bins = np.arange(start, end + width, width)
    # 혹시 값이 1.0을 조금 넘는 경우 포함
    tmp["_val_clip"] = tmp[col].clip(lower=start, upper=end)

    tmp["bin"] = pd.cut(tmp["_val_clip"], bins=bins, include_lowest=True, right=True)

    g = (tmp.groupby("bin")[TARGET]
            .agg(churn_rate="mean", n="size")
            .reset_index())
    g["churn_rate"] *= 100
    return g


def compute_churn_by_quantile_bins(df: pd.DataFrame, col: str, q: int = 10):
    tmp = df[[col, TARGET]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)

    # 동일값 많으면 qcut이 bin 개수 줄일 수 있음
    tmp["bin"] = pd.qcut(tmp[col], q=q, duplicates="drop")

    g = (tmp.groupby("bin")[TARGET]
            .agg(churn_rate="mean", n="size")
            .reset_index())
    g["churn_rate"] *= 100
    return g


def plot_churn_by_bins_line_st(g: pd.DataFrame, title: str):
    if g.empty:
        st.warning("표시할 bin 결과가 없어요.")
        return None

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(g)), g["churn_rate"], marker="o")
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels(g["bin"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_churn_by_bins_bar_st(g: pd.DataFrame, title: str):
    if g.empty:
        st.warning("표시할 bin 결과가 없어요.")
        return None

    fig, ax = plt.subplots()
    ax.bar(g["bin"].astype(str), g["churn_rate"])
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig
