import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from src.eda_interactive import plot_churn_by_category_st, compute_churn_by_quantile_bins, compute_churn_by_bins_equal_width,plot_churn_by_bins_bar_st,plot_churn_by_bins_line_st

# 경로 설정
ROOT_DIR = Path(__file__).resolve().parents[1]
EDA_DATA_DIR = ROOT_DIR / "data" / "preprocessed"

@st.cache_data
def load_eda_summary():
    # 요약 지표 로드
    return pd.read_pickle(EDA_DATA_DIR / "eda_summary.pkl")

@st.cache_data
def load_tab_data(file_name):
    # 각 탭에 필요한 요약 데이터 로드
    return pd.read_pickle(EDA_DATA_DIR / file_name)

def run_eda():
    # 0. 요약 데이터
    summary = load_eda_summary()

    st.title("📊 데이터 심층 인사이트 (EDA)")
    st.markdown("미리 계산된 데이터로 인사이트를 빠르게 확인하세요.")
    st.markdown("---")

    # 1. 상단 요약 지표
    c1, c2, c3 = st.columns(3)
    c1.metric("분석 대상 유저", f"{summary['total_users']:,} 명")
    c2.metric("평균 이탈률", f"{summary['churn_rate']:.1f}%")
    c3.metric("평균 청취 시간", f"{summary['avg_secs']:,.0f}초")

    # 2. 탭 구성
    tab1, tab2, tab3 = st.tabs(["🔍 핵심 변수 영향력", "🎧 사용 패턴 격차", "💳데이터 시각화"])

    with tab1:
        st.markdown("### 🔍 **모델이 주목한 이탈 핵심 요인 (SHAP)**")
        st.info("AI 모델이 유저의 이탈을 예측할 때 어떤 변수에 가장 큰 비중을 두었는지 보여줍니다.")

        # 1. SHAP 중요도 데이터 로드
        try:
            df_shap = load_tab_data("top_5_shap_features.pkl")
            
            # 시각화를 위해 데이터 정렬 (중요도 높은 순)
            df_shap = df_shap.sort_values(by='importance', ascending=True)

            # 2. Plotly 수평 바 차트 생성
            fig_shap = px.bar(
                df_shap,
                x='importance',
                y='feature',
                orientation='h',
                title="Top Feature Importance (SHAP Value)",
                labels={'importance': '평균 영향력 (Mean |SHAP Value|)', 'feature': '변수명'},
                color='importance',
                color_continuous_scale='Reds'
            )

            # 레이아웃 미세 조정
            fig_shap.update_layout(
                showlegend=False,
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis={'categoryorder': 'total ascending'}
            )

            # 차트 출력
            st.plotly_chart(fig_shap, use_container_width=True)

            # 3. 인사이트 요약
            st.markdown("#### **📌 분석 결과 해석**")
            top_1 = df_shap.iloc[-1]['feature']
            top_2 = df_shap.iloc[-2]['feature']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.success(f"**1순위 핵심 지표: {top_1}**\n\n이 지표의 변화가 유저 이탈 예측에 가장 결정적인 역할을 합니다.")
            with col_b:
                st.success(f"**2순위 핵심 지표: {top_2}**\n\n해당 수치가 특정 임계치를 넘을 경우 이탈 위험군으로 분류될 가능성이 높습니다.")

        except FileNotFoundError:
            st.warning("SHAP 결과 파일(top_5_shap_features.pkl)을 찾을 수 없습니다. 분석 스크립트를 먼저 실행해주세요.")

    with tab2:
        st.markdown("### **이탈자 vs 유지자: 청취 분포 비교**")
        # 미리 샘플링된 가벼운 데이터 로드
        df_sample = load_tab_data("eda_box_plot.pkl")
        
        fig_box = px.box(df_sample, x='is_churn', y='total_secs_mean', color='is_churn',
                         labels={'is_churn': '이탈 여부', 'total_secs_mean': '평균 청취 시간(초)'})
        st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        plt.rcParams["figure.figsize"] = (10, 4)
        plt.rcParams["axes.grid"] = True

        TARGET = "is_churn"

        df = load_tab_data("kkbox_data.pkl")
        st.markdown("### **데이터 시각화**")
        st.caption("버튼을 눌러 선택한 변수 기준으로 이탈률 그래프를 생성합니다. (matplotlib)")


        st.subheader("1) 카테고리 변수별 이탈률 (Bar)")

        # 후보 컬럼: 범주형/오브젝트/카테고리만 자동 후보로
        cat_candidates = ['gender','age_group','registered_via']


        if len(cat_candidates) == 0:
            st.info("범주형 컬럼(object/category)이 없어 카테고리 그래프 후보가 없습니다.")
        else:
            cat_col = st.selectbox("컬럼 선택", cat_candidates, index=0)
            top_n = st.slider("상위 N개만 표시", 5, 50, 20, step=5)
            min_n = st.number_input("최소 표본수(min_n) 필터", min_value=1, value=100, step=50)
            sort_by = st.radio("정렬 기준", ["churn", "n"], horizontal=True,
                            format_func=lambda x: "이탈률 높은 순" if x == "churn" else "표본 많은 순")

            run_cat = st.button("📊 카테고리 그래프 생성", use_container_width=True)

            if run_cat:
                fig, g = plot_churn_by_category_st(
                    df=df,
                    col=cat_col,
                    top_n=int(top_n),
                    min_n=int(min_n),
                    sort_by=sort_by
                )
                if fig is not None:
                    st.pyplot(fig, clear_figure=True)

        st.markdown("---")
        st.subheader("2) 수치형 변수 bin별 이탈률 (Line/Bar)")

        num_candidates = ['total_paid','total_secs_sum']

        if len(num_candidates) == 0:
            st.info("수치형 컬럼(number)이 없어 bin 그래프 후보가 없습니다.")
        else:
            num_col = st.selectbox("수치형 컬럼 선택", num_candidates, index=0)

            bin_mode = st.radio(
                "bin 방식",
                ["Quantile(qcut)", "Equal-width(등폭)"],
                horizontal=True
            )

            chart_type = st.radio("차트 타입", ["Line", "Bar"], horizontal=True)

            if bin_mode == "Quantile(qcut)":
                q = st.slider("q (분위수 bin 개수)", 4, 20, 10)
                run_num = st.button("📈 수치형(bin) 그래프 생성", use_container_width=True)

                if run_num:
                    g = compute_churn_by_quantile_bins(df, num_col, q=int(q))
                    title = f"Churn rate by {num_col} quantile bins (q={q})"
                    fig = plot_churn_by_bins_line_st(g, title) if chart_type == "Line" else plot_churn_by_bins_bar_st(g, title)
                    if fig is not None:
                        st.pyplot(fig, clear_figure=True)



            else:
                # 등폭(특히 auto_renew_rate 같은 0~1 비율 변수에 적합)
                width = st.select_slider("등폭 width", options=[0.05, 0.1, 0.2, 0.25], value=0.2)
                run_num = st.button("📈 수치형(등폭) 그래프 생성", use_container_width=True)

                if run_num:
                    g = compute_churn_by_bins_equal_width(df, num_col, width=float(width))
                    title = f"Churn rate by {num_col} equal-width bins (width={width})"
                    fig = plot_churn_by_bins_line_st(g, title) if chart_type == "Line" else plot_churn_by_bins_bar_st(g, title)
                    if fig is not None:
                        st.pyplot(fig, clear_figure=True)