import streamlit as st

def run_home():
    # 1. 헤더 섹션 (브랜드 아이덴티티)
    st.title("🎧 KeepTune AI : 구독 이탈 방지 솔루션")
    st.markdown("#### **\"데이터로 유저의 리듬을 지키다 (Keep the Tune)\"**")
    st.write("---")

    # 2. 프로젝트 핵심 성과 (KPI)
    st.subheader("📍 프로젝트 핵심 지표 (Key Performance Indicators)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="분석 로그 규모", value="1,812만 건", delta="Big Data")
    with kpi2:
        st.metric(label="이탈 예측 정확도", value="97.7%", delta="Hybrid AI")
    with kpi3:
        st.metric(label="최적화 변수", value="24개", delta="Optimized")
    with kpi4:
        st.metric(label="기대 방어 효율", value="25.4%", delta="Target")

    st.write("") 
    
    # 3. 비즈니스 가치 제안 (비교 카드)
    col1, col2 = st.columns(2)
    with col1:
        st.error("### 📉 기존 방식의 한계")
        st.markdown("""
        - **높은 유치 비용**: 신규 고객 한 명을 모시는 데 유지 비용의 5배 이상 소요
        - **판단 지연**: 사람이 일일이 수만 명의 이탈 징후를 포착하기 불가능
        - **무분별한 할인**: 타겟 없는 혜택 발송으로 인한 수익성 악화
        """)
    with col2:
        st.success("### 🚀 KeepTune의 가치")
        st.markdown("""
        - **선제적 방어**: 97%의 정확도로 떠날 유저를 미리 찾아내어 대응
        - **데이터 인사이트**: '정기 결제'와 '청취 시간' 등 핵심 이탈 원인 분석
        - **맞춤형 가이드**: 위험 단계별로 바로 적용 가능한 마케팅 시나리오 제공
        """)

    st.write("---")

    # 4. 분석 기술 스택 (XGBoost & ResNet 하이브리드 강조)
    st.subheader("⚙️ 분석 아키텍처 및 하이브리드 엔진")
    
    tab1, tab2 = st.tabs(["📊 데이터 처리 시스템", "🧠 하이브리드 예측 모델"])
    
    with tab1:
        st.markdown("""
        - **행동 로그 분석**: 일평균 노래 청취 시간, 접속 간격 등 활동 패턴 정밀 수집
        - **결제 시그널 추적**: 정기 결제(Auto-renew) 설정 여부 및 해지 시도 데이터 결합
        - **고속 연산 처리**: **Polars** 라이브러리를 활용하여 대용량 데이터를 지연 없이 전처리
        """)

    with tab2:
        st.info("**Advanced Hybrid Ensemble: XGBoost + ResNet**")
        st.write("정형 데이터 분석의 최강자와 딥러닝의 패턴 추출 능력을 결합했습니다.")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**1. XGBoost (Gradient Boosting)**")
            st.caption("유저의 결제 이력과 활동 수치 등 정형 데이터를 정교하게 분류하여 이탈 가능성을 계산합니다.")
        with col_m2:
            st.markdown("**2. ResNet (Deep Learning)**")
            st.caption("복합적으로 얽힌 유저 행동 사이의 숨겨진 패턴을 다차원적으로 분석하여 예측의 깊이를 더합니다.")

    st.write("---")

    # 5. 기대 효과 및 비전
    st.subheader("📈 기대 효과 및 비즈니스 임팩트")
    with st.expander("현업 적용 시 예상 시나리오 확인하기"):
        st.write("""
        - **이탈 고객 방어**: '정기 결제 해지' 유저 대상 집중 케어로 연간 이탈률 약 20% 감소 기대
        - **고객 가치 극대화**: 우량 유저의 이탈을 막아 서비스의 장기적인 수익 구조 안정화
        - **똑똑한 마케팅**: 주관적인 감이 아닌 '데이터'에 기반한 마케팅 예산 최적화
        """)

    st.write("---")
    st.caption("KeepTune AI Solution Project 2026")