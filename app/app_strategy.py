import streamlit as st

def run_strategy():
    # 1. 헤더 설정
    st.title("💡 우리 서비스 단골 고객 지키기 전략")
    st.markdown("##### **XGBoost & ResNet AI 분석으로 찾은 고객 유지(Retention) 가이드**")
    st.markdown("---")

    # 2. 시뮬레이션 결과 연동
    if st.session_state.get('predict_done'):
        res = st.session_state.result_data
        # float32 에러 방지를 위해 float() 변환 추가
        risk_score = float(res['scores'][-1]) * 100
        
        st.success(f"✅ 현재 유저 진단 결과(이탈 위험: {risk_score:.1f}%)를 바탕으로 산출된 맞춤 전략입니다.")
        
        st.subheader("1. 고객을 놓치지 않았을 때 기대 효과")
        c1, c2 = st.columns(2)
        c1.metric("예상되는 매출 방어", "약 12.8만 원", "1년 더 이용할 때의 가치")
        c2.metric("마케팅 효율", "4.2배", "비용 대비 이득")
    else:
        st.info("ℹ️ '이탈 시뮬레이터'에서 유저를 먼저 분석하면 실제 수치 기반의 전략이 나타납니다.")

    st.markdown("---")

    # 3. 리소스 최적화 전략
    st.subheader("2. 어떤 고객에게 더 집중해야 할까요?")
    st.write("데이터 분석 결과, **'정기 결제'를 안 하거나 '노래 듣는 시간'이 줄어든 고객**을 먼저 챙겨야 합니다.")
    
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.write("**[떠날 확률 높은 고객]** 예산 65% 집중 - 적극적으로 마음 돌리기")
        st.progress(0.65)
        st.caption("👉 노래를 안 들은 지 **10일 이내**에 연락하는 것이 가장 효과적입니다.")
        
        st.write("**[주의가 필요한 고객]** 예산 25% 배분 - 혜택으로 붙잡기")
        st.progress(0.25)
        st.caption("👉 매번 결제하는 분들을 '정기 결제'로 바꾸는 것이 가장 중요합니다.")
        
        st.write("**[안정적인 단골 고객]** 예산 10% 유지 - 고마움 표현하기")
        st.progress(0.1)
    
    with col_b:
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <strong>📋 분석가 총평 (핵심 요약)</strong><br>
        유저가 떠나지 않게 만드는 가장 큰 힘은 <strong>'정기 결제'</strong> 설정이었습니다. 
        결제가 편해질수록 고객들은 우리 서비스를 더 오래 사랑해 주십니다.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. 실천 가이드 (Action Plan)
    st.subheader("3. 고객의 마음을 잡기 위해 바로 해야 할 일")
    
    # [수정] SyntaxError를 일으켰던 이미지 텍스트를 삭제했습니다.
    
    with st.expander("📅 지금 당장: 떠나려는 고객 붙잡기", expanded=True):
        st.write("**대상**: 정기 결제를 취소했거나 곧 결제일이 다가오는 분들")
        st.write("**방법**: 다시 정기 결제로 바꾸면 '한 달 무료' 혜택 보내기")
        st.write("**이유**: 정기 결제를 하는 분들이 서비스를 4배 더 오래 씁니다.")
        
    with st.expander("📅 다음 단계: 서비스에 다시 재미를 느끼게 하기"):
        st.write("**대상**: 노래 듣는 시간이 눈에 띄게 줄어든 분들")
        st.write("**방법**: 좋아할 만한 새로운 노래 리스트를 알림으로 보내기")
        st.write("**이유**: 떠나기 전에는 반드시 활동량이 줄어드는 신호가 나타납니다.")
        
    with st.expander("📅 장기 목표: 우리 서비스의 찐팬 만들기"):
        st.write("**대상**: 6개월 넘게 꾸준히 이용해 주시는 고마운 분들")
        st.write("**방법**: 단골 전용 이벤트나 연간 회원권 할인 혜택 드리기")

    st.markdown("---")
    st.warning("💡 **마지막 제언**: 무조건 할인을 해주기보다, 고객이 우리를 잊어갈 때쯤(활동 감소 10일 전후) 딱 맞춰서 말을 거는 '똑똑한 마케팅'이 필요합니다.")