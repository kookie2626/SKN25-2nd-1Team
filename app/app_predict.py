import streamlit as st
import pandas as pd
import plotly.express as px
from src.predict import predict_churn  # 수정된 예측 엔진 임포트

def run_predict():
    st.title("🔮 KeepTune AI : 이탈 방어 시뮬레이터")
    st.markdown("##### **XGBoost & ResNet 하이브리드 진단 리포트**")
    st.markdown("---")

    # 1. 세션 상태 관리 (에러 방지용 초기화)
    if 'predict_done' not in st.session_state:
        st.session_state.predict_done = False
    if 'result_data' not in st.session_state:
        st.session_state.result_data = None

    # 2. 시나리오 설정 섹션 (세밀한 슬라이더 모드)
    with st.container():
        st.subheader("👤 시뮬레이션 대상 설정")
        col1, col2 = st.columns(2)
        
        with col1:
            # 💳 결제 설정
            auto_label = st.radio("💳 정기 결제(자동 갱신) 설정", ["활성 (구독 중)", "해지 (만료 예정)"], horizontal=True)
            auto_renew = 1.0 if "활성" in auto_label else 0.0
            
            st.write("") 
            # 🎧 청취 시간: 1분 단위 정밀 조절
            total_mins = st.slider(
                "🎧 일평균 노래 청취 시간 (분)", 
                min_value=0, max_value=720, value=30, step=1, 
                help="좌우로 밀거나, 키보드 방향키를 사용하면 1분 단위로 조절됩니다."
            )
            total_secs = float(total_mins * 60) # 모델 학습 단위인 '초'로 변환
            
        with col2:
            # ⚠️ 해지 비율: 1% 단위 정밀 조절
            cancel_rate = st.slider("⚠️ 과거 서비스 해지 시도 비율", 0.0, 1.0, 0.1, step=0.01)
            
            st.write("") 
            # 💰 결제 횟수
            txn_cnt = st.slider("💰 누적 결제 횟수 (회)", 1, 100, 10, step=1)

    input_data = {
        'is_auto_renew': auto_renew,
        'total_secs_mean': total_secs,
        'is_cancel': cancel_rate,
        'payment_plan_days': 30.0,
        'txn_cnt': float(txn_cnt),
        'total_paid': float(txn_cnt) * 30.0,  # 가정: 평균 결제액
        'total_secs_sum': total_secs * float(txn_cnt),  # 가정: 총 청취 시간
        'auto_renew_rate': auto_renew,
        'cancel_rate': cancel_rate,
    }

    # 3. AI 진단 실행 (XGBoost & ResNet 하이브리드)
    if st.button("🚀 하이브리드 AI 분석 시작", use_container_width=True):
        try:
            with st.spinner('XGBoost(트리) & ResNet(딥러닝) 엔진 가동 중...'):
                # 결과값은 (p_xgb, p_resnet, final_score) 튜플로 반환됨
                results = predict_churn(input_data)
                
                # [해결] 결과를 강제로 float 타입으로 변환하여 저장 (Streamlit 에러 방지)
                st.session_state.result_data = {
                    'scores': [float(v) for v in results], 
                    'input': input_data
                }
                st.session_state.predict_done = True
                st.toast("✅ 하이브리드 분석이 완료되었습니다!")
                
        except Exception as e:
            st.error("⚠️ 분석 엔진 가동 중 일시적인 오류가 발생했습니다.")
            st.info("results/ 폴더에 모델 파일들이 있는지 확인해주세요.")
            print(f"Debug Error: {e}")

    # 4. 분석 결과 출력
    if st.session_state.predict_done:
        res = st.session_state.result_data
        p_xgb, p_resnet, final_score = res['scores']
        risk_score = final_score * 100

        # 종합 지표 대시보드
        st.subheader("📊 종합 이탈 위험 진단")
        m1, m2, m3 = st.columns(3)
        m1.metric("최종 이탈 위험도", f"{risk_score:.1f}%")
        m2.metric("XGBoost 판정", f"{p_xgb*100:.1f}%", help="수치 기반 분석")
        m3.metric("ResNet 판정", f"{p_resnet*100:.1f}%", help="패턴 기반 분석")

        # [해결] float 변환을 통해 Progress Value 에러 원천 차단
        st.progress(float(final_score))

        # 5. 판단 근거 시각화 (Feature Importance)
        st.write("")
        st.subheader("🔍 AI의 판단 근거 (Feature Importance)")
        
        features = ['정기 결제 설정', '청취 시간', '해지 시도 이력', '결제 횟수']
        contributions = [
            -35 if auto_renew == 1 else 42.5, # EDA 가중치 42.5 반영
            -15 if total_mins > 45 else 12,
            25 if cancel_rate > 0.3 else -5,
            -10 if txn_cnt > 15 else 5
        ]
        
        fig = px.bar(
            x=contributions, y=features, orientation='h',
            color=contributions, color_continuous_scale='RdYlGn_r',
            labels={'x': '이탈 위험 기여도 (+: 위험 가중, -: 안전 요인)', 'y': '핵심 데이터'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # 6. 추천 솔루션 탭
        st.subheader("📋 추천 리텐션 솔루션")
        t1, t2 = st.tabs(["💡 즉각적 대응 전략", "📈 장기 고객 관리"])

        with t1:
            if auto_renew == 0:
                st.info("**[강력 권고] 정기 결제 전환 프로모션**")
                st.write("- 이탈 요인 중 '정기 결제 미설정' 비중이 압도적으로 높습니다.")
                st.write("- **솔루션**: 정기 결제 재설정 시 '1개월 무료' 혜택 즉시 노출")
            elif risk_score > 60:
                st.warning("**[주의] 활동성 복구 캠페인**")
                st.write("- 청취 시간이 임계치 미만입니다. 취향 맞춤 플레이리스트 알림을 발송하세요.")
            else:
                st.success("**[안정] 로열티 강화 프로그램**")
                st.write("- 현재 위험도가 낮습니다. 장기 이용 고객 전용 배지를 부여하세요.")

    st.markdown("---")
    st.caption("KeepTune v2.5 | Hybrid Ensemble (XGBoost + ResNet)")