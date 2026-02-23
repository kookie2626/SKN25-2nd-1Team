import streamlit as st
import sys
from pathlib import Path

# 1. 파일 경로 설정 (상대 경로 및 임포트 에러 방지)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 최상위 루트 폴더 기준
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 2. 페이지 모듈 임포트
from app.app_home import run_home
from app.app_eda import run_eda
from app.app_predict import run_predict
from app.app_strategy import run_strategy

def main():
    # --- [페이지 설정] ---
    st.set_page_config(page_title="KeepTune Dashboard", layout="wide", page_icon="🎧")

    # --- [🎨 깔끔한 배너형 버튼 스타일 CSS] ---
    st.markdown("""
        <style>
        /* 버튼을 배너처럼 보이게 하는 커스텀 스타일 */
        div.stButton > button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: transparent;
            color: #31333F;
            border: 1px solid #f0f2f6;
            text-align: left;
            padding-left: 20px;
            font-size: 1rem;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            background-color: #f0f2f6;
            border-color: #f0f2f6;
            color: #ff4b4b;
        }
        /* 현재 선택된 페이지 버튼 강조 (Streamlit 기본 버튼 한계로 호버 위주 설정) */
        </style>
    """, unsafe_allow_html=True)

    # --- [사이드바 구성] ---
    st.sidebar.title("🎧 KeepTune")
    st.sidebar.markdown("---")

    # [근혁님 로직] 페이지 상태 관리
    if 'page' not in st.session_state: 
        st.session_state.page = '대시보드'

    st.sidebar.subheader("메뉴")

    # [근혁님 로직] 버튼형 메뉴 (글자를 클릭하는 배너 느낌)
    if st.sidebar.button("🏠 대시보드", use_container_width=True): 
        st.session_state.page = '대시보드'
    if st.sidebar.button("🔍 유저 행동 인사이트", use_container_width=True): 
        st.session_state.page = '유저 행동 인사이트'
    if st.sidebar.button("🔮 이탈 위험도 시뮬레이터", use_container_width=True): 
        st.session_state.page = '이탈 위험도 시뮬레이터'
    if st.sidebar.button("🚀 비즈니스 전략", use_container_width=True): 
        st.session_state.page = '비즈니스 전략'

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 KeepTune. All rights reserved.")
    st.sidebar.caption("Hybrid Engine: XGBoost + ResNet")

    # --- [근혁님 로직] 페이지 전환 로직 ---
    if st.session_state.page == '대시보드': 
        run_home()
    elif st.session_state.page == '유저 행동 인사이트': 
        run_eda()
    elif st.session_state.page == '이탈 위험도 시뮬레이터': 
        run_predict()
    elif st.session_state.page == '비즈니스 전략': 
        run_strategy()

if __name__ == "__main__":
    main()