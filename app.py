import streamlit as st
import sys
import os
from pathlib import Path

# 1. 경로 설정 (Path Configuration)
# 프로젝트의 최상위 루트 폴더를 시스템 경로에 등록하여 
# app/ 폴더나 src/ 폴더 내의 모듈을 어디서든 참조할 수 있게 합니다.
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # sk25_2nd_project 폴더

# 시스템 경로의 최상단에 루트와 app 폴더 추가
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

def main():
    # 2. 실행 대상 파일 지정
    # main.py에서 app_main.py로 이름을 변경한 부분을 반영했습니다.
    main_file_path = ROOT / "app" / "main.py"

    # 파일 존재 여부 확인
    if not main_file_path.exists():
        st.error(f"❌ 실행할 파일을 찾을 수 없습니다: {main_file_path}")
        st.info("app/ 폴더 안에 'main.py' 파일이 있는지 확인해 주세요.")
        return

    # 3. 대시보드 로직 실행 (Execution)
    # exec를 통해 app_main.py의 내용을 현재의 전역 공간(globals)에서 실행합니다.
    try:
        with open(main_file_path, "r", encoding="utf-8") as f:
            exec(f.read(), globals())
    except Exception as e:
        st.error(f"⚠️ 대시보드 실행 중 오류가 발생했습니다: {e}")
        # 상세 에러 로그 출력 (디버깅용)
        st.exception(e)

if __name__ == "__main__":
    main()