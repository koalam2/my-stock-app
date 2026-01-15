# ... (기존 import 문들) ...
import json  # 이 줄이 꼭 필요합니다! 맨 위에 없다면 추가해주세요.

# ... (기존 코드 생략) ...

# 구글 시트 연결 설정 (수정된 버전)
@st.cache_resource
def init_connection():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # [수정] secrets에 인증 정보가 있는지 확인하고, 없으면 안내 메시지 출력
    if "gcp_json" in st.secrets:
        # 방법 1: JSON 문자열을 통째로 붙여넣은 경우 (추천)
        creds_dict = json.loads(st.secrets["gcp_json"])
    elif "gcp_service_account" in st.secrets:
        # 방법 2: 기존처럼 하나씩 키를 설정한 경우 (백업용)
        creds_dict = dict(st.secrets["gcp_service_account"])
    else:
        # 인증 정보가 없는 경우 에러 대신 안내 메시지 표시
        st.error("🚨 구글 시트 연동을 위한 인증 키(Secrets)를 찾을 수 없습니다.")
        st.markdown("""
        **해결 방법:**
        1. **내 컴퓨터에서 실행 중이라면:** 프로젝트 폴더에 `.streamlit/secrets.toml` 파일을 만들고 키를 저장하세요.
        2. **웹(Streamlit Cloud)에서 실행 중이라면:** 설정(Settings) > Secrets 메뉴에 키를 붙여넣으세요.
        """)
        st.stop() # 앱 실행을 여기서 멈춤
        
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client

# ... (나머지 코드는 그대로) ...
