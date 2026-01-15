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
    
    # [수정] secrets에 'gcp_json'이라는 키로 전체 JSON이 들어있는지 확인
    if "gcp_json" in st.secrets:
        # 방법 1: JSON 문자열을 통째로 붙여넣은 경우 (추천)
        creds_dict = json.loads(st.secrets["gcp_json"])
    else:
        # 방법 2: 기존처럼 하나씩 키를 설정한 경우 (백업용)
        creds_dict = dict(st.secrets["gcp_service_account"])
        
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client

# ... (나머지 코드는 그대로) ...
```

*깃허브(GitHub)에 가서 `app.py` 파일을 위 내용으로 수정(Commit)해 주세요.*

---

### 2단계: Streamlit Secrets 설정 변경

이제 Streamlit Cloud 설정 화면으로 가서, 내용을 싹 지우고 **아래 양식대로** 다시 붙여넣으세요.

1.  Streamlit Cloud > App Settings > **Secrets** 클릭.
2.  기존 내용을 지우고 아래 내용을 복사해서 넣습니다.
3.  **`"""` (따옴표 3개)** 사이에 메모장 내용을 통째로 붙여넣는 것이 핵심입니다!

```toml
# 1. 구글 시트 주소
sheet_url = "https://docs.google.com/spreadsheets/d/여기에_구글시트_주소를_넣으세요/edit"

# 2. 인증 키 (아래 따옴표 3개 사이에 메모장 내용을 통째로 붙여넣으세요!)
gcp_json = """
{
  "type": "service_account",
  "project_id": "stockwise-1234",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----...",
  "client_email": "...",
  "client_id": "...",
  "auth_uri": "...",
  "token_uri": "...",
  "auth_provider_x509_cert_url": "...",
  "client_x509_cert_url": "..."
}
"""
