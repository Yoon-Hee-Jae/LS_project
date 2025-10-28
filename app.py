# VSCode를 쓰신다면, 상단 메뉴에서 [Terminal] → [New Terminal] 클릭

# 아래 명령어 한 번만 입력:
# streamlit run app.py

# 이후에는 터미널을 닫지 말고, 코드를 수정하면
# Streamlit이 자동 새로고침(Hot Reload) 해줍니다!
# → 저장(ctrl+s)만 해도 웹이 자동 업데이트됩니다.


import streamlit as st

st.title("🎉 나의 첫 Streamlit 앱")
st.write("안녕하세요! 여기는 Streamlit으로 만든 첫 웹앱입니다 😄")

