import streamlit as st
import PIL
from funcs import load_css , linegaro , linesero , load_font , apply_custom_font
import webbrowser  
from css import hard_type_css


from intro import intro
from data_info import data_info
from transformer_info import transformer_info
from base_model import base_model


st.set_page_config(page_title='AI말평 경진대회 속성 기반 감정 분석', page_icon='🧊' , layout='wide')
# 외부 CSS 불러오기
load_css('./style.css')
font_path = "./fonts/AppleSDGothicNeoB.ttf"
encoded_font = load_font(font_path)

apply_custom_font(encoded_font, 'AppleSDGothicNeoB')
hard_type_css()
page_select = {
    "팀 소개 및 목표": intro,  # 괄호 없이 함수 참조 전달
    "데이터 들여다보기": data_info,
    "Transformer 에 대하여": transformer_info,
    "베이스 모델/코드 설명": base_model
}

# 사이드바에서 페이지 선택
st.sidebar.markdown("<p style='color: #efebe7; font-size: 25px;'>Select a page</p>", unsafe_allow_html=True)
selected_page = st.sidebar.radio("❕", page_select.keys(), key="value")
page_select[selected_page]()




