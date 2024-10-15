import os 
import pandas as pd
import streamlit as st
import base64
import numpy as np
# css
def load_css(file_name):
    with open(file_name ,encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 줄 가로,세로
def linegaro():
    st.markdown(
        """
        <div style="border-top: 1px solid #d2c5d3; width: 95%;"></div>
        """,
        unsafe_allow_html=True)
def linesero():
    st.markdown(
        """
        <div style="display: flex; align-items: center; border-right: 1px solid #D4BDAC; height: 100%;"></div>
        """,
        unsafe_allow_html=True)
# TTF 파일을 base64로 인코딩하는 함수
def load_font(ttf_path):
    with open(ttf_path, "rb") as font_file:
        encoded_font = base64.b64encode(font_file.read()).decode("utf-8")
    return encoded_font

# CSS를 사용하여 폰트 적용
def apply_custom_font(encoded_font, font_name):
    custom_css = f"""
    <style>
    @font-face {{
        font-family: '{font_name}';
        src: url(data:fonts/ttf;base64,{encoded_font}) format('truetype');
    }}
    html, body, [class*="css"]  {{
        font-family: '{font_name}';
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)