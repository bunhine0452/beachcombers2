import streamlit as st

def hard_type_css():
    body_css = """
        <style>
        /* 지정된 셀렉터에 둥근 모서리 적용 */
        #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1jicfl2.ea3mdgi5 > div {
            border-radius: 30px;
            border: 2px solid #fffcf2;
            padding: 25px;
            background-color: #fffcf2;
            box-shadow: 4px 4px 24px #252422;
            color: rgb(0, 0, 0);
        }
        </style>
    """

    # CSS 삽입
    st.markdown(body_css, unsafe_allow_html=True)
