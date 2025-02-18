import streamlit as st
from css import hard_type_css
import PyPDF2
import io   
import pandas as pd
def data_info():
    a,b,c,d = st.columns([0.05, 0.1, 0.01 ,1])
    with a:
        st.write('')
        st.image('https://kli.korean.go.kr/images/logo_kor_2.png')
    with b:
        st.write('')
        st.write('')
        st.image('https://kli.korean.go.kr/images/logo_kor_header.png', width=400)
    with d:
        st.title('데이터 들여다보기')
    st.markdown('#')  

    
    frame1,frame2 = st.columns([1,1])
    with frame1:
        st.markdown('#####')
        st.image('./img/data_form1.png', width=1000)
    with frame2:
        st.markdown('''
                    #### 데이터의 구조는 이와 같은 방식으로 이루어져 있습니다.
                    - ID : 리뷰 문장의 고유 번호
                    - sentence_form : 리뷰 문장
                    - annotation : 리뷰 문장에 대한 속성 및 감정 분석 결과
                        - < annotation 속 구조>
                            - 브랜드#가격 = 카테고리 분류
                            - 애플 = 카테고리가 분류된 단어,구절
                            - 0,2 = 해당 단어 또는 구절의 인덱스 시작/끝 번호 
                            - negative = 해당 단어 또는 구절에 대한 리뷰의 감정 분석 결과
                    ''')
    st.markdown('#####')
    frame1,frame2 = st.columns([1,1])
    with frame1:
        st.markdown('#### 개체/속성(ADC)')
        st.image('./img/data_form2.png', width=1000)
        st.markdown('#### 감정(ASC)')
        char = {'positive' : '긍정', 'negative' : '부정', 'neutral' : '중립'}
        df = pd.DataFrame([char])  # 데이터프레임으로 변환
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.markdown('#### 데이터 분포 시각화')
        st.image('./img/plot1.png', width=1300)
        st.image('./img/plot2.png', width=1300)
    with frame2:
        st.markdown('#### 감정에 대한 개체/속성 정보')
        st.markdown('''
                    개체는 총 4개의 분류로 나누어져있으며, 개체에 대한 속성은 총 7가지의 소분류로 나누어져있습니다.
                    이론 상 총 28가지의 카테고리가 존재하지만 실제 학습 데이터엔 25개의 카테고리만 존재하고 있습니다.                    
                    ''')
        # 기존 counts 딕셔너리를 데이터프레임으로 변환
        counts = {
            'Train(학습)' : 2999,
            'Dev(검증)' : 2792,
            'Test(예측)' : 2126
        }
        df = pd.DataFrame([counts], index=['문장수'])  # 데이터프레임으로 변환
        st.dataframe(df, use_container_width=True)
        st.markdown('''
                    데이터 세트의 문장수는 총 이렇게 이루어져 있습니다 여기서 알 수 있는 
                    사실은 데이터의 개수가 현저히 적다는 것이며, 데이터의 분포는 특정 개체/속성에 대해 다소 극심한 데이터 불균형을 이루고 있었으며,
                    더군다나, 감정에 대한 정보는 압도적으로 긍정에 대한 데이터가 많았으며
                    리뷰 속 이모티콘과 중복문자(ㅋㅋㅋ, ....) 와 같은 예측에 방해가 될 요소들이 포함되어 있어 이러한 요소들을 제거하는 작업이 필요하다는 것을 알 수 있었습니다.
                    ''')
        st.markdown('#### 문제 해결에 관한 고찰')
        st.markdown('데이터를 보았을때와 프로젝트에 필요한 기초 지식을 토대로 미리 여러 가설들을 설정하여 프로젝트의 목적에 달성하기 위해 노력했습니다.')
        pro , line ,sol = st.columns([1,0.1,1])
        with pro:
            st.markdown('##### 잠재적인 문제, 고려사항')
            st.markdown('''
                    - 문장
                        - 문장속 불용어(이모티콘, 중복단어)
                        - 애매한 문장
                    ''')
            st.write('')
            st.markdown('''
                        - 학습 데이터
                            - 속성과 감정 데이터 불균형
                            - 물리적인 데이터 개수 부족
                        ''')
        with line:
            st.markdown(
            """
            <div style="border-right: 1px solid  #d2c5d3; height: 300px;"></div>
            """,
            unsafe_allow_html=True)
        with sol:
            st.markdown('##### 해결 방안')
            st.markdown('''
                        - 문장
                            - 불용어 제거 또는 일부 변경
                            - 애매한 문장 제거
                        ''')
            st.write('')
            st.markdown('''
                        - 학습 데이터
                            - 데이터 증강
                        ''')
        st.markdown('''
                    ##### *데이터 증강 방법*
                    *본 과제에 유의사항엔 학습데이터를 언어 모델(LLM)을 통해 증강을 해도 된다고 명시되어 있었습니다.*
                    *따라서 저희 팀은 언어 모델을 통해 데이터를 증강하는 방법을 선택하였습니다.*
                    *또한 다른 팀들의 데이터 증강 방법인*
                    - *역번역*
                    - *반대 감정 문장 생성*
                    
                    *등 여러 의견들을 종합하여 추가로 도전해볼 계획을 세웠습니다.*  
                    ''')
