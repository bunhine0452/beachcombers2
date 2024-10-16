import streamlit as st
import PIL
from funcs import load_css , linegaro , linesero , load_font , apply_custom_font
import webbrowser  
from css import hard_type_css

def intro():
    a,b,c,d = st.columns([0.05, 0.1, 0.01 ,1])
    with a:
        st.write('')
        st.image('https://kli.korean.go.kr/images/logo_kor_2.png')
    with b:
        st.write('')
        st.write('')
        st.image('https://kli.korean.go.kr/images/logo_kor_header.png', width=400)
    with d:
        st.title('Project: AI말평 경진대회 속성 기반 감정 분석')
        st.write('*본 페이지는 PC웹 브라우저에 최적화 되었습니다.*')
    st.markdown('#')

    st.markdown('## 프로젝트 소개')
    linegaro()
    a , b , c = st.columns([1, 0.05 ,1])
    with a:
        st.markdown('### START')

        st.markdown('''    
                    
                    이 과제는 '2021년 말뭉치 감성 분석 및 연구'라는 국립국어원의 사업에서 구축된 데이터를 사용하여, '제품' 분야에서 감성 분석을 수행하는 것입니다. 감성 분석은 소비자가 제품에 대해 긍정적이거나 부정적인 감정을 느끼는지를 알아내는 작업입니다.

                    이 과제의 주요 목표는 두 가지입니다:
                    1. **속성 범주 탐지(Aspect Category Detection, ACD)**: 주어진 문장에서 어떤 속성이 분석의 대상이 되는지를 찾아내는 작업입니다. 이때 속성은 '개체#속성' 형식으로 표기됩니다. 예를 들어, '제품 전체#디자인'이라고 하면, 이는 '제품 전체'라는 개체의 '디자인'이라는 속성에 대해 논하고 있는 문장임을 의미합니다.
                    
                    2. **속성 감성 분류(Aspect Sentiment Classification, ASC)**: 속성 범주 탐지로 찾아낸 속성에 대해 소비자가 긍정적인 감정을 느꼈는지, 부정적인 감정을 느꼈는지를 분석하는 작업입니다. 예를 들어, '제품 전체#디자인'이 주석된 문장에서 필자가 그 디자인에 대해 긍정적이었는지 부정적이었는지를 판단합니다.

                    말뭉치 데이터는 모두 화자의 주관적인 의견이 반영된 문장으로 구성되어 있으며, 그 문장에서 어떤 개체와 속성에 대한 평가가 이루어졌는지를 탐지하는 것이 이 과제의 핵심입니다. 예를 들어, '본품#디자인', '브랜드#인지도' 등의 속성 범주가 문장에서 추출됩니다.''')
        if st.button('AI말평 과제 페이지'):
            webbrowser.open('https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=57')

        st.markdown('### 팀원 소개')
        tab1,tab2,tab3,tab4 = st.tabs([
            '이정화',
            '신상길',
            '신민석',
            '김현빈'
        ])
        with tab1:
            st.markdown('''
            팀원 이름을 눌러서 역할을 확인할 수 있습니다.
            ''')
        with tab2:
            st.markdown('''
            팀원 이름을 눌러서 역할을 확인할 수 있습니다.2
            ''')
        with tab3:
            st.markdown('''
            팀원 이름을 눌러서 역할을 확인할 수 있습니다.3
            ''')
    with b:
        st.markdown(
        """
        <div style="border-right: 1px solid  #d2c5d3; height: 520px;"></div>
        """,
        unsafe_allow_html=True)
    with c:
        c1,c2 = st.columns([1,1])
        with c1:
            st.markdown('### DATA')
            st.markdown('''- 리뷰 문장 ''')
        with c2:    
            st.markdown('### TASK')
            st.markdown('''- 카테고리 속 다중분류(ACD) ''')
            st.markdown('''- 감정 분석(ASC) ''')
        st.markdown('### RESULT(예시)')
        st.markdown('##### input ')
        st.code(''' 
            "sentence_form": "애플 브랜드는 비싸지만, 아이폰은 품질이 좋다"
                ''')
        st.markdown('##### output')
        st.code('''
            "annotation": 
                [["브랜드#가격", ["애플", 0, 2], "negative"], 
                ["본품#품질", ["아이폰", 15, 17], "positive"]]
        ''')
        st.markdown('### BASE_Knowledge')
        st.write('''프로젝트를 진행하기 앞서, 과제에서 기초적으로 요구하는 BERT모델에 대한 이해와 Transformers 라이브러리를 사용하여 모델을 구축하는 방법에 대해 학습하였습니다.''')
        st.write('''또한, 데이터를 증강시키기 위한 LLM 모델을 사용법을 응용하여 맞는 데이터를 생성하는 방법에 대해 학습이 필요했고 여러 전처리 기법들을 사용하였습니다.''')
        st.markdown('##### 사용된 모듈 및 환경')
        st.image('https://www.python.org/static/img/python-logo@2x.png', width=300)
        st.markdown('''- 모델링: transformers, datasets, torch, sklearn, nltk, konlpy, kobert, korean-roberta''')
        st.markdown('''- 데이터 증강: chatGPT(GPTs)''')
        st.markdown('#')  



