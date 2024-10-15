import streamlit as st
from css import hard_type_css
import PyPDF2
import io   

def p1():
    st.title('info.pdf 상세 분석')

    # PDF 파일 업로드
    uploaded_file = st.file_uploader("info.pdf 파일을 업로드하세요", type="pdf")
    
    if uploaded_file is not None:
        # PDF 파일 읽기
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        
        # 페이지 수 표시
        num_pages = len(pdf_reader.pages)
        st.write(f"총 페이지 수: {num_pages}")
        
        # 목차 표시 (가정: 첫 페이지에 목차가 있다고 가정)
        st.subheader("목차")
        toc = pdf_reader.pages[0].extract_text()
        st.text(toc)
        
        # 각 페이지 내용 표시
        for i in range(num_pages):
            st.subheader(f"페이지 {i+1}")
            page = pdf_reader.pages[i]
            text = page.extract_text()
            st.text(text)
            
            # 이미지 추출 (만약 있다면)
            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        st.image(xObject[obj]._data)
        
        # 키워드 검색 기능
        st.subheader("키워드 검색")
        keyword = st.text_input("검색할 키워드를 입력하세요")
        if keyword:
            results = []
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                if keyword.lower() in text.lower():
                    results.append(f"페이지 {i+1}: {text[:200]}...")
            
            if results:
                st.write("검색 결과:")
                for result in results:
                    st.write(result)
            else:
                st.write("검색 결과가 없습니다.")
        
        # PDF 메타데이터 표시
        st.subheader("PDF 메타데이터")
        metadata = pdf_reader.metadata
        for key, value in metadata.items():
            st.write(f"{key}: {value}")

    else:
        st.write("PDF 파일을 업로드해주세요.")
