import streamlit as st
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()


def get_api_key():
    # 환경 변수 가져오기
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        st.error("🚨 OpenAI API 키가 설정되지 않았습니다! .env 파일을 확인하세요.")
    else:
        st.success("✅ OpenAI API 키가 정상적으로 로드되었습니다.")

@st.cache_resource
def load_json_file():
    # JSON 데이터 로드 (과실비율 데이터)
    with open("accident_data.json", "r", encoding="utf-8") as file:
        accident_data = json.load(file)

    # JSON 데이터를 langchain Document로 변환
    documents = []
    for case in accident_data:
        doc = Document(
            page_content=f"사고유형: {case['사고유형']}\n자동차 A: {case['자동차 A']}\n자동차 B: {case['자동차 B']}\n사고 설명: {case['사고 설명']}\n과실 비율: {case['과실 비율']}"
        )
        documents.append(doc)

    # OpenAI Embeddings 사용하여 문서 벡터화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


@st.cache_data
def get_selected_docs(user_input):
    selected_docs = retriever.get_relevant_documents(user_input)
    return selected_docs


# Streamlit UI
st.title("🚗 교통사고 과실비율 AI 챗봇")
st.write("사고 상황을 설명하면 AI가 과실비율을 알려드립니다.")

user_input = st.text_area("✏️ 사고 상황을 입력하세요", "")

get_api_key()

# 문서 가져오기
vectorstore = load_json_file()

# 검색기(Retriever) 생성 (유사도 높은 문서만 사용)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

selected_docs = get_selected_docs(user_input)

# PromptTemplate 설정
prompt = PromptTemplate.from_template("""
당신은 사고 상황을 설명하면 과실비율을 알려주는 챗봇입니다.
질문을 보고 참고 문서의 사고 설명과 가장 유사한 사고유형을 찾고, 먼저 사고 설명을 말해줍니다.
그다음 과실비율을 알려줍니다.
유사한 사례가 없다면 "유사한 사례가 없습니다. 좀 더 구체적인 상황 설명이 필요합니다." 라고 말하세요.
대답은 한국어로 해주세요.

# 참고 문서: {documents}
                                      
# 질문: {question}
"""
)

# 언어모델(LLM) 생성
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

# 체인(Chain) 생성
chain = (
    prompt
    | llm
    | StrOutputParser()     # 답을 항상 문자열로 출력
)

if st.button("🚀 과실비율 분석하기"):
    result = chain.invoke({"documents": selected_docs, "question": user_input})
    st.markdown(f"📌 AI 과실비율 결과: \n\n{result}")