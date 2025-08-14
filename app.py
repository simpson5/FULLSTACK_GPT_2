###
# 오늘의 강의: 풀스택 GPT: #7.0부터 ~ #7.10까지
# 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
# 파일 업로드 및 채팅 기록을 구현합니다.
# 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
###

# LangChain 관련 imports - 각 컴포넌트의 역할을 이해하는 것이 중요
from langchain.prompts import ChatPromptTemplate  # 프롬프트 템플릿 생성용
from langchain.document_loaders import UnstructuredFileLoader  # 다양한 파일 형식 로드
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings  # 임베딩 생성 및 캐싱
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough  # LCEL 체인 구성요소
from langchain.storage import LocalFileStore  # 로컬 파일 기반 캐시 저장소
from langchain.text_splitter import CharacterTextSplitter  # 텍스트를 청크로 분할
from langchain.vectorstores.faiss import FAISS  # 벡터 검색을 위한 FAISS 벡터스토어
from langchain.chat_models import ChatOpenAI  # OpenAI 채팅 모델
from langchain.callbacks.base import BaseCallbackHandler  # 콜백 핸들러 베이스 클래스
import streamlit as st  # Streamlit 웹 앱 프레임워크

# Streamlit 페이지 설정 - 브라우저 탭에 표시될 제목과 아이콘 설정
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


# 스트리밍 응답을 구현하기 위한 커스텀 콜백 핸들러
# LLM이 생성하는 각 토큰을 실시간으로 화면에 표시
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""  # 누적될 메시지를 저장하는 변수

    def on_llm_start(self, *args, **kwargs):
        # LLM이 응답 생성을 시작할 때 호출
        # st.empty()로 빈 컨테이너를 생성하여 나중에 내용을 업데이트
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        # LLM이 응답 생성을 완료했을 때 호출
        # 완성된 메시지를 세션 상태에 저장
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        # 새로운 토큰(글자)이 생성될 때마다 호출
        self.message += token  # 기존 메시지에 새 토큰 추가
        self.message_box.markdown(self.message)  # 화면의 메시지 박스 업데이트


# ChatOpenAI 모델 초기화
# temperature: 0.1로 설정하여 더 일관된 응답 생성 (0=결정적, 1=창의적)
# streaming: True로 설정하여 실시간 스트리밍 응답 활성화
# callbacks: 스트리밍을 처리할 커스텀 콜백 핸들러 등록
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


# @st.cache_data 데코레이터: 비용이 많이 드는 연산을 캐싱
# 동일한 파일에 대해서는 함수를 재실행하지 않고 캐시된 결과 반환
# show_spinner: 처리 중 표시할 메시지
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    # 업로드된 파일을 로컬에 저장하는 과정
    file_content = file.read()  # 파일 내용을 바이너리로 읽기
    file_path = f"./.cache/files/{file.name}"  # 저장할 경로 설정
    # 파일을 바이너리 쓰기 모드로 열어 저장
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 임베딩 캐시를 위한 로컬 파일 저장소 설정
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    # 텍스트 분할기 설정 - tiktoken 인코더 사용
    # separator: 줄바꿈 문자를 기준으로 분할
    # chunk_size: 각 청크의 최대 토큰 수 (600)
    # chunk_overlap: 청크 간 중복 토큰 수 (100) - 컨텍스트 연속성 유지
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    # UnstructuredFileLoader: PDF, TXT, DOCX 등 다양한 형식 지원
    loader = UnstructuredFileLoader(file_path)
    # 문서를 로드하고 설정한 splitter로 분할
    docs = loader.load_and_split(text_splitter=splitter)
    
    # OpenAI 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()
    # 캐시 기능이 추가된 임베딩 생성 - 동일한 텍스트는 재계산하지 않음
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    # FAISS 벡터스토어 생성 - 빠른 유사도 검색을 위한 벡터 DB
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    # 검색기(retriever) 생성 - 질문과 관련된 문서를 찾는 역할
    retriever = vectorstore.as_retriever()
    return retriever


# 메시지를 세션 상태에 저장하는 함수
# Streamlit은 재실행 시 모든 변수가 초기화되므로 st.session_state 사용
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# 메시지를 화면에 표시하고 선택적으로 저장하는 함수
def send_message(message, role, save=True):
    # st.chat_message로 역할(human/ai)에 맞는 채팅 UI 생성
    with st.chat_message(role):
        st.markdown(message)  # 마크다운 형식으로 메시지 표시
    # save=True일 때만 메시지를 세션 상태에 저장
    if save:
        save_message(message, role)


# 저장된 대화 히스토리를 화면에 다시 그리는 함수
# Streamlit이 재실행될 때마다 이전 대화를 복원
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,  # 이미 저장된 메시지이므로 다시 저장하지 않음
        )


# 검색된 문서들을 하나의 문자열로 포맷팅
# 각 문서 내용을 두 줄 띄움(\n\n)으로 구분
def format_docs(docs):
    st.write(docs)
    st.write(docs[0].page_content)
    return "\n\n".join(document.page_content for document in docs)


# RAG(Retrieval Augmented Generation)를 위한 프롬프트 템플릿
# system: AI의 행동 지침 설정
# human: 사용자의 질문
# {context}와 {question}은 런타임에 실제 값으로 치환됨
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


# 페이지 제목 설정
st.title("DocumentGPT")

# 환영 메시지와 사용 안내
st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

# 사이드바에 파일 업로더 위젯 생성
# with 구문을 사용하여 사이드바 컨텍스트 내에서 위젯 생성
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],  # 허용된 파일 형식 지정
    )

# 파일이 업로드된 경우의 메인 로직
if file:
    # 파일을 임베딩하고 검색기 생성 (캐시됨)
    retriever = embed_file(file)
    # AI 준비 완료 메시지 (저장하지 않음)
    send_message("I'm ready! Ask away!", "ai", save=False)
    # 이전 대화 히스토리 표시
    paint_history()

    # 채팅 입력 위젯 생성
    message = st.chat_input("Ask anything about your file...")
    
    if message:
        # 사용자 메시지 표시 및 저장
        send_message(message, "human")
        
        # LCEL(LangChain Expression Language) 체인 구성
        # 1. retriever가 관련 문서 검색
        # 2. format_docs가 문서를 문자열로 포맷
        # 3. RunnablePassthrough가 질문을 그대로 전달
        # 4. prompt가 컨텍스트와 질문으로 프롬프트 생성
        # 5. llm이 최종 응답 생성
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        # AI 메시지 컨텍스트에서 체인 실행
        # 스트리밍 응답이 콜백 핸들러를 통해 실시간 표시됨
        with st.chat_message("ai"):
            chain.invoke(message)

# 파일이 없는 경우 대화 히스토리 초기화
else:
    st.session_state["messages"] = []