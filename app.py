###
# 05. Streamlit is 🔥
# 오늘의 강의: 풀스택 GPT: #7.0부터 ~ #7.10까지
# 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
# 파일 업로드, 채팅 기록, 그리고 ConversationBufferMemory를 통한 대화 기억 기능을 구현합니다.
# st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
###

# LangChain 관련 imports - 각 컴포넌트의 역할을 이해하는 것이 중요
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # 프롬프트 템플릿 생성용
from langchain.document_loaders import UnstructuredFileLoader  # 다양한 파일 형식 로드
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings  # 임베딩 생성 및 캐싱
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough  # LCEL 체인 구성요소
from langchain.storage import LocalFileStore  # 로컬 파일 기반 캐시 저장소
from langchain.text_splitter import CharacterTextSplitter  # 텍스트를 청크로 분할
from langchain.vectorstores.faiss import FAISS  # 벡터 검색을 위한 FAISS 벡터스토어
from langchain.chat_models import ChatOpenAI  # OpenAI 채팅 모델
from langchain.callbacks.base import BaseCallbackHandler  # 콜백 핸들러 베이스 클래스
from langchain.memory import ConversationBufferMemory  # 대화 기록 메모리
import streamlit as st  # Streamlit 웹 앱 프레임워크
import os  # 파일 시스템 작업용

# Streamlit 페이지 설정 - 브라우저 탭에 표시될 제목과 아이콘 설정
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


# 스트리밍 응답을 구현하기 위한 커스텀 콜백 핸들러 (메모리 통합)
# LLM이 생성하는 각 토큰을 실시간으로 화면에 표시하고 LangChain 메모리에 저장
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, question=""):
        self.message = ""  # 누적될 메시지를 저장하는 변수
        self.question = question  # 현재 질문 저장

    def on_llm_start(self, *args, **kwargs):
        # LLM이 응답 생성을 시작할 때 호출
        # st.empty()로 빈 컨테이너를 생성하여 나중에 내용을 업데이트
        self.message_box = st.empty()
        self.message = ""  # 메시지 초기화

    def on_llm_end(self, *args, **kwargs):
        # LLM이 응답 생성을 완료했을 때 호출
        # Streamlit 세션 상태에 저장
        save_message(self.message, "ai")
        # LangChain 메모리에 대화 저장
        if self.question and self.message:
            memory.save_context(
                {"input": self.question},      # 사용자 입력
                {"output": self.message}       # AI 응답
            )

    def on_llm_new_token(self, token, *args, **kwargs):
        # 새로운 토큰(글자)이 생성될 때마다 호출
        self.message += token  # 기존 메시지에 새 토큰 추가
        self.message_box.markdown(self.message)  # 화면의 메시지 박스 업데이트


# ChatOpenAI 모델은 이제 각 함수에서 필요시 동적으로 생성
# (세션의 API 키를 사용하여 생성)


# ConversationBufferMemory 초기화 - LangChain 메모리 시스템
# return_messages: True로 설정하여 HumanMessage/AIMessage 객체로 저장 (채팅 모델 호환)
# memory_key: 프롬프트 템플릿의 변수명과 일치해야 함
memory = ConversationBufferMemory(
    return_messages=True,      # 메시지 형태 반환
    memory_key="chat_history"  # 프롬프트 변수명
)


# @st.cache_data 데코레이터: 비용이 많이 드는 연산을 캐싱
# 동일한 파일에 대해서는 함수를 재실행하지 않고 캐시된 결과 반환
# show_spinner: 처리 중 표시할 메시지
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, api_key):
    """
    파일을 임베딩하고 벡터 스토어를 생성합니다.
    
    Parameters:
    - file: 업로드된 파일 객체
    - api_key: OpenAI API 키 (캐시 무효화를 위해 매개변수로 전달)
    
    Returns:
    - retriever: FAISS 검색기
    - docs: 문서 청크 리스트  
    - file_info: 파일 처리 정보 딕셔너리
    """
    # API 키 유효성 사전 검증
    if not api_key or not api_key.startswith('sk-'):
        raise ValueError("유효하지 않은 OpenAI API 키입니다. 'sk-'로 시작하는 키를 입력해주세요.")
    # 업로드된 파일을 로컬에 저장하는 과정
    file_content = file.read()  # 파일 내용을 바이너리로 읽기
    file_size_kb = len(file_content) / 1024  # KB 단위로 파일 크기 계산
    
    # 디렉토리가 없으면 생성 (Streamlit Cloud 호환)
    os.makedirs("./.cache/files", exist_ok=True)
    os.makedirs(f"./.cache/embeddings/{file.name}", exist_ok=True)
    
    file_path = f"./.cache/files/{file.name}"  # 저장할 경로 설정
    # 파일을 바이너리 쓰기 모드로 열어 저장
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 임베딩 캐시를 위한 로컬 파일 저장소 설정
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    # 문서 크기에 따른 적응형 청킹 전략
    if file_size_kb <= 10:  # 10KB 이하 - 작은 문서
        chunk_size, chunk_overlap = 2000, 200
    elif file_size_kb <= 50:  # 10-50KB - 중간 문서 (25KB 소설 포함)
        chunk_size, chunk_overlap = 4000, 400
    elif file_size_kb <= 200:  # 50-200KB - 큰 문서
        chunk_size, chunk_overlap = 6000, 600
    else:  # 200KB 초과 - 매우 큰 문서
        chunk_size, chunk_overlap = 8000, 800
    
    # 적응형 텍스트 분할기 설정
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # UnstructuredFileLoader: PDF, TXT, DOCX 등 다양한 형식 지원
    loader = UnstructuredFileLoader(file_path)
    # 문서를 로드하고 설정한 splitter로 분할
    docs = loader.load_and_split(text_splitter=splitter)
    
    # OpenAI 임베딩 모델 초기화 (명시적 API 키 사용)
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        # 캐시 기능이 추가된 임베딩 생성 - 동일한 텍스트는 재계산하지 않음
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    except Exception as e:
        st.error(f"❌ OpenAI API 인증 실패: {str(e)}")
        st.error("🔑 API 키를 확인하고 다시 시도해주세요.")
        raise e
    
    # FAISS 벡터스토어 생성 - 빠른 유사도 검색을 위한 벡터 DB
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    
    # 문서 크기에 따른 적응형 검색 전략
    num_chunks = len(docs)
    if num_chunks <= 3:  # 적은 청크 - 모든 청크 검색
        k = num_chunks
    elif num_chunks <= 10:  # 중간 청크 - 대부분 검색
        k = max(8, num_chunks - 2)
    else:  # 많은 청크 - 상위 15개 검색
        k = 15
    
    # 검색기(retriever) 생성 - 적응형 검색
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    
    # 파일 정보 반환 (UI 표시용)
    file_info = {
        "size_kb": round(file_size_kb, 1),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": num_chunks,
        "search_k": k
    }
    
    return retriever, docs, file_info


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
    return "\n\n".join(document.page_content for document in docs)


# 문서 전체 요약을 위한 함수
def summarize_document(docs):
    """
    문서 전체를 요약하여 사용자가 전체 내용을 파악할 수 있도록 함
    Map-Reduce 방식으로 긴 문서를 효율적으로 요약
    """
    if not docs:
        return "요약할 문서가 없습니다."
    
    # 요약을 위한 프롬프트 템플릿
    summary_prompt = ChatPromptTemplate.from_template(
        """다음 문서 내용을 한국어로 간결하게 요약해 주세요. 
        주요 내용과 핵심 포인트를 포함해서 3-5개의 문단으로 정리해 주세요.
        
        문서 내용:
        {text}
        
        요약:"""
    )
    
    # 각 청크별 요약 생성
    chunk_summaries = []
    # API 키가 세션에 있을 때만 요약 가능
    if not st.session_state.get("api_key"):
        return "❌ API Key가 설정되지 않았습니다. 먼저 API Key를 입력해주세요."
    
    summary_llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=st.session_state["api_key"]
    )
    
    # 청크를 그룹화하여 처리 (5개씩)
    for i in range(0, len(docs), 5):
        chunk_group = docs[i:i+5]
        combined_text = "\n\n".join([doc.page_content for doc in chunk_group])
        
        try:
            summary_chain = summary_prompt | summary_llm
            summary = summary_chain.invoke({"text": combined_text})
            chunk_summaries.append(summary.content)
        except Exception as e:
            chunk_summaries.append(f"요약 중 오류 발생: {str(e)}")
    
    # 모든 청크 요약을 최종 요약으로 통합
    if len(chunk_summaries) > 1:
        final_text = "\n\n".join(chunk_summaries)
        final_summary_chain = summary_prompt | summary_llm
        try:
            final_summary = final_summary_chain.invoke({"text": final_text})
            return final_summary.content
        except Exception as e:
            return f"최종 요약 중 오류 발생: {str(e)}\n\n개별 요약들:\n" + "\n\n".join(chunk_summaries)
    else:
        return chunk_summaries[0] if chunk_summaries else "요약을 생성할 수 없습니다."


# LangChain 메모리에서 대화 히스토리를 로드하는 헬퍼 함수
# LCEL 체인에서 사용되며, 체인 실행 시마다 자동으로 호출됨
def load_memory(_):
    """
    메모리에서 대화 히스토리를 로드하는 헬퍼 함수
    체인 입력 딕셔너리는 사용하지 않음 (_로 표시)
    """
    memory_vars = memory.load_memory_variables({})
    return memory_vars["chat_history"]


# RAG(Retrieval Augmented Generation)를 위한 프롬프트 템플릿 (메모리 포함)
# system: AI의 행동 지침 설정
# MessagesPlaceholder: 대화 기록이 삽입될 위치
# human: 사용자의 질문
# {context}, {chat_history}, {question}은 런타임에 실제 값으로 치환됨
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        당신은 도움이 되는 AI 어시스턴트입니다. 주어진 컨텍스트만을 사용하여 질문에 답변해주세요.
        컨텍스트에 없는 정보는 "모르겠습니다"라고 답변하고, 추측하지 마세요.
        이전 대화 내용을 기억하고 일관성 있는 답변을 제공하세요.
        
        컨텍스트:
        {context}
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),  # 대화 기록이 삽입될 위치
    ("human", "{question}")  # 사용자 질문이 전달될 플레이스홀더
])


# 페이지 제목 설정
st.title("DocumentGPT")

# 환영 메시지와 사용 안내 (파일이 업로드되지 않았을 때만 표시)
if "file_uploaded" not in st.session_state or not st.session_state.file_uploaded:
    if not st.session_state.get("api_key"):
        st.markdown(
            """
            ## 🚀 시작하기
            
            1. 먼저 사이드바에서 **OpenAI API Key**를 입력해주세요
            2. API Key 발급: [OpenAI Platform](https://platform.openai.com/api-keys)
            3. API Key 입력 후 문서를 업로드하세요
            """
        )
    else:
        st.markdown(
            """
            ## 📄 환영합니다!

            사이드바에서 문서를 업로드하여 시작하세요.

            업로드한 문서에 대해 무엇이든 질문해보세요!
            """
        )

# 사이드바에 API 키 입력 및 파일 업로더 위젯 생성
with st.sidebar:
    st.markdown("## 🔑 API Configuration")
    
    # API 키 입력 (세션 상태에 저장하여 유지)
    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.get("api_key", ""),
        type="password",
        placeholder="sk-...",
        help="https://platform.openai.com/api-keys 에서 발급받을 수 있습니다."
    )
    
    # API 키 저장 및 모델 초기화
    if api_key:
        st.session_state["api_key"] = api_key
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        
        # API 키 유효성 검증
        try:
            # 간단한 테스트로 API 키 확인
            test_llm = ChatOpenAI(
                temperature=0.1,
                openai_api_key=api_key
            )
            st.success("✅ API Key 설정 완료!")
        except Exception as e:
            st.error(f"❌ API Key 오류: 유효하지 않은 API Key입니다.")
            api_key = None
            if "api_key" in st.session_state:
                del st.session_state["api_key"]
    else:
        if "api_key" in st.session_state:
            del st.session_state["api_key"]
    
    st.markdown("---")
    st.markdown("## 📄 Document Upload")
    
    # API 키가 없으면 파일 업로드 비활성화
    if not api_key:
        st.warning("⚠️ 먼저 OpenAI API Key를 입력해주세요!")
        file = None
    else:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],  # 허용된 파일 형식 지정
        )
    
    # 문서가 업로드된 경우 요약 버튼 표시
    if file:
        st.markdown("---")
        st.markdown("## Document Analysis")
        if st.button("📄 문서 전체 요약"):
            st.session_state["show_summary"] = True
        
        if st.button("🔍 검색 정보"):
            st.session_state["show_search_info"] = True
    
    st.markdown("---")
    st.markdown("## About")
    st.markdown("DocumentGPT with Memory - Enhanced with conversation history")
    st.markdown("**적응형 전략**: 문서 크기에 따른 최적 청킹으로 전체 문서 완벽 분석!")
    st.markdown("• 10KB ↓: 2K토큰 • 25KB: 4K토큰 • 200KB: 6K토큰 • 200KB+: 8K토큰")
    st.markdown("[GitHub Repository](https://github.com/your-repo)")  # 7강 요구사항: 깃허브 링크

# 파일이 업로드된 경우의 메인 로직
if file:
    # 파일 업로드 상태 설정 (환영 메시지 숨기기 위함)
    st.session_state.file_uploaded = True
    
    # 파일을 임베딩하고 검색기, 문서 청크, 파일 정보 생성 (캐시됨)
    try:
        retriever, docs, file_info = embed_file(file, st.session_state["api_key"])
    except Exception as e:
        st.error("💥 문서 처리 중 오류가 발생했습니다.")
        st.error("🔧 해결 방법:")
        st.error("1️⃣ API 키가 올바른지 확인")
        st.error("2️⃣ OpenAI 계정에 충분한 크레딧이 있는지 확인")
        st.error("3️⃣ 인터넷 연결 상태 확인")
        st.stop()
    
    # 문서 정보 표시
    st.success(f"📄 **{file.name}** ({file_info['size_kb']}KB) 업로드 완료!")
    
    # 문서 요약 표시
    if st.session_state.get("show_summary", False):
        st.markdown("## 📄 문서 전체 요약")
        with st.spinner("문서를 요약하는 중..."):
            summary = summarize_document(docs)
            st.markdown(summary)
            st.markdown("---")
        st.session_state["show_summary"] = False
    
    # 검색 정보 표시
    if st.session_state.get("show_search_info", False):
        st.markdown("## 🔍 문서 검색 설정 정보")
        st.info(f"""
        **문서 처리 정보:**
        - 파일 크기: {file_info['size_kb']}KB
        - 총 청크 수: {file_info['num_chunks']}개
        - 청크 크기: {file_info['chunk_size']:,} 토큰 (적응형)
        - 청크 중복: {file_info['chunk_overlap']:,} 토큰
        - 검색 방식: 단순 유사도 검색
        - 검색 청크 수: {file_info['search_k']}개
        
        **적응형 전략:**
        - 문서 크기에 따른 최적 청크 크기 자동 조정
        - 청크 수에 따른 검색 범위 최적화
        - 문서 전체 커버리지 보장으로 결말까지 정확한 검색
        """)
        st.session_state["show_search_info"] = False
    
    # AI 준비 완료 메시지 (저장하지 않음)
    send_message("준비완료! 문서에 대해 질문해 주세요!", "ai", save=False)
    # 이전 대화 히스토리 표시
    paint_history()

    # 채팅 입력 위젯 생성
    message = st.chat_input("Ask anything about your file...")
    
    if message:
        # 사용자 메시지 표시 및 저장
        send_message(message, "human")
        
        # 현재 질문에 대한 콜백 핸들러 생성 (메모리 저장 포함)
        callback_handler = ChatCallbackHandler(question=message)
        
        # API 키 확인 후 LLM 인스턴스 생성
        if not st.session_state.get("api_key"):
            st.error("❌ API Key가 설정되지 않았습니다!")
            st.stop()
        
        # 질문별 LLM 인스턴스 생성 (콜백 핸들러 포함)
        llm_with_callback = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[callback_handler],
            openai_api_key=st.session_state["api_key"],  # 세션의 API 키 사용
        )
        
        # LCEL(LangChain Expression Language) 체인 구성 (메모리 포함)
        # 1. RunnablePassthrough.assign으로 동적 변수 주입
        # 2. chat_history: 메모리에서 대화 기록 로드
        # 3. context: 질문을 기반으로 관련 문서 검색 후 포맷
        # 4. prompt: 프롬프트 템플릿에 변수들 적용
        # 5. llm: 생성된 프롬프트를 LLM에 전달하여 응답 생성
        chain = (
            RunnablePassthrough.assign(
                chat_history=load_memory,    # 메모리에서 대화 기록 로드
                context=lambda x: format_docs(retriever.get_relevant_documents(x["question"]))
            )
            | prompt
            | llm_with_callback
        )
        # AI 메시지 컨텍스트에서 체인 실행
        # 스트리밍 응답이 콜백 핸들러를 통해 실시간 표시됨
        with st.chat_message("ai"):
            chain.invoke({"question": message})

# 파일이 없는 경우 대화 히스토리 초기화
else:
    # 파일 업로드 상태 해제 (환영 메시지 다시 표시)
    st.session_state.file_uploaded = False
    st.session_state["messages"] = []
    # LangChain 메모리도 초기화
    memory.clear()