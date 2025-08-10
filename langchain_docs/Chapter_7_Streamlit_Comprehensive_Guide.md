# Chapter 7: Streamlit 완벽 가이드

## 목차
1. [소개](#소개)
2. [Streamlit 기본 개념](#streamlit-기본-개념)
3. [데이터 흐름과 상태 관리](#데이터-흐름과-상태-관리)
4. [멀티페이지 애플리케이션](#멀티페이지-애플리케이션)
5. [채팅 인터페이스 구현](#채팅-인터페이스-구현)
6. [파일 업로드와 처리](#파일-업로드와-처리)
7. [대화 히스토리 관리](#대화-히스토리-관리)
8. [LangChain 체인 통합](#langchain-체인-통합)
9. [스트리밍 응답 구현](#스트리밍-응답-구현)
10. [실습 코드 예제](#실습-코드-예제)

## 소개

Streamlit은 Python으로 데이터 애플리케이션을 빠르게 만들 수 있는 오픈소스 라이브러리입니다. 이 챕터에서는 Streamlit과 LangChain을 결합하여 DocumentGPT라는 챗봇을 구축하는 과정을 다룹니다.

## Streamlit 기본 개념

### 1. st.write() - 만능 출력 함수

`st.write()`는 Streamlit의 "Swiss army knife"로, 거의 모든 것을 화면에 출력할 수 있습니다.

```python
import streamlit as st
from langchain.prompts import PromptTemplate

# 텍스트 출력
st.write("Hello, Streamlit!")

# 변수 출력
prompt_template = PromptTemplate.from_template("Tell me about {topic}")
st.write(prompt_template)  # 클래스 정의와 문서화까지 표시
```

### 2. Streamlit Magic

변수를 단독으로 작성하면 자동으로 화면에 출력됩니다.

```python
# st.write() 없이도 출력됨
"This is magic!"
prompt_template  # 변수명만 작성해도 출력
```

### 3. 주요 위젯들

```python
# 선택 박스
model = st.selectbox(
    "Choose your model",
    ["GPT-3", "GPT-4"]
)

# 텍스트 입력
name = st.text_input("What is your name?")

# 슬라이더
temperature = st.slider("Temperature", min_value=0.1, max_value=1.0)

# 체크박스
is_expensive = st.checkbox("Not cheap")
```

## 데이터 흐름과 상태 관리

### 핵심 개념: 전체 재실행

**Streamlit의 가장 중요한 특징**: 데이터가 변경될 때마다 **전체 Python 파일이 위에서 아래로 다시 실행**됩니다.

```python
from datetime import datetime

# 시간을 표시하면 데이터 변경 시마다 업데이트됨
st.title(datetime.now().strftime("%H:%M:%S"))

model = st.selectbox("Model", ["GPT-3", "GPT-4"])
st.write(f"You chose: {model}")

# 사용자가 선택을 변경할 때마다 전체 코드가 재실행되어
# 시간이 업데이트되고 선택 결과가 표시됨
```

### 조건부 위젯 표시

```python
is_expensive = st.checkbox("Not cheap")

if is_expensive:
    st.write("Expensive mode")
else:
    st.write("Cheap mode")
    # 다른 위젯들을 조건부로 표시
    st.slider("Temperature", 0.1, 1.0)
```

## 멀티페이지 애플리케이션

### 1. 페이지 설정

```python
# Home.py (메인 파일)
st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)
```

### 2. 페이지 구조

```
project/
├── Home.py              # 메인 페이지
└── pages/              # 이 폴더명은 반드시 "pages"여야 함
    ├── 01_DocumentGPT.py
    ├── 02_PrivateGPT.py
    └── 03_QuizGPT.py
```

파일명 앞의 숫자(01, 02, 03)는 정렬용이며 UI에는 표시되지 않습니다.

### 3. 사이드바 활용

```python
# with 패턴을 사용한 사이드바 구성
with st.sidebar:
    st.title("Sidebar Title")
    st.text_input("Input in sidebar")
    
# 또는
st.sidebar.title("Sidebar Title")
st.sidebar.text_input("Input in sidebar")
```

## 채팅 인터페이스 구현

### 1. 채팅 메시지 표시

```python
# 사용자 메시지
with st.chat_message("human"):
    st.write("Hello, AI!")

# AI 메시지
with st.chat_message("ai"):
    st.write("Hello, Human!")
```

### 2. 채팅 입력

```python
message = st.chat_input("Send a message to the AI")
if message:
    # 메시지 처리
    st.write(f"You said: {message}")
```

### 3. 상태 표시

```python
# 작업 진행 상태 표시
with st.status("Embedding file...", expanded=True) as status:
    time.sleep(2)
    st.write("Getting the file")
    time.sleep(2)
    st.write("Embedding the file")
    time.sleep(2)
    st.write("Caching the file")
    status.update(label="Complete!", state="complete", expanded=False)
```

## 파일 업로드와 처리

### 파일 업로더 구현

```python
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    # 파일 내용 읽기
    file_content = file.read()
    
    # 파일 저장
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
```

### 임베딩 함수 with 캐싱

```python
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    
    # 디렉토리 생성
    os.makedirs("./.cache/files", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # 임베딩 캐시 디렉토리
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    # 텍스트 분할
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    
    # 문서 로드 및 분할
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    
    # 임베딩 생성 및 캐싱
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    # 벡터 스토어 생성
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    
    return retriever
```

**`@st.cache_data` 데코레이터의 중요성**:
- 입력(파일)이 동일하면 함수를 재실행하지 않고 캐시된 결과 반환
- 비용이 많이 드는 연산(임베딩 생성)을 한 번만 수행

## 대화 히스토리 관리

### Session State 활용

```python
# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 메시지 저장 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

# 메시지 전송 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 대화 히스토리 표시
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,  # 이미 저장된 메시지는 다시 저장하지 않음
        )
```

### 파일 변경 시 대화 초기화

```python
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    # ... 채팅 로직
else:
    # 파일이 없으면 대화 초기화
    st.session_state["messages"] = []
```

## LangChain 체인 통합

### 문서 포맷 함수

```python
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
```

### 프롬프트 템플릿

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. 
            If you don't know the answer just say you don't know. 
            DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)
```

### LCEL 체인 구성

```python
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,  # 스트리밍 활성화
    callbacks=[ChatCallbackHandler()],
)

# 체인 구성
chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

# 체인 실행
if message:
    send_message(message, "human")
    with st.chat_message("ai"):
        chain.invoke(message)
```

## 스트리밍 응답 구현

### 커스텀 콜백 핸들러

```python
from langchain.callbacks.base import BaseCallbackHandler

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        # LLM 시작 시 빈 박스 생성
        self.message_box = st.empty()
    
    def on_llm_end(self, *args, **kwargs):
        # LLM 종료 시 메시지 저장
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        # 새 토큰 수신 시 메시지 업데이트
        self.message += token
        self.message_box.markdown(self.message)
```

**작동 원리**:
1. `on_llm_start`: AI 메시지 영역에 빈 박스 생성
2. `on_llm_new_token`: 각 토큰을 받을 때마다 박스 내용 업데이트
3. `on_llm_end`: 완성된 메시지를 세션 상태에 저장

## 실습 코드 예제

### 완전한 DocumentGPT 구현

```python
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    
    # 디렉토리 생성
    os.makedirs("./.cache/files", exist_ok=True)
    
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. 
            If you don't know the answer just say you don't know. 
            DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []
```

## 중요 포인트 정리

1. **데이터 흐름**: Streamlit은 데이터 변경 시 전체 코드를 재실행
2. **Session State**: 재실행 간에 데이터를 유지하는 유일한 방법
3. **캐싱**: `@st.cache_data`로 비용이 많이 드는 연산 최적화
4. **스트리밍**: 콜백 핸들러로 실시간 응답 구현
5. **파일 구조**: `pages/` 폴더로 멀티페이지 앱 구성

## 추가 개선 사항

### 보안 강화
```python
import re

def sanitize_filename(filename):
    # 파일명에서 위험한 문자 제거
    return re.sub(r'[^\w\s.-]', '', filename)
```

### 환경 변수 관리
`.streamlit/secrets.toml` 파일 생성:
```toml
OPENAI_API_KEY = "your-api-key"
```

**.gitignore에 추가**:
```
.streamlit/
.cache/
```

## 코드 챌린지

DocumentGPT에 메모리 기능을 추가해보세요:
- ConversationBufferMemory 활용
- 이전 대화 컨텍스트 유지
- 관련 질문에 대한 연속적인 답변 가능

이것으로 Streamlit과 LangChain을 활용한 DocumentGPT 구축 가이드를 마칩니다.