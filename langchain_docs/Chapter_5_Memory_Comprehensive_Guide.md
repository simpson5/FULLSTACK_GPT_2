# Chapter 5: Memory 완벽 가이드

## 목차
1. [메모리 개요](#메모리-개요)
2. [ConversationBufferMemory](#conversationbuffermemory)
3. [ConversationBufferWindowMemory](#conversationbufferwindowmemory)
4. [ConversationSummaryMemory](#conversationsummarymemory)
5. [ConversationSummaryBufferMemory](#conversationsummarybuffermemory)
6. [ConversationKGMemory](#conversationkgmemory)
7. [LLMChain과의 메모리 통합](#llmchain과의-메모리-통합)
8. [Chat 기반 메모리](#chat-기반-메모리)
9. [LCEL 기반 메모리](#lcel-기반-메모리)
10. [실습 코드 예제](#실습-코드-예제)

## 메모리 개요

### 메모리가 필요한 이유

**기본 OpenAI API의 한계**:
- **Stateless**: 각 요청이 독립적이며 이전 대화를 기억하지 못함
- **컨텍스트 손실**: 사용자가 후속 질문을 해도 이전 내용을 모름
- **ChatGPT의 차이점**: ChatGPT는 메모리 기능을 통해 대화의 연속성 제공

### LangChain 메모리 시스템

LangChain은 **5가지 이상의 메모리 클래스**를 제공하며, 각각 다른 방식으로 메모리를 관리합니다:

1. **ConversationBufferMemory**: 전체 대화 저장
2. **ConversationBufferWindowMemory**: 최근 N개 메시지만 저장
3. **ConversationSummaryMemory**: 대화 요약 저장
4. **ConversationSummaryBufferMemory**: 요약 + 최근 대화 결합
5. **ConversationKGMemory**: 지식 그래프 기반 메모리

### 메모리 API 공통 구조

모든 메모리 클래스는 동일한 API를 제공합니다:

```python
# 1. 메모리 생성
memory = SomeMemory()

# 2. 대화 저장
memory.save_context(
    {"input": "사용자 입력"},
    {"output": "AI 응답"}
)

# 3. 메모리 로드
memory.load_memory_variables({})

# 4. 반환 형식 설정
memory.return_messages = True  # Chat 모델용
# memory.return_messages = False  # 텍스트 완성용 (기본값)
```

## ConversationBufferMemory

### 개념

**가장 단순한 메모리 형태**로 전체 대화를 그대로 저장합니다.

**특징**:
- ✅ 구현이 간단함
- ✅ 모든 컨텍스트 보존
- ❌ 대화가 길어질수록 토큰 비용 증가
- ❌ 모델의 컨텍스트 윈도우 제한에 걸릴 수 있음

### 기본 사용법

```python
from langchain.memory import ConversationBufferMemory

# 메모리 생성
memory = ConversationBufferMemory()

# 대화 저장
memory.save_context(
    {"input": "Hi"},
    {"output": "How are you?"}
)

# 메모리 확인 (문자열 형식)
print(memory.load_memory_variables({}))
# 출력: {'history': 'Human: Hi\nAI: How are you?'}
```

### Chat 모델용 설정

```python
# Chat 모델용: 메시지 객체 반환
memory = ConversationBufferMemory(return_messages=True)

memory.save_context(
    {"input": "Hi"},
    {"output": "How are you?"}
)

print(memory.load_memory_variables({}))
# 출력: {'history': [HumanMessage(content='Hi'), AIMessage(content='How are you?')]}
```

### 실제 코드 예제

```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough

# 모델과 메모리 설정
model = ChatOpenAI()
memory = ConversationBufferMemory(return_messages=True)

# 프롬프트에 메모리 플레이스홀더 포함
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{message}"),
])

# 메모리 로드 함수
def load_memory(_):
    x = memory.load_memory_variables({})
    return {"history": x["history"]}

# 체인 구성
chain = RunnablePassthrough.assign(history=load_memory) | prompt | model

# 대화 실행
response = chain.invoke({"message": "Hi, I'm Bob"})
memory.save_context({"input": "Hi, I'm Bob"}, {"output": response.content})
```

## ConversationBufferWindowMemory

### 개념

**슬라이딩 윈도우 방식**으로 최근 N개의 메시지만 기억합니다.

**특징**:
- ✅ 메모리 크기 제한으로 비용 절약
- ✅ 최신 대화에 집중
- ❌ 오래된 중요한 정보 손실 가능

### 사용법

```python
from langchain.memory import ConversationBufferWindowMemory

# 최근 2개 대화만 기억
memory = ConversationBufferWindowMemory(
    k=2,  # 기억할 대화 수
    return_messages=True
)

# 여러 대화 저장
conversations = [
    ("Hello", "Hi there!"),
    ("What's your name?", "I'm Claude"),
    ("How are you?", "I'm doing well"),
    ("What's the weather?", "I don't know"),
]

for human, ai in conversations:
    memory.save_context({"input": human}, {"output": ai})

# 최근 2개만 기억됨
print(memory.load_memory_variables({}))
```

### 윈도우 크기 최적화

```python
# 대화 길이에 따른 적정 윈도우 크기
def get_optimal_window_size(avg_message_length, max_tokens=4000):
    """평균 메시지 길이를 기반으로 최적 윈도우 크기 계산"""
    estimated_tokens_per_message = avg_message_length * 0.75  # 대략적인 토큰 변환
    return min(10, max_tokens // (estimated_tokens_per_message * 2))  # 입력+출력 고려

# 동적 윈도우 크기 설정
window_size = get_optimal_window_size(avg_message_length=100)
memory = ConversationBufferWindowMemory(k=window_size, return_messages=True)
```

## ConversationSummaryMemory

### 개념

**대화를 요약**하여 저장하는 메모리입니다. LLM을 사용해 대화 내용을 압축합니다.

**특징**:
- ✅ 긴 대화도 압축하여 관리
- ✅ 핵심 정보는 유지
- ❌ 요약 과정에서 일부 세부사항 손실
- ❌ 요약 생성을 위한 추가 LLM 호출 필요

### 사용법

```python
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI

# LLM이 필요함 (요약 생성용)
llm = ChatOpenAI(temperature=0)

memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True
)

# 긴 대화 저장
memory.save_context(
    {"input": "Tell me about the history of artificial intelligence"},
    {"output": "Artificial intelligence has a rich history dating back to the 1950s. It began with Alan Turing's work on machine intelligence and the famous Turing Test. The field saw early enthusiasm in the 1960s with programs like ELIZA, but then experienced periods of reduced funding known as 'AI winters' in the 1970s and 1980s. The field resurged with machine learning in the 1990s and deep learning in the 2010s, leading to today's large language models like GPT."}
)

# 요약된 형태로 저장됨
print(memory.load_memory_variables({}))
```

### 커스텀 요약 프롬프트

```python
from langchain.prompts import PromptTemplate

# 커스텀 요약 템플릿
summary_template = """
대화를 다음 형식으로 간단히 요약해주세요:
- 주요 주제: 
- 핵심 질문:
- 중요한 답변:

대화 내용:
{text}

요약:
"""

memory = ConversationSummaryMemory(
    llm=llm,
    prompt=PromptTemplate.from_template(summary_template),
    return_messages=True
)
```

## ConversationSummaryBufferMemory

### 개념

**하이브리드 접근법**으로 요약과 최근 메시지를 결합합니다.

**작동 방식**:
1. 최근 N개 메시지는 그대로 보관
2. 오래된 메시지들은 요약하여 저장
3. 토큰 제한에 도달하면 자동으로 요약 생성

**특징**:
- ✅ 최근 대화의 세부사항 보존
- ✅ 오래된 대화의 핵심 정보 유지
- ✅ 메모리 크기 자동 관리
- ❌ 복잡한 내부 로직

### 사용법

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,  # 토큰 제한
    return_messages=True
)

# 여러 대화 저장 (자동으로 요약 관리)
for i in range(10):
    memory.save_context(
        {"input": f"Message {i}"},
        {"output": f"Response {i}"}
    )

# 요약 + 최근 메시지 확인
print(memory.load_memory_variables({}))
```

## ConversationKGMemory

### 개념

**지식 그래프(Knowledge Graph)** 방식으로 메모리를 관리합니다.

**특징**:
- 엔티티 간의 관계를 그래프로 저장
- 구조화된 정보 관리
- 복잡한 관계 추적 가능

### 사용법

```python
from langchain.memory import ConversationKGMemory

memory = ConversationKGMemory(
    llm=llm,
    return_messages=True
)

memory.save_context(
    {"input": "My name is Bob and I work at Google"},
    {"output": "Nice to meet you Bob! How do you like working at Google?"}
)

# 지식 그래프 확인
print(memory.kg.get_triples())
# 출력: [('Bob', 'works at', 'Google')]
```

## LLMChain과의 메모리 통합

### 기존 체인에 메모리 추가

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 메모리가 포함된 프롬프트
template = """
The following is a conversation between a human and an AI.

{history}
Human: {input}
AI:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["history", "input"]
)

# 메모리가 있는 체인
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=ConversationBufferMemory(),
    verbose=True
)

# 대화 실행 (메모리 자동 관리)
response1 = chain.run("Hi, I'm Alice")
response2 = chain.run("What's my name?")  # "Alice"를 기억함
```

## Chat 기반 메모리

### ChatPromptTemplate와 메모리

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat 형식 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who remembers conversation history."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Chat 메모리 설정
memory = ConversationBufferMemory(
    memory_key="history",  # 프롬프트의 변수명과 일치
    return_messages=True   # Chat 모델용
)

# 체인 구성
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
    prompt=prompt,
    verbose=True
)
```

## LCEL 기반 메모리

### 현대적 메모리 구현

LCEL(LangChain Expression Language)을 사용한 메모리 관리:

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from operator import itemgetter

# 메모리 로드 함수
def load_memory(input_dict):
    """메모리에서 대화 히스토리 로드"""
    memory_variables = memory.load_memory_variables({})
    input_dict["history"] = memory_variables["history"]
    return input_dict

def save_context(input_dict):
    """대화를 메모리에 저장"""
    memory.save_context(
        {"input": input_dict["input"]},
        {"output": input_dict["output"]}
    )
    return input_dict["output"]

# LCEL 체인 구성
chain = (
    RunnableLambda(load_memory)
    | {
        "input": itemgetter("input"),
        "history": itemgetter("history")
    }
    | prompt
    | model
    | RunnableLambda(lambda x: {"output": x.content, "input": "..."})
    | RunnableLambda(save_context)
)
```

### 고급 메모리 패턴

```python
class AdvancedMemoryChain:
    def __init__(self, llm, memory_type="buffer"):
        self.llm = llm
        self.memory = self._create_memory(memory_type)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
    
    def _create_memory(self, memory_type):
        """메모리 타입에 따라 적절한 메모리 생성"""
        if memory_type == "buffer":
            return ConversationBufferMemory(return_messages=True)
        elif memory_type == "window":
            return ConversationBufferWindowMemory(k=5, return_messages=True)
        elif memory_type == "summary":
            return ConversationSummaryMemory(llm=self.llm, return_messages=True)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
    
    def invoke(self, user_input):
        """메모리를 활용한 대화 실행"""
        # 메모리 로드
        memory_vars = self.memory.load_memory_variables({})
        
        # 프롬프트 생성
        messages = self.prompt.format_messages(
            history=memory_vars["history"],
            input=user_input
        )
        
        # LLM 호출
        response = self.llm.invoke(messages)
        
        # 메모리 저장
        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )
        
        return response
```

## 실습 코드 예제

### 완전한 메모리 챗봇 구현

```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

class MemoryChatbot:
    def __init__(self, memory_type="buffer", max_tokens=1000):
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = self._create_memory(memory_type, max_tokens)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with perfect memory."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{message}")
        ])
        
        # LCEL 체인 구성
        self.chain = (
            RunnablePassthrough.assign(history=self._load_memory)
            | self.prompt
            | self.llm
            | RunnableLambda(self._save_memory)
        )
    
    def _create_memory(self, memory_type, max_tokens):
        """메모리 타입별 생성"""
        if memory_type == "buffer":
            return ConversationBufferMemory(return_messages=True)
        elif memory_type == "window":
            return ConversationBufferWindowMemory(k=5, return_messages=True)
        elif memory_type == "summary":
            return ConversationSummaryMemory(llm=self.llm, return_messages=True)
        elif memory_type == "summary_buffer":
            return ConversationSummaryBufferMemory(
                llm=self.llm, 
                max_token_limit=max_tokens,
                return_messages=True
            )
    
    def _load_memory(self, inputs):
        """메모리에서 대화 히스토리 로드"""
        memory_vars = self.memory.load_memory_variables({})
        inputs["history"] = memory_vars["history"]
        return inputs
    
    def _save_memory(self, response):
        """대화를 메모리에 저장"""
        # 여기서 원래 입력을 저장해야 하지만, 
        # 실제 구현에서는 체인 외부에서 처리
        return response
    
    def chat(self, message):
        """메모리를 활용한 대화"""
        response = self.chain.invoke({"message": message})
        
        # 메모리에 저장
        self.memory.save_context(
            {"input": message},
            {"output": response.content}
        )
        
        return response.content
    
    def get_memory_stats(self):
        """메모리 통계 반환"""
        memory_vars = self.memory.load_memory_variables({})
        history = memory_vars["history"]
        
        if isinstance(history, str):
            return {"type": "string", "length": len(history)}
        else:
            return {"type": "messages", "count": len(history)}

# 사용 예제
chatbot = MemoryChatbot(memory_type="summary_buffer", max_tokens=500)

# 대화 테스트
print(chatbot.chat("Hi, my name is Alice and I love programming"))
print(chatbot.chat("What's my name and what do I love?"))
print(chatbot.chat("Tell me about Python programming"))
print(chatbot.chat("What did we talk about earlier?"))

# 메모리 상태 확인
print(f"Memory stats: {chatbot.get_memory_stats()}")
```

### 메모리 성능 비교

```python
import time
from typing import List, Dict

def benchmark_memory_types():
    """다양한 메모리 타입의 성능 비교"""
    llm = ChatOpenAI(temperature=0)
    
    memory_configs = {
        "buffer": ConversationBufferMemory(return_messages=True),
        "window": ConversationBufferWindowMemory(k=5, return_messages=True),
        "summary": ConversationSummaryMemory(llm=llm, return_messages=True),
        "summary_buffer": ConversationSummaryBufferMemory(
            llm=llm, max_token_limit=200, return_messages=True
        )
    }
    
    test_conversations = [
        ("Hello, I'm testing memory performance", "Hi! I'll help you test memory."),
        ("What's my previous message about?", "You mentioned testing memory performance."),
        ("How many messages have we exchanged?", "We've exchanged 2 messages so far."),
        ("Summarize our conversation", "We're testing different memory types for performance."),
    ]
    
    results = {}
    
    for memory_name, memory in memory_configs.items():
        start_time = time.time()
        
        # 대화 시뮬레이션
        for human_msg, ai_msg in test_conversations:
            memory.save_context({"input": human_msg}, {"output": ai_msg})
            _ = memory.load_memory_variables({})  # 메모리 로드 시간 측정
        
        end_time = time.time()
        
        # 메모리 크기 계산
        memory_vars = memory.load_memory_variables({})
        if isinstance(memory_vars["history"], str):
            memory_size = len(memory_vars["history"])
        else:
            memory_size = sum(len(msg.content) for msg in memory_vars["history"])
        
        results[memory_name] = {
            "execution_time": end_time - start_time,
            "memory_size": memory_size,
            "memory_type": type(memory).__name__
        }
    
    return results

# 벤치마크 실행
benchmark_results = benchmark_memory_types()
for memory_type, stats in benchmark_results.items():
    print(f"{memory_type:15} | Time: {stats['execution_time']:.3f}s | Size: {stats['memory_size']:4d} chars")
```

## 핵심 포인트 정리

### 메모리 타입 선택 가이드

1. **ConversationBufferMemory**: 
   - 짧은 대화, 모든 컨텍스트 필요
   - 높은 정확도, 높은 비용

2. **ConversationBufferWindowMemory**: 
   - 긴 대화, 최근 맥락 중요
   - 중간 정확도, 예측 가능한 비용

3. **ConversationSummaryMemory**: 
   - 매우 긴 대화, 핵심만 보존
   - 낮은 비용, 정보 손실 가능

4. **ConversationSummaryBufferMemory**: 
   - 최고의 균형
   - 최근 + 요약, 자동 관리

5. **ConversationKGMemory**: 
   - 구조화된 정보 관리
   - 복잡한 관계, 특수 용도

### 실무 적용 팁

1. **메모리 타입 선택**: 사용 사례와 비용을 고려하여 결정
2. **토큰 모니터링**: 메모리 크기를 정기적으로 확인
3. **테스트**: 실제 대화에서 메모리 성능 검증
4. **하이브리드 접근**: 상황에 따라 메모리 타입 동적 변경

이것으로 LangChain Memory의 완벽 가이드를 마칩니다. 다음 장에서는 RAG(Retrieval Augmented Generation)에 대해 학습하겠습니다.