# 📚 Templates API Reference

## 🎯 개요
LangChain의 프롬프트 템플릿 관련 클래스들의 완전한 API 레퍼런스입니다.

## 📋 목차
- [PromptTemplate](#prompttemplate)
- [ChatPromptTemplate](#chatprompttemplate)
- [FewShotPromptTemplate](#fewshotprompttemplate)
- [FewShotChatMessagePromptTemplate](#fewshotchatmessageprompttemplate)
- [PipelinePromptTemplate](#pipelineprompttemplate)

---

## PromptTemplate

### 클래스 정의
```python
from langchain.prompts import PromptTemplate

class PromptTemplate:
    def __init__(
        self,
        template: str,
        input_variables: List[str],
        template_format: str = "f-string",
        validate_template: bool = True
    )
```

### 매개변수
| 매개변수 | 타입 | 필수/선택 | 기본값 | 설명 |
|----------|------|-----------|--------|------|
| `template` | str | 필수 | - | 변수가 포함된 템플릿 문자열 |
| `input_variables` | List[str] | 필수 | - | 템플릿에서 사용할 변수명 리스트 |
| `template_format` | str | 선택 | "f-string" | 템플릿 형식 ("f-string", "jinja2") |
| `validate_template` | bool | 선택 | True | 템플릿 유효성 검사 여부 |

### 주요 메서드

#### `from_template(template: str) -> PromptTemplate`
```python
# 📋 기능: 문자열에서 PromptTemplate 생성 (변수 자동 추출)
# 📥 입력: 템플릿 문자열
# 📤 출력: PromptTemplate 인스턴스

template = PromptTemplate.from_template("Hello {name}, how are you?")
```

#### `format(**kwargs) -> str`
```python
# 📋 기능: 템플릿에 값을 대입하여 최종 문자열 생성
# 📥 입력: 변수명=값 형태의 키워드 인자
# 📤 출력: 완성된 프롬프트 문자열

result = template.format(name="Alice")
# 출력: "Hello Alice, how are you?"
```

#### `format_prompt(**kwargs) -> PromptValue`
```python
# 📋 기능: PromptValue 객체로 포맷팅
# 📥 입력: 변수명=값 형태의 키워드 인자
# 📤 출력: PromptValue 객체 (LCEL에서 사용)

prompt_value = template.format_prompt(name="Bob")
```

#### `save(file_path: str) -> None`
```python
# 📋 기능: 템플릿을 파일로 저장
# 📥 입력: 저장할 파일 경로 (.json, .yaml 확장자)

template.save("my_template.json")
```

### 사용 예제
```python
from langchain.prompts import PromptTemplate

# 1. 기본 생성
template = PromptTemplate(
    template="Tell me a {adjective} joke about {topic}",
    input_variables=["adjective", "topic"]
)

# 2. from_template 사용 (권장)
template = PromptTemplate.from_template("Tell me a {adjective} joke about {topic}")

# 3. 사용
joke_prompt = template.format(adjective="funny", topic="programming")
print(joke_prompt)
# 출력: "Tell me a funny joke about programming"
```

---

## ChatPromptTemplate

### 클래스 정의
```python
from langchain.prompts import ChatPromptTemplate

class ChatPromptTemplate:
    def __init__(
        self,
        messages: List[MessagePromptTemplate],
        input_variables: List[str] = None
    )
```

### 매개변수
| 매개변수 | 타입 | 필수/선택 | 설명 |
|----------|------|-----------|------|
| `messages` | List[MessagePromptTemplate] | 필수 | 메시지 템플릿들의 리스트 |
| `input_variables` | List[str] | 선택 | 입력 변수명 (자동 추론 가능) |

### 주요 메서드

#### `from_messages(messages: List[Tuple[str, str]]) -> ChatPromptTemplate`
```python
# 📋 기능: 메시지 튜플 리스트에서 ChatPromptTemplate 생성
# 📥 입력: (역할, 메시지) 튜플들의 리스트
# 📤 출력: ChatPromptTemplate 인스턴스

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Hello, my name is {name}"),
    ("ai", "Hello {name}! How can I help you today?"),
    ("human", "{user_input}")
])
```

#### `format_messages(**kwargs) -> List[BaseMessage]`
```python
# 📋 기능: 메시지 객체 리스트로 포맷팅
# 📥 입력: 변수명=값 형태의 키워드 인자
# 📤 출력: BaseMessage 객체들의 리스트

messages = template.format_messages(
    name="Alice",
    user_input="What's the weather like?"
)
```

### 지원되는 메시지 역할
| 역할 | 설명 | 사용 예시 |
|------|------|-----------|
| `"system"` | 시스템 지시사항 | 역할 정의, 규칙 설정 |
| `"human"` | 사용자 메시지 | 질문, 요청 |
| `"ai"` | AI 응답 메시지 | Few-shot 예제에서 사용 |

### 사용 예제
```python
from langchain.prompts import ChatPromptTemplate

# 1. 기본 대화 템플릿
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that speaks like a {character}."),
    ("human", "{user_input}")
])

# 2. 메시지 생성
messages = template.format_messages(
    character="pirate",
    user_input="What's your favorite food?"
)

# 3. LCEL과 함께 사용
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI()
chain = template | chat
response = chain.invoke({
    "character": "pirate", 
    "user_input": "What's your favorite food?"
})
```

---

## FewShotPromptTemplate

### 클래스 정의
```python
from langchain.prompts.few_shot import FewShotPromptTemplate

class FewShotPromptTemplate:
    def __init__(
        self,
        examples: List[Dict[str, str]] = None,
        example_selector: BaseExampleSelector = None,
        example_prompt: PromptTemplate,
        suffix: str,
        input_variables: List[str],
        prefix: str = "",
        example_separator: str = "\n\n",
        template_format: str = "f-string"
    )
```

### 매개변수
| 매개변수 | 타입 | 필수/선택 | 기본값 | 설명 |
|----------|------|-----------|--------|------|
| `examples` | List[Dict] | 선택* | None | 예제 리스트 |
| `example_selector` | BaseExampleSelector | 선택* | None | 동적 예제 선택기 |
| `example_prompt` | PromptTemplate | 필수 | - | 예제 포맷팅 템플릿 |
| `suffix` | str | 필수 | - | 사용자 입력 부분 |
| `input_variables` | List[str] | 필수 | - | 입력 변수명 |
| `prefix` | str | 선택 | "" | 예제 앞 텍스트 |
| `example_separator` | str | 선택 | "\n\n" | 예제 구분자 |

*참고: `examples`와 `example_selector` 중 하나는 반드시 제공해야 함*

### 사용 예제
```python
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

# 1. 예제 데이터
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# 2. 예제 포맷터
example_prompt = PromptTemplate.from_template(
    "Word: {word}\nAntonym: {antonym}"
)

# 3. Few-shot 템플릿 생성
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of each word.",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"]
)

# 4. 사용
result = prompt.format(input="big")
print(result)
```

---

## FewShotChatMessagePromptTemplate

### 클래스 정의
```python
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate

class FewShotChatMessagePromptTemplate:
    def __init__(
        self,
        examples: List[Dict[str, str]] = None,
        example_selector: BaseExampleSelector = None,
        example_prompt: ChatPromptTemplate,
        input_variables: List[str] = None
    )
```

### 매개변수
| 매개변수 | 타입 | 필수/선택 | 설명 |
|----------|------|-----------|------|
| `examples` | List[Dict] | 선택* | 대화 예제 리스트 |
| `example_selector` | BaseExampleSelector | 선택* | 동적 예제 선택기 |
| `example_prompt` | ChatPromptTemplate | 필수 | 대화 예제 포맷터 |
| `input_variables` | List[str] | 선택 | 입력 변수 (자동 추론) |

### 사용 예제
```python
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate

# 1. 대화 예제
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

# 2. 예제 포맷터
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 3. Few-shot 대화 템플릿
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. 최종 템플릿에 통합
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a wondrous wizard of math."),
    few_shot_prompt,
    ("human", "{input}"),
])
```

---

## PipelinePromptTemplate

### 클래스 정의
```python
from langchain.prompts.pipeline import PipelinePromptTemplate

class PipelinePromptTemplate:
    def __init__(
        self,
        final_prompt: PromptTemplate,
        pipeline_prompts: List[Tuple[str, PromptTemplate]],
        input_variables: List[str] = None
    )
```

### 매개변수
| 매개변수 | 타입 | 필수/선택 | 설명 |
|----------|------|-----------|------|
| `final_prompt` | PromptTemplate | 필수 | 최종 통합 템플릿 |
| `pipeline_prompts` | List[Tuple[str, PromptTemplate]] | 필수 | (이름, 템플릿) 구성 요소들 |
| `input_variables` | List[str] | 선택 | 입력 변수 (자동 추론) |

### 사용 예제
```python
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

# 1. 구성 요소 템플릿들
intro = PromptTemplate.from_template(
    "You are impersonating {character}."
)

example = PromptTemplate.from_template(
    "Here's how you talk:\nHuman: {example_q}\nYou: {example_a}"
)

start = PromptTemplate.from_template(
    "Now start!\nHuman: {question}\nYou:"
)

# 2. 최종 템플릿
final = PromptTemplate.from_template(
    "{intro}\n\n{example}\n\n{start}"
)

# 3. 파이프라인 구성
pipeline = PipelinePromptTemplate(
    final_prompt=final,
    pipeline_prompts=[
        ("intro", intro),
        ("example", example),
        ("start", start),
    ]
)

# 4. 사용
result = pipeline.format(
    character="pirate",
    example_q="What's your name?",
    example_a="Arrr! Captain Blackbeard!",
    question="What's your favorite treasure?"
)
```

---

## 🔧 유틸리티 함수

### load_prompt()
```python
from langchain.prompts import load_prompt

def load_prompt(file_path: str) -> PromptTemplate:
    """
    📋 기능: 파일에서 프롬프트 템플릿 로드
    📥 입력: 프롬프트 파일 경로 (.json, .yaml)
    📤 출력: PromptTemplate 인스턴스
    """

# 사용 예시
template = load_prompt("my_prompt.json")
```

### 검증 함수들
```python
# 템플릿 변수 검증
template.input_variables  # 필요한 변수 목록 확인

# 템플릿 포맷 검증
try:
    template.format(required_var="value")
except KeyError as e:
    print(f"필수 변수 누락: {e}")
```

---

## 🎯 Best Practices

### 1. 변수명 규칙
```python
# ✅ 좋은 예: 명확하고 설명적인 변수명
template = PromptTemplate.from_template(
    "Generate a {content_type} about {topic} for {target_audience}"
)

# ❌ 나쁜 예: 모호한 변수명
template = PromptTemplate.from_template(
    "Generate a {x} about {y} for {z}"
)
```

### 2. 에러 처리
```python
try:
    result = template.format(user_input="hello")
except KeyError as e:
    print(f"필수 변수 누락: {e}")
except Exception as e:
    print(f"템플릿 처리 오류: {e}")
```

### 3. 타입 힌트 사용
```python
from typing import Dict, List
from langchain.prompts import PromptTemplate

def create_prompt(template_str: str, variables: List[str]) -> PromptTemplate:
    return PromptTemplate(
        template=template_str,
        input_variables=variables
    )
```

---

## 🚨 주의사항

### 보안
- 사용자 입력을 직접 템플릿에 삽입하지 말고 변수를 통해 전달
- 민감한 정보가 포함된 템플릿 파일의 권한 관리 필요

### 성능
- 큰 예제를 가진 FewShotPromptTemplate은 토큰 제한 주의
- 자주 사용되는 템플릿은 미리 로드하여 재사용

### 호환성
- LangChain 버전에 따른 API 변경 사항 확인
- 저장된 템플릿 파일의 버전 호환성 검증

---

이 API 레퍼런스는 LangChain의 주요 프롬프트 템플릿 클래스들의 완전한 사용법을 제공합니다. 각 클래스의 매개변수와 메서드를 활용하여 강력하고 유연한 프롬프트 시스템을 구축할 수 있습니다.