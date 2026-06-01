# 🔍 Code Analysis: LINGO-Space Semantic Parser

**분석 대상:** `semantic_parser/parser.py`  
**공식 코드:** https://github.com/rirolab/LINGO-Space  
**논문:** LINGO-Space (AAAI 2024)  
**작성:** Myoungjin Kim | 2026

---

## 1. 전체 코드 (실제 구현)

```python
import os
import timeout_decorator
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = lambda instruction: f"""
Imagine you are a semantic parser.
I will give you an instruction that describes a space.
Given the instruction, first extract an action from the instruction.
Then, extract the source and the target of the action.
Note that a space is composed of reference objects and relation predicates.
Instruction: put the green blocks in a blue bowl
Output: {{'action': 'move', 'source': 'green blocks', 'target': [('blue bowl', 'in')]}}
Instruction: move to the front of the table
Output: {{'action': 'move', 'source': 'robot', 'target': [('table', 'front')]}}
Instruction: put the Pepsi Wild cherry box to the left of the gold box and put the Pepsi Wild cherry box to the right of the bull figure and put the Pepsi Wild cherry box below the gold box and put the Pepsi Wild cherry box below the gold box
Output: {{'action': 'move', 'source': 'Pepsi Wild cherry box', 'target': [('gold box', 'left'), ('bull figure', 'right'), ('gold box', 'below'), ('gold box', 'below')]}}
Instruction: {instruction}
Output:
"""

@timeout_decorator.timeout(240)
def send_query(query, model="gpt-4", temparature=0.2):
    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=temparature
    )
    return chat_completion['choices'][0]['message']['content']

def parse(instruction):
    query = prompt(instruction)
    output = send_query(query)
    output = eval(output)
    return output
```

---

## 2. 코드가 하는 일 (한 줄씩 분석)

### ① API Key 설정
```python
openai.api_key = os.getenv("OPENAI_API_KEY")
```
환경변수에서 API Key를 읽는다. 코드에 직접 Key를 넣지 않는 보안 관행.

---

### ② 프롬프트 구성 (핵심)
```python
prompt = lambda instruction: f"""
Imagine you are a semantic parser.
...
Instruction: {instruction}
Output:
"""
```

`lambda`를 사용해서 **명령어를 받으면 완성된 프롬프트를 반환하는 함수**를 만든다.

프롬프트 내부 구조:

```
[역할 부여]
"Imagine you are a semantic parser."
"...a space is composed of reference objects and relation predicates."

[Few-shot 예시 3개]
예시 1: 단순 명령 (관계 1개)
  "put the green blocks in a blue bowl"
  → {'action': 'move', 'source': 'green blocks', 'target': [('blue bowl', 'in')]}

예시 2: 로봇 자체 이동
  "move to the front of the table"
  → {'action': 'move', 'source': 'robot', 'target': [('table', 'front')]}

예시 3: 복합 명령 (관계 4개)
  "put the Pepsi Wild cherry box to the left of..."
  → {'action': 'move', 'source': '...', 'target': [(...), (...), (...), (...)]}

[실제 입력]
Instruction: {instruction}
Output:         ← GPT가 여기서부터 채워넣음
```

예시 3번이 특히 중요하다.  
같은 객체에 대한 관계가 **4번 반복**되는 극단적인 케이스를 보여줌으로써,  
GPT가 "몇 개든 관계를 다 뽑아야 한다"는 걸 학습하게 한다.

---

### ③ API 호출 + 타임아웃
```python
@timeout_decorator.timeout(240)
def send_query(query, model="gpt-4", temparature=0.2):
    chat_completion = openai.ChatCompletion.create(...)
    return chat_completion['choices'][0]['message']['content']
```

- `@timeout_decorator.timeout(240)`: 240초(4분) 안에 응답 없으면 에러 처리
- `temperature=0.2`: 낮은 온도 → 일관성 있는 출력 유도 (창의성 낮춤)
  - PLAN-A 프로젝트에서는 `temperature=0.5`를 사용했는데, 파싱 같은 정확성이 중요한 작업에서는 더 낮은 값이 적합하다

---

### ④ 파싱 실행
```python
def parse(instruction):
    query = prompt(instruction)   # 프롬프트 완성
    output = send_query(query)    # GPT 호출
    output = eval(output)         # 문자열 → Python 딕셔너리
    return output
```

`eval(output)` 이 핵심이다.  
GPT가 문자열로 반환한 결과를:
```
"{'action': 'move', 'source': 'cyan bowl', 'target': [('chocolate', 'above')]}"
```
Python 딕셔너리로 변환한다:
```python
{'action': 'move', 'source': 'cyan bowl', 'target': [('chocolate', 'above')]}
```

---

## 3. PLAN-A와 실제 코드 수준에서의 비교

| 항목 | PLAN-A | LINGO-Space |
|------|--------|-------------|
| API 호출 방식 | `http.Request` (raw HTTP) | `openai.ChatCompletion.create` (SDK) |
| Few-shot 예시 수 | 2개 | 3개 |
| temperature | 0.5 | 0.2 |
| 출력 파싱 | `json.decode()` | `eval()` |
| 타임아웃 처리 | 없음 | 240초 데코레이터 |
| 이후 처리 | 캘린더 저장 (끝) | GNN 입력으로 전달 |

PLAN-A 코드에서는 `temperature=0.5`를 사용했는데, 일정 추출처럼 정확성이 중요한 작업에서는 LINGO-Space처럼 낮은 값(0.2)이 더 적합하다는 걸 이 코드를 보며 이해했다.

---

## 4. 이 코드에서 배운 점

**① lambda로 프롬프트 템플릿 만들기**

```python
# LINGO-Space 방식
prompt = lambda instruction: f"...{instruction}..."

# 사용할 때
query = prompt("put the cup near the plate")
```

함수를 따로 정의하지 않고 lambda로 간결하게 템플릿화하는 패턴.  
프롬프트가 길어질수록 이렇게 분리하면 가독성이 좋아진다.

**② temperature의 의미**

- `temperature=0`: 항상 같은 답 (deterministic)
- `temperature=0.2`: 거의 일관적, 약간의 유연성
- `temperature=0.5`: 균형
- `temperature=1.0`: 창의적, 다양한 답

파싱처럼 정해진 형식이 중요할 때는 낮은 값을 써야 한다.

**③ eval()의 위험성과 트레이드오프**

`eval()`은 문자열을 Python 코드로 실행시키는 함수다.  
보안상 위험할 수 있지만 (악의적인 코드 실행 가능),  
여기서는 GPT 출력을 딕셔너리로 빠르게 변환하는 실용적 선택이다.  
더 안전한 방법은 `json.loads()`이지만, GPT 출력이 JSON이 아닌 Python dict 형식이라 eval을 쓴 것으로 보인다.

---

## 5. 다음 단계

- [ ] Colab 데모 실행: https://colab.research.google.com/drive/14Nl0sozJ3JpfwxkfwGk_0s8k8DOqTEVN
- [ ] 다양한 명령어 입력해서 파싱 결과 직접 확인
- [ ] temperature를 바꿔가며 출력 변화 관찰
- [ ] Spatial-Distribution Estimator 코드 분석으로 이어가기

---

## 참고

- 공식 코드: https://github.com/rirolab/LINGO-Space
- 논문: https://arxiv.org/abs/2402.01183
- Colab 데모: https://colab.research.google.com/drive/14Nl0sozJ3JpfwxkfwGk_0s8k8DOqTEVN
