# 📄 Paper Review: LINGO-Space

**논문 제목:** LINGO-Space: Language-Conditioned Incremental Grounding for Space  
**저자:** Dohyun Kim, Nayoung Oh, Deokmin Hwang, Daehyung Park (KAIST)  
**게재:** AAAI 2024  
**arxiv:** https://arxiv.org/abs/2402.01183  
**공식 코드:** https://github.com/rirolab/LINGO-Space  
**리뷰 작성:** Myoungjin Kim | 2026

---

## 1. 왜 이 논문을 읽었나 (Personal Motivation)

RIRO Lab의 연구를 공부하면서 처음 접한 논문이다.  
나는 이전에 PLAN-A 프로젝트에서 GPT-4o에 few-shot prompting을 적용해 자연어 대화에서 구조화된 일정 정보를 추출하는 시스템 설계에 참여한 적이 있다.  
C.M.D.M.S 프로젝트에서는 SAHI + YOLOv5를 이용한 객체 탐지를 조사했다.  

LINGO-Space를 보면서 두 경험이 하나로 수렴되는 느낌을 받았다.  
자연어를 파싱하는 LLM과 공간을 인식하는 비전 시스템이 결합되어 로봇이 "어디에 행동할지"를 결정하는 구조가 바로 내가 관심 가지는 방향이었다.

---

## 2. 핵심 문제 (Problem)

기존 로봇 언어 이해 연구의 대부분은 **instance grounding** — 즉, "어떤 객체인지" 찾는 문제에 집중해왔다.

이 논문은 다른 문제를 푼다: **space grounding** — "어느 공간에 행동해야 하는지" 찾는 문제.

예시:
> "컵을 테이블 위, 접시 가까이에 놓아줘"

이 명령에서 로봇이 찾아야 하는 건 특정 객체가 아니라, **빈 공간의 위치**다.

### 어려운 이유 3가지

| 문제 | 설명 | 예시 |
|------|------|------|
| Positional ambiguity | 거리 정보가 없음 | "오른쪽에" → 10cm? 30cm? |
| Referential ambiguity | 비슷한 객체가 여러 개 | "컵 근처" → 어떤 컵? |
| Compositional ambiguity | 여러 조건이 동시에 주어짐 | "접시 위쪽이면서 컵 왼쪽" |

---

## 3. 핵심 아이디어 (Key Idea)

> **위치를 하나의 좌표로 출력하지 않고, 확률 분포로 표현한다.**  
> **그리고 새로운 표현이 들어올수록 그 분포를 점점 좁혀나간다.**

### 기존 방식 vs LINGO-Space

```
기존 방식:
"접시 오른쪽" → (x=30, y=0) 하나의 좌표 출력
→ 틀리면 끝

LINGO-Space:
"접시 오른쪽" → 오른쪽 방향으로 퍼진 확률 분포
+ "컵 앞쪽" → 두 조건의 교집합으로 분포가 좁아짐
+ "조금 더 가까이" → 더 좁아짐
→ 최종 분포의 최댓값이 목표 위치
```

이 점진적 업데이트(incremental update) 구조가 논문 이름의 **Incremental**에 해당한다.

---

## 4. 시스템 구조 (Architecture)

전체 파이프라인은 3개의 모듈로 구성된다.

```
자연어 명령 입력
"put the cyan bowl above the chocolate and left of the silver spoon"
        │
        ▼
┌─────────────────────────┐
│   1. Scene Graph        │   카메라로 환경 인식
│      Generator          │   → 객체 위치 + 관계를 그래프로 표현
│   (GroundingDINO + CLIP)│
└────────────┬────────────┘
             │
        ▼
┌─────────────────────────┐
│   2. Semantic Parser    │   자연어 명령 분해
│      (ChatGPT)          │   → [("chocolate","above"), ("silver spoon","left")]
│   LLM 기반 파싱          │   각 관계 표현을 순서대로 분리
└────────────┬────────────┘
             │
        ▼
┌─────────────────────────┐
│   3. Spatial-Dist.      │   관계 표현 하나씩 처리하면서
│      Estimator          │   확률 분포를 점진적으로 업데이트
│   (Graph Neural Network)│
└────────────┬────────────┘
             │
        ▼
   최종 목표 위치 (x*, y*)
```

### 확률 분포: Polar Distribution

위치를 표현하는 방식이 독특하다. 직교 좌표(x, y) 대신 **극좌표(거리 d, 각도 φ)** 를 사용한다.

```
예: "접시 오른쪽에"
→ 접시를 중심으로
  거리 d: Gaussian 분포 (적당히 가까운 거리)
  각도 φ: Von Mises 분포 (오른쪽 방향, 약 0도)
```

여러 객체에 대한 분포를 가중합(mixture)하여 최종 분포를 만든다.

---

## 5. 내가 이해한 핵심 연결고리

### PLAN-A와의 연결

PLAN-A에서 GPT-4o는 대화 전체를 보고 일정 정보를 한 번에 추출했다.  
하지만 LINGO-Space의 Semantic Parser는 ChatGPT를 이용해 복합 명령을 **순서 있는 단위 표현들로 분해**한다.

두 시스템 모두 LLM의 **in-context learning(few-shot prompting)** 능력을 활용해 모호한 자연어를 구조화된 형태로 변환한다는 공통점이 있다.

차이점은, LINGO-Space는 분해된 표현들을 **순서대로 처리하며 확률 분포를 갱신**한다는 점이다. 이것이 훨씬 더 robust한 방식이다.

### few-shot prompting → incremental grounding

```
PLAN-A:    모호한 대화 → (few-shot prompting) → 구조화된 일정 정보
                                                 (한 번에 해결)

LINGO-Space: 모호한 공간 표현 → (semantic parsing) → 분해된 표현들
                               → (incremental update) → 점점 좁아지는 분포
                                                       (단계적으로 해결)
```

---

## 6. 실험 결과 (Results)

- 20가지 테이블탑 조작 벤치마크 테스트에서 기존 방법 대비 높은 성능
- 분포 업데이트(incremental)가 실제로 공간을 더 정확하게 좁혀주는 것을 확인
- Boston Dynamics Spot 로봇으로 실제 환경 내비게이션 실험 성공

---

## 7. 한계 및 후속 연구 방향

- 3D 공간이 아닌 2D 공간에서만 실험됨 (실제 조작에는 높이 정보도 필요)
- Scene graph 생성이 완벽하다고 가정 (실제로는 오류 발생 가능)
- 후속 연구로 C2F-SPACE (Coarse-to-Fine)가 제안됨 — 더 정밀한 공간 그라운딩

---

## 8. 배운 점 & 느낀 점

이 논문을 읽으면서 가장 인상 깊었던 부분은 **"모호함을 해결하는 방식"** 이었다.

기존 방법들이 모호함을 "없애려" 했다면, LINGO-Space는 모호함을 **확률 분포로 명시적으로 표현**하고 추가 정보가 올 때마다 업데이트한다. 이것이 더 정직하고 현실적인 접근이라고 느꼈다.

로봇이 단순히 명령을 실행하는 것이 아니라, 불확실한 상황에서도 **점진적으로 이해를 구체화**하면서 행동할 수 있다는 점이 Embodied AI의 핵심 방향 중 하나라는 것을 이해하게 됐다.

---

## 9. 참고 자료

- 논문 PDF: https://arxiv.org/pdf/2402.01183
- 공식 코드: https://github.com/rirolab/LINGO-Space
- 프로젝트 페이지 + 데모 영상: https://lingo-space.github.io
- RIRO Lab: https://rirolab.kaist.ac.kr
