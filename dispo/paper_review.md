# 📄 Paper Review: DiSPo

**논문 제목:** DiSPo: Diffusion-SSM based Policy Learning for Coarse-to-Fine Action Discretization  
**저자:** Nayoung Oh, Jaehyeong Jang, Moonkyeong Jung, Daehyung Park (KAIST)  
**게재:** ICRA 2026  
**arxiv:** https://arxiv.org/abs/2409.14719  
**프로젝트 페이지:** https://robo-dispo.github.io  
**리뷰 작성:** Myoungjin Kim | 2026

---

## 1. 왜 이 논문을 읽었나 (Personal Motivation)

박대형 교수님 강연에서 직접 언급된 연구다.

강연에서 교수님은 현재 VLA 모델의 핵심 한계를 이렇게 설명했다:

> "저희가 사용하는 많은 모델들, 예를 들어 트랜스포머 같은 것들은 이런 템플러 스트럭처를 이해하고 하는 건 아니에요. 그냥 다이나믹스를 이해하는 게 아니라 이미지, 어떤 패턴들을 그대로 다시 보여주는 정도에 가깝습니다."

즉 현재 모델들은 **"정밀하게 움직여야 할 때와 거칠게 움직여야 할 때"를 구분하지 못한다**는 것이다.  
DiSPo는 바로 이 문제를 해결하는 연구다.

---

## 2. 핵심 문제 (Problem)

로봇의 실제 작업은 두 가지 종류의 동작이 섞여 있다:

```
Coarse (거친) 동작   →  물체가 있는 방향으로 팔을 크게 이동
Fine (정밀한) 동작   →  물체를 잡는 마지막 몇 mm
```

**박교수님 강연 표현:** "라스트 인치(Last Inch)" — 물체를 잡기 딱 직전 순간이 가장 중요

### 기존 방법의 한계

| 방법 | 문제점 |
|------|--------|
| 정밀한 데이터로만 학습 | 데이터 수집 비용이 매우 높음 (고주파 센서 필요) |
| 거친 데이터로만 학습 | 정밀한 동작을 재현 못 함 |
| 트랜스포머 기반 모델 | Temporal structure 이해 못 함 (정밀/거친 동작 구분 불가) |

### DiSPo의 목표

> **"거친 시연 데이터만 있어도, 사용자가 원하는 정밀도로 동작을 생성할 수 있는 모델"**

---

## 3. 핵심 아이디어 — 두 기술의 결합

DiSPo = **Diffusion Policy** + **SSM (Mamba)**

### Diffusion Policy란?

노이즈에서 시작해서 점점 노이즈를 제거하며 행동을 생성하는 방법.

```
랜덤 노이즈 a(K)
      ↓ 노이즈 제거 (K번 반복)
      ↓
최종 행동 a(0)
```

표현력이 강해서 복잡하고 다양한 행동을 학습할 수 있다는 장점이 있다.

### SSM (State Space Model) — Mamba란?

연속적인 다이나믹스를 이해하는 모델.  
핵심은 **Step Size (∆)** 라는 파라미터다.

```
Step Size가 크다  →  시간을 크게 건너뜀  →  거친(Coarse) 동작
Step Size가 작다  →  시간을 세밀하게 봄  →  정밀한(Fine) 동작
```

트랜스포머는 이 Step Size 개념이 없어서 temporal structure를 이해하지 못한다.  
Mamba는 입력에 따라 Step Size를 동적으로 조절할 수 있다.

### DiSPo의 핵심 혁신 — Step-Scale Factor

```
사용자가 rt = 0.5 입력  →  현재보다 2배 정밀한 동작 생성
사용자가 rt = 2.0 입력  →  현재보다 2배 거친 동작 생성
```

즉 **학습 후에도 사용자가 정밀도를 실시간으로 조절**할 수 있다.

---

## 4. 시스템 구조

```
입력:
  - 관찰 o (카메라 이미지 + 로봇 관절 상태)
  - Step-Scale Factor rt (사용자 지정 정밀도)
  - 노이즈 a(K) (랜덤 가우시안 노이즈)

         ↓
┌─────────────────────────────┐
│   DiSPo Block × NM개        │
│                             │
│   adaLN (정규화)             │
│      ↓                      │
│   Step-scaled Mamba         │
│   (∆t = rt × SoftPlus(...)) │
└─────────────────────────────┘
         ↓
   노이즈 예측 εˆ(k)
         ↓
   다음 단계 행동 a(k-1)
         ↓
   (K번 반복 후)
         ↓
최종 행동 a(0)
```

핵심은 Mamba 블록 안의 Step Size 계산:
```
∆t = rt × SoftPlus(f∆(u))
```
사용자가 입력한 rt가 Step Size를 직접 스케일링한다.

---

## 5. 학습 방법 — 두 단계

### 1단계: 사전학습 (Sample-Rate Augmentation)

거친 데이터(저주파)를 다양한 주파수로 리샘플링해서 학습 데이터를 증강한다.

```
원본 데이터: 5Hz (초당 5번 샘플링)
증강 데이터: 2.5Hz, 1.25Hz ... 다양한 버전 생성
→ 다양한 granularity를 이해하도록 학습
```

### 2단계: 파인튜닝 (Pseudo Demonstration)

1단계에서 학습한 모델로 **가짜 정밀 데이터(Pseudo Demonstration)**를 생성하고,  
이걸로 다시 학습해서 정밀도를 높인다.

```
거친 데이터 → 사전학습된 DiSPo → 정밀한 가짜 데이터 생성
                                          ↓
                               이 데이터로 파인튜닝
```

---

## 6. 실험 결과

### 주요 성과
- 기존 방법 대비 최대 **81% 성공률 향상**
- 1~2mm 갭에 부품 끼우기 (Clamp Passing) 성공
- 사용자가 rt 값으로 정밀도를 실시간 조절 가능

### 박교수님 강연에서 언급한 결과
> "학생들한테 실제로 요걸 부딪히지 말고 끼워라 하면은 20% 정도의 성공률인데 저희 모델이 이제 비전 기반으로 이런 매니플레이션을 할 수 있었다."

---

## 7. 강연 내용과의 연결

박교수님이 강연에서 DiSPo를 언급한 맥락:

```
현재 VLA의 문제
  → Temporal Structure를 이해 못 함
  → 정밀/거친 동작을 구분 못 함
        ↓
DiSPo가 이를 해결
  → SSM(Mamba)으로 Temporal Structure 이해
  → Step-Scale Factor로 정밀도 조절
        ↓
교수님의 비전
  → 이런 Expert Model을 범용 VLA 모델에 플러그인
  → "매트릭스에서 헬기 조종 프로그램 다운로드처럼"
```

즉 DiSPo는 범용 모델을 보완하는 **"정밀 조작 Expert Module"** 역할을 한다.

---

## 8. 배경 개념 정리

DiSPo를 이해하기 위해 새로 공부한 개념들:

### Imitation Learning (모방 학습)
로봇이 사람의 시연을 보고 행동을 학습하는 방법.  
→ 강화학습처럼 보상을 설계할 필요 없이 시연 데이터만 있으면 됨

### Diffusion Model
이미지 생성 AI(DALL-E, Stable Diffusion)의 핵심 기술.  
랜덤 노이즈 → 점진적 노이즈 제거 → 원하는 출력.  
로봇 분야에서는 이걸 행동(Action) 생성에 적용

### State Space Model (SSM)
연속적인 시계열 데이터를 처리하는 모델.  
제어공학에서 오래 쓰인 개념을 딥러닝에 적용.  
Mamba는 최신 SSM으로 트랜스포머의 대안으로 주목받고 있음

### Mamba의 핵심
트랜스포머와 달리 **입력에 따라 Step Size(∆)를 동적으로 변경** 가능.  
이게 DiSPo에서 정밀도 조절의 핵심 메커니즘이 됨

---

## 9. 한계 및 후속 연구

- 현재는 2D 시각 정보 위주 → 촉각(Tactile) 데이터 통합 필요
- Step-Scale Factor를 수동으로 입력해야 함 → 자동화 필요
- 후속 연구: DiSPo-VLA (VLA 모델과 결합) 방향으로 발전 중

---

## 10. 배운 점 & 느낀 점

강연을 먼저 듣고 논문을 읽으니 왜 이 연구가 필요한지가 훨씬 명확하게 이해됐다.

박교수님이 강연에서 말한 **"현재 VLA가 Temporal Structure를 이해 못 한다"** 는 한계가 DiSPo의 출발점이었다.  
트랜스포머는 정밀한 동작과 거친 동작을 구분하지 못하는데, Mamba의 Step Size 메커니즘이 이를 해결한다는 게 핵심 인사이트였다.

LINGO-Space가 공간의 모호성을 확률 분포로 다뤘다면,  
DiSPo는 시간의 모호성(언제 정밀하게, 언제 거칠게)을 Step Size로 다룬다는 점에서  
두 연구 모두 **"모호성을 명시적으로 표현하고 제어한다"** 는 공통된 철학이 있다고 느꼈다.

---

## 참고

- 논문: https://arxiv.org/abs/2409.14719
- 프로젝트 페이지 + 데모 영상: https://robo-dispo.github.io
- 박대형 교수님 강연: https://www.youtube.com/watch?v=3Eg7JbZ674g
- LINGO-Space 리뷰: ../lingo-space/paper_review.md
