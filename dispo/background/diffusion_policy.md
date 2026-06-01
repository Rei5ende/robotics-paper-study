# 📚 Background Study: Diffusion Policy

> DiSPo의 핵심 구성 요소인 Diffusion Policy를 이해하기 위한 배경 공부 노트  
> 원본 논문: Diffusion Policy (Chi et al., RSS 2023)  
> 관련 논문: DiSPo (ICRA 2026)

---

## 1. Diffusion Policy가 왜 필요한가 — 기존 방법의 한계

### 기존 Imitation Learning의 문제: Multimodal Action Distribution

로봇이 같은 상황에서도 **여러 가지 올바른 행동**이 존재할 수 있다.

```
예: 테이블 위 컵을 집으라는 명령

  행동 A: 왼쪽에서 접근해서 집기
  행동 B: 오른쪽에서 접근해서 집기
  행동 C: 위에서 바로 집기

→ 셋 다 올바른 행동
→ 이걸 "Multimodal Action Distribution"이라고 함
```

### 기존 방법(Behavior Cloning)의 실패

가장 단순한 모방 학습 방법인 Behavior Cloning은 평균을 출력하려 한다.

```
시연 데이터:
  상황 S → 행동 A (왼쪽 접근)
  상황 S → 행동 B (오른쪽 접근)

Behavior Cloning 학습 결과:
  상황 S → (A + B) / 2  ← 중간 어딘가로 가는 행동
  → 실제로는 아무것도 제대로 못 하는 행동
```

여러 올바른 행동 중 하나를 선택하지 못하고 **평균값을 출력**해서 실패한다.

---

## 2. Diffusion Model이란? — 이미지 생성 AI의 핵심 기술

Diffusion Policy를 이해하려면 먼저 Diffusion Model을 알아야 한다.

### 핵심 아이디어

```
Forward Process (노이즈 추가):
  원본 이미지 x0
      ↓ 노이즈 조금 추가
      x1
      ↓ 노이즈 조금 더 추가
      x2
      ↓ ...
      xT  ← 완전한 랜덤 노이즈

Reverse Process (노이즈 제거):
  랜덤 노이즈 xT
      ↓ 노이즈 조금 제거 (모델이 학습)
      x_{T-1}
      ↓ ...
      x0  ← 원본 이미지 복원
```

모델은 **"노이즈를 어떻게 제거할지"** 를 학습한다.  
한 번에 완성된 이미지를 만드는 게 아니라 **조금씩 점진적으로 개선**한다.

### DDPM (Denoising Diffusion Probabilistic Model)

Diffusion Policy가 사용하는 핵심 알고리즘.

```
학습 시:
  1. 원본 데이터 x0 선택
  2. 랜덤 노이즈 ε 생성
  3. x0에 ε를 더해서 xk 생성
  4. 모델이 "xk에서 ε를 예측"하도록 학습

추론 시:
  1. 랜덤 노이즈 xK에서 시작
  2. 모델이 노이즈를 예측하고 제거
  3. K번 반복 → 최종 출력 x0
```

---

## 3. Diffusion Policy — 로봇 행동에 Diffusion 적용

Chi et al. (2023)은 이 아이디어를 **로봇 행동(Action) 생성**에 적용했다.

### 핵심 아이디어

```
이미지 생성:  랜덤 노이즈 → (노이즈 제거 반복) → 이미지
Diffusion Policy: 랜덤 노이즈 → (노이즈 제거 반복) → 로봇 행동
```

행동을 "생성해야 할 데이터"로 보고, 이미지 생성과 동일한 방식으로 접근한다.

### Multimodal 문제 해결

```
상황 S에서 가능한 행동: A (왼쪽), B (오른쪽)

Behavior Cloning: 평균 → 실패
Diffusion Policy: 랜덤 노이즈에서 시작
                  → 어떤 시작점이냐에 따라 A 또는 B 중 하나로 수렴
                  → 둘 다 올바른 행동이므로 성공
```

랜덤성이 오히려 **다양한 올바른 행동 중 하나를 자연스럽게 선택**하게 해준다.

---

## 4. Diffusion Policy 구조

```
입력:
  - 관찰 o (카메라 이미지 + 로봇 관절 상태)
  - 노이즈가 추가된 행동 a^(k)

         ↓
┌─────────────────────────┐
│   노이즈 예측 모델        │
│   (CNN 또는 Transformer) │
│                         │
│   조건: 관찰 o           │
│   입력: 노이즈 행동 a^(k) │
│   출력: 노이즈 예측 ε̂    │
└─────────────────────────┘
         ↓
   노이즈 제거:
   a^(k-1) = a^(k) - ε̂ + 약간의 노이즈

         ↓ (K번 반복)

최종 행동 a^(0)
```

### 두 가지 구현 방식

**CNN 기반 (U-Net):**
- 빠른 추론 속도
- 로컬 패턴에 강함

**Transformer 기반:**
- 더 긴 시퀀스 처리 가능
- 복잡한 행동 패턴에 강함

DiSPo는 여기서 Transformer 대신 **Mamba(SSM)** 를 사용했다.

---

## 5. Diffusion Policy의 장점 3가지

### ① Multimodal Action Distribution 처리
앞서 설명한 것처럼 여러 올바른 행동을 자연스럽게 처리한다.

### ② 고차원 행동 공간
로봇 팔의 관절이 7개면 행동 공간이 7차원이다.  
Diffusion은 고차원 공간에서도 잘 동작한다.

### ③ 안정적인 학습
기존 GAN 기반 방법들은 학습이 불안정했는데, Diffusion은 상대적으로 안정적이다.

---

## 6. Diffusion Policy의 한계 — DiSPo가 해결하려 한 것

### 속도 문제
노이즈 제거를 K번 반복해야 해서 추론이 느리다.

```
K = 100번 반복 → 실시간 제어에 부적합
DiSPo의 해결: 거친 동작은 빠르게, 정밀한 동작만 느리게
```

### Temporal Structure 부재
행동 시퀀스를 생성할 때 시간적 구조를 고려하지 않는다.

```
기존 Diffusion Policy:
  행동 시퀀스를 그냥 "고차원 벡터"로 취급
  → 0.1초 간격인지 1초 간격인지 모름

DiSPo:
  Mamba의 Step Size로 시간 구조 명시적 모델링
  → 정밀도 조절 가능
```

---

## 7. 전체 흐름 정리

```
Behavior Cloning
  → Multimodal 문제로 실패
        ↓
Diffusion Model (이미지 생성에서 옴)
  → 점진적 노이즈 제거로 Multimodal 처리
        ↓
Diffusion Policy (Chi et al. 2023)
  → 행동 생성에 Diffusion 적용
  → 강력하지만 느리고 Temporal Structure 없음
        ↓
DiSPo (RIRO Lab, 2024)
  → Diffusion + Mamba(SSM) 결합
  → Temporal Structure 이해 + 정밀도 조절
```

---

## 8. 배운 점

Diffusion Policy를 공부하면서 인상 깊었던 건 **"랜덤성을 약점이 아닌 강점으로 활용"** 한다는 점이었다.

기존 방법들은 데이터의 다양성(Multimodal)을 처리하기 어려워서 평균값으로 수렴했는데, Diffusion은 오히려 랜덤 노이즈에서 시작함으로써 이 다양성을 자연스럽게 다룬다.

SSM 노트에서 공부한 Step Size와 연결해서 보면:
- Diffusion은 **행동의 다양성** 문제를 해결
- SSM은 **시간의 구조** 문제를 해결
- DiSPo는 두 가지를 결합해서 **다양하면서도 정밀한** 행동을 생성

---

## 9. 다음 공부할 것

- [ ] TCL (Transferable Constraint Learning) 리뷰 → ICL 계열 연구로 이동
- [ ] DDPM 원본 논문 훑기: https://arxiv.org/abs/2006.11239
- [ ] Diffusion Policy 공식 코드: https://github.com/real-stanford/diffusion_policy

---

## 참고 자료

- Diffusion Policy 논문: https://arxiv.org/abs/2303.04137
- DDPM 논문: https://arxiv.org/abs/2006.11239
- 공식 코드: https://github.com/real-stanford/diffusion_policy
- DiSPo 리뷰: ../../paper_review.md
- SSM 배경 공부: ./ssm.md
