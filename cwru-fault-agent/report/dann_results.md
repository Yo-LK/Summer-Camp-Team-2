# 전이학습 B: DANN 실험 결과

> **참고**: 이 문서의 표에 적힌 수치는 초기 검증 실행(샌드박스) 기준입니다.
> **실제 최종 결과는 `experiments/runs/dann_results.csv`,
> `experiments/runs/cnn_vs_dann_comparison.csv`(로컬 재현 실행 결과)를 기준으로 삼습니다.**
> 신경망은 랜덤 초기화·미니배치 순서에 따라 실행마다 수치가 달라질 수 있으나(단일 시드
> 결과이기 때문, 4절 한계점 참고), 핵심 결론 — "target_3hp에서 DANN이 뚜렷하게 개선되고,
> target_adaptation 표본이 적은 target_0hp에서는 불안정하다" — 은 두 실행 모두에서
> 동일하게 재현되었습니다.

보고서의 "방법론" 및 "실험 결과" 섹션에 인용 가능하도록 정리했습니다.

---

## 1. 방법론

### 1.1 개요
DANN(Domain-Adversarial Neural Network, Ganin & Lempitsky, 2015)은 CORAL/TCA와 근본적으로
다른 접근이다. CORAL/TCA가 **손으로 설계한 특징(hand-crafted feature)의 통계량을 사후에
정렬**하는 반면, DANN은 **신경망이 특징 표현 자체를 도메인-불변하게 학습**하도록 강제한다.

### 1.2 구조
기존 CNN 베이스라인(`src/models/cnn/model.py`)의 `feature_extractor`를 그대로 재사용하고,
두 개의 head를 붙였다.

```
                    ┌─→ label_classifier (고장유형 분류, source 라벨만 사용)
raw signal(8192) → feature_extractor(→128차원)
                    └─→ Gradient Reversal Layer → domain_discriminator (source/target 판별, 라벨 불필요)
```

**Gradient Reversal Layer(GRL)**가 핵심 트릭이다. 순전파 시에는 항등함수처럼 통과시키지만,
역전파 시에는 그래디언트에 $-\lambda$를 곱해 부호를 반전시킨다. 그 결과:
- `domain_discriminator`는 정상적으로 "도메인을 더 잘 맞추는" 방향으로 학습되고
- 그 그래디언트가 반전되어 `feature_extractor`에 전달되므로, `feature_extractor`는 반대로
  "도메인을 구분 못 하게 만드는" 방향(도메인-불변 표현)으로 학습된다

이 둘이 동시에 이루어지는 것이 별도의 min-max 최적화 루프 없이 **하나의 역전파**로
구현되는 것이 GRL의 핵심 이점이다.

### 1.3 학습 설정
- **웜스타트**: CNN 베이스라인(`train_baseline.py`)에서 학습한 `feature_extractor` 가중치를
  초기값으로 불러와서 시작 (완전 랜덤 초기화보다 안정적이고 빠르게 수렴)
- **λ 스케줄링** (Ganin & Lempitsky, 2015): λ_p = 2/(1+e^(-10p)) − 1
  (p: 학습 진행률 0→1). 학습 초반엔 도메인 적대적 신호를 약하게 주고, 후반으로 갈수록
  강하게 줘서 분류기가 먼저 어느 정도 안정된 후에 도메인 정렬이 걸리도록 함
- **데이터 사용 원칙**: `target_adaptation`(라벨 없음)은 도메인 판별 손실에만 사용,
  `target_test`는 최종 평가에만 사용 (CORAL/TCA와 동일한 원칙, 데이터 누수 방지)
- 클래스 불균형 대응을 위해 분류 손실에 `source_train` 기준 클래스 가중치 적용

---

## 2. 결과

### 2.1 CNN 베이스라인(전이학습 없음) vs DANN

| task | CNN 베이스라인 F1 | DANN F1 | 개선폭 |
|---|---|---|---|
| target_0hp | 0.675 | 0.667 | −0.9%p |
| target_1hp | 0.914 | 0.914 | −0.1%p |
| target_2hp | 0.999 | 0.999 | −0.1%p |
| target_3hp | 0.861 | **0.935** | **+7.4%p** |

(원본 데이터: `experiments/runs/cnn_vs_dann_comparison.csv`,
요약: `cnn_vs_dann_summary.csv`, 그래프: `cnn_vs_dann_comparison.png`)

### 2.2 학습 중 도메인 판별 정확도(domain_acc) 추이
`domain_acc`는 도메인 판별기가 source/target을 얼마나 잘 구분하는지 나타낸다
(1.0=완전히 구분됨, 0.5=완전히 구분 불가 = feature가 도메인-불변이라는 뜻, 이상적).

- **target_3hp**: 학습 전반에 걸쳐 0.50~0.59 사이에서 안정적으로 유지됨 → GRL이 의도대로
  도메인-불변 표현을 성공적으로 학습시켰다는 근거
- **target_0hp**: 마찬가지로 0.47~0.63 사이로 판별기는 잘 속고 있었으나, `val_f1` 자체가
  0.35~0.90 사이로 **에폭마다 크게 요동**침 (아래 3.2절에서 해석)

---

## 3. 해석

### 3.1 DANN은 "domain gap이 크지만 적응 데이터가 충분한" 상황에서 가장 효과적
target_3hp는 CORAL/TCA 실험에서도 SVM 기준 가장 큰 개선(+17.7%p)을 보였던 태스크로,
CNN+DANN 조합에서도 **가장 뚜렷한 개선(+7.4%p)**을 보였다. `target_adaptation` 표본이
308개로 4개 태스크 중 가장 많다는 점도 안정적인 도메인 정렬에 기여했을 것으로 보인다.

target_1hp, target_2hp는 이미 CNN 베이스라인만으로도 F1 0.91~0.99로 포화 상태였기 때문에
DANN이 추가로 개선할 여지가 거의 없었다 (천장 효과, ceiling effect).

### 3.2 target_0hp: 적응 데이터 부족이 DANN에서도 동일하게 문제가 됨
target_0hp는 `target_adaptation` 표본이 98개로 다른 태스크의 1/3 수준인데, 이 태스크에서만
**val_f1이 에폭마다 0.35~0.90으로 요동**치는 불안정한 학습 양상을 보였다. 도메인 판별기에
줄 수 있는 target 표본이 적어 미니배치마다 도메인 손실의 분산이 크고, 이것이 GRL을 통해
`feature_extractor`에 불안정한 그래디언트로 전달된 것으로 해석된다.

**→ CORAL/TCA에서도 동일하게 target_0hp가 가장 불안정했다는 점과 일치하는 결과**이며,
이는 개별 방법론의 결함이 아니라 "이 태스크 자체가 적응 데이터 부족으로 어렵다"는
구조적 원인임을 뒷받침한다.

### 3.3 CORAL/TCA와의 대조 (역할 4번 결과와 비교)
같은 4개 태스크에 대해 정리하면:

| task | RF+CORAL Δ | SVM+CORAL Δ | CNN+DANN Δ |
|---|---|---|---|
| target_0hp | −9.3%p | +9.2%p | −0.9%p |
| target_1hp | −5.2%p | −0.2%p | −0.1%p |
| target_2hp | −1.8%p | −0.3%p | −0.1%p |
| target_3hp | −2.8%p | **+17.7%p** | **+7.4%p** |

흥미롭게도 **CORAL(SVM)과 DANN(CNN) 모두 target_3hp에서 가장 큰 개선을 보였다** — 서로
완전히 다른 메커니즘(통계량 정렬 vs 적대적 표현 학습)임에도 "어떤 태스크에서 전이학습이
잘 통하는가"에 대해 일관된 결론에 도달했다는 점이 흥미롭다. 반면 RandomForest는 CORAL 적용
시 4개 태스크 전부에서 손해를 봤는데, 이는 트리 모델이 선형 변환·적대적 표현학습 어느 쪽과도
궁합이 좋지 않을 가능성을 시사한다 (본 프로젝트에서 RF+DANN 조합은 시도하지 않음 — DANN은
신경망 전용 구조이기 때문).

### 3.4 결론 (보고서 요약 문단 초안)

> DANN을 CNN 베이스라인 위에 구현하여 4개 leave-one-load-out 태스크에서 평가한 결과,
> CORAL/TCA와 유사하게 효과가 태스크에 따라 갈렸다. 이미 CNN 베이스라인만으로 F1 0.91
> 이상을 달성한 쉬운 태스크(target_1hp, target_2hp)에서는 개선 여지가 없었던 반면,
> 가장 어려운 두 태스크 중 적응 데이터가 충분했던 target_3hp(target_adaptation 308개)에서는
> macro-F1을 0.861→0.935로 7.4%p 끌어올렸다. 반대로 적응 데이터가 극히 적은 target_0hp
> (98개)에서는 CORAL/TCA와 마찬가지로 불안정한 결과를 보여, 적응 데이터의 양이 특징 기반
> 방법과 신경망 기반 방법 모두에 공통적으로 중요한 제약 조건임을 확인했다. 이는 방법론
> 자체보다 시나리오(태스크 난이도·가용 적응 데이터양)가 전이학습의 성패를 더 크게 좌우할
> 수 있음을 시사한다.

---

## 4. 한계점 (보고서에 명시 권장)

1. **단일 시드** 결과다. CORAL/TCA와 마찬가지로 신뢰구간 확보를 위해 여러 시드 반복 실행이
   필요하다.
2. **RandomForest+DANN 비교는 불가능**하다 (DANN은 신경망 구조에 종속적). 따라서 "분류기
   종류에 따른 CORAL 효과 차이"에 대응하는 "분류기 종류에 따른 DANN 효과 차이" 비교는
   본 실험 범위에 포함하지 않았다.
3. target_0hp의 학습 불안정성(val_f1 요동)은 조기 종료(early stopping) 기준을 `source_validation`
   성능으로만 판단하는 현재 구현의 한계와도 맞물려 있을 수 있다 — source_validation은
   domain shift와 무관하므로, "가장 domain-invariant한 체크포인트"를 고르는 기준으로는
   불완전하다. 이는 CNN 베이스라인 실험(`cnn_baseline_results.csv`)에서도 모든 태스크의
   `best_val_f1`이 1.0에 도달했다는 관찰과 일맥상통한다.

---

## 5. 산출물 목록

- `src/models/cnn/dann_model.py` — DANN, GradientReversalLayer, DomainDiscriminator 구현
- `src/models/cnn/train_dann.py` — 4태스크 DANN 학습 스크립트 (CNN 베이스라인 웜스타트)
- `experiments/runs/dann_results.csv` — DANN 결과 (4행)
- `experiments/runs/cnn_vs_dann_comparison.csv` — CNN 베이스라인+DANN 통합 결과
- `experiments/runs/cnn_vs_dann_summary.csv` — 개선폭 요약
- `experiments/runs/cnn_vs_dann_comparison.png` — 시각화
- `experiments/runs/checkpoints/dann_*.pt` — 학습된 DANN 체크포인트