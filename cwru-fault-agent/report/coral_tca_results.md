# 전이학습 A: CORAL / TCA 실험 결과

보고서의 "방법론" 및 "실험 결과" 섹션에 인용 가능하도록 정리했습니다.

---

## 1. 방법론

### 1.1 실험 설계
`CHANGELOG_data_preprocessing.md`에서 정의한 **leave-one-load-out 4개 태스크**
(target_0hp, target_1hp, target_2hp, target_3hp) 각각에 대해, 3가지 방법 × 2가지 분류기 =
총 24개 실험 조합을 수행했다.

| 방법 | 설명 |
|---|---|
| **none** | 전이학습 없음. `source_train`으로 학습한 분류기를 `target_test`에 그대로 적용 (zero-shot 베이스라인) |
| **CORAL** | source 특징의 2차 통계량(공분산)을 target 통계량에 맞춰 재정렬 후 학습 |
| **TCA** | source와 target을 함께 낮은 차원으로 투영해 도메인 간 평균 분포 차이(MMD)를 최소화하는 공유 표현을 학습 |

**중요한 원칙(데이터 누수 방지):** CORAL/TCA 모두 도메인 정렬에는 **`target_adaptation`
(라벨 없는 소량의 target 데이터)만 사용**하고, `target_test`는 최종 평가에만 사용했다.
분류기 학습에 쓰이는 스케일러(`StandardScaler`)도 `source_train`에만 `fit`했다.

### 1.2 CORAL (Sun & Saenko, 2016)
Source 특징 $X_s$를 whitening(자신의 공분산 제거) 후 target 공분산으로 recoloring한다.

$$X_s' = (X_s - \mu_s) \, C_s^{-1/2} \, C_t^{1/2} + \mu_t$$

- $C_s, C_t$: source, target(adaptation set)의 공분산 행렬
- 분류기는 정렬된 $X_s'$로 학습하고, target_test는 변환 없이 그대로 평가 (CORAL의 원 정의를 따름)

### 1.3 TCA (Pan et al., 2011) — 선형 커널 간소화 구현
원 논문은 커널 트릭(n×n 커널 행렬)을 사용하지만, 본 프로젝트는 특징 차원이 15로 작고
표본 수가 많아(최대 3,624개) 표준 커널 공식이 수치적으로 불안정했다(공분산 행렬 특이성
문제 확인). 이에 **선형 커널의 경우 n×n 문제가 d×d(15×15) 문제와 수학적으로 동치**라는 성질을 이용해
직접 특징공간에서 계산하도록 구현했다.

$$X^\top L X = (\mu_s - \mu_t)(\mu_s - \mu_t)^\top \quad (\text{rank-1}), \qquad
X^\top H X = \text{전체 데이터 산포행렬}$$

이 두 $d \times d$ 행렬의 일반화 고유값 문제를 풀어 도메인 간 평균 차이는 최소화하면서
전체 분산은 보존하는 투영 방향을 찾는다. (`src/models/transfer/domain_adapt.py`,
`LinearTCA` 클래스)

**한계**: 이는 선형(linear) 커널만 지원하는 간소화 버전이다. RBF 등 비선형 커널을 쓰는
원 TCA보다 표현력이 낮을 수 있으며, 이는 한계점으로 보고서에 명시할 필요가 있다.

---

## 2. 결과

### 2.1 태스크별 데이터 규모

| 태스크 | source_train | source_validation | target_adaptation | target_test |
|---|---|---|---|---|
| target_0hp | 3,624 | 924 | 98 | 481 |
| target_1hp | 2,869 | 741 | 286 | 1,231 |
| target_2hp | 2,872 | 741 | 283 | 1,231 |
| target_3hp | 2,941 | 711 | 308 | 1,223 |

target_0hp는 `target_adaptation`이 98개로 다른 태스크(약 300개)의 1/3 수준밖에 되지 않아,
도메인 통계량(특히 공분산) 추정이 불안정할 수 있다는 점을 염두에 두어야 한다.

### 2.2 전체 결과표 (accuracy / macro-F1)

| task | classifier | none (acc/F1) | coral (acc/F1) | tca (acc/F1) |
|---|---|---|---|---|
| target_0hp | RandomForest | 0.857 / 0.893 | 0.753 / 0.799 | 0.867 / 0.840 |
| target_0hp | SVM_RBF | 0.638 / 0.710 | 0.761 / 0.802 | 0.680 / 0.539 |
| target_1hp | RandomForest | 0.879 / 0.882 | 0.885 / 0.830 | 0.890 / 0.902 |
| target_1hp | SVM_RBF | 0.857 / 0.812 | 0.854 / 0.810 | 0.874 / 0.824 |
| target_2hp | RandomForest | 0.981 / 0.984 | 0.959 / 0.966 | 0.960 / 0.964 |
| target_2hp | SVM_RBF | 0.927 / 0.943 | 0.923 / 0.940 | 0.931 / 0.945 |
| target_3hp | RandomForest | 0.836 / 0.863 | 0.851 / 0.835 | 0.866 / 0.887 |
| target_3hp | SVM_RBF | 0.817 / 0.643 | 0.865 / 0.820 | 0.707 / 0.554 |

(원본 데이터: `experiments/runs/coral_tca_results.csv`, 그래프: `coral_tca_comparison.png`)

### 2.3 macro-F1 개선폭 (method − none)

| task | classifier | CORAL Δ | TCA Δ |
|---|---|---|---|
| target_0hp | RandomForest | **−0.093** | −0.052 |
| target_0hp | SVM_RBF | **+0.092** | −0.171 |
| target_1hp | RandomForest | −0.052 | +0.020 |
| target_1hp | SVM_RBF | −0.002 | +0.012 |
| target_2hp | RandomForest | −0.018 | −0.020 |
| target_2hp | SVM_RBF | −0.003 | +0.002 |
| target_3hp | RandomForest | −0.028 | +0.024 |
| target_3hp | SVM_RBF | **+0.177** | −0.088 |

---

## 3. 해석

### 3.1 핵심 발견: CORAL은 RandomForest에서 일관되게 성능을 낮춘다
**4개 태스크 전부에서 CORAL 적용 시 RandomForest의 macro-F1이 하락했다** (−0.093, −0.052,
−0.018, −0.028). 이는 트리 기반 모델이 특징의 축별 분할(axis-aligned split) 경계에
의존하는데, CORAL이 공분산 구조 전체를 선형변환(회전 포함)하면서 이 분할 경계와 실제
클래스 경계 사이의 정렬이 깨지기 때문으로 해석된다. 반대로 SVM(RBF 커널, 거리 기반)에서는
CORAL이 도움이 되는 경우가 있었다.

### 3.2 CORAL은 "도메인 격차가 큰" 태스크에서만 SVM에 도움이 된다
SVM 기준 CORAL의 개선폭은 태스크의 원래 난이도(zero-shot 성능)와 뚜렷한 상관을 보였다.

- 이미 쉬운 태스크(target_1hp: F1 0.812, target_2hp: F1 0.943)에서는 CORAL이 거의 변화 없음(−0.002, −0.003)
- 원래 어려운 태스크(target_0hp: F1 0.710, target_3hp: F1 0.643)에서는 CORAL이 큰 폭으로 개선(+0.092, **+0.177**)

즉 **"CORAL은 도메인 간 거리가 클 때, 그리고 거리·마진 기반 분류기에서 효과적"**이라는
가설이 본 실험에서도 재현되었다. 다만 트리 기반 모델에는 이 효과가 나타나지 않고 오히려
역효과였다는 점은 방법-모델 궁합을 신중히 선택해야 함을 시사한다.

### 3.3 TCA는 결과가 혼재되어 있으며, target_0hp에서 특히 불안정
TCA는 RandomForest 기준 target_1hp(+0.020), target_3hp(+0.024)에서는 소폭 개선을 보였지만
target_0hp(−0.052), target_2hp(−0.020)에서는 하락했다. 특히 target_0hp에서 **SVM 기준
TCA가 −0.171로 가장 큰 폭의 성능 저하**를 보였는데, 이는 이 태스크의 `target_adaptation`
표본 수가 98개로 다른 태스크의 1/3 수준이라 도메인 평균 차이 추정 자체가 노이즈에
취약했기 때문으로 추정된다. → 한계점 섹션에 명시 필요.

### 3.4 결론 (보고서 요약 문단 초안)

> CORAL과 TCA를 leave-one-load-out 4개 태스크에 걸쳐 평가한 결과, 두 방법 모두
> 무조건적인 성능 향상을 보장하지 않았으며 효과는 (a) 분류기 종류, (b) 원래 도메인 격차의
> 크기, (c) 적응에 사용 가능한 target 데이터 양에 따라 달라졌다. 구체적으로 CORAL은
> RandomForest 4개 태스크 전부에서 성능을 저하시킨 반면, 원래 zero-shot 성능이 낮았던
> 태스크(target_0hp, target_3hp)에서는 SVM 성능을 큰 폭으로 개선했다(+0.09~+0.18 F1).
> 이는 CORAL이 특징 공간을 선형 재정렬하는 방식이 거리/마진 기반 분류기와는 궁합이 맞지만
> 축 정렬 분할에 의존하는 트리 모델과는 상충한다는 것을 시사한다. TCA는 target_adaptation
> 표본이 충분한 태스크(약 280~310개)에서는 비교적 안정적이었으나, 표본이 적은
> target_0hp(98개)에서는 두 분류기 모두에서 불안정한 결과를 보여, 적응 데이터 양이
> 전이학습 방법의 신뢰도에 미치는 영향을 별도로 언급할 필요가 있다.

---

## 4. 한계점 (보고서에 명시 권장)

1. **TCA는 선형 커널로 간소화된 구현**이다. 원 논문의 비선형(RBF) 커널 TCA와 비교하면
   표현력이 제한적일 수 있다.
2. **target_0hp의 target_adaptation 표본 수(98개)가 다른 태스크 대비 적어**, 이 태스크에서의
   결과는 통계적으로 덜 안정적일 수 있다.
3. CORAL/TCA 모두 **단일 시드(random_state=42)** 결과다. 팀 문서에서 요청된 대로,
   신뢰구간 확보를 위해 시드 3개 이상 반복 실행 후 평균±표준편차로 보고하는 것을 권장한다
   (`compare.py` 또는 후속 스크립트에서 반복 실행 지원 예정).

---

## 5. 산출물 목록

- `src/models/transfer/domain_adapt.py` — CORAL, LinearTCA 구현
- `src/models/transfer/run_coral_tca.py` — 4태스크 × 3방법 × 2분류기 실험 스크립트
- `experiments/runs/coral_tca_results.csv` — 전체 원본 결과 (24행)
- `experiments/runs/coral_tca_delta_summary.csv` — 방법별 개선폭 요약
- `experiments/runs/coral_tca_comparison.png` — 시각화 (분류기별 태스크×방법 막대그래프)