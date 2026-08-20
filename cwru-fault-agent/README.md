# CWRU 베어링 고장진단 — 데이터/베이스라인/전이학습 (진행 상황)

> 이 문서는 데이터 파이프라인부터 CORAL/TCA/DANN까지 완료된 작업을 팀 공유용으로
> 정리한 것입니다. Agent 구현과 최종 보고서/발표자료는 아직 남은 단계입니다.

---

## 1. 한눈에 보기: 지금까지 한 일

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 데이터 EDA (클래스 분포, 도메인 구성, 신호 품질) | ✅ |
| 2 | 원신호 → 특징 재현 검증 | ✅ |
| 3 | **전통적 방법** 베이스라인 (RandomForest, SVM) | ✅ |
| 3.5 | 데이터 전처리 확장 (4개 전이 태스크, 문제 레코드 플래그) | ✅ |
| 4 | **전이학습 기법 A**: CORAL, TCA | ✅ |
| 4.5 | CNN 베이스라인 (DANN용 대조군) | ✅ |
| 5 | **전이학습 기법 B**: DANN | ✅ |
| 6 | Agent (도구 5개 + ReAct 루프) | ⬜ 다음 단계 |
| 7 | 최종 보고서/발표자료 통합 | ⬜ |

과제 요구사항인 **"전이학습 기법 2개 이상 + 전통적 방법과 비교"**는 CORAL, TCA, DANN
3개 기법으로 충족했습니다.

---

## 2. 용어 정리 (헷갈리기 쉬운 부분)

**분류기(모델)** 와 **전이학습 기법**은 서로 다른 층위입니다.

```
분류기(모델) — "무엇으로 분류할지"
  ├─ RandomForest ┐
  ├─ SVM          ┴─ 전통적 방법 (신경망 아님)
  └─ CNN              └─ 신경망 방법

전이학습 기법 — "도메인 차이를 어떻게 극복할지" (분류기 위에 얹는 것)
  ├─ CORAL   (RandomForest/SVM 위에 적용)
  ├─ TCA     (RandomForest/SVM 위에 적용)
  └─ DANN    (CNN 위에 적용, 신경망 전용 구조)
```

즉 "전통적 방법 2개(RF, SVM) + 전이학습 3개(CORAL/TCA/DANN)"를 조합해서 실험한 것이며,
CNN은 별도의 네 번째 기법이 아니라 DANN을 구현하기 위한 신경망 구조입니다.

---

## 3. 데이터 구성

- CWRU 베어링 데이터셋 중 **48kHz, Drive-End** 조건만 사용 (56개 레코드)
- **도메인 = 부하(load_hp)**: 0, 1, 2, 3 hp 4가지 조건
- Leave-one-load-out 방식으로 **4개 전이 태스크** 정의 (`window_index.csv`의
  `split_t0`~`split_t3`, 기존 `split` 컬럼 = `split_t3`와 동일 태스크)

| 태스크 | source_train | source_validation | target_adaptation | target_test |
|---|---|---|---|---|
| target_0hp | 3,624 | 924 | 98 | 481 |
| target_1hp | 2,869 | 741 | 286 | 1,231 |
| target_2hp | 2,872 | 741 | 283 | 1,231 |
| target_3hp | 2,941 | 711 | 308 | 1,223 |

**적응 원칙(전 실험 공통)**: `target_adaptation`(라벨 없음)만 도메인 적응에 사용하고,
`target_test`는 최종 평가에만 사용 — 데이터 누수 방지.

### 발견 및 수정한 데이터 이슈
- CWRU 원본 `.mat` 4개 파일(레코드 99, 174, 175, 217)에서 내부 변수명이 파일 번호와
  어긋나는 알려진 문제 발견 → `src/data/loader.py`에서 `data_audit.csv`의 정답 길이와
  대조해 자동 보정 (자세한 내용: 커밋 로그 및 코드 주석 참고)
- `known_issue` 컬럼을 `data_audit.csv`에 추가 (클리핑 5개, 전기노이즈 1개, DE/FE 중복 5개
  레코드 플래그, 삭제는 안 함)

---

## 4. 실험 결과 요약

### 4.1 전통적 방법 (RandomForest / SVM) — zero-shot vs CORAL vs TCA

4개 태스크 × RandomForest/SVM × {none, CORAL, TCA} 전체 결과: `experiments/runs/coral_tca_results.csv`

**핵심 발견**
- CORAL은 **RandomForest에서 4개 태스크 전부 성능 하락** (트리 모델과 궁합이 안 맞음)
- CORAL은 **SVM에서, 원래 도메인 격차가 컸던 태스크(target_0hp, target_3hp)**에서 크게 개선
  (target_3hp: F1 0.643 → 0.820, **+17.7%p**)
- TCA는 결과가 더 불안정, 특히 `target_adaptation` 표본이 적은 target_0hp(98개)에서 취약

→ 자세한 방법론/해석: `report/coral_tca_results.md`

### 4.2 CNN 베이스라인 vs DANN

4개 태스크 결과: `experiments/runs/cnn_vs_dann_comparison.csv` (로컬 재현 실행 기준 최종)

| task | CNN(적응없음) F1 | DANN F1 | 개선폭 |
|---|---|---|---|
| target_0hp | 0.561 | 0.423 | −13.8%p (적응데이터 98개, 불안정) |
| target_1hp | 0.857 | 0.928 | +7.0%p |
| target_2hp | 0.997 | 1.000 | +0.3%p |
| target_3hp | 0.843 | 0.903 | +6.0%p |

**핵심 발견**
- target_3hp에서 DANN이 뚜렷하게 개선 — **CORAL(SVM)도 DANN(CNN)도 target_3hp에서
  가장 크게 개선됐다는 점이 일치** (서로 다른 메커니즘인데 같은 결론)
- target_0hp는 CORAL/TCA/DANN 전부 공통적으로 불안정 → 방법론 문제가 아니라
  **"이 태스크는 적응 데이터(98개)가 절대적으로 부족하다"는 구조적 원인**으로 판단
- 신경망(CNN)은 `source_validation`에서 거의 항상 100%(과적합)를 찍는데도 target 성능은
  들쭉날쭉함 — 전통 ML보다 도메인 시프트에 더 취약할 수 있다는 근거

→ 자세한 방법론/해석: `report/dann_results.md`

---

## 5. 프로젝트 구조

```
cwru-fault-agent/
├── data/                                   # 원본 CSV/JSON + .mat (raw/, git 제외)
│   ├── metadata.csv, window_index.csv, features.csv, ...
│   └── raw/                                 # .mat 56개 (git 제외, 각자 로컬에 준비 필요)
│
├── src/
│   ├── config.py                            # 경로 자동 설정 (프로젝트 어디에 클론해도 동작)
│   ├── data/
│   │   ├── loader.py                        # .mat 로더 (CWRU 변수명 버그 수정 포함)
│   │   ├── build_raw_window_cache.py        # CNN용 원신호 캐시 생성
│   │   └── extend_splits_and_flags.py       # split_t0~t3, known_issue 생성 스크립트
│   ├── features/extractor.py                # 시간/주파수/envelope 특징 추출
│   └── models/
│       ├── baseline_ours.py                 # 전통 ML (RandomForest/SVM) 단일 태스크
│       ├── transfer/
│       │   ├── domain_adapt.py              # CORAL, TCA 구현
│       │   └── run_coral_tca.py             # 4태스크 × 3방법 × 2분류기 실험
│       ├── cnn/
│       │   ├── model.py                     # 1D-CNN (feature_extractor + classifier)
│       │   ├── train_baseline.py            # CNN 베이스라인 (DANN 대조군)
│       │   ├── dann_model.py                # DANN (GRL + 도메인판별기)
│       │   └── train_dann.py                # DANN 4태스크 학습
│       └── team/                            # 팀원 제공 코드 (참고용, 공식 결과 아님)
│
├── notebooks/
│   ├── 01_eda.py
│   └── 02_feature_validation.py
│
├── experiments/runs/                        # 전체 실행 결과 (csv, png)
│   └── checkpoints/                         # 학습된 모델 (.pt, git 제외)
│
└── report/                                   # 보고서용 상세 문서
    ├── CHANGELOG_data_preprocessing.md       # 데이터 변경 이력 + 사유
    ├── coral_tca_results.md                  # CORAL/TCA 방법론 + 결과 + 해석
    └── dann_results.md                       # DANN 방법론 + 결과 + 해석
```

---

## 6. 로컬에서 재현하는 방법

```powershell
# 1. 패키지 설치
pip install torch pandas numpy scipy scikit-learn seaborn joblib matplotlib

# 2. .mat 56개를 data/raw/normal/, data/raw/fault/ 에 배치
#    (metadata.csv의 relative_path 기준, 각자 CWRU 공식 사이트에서 다운로드)

# 3. EDA
python notebooks/01_eda.py

# 4. 특징 검증 (선택, 시간 다소 소요)
python notebooks/02_feature_validation.py

# 5. 전통 ML 베이스라인
python src/models/baseline_ours.py

# 6. CORAL/TCA (4태스크 전체)
python src/models/transfer/run_coral_tca.py

# 7. CNN 베이스라인 캐시 생성 + 학습
python src/data/build_raw_window_cache.py
python src/models/cnn/train_baseline.py

# 8. DANN (CNN 베이스라인 체크포인트 필요, 7번 먼저 실행)
python src/models/cnn/train_dann.py
```

---

## 7. 중요한 설계 결정 사항 (팀 논의 필요 시 참고)

1. **CORAL/TCA 통계량 추정 방식**: 팀원 코드(`coral.py`)는 target 전체(평가셋 포함, transductive)를
   통계 추정에 사용하지만, 우리는 `target_adaptation`(라벨없는 소량 데이터)만 사용하는
   방식(inductive)으로 통일했습니다. 프로젝트의 `class_mapping.json`이 애초에
   `target_adaptation`/`target_test`를 분리해둔 설계 의도와 일치하고, 배포 시나리오상
   더 현실적이기 때문입니다. → 최종 보고서 수치는 우리 방식 기준입니다.
2. **`split_t3` vs 기존 `split`**: 새로 만든 `split_t3`가 기존 `split`과 1.8%(95/5,183 윈도우)
   불일치하여, target=3hp 태스크는 기존 검증된 `split` 컬럼을 그대로 사용하기로 했습니다.
3. **재현성**: 모든 신경망 결과는 현재 단일 시드 기준입니다. 신뢰구간 확보를 위해서는
   시드 3개 이상 반복 실행이 필요합니다 (미착수).

---

## 8. 다음 단계

- [ ] Agent 구현: 도구 5개(`list_available_data`, `load_signal`, `extract_features`,
      `detect_domain_shift`, `classify`, `adapt_model`) + ReAct 루프
- [ ] 데모 시나리오 1개 이상 (예: "2hp 신호 진단해줘" → 도메인 시프트 감지 → 적응 → 재분류)
- [ ] CORAL/TCA, DANN 각각 시드 3회 반복 실행 후 평균±표준편차로 재보고
- [ ] `envelope_rule_team.py`(물리 규칙 기반 진단) 실행 및 결과 통합 — 도메인 시프트에
      원천적으로 영향받지 않는 대조군으로서 보고서 스토리를 보강할 수 있음
- [ ] 최종 보고서 통합 (배경 → 베이스라인 격차 → CORAL/TCA 조건부 효과 → DANN과 대조 → 결론)
- [ ] 발표자료(~15분) 준비
