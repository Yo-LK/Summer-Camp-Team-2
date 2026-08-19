# CWRU 베어링 고장진단 ReAct 에이전트 — 프로젝트 구조

## 0. 확정된 데이터 구조 (실제 파일 검증 완료)

| 요소 | 내용 |
|---|---|
| Source domain | load_hp = 0, 1, 2 → `source_train`(2,941창) / `source_validation`(711창) |
| Target domain | load_hp = 3 → `target_adaptation`(308창) / `target_test`(1,223창) |
| 클래스 (fault_type) | normal(0) / inner_race(1) / ball(2) / outer_race(3) |
| 심각도 (severity) | normal(0) / 0.007(1) / 0.014(2) / 0.021(3) |
| 정규화 | global z-score, `source_train`에서만 계산 (mean≈0.0277, std≈0.4977) — **target 통계는 절대 훈련에 쓰지 않음** |
| 특징 | features.csv에 시간/주파수 영역 특징 + BPFI/BPFO/BSF/FTF 이론 주파수 및 envelope 값 이미 계산됨 |

이 구성 자체가 바로 "Training/Test가 다른 운전 조건(부하)에서 온다"는 과제의 핵심 챌린지입니다. load_hp가 다르면 speed_rpm도 달라지므로 BPFI/BPFO 등 고장 주파수 자체가 이동합니다(→ shaft_frequency_hz 기반 정규화된 특징이 중요해지는 이유).

---

## 1. 디렉터리 구조

```
cwru-fault-agent/
├── data/
│   ├── raw/                        # 원본 .mat (이미 metadata.csv가 가리키는 파일들)
│   └── processed/                  # features.csv, window_index.csv 등 (프로젝트 제공분 그대로 사용)
│
├── src/
│   ├── data/
│   │   ├── loader.py                # .mat 로딩, DE/FE/RPM 채널 파싱
│   │   ├── windowing.py             # window_index.csv 로직 재현 (슬라이딩 윈도우)
│   │   └── dataset.py               # PyTorch/sklearn용 Dataset 래퍼 (split별 로딩)
│   │
│   ├── features/
│   │   ├── time_domain.py           # mean/std/rms/kurtosis/crest_factor/impulse_factor
│   │   ├── freq_domain.py           # FFT, dominant_frequency, spectral_centroid/spread
│   │   ├── envelope.py              # envelope spectrum + BPFI/BPFO/BSF/FTF 계산 (Ball_Bearing.pdf 근거)
│   │   └── extractor.py             # 위 모듈 통합, features.csv 스키마와 1:1 매칭
│   │
│   ├── models/
│   │   ├── classical.py             # 베이스라인: SVM, Random Forest, XGBoost (전통적 방법)
│   │   ├── deep_baseline.py         # 1D-CNN 또는 MLP (딥러닝 베이스라인, 전이학습 없이 source만 학습)
│   │   └── transfer/
│   │       ├── finetune.py          # 방법 1: source 사전학습 → target_adaptation으로 fine-tuning
│   │       ├── dann.py              # 방법 2: Domain-Adversarial NN (gradient reversal)
│   │       └── coral.py             # 방법 3(옵션): CORAL / 특징 분포 정렬, 비교군 확대용
│   │
│   ├── agent/
│   │   ├── tools/
│   │   │   ├── data_loader_tool.py      # Tool: 조건별 데이터 로드
│   │   │   ├── feature_extractor_tool.py # Tool: 신호 → 특징 벡터
│   │   │   ├── classifier_tool.py        # Tool: 특징 → 고장유형/심각도/확률
│   │   │   └── transfer_tool.py          # Tool: 도메인 차이 감지 시 적응 수행
│   │   ├── prompts.py               # ReAct 시스템 프롬프트, Thought 템플릿
│   │   └── react_agent.py           # Thought→Action→Observation 루프 구현
│   │
│   ├── evaluation/
│   │   ├── metrics.py                # accuracy, F1, confusion matrix, per-domain 성능
│   │   └── compare.py                # 전통 ML vs 전이학습 A/B 비교 스크립트
│   │
│   └── config.py                     # class_mapping.json, normalization.json 로더/상수
│
├── notebooks/
│   ├── 01_eda.ipynb                  # metadata/data_audit 기반 탐색
│   ├── 02_feature_validation.ipynb   # 직접 계산 vs features.csv 값 검증
│   ├── 03_baseline_results.ipynb
│   └── 04_transfer_learning_results.ipynb
│
├── experiments/
│   └── runs/                         # 실험별 결과, 모델 체크포인트, 로그
│
├── report/
│   └── final_report.md/.docx         # 배경, 모듈 설명, 성능 분석, 결론
│
├── slides/
│   └── presentation.pptx             # ~15분 발표용
│
├── metadata.json / class_mapping.json / normalization.json   # (제공된 그대로 사용)
├── requirements.txt
└── README.md
```

---

## 2. 단계별 로드맵 (권장 순서)

| 단계 | 내용 | 산출물 |
|---|---|---|
| **1. EDA** | metadata/data_audit/window_index로 클래스·도메인 분포, 신호 품질 확인 | `01_eda.ipynb` |
| **2. Feature 검증** | 원신호에서 직접 특징 계산 → features.csv 값과 일치 확인 (파이프라인 신뢰성 확보) | `extractor.py`, `02_feature_validation.ipynb` |
| **3. 베이스라인 분류기** | source_train 학습 → source_validation 검증 (전통 ML + 딥러닝 각 1개) | `classical.py`, `deep_baseline.py` |
| **4. Zero-shot 전이 실패 확인** | 베이스라인을 target_test에 그대로 적용 → 성능 하락 정량 확인 (도메인 갭 입증) | `compare.py` 1차 결과 |
| **5. 전이학습 2종 구현** | fine-tuning + DANN(또는 CORAL) — target_adaptation으로 적응, target_test로 평가 | `transfer/*.py` |
| **6. ReAct 에이전트 조립** | 4개 Tool 구현 → 에이전트가 조건 인식→도구 선택→(필요시)적응→분류 자동 수행 | `react_agent.py` |
| **7. 성능 비교 및 리포트** | 전통 ML vs 전이학습 vs 무적응 딥러닝 비교표/그래프 | `report/`, `slides/` |

---

## 3. 핵심 설계 포인트

- **도메인 갭을 먼저 "보여주는" 것이 중요**: 베이스라인 모델을 target_test에 무작정 돌려서 성능이 떨어지는 걸 수치로 보여줘야, 전이학습이 필요한 이유와 효과를 설득력 있게 제시할 수 있습니다.
- **정규화 유출 주의**: normalization.json에 명시된 대로 mean/std는 source_train 기준 고정값만 사용 — target 데이터로 재계산하면 안 됩니다(데이터 누수).
- **Tool은 독립적으로 먼저 함수/스크립트로 완성 → 이후 에이전트가 호출하는 구조**로 가면 디버깅이 쉽습니다. 즉 3~5단계를 순수 코드로 검증한 뒤 6단계에서 감싸는 순서를 권장합니다.
- **전이학습 비교군은 최소 2개**: fine-tuning(간단, 강력한 베이스라인) + DANN(적대적 학습) 조합이 난이도 대비 결과 대비가 뚜렷해서 추천됩니다. 여유 있으면 CORAL을 3번째로 추가해 비교표를 풍부하게 만들 수 있습니다.

---

## 4. requirements.txt (초안)

```
numpy
scipy
pandas
scikit-learn
torch
matplotlib
seaborn
xgboost
```
(에이전트 프레임워크는 LangChain/자체 구현 중 선택 필요 — 확정되면 추가)
