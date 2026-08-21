# CWRU 베어링 고장 진단

**언어:** [English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## 프로젝트 목표

다음을 수행할 수 있는 에이전트(agent)를 구축합니다:

1. CWRU 베어링 데이터셋 구조를 로드하고 이해하기
2. 진동 신호 분석하기
3. 적절한 고장 진단 방법 선택하기
4. 서로 다른 운전 조건(모터 부하) 간에 전이 학습 적용하기

**현재 상태:** 전체 데이터 파이프라인(다운로드 → 분할/윈도우/특징 추출 → 베이스라인 → 적응 방법)이 완료되었으며, 에이전트의 의사결정 정책에 바로 활용할 수 있는 하나의 통합 결과 테이블(`data/agent_policy_table.csv`, 12개 페어 × 10개 방법)을 생성했습니다. 이 에이전트/오케스트레이션 레이어 — 프로젝트 목표에서 실제로 "에이전트를 구축"하는 부분 — 이 아직 만들어지지 않은 다음 단계입니다. 지금까지는 모두 에이전트가 필요로 할 입력들을 만들어내는, 수동으로 실행하는 노트북 파이프라인입니다.

## 4가지 목표 대비 진행 상황

| # | 목표 | 상태 | 비고 |
|---|---|---|---|
| 1 | 데이터셋 구조 로드 및 이해 | ✅ 완료 | `data_download.ipynb`가 데이터를 다운로드하고, 라벨링하고, 검사함 |
| 2 | 진동 신호 분석 | 🟡 부분 완료 | 추출 파이프라인(원시 신호/FFT/포락선/고장 주파수 피크)은 존재하지만, 그 위에서의 분석은 아직 없음 — 이 특징들이 실제로 정상/고장 윈도우를 구분해내는지 검증하는 것이 전혀 없음 |
| 3 | 적절한 고장 진단 방법 선택 | 🟢 거의 완료 | 10가지 방법을 실제 페어별 수치로 정면 비교했고, `data/agent_policy_table.csv`로 통합함 — 남은 격차는 "선택"이 아직 사람이 표를 읽는 방식이지, 정책/에이전트가 하는 것이 아니라는 점 |
| 4 | 운전 조건 간 전이 학습 적용 | 🟢 거의 완료 | 두 가지 실제 적응 방법(파인튜닝된 CNN, CORAL+랜덤 포레스트)을 구현하여 12개 페어 전체에서 평가함; 가장 우수한 변형(부분 동결 CNN, FFT 특징 기반 CORAL+RF)은 target-only 상한값과의 격차를 거의 다 좁힘 — 아래 결과 참고 |

## 지금까지 만든 것

### `data_download.ipynb` — 다운로드 및 탐색

- [CWRU 베어링 데이터 센터](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)에서 CWRU의 **정상 베이스라인** 데이터(4개 파일, 0–3 hp 부하별 1개씩)와 **48 kHz 구동측(drive-end) 고장** 데이터(52개 파일: 내륜/볼/외륜 고장, 고장 직경 0.007"/0.014"/0.021", 각각 0–3 hp에서 1개씩)를 다운로드하며, 이미 디스크에 있는 파일은 건너뜁니다.
- 각 파일을 고장 위치, 직경, 부하, RPM, (외륜 고장의 경우) 시계 방향 위치를 인코딩한 설명적인 이름으로 저장합니다 — 예: `48k_drive_end_fault_inner_race_0.007in_0hp_1797rpm_109.mat` — 라벨이 붙은 하위 폴더 `data/normal_baseline_data/`와 `data/48k_drive_end_fault/`에 나누어 저장합니다.
- 원시 `.mat` 파일 하나의 구조를 검사합니다(CWRU 내부 변수명: `X{n}_DE_time`, `X{n}_FE_time`, `X{n}RPM`; 이 데이터셋에는 `BA_time` 채널이 없음).
- 파일명에서 직접 메타데이터를 파싱하여 `manifest` DataFrame(파일당 1행)을 구성하며, 파일별 신호 길이, 지속 시간, 기록된 RPM을 포함합니다.
- 예시 한 쌍에 대해 정상 대 고장 신호를 **시간 영역**과 **FFT**(0–3 kHz)로 각각 플로팅합니다.
- 모든 파일을 하나의 **`data/combined_dataset.mat`**(370 MB)로 통합합니다 — 파일마다 하나의 MATLAB 구조체이며, 신호는 원본 전체 길이를 유지하고 `DE_time`/`FE_time`/`BA_time`과 모든 고장 메타데이터를 함께 담습니다. 통합 대상 소스 디렉터리(`SOURCE_DIRS`)는 암묵적이 아니라 명시적으로 지정됩니다.

### `data_splitting_preprocessing.ipynb` — 분할, 윈도우, 특징 추출

- `scipy.io.loadmat(..., simplify_cells=True)`를 통해 `data/combined_dataset.mat`을 `df`(파일당 1행)로 로드합니다.
- **파일별 시간 기준 train/test 분할**(`split_signal_train_test`, 80/20): 윈도우로 나누기 **전에** 각 파일의 원시 신호를 먼저 분할하여, 고정 크기 윈도우가 train/test 경계를 넘나들거나 그 사이에 정보가 새는 일이 없도록 합니다 — 모든 파일에 대해 train+test를 다시 이어 붙여 원본 신호와 일치하는지 검증했습니다.
- **고정 크기 윈도우 분할**(`segment_signal`, 기본값 `window_size=4096`, 겹치지 않음, 남는 뒷부분은 버림)을 `DE_time`에만 적용합니다(구동측 채널; `FE_time`/`BA_time`은 `combined_dataset.mat`에 담겨 있지만 아직 윈도우로 나누지 않음).
- 윈도우는 `load_hp`별로 묶여 `windows_by_load`가 되고, 한 번만 **`data/windows_by_load.pkl`**(184 MB)로 저장됩니다 — 페어마다 중복 저장하지 않습니다.
- `make_splits(source_load, target_load)`는 `{0, 1, 2, 3}` hp 사이의 **12개 순서쌍(ordered pair)** 중 원하는 것에 대해 4개 버킷 딕셔너리(`source_train`, `source_test`, `target_labeled`, `target_test`)를 즉석에서 구성해 주므로, 다운스트림 학습 루프가 0→1, 0→2, …, 3→2를 순회하며 결과를 집계할 수 있고, 하나의 고정된 source/target 조합에 얽매이지 않습니다.
- **특징 추출** — `windows_by_load`의 모든 윈도우(총 5,601개)에 대해 한 번의 순회로 계산되는 4가지 방법:
  1. **원시 시간 영역** — 윈도우 자체; 진폭 패턴을 직접 포착합니다.
  2. **FFT 크기 스펙트럼** — 주파수 성분을 포착하지만, 대개 고장 충격 성분은 광대역 구조 공진에 묻혀 있습니다.
  3. **포락선(envelope) 스펙트럼** — 윈도우에 `|Hilbert(window)|`를 적용한 뒤 그 포락선에 FFT를 취합니다. 이 복조 과정을 통해 고장 충격의 **발생 빈도**가 반송파 공진과 분리된 깨끗한 스펙트럼 선으로 나타납니다.
  4. **고장 주파수 피크(BPFO/BPFI/BSF)** — CWRU가 공개한 SKF 6205 구동측 베어링의 차수(order) 배수(회전 속도의 3.5848×/5.4152×/2.357×)에 해당하는 이론적 외륜/내륜/볼 자전 고장 주파수와 그 2, 3차 고조파 지점에서 포락선 스펙트럼의 크기를 읽어오되, 항상 어떤 빈(bin)에는 걸리도록 충분히 넓은 허용 오차(약 12.7 Hz — FFT 빈 간격이 11.72 Hz이며, 이전의 ±5 Hz 허용 오차는 그 절반보다 좁아서 일부 고조파가 신호 내용과 무관하게 항상 `0.0`으로 읽히는 문제가 있었습니다)를 사용합니다. 이후 **해당 윈도우 자체의 시간 영역 RMS로 정규화**하여, 결과가 각 윈도우의 전체 진동 진폭이 아니라 상대적인 스펙트럼 집중도를 반영하도록 합니다 — 조밀한 포락선 스펙트럼을 윈도우당 물리적 의미를 지니면서도 도메인 간 비교가 가능한 9개의 숫자로 압축합니다.
  - pickle이 아니라 **`.npz`**로 저장합니다 — 이들은 모델에 입력할 수치 특징 행렬이지, 앞서 사용한 메타데이터 위주의 구조가 아닙니다. 각 윈도우의 메타데이터(`load_hp`, `split`, `category`, `fault_location`, `filename`, 계산된 RPM 등)는 같은 파일 안에 병렬 배열로 함께 저장됩니다: `data/features_time.npz`, `data/features_fft.npz`, `data/features_envelope.npz`, `data/features_fault_freq.npz`. 네 파일 모두 행 순서가 동일하므로, `i`번째 행은 어느 파일에서든 같은 윈도우를 가리킵니다.
  - 특징은 `windows_by_load`와 마찬가지로 부하/분할 단위로 계산되며, **아직 source/target 페어 단위로 구체화되지 않았습니다** — 특정 (source_load, target_load)에 대해 `source_train`/`source_test`/`target_labeled`/`target_test`를 구성하려면, `make_splits`가 원시 윈도우에 대해 하는 것과 같은 방식으로 이 배열들을 `load_hp`/`split` 기준으로 필터링해야 합니다. 이 필터링과, 그 결과에 대한 모델 학습/적응/평가는 (당시 아직 작성되지 않았던) 학습 노트북의 몫으로 남겨두었습니다.

### `model_training.ipynb` — 베이스라인 모델

`data/windows_by_load.pkl`의 원시 `DE_time` 윈도우로 1D CNN(10개 클래스: `normal` + {`inner_race`, `ball`, `outer_race`} × {0.007", 0.014", 0.021"}, 외륜 고장의 시계 방향 위치는 직경 클래스에 통합됨)을 학습합니다. 아키텍처는 `EmbeddingExtractor`(합성곱 스택 → 고정 크기 임베딩)와 `LabelPredictor`(임베딩 → 클래스 로짓)라는 두 개의 독립된 `nn.Module`로 분리되어 있어, 이후의 도메인 적응 방법이 분류 헤드를 건드리지 않고도 임베딩 출력에 직접 연결할 수 있습니다(예: source/target 임베딩 사이의 MMD 계산, 또는 그 위에 gradient-reversal 도메인 분류기 추가).

**12개 순서쌍 (source_load, target_load)** 전체에 대해 평가한 두 가지 적응 없는 기준점:

- **베이스라인 1 — source-only (하한선)**: 한 부하의 전체 train 분할로 학습하고, **다른** 부하의 test 분할로 평가합니다(적응 없음). 평균 정확도 **69.7%**, **39.8%**(2→0)부터 **91.9%**(1→2)까지 분포 — 이것이 순수하게 도메인 시프트로 인해 발생하는 비용이며, 어떤 페어를 고르느냐에 따라 크게 달라집니다.
- **베이스라인 2 — target-only, 희소(클래스당 10%)**: 부하별로 별도의 모델을 학습하되, 해당 부하 train 윈도우의 **클래스당 10%**만 사용하고, 자신의 부하의 test 분할로 도메인 내(in-domain) 평가를 합니다. 평균 정확도 **82.7%**(75.2%–90.6%). 이전 버전에서는 이 베이스라인을 target 부하의 **전체** train 분할(부하당 약 1300개 윈도우)로 학습했었는데, 약 99.9%가 나왔습니다 — 이는 사실상 상한선이지 "라벨이 희소한" 상황의 기준값이 아니므로, 지금 실제로 사용하는 클래스당 10% 베이스라인으로 대체되었습니다.
- 모델은 단 **8개만 학습**합니다(베이스라인당 4개, 부하마다 1개) — 특정 부하에 대해 `source_train`과 `target_labeled`는 동일한 기반 데이터이므로, 해당 부하가 source로 등장하는 3개 페어 전체에서 베이스라인 1의 모델을 재사용할 수 있고, 결과는 페어마다 재학습하는 대신 이미 학습된 모델을 평가하여 만들어집니다.
- 체크포인트는 **`models/`**(`baseline1_full_load{0-3}.pt`, `baseline2_scarce_load{0-3}.pt`)에, 결과는 **`data/baseline_results.csv`**(페어당 1행, 두 베이스라인의 정확도/매크로 F1)에 저장됩니다.

### `domain_adaptation_evaluation.ipynb` — 적응 CNN, CORAL + 랜덤 포레스트, RF 베이스라인, 전체 비교

두 베이스라인 위에 실제 도메인 적응 방법을 구축하고, 12개 페어 전체에 대해 동일한 `target_test`로 모든 것을 비교합니다:

- **적응 CNN, 베이스라인 1로부터 파인튜닝** — 베이스라인 1의 source 학습 가중치를 불러온 뒤, 베이스라인 2가 사용한 것과 동일한 희소(클래스당 10%) target 하위 집합으로 원래보다 10배 낮은 학습률로 파인튜닝합니다. 두 가지 변형: **완전 동결**(`label_predictor`만 재학습)과 **부분 동결**(마지막 합성곱 블록의 가중치와 BatchNorm 이동 통계량도 함께 동결 해제).
- **적응 고전 머신러닝 — CORAL + 랜덤 포레스트** — `source_train`의 공분산을 target의 공분산에 맞춰 정렬(CORAL)한 뒤, 정렬된 결과로 랜덤 포레스트를 학습합니다. 세 가지 특징 세트에서 각각 실행: 9차원 `features_fault_freq`(BPFO/BPFI/BSF 피크), 그리고 `features_fft`/`features_envelope`(각 2049차원 — CORAL 전에 먼저 PCA로 20개 성분까지 축소; 그렇지 않으면 CORAL에 필요한 공분산 행렬이 심각하게 계수 부족(rank-deficient) 상태가 됩니다). **CORAL은 target 도메인에 대해 비지도(unsupervised)입니다** — 정렬 과정은 target의 *특징* 분포만 필요할 뿐 라벨은 전혀 필요하지 않습니다 — 따라서 실제로 라벨이 필요한 위의 방법들과 달리, CORAL의 target 쪽 통계량은 (라벨이 있는 방법들만 제한되는) 희소한 클래스당 10% 하위 집합이 아니라 **전체 target train 분할**(부하당 536–1322개 윈도우)에서 계산됩니다.
- **고전 머신러닝 베이스라인(적응 없음)** — `source_train`만으로 학습한 일반 랜덤 포레스트를, 각 특징 세트별로 target 도메인에 대해 제로샷으로 평가합니다 — 베이스라인 1의 고전 머신러닝 버전입니다. 이것이 필요한 이유는, 이것이 없으면 CORAL이 실제로 얼마나 기여했는지, 아니면 단지 해당 특징 세트 + RF 자체가 이미 얻고 있는 성능인지 구분할 방법이 없기 때문입니다 — 아래 핵심 요약을 참고하세요.
- 정합성 검사(sanity check)에서 희소 하위 집합 선택 로직으로 베이스라인 2의 수치를 다시 계산해 `baseline_results.csv`와 일치함을 확인합니다(작은 차이는 GPU 학습의 비결정성 때문), 이를 통해 `windows_by_load.pkl`(원시 윈도우)과 `features_*.npz` 파일(사전 추출된 특징) 사이의 인덱스 기반 윈도우 대응이 올바르다는 것을 검증합니다.
- **80개 모델을 `models/`에 저장**: 적응 CNN 체크포인트 24개(동결 모드 2가지 × 페어 12개) + CORAL+RF 번들 36개 + 일반 RF 적응-없음 번들 12개(특징 세트 3가지 × 페어 12개 / 부하 4개, `{clf, scaler, pca}`를 joblib으로 덤프), 여기에 앞서 언급한 베이스라인 체크포인트 8개가 더해집니다.
- **통합 정책 테이블** — 10가지 방법 전체의 정확도를 하나의 넓은 테이블로 재구성합니다. 페어당 1행, 방법당 1열이며, **`data/agent_policy_table.csv`**로 저장됩니다. 이것이 다음 단계(주어진 source→target 페어에 대해 어떤 방법을 신뢰할지 결정하는 에이전트)를 위해 의도된 인계 산출물입니다 — 아래 결과를 참고하세요.

**`data/agent_policy_table.csv` — 페어당 1행, 방법당 1열(정확도):**

| source→target | Baseline1 (no adapt) | Baseline2 (target-only) | CNN partial-freeze | CNN full-freeze | RF no-adapt (fft) | RF no-adapt (envelope) | RF no-adapt (fault_freq) | CORAL+RF (fft) | CORAL+RF (envelope) | CORAL+RF (fault_freq) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0→1 | 0.615 | 0.814 | 0.829 | 0.699 | 0.758 | 0.665 | 0.562 | 0.655 | 0.590 | 0.568 |
| 0→2 | 0.668 | 0.839 | 0.901 | 0.758 | 0.795 | 0.658 | 0.655 | 0.770 | 0.671 | 0.537 |
| 0→3 | 0.494 | 0.752 | 0.736 | 0.680 | 0.730 | 0.615 | 0.599 | 0.680 | 0.562 | 0.547 |
| 1→0 | 0.617 | 0.906 | 0.641 | 0.602 | 0.781 | 0.617 | 0.586 | 0.750 | 0.523 | 0.602 |
| 1→2 | 0.919 | 0.839 | 0.910 | 0.922 | 0.811 | 0.860 | 0.596 | 0.898 | 0.835 | 0.739 |
| 1→3 | 0.904 | 0.752 | 0.941 | 0.904 | 0.826 | 0.826 | 0.624 | 0.661 | 0.839 | 0.665 |
| 2→0 | 0.398 | 0.906 | 0.336 | 0.508 | 0.734 | 0.680 | 0.570 | 0.711 | 0.500 | 0.531 |
| 2→1 | 0.839 | 0.814 | 0.904 | 0.857 | 0.854 | 0.829 | 0.655 | 0.907 | 0.811 | 0.724 |
| 2→3 | 0.696 | 0.752 | 0.981 | 0.860 | 0.888 | 0.907 | 0.727 | 0.876 | 0.876 | 0.767 |
| 3→0 | 0.570 | 0.906 | 0.648 | 0.602 | 0.703 | 0.461 | 0.617 | 0.695 | 0.586 | 0.555 |
| 3→1 | 0.780 | 0.814 | 0.891 | 0.786 | 0.717 | 0.602 | 0.581 | 0.826 | 0.814 | 0.637 |
| 3→2 | 0.860 | 0.839 | 1.000 | 0.904 | 0.823 | 0.792 | 0.801 | 1.000 | 0.991 | 0.786 |

모든 페어에서 이기는 단일 방법은 없습니다 — 예를 들어 `RF no-adapt (fft)`는 1→0 페어에서 가장 우수한 제로샷 방법입니다(0.781, 모든 CNN 및 CORAL 변형을 직접 앞섭니다) — 이러한 페어별 편차야말로 에이전트 정책이 실제로 조건으로 삼아야 할 신호입니다.

**결과 — 12개 페어 전체 평균 정확도:**

| 방법 | 평균 정확도 |
|---|---|
| 베이스라인 2 — target-only, 클래스당 10% | 82.7% |
| 적응 CNN (부분 동결) | 81.0% |
| CORAL + 랜덤 포레스트 (FFT 특징) | 78.6% |
| RF, 적응 없음 (FFT 특징) | 78.5% |
| 적응 CNN (완전 동결) | 75.7% |
| CORAL + 랜덤 포레스트 (포락선 특징) | 71.6% |
| RF, 적응 없음 (포락선 특징) | 70.9% |
| 베이스라인 1 — source-only | 69.7% |
| CORAL + 랜덤 포레스트 (fault-freq 특징) | 63.8% |
| RF, 적응 없음 (fault-freq 특징) | 63.1% |

![방법별 평균 정확도](assets/mean_accuracy.png)

**페어별 전체 결과**(위 평균값뿐 아니라 12개 페어 각각의 데이터)는 **`data/full_comparison_results.csv`**와 아래 차트에 있습니다:

![CNN 방법, 12개 부하 페어 전체](assets/cnn_methods_per_pair.png)

![CORAL + 랜덤 포레스트 방법, 12개 부하 페어 전체](assets/coral_methods_per_pair.png)

![CORAL이 일반 RF 대비 실제로 도움이 되는지, 특징 세트별로](assets/rf_coral_vs_noadapt.png)

**핵심 요약:**

- `features_fault_freq.npz`에서 두 가지 구현 결함을 발견하여 수정했습니다: 피크를 읽을 때 사용하는 허용 오차가 FFT 빈 간격보다 좁았던 문제(신호 내용과 무관하게 일부 고조파가 조용히 0으로 처리됨)와, 윈도우별 진폭 정규화가 없었던 문제(원시 크기 값이 각 부하의 전체 진동 진폭에 지배되어, source/target 공분산 스케일이 약 300배 어긋나면서 CORAL이 정렬 과정에서 대부분의 신호를 무너뜨렸습니다). 두 가지를 모두 고치자 CORAL+RF(fault-freq)가 49.1%에서 60%대 초반까지 올라갔습니다.
- RF 적응-없음 베이스라인을 추가한 것은 선택이 아니라 필수였습니다: 이를 통해 **CORAL이 실제로 기여하는 부분은 작고 일관되지도 않다**는 것이 드러났습니다. 평균적으로 CORAL은 특징 세트별로 일반 RF보다 약 1–3점 앞설 뿐이고, 페어별로 보면 과반도 이기지 못합니다(FFT: 4/12, 포락선: 5/12, fault-freq: 7/12) — 이 베이스라인이 없었다면 CORAL+RF의 수치만 보고 실제 근거보다 훨씬 더 "적응이 잘 작동한다"는 인상을 받았을 것입니다.
- CORAL이 **전체** target train 분할을 사용하도록 한 것(실제로는 필요하지도 않은 희소 라벨 하위 집합 대신)은 이론적으로는 올바른 수정이었지만, 실제로는 안정적인 개선을 가져오지 못했습니다: FFT와 포락선은 평균적으로 오히려 소폭 하락했고(데이터는 늘었지만, 이제는 클래스가 균형 잡힌 희소 하위 표본이 아니라 불균형한 클래스 혼합을 반영하게 됨), fault-freq는 소폭 개선되었습니다. 원칙적으로는 맞지만, 이 데이터 규모에서 확실한 승리는 아닙니다.
- CNN의 마지막 합성곱 블록을 부분 동결 해제한 것(75.7% → 81.0%)은 여전히 명확하고 진짜배기인 개선으로 남아 있습니다.
- CNN이든 고전적 방법이든, 어떤 적응 방법도 평균적으로 베이스라인 2를 명확히 **능가하지는** 못합니다 — 가장 우수한 방법들도 그것을 넘어서기보다는 근접할 뿐입니다. 이 과제에 한해서는, source 지식을 활용하는 것이 동일한 양의 희소한 target 데이터로 직접 학습하는 것보다 더 나은 결과를 가져다준다는 것이 아직 입증되지 않았습니다.
- 도메인 격차는 페어마다 다르며 균일하지 않습니다: 예를 들어 2→0에서는 베이스라인 1(39.8%)이 오히려 부분 동결 적응 CNN(33.6%)을 능가합니다 — 적응이 항상 도움이 되는 것은 아니며, 때로는 아무것도 하지 않는 것보다 더 나쁠 수도 있습니다.

## 데이터 레이아웃

```
data/
├── normal_baseline_data/       # 4개 파일: normal_{0,1,2,3}hp.mat
├── 48k_drive_end_fault/        # 52개 파일: 48k_drive_end_fault_<location>_<diameter>in_<load>hp_<rpm>rpm[_<position>]_<file_number>.mat
├── combined_dataset.mat        # 56개 파일 전체 통합, 파일당 1개 구조체, 전체 길이 신호 + 메타데이터
├── windows_by_load.pkl         # DE_time 윈도우(크기 4096), load_hp별로 그룹화, 파일별 train/test 분할
├── features_time.npz           # 원시 윈도우, (5601, 4096), + 윈도우별 메타데이터
├── features_fft.npz            # FFT 크기 스펙트럼, (5601, 2049), + 윈도우별 메타데이터
├── features_envelope.npz       # Hilbert 포락선 스펙트럼, (5601, 2049), + 윈도우별 메타데이터
├── features_fault_freq.npz     # BPFO/BPFI/BSF 피크 크기(RMS 정규화 적용), (5601, 9), + 윈도우별 메타데이터
├── baseline_results.csv        # 베이스라인 1 & 2 정확도/매크로 F1, 부하 페어당 1행 (총 12개)
├── full_comparison_results.csv # 10가지 방법 전체 정확도/매크로 F1, 부하 페어당 1행 (총 12개)
└── agent_policy_table.csv      # 넓은 테이블: 페어당 1행 × 방법 10열, 정확도만 — 에이전트 인계 산출물

models/                          # gitignore 대상 아님 — 총 80개 파일
├── baseline1_full_load{0-3}.pt          # 베이스라인 1 체크포인트 (4개)
├── baseline2_scarce_load{0-3}.pt        # 베이스라인 2 체크포인트 (4개)
├── adapted_cnn_{full,partial}_{S}to{T}.pt   # 적응 CNN 체크포인트 (2가지 모드 × 12페어 = 24개)
├── rf_noadapt_{fault_freq,fft,envelope}_load{0-3}.joblib  # 일반 RF 적응-없음 번들 (3가지 특징 세트 × 부하 4개 = 12개)
└── coral_rf_{fault_freq,fft,envelope}_{S}to{T}.joblib  # CORAL+RF 번들 (3가지 특징 세트 × 12페어 = 36개)

assets/                          # gitignore 대상 아님 — README 차트 이미지
├── mean_accuracy.png
├── cnn_methods_per_pair.png
├── coral_methods_per_pair.png
└── rf_coral_vs_noadapt.png
```

`data/`는 gitignore 대상입니다(대용량 바이너리 파일) — 그 안의 모든 것은 `data_download.ipynb` → `data_splitting_preprocessing.ipynb` → `model_training.ipynb` → `domain_adaptation_evaluation.ipynb` 순서로 실행하면 재현 가능합니다. `models/`와 `assets/`는 gitignore 대상이 아닙니다 — `assets/`에는 이 README의 이미지가 들어 있고, `models/`는 `data/`만큼의 대용량 바이너리 데이터는 아니기 때문입니다.

## 환경 설정

**옵션 A — conda:**

```bash
conda create -n cwru python=3.10
conda activate cwru
pip install -r requirements.txt
```

**옵션 B — venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`의 `torch`는 플랫폼에 맞는 CUDA 빌드를 `pip`가 자동으로 찾아 설치합니다; CPU 전용 환경에서는 CPU 빌드가 설치되며, 어느 쪽이든 별도로 손댈 부분은 없습니다.

실행 순서: `data_download.ipynb`(`data/` 채우기) → `data_splitting_preprocessing.ipynb` → `model_training.ipynb`(`models/`에 베이스라인 체크포인트 8개 생성) → `domain_adaptation_evaluation.ipynb`(`models/`에 체크포인트/번들 72개 추가, `assets/*.png` 재생성, `data/agent_policy_table.csv` 생성).

## 데모 실행

`data/`와 `models/`가 채워진 뒤 (또는 미리 빌드된 `models/` 디렉터리가 이미 있다면), Streamlit UI를 실행합니다:

```bash
streamlit run demo.py
```

브라우저 탭이 열리며 (기본 주소 `http://localhost:8501`) 원시 `.mat` 진동 신호 파일을 업로드할 수 있습니다. 에이전트는 다음을 수행합니다:

- 신호에 기록된 RPM으로부터 작동 조건(부하)을 추론하며, RPM이 없거나 4가지 알려진 부하 중 어느 것과도 충분히 가깝지 않으면 수동 선택으로 대체합니다
- THOUGHT → ACTION → OBSERVATION → DECISION 진단 루프(`agent/diagnose.py`, `agent/react_loop.py`)를 실행하여 `data/agent_policy_table.csv`에서 검증된 최적의 도구 체인을 선택하고, 신뢰도가 낮으면 다른 방법으로 대체합니다
- 예측된 고장 클래스, 신뢰도, 사용된 방법, 그리고 (이 프로젝트 자체 데이터셋의 파일인 경우) 파일명에서 추론한 실제 클래스를 비교용으로 함께 보고합니다

## 아직 부족한 부분 / 다음 단계

- **에이전트 그 자체** — 프로젝트 목표 대비 가장 큰 공백입니다. `data/agent_policy_table.csv`는 (source_load, target_load) 페어별로 어떤 방법을 선택할지 결정하는 정책에 데이터를 제공하기 위해 존재하지만, 아직 그 테이블을 읽고 행동하는 것은 아무것도 없습니다. 이것이 다음으로 계획된 작업입니다.
- **목표 2 (신호 분석)**는 이제 추출 파이프라인(원시/FFT/포락선/고장 주파수 피크)은 갖췄지만, 그 위에서의 *분석*은 아직 없습니다 — 고장 유형/심각도별로 추출된 특징을 비교하는 시각화나 통계가 없고, BPFO/BPFI/BSF 피크가 실제로 정상과 고장 윈도우를 구분해내는지 검증도 없으며, 고전적인 시간 영역 통계 특징(RMS, 첨도, 왜도, crest factor)도 없습니다.
- **목표 3 (방법 선택)**은 에이전트에 필요한 비교 데이터(`agent_policy_table.csv`)는 갖췄지만, "선택" 로직 자체는 아직 없습니다 — 그것이 위에서 말한 에이전트 작업입니다. 그 외 아직 시도하지 않은 것: `features_time`(4번째로 추출한 특징 세트)은 어디에도 사용된 적이 없고, `FE_time`(팬측) 신호나 DE+FE 결합 신호로 어떤 방법도 시도해본 적이 없습니다.
- **목표 4 (전이 학습)**은 이제 실제 적응 방법 두 가지(파인튜닝 CNN, CORAL+랜덤 포레스트)가 있고 둘 다 경쟁력이 있지만, 둘 다 평균적으로 베이스라인 2를 **능가하지는** 못합니다 — "source 지식을 활용하는 것"이 "희소한 target 라벨을 그냥 직접 쓰는 것"보다 이 과제에서 실제로 얼마나 더 나은 가치를 주는지는 아직 입증되지 않았습니다. 아직 시도하지 않은 것: MMD/DANN 방식의 적대적 도메인 적응, 적응 CNN에서 동결 해제된 합성곱 블록과 헤드에 서로 다른 학습률 적용, 그리고 (FFT + fault-freq를 이어붙이는 식으로) 단일 특징 세트가 아니라 결합된 특징 세트에 대한 CORAL 적용.
- **목표 1**은 완료되었습니다 — 범위는 구동측 고장 데이터와 정상 베이스라인 데이터로 한정합니다.

## 에이전트를 위한 계획된 구조

지금까지의 모든 것은 노트북 안에 존재합니다. 다음 단계에서는 이 노트북들에 있는 재사용 가능한 로직을 임포트 가능한 `src/` 패키지로 뽑아낸 뒤, 그 위에 실제 에이전트를 구축합니다:

```
src/                                # 재사용 가능한, 임포트 가능한 코드 — 에이전트가 사용
├── __init__.py
├── data_loading.py                 # load_mat_file(), build_raw_df()
├── preprocessing.py                # split_signal_train_test(), segment_signal()
├── feature_extraction.py           # extract_time(), extract_fft(),
│                                      extract_envelope(), extract_fault_freq()
├── models.py                       # CNN1D 클래스 정의
├── adaptation.py                   # fine_tune(), coral_transform()
└── evaluate.py                     # 정확도/F1/혼동 행렬 헬퍼 함수

agent/                               # 에이전트 워크플로우 — 실제 산출물
├── __init__.py
├── tools.py                         # src/의 함수들을 에이전트가 호출 가능한 도구로 감쌈
├── policy.py                        # comparison_table.csv를 로드하고,
│                                       source→target → 최적 방법 룩업을 구성/조회
├── react_loop.py                    # THOUGHT/ACTION/OBSERVATION/DECISION 오케스트레이션
└── diagnose.py                      # 메인 엔트리 포인트: diagnose(signal, condition)

demo.py                              # Streamlit UI 데모 엔트리 포인트 — 실행: `streamlit run demo.py`
```

- `src/`에는 현재 4개의 노트북(`data_download.ipynb`, `data_splitting_preprocessing.ipynb`, `model_training.ipynb`, `domain_adaptation_evaluation.ipynb`)에 흩어져 있는 로직을 임포트 가능한 모듈로 뽑아내어 담습니다 — 이를 통해 `agent/`가 노트북 코드를 중복 작성하지 않고 직접 호출할 수 있습니다.
- `agent/policy.py`의 `comparison_table.csv`는 이 저장소에 이미 존재하는 **`data/agent_policy_table.csv`**를 가리킵니다 — `domain_adaptation_evaluation.ipynb`에서 이미 만들고 검증한 룩업 테이블입니다.
- `agent/react_loop.py`는 THOUGHT/ACTION/OBSERVATION/DECISION 오케스트레이션 루프입니다 — 이것이 단순한 정적 룩업이 아니라 실제로 "에이전트"이게 만드는 부분입니다.
- `agent/diagnose.py`는 호출자가 사용하는 엔트리 포인트입니다: `diagnose(signal, condition)`.
