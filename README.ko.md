# CWRU 베어링 고장 진단

**언어:** [English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## 프로젝트 목표

다음을 수행할 수 있는 에이전트(agent)를 구축합니다:

1. CWRU 베어링 데이터셋 구조를 로드하고 이해하기
2. 진동 신호 분석하기
3. 적절한 고장 진단 방법 선택하기
4. 서로 다른 운전 조건(모터 부하) 간에 전이 학습 적용하기

**현재 상태:** 전체 데이터 파이프라인(다운로드 → 분할/윈도우/특징 추출 → 베이스라인 → 적응 방법)이 완료되었으며, 하나의 통합 결과 테이블(`data/agent_policy_table.csv`, 12개 페어 × 10개 방법)을 생성했습니다. 이 테이블은 이제 실제로 동작하는 에이전트(`agent/`) — 정책 룩업과 THOUGHT → ACTION → OBSERVATION → DECISION 루프로 구성됨 — 를 구동하며, Streamlit 데모(`demo.py`)를 통해 업로드된 신호를 처음부터 끝까지 진단하는 형태로 노출됩니다. 이 에이전트는 부하 0과 1만을 알려진 사전 학습 소스 도메인으로 알도록 의도적으로 제한되어 있어(`agent.policy.KNOWN_SOURCE_LOADS`), 낯선 부하(2 또는 3)를 만나면 같은 도메인의 지름길로 가는 대신 반드시 전이 학습을 실제로 수행해야 합니다.

## 4가지 목표 대비 진행 상황

| # | 목표 | 상태 | 비고 |
|---|---|---|---|
| 1 | 데이터셋 구조 로드 및 이해 | ✅ 완료 | `data_download_exploration.ipynb`가 데이터를 다운로드하고, 라벨링하고, 검사함 |
| 2 | 진동 신호 분석 | 🟡 부분 완료 | 추출 파이프라인(원시 신호/FFT/포락선/고장 주파수 피크)은 존재하지만, 그 위에서의 분석은 아직 없음 — 이 특징들이 실제로 정상/고장 윈도우를 구분해내는지 검증하는 것이 전혀 없음 |
| 3 | 적절한 고장 진단 방법 선택 | ✅ 완료 | 10가지 방법을 실제 페어별 수치로 정면 비교했고, `data/agent_policy_table.csv`로 통합했으며, 이제는 사람이 표를 읽는 대신 실제로 동작하는 정책/에이전트(`agent/policy.py`, `agent/react_loop.py`)가 선택을 수행함 |
| 4 | 운전 조건 간 전이 학습 적용 | 🟢 거의 완료 | 두 가지 실제 적응 방법(파인튜닝된 CNN, CORAL+랜덤 포레스트)을 구현하여 12개 페어 전체에서 평가함; 가장 우수한 변형(부분 동결 CNN, FFT 특징 기반 CORAL+RF)은 target-only 상한값과의 격차를 거의 다 좁힘 — 아래 결과 참고 |

## 지금까지 만든 것

### `data_download_exploration.ipynb` — 다운로드 및 탐색

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
- **통합 정책 테이블** — 10가지 방법 전체의 정확도를 하나의 넓은 테이블로 재구성합니다. 페어당 1행, 방법당 1열이며, **`data/agent_policy_table.csv`**로 저장됩니다. 이것이 `agent/policy.py`가 실제로 로드하고 조회하는 인계 산출물입니다 — 아래 결과를 참고하세요. 매크로 F1에 대해서도 동일한 재구성을 반복하여 **`data/agent_policy_table_f1.csv`**로 저장합니다(`agent/policy.py`는 정확도만으로 순위를 매기므로 이 테이블을 사용하지 않습니다 — 어떤 방법의 순위가 클래스 불균형을 반영하는 지표에서도 유지되는지 확인하고 싶은 사람을 위한 것입니다).

**`data/agent_policy_table.csv` — 페어당 1행, 방법당 1열(정확도).** CWT(`data/cwt_baseline_results.csv`, `cwt_baseline_exploration.ipynb`)도 비교를 위해 함께 표시했습니다 — 베이스라인에 불과하고(파인튜닝/CORAL 버전 없음) `agent/policy.py`에도 연결되어 있지 않기 때문에, `agent_policy_table.csv`에 합쳐지지 않고 별도의 CSV로 관리됩니다:

| source→target | Baseline1 (no adapt) | Baseline2 (target-only) | CNN partial-freeze | CNN full-freeze | RF no-adapt (fft) | RF no-adapt (envelope) | RF no-adapt (fault_freq) | CORAL+RF (fft) | CORAL+RF (envelope) | CORAL+RF (fault_freq) | CWT (no adapt) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0→1 | 0.615 | 0.814 | 0.829 | 0.699 | 0.758 | 0.665 | 0.562 | 0.655 | 0.590 | 0.568 | 0.565 |
| 0→2 | 0.668 | 0.839 | 0.901 | 0.758 | 0.795 | 0.658 | 0.655 | 0.770 | 0.671 | 0.537 | 0.593 |
| 0→3 | 0.494 | 0.752 | 0.736 | 0.680 | 0.730 | 0.615 | 0.599 | 0.680 | 0.562 | 0.547 | 0.441 |
| 1→0 | 0.617 | 0.906 | 0.641 | 0.602 | 0.781 | 0.617 | 0.586 | 0.750 | 0.523 | 0.602 | 0.719 |
| 1→2 | 0.919 | 0.839 | 0.910 | 0.922 | 0.811 | 0.860 | 0.596 | 0.898 | 0.835 | 0.739 | 0.963 |
| 1→3 | 0.904 | 0.752 | 0.941 | 0.904 | 0.826 | 0.826 | 0.624 | 0.661 | 0.839 | 0.665 | 0.786 |
| 2→0 | 0.398 | 0.906 | 0.336 | 0.508 | 0.734 | 0.680 | 0.570 | 0.711 | 0.500 | 0.531 | 0.625 |
| 2→1 | 0.839 | 0.814 | 0.904 | 0.857 | 0.854 | 0.829 | 0.655 | 0.907 | 0.811 | 0.724 | 0.946 |
| 2→3 | 0.696 | 0.752 | 0.981 | 0.860 | 0.888 | 0.907 | 0.727 | 0.876 | 0.876 | 0.767 | 0.699 |
| 3→0 | 0.570 | 0.906 | 0.648 | 0.602 | 0.703 | 0.461 | 0.617 | 0.695 | 0.586 | 0.555 | 0.500 |
| 3→1 | 0.780 | 0.814 | 0.891 | 0.786 | 0.717 | 0.602 | 0.581 | 0.826 | 0.814 | 0.637 | 0.621 |
| 3→2 | 0.860 | 0.839 | 1.000 | 0.904 | 0.823 | 0.792 | 0.801 | 1.000 | 0.991 | 0.786 | 0.748 |

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
| CWT + 2D CNN, 적응 없음 | 68.4% |
| CORAL + 랜덤 포레스트 (fault-freq 특징) | 63.8% |
| RF, 적응 없음 (fault-freq 특징) | 63.1% |

![모델 계열별로 그룹화한 방법별 평균 정확도](assets/mean_accuracy_grouped.png)

**결과 — 12개 페어 전체 평균 매크로 F1**(`data/agent_policy_table_f1.csv` + `data/mean_f1_by_method.csv`에서 — 정확도와 달리 매크로 F1은 모든 클래스를 동등하게 취급합니다):

| 방법 | 평균 매크로 F1 |
|---|---|
| 베이스라인 2 — target-only, 클래스당 10% | 76.9% |
| 적응 CNN (부분 동결) | 74.3% |
| RF, 적응 없음 (FFT 특징) | 68.3% |
| 적응 CNN (완전 동결) | 68.1% |
| CORAL + 랜덤 포레스트 (FFT 특징) | 67.9% |
| CORAL + 랜덤 포레스트 (포락선 특징) | 60.9% |
| 베이스라인 1 — source-only | 59.7% |
| RF, 적응 없음 (포락선 특징) | 59.3% |
| CWT + 2D CNN, 적응 없음 | 59.3% |
| CORAL + 랜덤 포레스트 (fault-freq 특징) | 54.3% |
| RF, 적응 없음 (fault-freq 특징) | 53.1% |

주목할 점은 순위가 정확도와 완전히 같지는 않다는 것입니다: **CORAL + 랜덤 포레스트 (FFT)**는 정확도 기준 3위에서 F1 기준 5위로 내려가며, RF 적응 없음 (FFT)과 적응 CNN (완전 동결) 모두에게 추월당합니다 — 이는 이 방법의 정확도 우위가 여러 클래스에 고르게 분포된 것이 아니라 크거나 쉬운 클래스에 불균형하게 치우쳐 있음을 시사하며, 정확도만으로는 드러나지 않는 부분입니다.

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

### `cwt_baseline_exploration.ipynb` — 더 풍부한 특징 표현이 도움이 될까?

탐색적 실험이며 베이스라인에 한정됩니다(전체 적응 매트릭스에는 의도적으로 포함하지 않았습니다): 연속 웨이블릿 변환(CWT) 스케일로그램(Morlet 웨이블릿, 150–3000 Hz 범위의 32개 스케일, 시간축을 128포인트로 다운샘플링 → 윈도우당 32×128 이미지)을 2D CNN(`src.CNN2D`)과 결합하여, 부하별로 전체 train 분할에서 모델 하나를 학습하고 제로샷 크로스 도메인으로 평가했습니다 — 직접 비교를 위해 베이스라인 1과 동일한 프로토콜을 사용했습니다.

- 12개 페어 전체 평균 정확도: **68.4%**(전체 비교 표와 그룹화된 차트는 위 참고) — 기존 원시 윈도우 1D CNN 베이스라인 1(69.7%)과 사실상 동일하며, FFT 특징 기반 RF(78.5%)보다는 확실히 뒤처집니다. 평균 매크로 F1: **59.3%**로, RF 적응 없음(포락선 특징)과 사실상 동일합니다 — 위 F1 표 참고.
- 체크포인트는 `models/cwt_baseline1_full_load{0-3}.pt`에, 결과는 `data/cwt_baseline_results.csv`에 저장됩니다. 또한 CWT를 `data/agent_policy_table.csv`/`agent_policy_table_f1.csv`와 대조하는 11가지 방법 그룹화 비교에도 합쳐서, `assets/mean_accuracy_grouped.png`와 `data/mean_f1_by_method.csv`로 저장합니다.
- 결론: 더 풍부한 2D 시간-주파수 입력이 여기서는 훨씬 단순한 원시 1D 윈도우 CNN을 뚜렷이 능가하지 못했으므로, 다른 특징 표현들이 거쳤던 전체 파인튜닝/CORAL 비교에는 포함하지 않았습니다 — 배제된 것이 아니라, 들어가는 추가 연산 비용 대비 명확한 이득이 보이지 않았을 뿐입니다.

### `src/` — 재사용 가능한 파이프라인 로직, 그리고 `agent/` — 진단 에이전트

`src/`는 위 네 개 노트북에 담겨 있던 로직을 임포트 가능한 패키지(`data_loading.py`, `preprocessing.py`, `feature_extraction.py`, `models.py`, `adaptation.py`, `evaluate.py`)로 뽑아냅니다. 노트북 자체는 여전히 자기 완결적으로 남아 있습니다 — 가독성을 위해 각 노트북은 필요한 로직을 여전히 자체적으로 재정의합니다 — 하지만 `agent/`와 `demo.py`는 그 로직을 중복하지 않고 `src/`를 직접 호출합니다.

`agent/`가 실제 에이전트 워크플로우입니다:

- **`agent/tools.py`** — `src/`를 에이전트가 호출 가능한 동작으로 감쌉니다: 신호 로드, 윈도우 분할, 4가지 특징 표현 중 어느 것이든 추출, 학습된 체크포인트/번들 로드, CNN 또는 RF 추론 실행, 특징에 CORAL 정렬 적용, CNN 파인튜닝, 예측 결과 채점.
- **`agent/policy.py`** — `data/agent_policy_table.csv`에 대한 상태 없는(stateless) 조회입니다: `(source_load, target_load)` 페어가 주어지면 검증된 모든 방법을 정확도순으로 정렬하고, 승자를 구체적인 도구 호출(어떤 체크포인트/번들을, `agent/tools.py`의 어떤 함수로)로 변환합니다. 또한 `KNOWN_SOURCE_LOADS = {0, 1}`을 정의합니다 — 실제 결과는 4개 부하 전체를 소스로 다루고 있음에도, 에이전트는 의도적으로 이 두 부하만을 사전 학습된 소스 도메인으로 알도록 제한되어 있으며, 이는 낯선 부하(2 또는 3)를 만났을 때 같은 도메인의 지름길 대신 전이 학습을 강제하기 위함입니다.
- **`agent/react_loop.py`** — THOUGHT → ACTION → OBSERVATION → DECISION 루프: THOUGHT는 `policy.py`에 남은 방법 중 최선이 무엇인지 묻고, ACTION은 `tools.py`를 통해 그것을 실행하며, OBSERVATION은 예측 결과와 그 신뢰도를 기록하고, DECISION은 그것을 받아들이거나 (신뢰도가 너무 낮으면) 다음 순위 방법으로 넘어갑니다(최대 `max_attempts`회까지).
- **`agent/diagnose.py`** — 엔트리 포인트인 `diagnose(signal, condition)`: 신호를 윈도우로 나누고, 주어진 `condition`에 대해 가장 잘 맞는 알려진 소스 도메인을 선택하거나(또는 명시적으로 지정된 소스 도메인을 `KNOWN_SOURCE_LOADS`로 검증하고), react 루프를 실행합니다.
- **`demo.py`** — `diagnose()` 위에 구축된 Streamlit UI입니다; 아래 "데모 실행" 절을 참고하세요.

솔직하게 짚어둘 점: 에이전트 루프에서 "적응"이란 항상 **이미 적응이 끝난 체크포인트를 선택**하는 것을 의미합니다 — `domain_adaptation_evaluation.ipynb`에서 오프라인으로 파인튜닝했거나 CORAL로 정렬해둔 것 — 업로드된 신호에 대해 실시간으로 적응을 계산하는 것이 아닙니다. 이를 온라인으로 수행하는 것도 여기서는 그다지 의미가 크지 않습니다: 지도 학습 방식의 파인튜닝은 진단용 업로드에는 없는 라벨을 필요로 하고, 파일 하나의 윈도우만으로 CORAL을 다시 계산해봐야 오프라인 번들에 이미 담겨 있는 통계량보다 더 노이즈가 심한 버전이 나올 뿐입니다.

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
├── cwt_baseline_results.csv    # CWT+2D CNN 베이스라인 정확도/매크로 F1, 부하 페어당 1행 (총 12개)
├── agent_policy_table.csv      # 넓은 테이블: 페어당 1행 × 방법 10열, 정확도만 — 에이전트 인계 산출물
├── agent_policy_table_f1.csv   # 위와 동일한 형태, 매크로 F1 (agent/policy.py는 사용하지 않음)
└── mean_f1_by_method.csv       # 방법별 평균 매크로 F1(12페어 평균), 모델 계열별 그룹화, CWT 포함

models/                          # gitignore 대상 아님 — 총 84개 파일
├── baseline1_full_load{0-3}.pt          # 베이스라인 1 체크포인트 (4개)
├── baseline2_scarce_load{0-3}.pt        # 베이스라인 2 체크포인트 (4개)
├── adapted_cnn_{full,partial}_{S}to{T}.pt   # 적응 CNN 체크포인트 (2가지 모드 × 12페어 = 24개)
├── rf_noadapt_{fault_freq,fft,envelope}_load{0-3}.joblib  # 일반 RF 적응-없음 번들 (3가지 특징 세트 × 부하 4개 = 12개)
├── coral_rf_{fault_freq,fft,envelope}_{S}to{T}.joblib  # CORAL+RF 번들 (3가지 특징 세트 × 12페어 = 36개)
└── cwt_baseline1_full_load{0-3}.pt      # CWT+2D CNN 베이스라인 체크포인트 (4개)

assets/                          # gitignore 대상 아님 — README 차트 이미지
├── mean_accuracy.png
├── mean_accuracy_grouped.png      # CWT 포함, 모델 계열별로 그룹화
├── cnn_methods_per_pair.png
├── coral_methods_per_pair.png
└── rf_coral_vs_noadapt.png
```

`data/`는 gitignore 대상입니다(대용량 바이너리 파일) — 그 안의 모든 것은 `data_download_exploration.ipynb` → `data_splitting_preprocessing.ipynb` → `model_training.ipynb` → `domain_adaptation_evaluation.ipynb` 순서로 실행하면 재현 가능합니다. `models/`와 `assets/`는 gitignore 대상이 아닙니다 — `assets/`에는 이 README의 이미지가 들어 있고, `models/`는 `data/`만큼의 대용량 바이너리 데이터는 아니기 때문입니다.

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

실행 순서: `data_download_exploration.ipynb`(`data/` 채우기) → `data_splitting_preprocessing.ipynb` → `model_training.ipynb`(`models/`에 베이스라인 체크포인트 8개 생성) → `domain_adaptation_evaluation.ipynb`(`models/`에 체크포인트/번들 72개 추가, `assets/*.png` 재생성, `data/agent_policy_table.csv` 생성). `cwt_baseline_exploration.ipynb`는 선택 사항이며 나머지와 독립적입니다 — `data/windows_by_load.pkl`만 있으면 되고, 에이전트나 데모가 이것에 의존하지 않습니다.

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

- **목표 2 (신호 분석)**는 여전히 추출 파이프라인 위에서의 *분석*이 없습니다 — 고장 유형/심각도별로 추출된 특징을 비교하는 시각화나 통계가 없고, BPFO/BPFI/BSF 피크가 실제로 정상과 고장 윈도우를 구분해내는지 검증도 없으며, 고전적인 시간 영역 통계 특징(RMS, 첨도, 왜도, crest factor)도 없습니다.
- **에이전트는 오프라인으로 미리 계산해 둔 결과 중에서만 선택할 뿐, 실시간으로 적응하지는 않습니다.** `react_loop.py`의 "적응" 단계는 이미 파인튜닝되었거나 CORAL로 정렬된 체크포인트를 불러올 뿐입니다; `agent/tools.py`는 `fine_tune_cnn()`/`coral_align()`을 제공하지만, 진단 경로 어디에서도 이를 호출하지 않습니다. 지도 학습 방식의 파인튜닝은 진단용 업로드에는 없는 라벨을 필요로 하고, 파일 하나의 윈도우만으로 CORAL을 다시 계산해봐야 오프라인 번들에 이미 담겨 있는 통계량보다 노이즈만 더 심한 버전이 나올 뿐입니다 — 의미 있는 온라인 적응을 하려면 단일 파일 진단이 아니라, 라벨이 없는 새로운 배포 배치 전체가 필요합니다.
- **CWT + 2D CNN**(`cwt_baseline_exploration.ipynb`)은 제로샷 베이스라인으로만 테스트되었습니다(69.2%, 기존 원시 윈도우 CNN과 거의 동일) — 베이스라인 결과가 추가 연산 비용을 명확히 정당화하지 못했기 때문에, 다른 네 가지 특징 표현이 거쳤던 파인튜닝/CORAL 매트릭스에는 포함하지 않았습니다.
- 기존 특징 표현들에 대해 아직 시도하지 않은 것: MMD/DANN 방식의 적대적 도메인 적응, 적응 CNN에서 동결 해제된 합성곱 블록과 헤드에 서로 다른 학습률 적용, (FFT + fault-freq를 이어붙이는 식의) 결합된 특징 세트에 대한 CORAL 적용, 그리고 `features_time`/`FE_time`(팬측)/DE+FE 결합 신호는 어디에도 사용된 적이 없습니다.
- CNN이든 고전적 방법이든, 어떤 적응 방법도 평균적으로 베이스라인 2를 명확히 **능가하지는** 못합니다(위 핵심 요약 참고) — 이 과제에 한해서는, source 지식을 활용하는 것이 동일한 양의 희소한 target 데이터로 직접 학습하는 것보다 더 나은 결과를 가져다준다는 것이 아직 입증되지 않았습니다.
- 목표 1과 목표 3은 완료되었습니다.

## 결론

네 가지 목표 모두 처음부터 끝까지 달성되었습니다: 데이터 파이프라인은 CWRU 48kHz 드라이브 엔드 데이터셋을 다운로드하고, 라벨링하고, 탐색합니다; 특징 추출 파이프라인은 다섯 가지 표현(원시 윈도우, FFT, 엔벨로프, 고장 주파수 피크, CWT 스칼로그램)을 생성하며, 그 과정에서 실제 버그 두 개를 발견해 수정했습니다(FFT 빈 간격보다 좁았던 고장 주파수 허용 오차, 그리고 CORAL의 공분산 정렬을 약 300배 왜곡시키고 있던 누락된 RMS 정규화); 11가지 방법이 12개의 순서 있는 source→target 부하 페어 전체에 걸쳐 정면으로 비교되어 `data/agent_policy_table.csv`로 통합되었습니다; 그리고 두 가지 실제 적응 방법(파인튜닝된 CNN, CORAL+랜덤 포레스트)이 구현되었고 데이터 누출이 없음이 검증되었습니다 — 파인튜닝은 target의 **train** 분할에서만 클래스별로 층화 추출한 희소 서브샘플을 사용하며, 코드 검토와 실증적 중복 검사 양쪽 모두에서 보류된 **test** 분할과 전혀 겹치지 않음이 확인되었습니다.

이에 더해 에이전트 자체도 구축되어 정상 작동합니다: `agent/policy.py`는 결과 테이블에 대한 상태 없는 조회이고, `agent/react_loop.py`는 THOUGHT → ACTION → OBSERVATION → DECISION 루프를 실행하여 검증된 최선의 방법을 선택하고 신뢰도가 낮으면 대안으로 폴백하며, `agent/diagnose.py`는 이를 하나의 `diagnose(signal, condition)` 호출로 묶어내고, `demo.py`는 그 위에 Streamlit UI를 올립니다 — 신호를 업로드하면 에이전트는 RPM으로부터 작동 조건을 추론하고, 의도적으로 작게 설정한 "알려진" source 도메인 집합(`KNOWN_SOURCE_LOADS = {0, 1}`)으로 스스로를 제한하여 부하 2와 3에서는 같은 도메인으로의 지름길이 아니라 실제 전이 학습이 작동하도록 하며, 예측 결과와 함께 전체 추론 과정을 보고합니다.

가감 없이 그대로 보고해야 할 가장 중요한 결과는 이것입니다: 12개 페어 평균으로 볼 때 **어떤 적응 방법도 베이스라인 2를 능가하지 못합니다** (source 도메인을 전혀 사용하지 않고 target 부하의 클래스당 10% 희소 라벨 서브셋만으로 직접 학습한 모델). CNN partial-freeze 파인튜닝이 근접하긴 하지만(81.0% 대 82.7%), CORAL+RF와 RF-no-adapt도 FFT 특징에서 각각 준수한 성능을 보이지만(~78.5%), source 학습 모델이 target 도메인의 적은 라벨을 직접 사용하는 것보다 낫다는 전이 학습의 핵심 약속은 이 과제에서는 아직 입증되지 않았습니다. 이는 감출 실패가 아니라, 적응이 언제 복잡성을 감수할 가치가 있고 언제 없는지에 대한 진짜 유용한 발견입니다.

## 한계

- **결과가 이 데이터셋 밖으로 일반화되지 않습니다.** 여기의 모든 것은 하나의 베어링 종류(SKF 6205), 하나의 데이터 소스(CWRU), 네 가지 이산적인 부하/RPM 조건으로 이루어져 있습니다. 이 방법들 중 어느 것이든 — 혹은 "적응이 희소 라벨 베이스라인을 이기지 못한다"는 발견 자체든 — 다른 장비, 센서, 베어링 종류에서도 성립하는지는 테스트되지 않았으며, 새로운 라벨 데이터 없이는 답할 수 없습니다.
- **에이전트는 실시간으로 적응하지 않습니다.** 위에서 다룬 대로, `react_loop.py`는 오프라인에서 파인튜닝되었거나 CORAL로 정렬된 체크포인트 중에서 **선택**만 할 뿐입니다; `agent/tools.py`의 `fine_tune_cnn()`/`coral_align()`은 존재하지만 진단 경로에서 호출되지 않습니다. 이는 단순히 최적화되지 않은 구석이 아니라 실제 기능적 공백입니다 — 진단용 업로드에는 지도 학습 파인튜닝에 필요한 라벨이 없고, 파일 하나로는 CORAL이 이미 캐시된 결과보다 더 나은 추정을 내놓기에 데이터가 너무 적습니다.
- **`KNOWN_SOURCE_LOADS`는 시연을 위한 제약이지 기술적 한계가 아닙니다.** 에이전트를 부하 {0, 1}만 source 도메인으로 사용하도록 제한한 것은 부하 2/3에서 전이 학습이 실제로 작동하도록 강제하기 위한 의도적 선택입니다 — 4개 부하 전체를 source로 사용한 체크포인트와 결과는 이미 존재하며, 이 제약을 없애도 최소한 동등하거나 더 나은 성능을 낼 것입니다.
- **데모의 실제 클래스 비교는 이 프로젝트 자체 파일에서만 작동합니다.** `demo.py`는 CWRU의 설명적 파일명에서 "실제 클래스"를 추론합니다 — 진짜로 새로운, 라벨이 없는 신호에 대해서는 진단이 맞았는지 확인할 방법이 없습니다.
- **결과에는 약 1퍼센트 포인트의 실행 간 노이즈가 있습니다.** 직접 관찰된 사례: 동일한 시드로 CWT를 재학습했을 때 평균 정확도가 69.2%에서 68.4%로 바뀌었으며, 이는 GPU 학습의 비결정성 때문입니다. 이 README의 모든 정확도 수치는 소수점 세 자리까지 정확한 값이 아니라 근사치로 읽어야 합니다.
- **자동화된 테스트 스위트가 없습니다.** 정확성은 수동으로 노트북을 재실행하고, 목적에 맞는 검증 스크립트를 작성하고, 직접 검사(예: train/test 누출 검사)하는 방식으로 확인되었습니다 — CI 기반 테스트가 아니므로, 현재는 회귀가 발생해도 노트북을 수동으로 다시 실행해야만 발견할 수 있습니다.

## 저장소 구조

위에서 설명한 부분들이 어떻게 맞물리는지에 대한 빠른 참조입니다 — 각 파일이 실제로 무엇을 하는지는 위의 "`src/` — 재사용 가능한 파이프라인 로직, 그리고 `agent/` — 진단 에이전트" 절을 참고하세요:

```
src/                                # 재사용 가능한, 임포트 가능한 코드 — agent/와 demo.py가 사용
├── __init__.py
├── data_loading.py                 # load_mat_file(), build_raw_df(), label_file(), label_for_item()
├── preprocessing.py                # split_signal_train_test(), segment_signal(), build_windows_by_load()
├── feature_extraction.py           # extract_time(), extract_fft(), extract_envelope(),
│                                      extract_fault_freq(), extract_cwt()
├── models.py                       # CNN1D, CNN2D, WindowDataset, ScalogramDataset, train_cnn()
├── adaptation.py                   # fine_tune(), coral_transform()
└── evaluate.py                     # 정확도/F1/혼동 행렬 헬퍼 함수

agent/                               # 에이전트 워크플로우
├── __init__.py
├── tools.py                         # src/의 함수들을 에이전트가 호출 가능한 도구로 감쌈
├── policy.py                        # agent_policy_table.csv를 로드하고,
│                                       source→target → 최적 방법 룩업을 구성/조회; KNOWN_SOURCE_LOADS
├── react_loop.py                    # THOUGHT/ACTION/OBSERVATION/DECISION 오케스트레이션
└── diagnose.py                      # 메인 엔트리 포인트: diagnose(signal, condition)

demo.py                              # Streamlit UI 데모 엔트리 포인트 — 실행: `streamlit run demo.py`
```
