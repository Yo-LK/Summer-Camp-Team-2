# 데이터 전처리 변경 이력 (CHANGELOG)

보고서의 "데이터 전처리" 또는 "방법론" 섹션에 그대로 인용 가능하도록 정리한 문서입니다.

---

## 변경 1: `window_index.csv`에 `split_t0` ~ `split_t3` 컬럼 추가

### 배경 / 문제 제기
기존 `window_index.csv`는 도메인 전이 태스크를 **`0,1,2hp(source) → 3hp(target)` 단 1개 조합**으로만
정의하고 있었다. 이 상태로 전이학습(CORAL/TCA, DANN)의 효과를 평가하면, 관측된 성능 향상이
"전이학습 방법론 자체의 효과"인지 "우연히 비교적 쉬운 도메인 조합을 골랐기 때문"인지 구분할 수 없다는
문제가 제기되었다. 실제로 부하(load) 조합별 zero-shot 성능 편차가 크게 나타나는 것으로 보고되었다
(예: 2hp→3hp는 상대적으로 쉬운 반면, 0hp→3hp, 3hp→0hp 방향은 훨씬 어려움).

### 조치
Leave-one-load-out(LOLO) 방식으로 태스크를 4개로 확장했다.

| 컬럼 | target(적응 대상) | source(학습 도메인) |
|---|---|---|
| `split_t0` | load_hp = 0 | load_hp = 1, 2, 3 |
| `split_t1` | load_hp = 1 | load_hp = 0, 2, 3 |
| `split_t2` | load_hp = 2 | load_hp = 0, 1, 3 |
| `split_t3` | load_hp = 3 | load_hp = 0, 1, 2 |

각 컬럼 내부의 값은 기존과 동일한 4범주(`source_train` / `source_validation` /
`target_adaptation` / `target_test`)를 사용한다.

**분할 규칙** (레코드 단위, 시간순 분할 — 특정 시점 이후를 validation/test로 분리하여
윈도우 셔플로 인한 시간적 누수를 방지):
- source 도메인 내부: 레코드별 시간순 80% → `source_train`, 나머지 20% → `source_validation`
- target 도메인 내부: 레코드별 시간순 20% → `target_adaptation`, 나머지 80% → `target_test`
- 윈도우가 8,192샘플·50% overlap으로 생성되어 있으므로, 분할 경계에 걸친 윈도우 1개는
  train/validation 어느 쪽에도 포함하지 않고 제외한다 (경계 데이터 누수 방지).
  → 레코드 56개 × 1개 = 총 56개 윈도우가 각 `split_tX` 컬럼에서 결측(NaN) 처리됨
  (학습·평가 어디에도 사용하지 않음. 버그가 아니라 의도된 처리).

### 검증 중 발견한 이슈 및 결정 사항
`split_t3`(target=3hp)는 기존 `split` 컬럼과 동일한 태스크 정의이므로 완전히 일치할 것으로
예상되었으나, 검증 결과 **95개 윈도우(전체의 1.8%)에서 값이 불일치**했다:
- 30건: 기존 `split`=`source_train` → 재생성한 `split_t3`=`source_validation`
- 9건: 기존 `split`=`target_adaptation` → 재생성한 `split_t3`=`target_test`

원인은 두 스크립트 간 분할 경계(cut point) 계산 방식의 미세한 차이로 추정된다(반올림 또는
경계 인덱스 처리 방식 차이). 이 차이가 기존에 보고된 베이스라인 성능(RandomForest 기준
target_test macro-F1 85.1%)의 재현성에 영향을 줄 수 있다고 판단하여, 다음과 같이 결정했다:

> **기존 `split` 컬럼은 변경하지 않고 그대로 유지하며, target=3hp 태스크의 기준(baseline)은
> 계속 `split`을 사용한다. 새로 추가된 `split_t3`는 참고용으로만 저장하고 실제 실험에는
> `split_t0`, `split_t1`, `split_t2`, `split`(기존, target=3hp) 이렇게 4개 조합을 사용한다.**

이는 이미 검증을 마친 베이스라인 수치와의 일관성을 유지하기 위한 결정이다.

### 변경 파일
- `data/window_index.csv` (컬럼 5183행 × 11열 → 5183행 × 15열, `split_t0`~`split_t3` 추가)
- `data/window_index_backup.csv` (변경 전 원본 백업)
- 변경 스크립트: `src/data/extend_splits_and_flags.py`

---

## 변경 2: `data_audit.csv`에 `known_issue` 컬럼 추가

### 배경 / 문제 제기
기존 데이터 품질 점검(`data_audit.csv`)은 NaN·무한값·상수 신호 여부만 검사했다. 그러나 CWRU
데이터셋을 다룬 선행 문헌(Smith & Randall, 2015)의 Table 3에서는 이 자동 검사로는 걸러지지
않는, 신호 자체의 물리적 이상이 있는 레코드 11개를 명시적으로 지적하고 있다.

### 조치
레코드를 **삭제하거나 학습에서 제외하지 않고**, 문제 유형을 라벨링하는 `known_issue` 컬럼만
추가했다. 포함/제외 여부에 따른 성능 차이를 이후 비교 실험(ablation)으로 검증하기 위함이다.

| known_issue 값 | 해당 record_id | 설명 |
|---|---|---|
| `clipped` | 191, 214, 215, 228, 229 | 신호 진폭이 센서/ADC 포화로 잘려나간(clipping) 것으로 보고된 레코드 |
| `electrical_noise` | 177 | 전기적 노이즈 혼입이 보고된 레코드 |
| `de_fe_duplicate` | 189, 201, 213, 226, 238 | DE(Drive-End)와 FE(Fan-End) 채널이 스케일 상수배 관계로, 사실상 중복 정보로 보이는 레코드 |
| `none` | 나머지 45개 | 알려진 이슈 없음 |

우리 프로젝트의 56개 서브셋(48kHz, Drive-End) 기준으로 위 11개 레코드가 모두 포함되어 있는
것을 확인했다 (플래그 대상 11개 전부 audit 테이블에 존재).

### 활용 방침
- 기본 실험(베이스라인, CORAL/TCA, DANN)은 `known_issue` 유무와 관계없이 56개 레코드 전체를
  사용하여 기존 수치와의 일관성을 유지한다.
- 추가 ablation으로 `known_issue != 'none'`인 레코드를 제외했을 때 성능이 유의미하게
  달라지는지 별도로 검증하고, 그 결과를 보고서의 한계점(limitation) 또는 부록에 기술한다.

### 변경 파일
- `data/data_audit.csv` (컬럼 18개 → 19개, `known_issue` 추가)
- `data/data_audit_backup.csv` (변경 전 원본 백업)
- 변경 스크립트: `src/data/extend_splits_and_flags.py` (변경 1과 동일 스크립트에서 함께 처리)

---

## 요약 (보고서용 한 문단)

> 단일 도메인 전이 태스크(0,1,2hp→3hp)만으로는 전이학습 기법의 효과를 방법론 자체의 효과와
> 우연한 태스크 선택 효과로 구분하기 어렵다는 문제 제기에 따라, leave-one-load-out 방식으로
> 4개의 전이 태스크(`split_t0`~`split_t3`, 각 부하 조건을 한 번씩 target으로 지정)를
> `window_index.csv`에 추가했다. 검증 과정에서 새로 생성한 `split_t3`가 기존 `split`
> 컬럼과 1.8%(95/5,183 윈도우) 불일치하는 것을 발견했으며, 재현성을 위해 기존 `split`
> 컬럼(target=3hp 태스크의 기준)은 그대로 유지하고 나머지 3개 태스크만 신규 컬럼을 사용하기로
> 결정했다. 또한 선행 문헌에서 보고된 클리핑·전기노이즈·채널중복 이슈가 있는 11개 레코드에
> `known_issue` 플래그를 추가하여, 향후 해당 레코드 포함/제외에 따른 성능 비교(ablation)를
> 수행할 수 있도록 했다.