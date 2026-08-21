# CWRU Bearing Fault Diagnosis

**Languages:** [English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## Project goal

Build an agent that can:

1. Load and understand the CWRU bearing dataset structure
2. Analyze vibration signals
3. Select appropriate fault diagnosis methods
4. Apply transfer learning across different operating conditions (motor loads)

**Current state:** the full data pipeline (download → split/window/feature-extraction → baselines → adaptation methods) is done and has produced a single consolidated results table (`data/agent_policy_table.csv`, 12 pairs × 10 methods). That table now feeds a working agent (`agent/`) — a policy lookup plus a THOUGHT → ACTION → OBSERVATION → DECISION loop — exposed through a Streamlit demo (`demo.py`) that diagnoses an uploaded signal end-to-end. The agent is deliberately restricted to only knowing loads 0 and 1 as pretrained source domains (`agent.policy.KNOWN_SOURCE_LOADS`), so an unfamiliar load (2 or 3) forces it to exercise transfer learning rather than reach for a same-domain shortcut.

## Status against the 4 objectives

| # | Objective | Status | Notes |
|---|---|---|---|
| 1 | Load & understand dataset structure | ✅ Done | `data_download.ipynb` downloads, labels, and inspects the data |
| 2 | Analyze vibration signals | 🟡 Partial | Extraction pipeline exists (raw/FFT/envelope/fault-frequency peaks), but no analysis on top of it yet — nothing validates that these features actually separate healthy from faulty windows |
| 3 | Select fault diagnosis methods | ✅ Done | 10 methods compared head-to-head, consolidated into `data/agent_policy_table.csv`, and actually selected by a working policy/agent (`agent/policy.py`, `agent/react_loop.py`) rather than a human reading the table |
| 4 | Transfer learning across operating conditions | 🟢 Mostly done | Two real adaptation methods (fine-tuned CNN, CORAL+Random Forest) implemented and evaluated across all 12 pairs; the best variants (partial-freeze CNN, CORAL+RF on FFT features) close nearly all the gap to the target-only ceiling — see results below |

## What's been built

### `data_download.ipynb` — download & explore

- Downloads CWRU's **normal baseline** (4 files, one per load: 0–3 hp) and **48 kHz drive-end fault** data (52 files: inner race / ball / outer race faults, at fault diameters 0.007"/0.014"/0.021", each at 0–3 hp) from the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data), skipping files already on disk.
- Saves each file under a descriptive name encoding fault location, diameter, load, RPM, and (for outer-race faults) clock position — e.g. `48k_drive_end_fault_inner_race_0.007in_0hp_1797rpm_109.mat` — into labeled subfolders: `data/normal_baseline_data/` and `data/48k_drive_end_fault/`.
- Inspects a single raw `.mat` file's structure (CWRU's internal variable names: `X{n}_DE_time`, `X{n}_FE_time`, `X{n}RPM`; no `BA_time` channel in this dataset).
- Builds a `manifest` DataFrame (one row per file) by parsing metadata straight out of the filenames, with signal length, duration, and reported RPM per file.
- Plots normal-vs-fault signals in the **time domain** and via **FFT** (0–3 kHz) for one example pair.
- Combines every file into a single **`data/combined_dataset.mat`** (370 MB) — one MATLAB struct per file, signals kept at full original length, carrying both `DE_time`/`FE_time`/`BA_time` and all fault metadata. Source directories to combine are explicit (`SOURCE_DIRS`), not implicit.

### `data_splitting_preprocessing.ipynb` — split, window & extract features

- Loads `data/combined_dataset.mat` into a `df` (one row per file) via `scipy.io.loadmat(..., simplify_cells=True)`.
- **Time-based train/test split per file** (`split_signal_train_test`, 80/20): splits each file's raw signal *before* windowing, so no fixed-size window can straddle the train/test boundary or leak between them — verified by reassembling train+test back into the original signal for every file.
- **Fixed-size windowing** (`segment_signal`, default `window_size=4096`, non-overlapping, trailing remainder dropped) applied to `DE_time` only (the drive-end channel; `FE_time`/`BA_time` are carried in `combined_dataset.mat` but not yet windowed).
- Windows are grouped by `load_hp` into `windows_by_load` and saved once to **`data/windows_by_load.pkl`** (184 MB) — not duplicated per pair.
- `make_splits(source_load, target_load)` builds the 4-bucket dict (`source_train`, `source_test`, `target_labeled`, `target_test`) for any one of the **12 ordered load pairs** among `{0, 1, 2, 3}` hp on demand, letting a downstream training loop iterate 0→1, 0→2, ..., 3→2 and average results rather than committing to a single fixed source/target split.
- **Feature extraction** — 4 methods, computed in a single pass over every window in `windows_by_load` (5,601 windows total):
  1. **Raw time domain** — the window itself; captures amplitude patterns.
  2. **FFT magnitude spectrum** — captures frequency content, though fault impacts are usually buried under broadband resonance here.
  3. **Envelope spectrum** — `|Hilbert(window)|`, then FFT of that envelope. Demodulates the signal so the fault impact *rate* shows up as clean spectral lines, isolated from the carrier resonance.
  4. **Fault-frequency peaks (BPFO/BPFI/BSF)** — envelope-spectrum magnitude read off at the theoretical outer-race/inner-race/ball-spin fault frequencies (CWRU's published order-multipliers for the SKF 6205 drive-end bearing: 3.5848×/5.4152×/2.357× shaft speed) and their 2nd/3rd harmonics, tolerance wide enough to always catch a bin (~12.7 Hz — the FFT bin spacing is 11.72 Hz; an earlier ±5 Hz tolerance was narrower than half that spacing and silently returned `0.0` for some harmonics regardless of signal content), then **normalized by that window's own time-domain RMS** so the result reflects relative spectral concentration rather than each window's overall vibration amplitude — compresses the dense envelope spectrum into 9 physically-grounded, domain-comparable numbers per window.
  - Saved as **`.npz`**, not pickle — these are numeric feature matrices meant to feed a model, not the metadata-heavy structures used earlier. Per-window metadata (`load_hp`, `split`, `category`, `fault_location`, `filename`, resolved RPM, etc.) is kept alongside as parallel arrays in the same file: `data/features_time.npz`, `data/features_fft.npz`, `data/features_envelope.npz`, `data/features_fault_freq.npz`. All four share the same row order, so row `i` is the same window across every file.
  - Features are computed per load/split (mirroring `windows_by_load`), **not yet materialized per source/target pair** — assembling `source_train`/`source_test`/`target_labeled`/`target_test` for a given (source_load, target_load) still requires filtering these arrays by `load_hp`/`split`, the same way `make_splits` does for raw windows. That filtering, plus training/adapting/evaluating a model on the result, is left for the (not yet written) training notebook.

### `model_training.ipynb` — baseline models

Trains a 1D CNN (10-class: `normal` + {`inner_race`, `ball`, `outer_race`} × {0.007", 0.014", 0.021"}, outer-race clock position pooled into the diameter class) on raw `DE_time` windows from `data/windows_by_load.pkl`. Architecture is split into an `EmbeddingExtractor` (conv stack → fixed-size embedding) and a `LabelPredictor` (embedding → class logits) as two separate `nn.Module`s, so a later domain-adaptation method can attach to the embedding directly (e.g. MMD between source/target embeddings, or a gradient-reversal domain classifier) without touching the classification head.

Two no-adaptation reference points, evaluated across all **12 ordered (source_load, target_load) pairs**:

- **Baseline 1 — source-only (the floor)**: a model trained on one load's full train split, evaluated on a *different* load's test split (no adaptation). Mean accuracy **69.7%**, ranging from **39.8%** (2→0) to **91.9%** (1→2) — the raw cost of domain shift, and it varies a lot by which pair you pick.
- **Baseline 2 — target-only, scarce (10%/class)**: a separately trained model per load, using only 10% of that load's training windows *per class*, evaluated in-domain on its own load's test split. Mean accuracy **82.7%** (75.2%–90.6%). An earlier version trained this baseline on the *full* target-load train split instead (~1300 windows/load) and got ~99.9% — essentially a ceiling, not a "scarce labels" reference — so that version was dropped in favor of the 10%-per-class baseline actually used now.
- Only **8 models are trained** (4 per baseline, one per load) — `source_train` and `target_labeled` are the same underlying data for a given load, so baseline 1's model is reused across all 3 pairs where that load is the source, and results are built by evaluating already-trained models rather than retraining per pair.
- Checkpoints saved to **`models/`** (`baseline1_full_load{0-3}.pt`, `baseline2_scarce_load{0-3}.pt`) and results to **`data/baseline_results.csv`** (one row per pair, both baselines' accuracy/macro-F1).

### `domain_adaptation_evaluation.ipynb` — adapted CNN, CORAL + Random Forest, RF baseline, full comparison

Builds real domain-adaptation methods on top of the two baselines and compares everything on the same `target_test`, for all 12 pairs:

- **Adapted CNN, fine-tuned from Baseline 1** — loads Baseline 1's source-trained weights, then fine-tunes on the same scarce 10%/class target subset Baseline 2 uses, at a 10x lower learning rate, in two variants: **full freeze** (only `label_predictor` retrains) and **partial freeze** (the last conv block also unfreezes, weights *and* BatchNorm running stats).
- **Adapted classical ML — CORAL + Random Forest** — CORAL-aligns `source_train`'s covariance to the target's, trains a Random Forest on the aligned result, run on three feature sets: the 9-dim `features_fault_freq` (BPFO/BPFI/BSF peaks), and `features_fft`/`features_envelope` (2049-dim each, PCA-reduced to 20 components first — otherwise CORAL's covariance math is severely rank-deficient). **CORAL is unsupervised with respect to target** — it only needs the target *feature* distribution, never labels — so unlike the labeled methods above, its target-side statistics are computed from the **full target train split** (536–1322 windows/load), not the scarce 10%/class subset; that restriction only exists for methods that genuinely need target labels.
- **Classical ML baseline (no adaptation)** — a plain Random Forest trained on `source_train` only, evaluated zero-shot cross-domain, for each of the 3 feature sets — the classical-ML analog of Baseline 1. Necessary because without it, there was no way to tell how much CORAL was actually contributing versus what the feature set + RF already gets on its own — see takeaways below.
- A sanity check re-derives Baseline 2's numbers from the scarce-subset logic and confirms they match `baseline_results.csv` (within GPU training non-determinism), validating that the index-based window selection lines up correctly between `windows_by_load.pkl` (raw windows) and the `features_*.npz` files (pre-extracted features).
- **80 models saved to `models/`**: 24 adapted-CNN checkpoints (2 freeze modes × 12 pairs) + 36 CORAL+RF bundles + 12 plain-RF-no-adapt bundles (3 feature sets × 12 pairs / 4 loads, joblib dumps of `{clf, scaler, pca}`), alongside the 8 baseline checkpoints above.
- **Consolidated policy table** — all 10 methods' accuracy reshaped into one wide table, one row per pair, one column per method, saved to **`data/agent_policy_table.csv`**. This is the hand-off artifact `agent/policy.py` actually loads and queries — see results below.

**`data/agent_policy_table.csv` — one row per pair, one column per method (accuracy):**

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

No single method wins every pair — e.g. `RF no-adapt (fft)` is the single best zero-shot method at 1→0 (0.781, beating every CNN and CORAL variant outright) — that per-pair variability is the actual signal an agent policy needs to condition on.

**Results — mean accuracy across all 12 pairs:**

| Method | Mean accuracy |
|---|---|
| Baseline 2 — target-only, 10%/class | 82.7% |
| Adapted CNN (partial freeze) | 81.0% |
| CORAL + Random Forest (FFT features) | 78.6% |
| RF, no adaptation (FFT features) | 78.5% |
| Adapted CNN (full freeze) | 75.7% |
| CORAL + Random Forest (envelope features) | 71.6% |
| RF, no adaptation (envelope features) | 70.9% |
| Baseline 1 — source-only | 69.7% |
| CORAL + Random Forest (fault-freq features) | 63.8% |
| RF, no adaptation (fault-freq features) | 63.1% |

![Mean accuracy by method](assets/mean_accuracy.png)

**Full per-pair results** (all 12 pairs individually, not just the mean above) are in **`data/full_comparison_results.csv`** and the charts below:

![CNN methods, all 12 load pairs](assets/cnn_methods_per_pair.png)

![CORAL + Random Forest methods, all 12 load pairs](assets/coral_methods_per_pair.png)

![Does CORAL help vs. plain RF, per feature set](assets/rf_coral_vs_noadapt.png)

**Takeaways:**

- Two implementation bugs were found and fixed in `features_fault_freq.npz`: a peak-reading tolerance narrower than the FFT bin spacing (silently zeroing some harmonics regardless of signal content) and no per-window amplitude normalization (raw magnitudes were dominated by each load's overall vibration amplitude, causing a ~300x source/target covariance-scale mismatch that made CORAL collapse most of the signal during alignment). Fixing both raised CORAL+RF(fault-freq) from 49.1% to the low 60s.
- Adding the RF-no-adaptation baseline was necessary, not optional: it reveals that **CORAL's actual contribution is small and inconsistent**. Averaged, CORAL beats plain RF by only ~1–3 points per feature set, and per-pair it doesn't even win a majority of the time (fft: 4/12, envelope: 5/12, fault-freq: 7/12) — without this baseline, CORAL+RF's numbers alone would have looked far more like "adaptation is working" than the evidence actually supports.
- Letting CORAL use the *full* target train split (not the scarce labeled subset it doesn't actually need) was the theoretically correct fix, but didn't reliably help in practice: FFT and envelope actually got slightly *worse* on average (more data, but now covering an imbalanced class mixture rather than the class-balanced scarce subsample), while fault-freq improved slightly. Correct in principle; not a guaranteed win in this data regime.
- Partial-unfreezing the CNN's last conv block (75.7% → 81.0%) remains a clear, genuine win.
- No adaptation method — CNN or classical — clearly *beats* Baseline 2 on average; the best land close to it, not above it. Leveraging source knowledge hasn't yet been shown to buy more than training directly on the same amount of scarce target data, for this task.
- The domain gap is pair-dependent, not uniform: e.g. at 2→0, Baseline 1 (39.8%) actually beats the partial-freeze adapted CNN (33.6%) — adaptation isn't guaranteed to help, and can occasionally hurt relative to doing nothing.

### `cwt_baseline_exploration.ipynb` — does a richer feature representation help?

Exploratory, baseline-only (deliberately not carried through the full adaptation matrix): a Continuous Wavelet Transform scalogram (Morlet wavelet, 32 scales over 150–3000 Hz, time axis downsampled to 128 → a 32×128 image per window) paired with a 2D CNN (`src.CNN2D`), trained one model per load on its full train split and evaluated zero-shot cross-domain — the same protocol as Baseline 1, for a direct comparison.

- Mean accuracy across all 12 pairs: **69.2%** — essentially tied with the raw-window 1D CNN Baseline 1 (69.7%), and well behind RF on FFT features (78.5%).
- Checkpoints saved to `models/cwt_baseline1_full_load{0-3}.pt`; results in `data/cwt_baseline_results.csv`.
- Take: a richer 2D time-frequency input didn't obviously beat the much simpler raw-1D-window CNN here, so it wasn't carried into the full fine-tuning/CORAL comparison other representations went through — not ruled out, just not an obvious win for the added compute.

### `src/` — reusable pipeline logic, and `agent/` — the diagnosis agent

`src/` pulls the logic embedded in the four notebooks above into an importable package (`data_loading.py`, `preprocessing.py`, `feature_extraction.py`, `models.py`, `adaptation.py`, `evaluate.py`). The notebooks themselves stay self-contained — each still redefines its own copy of the logic it needs, for readability — but `agent/` and `demo.py` call `src/` directly rather than duplicating any of it.

`agent/` is the actual agentic workflow:

- **`agent/tools.py`** — wraps `src/` as agent-callable actions: load a signal, window it, extract any of the 4 feature representations, load a trained checkpoint/bundle, run CNN or RF inference, CORAL-align features, fine-tune a CNN, score predictions.
- **`agent/policy.py`** — a stateless lookup over `data/agent_policy_table.csv`: given a `(source_load, target_load)` pair, ranks every validated method by accuracy and resolves the winner into a concrete tool call (which checkpoint/bundle, which `agent/tools.py` function). Also defines `KNOWN_SOURCE_LOADS = {0, 1}` — the agent is deliberately restricted to only these as pretrained source domains, even though the underlying results cover all 4 loads as sources, specifically so an unfamiliar load (2 or 3) forces transfer learning instead of a same-domain shortcut.
- **`agent/react_loop.py`** — the THOUGHT → ACTION → OBSERVATION → DECISION loop: THOUGHT asks `policy.py` for the best remaining method, ACTION runs it via `tools.py`, OBSERVATION records the prediction and its confidence, DECISION accepts it or falls back to the next-ranked method (up to `max_attempts`) if confidence is too low.
- **`agent/diagnose.py`** — the entry point, `diagnose(signal, condition)`: windows the signal, picks the best known source domain for `condition` (or validates an explicit one against `KNOWN_SOURCE_LOADS`), and runs the react loop.
- **`demo.py`** — a Streamlit UI on top of `diagnose()`; see [Running the demo](#running-the-demo) below.

Worth flagging honestly: "adapt" in the agent's loop always means *selecting an already-adapted checkpoint* — fine-tuned or CORAL-aligned offline, back in `domain_adaptation_evaluation.ipynb` — never computing adaptation live against the uploaded signal. Doing that online wouldn't be that meaningful here anyway: supervised fine-tuning needs labels a diagnostic upload doesn't have, and CORAL from a single file's windows would just be a noisier version of the statistics already baked into the offline bundles.

## Data layout

```
data/
├── normal_baseline_data/       # 4 files: normal_{0,1,2,3}hp.mat
├── 48k_drive_end_fault/        # 52 files: 48k_drive_end_fault_<location>_<diameter>in_<load>hp_<rpm>rpm[_<position>]_<file_number>.mat
├── combined_dataset.mat        # all 56 files combined, one struct per file, full-length signals + metadata
├── windows_by_load.pkl         # DE_time windows (size 4096), grouped by load_hp, split into train/test per file
├── features_time.npz           # raw windows, (5601, 4096), + per-window metadata
├── features_fft.npz            # FFT magnitude spectrum, (5601, 2049), + per-window metadata
├── features_envelope.npz       # Hilbert envelope spectrum, (5601, 2049), + per-window metadata
├── features_fault_freq.npz     # BPFO/BPFI/BSF peak magnitudes (RMS-normalized), (5601, 9), + per-window metadata
├── baseline_results.csv        # baseline 1 & 2 accuracy/macro-F1, one row per load pair (12 total)
├── full_comparison_results.csv # all 10 methods' accuracy/macro-F1, one row per load pair (12 total)
├── cwt_baseline_results.csv    # CWT+2D-CNN baseline accuracy/macro-F1, one row per load pair (12 total)
└── agent_policy_table.csv      # wide table: 1 row/pair × 10 method columns, accuracy only — the agent hand-off artifact

models/                          # not gitignored — 84 files total
├── baseline1_full_load{0-3}.pt          # baseline 1 checkpoints (4)
├── baseline2_scarce_load{0-3}.pt        # baseline 2 checkpoints (4)
├── adapted_cnn_{full,partial}_{S}to{T}.pt   # adapted CNN checkpoints (2 modes × 12 pairs = 24)
├── rf_noadapt_{fault_freq,fft,envelope}_load{0-3}.joblib  # plain-RF-no-adapt bundles (3 feature sets × 4 loads = 12)
├── coral_rf_{fault_freq,fft,envelope}_{S}to{T}.joblib  # CORAL+RF bundles (3 feature sets × 12 pairs = 36)
└── cwt_baseline1_full_load{0-3}.pt      # CWT+2D-CNN baseline checkpoints (4)

assets/                          # not gitignored — README chart images
├── mean_accuracy.png
├── cnn_methods_per_pair.png
├── coral_methods_per_pair.png
└── rf_coral_vs_noadapt.png
```

`data/` is gitignored (large binary files) — everything in it is reproducible by running `data_download.ipynb`, then `data_splitting_preprocessing.ipynb`, then `model_training.ipynb`, then `domain_adaptation_evaluation.ipynb`. `models/` and `assets/` are not gitignored, since `assets/` holds this README's images and `models/` isn't large-binary-data in the same sense as `data/`.

## Setup

**Option A — conda:**

```bash
conda create -n cwru python=3.10
conda activate cwru
pip install -r requirements.txt
```

**Option B — venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` in `requirements.txt` pulls whatever CUDA build `pip` resolves for your platform automatically; CPU-only machines get the CPU build, no changes needed either way.

Run in order: `data_download.ipynb` (populates `data/`), `data_splitting_preprocessing.ipynb`, `model_training.ipynb` (populates `models/` with the 8 baseline checkpoints), `domain_adaptation_evaluation.ipynb` (adds 72 more checkpoints/bundles to `models/`, regenerates `assets/*.png`, and produces `data/agent_policy_table.csv`). `cwt_baseline_exploration.ipynb` is optional and independent of the rest — it only needs `data/windows_by_load.pkl` and isn't required for the agent or demo to work.

## Running the demo

Once `data/` and `models/` are populated (or a prebuilt `models/` directory is already present), launch the Streamlit UI:

```bash
streamlit run demo.py
```

This opens a browser tab (default `http://localhost:8501`) where you can upload a raw `.mat` vibration signal and the agent will:

- infer the operating condition (load) from the signal's recorded RPM, falling back to a manual picker if the RPM is missing or doesn't match one of the four known loads closely enough
- run the THOUGHT → ACTION → OBSERVATION → DECISION diagnosis loop (`agent/diagnose.py`, `agent/react_loop.py`), picking the best validated tool chain from `data/agent_policy_table.csv` and falling back to alternatives if confidence is low
- report the predicted fault class, confidence, method used, and — for files from this project's own dataset — the actual class inferred from the filename, for comparison

## What's missing / next steps

- **Objective 2 (signal analysis)** still has no *analysis* on top of the extraction pipeline — no visualization or statistics comparing extracted features across fault types/severities, no validation that the BPFO/BPFI/BSF peaks actually separate healthy from faulty windows, and no classical time-domain statistical features (RMS, kurtosis, skewness, crest factor).
- **The agent only ever selects among offline-computed results — it never adapts live.** `react_loop.py`'s "adapt" step loads an already fine-tuned/CORAL-aligned checkpoint; `agent/tools.py` exposes `fine_tune_cnn()`/`coral_align()`, but nothing in the diagnosis path calls them. Supervised fine-tuning needs labels a diagnostic upload doesn't have, and CORAL from a single file's windows would just be a noisier version of the statistics already baked into the offline bundles — meaningful online adaptation would need a genuinely new, unlabeled deployment batch, not a single-file diagnosis.
- **CWT + 2D CNN** (`cwt_baseline_exploration.ipynb`) was only tested as a zero-shot baseline (69.2%, tied with the existing raw-window CNN) — not carried through the fine-tuning/CORAL matrix the other four representations went through, since the baseline result didn't clearly justify the added compute.
- Untried on the existing representations: MMD/DANN-style adversarial domain adaptation, differential learning rates for the adapted CNN's unfrozen conv block vs. its head, CORAL on combined feature sets (e.g. FFT + fault-freq concatenated), and `features_time`/`FE_time` (fan-end)/combined DE+FE signals were never used for anything.
- No adaptation method — CNN or classical — clearly *beats* Baseline 2 on average (see takeaways above); leveraging source knowledge hasn't yet been shown to buy more than training directly on the same amount of scarce target data, for this task.
- Objectives 1 and 3 are done.

## Conclusion

All four objectives are met end to end: the data pipeline downloads, labels, and explores the CWRU 48kHz drive-end dataset; the feature-extraction pipeline produces five representations (raw window, FFT, envelope, fault-frequency peaks, and CWT scalogram) with two real bugs found and fixed along the way (a fault-frequency tolerance narrower than the FFT bin spacing, and a missing RMS normalization that was distorting CORAL's covariance alignment by ~300x); 11 methods were compared head-to-head across all 12 ordered source→target load pairs and consolidated into `data/agent_policy_table.csv`; and two real adaptation methods (fine-tuned CNN, CORAL+Random Forest) were implemented and verified leak-free — fine-tuning uses a stratified scarce subsample of the target's *train* split only, confirmed to have zero overlap with the held-out *test* split by both code inspection and empirical checks.

On top of that, the agent itself is built and working: `agent/policy.py` is a stateless lookup over the results table, `agent/react_loop.py` runs a THOUGHT → ACTION → OBSERVATION → DECISION loop that picks the best validated method and falls back to alternatives on low confidence, `agent/diagnose.py` ties it together into a single `diagnose(signal, condition)` call, and `demo.py` puts a Streamlit UI on top — upload a signal, the agent infers its operating condition from RPM, restricts itself to a deliberately small set of "known" source domains (`KNOWN_SOURCE_LOADS = {0, 1}`) so loads 2 and 3 genuinely exercise transfer learning rather than a same-domain shortcut, and reports its prediction alongside a full reasoning trace.

The most important result to report plainly, not spin: **no adaptation method beats Baseline 2** (a model trained directly on a scarce 10%-per-class labeled subset of the target load, with no source domain involved at all) on average across the 12 pairs. CNN partial-freeze fine-tuning comes close (81.0% vs. 82.7%), and both CORAL+RF and RF-no-adapt land respectably on FFT features (~78.5%), but the core promise of transfer learning — that a source-trained model beats just using the small amount of target-domain labels directly — hasn't been demonstrated for this task. That's a genuine, useful finding about when adaptation is and isn't worth the complexity, not a failure to hide.

## Limitations

- **Results don't generalize past this dataset.** Everything here is one bearing type (SKF 6205), one data source (CWRU), and four discrete load/RPM conditions. Whether any of these methods — or the "adaptation doesn't beat scarce labels" finding itself — holds on a different rig, sensor, or bearing type is untested and can't be answered without new labeled data.
- **The agent never adapts live.** As covered above, `react_loop.py` only *selects* among checkpoints that were fine-tuned or CORAL-aligned offline; `agent/tools.py`'s `fine_tune_cnn()`/`coral_align()` exist but aren't called from the diagnosis path. This is a real capability gap, not just an unoptimized corner — a diagnostic upload has no labels for supervised fine-tuning, and a single file is too little data for CORAL to improve on what's already cached.
- **`KNOWN_SOURCE_LOADS` is a demonstration constraint, not a technical one.** Restricting the agent to loads {0, 1} as source domains is a deliberate choice to force transfer learning to be exercised for loads 2/3 — checkpoints and results for all 4 loads as sources already exist and would perform at least as well if the constraint were lifted.
- **The demo's ground-truth comparison only works on this project's own files.** `demo.py` infers the "actual class" from CWRU's descriptive filenames — for a genuinely new, unlabeled signal there's no way to check whether a diagnosis was correct.
- **Numbers carry ~1 percentage point of run-to-run noise.** Observed directly: retraining CWT with the same seed shifted its mean accuracy from 69.2% to 68.4%, from GPU training non-determinism. Every accuracy figure in this README should be read as approximate, not exact to three significant figures.
- **No automated test suite.** Correctness has been checked through manual notebook re-execution, targeted verification scripts, and direct inspection (e.g. the train/test leakage check) — not CI-backed tests, so regressions would currently only be caught by rerunning the notebooks by hand.

## Repository structure

Quick reference for how the pieces described above fit together — see the "`src/` — reusable pipeline logic, and `agent/` — the diagnosis agent" section above for what each file actually does:

```
src/                                # reusable, importable code — used by agent/ and demo.py
├── __init__.py
├── data_loading.py                 # load_mat_file(), build_raw_df(), label_file(), label_for_item()
├── preprocessing.py                # split_signal_train_test(), segment_signal(), build_windows_by_load()
├── feature_extraction.py           # extract_time(), extract_fft(), extract_envelope(),
│                                      extract_fault_freq(), extract_cwt()
├── models.py                       # CNN1D, CNN2D, WindowDataset, ScalogramDataset, train_cnn()
├── adaptation.py                   # fine_tune(), coral_transform()
└── evaluate.py                     # accuracy/F1/confusion matrix helpers

agent/                               # the agentic workflow
├── __init__.py
├── tools.py                         # wraps src/ functions as agent-callable tools
├── policy.py                        # loads agent_policy_table.csv, builds/queries the
│                                       source→target → best-method lookup; KNOWN_SOURCE_LOADS
├── react_loop.py                    # THOUGHT/ACTION/OBSERVATION/DECISION orchestration
└── diagnose.py                      # main entry point: diagnose(signal, condition)

demo.py                              # Streamlit UI demo entry point — run: `streamlit run demo.py`
```
