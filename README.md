# CWRU Bearing Fault Diagnosis

**Languages:** [English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## Project goal

Build an agent that can:

1. Load and understand the CWRU bearing dataset structure
2. Analyze vibration signals
3. Select appropriate fault diagnosis methods
4. Apply transfer learning across different operating conditions (motor loads)

**Current state:** the full data pipeline (download → split/window/feature-extraction → baselines → adaptation methods) is done and has produced a single consolidated results table (`data/agent_policy_table.csv`, 12 pairs × 7 methods) that's ready to feed an agent's decision policy. That agent/orchestration layer — the actual "build an agent" part of the project goal — is the next unbuilt piece; everything so far is a manually-run notebook pipeline that produces the inputs an agent would need.

## Status against the 4 objectives

| # | Objective | Status | Notes |
|---|---|---|---|
| 1 | Load & understand dataset structure | ✅ Done | `data_download.ipynb` downloads, labels, and inspects the data |
| 2 | Analyze vibration signals | 🟡 Partial | Extraction pipeline exists (raw/FFT/envelope/fault-frequency peaks), but no analysis on top of it yet — nothing validates that these features actually separate healthy from faulty windows |
| 3 | Select fault diagnosis methods | 🟢 Mostly done | 7 methods compared head-to-head with real per-pair numbers, consolidated into `data/agent_policy_table.csv` — the remaining gap is that "selection" is still a human reading a table, not a policy/agent |
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
  4. **Fault-frequency peaks (BPFO/BPFI/BSF)** — envelope-spectrum magnitude read off at the theoretical outer-race/inner-race/ball-spin fault frequencies (CWRU's published order-multipliers for the SKF 6205 drive-end bearing: 3.5848×/5.4152×/2.357× shaft speed) and their 2nd/3rd harmonics, ±5 Hz tolerance — compresses the dense envelope spectrum into 9 physically-grounded numbers per window.
  - Saved as **`.npz`**, not pickle — these are numeric feature matrices meant to feed a model, not the metadata-heavy structures used earlier. Per-window metadata (`load_hp`, `split`, `category`, `fault_location`, `filename`, resolved RPM, etc.) is kept alongside as parallel arrays in the same file: `data/features_time.npz`, `data/features_fft.npz`, `data/features_envelope.npz`, `data/features_fault_freq.npz`. All four share the same row order, so row `i` is the same window across every file.
  - Features are computed per load/split (mirroring `windows_by_load`), **not yet materialized per source/target pair** — assembling `source_train`/`source_test`/`target_labeled`/`target_test` for a given (source_load, target_load) still requires filtering these arrays by `load_hp`/`split`, the same way `make_splits` does for raw windows. That filtering, plus training/adapting/evaluating a model on the result, is left for the (not yet written) training notebook.

### `model_training.ipynb` — baseline models

Trains a 1D CNN (10-class: `normal` + {`inner_race`, `ball`, `outer_race`} × {0.007", 0.014", 0.021"}, outer-race clock position pooled into the diameter class) on raw `DE_time` windows from `data/windows_by_load.pkl`. Architecture is split into an `EmbeddingExtractor` (conv stack → fixed-size embedding) and a `LabelPredictor` (embedding → class logits) as two separate `nn.Module`s, so a later domain-adaptation method can attach to the embedding directly (e.g. MMD between source/target embeddings, or a gradient-reversal domain classifier) without touching the classification head.

Two no-adaptation reference points, evaluated across all **12 ordered (source_load, target_load) pairs**:

- **Baseline 1 — source-only (the floor)**: a model trained on one load's full train split, evaluated on a *different* load's test split (no adaptation). Mean accuracy **69.7%**, ranging from **39.8%** (2→0) to **91.9%** (1→2) — the raw cost of domain shift, and it varies a lot by which pair you pick.
- **Baseline 2 — target-only, scarce (10%/class)**: a separately trained model per load, using only 10% of that load's training windows *per class*, evaluated in-domain on its own load's test split. Mean accuracy **82.7%** (75.2%–90.6%). An earlier version trained this baseline on the *full* target-load train split instead (~1300 windows/load) and got ~99.9% — essentially a ceiling, not a "scarce labels" reference — so that version was dropped in favor of the 10%-per-class baseline actually used now.
- Only **8 models are trained** (4 per baseline, one per load) — `source_train` and `target_labeled` are the same underlying data for a given load, so baseline 1's model is reused across all 3 pairs where that load is the source, and results are built by evaluating already-trained models rather than retraining per pair.
- Checkpoints saved to **`models/`** (`baseline1_full_load{0-3}.pt`, `baseline2_scarce_load{0-3}.pt`) and results to **`data/baseline_results.csv`** (one row per pair, both baselines' accuracy/macro-F1).

### `domain_adaptation.ipynb` — adapted CNN, CORAL + Random Forest, full comparison

Builds real domain-adaptation methods on top of the two baselines and compares everything on the same `target_test`, for all 12 pairs:

- **Adapted CNN, fine-tuned from Baseline 1** — loads Baseline 1's source-trained weights, then fine-tunes on the same scarce 10%/class target subset Baseline 2 uses, at a 10x lower learning rate, in two variants: **full freeze** (only `label_predictor` retrains) and **partial freeze** (the last conv block also unfreezes, weights *and* BatchNorm running stats).
- **Adapted classical ML — CORAL + Random Forest** — CORAL-aligns `source_train`'s covariance to `target_labeled`'s, trains a Random Forest on the aligned result, run on three feature sets: the 9-dim `features_fault_freq` (BPFO/BPFI/BSF peaks), and `features_fft`/`features_envelope` (2049-dim each, PCA-reduced to 20 components first — otherwise CORAL's covariance math is severely rank-deficient against as few as 56 target samples).
- A sanity check re-derives Baseline 2's numbers from the same scarce-subset logic and confirms they match `baseline_results.csv` (within GPU training non-determinism), validating that the index-based window selection lines up correctly between `windows_by_load.pkl` (raw windows) and the `features_*.npz` files (pre-extracted features).
- **68 models saved to `models/`**: 24 adapted-CNN checkpoints (2 freeze modes × 12 pairs) + 36 CORAL+RF bundles (3 feature sets × 12 pairs, joblib dumps of `{clf, scaler, pca}`), alongside the 8 baseline checkpoints above.
- **Consolidated policy table** — all 7 methods' accuracy reshaped into one wide table, one row per pair, one column per method, saved to **`data/agent_policy_table.csv`**. This is the intended hand-off artifact for the next phase (an agent that picks which method to trust for a given source→target pair) — see results below.

**`data/agent_policy_table.csv` — one row per pair, one column per method (accuracy):**

| source→target | Baseline1 (no adapt) | Baseline2 (target-only) | CNN partial-freeze | CNN full-freeze | CORAL+RF (fft) | CORAL+RF (envelope) | CORAL+RF (fault_freq) |
|---|---|---|---|---|---|---|---|
| 0→1 | 0.615 | 0.814 | 0.829 | 0.699 | 0.655 | 0.590 | 0.606 |
| 0→2 | 0.668 | 0.839 | 0.901 | 0.758 | 0.755 | 0.689 | 0.528 |
| 0→3 | 0.494 | 0.752 | 0.736 | 0.680 | 0.748 | 0.640 | 0.301 |
| 1→0 | 0.617 | 0.906 | 0.641 | 0.602 | 0.742 | 0.570 | 0.531 |
| 1→2 | 0.919 | 0.839 | 0.910 | 0.922 | 0.910 | 0.857 | 0.627 |
| 1→3 | 0.904 | 0.752 | 0.941 | 0.904 | 0.792 | 0.870 | 0.488 |
| 2→0 | 0.398 | 0.906 | 0.336 | 0.508 | 0.750 | 0.523 | 0.469 |
| 2→1 | 0.839 | 0.814 | 0.904 | 0.857 | 0.922 | 0.826 | 0.522 |
| 2→3 | 0.696 | 0.752 | 0.981 | 0.860 | 0.953 | 0.941 | 0.547 |
| 3→0 | 0.570 | 0.906 | 0.648 | 0.602 | 0.711 | 0.578 | 0.469 |
| 3→1 | 0.780 | 0.814 | 0.891 | 0.786 | 0.835 | 0.845 | 0.422 |
| 3→2 | 0.860 | 0.839 | 1.000 | 0.904 | 1.000 | 0.984 | 0.382 |

No single method wins every pair (e.g. CNN partial-freeze wins 2→3 and 3→2; Baseline2 wins 1→0, 2→0, 3→0 by a wide margin; CORAL+RF fault-freq never wins anywhere) — that per-pair variability is the actual signal an agent policy needs to condition on.

**Results — mean accuracy across all 12 pairs:**

| Method | Mean accuracy |
|---|---|
| Baseline 2 — target-only, 10%/class | 82.7% |
| CORAL + Random Forest (FFT features) | 81.5% |
| Adapted CNN (partial freeze) | 81.0% |
| Adapted CNN (full freeze) | 75.7% |
| CORAL + Random Forest (envelope features) | 74.3% |
| Baseline 1 — source-only | 69.7% |
| CORAL + Random Forest (fault-freq features) | 49.1% |

![Mean accuracy by method](assets/mean_accuracy.png)

**Full per-pair results** (all 12 pairs individually, not just the mean above) are in **`data/full_comparison_results.csv`** and the two charts below:

![CNN methods, all 12 load pairs](assets/cnn_methods_per_pair.png)

![CORAL + Random Forest methods, all 12 load pairs](assets/coral_methods_per_pair.png)

**Takeaways:**

- Partial-unfreezing the CNN's last conv block (75.7% → 81.0%) and switching CORAL from the compact fault-frequency features to the full FFT spectrum (49.1% → 81.5%) were both large, genuine improvements over the first attempt at each method — the fault-frequency feature set was the bottleneck for CORAL, not CORAL itself.
- Neither adaptation method clearly *beats* Baseline 2 on average — the best variants land essentially tied with it. Leveraging source knowledge doesn't yet buy more than just training directly on the same amount of scarce target data does, for this task.
- The domain gap is pair-dependent, not uniform: e.g. at 2→0, Baseline 1 (39.8%) actually beats the partial-freeze adapted CNN (33.6%) — adaptation isn't guaranteed to help, and can occasionally hurt relative to doing nothing.

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
├── features_fault_freq.npz     # BPFO/BPFI/BSF peak magnitudes, (5601, 9), + per-window metadata
├── baseline_results.csv        # baseline 1 & 2 accuracy/macro-F1, one row per load pair (12 total)
├── full_comparison_results.csv # all 7 methods' accuracy/macro-F1, one row per load pair (12 total)
└── agent_policy_table.csv      # wide table: 1 row/pair × 7 method columns, accuracy only — the agent hand-off artifact

models/                          # not gitignored — 68 files total
├── baseline1_full_load{0-3}.pt          # baseline 1 checkpoints (4)
├── baseline2_scarce_load{0-3}.pt        # baseline 2 checkpoints (4)
├── adapted_cnn_{full,partial}_{S}to{T}.pt   # adapted CNN checkpoints (2 modes × 12 pairs = 24)
└── coral_rf_{fault_freq,fft,envelope}_{S}to{T}.joblib  # CORAL+RF bundles (3 feature sets × 12 pairs = 36)

assets/                          # not gitignored — README chart images
├── mean_accuracy.png
├── cnn_methods_per_pair.png
└── coral_methods_per_pair.png
```

`data/` is gitignored (large binary files) — everything in it is reproducible by running `data_download.ipynb`, then `data_splitting_preprocessing.ipynb`, then `model_training.ipynb`, then `domain_adaptation.ipynb`. `models/` and `assets/` are not gitignored, since `assets/` holds this README's images and `models/` isn't large-binary-data in the same sense as `data/`.

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

Run in order: `data_download.ipynb` (populates `data/`), `data_splitting_preprocessing.ipynb`, `model_training.ipynb` (populates `models/` with the 8 baseline checkpoints), `domain_adaptation.ipynb` (adds 60 more checkpoints/bundles to `models/`, regenerates `assets/*.png`, and produces `data/agent_policy_table.csv`).

## What's missing / next steps

- **The agent itself** — this is the biggest gap relative to the project goal. `data/agent_policy_table.csv` exists specifically to feed a policy that picks a method per (source_load, target_load) pair, but nothing reads that table and acts on it yet. This is the next planned piece of work.
- **Objective 2 (signal analysis)** has an extraction pipeline now (raw/FFT/envelope/fault-frequency peaks) but no *analysis* on top of it — no visualization or statistics comparing extracted features across fault types/severities, no validation that the BPFO/BPFI/BSF peaks actually separate healthy from faulty windows, and no classical time-domain statistical features (RMS, kurtosis, skewness, crest factor).
- **Objective 3 (method selection)** has the comparison data an agent needs (`agent_policy_table.csv`), but the "selection" logic itself doesn't exist yet — that's the agent work above. Also untested: `features_time` (the 4th extracted feature set) was never used for anything, and no method has been tried on `FE_time` (fan-end) or combined DE+FE signals.
- **Objective 4 (transfer learning)** has two real adaptation methods now (fine-tuned CNN, CORAL+Random Forest) and both are competitive, but neither *beats* Baseline 2 on average — the actual value-add of "leveraging source knowledge" over "just use the scarce target labels directly" hasn't been demonstrated yet for this task. Untried: MMD/DANN-style adversarial domain adaptation, differential learning rates for the adapted CNN's unfrozen conv block vs. its head, and CORAL on combined feature sets (e.g. FFT + fault-freq concatenated) rather than one at a time.
- **Objective 1** is done — scoped to the drive-end fault data and normal baseline data.
