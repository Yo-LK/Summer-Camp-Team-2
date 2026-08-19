"""
Rule-based bearing diagnosis via squared envelope spectrum.
Reproduces Method 1 of Smith & Randall (2015) as the "traditional method" comparison.

No training. Uses only rpm + bearing kinematics, so it is inherently immune to the
load-domain shift that hurts the ML baselines -- that contrast is the point of
including it in the report.

Usage
-----
  python envelope_rule.py --root ./data --metadata metadata.csv
  python envelope_rule.py --root ./data --metadata metadata.csv --channel FE
"""
from __future__ import annotations

import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import hilbert, decimate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# fault frequencies as multiples of shaft speed (Smith & Randall Table 2)
MULT = {
    "DE": {"BPFI": 5.415, "BPFO": 3.585, "BSF": 2.357, "FTF": 0.3983},
    "FE": {"BPFI": 4.947, "BPFO": 3.053, "BSF": 1.994, "FTF": 0.3816},
}
CLASSES = ["normal", "inner_race", "ball", "outer_race"]


def load_mat(path: Path) -> dict:
    m = sio.loadmat(str(path))
    out = {}
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if k.endswith("RPM"):
            out["rpm"] = int(np.ravel(v)[0])
            continue
        hit = re.search(r"_(DE|FE|BA)_time$", k)
        if hit:
            out[hit.group(1)] = np.ravel(v).astype(np.float64)
    return out


def squared_envelope_spectrum(x: np.ndarray, fs: int):
    x = x - x.mean()
    env = np.abs(hilbert(x)) ** 2
    env -= env.mean()
    n = len(env)
    spec = np.abs(np.fft.rfft(env * np.hanning(n))) / n
    return np.fft.rfftfreq(n, 1 / fs), spec


def band_peak(freq, spec, f0, tol=0.02):
    """Peak within +/-tol of f0. Slip shifts real fault frequencies by 1-2%,
    so reading a single FFT bin at the kinematic value systematically misses them."""
    m = (freq > f0 * (1 - tol)) & (freq < f0 * (1 + tol))
    return float(spec[m].max()) if m.any() else 0.0


def harmonic_score(freq, spec, f0, n_harm=3, tol=0.02, fmax=None):
    fmax = fmax or freq[-1]
    return sum(band_peak(freq, spec, f0 * h, tol)
               for h in range(1, n_harm + 1) if f0 * h < fmax)


def noise_floor(freq, spec, fmax=500.0):
    m = (freq > 5) & (freq <= fmax)
    return float(np.median(spec[m]))


def diagnose(x, fs, rpm, channel="DE", n_harm=3, tol=0.02, snr_threshold=6.0):
    """Return (predicted_class, score_dict). Normal if no fault frequency stands
    sufficiently above the local noise floor."""
    fr = rpm / 60.0
    freq, spec = squared_envelope_spectrum(x, fs)
    floor = noise_floor(freq, spec)

    raw = {name: harmonic_score(freq, spec, mult * fr, n_harm, tol, fmax=500.0)
           for name, mult in MULT[channel].items()}
    snr = {k: v / (floor * n_harm + 1e-12) for k, v in raw.items()}

    cand = {"inner_race": snr["BPFI"], "outer_race": snr["BPFO"], "ball": snr["BSF"]}
    best = max(cand, key=cand.get)
    pred = best if cand[best] >= snr_threshold else "normal"
    return pred, {"snr": snr, "floor": floor, "fr": fr}


def run(args):
    root = Path(args.root)
    meta = pd.read_csv(args.metadata)
    rows = []

    for _, r in meta.iterrows():
        path = root / r["relative_path"] if "relative_path" in r else root / r["filename"]
        if not path.exists():
            hits = list(root.rglob(r["filename"]))
            if not hits:
                print(f"[skip] not found: {r['filename']}")
                continue
            path = hits[0]

        d = load_mat(path)
        ch = args.channel
        if ch not in d:
            print(f"[skip] {path.name}: channel {ch} absent")
            continue

        x, fs = d[ch], int(r["sampling_rate_hz"])
        if args.target_fs and fs != args.target_fs:
            q = fs // args.target_fs
            # anti-aliased decimation. plain x[::q] folds >Nyquist content back
            # into the band and creates fake envelope peaks.
            x, fs = decimate(x, q, ftype="fir", zero_phase=True), args.target_fs

        rpm = int(d.get("rpm", r["speed_rpm"]))
        pred, info = diagnose(x, fs, rpm, ch, args.harmonics, args.tolerance, args.threshold)
        rows.append({
            "record_id": int(r["record_id"]), "channel": ch,
            "true": r["fault_type"], "pred": pred,
            "load_hp": int(r["load_hp"]), "fault_size_in": float(r["fault_size_in"]),
            "outer_race_position": r.get("outer_race_position", "NA"),
            "correct": pred == r["fault_type"],
            **{f"snr_{k}": round(v, 2) for k, v in info["snr"].items()},
        })
        print(f"  {r['record_id']:>5}  true={r['fault_type']:<11} pred={pred:<11} "
              f"{'OK ' if pred == r['fault_type'] else 'XX '}"
              f"BPFI={info['snr']['BPFI']:6.1f} BPFO={info['snr']['BPFO']:6.1f} "
              f"BSF={info['snr']['BSF']:6.1f}")

    res = pd.DataFrame(rows)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "envelope_rule_records.csv", index=False)

    acc = accuracy_score(res.true, res.pred)
    f1 = f1_score(res.true, res.pred, average="macro", labels=CLASSES, zero_division=0)
    per_class = dict(zip(CLASSES, f1_score(res.true, res.pred, average=None,
                                           labels=CLASSES, zero_division=0)))

    print(f"\naccuracy {acc:.4f}   macro-F1 {f1:.4f}")
    print("\nper-class F1:")
    for k, v in per_class.items():
        print(f"  {k:<12} {v:.4f}")
    print("\nconfusion matrix (rows=true, cols=pred, order "
          f"{CLASSES}):\n{confusion_matrix(res.true, res.pred, labels=CLASSES)}")
    print("\naccuracy by load (should be flat -- no domain shift for a physics rule):")
    print(res.groupby("load_hp").correct.mean().round(4).to_string())
    print("\naccuracy by fault size:")
    print(res.groupby("fault_size_in").correct.mean().round(4).to_string())

    with open(out / "baseline_raw_envelope_rule.json", "w") as fh:
        json.dump([{
            "mode": "rule_based", "n_classes": 4, "task": "all_loads",
            "source": [], "target": -1, "method": "envelope_rule",
            "seed": 0, "adaptation": "none",
            "accuracy": float(acc), "f1_macro": float(f1),
            "per_class_f1": {k: float(v) for k, v in per_class.items()},
        }], fh, indent=2)
    print(f"\nsaved -> {out}/envelope_rule_records.csv")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="directory containing the .mat files")
    p.add_argument("--metadata", default="metadata.csv")
    p.add_argument("--channel", default="DE", choices=["DE", "FE"])
    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--threshold", type=float, default=6.0, help="SNR cutoff for 'normal'")
    p.add_argument("--target-fs", type=int, default=0, help="decimate everything to this fs")
    p.add_argument("--outdir", default="results")
    run(p.parse_args())
