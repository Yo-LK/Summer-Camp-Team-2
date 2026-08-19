"""
CWRU bearing data visualizer.

Usage:
    python cwru_viz.py <file.mat> [--fs 12000] [--rpm 1797] [--ch DE] [--out fig.png]
    python cwru_viz.py --dir CWRU_12k --grid          # class overview grid
"""
import re
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# drive end SKF 6205-2RS JEM (x shaft speed)
FAULT_MULT_DE = {"BPFI": 5.415, "BPFO": 3.585, "FTF": 0.3983, "BSF": 2.357}
FAULT_MULT_FE = {"BPFI": 4.947, "BPFO": 3.053, "FTF": 0.3816, "BSF": 1.994}


def load_mat(path):
    """Return {'DE':arr,'FE':arr,'BA':arr,'rpm':int}. Keys absent if not in file."""
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


def squared_envelope_spectrum(x, fs):
    """Method 1 from Smith & Randall 2015: spectrum of the squared envelope."""
    x = x - x.mean()
    env = np.abs(hilbert(x)) ** 2
    env -= env.mean()
    n = len(env)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(env * win)) / n
    freq = np.fft.rfftfreq(n, 1 / fs)
    return freq, spec


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    rms = np.sqrt(np.mean(xc ** 2))
    return {
        "rms": rms,
        "peak": np.max(np.abs(xc)),
        "kurtosis": np.mean(xc ** 4) / (rms ** 4),   # normalised 4th moment, K=3 for gaussian
        "skew": np.mean(xc ** 3) / (rms ** 3),
        "crest": np.max(np.abs(xc)) / rms,
    }


def plot_record(path, fs, rpm=None, ch="DE", fmax=500, tlen=0.5, out=None):
    d = load_mat(path)
    if ch not in d:
        raise SystemExit(f"channel {ch} not in {path}; available: "
                         f"{[k for k in d if k != 'rpm']}")
    x = d[ch]
    rpm = rpm or d.get("rpm")
    fr = rpm / 60.0 if rpm else None
    mult = FAULT_MULT_DE if ch == "DE" else FAULT_MULT_FE

    t = np.arange(len(x)) / fs
    n_show = int(tlen * fs)
    freq, spec = squared_envelope_spectrum(x, fs)
    s = stats(x)

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))

    ax[0].plot(t[:n_show], x[:n_show], lw=0.6)
    ax[0].set(xlabel="Time (s)", ylabel="Accel",
              title=f"{Path(path).stem}  {ch}  fs={fs}Hz  rpm={rpm}   "
                    f"K={s['kurtosis']:.2f}  RMS={s['rms']:.3f}  crest={s['crest']:.2f}")
    if fr:
        for k in range(1, int(tlen * fr) + 1):
            ax[0].axvline(k / fr, color="r", ls=":", lw=0.7)

    ax[1].plot(t, x, lw=0.3)
    ax[1].set(xlabel="Time (s)", ylabel="Accel", title="Full record (check clipping / nonstationarity)")

    m = freq <= fmax
    ax[2].plot(freq[m], spec[m], lw=0.8)
    ax[2].set(xlabel="Frequency (Hz)", ylabel="|Env|^2",
              title="Squared envelope spectrum")
    if fr:
        colors = {"BPFI": "tab:red", "BPFO": "tab:green", "BSF": "tab:purple", "FTF": "tab:orange"}
        for name, mlt in mult.items():
            for h in range(1, 4):
                f0 = mlt * fr * h
                if f0 > fmax:
                    break
                ax[2].axvline(f0, color=colors[name], ls="--", lw=0.8,
                              label=name if h == 1 else None)
        for h in range(1, 4):
            ax[2].axvline(fr * h, color="k", ls=":", lw=0.7,
                          label="fr" if h == 1 else None)
        ax[2].legend(fontsize=8, ncol=5)

    fig.tight_layout()
    out = out or f"{Path(path).stem}_{ch}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
    print({k: round(v, 3) for k, v in s.items()})
    return out


def plot_grid(root, fs=12000, ch="DE", n_per=1, fmax=500, out="grid.png"):
    """One envelope spectrum per subfolder (class overview)."""
    root = Path(root)
    files = sorted(root.rglob("*.mat"))
    picked, seen = [], set()
    for f in files:
        key = f.parent.relative_to(root).as_posix()
        if list(k for k, _ in picked).count(key) < n_per:
            picked.append((key, f))
        seen.add(key)
    rows = len(picked)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.1 * rows), squeeze=False)
    for i, (key, f) in enumerate(picked):
        d = load_mat(f)
        if ch not in d:
            continue
        x = d[ch]
        rpm = d.get("rpm", 1797)
        fr = rpm / 60
        t = np.arange(len(x)) / fs
        axes[i][0].plot(t[: int(0.3 * fs)], x[: int(0.3 * fs)], lw=0.5)
        axes[i][0].set_ylabel(f"{key}\n{f.stem}", fontsize=7)
        freq, spec = squared_envelope_spectrum(x, fs)
        m = freq <= fmax
        axes[i][1].plot(freq[m], spec[m], lw=0.6)
        for name, mlt in FAULT_MULT_DE.items():
            axes[i][1].axvline(mlt * fr, ls="--", lw=0.6,
                               color={"BPFI": "r", "BPFO": "g", "BSF": "purple", "FTF": "orange"}[name])
        axes[i][0].tick_params(labelsize=6)
        axes[i][1].tick_params(labelsize=6)
    axes[0][0].set_title("Time signal (0.3 s)")
    axes[0][1].set_title("Squared envelope spectrum  (r=BPFI g=BPFO p=BSF o=FTF)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?")
    p.add_argument("--dir")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--fs", type=int, default=12000)
    p.add_argument("--rpm", type=int)
    p.add_argument("--ch", default="DE")
    p.add_argument("--fmax", type=float, default=500)
    p.add_argument("--out")
    a = p.parse_args()
    if a.grid:
        plot_grid(a.dir, fs=a.fs, ch=a.ch, fmax=a.fmax, out=a.out or "grid.png")
    else:
        plot_record(a.path, a.fs, a.rpm, a.ch, a.fmax, out=a.out)
