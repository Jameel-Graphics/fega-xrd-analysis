"""
analyze_samples.py
Analyse several Fe-27Ga XRD scans stored as PANalytical .xrdml files and write
  results/peak_summary.csv     one row per sample (main peaks, lattice parameters)
  results/<sample>_peaks.csv   fitted peaks per sample
  figures/overview.png         all patterns, each normalised to its own maximum
  figures/zoom_40_52.png       the region around the strongest reflections

Usage:  python src/analyze_samples.py path/to/folder_with_xrdml_files

Some exported files contain only the intensity counts (no angle header). For those the
script ASSUMES the scan range of the other files (same number of points) and flags this.
"""
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xrd_analysis import analyse, d_spacing

NS = "{http://www.xrdml.com/XRDMeasurement/1.3}"
WAVELENGTH = 1.540598  # Cu K-alpha1, from the xrdml header

# (name, hkl list on the cubic cell, allowed lattice-parameter range in angstrom)
# FCC-type is fitted first; BCC-type is then fitted only to the peaks FCC cannot explain.
FAMILIES = {
    "FCC-type": ([(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2)], (3.60, 3.80)),
    "BCC-type": ([(1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 0)], (2.85, 3.00)),
}


# ------------------------------------------------------------------------------- loading
def load_xrdml(path, fallback_axis=None):
    """Return two_theta, counts, info. Works for full xrdml files and bare-number exports."""
    text = open(path, encoding="utf-8-sig", errors="ignore").read()
    info = {"counting_time_s": np.nan, "axis": "from file header"}
    if text.lstrip().startswith("<?xml"):
        root = ET.fromstring(text.encode("utf-8"))
        dp = next(root.iter(NS + "dataPoints"))
        for pos in dp.findall(NS + "positions"):
            if pos.get("axis") == "2Theta":
                start = float(pos.find(NS + "startPosition").text)
                end = float(pos.find(NS + "endPosition").text)
        counts = np.array(dp.find(NS + "intensities").text.split(), dtype=float)
        ct = dp.find(NS + "commonCountingTime")
        info["counting_time_s"] = float(ct.text) if ct is not None else np.nan
        return np.linspace(start, end, len(counts)), counts, info
    # bare numbers: strip any leftover tags and read every number
    counts = np.array(re.findall(r"\d+(?:\.\d+)?", re.sub(r"<[^>]+>", " ", text)), dtype=float)
    if fallback_axis is None or len(counts) != fallback_axis[2]:
        raise ValueError(f"{path}: no angle header and no matching fallback scan range")
    info["axis"] = "ASSUMED (no header; scan range copied from other files)"
    return np.linspace(fallback_axis[0], fallback_axis[1], len(counts)), counts, info


# --------------------------------------------------------------------- phase-family matching
def best_family_fit(peaks, hkls, a_range, tol=0.015, min_rel=5.0, exclude=()):
    """Find the cubic lattice parameter for which most strong peaks match the given family."""
    strong = peaks[(peaks["rel_intensity_%"] >= min_rel) & (peaks["fwhm_deg"] < 2.5)]
    strong = strong.drop(index=[i for i in exclude if i in strong.index])
    best = None
    for a in np.arange(a_range[0], a_range[1], 0.0005):
        matched = {}
        for idx, row in strong.iterrows():
            for hkl in hkls:
                a_peak = row["d_A"] * np.sqrt(sum(i * i for i in hkl))
                if abs(a_peak - a) < tol and (hkl not in matched or abs(a_peak - a) < matched[hkl][1]):
                    matched[hkl] = (row["two_theta"], abs(a_peak - a), a_peak, row["rel_intensity_%"], idx)
        score = sum(v[3] for v in matched.values()) if len(matched) >= 2 else 0
        if best is None or score > best[0]:
            best = (score, a, matched)
    if best is None or best[0] == 0:
        return None
    a_mean = float(np.mean([v[2] for v in best[2].values()]))
    return {"a_A": a_mean, "n_peaks": len(best[2]), "hkl": sorted(best[2].keys()),
            "used": [v[4] for v in best[2].values()],
            "two_theta": [round(best[2][h][0], 2) for h in sorted(best[2].keys())]}


def main(folder):
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    files = sorted(glob.glob(os.path.join(folder, "*.xrdml")))
    # scan range taken from the first file that has a header
    fallback = None
    for f in files:
        try:
            tt, y, _ = load_xrdml(f)
            fallback = (tt[0], tt[-1], len(tt))
            break
        except Exception:
            continue

    rows, patterns = [], {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        tt, y, info = load_xrdml(f, fallback_axis=fallback)
        res = analyse(tt, y, wavelength=WAVELENGTH, prominence_frac=0.03)
        peaks = res["peaks"]
        peaks.round(4).to_csv(f"results/{name}_peaks.csv", index=False)
        patterns[name] = (tt, y, res["corrected"], info)

        row = {"sample": name, "angle_axis": info["axis"], "counting_time_s": info["counting_time_s"],
               "max_counts": int(y.max())}
        used = []
        for fam, (hkls, rng) in FAMILIES.items():
            fit = best_family_fit(peaks, hkls, rng, exclude=used)
            if fit:
                used += fit["used"]
            row[f"{fam}_a_A"] = round(fit["a_A"], 3) if fit else np.nan
            row[f"{fam}_matched"] = (", ".join(f"{h}@{t}" for h, t in zip(fit["hkl"], fit["two_theta"]))
                                    if fit else "")
        left = peaks[(peaks["rel_intensity_%"] >= 5) & (peaks["fwhm_deg"] < 2.5)].drop(
            index=[i for i in used if i in peaks.index])
        row["unassigned_strong_peaks_2theta"] = ", ".join(f"{v:.2f}" for v in left["two_theta"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv("results/peak_summary.csv", index=False)
    print(summary.to_string(index=False))

    # ----- overview figure (each pattern normalised to its own maximum, shifted for clarity)
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, (name, (tt, y, corr, info)) in enumerate(patterns.items()):
        ax.plot(tt, y / y.max() + 1.1 * k, lw=0.8, label=name)
    ax.set_xlabel("2\u03b8 (degrees)")
    ax.set_ylabel("Normalised intensity (shifted)")
    ax.set_title("Fe-27Ga XRD patterns, each normalised to its own maximum")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/overview.png", dpi=160)

    # ----- zoom on the strongest reflections
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, (name, (tt, y, corr, info)) in enumerate(patterns.items()):
        m = (tt > 40) & (tt < 52)
        ax.plot(tt[m], corr[m] / corr[m].max() + 1.1 * k, lw=1.0, label=name)
    ax.set_xlabel("2\u03b8 (degrees)")
    ax.set_ylabel("Normalised intensity (shifted)")
    ax.set_title("Zoom 40-52 deg (background removed)")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig("figures/zoom_40_52.png", dpi=160)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/real")
