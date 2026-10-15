"""Command line entry point:  python src/run_analysis.py --input data/your_file.csv"""
import argparse
import os

import matplotlib.pyplot as plt

from xrd_analysis import CU_KALPHA1, analyse, load_pattern


def main():
    ap = argparse.ArgumentParser(description="Peak analysis for Fe-Ga XRD patterns")
    ap.add_argument("--input", required=True, help="CSV/TXT with 2theta and intensity columns")
    ap.add_argument("--wavelength", type=float, default=CU_KALPHA1, help="X-ray wavelength in angstrom")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--figures", default="figures", help="figure folder")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    tt, y = load_pattern(args.input)
    res = analyse(tt, y, wavelength=args.wavelength)
    peaks = res["peaks"]
    peaks.round(4).to_csv(f"{args.out}/{name}_peaks.csv", index=False)

    print(f"\n{name}: {len(peaks)} peaks found")
    print(peaks[["two_theta", "d_A", "rel_intensity_%", "fwhm_deg", "candidate_label"]].round(3).to_string(index=False))
    print(f"\nMean A2 lattice parameter: {res['a_A2']:.4f} angstrom")
    print("Ordering check (candidate superlattice peaks):")
    for k, v in res["ordering"].items():
        print(f"  {k:24s} expected {v['expected_2theta']:6.2f} deg  found: {v['found']}  (rel. int. {v['rel_intensity_%']}%)")

    corrected_max = res["corrected"].max()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tt, y, color="0.7", lw=0.8, label="raw")
    ax.plot(tt, res["corrected"], color="#14284B", lw=1.1, label="smoothed, background removed")
    for _, r in peaks.iterrows():
        ax.axvline(r["two_theta"], color="#1B7F8C", ls=":", lw=0.8)
        if r["candidate_label"]:
            ax.text(r["two_theta"], r["height"] * 1.04, r["candidate_label"], ha="center", fontsize=8, color="#1B7F8C")
    for k, v in res["ordering"].items():
        ax.axvline(v["expected_2theta"], color="#C0392B", ls="--", lw=0.7, alpha=0.7)
        ax.text(v["expected_2theta"] + 0.5, corrected_max * 0.55, k, rotation=90, fontsize=7, color="#C0392B", va="top")
    ax.set_xlabel("2\u03b8 (degrees)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"{name}\npeaks (teal) and expected ordering positions (red dashed)", fontsize=10)
    ax.legend(frameon=False)
    ax.set_xlim(tt.min(), tt.max())
    fig.tight_layout()
    fig.savefig(f"{args.figures}/{name}.png", dpi=160)
    print(f"\nSaved {args.out}/{name}_peaks.csv and {args.figures}/{name}.png")


if __name__ == "__main__":
    main()
