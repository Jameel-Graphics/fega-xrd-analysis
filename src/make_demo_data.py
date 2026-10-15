"""
Creates SYNTHETIC XRD patterns so the repository runs end to end without real data.
They are NOT measured data and say nothing about real Fe-Ga samples.
"""
import os

import numpy as np
import pandas as pd

from xrd_analysis import CU_KALPHA1, pseudo_voigt


def make_pattern(a=2.91, ordered_fraction=0.0, seed=0):
    rng = np.random.default_rng(seed)
    tt = np.arange(20, 95, 0.02)
    y = 120 + 600 * np.exp(-(tt - 20) / 40)  # smooth background

    def pos(hkl, cell):
        d = cell / np.sqrt(sum(i * i for i in hkl))
        return 2 * np.degrees(np.arcsin(CU_KALPHA1 / (2 * d)))

    for hkl, amp in [((1, 1, 0), 10000), ((2, 0, 0), 1800), ((2, 1, 1), 3200)]:
        y += pseudo_voigt(tt, amp, pos(hkl, a), 0.35, 0.5, 0)
    if ordered_fraction > 0:  # weak superlattice peaks (D0_3 on the doubled cell)
        y += pseudo_voigt(tt, 900 * ordered_fraction, pos((1, 1, 1), 2 * a), 0.35, 0.5, 0)
        y += pseudo_voigt(tt, 600 * ordered_fraction, pos((1, 0, 0), a), 0.35, 0.5, 0)
    y += rng.normal(0, np.sqrt(y) * 0.4)
    return tt, y


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    for name, frac, seed in [("demo_disordered", 0.0, 2), ("demo_ordered", 1.0, 1)]:
        tt, y = make_pattern(ordered_fraction=frac, seed=seed)
        pd.DataFrame({"two_theta": tt, "intensity": y.round(1)}).to_csv(
            os.path.join(data_dir, f"{name}_SYNTHETIC.csv"), index=False, header=False)
    print("wrote synthetic demo files to data/")
