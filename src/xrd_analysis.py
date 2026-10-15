"""
xrd_analysis.py
Small, readable toolkit for analysing powder XRD patterns of BCC-based Fe-Ga alloys.

What it does
  1. loads a pattern (two columns: 2-theta in degrees, intensity)
  2. smooths it and removes the background
  3. detects peaks and fits each one with a pseudo-Voigt profile
  4. converts peak positions to d-spacings (Bragg's law) and a lattice parameter
  5. checks whether weak superlattice peaks (B2 / D0_3 ordering) are present
  6. estimates crystallite size with the Scherrer equation (rough estimate only)

Phase labels produced here are CANDIDATE labels for screening. They do not replace
proper phase identification (reference patterns, Rietveld refinement, expert judgement).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

CU_KALPHA1 = 1.5406  # wavelength in angstrom (change if your instrument differs)


# ----------------------------------------------------------------------------- loading
def load_pattern(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a text/CSV file with two numeric columns: 2-theta and intensity."""
    df = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    two_theta = df.iloc[:, 0].to_numpy(dtype=float)
    intensity = df.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(two_theta)
    return two_theta[order], intensity[order]


# ------------------------------------------------------------------------ preprocessing
def smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    window = window if window % 2 == 1 else window + 1
    return savgol_filter(y, window_length=window, polyorder=poly)


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens, 2005)."""
    n = len(y)
    d = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    penalty = lam * d.dot(d.transpose())
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        W = diags(w, 0, shape=(n, n))
        z = spsolve(csc_matrix(W + penalty), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


# --------------------------------------------------------------------------- peak fitting
def pseudo_voigt(x, amp, center, fwhm, eta, offset):
    """Weighted sum of a Lorentzian and a Gaussian with the same FWHM."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - center) / (fwhm / 2)) ** 2)
    return offset + amp * (eta * lorentz + (1 - eta) * gauss)


def fit_peak(two_theta, y, center_guess, half_window=1.2):
    """Fit one peak; returns a dict of fitted parameters or None if the fit fails."""
    mask = (two_theta > center_guess - half_window) & (two_theta < center_guess + half_window)
    if mask.sum() < 8:
        return None
    x, yy = two_theta[mask], y[mask]
    p0 = [yy.max() - yy.min(), center_guess, 0.3, 0.5, yy.min()]
    bounds = ([0, center_guess - half_window, 0.02, 0, -np.inf],
              [np.inf, center_guess + half_window, 3.0, 1, np.inf])
    try:
        popt, _ = curve_fit(pseudo_voigt, x, yy, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        return None
    amp, center, fwhm, eta, _ = popt
    return {"two_theta": center, "fwhm_deg": fwhm, "height": amp, "eta": eta}


# ---------------------------------------------------------------------------- crystallography
def d_spacing(two_theta_deg, wavelength=CU_KALPHA1):
    """Bragg's law: d = lambda / (2 sin(theta))."""
    theta = np.radians(np.asarray(two_theta_deg) / 2.0)
    return wavelength / (2.0 * np.sin(theta))


def lattice_param_cubic(d, hkl):
    h, k, l = hkl
    return d * np.sqrt(h * h + k * k + l * l)


def scherrer_size_nm(fwhm_deg, two_theta_deg, wavelength=CU_KALPHA1, k=0.9):
    """Rough crystallite size. Ignores instrument broadening, so it is a lower bound."""
    beta = np.radians(fwhm_deg)
    theta = np.radians(np.asarray(two_theta_deg) / 2.0)
    return (k * wavelength / (beta * np.cos(theta))) / 10.0  # angstrom -> nm


# BCC (A2) fundamental reflections, indexed on the simple A2 cell
A2_REFLECTIONS = {(1, 1, 0): "A2 (110)", (2, 0, 0): "A2 (200)", (2, 1, 1): "A2 (211)"}


def superlattice_positions(a_a2, wavelength=CU_KALPHA1):
    """
    Expected 2-theta of weak ordering peaks, given the A2 lattice parameter.
      B2   : (100) and (111) on the A2 cell
      D0_3 : doubled cell (a' = 2a), lowest peak is (111)' at d = a'/sqrt(3)
    """
    def tth(d):
        s = wavelength / (2 * d)
        return 2 * np.degrees(np.arcsin(s)) if s < 1 else np.nan

    a_d03 = 2 * a_a2
    return {
        "D0_3 (111)": tth(a_d03 / np.sqrt(3)),
        "B2 (100) / D0_3 (200)": tth(a_a2),
        "B2 (111)": tth(a_a2 / np.sqrt(3)),
    }


# ----------------------------------------------------------------------------------- pipeline
def analyse(two_theta, intensity, wavelength=CU_KALPHA1, prominence_frac=0.02):
    ys = smooth(intensity)
    bg = baseline_als(ys)
    corrected = np.clip(ys - bg, 0, None)

    idx, _ = find_peaks(corrected, prominence=prominence_frac * corrected.max())
    fitted = []
    for i in idx:
        res = fit_peak(two_theta, corrected, two_theta[i])
        if res is not None:
            fitted.append(res)

    peaks = pd.DataFrame(fitted).sort_values("two_theta").reset_index(drop=True)
    peaks["d_A"] = d_spacing(peaks["two_theta"], wavelength)
    peaks["rel_intensity_%"] = 100 * peaks["height"] / peaks["height"].max()
    peaks["crystallite_nm"] = scherrer_size_nm(peaks["fwhm_deg"], peaks["two_theta"], wavelength)

    # Match strong peaks to A2 fundamentals using a typical Fe-Ga lattice parameter (~2.90 A)
    labels, a_values = [], []
    for _, row in peaks.iterrows():
        label, a_val = "", np.nan
        for hkl, name in A2_REFLECTIONS.items():
            a_guess = lattice_param_cubic(row["d_A"], hkl)
            if 2.80 < a_guess < 3.00 and row["rel_intensity_%"] > 10:
                label, a_val = name, a_guess
                break
        labels.append(label)
        a_values.append(a_val)
    peaks["candidate_label"] = labels
    peaks["a_A2_A"] = a_values

    a_mean = float(np.nanmean(peaks["a_A2_A"])) if peaks["a_A2_A"].notna().any() else np.nan

    order = {}
    if not np.isnan(a_mean):
        for name, pos in superlattice_positions(a_mean, wavelength).items():
            near = peaks[(peaks["two_theta"] - pos).abs() < 0.4]
            order[name] = {
                "expected_2theta": round(float(pos), 2),
                "found": bool(len(near)),
                "rel_intensity_%": round(float(near["rel_intensity_%"].max()), 2) if len(near) else 0.0,
            }
    return {"corrected": corrected, "baseline": bg, "peaks": peaks, "a_A2": a_mean, "ordering": order}
