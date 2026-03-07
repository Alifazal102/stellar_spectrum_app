from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# Order matches `label_names` in the notebook
LABEL_NAMES = ["TEFF", "LOGG", "FE_H", "MG_FE", "SI_FE"]
LATEX_LABEL_NAMES = [
    "$T_{eff}$ (K)",
    "$\\log g$ (dex)",
    "$[Fe/H]$ (dex)",
    "$[Mg/Fe]$ (dex)",
    "$[Si/Fe]$ (dex)",
]


class ModelFilesMissing(RuntimeError):
    """
    Raised when the precomputed model files are not available.
    This is caught in the Streamlit app to show a friendly message.
    """


@lru_cache(maxsize=1)
def load_model_parameters() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load precomputed model parameters and wavelength grid.

    Expected files in `data/` (relative to this file):
      - optimal_theta_lams2.npy  (theta_lams in the notebook)
      - optimal_s2_lams2.npy    (s2_lams in the notebook)
      - master_wavelengths.npy  (master_wl in the notebook)
    """
    theta_path = DATA_DIR / "optimal_theta_lams2.npy"
    s2_path = DATA_DIR / "optimal_s2_lams2.npy"
    wl_path = DATA_DIR / "master_wavelengths.npy"

    missing = [p.name for p in (theta_path, s2_path, wl_path) if not p.exists()]
    if missing:
        raise ModelFilesMissing(
            "Missing model files in the `data/` folder: "
            + ", ".join(missing)
            + ".\n\n"
            + "From your notebook, save the arrays like:\n"
            + "    np.save('data/optimal_theta_lams2.npy', optimal_theta_lams)\n"
            + "    np.save('data/optimal_s2_lams2.npy', optimal_s2lams)\n"
            + "    np.save('data/master_wavelengths.npy', master_wl)\n"
        )

    theta_lams = np.load(theta_path, allow_pickle=False)
    s2_lams = np.load(s2_path, allow_pickle=False)
    master_wl = np.load(wl_path, allow_pickle=False)
    return theta_lams, s2_lams, master_wl


def model_spec(labels: np.ndarray) -> np.ndarray:
    """
    Predict a normalized spectrum for a single star.

    This is a direct translation of `model_spec` from the notebook:

        def model_spec(labels):
            products = np.outer(labels, labels)
            triu_indices = np.triu_indices(5)
            unique_products = products[triu_indices]
            label_vector = np.hstack((np.array([1]), labels, unique_products))
            pred_fluxes = theta_lams.dot(label_vector)
            return pred_fluxes
    """
    theta_lams, _, _ = load_model_parameters()

    labels = np.asarray(labels, dtype=float)
    if labels.shape != (5,):
        raise ValueError(
            f"Expected labels shape (5,), got {labels.shape}. "
            f"Order should be {LABEL_NAMES}."
        )

    products = np.outer(labels, labels)
    triu_indices = np.triu_indices(5)
    unique_products = products[triu_indices]

    label_vector = np.hstack((np.array([1.0]), labels, unique_products))
    pred_fluxes = theta_lams.dot(label_vector)
    return pred_fluxes


def get_default_label_vector() -> np.ndarray:
    """
    Reasonable default labels for an example red-giant-like star.
    Adjust these to any star you want to highlight in the portfolio.
    """
    # Roughly within the ranges present in the APOGEE training set.
    teff = 4500.0   # K
    logg = 2.5      # dex
    fe_h = -0.2     # dex
    mg_fe = 0.1     # dex
    si_fe = 0.1     # dex
    return np.array([teff, logg, fe_h, mg_fe, si_fe], dtype=float)


def get_label_ranges() -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate label ranges for UI sliders.
    You can tweak these if you want tighter bounds.
    """
    mins = np.array([3500.0, 0.0, -2.5, -0.2, -0.2], dtype=float)
    maxs = np.array([6500.0, 5.0, 0.5, 0.6, 0.6], dtype=float)
    return mins, maxs

