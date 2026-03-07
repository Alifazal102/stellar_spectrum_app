## Stellar Spectrum Streamlit Application

This repository contains a Streamlit application that presents a data-driven
model for predicting normalized APOGEE H-band spectra from five stellar labels.
The model is a quadratic function of the labels with wavelength-dependent
coefficients that were optimized offline and saved as NumPy arrays.

### 1. Required precomputed model files

The application expects the following arrays to be available in the `data/`
directory:

- `optimal_theta_lams2.npy` – coefficient matrix `theta_lams`
- `optimal_s2_lams2.npy` – per-pixel variance terms `s2_lams`
- `master_wavelengths.npy` – wavelength grid `master_wl`

If these arrays are produced in a separate workflow or notebook, they can be
saved with:

```python
import numpy as np

np.save('data/optimal_theta_lams2.npy', optimal_theta_lams)
np.save('data/optimal_s2_lams2.npy', optimal_s2lams)
np.save('data/master_wavelengths.npy', master_wl)
```

The Streamlit interface will not start unless all three files are present.

### 2. Local installation and execution

From the project root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This command launches a browser-based interface with sliders controlling the
five stellar labels and a live-updating plot of the predicted spectrum.

### 3. Deployment for review

1. Push the project to GitHub, including the `data/*.npy` model files.
2. On [Streamlit Community Cloud](https://streamlit.io/cloud), create a new
   application pointing to the repository.
3. Set the entrypoint to `app.py`.

After deployment, the application is accessible through a shareable URL suitable
for inclusion in resumes, portfolios, or technical screening materials.

