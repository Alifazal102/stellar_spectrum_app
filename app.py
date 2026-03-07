import numpy as np
import streamlit as st

from stellar_model import (
    LABEL_NAMES,
    LATEX_LABEL_NAMES,
    ModelFilesMissing,
    get_default_label_vector,
    get_label_ranges,
    load_model_parameters,
    model_spec,
)


st.set_page_config(
    page_title="Stellar Spectrum Label Inference",
    layout="wide",
)


def main() -> None:
    st.title("Predicting Stellar Spectra from Labels")
    st.markdown(
        """
        This application presents a data-driven model for APOGEE H-band spectra.

        Given a set of stellar labels (effective temperature, surface gravity,
        and elemental abundances), it uses a quadratic spectral model to
        predict the normalized stellar spectrum.
        """
    )

    try:
        _, _, master_wl = load_model_parameters()
    except ModelFilesMissing as exc:
        st.error(str(exc))
        st.stop()

    default_labels = get_default_label_vector()
    mins, maxs = get_label_ranges()

    with st.sidebar:
        st.header("Stellar labels")
        st.caption(
            "Move the sliders to explore how the predicted spectrum changes "
            "with each physical parameter."
        )

        current_labels = []
        for i, (name, latex_label) in enumerate(zip(LABEL_NAMES, LATEX_LABEL_NAMES)):
            min_val = float(mins[i])
            max_val = float(maxs[i])
            default_val = float(default_labels[i])
            step = (max_val - min_val) / 200.0

            value = st.slider(
                label=f"{latex_label} [{name}]",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=step,
            )
            current_labels.append(value)

        current_labels = np.array(current_labels, dtype=float)

    col_plot, col_info = st.columns([3, 2])

    with col_plot:
        st.subheader("Model-predicted normalized spectrum")
        pred_fluxes = model_spec(current_labels)

        # Downsample the spectrum to improve performance and readability.
        wl = np.asarray(master_wl, dtype=float)
        flux = np.asarray(pred_fluxes, dtype=float)
        if wl.size > 6000:
            stride = wl.size // 6000 + 1
            wl = wl[::stride]
            flux = flux[::stride]

        st.line_chart(
            {
                "Wavelength (Å)": wl,
                "Normalized flux": flux,
            },
            x="Wavelength (Å)",
        )

    with col_info:
        st.subheader("Current label vector")
        pretty = {
            latex: float(val)
            for latex, val in zip(LATEX_LABEL_NAMES, current_labels)
        }
        st.json(pretty)

        st.markdown("---")
        st.subheader("About this model")
        st.markdown(
            """
            - **Model form**: quadratic in five stellar labels with a shared set of
              coefficients \\(\\theta_\\lambda\\) per wavelength pixel.
            - **Training**: coefficients are optimized in an offline workflow using
              weighted least squares and stored as NumPy arrays.
            - **Usage here**: the interface performs only fast matrix multiplications
              to generate spectra in real time; no retraining occurs in this app.
            """
        )

        # Deployment and local execution instructions are described in
        # `README_streamlit.md` for reviewers who wish to reproduce the app.


if __name__ == "__main__":
    main()

