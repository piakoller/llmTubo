# ui/sidebar.py
import streamlit as st
import pandas as pd

def display_sidebar(patient_df: pd.DataFrame | None) -> tuple[str | None, str, str]:
    """Displays the sidebar and returns selected patient ID, guideline, and location."""
    st.sidebar.header("Einstellungen")

    selected_patient_id = None
    if patient_df is not None and not patient_df.empty:
        # Ensure IDs are unique and handle potential NaN/None, convert to string
        patient_ids = [str(pid) for pid in patient_df["ID"].unique() if pd.notna(pid)]
        if patient_ids:
            selected_patient_id = st.sidebar.selectbox(
                "Patienten-ID auswählen",
                options=patient_ids,
                key="patient_id_selector"
            )
        else:
            st.sidebar.warning("Keine gültigen Patienten-IDs in den Daten gefunden.")
    else:
        st.sidebar.warning("Keine Patientendaten zum Auswählen vorhanden.")

    guideline_provider = st.sidebar.selectbox(
        "Leitlinie wählen",
        options=["ESMO", "Onkopedia", "S3"],
        key="guideline_selector"
    )
    selected_location = st.sidebar.text_input(
        "Suche Klinische Studien in (Land/Ort)",
        value="Bern, Switzerland",
        key="location_search_input"
    )
    return selected_patient_id, guideline_provider, selected_location