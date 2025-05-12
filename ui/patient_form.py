# ui/patient_form.py
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def display_patient_form(patient_df: pd.DataFrame, selected_patient_id: str) -> dict | None:
    """
    Displays patient data form for the selected patient and returns current form values.
    Returns None if patient_id is not found.
    """
    try:
        # Ensure selected_patient_id is compared with string IDs if IDs are mixed type
        patient_row = patient_df[patient_df["ID"].astype(str) == str(selected_patient_id)].iloc[0]
    except IndexError:
        st.error(f"Patient mit ID {selected_patient_id} nicht in den geladenen Daten gefunden.")
        logger.warning(f"Patient ID {selected_patient_id} not found during form display.")
        return None # Critical to return None if patient not found

    st.header(f"Patient: {selected_patient_id}")
    col1, col2 = st.columns([2, 1])
    form_data = {}

    with col1:
        form_data["main_diagnosis_text"] = st.text_area("Hauptdiagnose (Text)", value=str(patient_row.get("main_diagnosis_text", "")), height=200, key=f"form_diag_text_{selected_patient_id}")
        form_data["secondary_diagnoses"] = st.text_area("Relevante Nebendiagnosen", value=str(patient_row.get("secondary_diagnoses", "")), height=80, key=f"form_sec_diag_{selected_patient_id}")
        form_data["clinical_info"] = st.text_area("Klinische Angaben / Fragestellung", value=str(patient_row.get("clinical_info", "")), height=80, key=f"form_clin_info_{selected_patient_id}")
        form_data["pet_ct_report"] = st.text_area("PET-CT Bericht", value=str(patient_row.get("pet_ct_report", "")), height=200, key=f"form_pet_ct_{selected_patient_id}")
    with col2:
        form_data["presentation_type"] = st.text_input("Vorstellungsart", value=str(patient_row.get("presentation_type", "")), key=f"form_pres_type_{selected_patient_id}")
        form_data["main_diagnosis"] = st.text_input("Diagnose-Kürzel", value=str(patient_row.get("main_diagnosis", "")), key=f"form_main_diag_code_{selected_patient_id}")
        form_data["ann_arbor_stage"] = st.text_input("Ann-Arbor Stadium", value=str(patient_row.get("ann_arbor_stage", "")), key=f"form_aa_stage_{selected_patient_id}")
        form_data["accompanying_symptoms"] = st.text_input("Begleitsymptome", value=str(patient_row.get("accompanying_symptoms", "")), key=f"form_symptoms_{selected_patient_id}")
        form_data["prognosis_score"] = st.text_input("Prognose-Score", value=str(patient_row.get("prognosis_score", "")), key=f"form_prog_score_{selected_patient_id}")
    return form_data