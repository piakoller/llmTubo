# app.py
import streamlit as st
import time
import logging

# Project local imports
import config # For TUBO_EXCEL_FILE_PATH
from utils.logging_setup import setup_logging
from data_loader import load_patient_data
from ui.sidebar import display_sidebar
from ui.patient_form import display_patient_form
from ui.results_display import (
    display_therapie_results,
    display_studien_results,
    display_report_download
)
from core.agent_manager import AgentWorkflowManager

# Setup logging as the first thing
setup_logging()
logger = logging.getLogger(__name__)


def main():
    st.set_page_config(page_title="Tumorboard Therapie Empfehlung", page_icon="🧬", layout="wide")
    st.title("Tumorboard Therapie Empfehlung")

    # Load data once
    # df_patients is now application-level, passed to UI components
    if 'df_patients' not in st.session_state:
        st.session_state.df_patients = load_patient_data(config.TUBO_EXCEL_FILE_PATH)

    df_patients = st.session_state.df_patients

    if df_patients is None or df_patients.empty:
        st.error("Patientendaten konnten nicht geladen werden oder sind leer. Bitte App neu starten oder Konfiguration prüfen.")
        st.stop()

    # --- Sidebar ---
    selected_patient_id, guideline_provider, selected_location_search = display_sidebar(df_patients)

    if not selected_patient_id:
        st.info("Bitte einen Patienten aus der Seitenleiste auswählen.")
        st.stop()

    # --- Patient Data Form ---
    # The key for patient_form_values ensures it re-renders if selected_patient_id changes
    patient_form_values = display_patient_form(df_patients, selected_patient_id)
    if not patient_form_values: # Patient ID might have become invalid after selection
        st.error(f"Formular für Patient {selected_patient_id} konnte nicht angezeigt werden. Bitte ID prüfen.")
        st.stop()

    # Combine all patient-related data for the agent manager
    current_patient_data_for_agents = {
        "id": selected_patient_id,
        **patient_form_values, # Values from the editable form
        "guideline": guideline_provider,
        "location": selected_location_search
    }

    # --- "Generate" Button and Agent Workflow ---
    if st.button("Multi-Agenten Empfehlung generieren", key="generate_button_main"):
        st.session_state.workflow_results = None # Clear previous results
        st.session_state.workflow_runtimes = {}
        st.session_state.workflow_errors = {}
        st.session_state.workflow_user_geopoint_exists = False # For UI display
        st.session_state.report_agent_instance = None


        start_time_total = time.perf_counter()
        
        # Use a status placeholder for overall progress
        status_placeholder = st.empty()
        status_placeholder.info("Starte Multi-Agenten-Workflow...")

        with st.spinner("Verarbeite Patientenanfrage..."):
            manager = AgentWorkflowManager(current_patient_data_for_agents)
            try:
                status_placeholder.info("Führe Agenten aus (Diagnostik, Studien, Therapie)...")
                manager.run_workflow() # This now handles initialization and running

                st.session_state.workflow_results = manager.results
                st.session_state.workflow_runtimes = manager.runtimes
                st.session_state.workflow_errors = manager.errors
                st.session_state.workflow_user_geopoint_exists = manager.user_geopoint is not None
                st.session_state.report_agent_instance = manager.report_agent # Store for report download

                if manager.errors.get("Initialization"):
                    st.error(f"Fehler bei der Initialisierung: {manager.errors['Initialization']}")
                    status_placeholder.empty()
                    st.stop()
                
                status_placeholder.success("Agenten-Lauf abgeschlossen!")
                logger.info("Agent workflow completed successfully via manager.")

            except Exception as e:
                st.error(f"Ein unerwarteter Fehler ist im Workflow aufgetreten: {e}")
                logger.error("Unexpected error in main workflow execution", exc_info=True)
                status_placeholder.empty()
                st.session_state.workflow_errors["GlobalWorkflowError"] = e


        st.session_state.total_runtime = time.perf_counter() - start_time_total


    # --- Display Results (if available in session_state) ---
    if st.session_state.get('workflow_results'):
        results = st.session_state.workflow_results
        errors = st.session_state.workflow_errors
        runtimes = st.session_state.workflow_runtimes
        report_agent_instance = st.session_state.get('report_agent_instance')


        display_therapie_results(results.get("Therapie"), errors.get("Therapie"))
        st.divider()
        display_studien_results(
            results.get("Studien"),
            errors.get("Studien"),
            st.session_state.workflow_user_geopoint_exists,
            current_patient_data_for_agents.get('location') # Original search string
        )
        st.divider()

        if report_agent_instance: # Check if ReportAgent was successfully initialized
            display_report_download(
                report_agent=report_agent_instance,
                diagnostik_output=results.get("Diagnostik"),
                therapie_output=results.get("Therapie"),
                patient_id_for_filename=current_patient_data_for_agents['id'],
                patient_main_diagnosis_text=current_patient_data_for_agents.get('main_diagnosis_text', 'N/A'),
                runtimes=runtimes # Pass dict to be potentially updated
            )
        else:
            st.warning("Bericht-Agent wurde nicht initialisiert, Download nicht verfügbar.")

        # Display Runtimes
        total_runtime = st.session_state.get('total_runtime', 0)
        runtime_info = " | ".join(f"{name}: {rt:.2f}s" for name, rt in runtimes.items() if rt is not None)
        st.info(f"⏱️ Laufzeiten: {runtime_info} | Gesamt: {total_runtime:.2f}s")


if __name__ == "__main__":
    main()