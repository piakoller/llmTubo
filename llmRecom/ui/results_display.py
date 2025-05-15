# ui/results_display.py
import streamlit as st
import os
import logging
from datetime import date
import time

from agents.report_agent import ReportAgent # Assuming ReportAgent is in agents directory
import config

logger = logging.getLogger(__name__)

def display_therapie_results(therapie_output: str | None, error: Exception | None):
    """Displays therapy recommendation results."""
    st.markdown("### 🧬 Therapieempfehlung")
    if error:
        st.error(f"Konnte keine Therapieempfehlung generieren: {error}")
        logger.error(f"Error in TherapieAgent result: {error}", exc_info=error if isinstance(error, Exception) else None)
    elif therapie_output:
        st.markdown(therapie_output)
    else:
        st.warning("Therapie Agent hat keine Empfehlung generiert oder es gab einen Fehler (keine Ausgabe).")

def display_studien_results(
    study_list: list | None,
    error: Exception | None,
    user_has_geopoint: bool, # Simplified from passing the geopoint itself
    location_search_string: str | None
):
    """Displays clinical trial study results."""
    st.markdown("### 🔬 Empfohlene klinische Studien")

    if location_search_string:
        if user_has_geopoint:
            st.info(f"Suche auf ClinicalTrials.gov nach Studien in '{location_search_string}'. Studien sind nach Entfernung sortiert.")
        else:
            st.warning(f"Konnte '{location_search_string}' nicht geokodieren. Studien sind nicht nach Entfernung sortiert.")
    else:
        st.info("Kein Ort für die Studiensuche angegeben. Studien werden nicht nach Entfernung sortiert.")

    if error:
        st.error(f"Fehler bei der Studiensuche: {error}")
        logger.error(f"Error in StudienAgent result: {error}", exc_info=error if isinstance(error, Exception) else None)
        return
    if not study_list: # study_list can be None or empty list
        st.info("Keine passenden klinischen Studien gefunden oder Agent hat keine zurückgegeben.")
        return

    try:
        st.write("") # Vertical space
        for i, study in enumerate(study_list):
            title = study.get("title", "N/A")
            nct_id = study.get("nct_id", "N/A")
            status = study.get("status", "Unbekannt")
            summary = study.get("summary", "Keine Zusammenfassung.")
            locations_data = study.get("locations", []) # List of dicts {"name": ..., "distance_km": ...}
            min_distance_km = study.get("min_distance_km") # Overall min distance for the study

            # Display only a few locations, sorted by distance if available
            # Locations are already pre-sorted within the StudienAgent if distance is available
            locations_to_display = locations_data[:config.MAX_LOCATIONS_TO_DISPLAY_PER_STUDY]
            more_locations_count = len(locations_data) - len(locations_to_display)

            link_url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id != "N/A" else None

            st.subheader(f"Studie {i+1}: {title}")
            meta_col1, meta_col2 = st.columns([1, 4])
            with meta_col1:
                st.markdown("**NCT ID:**")
                st.markdown("**Status:**")
                st.markdown("**Nächster Ort:**")
            with meta_col2:
                st.markdown(f"`{nct_id}`" if nct_id != "N/A" else "N/A")
                st.markdown(status)
                st.markdown(f"{min_distance_km:.1f} km" if min_distance_km is not None else "Entfernung unbekannt")

            st.markdown(f"**Standorte (Top {len(locations_to_display)}):**")
            if locations_to_display:
                for loc_data in locations_to_display:
                    loc_name = loc_data.get("name", "Keine Ortsangaben verfügbar")
                    loc_distance = loc_data.get("distance_km")
                    dist_str = f"({loc_distance:.1f} km)" if loc_distance is not None else "(Entfernung unbekannt)"
                    st.markdown(f"- {loc_name} {dist_str}")
                if more_locations_count > 0:
                    st.markdown(f"... und {more_locations_count} weitere Standorte.")
            else:
                st.markdown("Keine Standorte für diese Studie verfügbar oder geokodierbar.")

            if link_url:
                st.markdown(f"**Link:** [Zur Studie auf ClinicalTrials.gov]({link_url})", unsafe_allow_html=True)
            with st.expander("Kurzbeschreibung"):
                st.markdown(summary)
            st.divider()
    except Exception as e:
        st.error(f"Fehler bei der Darstellung der Studiendaten: {e}")
        logger.error("Error displaying study data", exc_info=True)
        # For debugging, show the raw data that caused the error
        st.json(study_list if isinstance(study_list, (list, dict)) else str(study_list))


def display_report_download(
    report_agent: ReportAgent, # Pass the initialized agent
    diagnostik_output: str | None,
    therapie_output: str | None,
    patient_id_for_filename: str,
    patient_main_diagnosis_text: str, # For the report content
    runtimes: dict # Pass the runtimes dict to update it
):
    """Handles report generation and offers download."""
    st.markdown("### 📄 Bericht")
    if not therapie_output or not diagnostik_output:
        st.warning("Bericht kann nicht generiert werden, da diagnostische Zusammenfassung oder Therapieempfehlung fehlt.")
        return

    try:
        with st.spinner("Generiere Bericht..."):
            start_report_time = st.session_state.get("report_start_time", 0) # Use session state if needed for timing across reruns
            if start_report_time == 0: # First time
                 start_report_time = time.perf_counter()
                 st.session_state.report_start_time = start_report_time


            llm_input_context = f"Diagnostische Zusammenfassung:\n{diagnostik_output}\n\nNeue Therapie Empfehlung:\n{therapie_output}"
            board_date_str = date.today().strftime("%d.%m.%Y")

            # Prepare patient data structure expected by ReportAgent
            # Extract more details if available and needed by ReportAgent's prompt
            report_patient_data_dict = {
                'last_name': patient_id_for_filename.split('_')[0] if '_' in patient_id_for_filename else patient_id_for_filename,
                'first_name': "", # Add if available
                'dob': "", # Add if available
                'pid': patient_id_for_filename,
                'main_diagnosis_text': patient_main_diagnosis_text
            }

            report_text = report_agent.generate_report_text(
                context=llm_input_context,
                patient_data=report_patient_data_dict,
                board_date=board_date_str
            )
            # Sanitize patient_id for filename if it contains special characters
            safe_patient_id = "".join(c if c.isalnum() or c in ['_', '-'] else '' for c in patient_id_for_filename)
            report_filename_base = f"{safe_patient_id}_bericht_{board_date_str.replace('.', '')}"
            report_filepath = report_agent.save_report(report_text, report_filename_base)

            if "Report" not in runtimes: # Only set once
                 runtimes["Report"] = time.perf_counter() - start_report_time
                 del st.session_state.report_start_time # Clean up

            logger.info(f"Report generated and saved to {report_filepath} in {runtimes.get('Report', 0):.2f}s.")

        if os.path.exists(report_filepath):
            with open(report_filepath, "r", encoding="utf-8") as f:
                st.download_button(
                    label=f"📥 Bericht als {config.REPORT_FILE_TYPE.upper()} herunterladen",
                    data=f.read(),
                    file_name=f"{report_filename_base}.{config.REPORT_FILE_TYPE}",
                    mime=f"text/{config.REPORT_FILE_TYPE.lower()}",
                    key=f"download_report_{patient_id_for_filename}"
                )
        else:
            st.error(f"⚠️ Berichtdatei ({report_filepath}) konnte nicht gefunden werden.")
    except Exception as e:
        st.error(f"Fehler beim Erstellen oder Speichern des Berichts: {e}")
        logger.error("Report generation/saving failed.", exc_info=True)
        if 'report_start_time' in st.session_state:
            del st.session_state.report_start_time # Clean up on error