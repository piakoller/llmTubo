# appOllamaAgent.py
import pandas as pd
import streamlit as st
import time
import threading
import logging
import os

from langchain_ollama import OllamaLLM

# Assuming multiagent.py and patient.py are in the same directory or accessible
from multiagent import DiagnostikAgent, StudienAgent, TherapieAgent, ReportAgent, AgentRunner, get_geopoint
from patient import Patient
from settings import TUBO_EXCEL_FILE_PATH # Assuming settings.py exists

# --- Configuration ---
LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0.7
REPORT_DIR = "generated_report"
REPORT_FILE_TYPE = "md" # Stick to markdown as per ReportAgent implementation

DEBUG_MODE = True # Set to True for detailed logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s"
)

def load_data(file_path):
    """Loads patient data from the specified Excel file."""
    try:
        # Skiprows might need adjustment based on the actual file structure
        return pd.read_excel(file_path, skiprows=8)
    except FileNotFoundError:
        st.error(f"Error: Excel file not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None

# --- Streamlit App ---
st.set_page_config(page_title="Tumorboard Therapie Empfehlung", page_icon="🧬", layout="wide")
st.title("Tumorboard Therapie Empfehlung")

# --- Data Loading ---
df_patients = load_data(TUBO_EXCEL_FILE_PATH)

if df_patients is None:
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Einstellungen")
    selected_patient_id = st.selectbox("Patienten-ID auswählen", df_patients["ID"].unique().tolist())
    guideline_provider = st.selectbox("Leitlinie wählen", ["ESMO", "Onkopedia", "S3"])
    selected_location = st.text_input("Suche Klinische Studien in (Land/Ort)", value="Bern, Switzerland")

# --- Patient Data Display and Input ---
try:
    patient_row = df_patients[df_patients["ID"] == selected_patient_id].iloc[0]
except IndexError:
    st.error(f"Patient with ID {selected_patient_id} not found.")
    st.stop()

st.header(f"Patient: {selected_patient_id}")
col1, col2 = st.columns([2, 1])

with col1:
    main_diagnosis_text = st.text_area("Hauptdiagnose (Text)", value=patient_row.get("main_diagnosis_text", ""), height=200)
    secondary_diagnoses = st.text_area("Relevante Nebendiagnosen", value=patient_row.get("secondary_diagnoses", ""), height=80)
    clinical_info = st.text_area("Klinische Angaben / Fragestellung", value=patient_row.get("clinical_info", ""), height=80)
    pet_ct_report = st.text_area("PET-CT Bericht", value=patient_row.get("pet_ct_report", ""), height=200)

with col2:
    presentation_type = st.text_input("Vorstellungsart", value=patient_row.get("presentation_type", ""))
    main_diagnosis = st.text_input("Diagnose-Kürzel", value=patient_row.get("main_diagnosis", ""))
    ann_arbor_stage = st.text_input("Ann-Arbor Stadium", value=patient_row.get("ann_arbor_stage", ""))
    accompanying_symptoms = st.text_input("Begleitsymptome", value=patient_row.get("accompanying_symptoms", ""))
    prognosis_score = st.text_input("Prognose-Score", value=str(patient_row.get("prognosis_score", "")))


# --- Create Patient Object ---
# Using a dictionary for simplicity here, but Patient class is also fine
patient_data = {
    "id": selected_patient_id,
    "main_diagnosis_text": main_diagnosis_text,
    "secondary_diagnoses": secondary_diagnoses,
    "clinical_info": clinical_info,
    "pet_ct_report": pet_ct_report,
    "presentation_type": presentation_type,
    "main_diagnosis": main_diagnosis,
    "ann_arbor_stage": ann_arbor_stage,
    "accompanying_symptoms": accompanying_symptoms,
    "prognosis_score": prognosis_score,
    "guideline": guideline_provider,
    "location": selected_location
}

# Alternative: Use the Patient class if it provides more methods/logic
# patient = Patient(**patient_data)


# --- Agent Execution Logic ---
if st.button("Multi-Agenten Empfehlung generieren"):
    start_time_total = time.perf_counter()
    # Perform geocoding once here before initializing agents
    user_geopoint = None
    if patient_data.get('location'):
        with st.spinner(f"Geokodierung von '{patient_data['location']}'..."):
            user_geopoint = get_geopoint(patient_data['location'])
            if not user_geopoint:
                 st.warning(f"Konnte '{patient_data['location']}' nicht geokodieren. Studien können nicht nach Entfernung sortiert werden.")

    with st.spinner("Initialisiere LLM und Agenten..."):
        try:
            # Initialize LLM
            llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            logging.info(f"OllamaLLM initialized with model {LLM_MODEL}")

            # Initialize Agents with the shared LLM instance
            diagnostik_agent = DiagnostikAgent(llm)
            studien_agent = StudienAgent(llm, location=patient_data.get('location'))
            therapie_agent = TherapieAgent(llm, patient_data['guideline'])
            report_agent = ReportAgent(llm, output_dir=REPORT_DIR, file_type=REPORT_FILE_TYPE)
            logging.info("Agents initialized.")

        except Exception as e:
            st.error(f"Fehler bei der Initialisierung des LLM oder der Agenten: {e}")
            logging.error(f"LLM/Agent initialization failed: {e}", exc_info=True)
            st.stop()

    with st.spinner("Agenten führen Aufgaben aus..."):
        # --- Context Building ---
        # Context for Diagnostik and Therapie (Therapie uses Diagnostik output)
        base_context = f"""
            Patient ID: {patient_data['id']}
            Leitlinie: {patient_data['guideline']}
            Diagnose: {patient_data['main_diagnosis']} ({patient_data['main_diagnosis_text']})
            Stadium: {patient_data['ann_arbor_stage']}
            Nebendiagnosen: {patient_data['secondary_diagnoses']}
            Klinik/Fragestellung: {patient_data['clinical_info']}
            PET-CT Bericht: {patient_data['pet_ct_report']}
            Begleitsymptome: {patient_data['accompanying_symptoms']}
            Prognose Score: {patient_data['prognosis_score']}
            Vorstellungsart: {patient_data['presentation_type']}
        """
        # Context specifically for StudienAgent
        studien_context = f"""
            Diagnose: {patient_data['main_diagnosis']}
        """
        # If needed include more details for the StudienAgent context
        # Symptome: {patient_data['accompanying_symptoms']}
        # Stadium: {patient_data['ann_arbor_stage']}
        
        logging.info("Contexts prepared.")

        # --- Agent Execution ---
        results = {}
        runtimes = {}
        errors = {}
        threads = []

        if DEBUG_MODE:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug mode is enabled. Only StudienAgent will run.")

            # StudienAgent synchron ausführen
            studien_runner = AgentRunner(studien_agent, "Studien", studien_context)
            studien_runner.run()

            # Ergebnisse anzeigen
            runtimes["Studien"] = studien_runner.runtime
            results["Studien"] = studien_runner.result

            if studien_runner.exception:
                errors["Studien"] = studien_runner.exception
                st.error(f"Studien Agent fehlgeschlagen: {studien_runner.exception}")
            else:
                st.success(f"Studien Agent erfolgreich in {runtimes['Studien']:.2f}s ausgeführt.")
                st.markdown("### 🔬 Empfohlene klinische Studien")

                location_info = patient_data.get('location')
                if location_info and user_geopoint:
                    st.info(f"Suche auf ClinicalTrials.gov nach Studien in '{location_info}'. Studien sind nach der Entfernung des nächstgelegenen Standorts zu Ihrem Suchort sortiert. Es werden die ersten 3 Standorte pro Studie angezeigt.")
                elif location_info and not user_geopoint:
                    st.warning(f"Konnte '{location_info}' nicht geokodieren. Studien konnten nicht nach Entfernung sortiert werden.")
                else:
                    st.info("Kein Ort für die Studiensuche angegeben.")

                study_list = results.get("Studien", [])
                if not study_list:
                    st.info("Keine passenden klinischen Studien gefunden oder Agent hat keine zurückgegeben.")
                else:
                    for i, study in enumerate(study_list):
                        title = study.get("title", "N/A")
                        nct_id = study.get("nct_id", "NCT ID nicht verfügbar")
                        status = study.get("status", "Status unbekannt")
                        summary = study.get("summary", "Keine Zusammenfassung verfügbar.")
                        locations_with_distances = study.get("locations", [])
                        min_distance_km = study.get("min_distance_km")
                        locations_to_display = locations_with_distances[:3]
                        more_locations_count = len(locations_with_distances) - len(locations_to_display)
                        link_url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None

                        st.subheader(f"Studie {i+1}: {title}")

                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.markdown("**NCT ID:**")
                            st.markdown("**Status:**")
                            st.markdown("**Nächster Ort:**")
                        with col2:
                            st.markdown(f"`{nct_id}`")
                            st.markdown(status)
                            if min_distance_km is not None:
                                st.markdown(f"{min_distance_km:.1f} km entfernt")
                            else:
                                st.markdown("Entfernung unbekannt")

                        st.markdown("**Standorte:**")
                        if locations_to_display:
                            sorted_locations = sorted(locations_to_display, key=lambda loc: loc.get("distance_km", float('inf')))
                            for loc_data in sorted_locations:
                                loc_name = loc_data.get("name", "Keine Ortsangaben verfügbar")
                                loc_distance = loc_data.get("distance_km")
                                if loc_distance is not None:
                                    st.markdown(f"- {loc_name} ({loc_distance:.1f} km)")
                                else:
                                    st.markdown(f"- {loc_name} (Entfernung unbekannt)")
                            if more_locations_count > 0:
                                st.markdown(f"... und {more_locations_count} weitere Standorte.")
                        else:
                            st.markdown("Keine Standorte verfügbar")

                        if link_url:
                            st.markdown(f"**Link:** [Zur Studie auf ClinicalTrials.gov]({link_url})", unsafe_allow_html=True)
                        else:
                            st.markdown("**Link:** Nicht verfügbar")

                        with st.expander("Kurzbeschreibung"):
                            st.markdown(summary)

                        st.divider()


                        st.stop()

        else:
            logging.getLogger().setLevel(logging.INFO)

            # Step 1 & 2: Start Diagnostik and Studien Agents in Parallel
            logging.info("Starting Diagnostik and Studien agents in parallel.")
            diag_runner = AgentRunner(diagnostik_agent, "Diagnostik", base_context)
            studien_runner = AgentRunner(studien_agent, "Studien", studien_context)

            diag_thread = threading.Thread(target=diag_runner.run, name="DiagnostikThread")
            studien_thread = threading.Thread(target=studien_runner.run, name="StudienThread")

            threads.extend([diag_thread, studien_thread])
            diag_thread.start()
            studien_thread.start()

            # Step 3: Wait specifically for Diagnostik to finish
            diag_thread.join()
            runtimes["Diagnostik"] = diag_runner.runtime
            if diag_runner.exception:
                errors["Diagnostik"] = diag_runner.exception
                st.error(f"Diagnostik Agent fehlgeschlagen: {diag_runner.exception}")
                logging.error("Diagnostik Agent failed.", exc_info=diag_runner.exception)
                # Decide if stopping is necessary - Therapy depends on it
                st.stop()
            else:
                results["Diagnostik"] = diag_runner.result
                logging.info(f"Diagnostik Agent finished in {runtimes['Diagnostik']:.2f}s. Starting Therapie Agent.")
                diagnostik_output = results["Diagnostik"]

                # Step 4: Start Therapie Agent (needs Diagnostik output)        
                therapie_runner = AgentRunner(therapie_agent, "Therapie", patient_data['guideline'], diagnostik_output)
                therapie_thread = threading.Thread(target=therapie_runner.run, name="TherapieThread")
                threads.append(therapie_thread)
                therapie_thread.start()

            # Step 5 & 6: Wait for any remaining threads (Studien and Therapie)
            # We already joined diag_thread, so we wait for the others in the list
            active_threads = [t for t in threads if t.is_alive()]
            for t in active_threads:
                t.join()
            logging.info("Studien and Therapie agents finished.")

        # Collect results and handle errors for Studien
        runtimes["Studien"] = studien_runner.runtime
        results["Studien"] = studien_runner.result
        if studien_runner.exception:
            errors["Studien"] = studien_runner.exception
            st.warning(f"Studien Agent fehlgeschlagen: {studien_runner.exception}")
            logging.warning("Studien Agent failed.", exc_info=studien_runner.exception)
            results["Studien"] = [] # Ensure empty list on failure

        # Collect results and handle errors for Therapie (check if runner exists)
        if 'therapie_runner' in locals():
            runtimes["Therapie"] = therapie_runner.runtime
            results["Therapie"] = therapie_runner.result
            if therapie_runner.exception:
                errors["Therapie"] = therapie_runner.exception
                st.error(f"Therapie Agent fehlgeschlagen: {therapie_runner.exception}")
                logging.error("Therapie Agent failed.", exc_info=therapie_runner.exception)
        else:
             # This case happens if Diagnostik failed and we stopped
             logging.warning("Therapie agent was not started due to prior failure.")
             results["Therapie"] = None # Ensure it's None

        # --- Display Results ---
        st.success("Agenten-Lauf abgeschlossen!")

        # Therapieempfehlung
        st.markdown("### 🧬 Therapieempfehlung")
        if "Therapie" not in errors and results.get("Therapie"):
            st.markdown(results["Therapie"])
        elif results.get("Therapie") is None and "Therapie" not in errors:
             st.warning("Therapie Agent hat keine Empfehlung generiert.")
        else:
            st.error("Konnte keine Therapieempfehlung generieren.")

        # Studienempfehlung
        st.markdown("### 🔬 Empfohlene klinische Studien")

        location_info = patient_data.get('location')
        if location_info and user_geopoint:
             st.info(f"Suche auf ClinicalTrials.gov nach Studien in '{location_info}'. Studien sind nach der Entfernung des nächstgelegenen Standorts zu Ihrem Suchort sortiert. Es werden die ersten 3 Standorte pro Studie angezeigt.")
        elif location_info and not user_geopoint:
             st.warning(f"Konnnte '{location_info}' nicht geokodieren. Studien konnten nicht nach Entfernung sortiert werden. Die Anzeige erfolgt basierend auf der API-Reihenfolge.")
        else:
             st.info("Kein Ort für die Studiensuche angegeben.")

        study_list = results.get("Studien", []) # Default to empty list

        if "Studien" in errors:
             st.warning(f"Suche nach klinischen Studien fehlgeschlagen: {errors['Studien']}")
        elif not study_list:
             st.info("Keine passenden klinischen Studien gefunden oder Agent hat keine zurückgegeben.")
        else:
            try:
                # Add a little vertical space before the first study
                st.write("")

                # --- MODIFIED display loop to handle list of dicts ---
                for i, study in enumerate(study_list):
                    # Access data using dictionary keys
                    title = study.get("title", "N/A")
                    nct_id = study.get("nct_id", "NCT ID nicht verfügbar")
                    status = study.get("status", "Status unbekannt")
                    summary = study.get("summary", "Keine Zusammenfassung verfügbar.")
                    locations_with_distances = study.get("locations", []) # List of dicts: [{"name": "...", "distance_km": ...}]
                    min_distance_km = study.get("min_distance_km") # Overall min distance for the study

                    # Limit the number of locations displayed
                    locations_to_display = locations_with_distances[:3]
                    more_locations_count = len(locations_with_distances) - len(locations_to_display)

                    link_url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id and nct_id != "NCT ID nicht verfügbar" else None


                    st.subheader(f"Studie {i+1}: {title}")

                    # Display key metadata and distances
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown("**NCT ID:**")
                        st.markdown("**Status:**")
                        # Add Min Distance label
                        st.markdown("**Nächster Ort:**")
                    with col2:
                         st.markdown(f"`{nct_id}`")
                         st.markdown(status)
                         # Display the minimum distance for the study
                         if min_distance_km is not None:
                              st.markdown(f"{min_distance_km:.1f} km entfernt") # Format distance to 1 decimal place
                         else:
                              st.markdown("Entfernung unbekannt")


                    # Display individual Locations with their distances, sorted by distance
                    st.markdown("**Standorte:**") # Label for individual locations
                    if locations_to_display:
                        # Sort locations by distance (ascending)
                        sorted_locations = sorted(locations_to_display, key=lambda loc: loc.get("distance_km", float('inf')))
                        for loc_data in sorted_locations:
                            loc_name = loc_data.get("name", "Keine Ortsangaben verfügbar")
                            loc_distance = loc_data.get("distance_km")
                            if loc_distance is not None:
                                st.markdown(f"- {loc_name} ({loc_distance:.1f} km)")
                            else:
                                st.markdown(f"- {loc_name} (Entfernung unbekannt)")
                        if more_locations_count > 0:
                            st.markdown(f"... und {more_locations_count} weitere Standorte.") # Indicate if more exist
                    else:
                        st.markdown("Keine Standorte verfügbar")

                    # Add Link below locations
                    if link_url:
                        st.markdown(f"**Link:** [Zur Studie auf ClinicalTrials.gov]({link_url})", unsafe_allow_html=True)
                    else:
                        st.markdown("**Link:** Nicht verfügbar")


                    with st.expander("Kurzbeschreibung"):
                        st.markdown(summary)

                    st.divider()
                # --- End of Study Loop ---

            except (TypeError, ValueError) as e:
                # Error handling specific to processing the study list for display
                st.error(f"Fehler bei der Darstellung der Studiendaten: {e}")
                st.text("Empfangene Rohdaten des Studien-Agenten:")
                # Display raw data for debugging if formatting failed
                st.json(study_list if isinstance(study_list, list) else str(study_list))
                logging.error("Error processing or displaying study data (non-table format)", exc_info=True)
            except Exception as e:
                 # Catch any other unexpected errors during display
                 st.error(f"Ein unerwarteter Fehler ist bei der Anzeige der Studien aufgetreten: {e}")
                 logging.error("Error displaying study data (non-table format)", exc_info=True)


        # --- Generate and Offer Report Download ---
        st.markdown("### 📄 Bericht")
        if "Therapie" not in errors and results.get("Therapie"):
            try:
                with st.spinner("Generiere Bericht..."):
                    start_report_time = time.perf_counter()
                    # Combine relevant info for the report context for the LLM
                    llm_input_context = f"Diagnostische Zusammenfassung:\n{diagnostik_output}\n\nNeue Therapie Empfehlung:\n{results['Therapie']}"

                    # Get current date for the board meeting
                    from datetime import date
                    board_date_str = date.today().strftime("%d.%m.%Y") # Format TT.MM.JJJJ

                    # Prepare patient data dict (ensure keys match those used in ReportAgent)
                    # You might need to fetch more patient details if needed
                    report_patient_data = {
                        'last_name': patient_data.get('id', selected_patient_id).split('_')[0] if '_' in patient_data.get('id', selected_patient_id) else patient_data.get('id', selected_patient_id), # Example: Extract last name if ID format is Name_ID
                        'first_name': "", # Add if available
                        'dob': "", # Add if available
                        'pid': selected_patient_id, # Use the ID as PID or fetch real PID
                        'main_diagnosis_text': patient_data.get('main_diagnosis_text', 'N/A')
                    }


                    # Call generate_report_text with new arguments
                    report_text = report_agent.generate_report_text(
                        context=llm_input_context,
                        patient_data=report_patient_data,
                        board_date=board_date_str
                    )

                    report_filename_base = f"{selected_patient_id}_bericht_{board_date_str}" # Add date to filename
                    report_filepath = report_agent.save_report(report_text, report_filename_base)
                    runtimes["Report"] = time.perf_counter() - start_report_time
                    logging.info(f"Report generated and saved to {report_filepath} in {runtimes['Report']:.2f}s.")

                # (Download button logic remains the same, adjust filename if needed)
                if os.path.exists(report_filepath):
                    with open(report_filepath, "r", encoding="utf-8") as f:
                         st.download_button(
                            label=f"📥 Bericht als {REPORT_FILE_TYPE.upper()} herunterladen",
                            data=f.read(),
                            file_name=f"{report_filename_base}.{REPORT_FILE_TYPE}", # Use updated filename
                            mime=f"text/{REPORT_FILE_TYPE}",
                        )
                else:
                     st.error(f"⚠️ Berichtdatei ({report_filepath}) konnte nicht gefunden werden nach dem Speichern.")
            except Exception as e:
                st.error(f"Fehler beim Erstellen oder Speichern des Berichts: {e}")
                logging.error("Report generation/saving failed.", exc_info=True)
        else:
            st.warning("Bericht kann nicht generiert werden, da die Therapieempfehlung fehlt oder fehlerhaft ist.")

        # --- Display Runtimes ---
        total_runtime = time.perf_counter() - start_time_total
        runtime_info = " | ".join(f"{name}: {rt:.2f}s" for name, rt in runtimes.items() if rt is not None)
        st.info(f"⏱️ Laufzeiten: {runtime_info} | Gesamt: {total_runtime:.2f}s")