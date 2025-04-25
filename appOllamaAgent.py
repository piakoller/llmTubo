import pandas as pd
import streamlit as st
import requests
import time
import threading
import logging
import os

from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from patient import Patient
from settings import TUBO_EXCEL_FILE_PATH
from multiagent import DiagnostikAgent, StudienAgent, TherapieAgent, ReportAgent

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# LLM mit Ollama
def getLLMResponse(input_text, no_words, category):
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    template = """Write a {category} on {input_text} in less than {no_words} words"""
    prompt = PromptTemplate(input_variables=["input_text", "no_words", "category"], template=template)
    response = llm.invoke(prompt.format(category=category, input_text=input_text, no_words=no_words))
    return response

# ClinicalTrials.gov search
def search_clinical_trials(diagnosis, symptoms, stage):
    search_terms = f"{diagnosis} {symptoms} {stage}".strip().replace("  ", " ")
    query = search_terms.replace(" ", "+")
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term={query}&pageSize=5"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        studies = data.get("studies", [])
        trial_summaries = []
        for study in studies:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            desc_module = protocol.get("descriptionModule", {})
            title = id_module.get("officialTitle", "No title")
            nct_id = id_module.get("nctId", "")
            status = status_module.get("overallStatus", "Unknown")
            summary = desc_module.get("briefSummary", "No summary")
            trial_summaries.append((title, nct_id, status, summary))
        return trial_summaries
    except requests.RequestException as e:
        print(f"API error: {e}")
        return []

# Hilfsfunktionen
class AgentRunner:
    def __init__(self, agent, name, context=""):
        self.agent = agent
        self.name = name
        self.context = context
        self.result = None
        self.runtime = None
        self.exception = None

    def run(self):
        logging.info(f"Starte Agent: {self.name}")
        start = time.perf_counter()
        try:
            self.result = self.agent.respond(self.context)
        except Exception as e:
            logging.error(f"Fehler im Agent '{self.name}': {e}")
            self.exception = e
        finally:
            self.runtime = time.perf_counter() - start
            logging.info(f"{self.name} abgeschlossen in {self.runtime:.2f}s")

def combine_contexts(*contexts):
    combined = []
    for ctx in contexts:
        if isinstance(ctx, list):
            combined.append("\n\n".join(ctx))
        elif isinstance(ctx, str):
            combined.append(ctx)
    return "\n\n".join(combined)

# Load patient data
df = pd.read_excel(TUBO_EXCEL_FILE_PATH, skiprows=8)

# Streamlit setup
TITLE = "Tumorboard Therapie Empfehlung"
st.set_page_config(page_title=TITLE, page_icon="🧬", layout="wide")
st.title(TITLE)

selected_patient_id = st.sidebar.selectbox("Patienten-ID auswählen", df["ID"].tolist())
guideline_provider = st.sidebar.selectbox("Leitlinie wählen", ["ESMO", "Onkopedia", "S3"])

# Get selected patient row
patient_row = next(df[df["ID"] == selected_patient_id].itertuples())

# Layout
col1, col2 = st.columns([2, 1])
with col1:
    main_diagnosis_text = st.text_area("Hauptdiagnose (Text)", value=patient_row.main_diagnosis_text, height=200)
    secondary_diagnoses = st.text_area("Relevante Nebendiagnosen", value=patient_row.secondary_diagnoses, height=80)
    clinical_info = st.text_area("Klinische Angaben / Fragestellung", value=patient_row.clinical_info, height=80)
    pet_ct_report = st.text_area("PET-CT Bericht", value=patient_row.pet_ct_report, height=200)

with col2:
    presentation_type = st.text_input("Vorstellungsart", value=patient_row.presentation_type)
    main_diagnosis = st.text_input("Diagnose-Kürzel", value=patient_row.main_diagnosis)
    ann_arbor_stage = st.text_input("Ann-Arbor Stadium", value=patient_row.ann_arbor_stage)
    accompanying_symptoms = st.text_input("Begleitsymptome", value=patient_row.accompanying_symptoms)
    prognosis_score = st.text_input("Prognose-Score", value=str(patient_row.prognosis_score))

# Create patient object
patient = Patient(
    id=selected_patient_id,
    main_diagnosis_text=main_diagnosis_text,
    secondary_diagnoses=secondary_diagnoses,
    clinical_info=clinical_info,
    pet_ct_report=pet_ct_report,
    presentation_type=presentation_type,
    main_diagnosis=main_diagnosis,
    ann_arbor_stage=ann_arbor_stage,
    accompanying_symptoms=accompanying_symptoms,
    prognosis_score=prognosis_score,
)

# Generate recommendation and search trials
if st.button("Multi-Agenten Empfehlung generieren"):
    with st.spinner("Agenten arbeiten..."):
        base_context = f"""
        Patient ID: {patient.id}
        Diagnose: {patient.main_diagnosis}
        Klinik: {patient.clinical_info}
        PET-CT: {patient.pet_ct_report}
        Symptome: {patient.accompanying_symptoms}
        Prognose: {patient.prognosis_score}
        """

        diagnostik_agent = DiagnostikAgent()
        studien_agent = StudienAgent()
        therapie_agent = TherapieAgent()

        # Diagnostik synchron starten
        diag_runner = AgentRunner(diagnostik_agent, "Diagnostik", base_context)
        diag_runner.run()

        if diag_runner.exception:
            st.error("Diagnostik fehlgeschlagen. Bitte prüfen.")
            st.stop()

        # Studien Agent parallel starten
        studien_runner = AgentRunner(studien_agent, "Studien", diag_runner.result)
        studien_thread = threading.Thread(target=studien_runner.run)
        studien_thread.start()

        studien_thread.join()

        if studien_runner.exception:
            st.error("Fehler beim Studien-Agent.")
            st.stop()

       # Therapie Agent
        combined_context = combine_contexts(diag_runner.result, studien_runner.result)
        therapie_runner = AgentRunner(therapie_agent, "Therapie", combined_context)
        therapie_runner.run()

        # Therapieempfehlung anzeigen
        st.success("Empfehlung generiert!")
        st.markdown("### 🧬 Therapieempfehlung")
        st.markdown(therapie_runner.result)

        # Bericht generieren
        report_agent = ReportAgent(output_dir="generated_report", file_type="pdf")
        report_markdown = report_agent.generate_report_text(therapie_runner.result)
        report_agent.save_report(report_markdown, selected_patient_id)

        # PDF-Datei laden und Download-Button anzeigen
        pdf_path = os.path.join("generated_report", f"{selected_patient_id}.pdf")

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Bericht als PDF herunterladen",
                data=pdf_bytes,
                file_name=f"{selected_patient_id}_bericht.pdf",
                mime="application/pdf",
            )
        else:
            st.error("⚠️ PDF konnte nicht gefunden werden.")

        # Laufzeiten anzeigen
        st.info(f"⏱️ Diagnostik: {diag_runner.runtime:.2f}s | Studien: {studien_runner.runtime:.2f}s | Therapie: {therapie_runner.runtime:.2f}s")
