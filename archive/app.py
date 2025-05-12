import pandas as pd
import streamlit as st
import requests
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

from patient import Patient
from settings import TUBO_EXCEL_FILE_PATH

# LLM
def getLLMResponse(input_text, no_words, category):
    llm = CTransformers(
        model='models/mixtral-8x7b-instruct.gguf.q4_K_M.bin',
        model_type='mistral',
        config={'max_new_tokens': 512, 'temperature': 0.7}
    )
    template = """Write a {category} on {input_text} in less than {no_words} words"""
    prompt = PromptTemplate(input_variables=["input_text", "no_words", "category"], template=template)
    response = llm(prompt.format(category=category, input_text=input_text, no_words=no_words))
    return response

# ClinicalTrials.gov search
def search_clinical_trials(diagnosis, symptoms, stage):
    query = f"{diagnosis} {symptoms} {stage}".replace(" ", "+")
    url = f"https://clinicaltrials.gov/api/query/full_studies?expr={query}&min_rnk=1&max_rnk=5&fmt=json"
    response = requests.get(url)
    
    if response.status_code == 200:
        studies = response.json()["FullStudiesResponse"]["FullStudies"]
        trial_summaries = []
        for study in studies:
            info = study["Study"]["ProtocolSection"]
            title = info["IdentificationModule"].get("OfficialTitle", "No title")
            nct_id = info["IdentificationModule"].get("NCTId", "")
            status = info["StatusModule"].get("OverallStatus", "Unknown")
            summary = info["DescriptionModule"].get("BriefSummary", "No summary")
            trial_summaries.append((title, nct_id, status, summary))
        return trial_summaries
    else:
        return []

# Load patient data
df = pd.read_excel(TUBO_EXCEL_FILE_PATH, skiprows=8)

# Streamlit setup
TITLE = "Tumor Board Therapy Recommendation"
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
if st.button("Empfehlung generieren"):
    st.info("Modell wird ausgeführt…")
    
    input_text = f"""
    Diagnose: {patient.main_diagnosis}
    Beschreibung: {patient.main_diagnosis_text}
    Nebendiagnosen: {patient.secondary_diagnoses}
    Klinik: {patient.clinical_info}
    PET-CT: {patient.pet_ct_report}
    Symptome: {patient.accompanying_symptoms}
    Prognose: {patient.prognosis_score}
    Leitlinie: {guideline_provider}
    """
    recommendation = getLLMResponse(input_text=input_text, no_words="200", category="recommendation")
    st.markdown("### Generierte Empfehlung")
    st.markdown(recommendation)

    # Search ClinicalTrials.gov
    st.markdown("---")
    st.markdown("### 🔍 Relevante klinische Studien (clinicaltrials.gov)")
    trials = search_clinical_trials(patient.main_diagnosis, patient.accompanying_symptoms, patient.ann_arbor_stage)

    if trials:
        for title, nct_id, status, summary in trials:
            st.markdown(f"**{title}**  \nStatus: `{status}`  \n[NCT ID: {nct_id}](https://clinicaltrials.gov/study/{nct_id})  \n_{summary}_  \n")
    else:
        st.warning("Keine passenden klinischen Studien gefunden.")
