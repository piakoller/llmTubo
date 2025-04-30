from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import requests
import os
import urllib.parse

# Basis-Agent-Klasse
class Agent:
    def __init__(self, name, role_description):
        self.name = name
        self.role_description = role_description
        self.llm = OllamaLLM(model="llama3.2", temperature=0.7)

    def respond(self, context):
        prompt = PromptTemplate(
            input_variables=["role", "context"],
            template="""
            Sie sind ein Experten-Agent. Ihre Rolle ist: {role}
            Basierend auf dem folgenden Kontext, antworten Sie angemessen:

            {context}
            """
                )
        return self.llm.invoke(prompt.format(role=self.role_description, context=context))


# Konkrete Agenten

class DiagnostikAgent(Agent):
    def __init__(self):
        super().__init__("Diagnostik", "Analysiere Diagnose, Symptome und klinische Berichte.")

class StudienAgent(Agent):
    def __init__(self):
        super().__init__(
            "Studienrecherche",
            "Filtere relevante klinische Studien von clinicaltrials.gov basierend auf Diagnose und Patientenprofil und präsentiere die Ergebnisse strukturiert."
        )

    # def search_clinical_trials(self, diagnosis, symptoms, stage):
    #     search_terms = f"{diagnosis} {symptoms} {stage}".strip().replace("  ", " ")
    #     query = urllib.parse.quote_plus(search_terms)  # URL-sichere Kodierung
    #     url = f"https://clinicaltrials.gov/api/v2/studies?query.term={query}&pageSize=5"
    #     try:
    #         response = requests.get(url)
    #         response.raise_for_status()
    #         data = response.json()
    #         studies = data.get("studies", [])
    #         trial_summaries = []
    #         for study in studies:
    #             protocol = study.get("protocolSection", {})
    #             id_module = protocol.get("identificationModule", {})
    #             status_module = protocol.get("statusModule", {})
    #             desc_module = protocol.get("descriptionModule", {})
    #             title = id_module.get("officialTitle", "No title")
    #             nct_id = id_module.get("nctId", "")
    #             status = status_module.get("overallStatus", "Unknown")
    #             summary = desc_module.get("briefSummary", "No summary")
    #             trial_summaries.append({"Titel": title, "Ort": "Noch keine Information", "Sponsor": "Noch keine Information", "Link": f"https://clinicaltrials.gov/study/{nct_id}"})
    #         return trial_summaries
    #     except requests.RequestException as e:
    #         print(f"API error: {e}")
    #         return []

    # ClinicalTrials.gov search
    def search_clinical_trials(self, diagnosis, symptoms, stage):
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

    def respond(self, diagnosis_context):
        diagnosis, symptoms, stage = "", "", ""
        for line in diagnosis_context.split("\n"):
            if "Diagnose" in line and ":" in line:
                diagnosis = line.split(":", 1)[1].strip()
            elif "Symptome" in line and ":" in line:
                symptoms = line.split(":", 1)[1].strip()
            elif ("Stadium" in line or "Staging" in line) and ":" in line:
                stage = line.split(":", 1)[1].strip()

        trials = self.search_clinical_trials(diagnosis, symptoms, stage)
        # trials = self.search_clinical_trials(patient.main_diagnosis, patient.accompanying_symptoms, patient.ann_arbor_stage)

        if not trials:
            return "Keine passenden klinischen Studien gefunden."

        # Gib eine Liste von Dictionaries zurück, passend zur extract_study_data Funktion
        return trials

    
class TherapieAgent(Agent):
    def __init__(self):
        super().__init__("Therapieempfehlung", "Erstelle eine Therapieempfehlung basierend auf allen bereitgestellten Informationen.")

class ReportAgent:
    def __init__(self, output_dir="/generated_report", file_type="pdf"):
        self.llm = OllamaLLM(model="llama3.2", temperature=0.3)
        self.output_dir = output_dir
        self.file_type = file_type

    def generate_report_text(self, context):
        prompt = PromptTemplate(
            input_variables=["context"],
            template="""
            Basierend auf der folgenden Therapieempfehlung, erstelle einen strukturierten medizinischen Bericht mit Abschnitten wie "Hintergrund", "Therapieplan" und "Referenzen".

            Empfehlungskontext:
            {context}
            """
        )
        return self.llm.invoke(prompt.format(context=context))

    def save_report(self, markdown_text, patient_id):
        self._save_markdown_file(markdown_text, self.output_dir, patient_id)

    def _save_markdown_file(self, markdown_text, output_path, filename):
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, f"{filename}.md")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(markdown_text)
        except OSError as e:
            raise RuntimeError(f"Fehler beim Speichern des Berichts: {e}")

