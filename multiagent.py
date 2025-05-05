import requests
import os
import urllib.parse
import time
import logging
from abc import ABC, abstractmethod

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# --- Configuration ---
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_PAGE_SIZE = 5
REQUESTS_TIMEOUT = 30 # Timeout for API calls in seconds

# --- Logging ---
# Logger can be configured externally, but setting a default here is helpful
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s")


# --- Agent Runner ---
class AgentRunner:
    """Handles the execution of an agent, capturing results, runtime, and exceptions."""
    def __init__(self, agent: 'Agent', name: str, context: str):
        self.agent = agent
        self.name = name
        self.context = context
        self.result = None
        self.runtime = None
        self.exception = None

    def run(self):
        """Executes the agent's respond method and records metrics."""
        logger.info(f"Agent '{self.name}' started.")
        start_time = time.perf_counter()
        try:
            self.result = self.agent.respond(self.context)
        except Exception as e:
            logger.error(f"Agent '{self.name}' encountered an error: {e}", exc_info=True)
            self.exception = e
        finally:
            self.runtime = time.perf_counter() - start_time
            status = "failed" if self.exception else "completed"
            logger.info(f"Agent '{self.name}' {status} in {self.runtime:.2f}s.")


# --- Base Agent Class ---
class Agent(ABC):
    """Abstract base class for all agents."""
    def __init__(self, name: str, role_description: str, llm: OllamaLLM):
        self.name = name
        self.role_description = role_description
        if llm is None:
            raise ValueError("LLM instance must be provided to the agent.")
        self.llm = llm
        logger.info(f"Agent '{self.name}' initialized.")

    @abstractmethod
    def respond(self, context: str) -> any:
        """
        Process the given context and return a response.
        The response type depends on the specific agent.
        """
        pass

    def _invoke_llm(self, prompt_template: str, context_vars: dict) -> str:
        """Helper method to invoke the LLM with a specific template."""
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        try:
            response = chain.invoke(context_vars)
            return response
        except Exception as e:
            logger.error(f"LLM invocation failed for agent {self.name}: {e}", exc_info=True)
            # Re-raise or return a specific error message
            raise RuntimeError(f"LLM call failed for {self.name}") from e


# --- Concrete Agent Implementations ---

class DiagnostikAgent(Agent):
    """Analyzes diagnosis, symptoms, and clinical reports."""
    def __init__(self, llm: OllamaLLM):
        super().__init__("Diagnostik", "Analysiere Diagnose, Symptome und klinische Berichte und fasse die wichtigsten Punkte zusammen.", llm)

    def respond(self, context: str) -> str:
        """Uses LLM to analyze the provided context."""
        template = """
        Du bist ein medizinischer Experte für Diagnostik. Deine Rolle ist: {role}
        Analysiere den folgenden Patientenkontext und fasse die relevanten diagnostischen Informationen, das Stadium und wichtige klinische Punkte klar und prägnant zusammen.

        Kontext:
        ---
        {context}
        ---

        Zusammenfassung der Diagnostik:
        """
        return self._invoke_llm(template, {"role": self.role_description, "context": context})


class StudienAgent(Agent):
    """Searches for relevant clinical trials on clinicaltrials.gov."""
    def __init__(self, llm: OllamaLLM):
        # This agent currently doesn't use the LLM directly, but keeps the interface consistent
        super().__init__(
            "Studienrecherche",
            "Filtere relevante klinische Studien von clinicaltrials.gov basierend auf Diagnose, Symptomen und Stadium.",
            llm
        )

    def _search_clinical_trials(self, diagnosis: str, symptoms: str, stage: str) -> list[tuple[str, str, str, str]]:
        """Performs the search on clinicaltrials.gov API."""
        if not diagnosis: # Basic validation
             logger.warning("Skipping clinical trial search: Diagnosis is missing.")
             return []

        search_terms = f"{diagnosis} {symptoms} {stage}".strip().replace("  ", " ")
        query = urllib.parse.quote_plus(search_terms)
        url = f"{CLINICAL_TRIALS_API_URL}?query.term={query}&pageSize={CLINICAL_TRIALS_PAGE_SIZE}"
        logger.info(f"Searching clinical trials with URL: {url}")

        try:
            response = requests.get(url, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            studies = data.get("studies", [])
            trial_summaries = []
            for study in studies:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                desc_module = protocol.get("descriptionModule", {})

                title = id_module.get("officialTitle", "N/A")
                nct_id = id_module.get("nctId", "N/A")
                status = status_module.get("overallStatus", "Unknown")
                summary = desc_module.get("briefSummary", "No summary provided.")

                trial_summaries.append((title, nct_id, status, summary))

            logger.info(f"Found {len(trial_summaries)} clinical trials for query: '{search_terms}'")
            return trial_summaries

        except requests.exceptions.Timeout:
             logger.error(f"API request timed out for clinical trials search: {url}")
             return []
        except requests.exceptions.RequestException as e:
            logger.error(f"API error during clinical trials search ({url}): {e}")
            return []
        except Exception as e: # Catch potential JSON parsing errors etc.
            logger.error(f"Unexpected error during clinical trials processing: {e}", exc_info=True)
            return []

    def _parse_context(self, diagnosis_context: str) -> tuple[str, str, str]:
        """Extracts diagnosis, symptoms, and stage from the context string."""
        diagnosis, symptoms, stage = "", "", ""
        context_lines = diagnosis_context.strip().split('\n')
        for line in context_lines:
            line_lower = line.lower()
            if ":" in line:
                 key, value = line.split(":", 1)
                 key = key.strip().lower()
                 value = value.strip()
                 if "diagnose" in key:
                     diagnosis = value
                 elif "symptome" in key:
                     symptoms = value
                 elif "stadium" in key: # Check for 'stadium'
                     stage = value

        logger.debug(f"Parsed context for studies: D='{diagnosis}', S='{symptoms}', St='{stage}'")
        return diagnosis, symptoms, stage


    def respond(self, diagnosis_context: str) -> list[tuple[str, str, str, str]]:
        """Parses context, searches trials, and returns results."""
        diagnosis, symptoms, stage = self._parse_context(diagnosis_context)

        trials = self._search_clinical_trials(diagnosis, symptoms, stage)

        if not trials:
            logger.info("No suitable clinical trials found or search failed.")
            return [] # Return empty list consistently

        # Return list of tuples directly, matching expected format in app
        return trials


class TherapieAgent(Agent):
    """Generates a therapy recommendation based on provided information."""
    def __init__(self, llm: OllamaLLM):
        super().__init__(
            "Therapieempfehlung",
            "Erstelle eine detaillierte, leitlinienbasierte Therapieempfehlung basierend auf der Diagnoseanalyse.",
             llm
         )

    def respond(self, context: str) -> str:
        """Uses LLM to generate therapy recommendations."""
        template = """
        Du bist ein Onkologie-Experte, spezialisiert auf Therapieplanung. Deine Rolle ist: {role}
        Basierend auf der folgenden diagnostischen Zusammenfassung und den ursprünglichen Patienteninformationen (implizit im Kontext enthalten), erstelle eine umfassende und begründete Therapieempfehlung. Berücksichtige die angegebene Leitlinie.

        Diagnostische Zusammenfassung / Kontext:
        ---
        {context}
        ---

        Therapieempfehlung (inkl. Begründung):
        """
        return self._invoke_llm(template, {"role": self.role_description, "context": context})


class ReportAgent:
    """Generates and saves a structured medical report."""
    def __init__(self, llm: OllamaLLM, output_dir: str = "generated_report", file_type: str = "md"):
        if llm is None:
             raise ValueError("LLM instance must be provided to the ReportAgent.")
        self.llm = llm
        self.output_dir = output_dir
        # Ensure file type is supported (currently only md)
        self.file_type = file_type.lower()
        if self.file_type not in ["md", "markdown"]:
            logger.warning(f"Unsupported file type '{file_type}'. Defaulting to 'md'.")
            self.file_type = "md"
        logger.info(f"ReportAgent initialized. Output directory: '{output_dir}', File type: '{self.file_type}'")


    def generate_report_text(self, context: str, patient_data: dict, board_date: str) -> str:
        """Uses LLM to structure the context into a report format, incorporating patient data."""
        logger.info("Generating report text...")

        # Prepare patient info string for the prompt
        # Adjust keys based on your actual patient_data dictionary
        patient_info_str = f"""
            Patient: {patient_data.get('last_name', 'Nachname')}, {patient_data.get('first_name', 'Vorname')}
            Geburtsdatum: {patient_data.get('dob', 'XX.XX.XXXX')}
            PID: {patient_data.get('pid', 'XXXXXXXX')}
            """

        # The main diagnosis might be in patient_data or context, adjust as needed
        main_diagnosis_str = patient_data.get('main_diagnosis_text', 'Hauptdiagnose nicht spezifiziert')

        prompt = PromptTemplate(
            input_variables=["patient_info", "board_date", "main_diagnosis", "context"],
            template="""
            Du bist ein medizinischer Sekretär/Assistent, der Tumorboard-Protokolle im Stil des Inselspitals Bern (Schweiz) verfasst.
            Erstelle ein Protokoll für das Endokrine Tumorboard vom {board_date}.

            **Patientendaten:**
            {patient_info}

            **Hauptdiagnose (basierend auf vorliegenden Informationen):**
            1. {main_diagnosis}

            Der folgende `{context}` enthält eine Zusammenfassung der aktuellen diagnostischen Situation durch einen Experten und die daraus resultierende aktuelle Therapieempfehlung des Tumorboards.

            **Eingabeinformationen (Aktuelle Beurteilung & Empfehlung):**
            ---
            {context}
            ---

            **Deine Aufgabe:**
            Formatiere **nur** die Informationen aus dem `{context}` in ein medizinisches Protokoll im Markdown-Format.
            Orientiere dich am Stil des Inselspitals, insbesondere bei den Abschnitten 'Beurteilung' und 'Empfehlungen'.
            Fasse die diagnostische Einschätzung aus dem `{context}` unter der Überschrift '## Beurteilung' zusammen.
            Liste die aktuelle Therapieempfehlung(en) aus dem `{context}` unter der Überschrift '## Empfehlungen' auf, idealerweise als Bullet Points (`*` oder `•`).

            **WICHTIG:**
            *   Füge **keine** historischen Daten (frühere Scans, Behandlungen, Empfehlungen, Labore, Verläufe) hinzu, die nicht explizit im `{context}` stehen.
            *   Erfinde **keine** Patientendaten, Adressen, Board-Teilnehmer oder Abteilungsleiter.
            *   Generiere **keinen** Briefkopf oder Fusszeile.
            *   Konzentriere dich darauf, die *aktuelle Beurteilung* und die *neuen Empfehlungen* basierend auf dem `{context}` klar darzustellen.

            **Protokoll (Markdown):**

            ## Beurteilung
            [Hier die Zusammenfassung der diagnostischen Einschätzung aus dem context einfügen]

            ## Empfehlungen
            [Hier die aktuelle(n) Therapieempfehlung(en) aus dem context als Bullet Points auflisten]

            """
        )
        chain = prompt | self.llm
        try:
            report_text = chain.invoke({
                "patient_info": patient_info_str,
                "board_date": board_date,
                "main_diagnosis": main_diagnosis_str,
                "context": context
                })
            logger.info("Report text generated successfully.")
            # Prepend a basic header to the LLM output
            header = f"# Endokrines Tumorboard vom {board_date}\n\n"
            full_report = header + patient_info_str + f"\n## Diagnosen\n1. {main_diagnosis_str}\n\n" + report_text
            return full_report
        except Exception as e:
            logger.error(f"LLM invocation failed during report generation: {e}", exc_info=True)
            raise RuntimeError("LLM call failed during report generation") from e

    def save_report(self, report_text: str, filename_base: str) -> str:
        """Saves the generated report text to a file."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f"{filename_base}.{self.file_type}")
        logger.info(f"Attempting to save report to: {filepath}")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Report successfully saved to {filepath}")
            return filepath
        except OSError as e:
            logger.error(f"Failed to save report to {filepath}: {e}", exc_info=True)
            raise RuntimeError(f"Error saving report file: {e}") from e
        except Exception as e: # Catch any other unexpected errors during file writing
             logger.error(f"An unexpected error occurred while saving the report to {filepath}: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error saving report file: {e}") from e