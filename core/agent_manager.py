# core/agent_manager.py
import logging
import threading
import time
from langchain_ollama import OllamaLLM

import config
from agents.base_agent import AgentRunner
from agents.diagnostik_agent import DiagnostikAgent
from agents.studien_agent import StudienAgent
from agents.therapie_agent import TherapieAgent
from agents.report_agent import ReportAgent
from services.geocoding_service import get_geopoint

logger = logging.getLogger(__name__)

class AgentWorkflowManager:
    def __init__(self, patient_data: dict):
        self.patient_data = patient_data
        self.llm = None
        self.diagnostik_agent = None
        self.studien_agent = None
        self.therapie_agent = None
        self.report_agent = None # Initialize ReportAgent here
        self.user_geopoint = None

        self.results = {}
        self.runtimes = {}
        self.errors = {}

    def _initialize_components(self):
        """Initializes LLM and all agents."""
        logger.info("Initializing LLM and Agents...")
        self.llm = OllamaLLM(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        logger.info(f"OllamaLLM initialized with model {config.LLM_MODEL}")

        self.diagnostik_agent = DiagnostikAgent(self.llm)
        self.studien_agent = StudienAgent(self.llm, search_location_str=self.patient_data.get('location'))
        self.therapie_agent = TherapieAgent(self.llm, guideline_provider=self.patient_data['guideline'])
        self.report_agent = ReportAgent(self.llm, output_dir=config.REPORT_DIR, file_type=config.REPORT_FILE_TYPE)
        logger.info("All Agents initialized.")

        if self.patient_data.get('location'):
            logger.info(f"Attempting geocoding for: {self.patient_data['location']}")
            self.user_geopoint = get_geopoint(self.patient_data['location'])
            if not self.user_geopoint:
                logger.warning(f"Could not geocode '{self.patient_data['location']}'.")
            else:
                logger.info(f"Geocoded '{self.patient_data['location']}' to {self.user_geopoint}")


    def _prepare_contexts(self) -> tuple[str, str]:
        """Prepares contexts for different agents."""
        logger.info("Preparing agent contexts...")
        # Base context for Diagnostik and subsequently Therapie
        base_context = f"""
            Patient ID: {self.patient_data['id']}
            Leitlinie: {self.patient_data['guideline']}
            Diagnose-Kürzel: {self.patient_data.get('main_diagnosis', 'N/A')}
            Hauptdiagnose (Text): {self.patient_data.get('main_diagnosis_text', 'N/A')}
            Stadium: {self.patient_data.get('ann_arbor_stage', 'N/A')}
            Nebendiagnosen: {self.patient_data.get('secondary_diagnoses', 'N/A')}
            Klinik/Fragestellung: {self.patient_data.get('clinical_info', 'N/A')}
            PET-CT Bericht: {self.patient_data.get('pet_ct_report', 'N/A')}
            Begleitsymptome: {self.patient_data.get('accompanying_symptoms', 'N/A')}
            Prognose Score: {self.patient_data.get('prognosis_score', 'N/A')}
            Vorstellungsart: {self.patient_data.get('presentation_type', 'N/A')}
        """
        # Context specifically for StudienAgent - focused on diagnosis
        # Ensure there's a fallback if main_diagnosis_text is empty
        studien_diagnosis_term = self.patient_data.get('main_diagnosis_text', '').strip()
        if not studien_diagnosis_term:
            studien_diagnosis_term = self.patient_data.get('main_diagnosis', '').strip()
        if not studien_diagnosis_term:
            logger.warning("No diagnosis term found for StudienAgent context. Using 'tumor' as a generic fallback.")
            studien_diagnosis_term = "tumor" # Generic fallback, less ideal

        studien_context = studien_diagnosis_term # Pass only the diagnosis string
        logger.info(f"Base context prepared. Studien context based on: '{studien_context}'")
        return base_context, studien_context

    def run_workflow(self):
        """Executes the full multi-agent workflow."""
        self.results = {}
        self.runtimes = {}
        self.errors = {}
        threads = []

        try:
            self._initialize_components()
        except Exception as e:
            logger.error(f"Critical error during component initialization: {e}", exc_info=True)
            self.errors["Initialization"] = e
            return # Cannot proceed

        base_context, studien_context = self._prepare_contexts()

        # 1. Diagnostik and Studien Agents (Parallel)
        logger.info("Starting Diagnostik and Studien agents in parallel.")
        diag_runner = AgentRunner(self.diagnostik_agent, "Diagnostik", base_context)
        # Ensure studien_context (diagnosis string) is passed to StudienAgent
        studien_runner = AgentRunner(self.studien_agent, "Studien", studien_context)


        diag_thread = threading.Thread(target=diag_runner.run, name="DiagnostikThread")
        studien_thread = threading.Thread(target=studien_runner.run, name="StudienThread")
        threads.extend([diag_thread, studien_thread])
        diag_thread.start()
        studien_thread.start()

        # 2. Wait for Diagnostik (Therapie depends on it)
        diag_thread.join()
        self.runtimes["Diagnostik"] = diag_runner.runtime
        if diag_runner.exception:
            self.errors["Diagnostik"] = diag_runner.exception
            logger.error("Diagnostik Agent failed. Therapie Agent will not run.", exc_info=diag_runner.exception)
            # Ensure studien thread is also joined if we might return early
            if studien_thread.is_alive(): studien_thread.join()
            self.runtimes["Studien"] = studien_runner.runtime # Collect runtime
            if studien_runner.exception: self.errors["Studien"] = studien_runner.exception
            self.results["Studien"] = studien_runner.result if not studien_runner.exception else []
            return # Stop if Diagnostik fails
        else:
            self.results["Diagnostik"] = diag_runner.result
            logger.info(f"Diagnostik Agent finished. Output snippet: {str(self.results['Diagnostik'])[:100]}...")

            # 3. Therapie Agent (Sequential after Diagnostik)
            logger.info("Starting Therapie agent.")
            # Pass the actual diagnostik_output as context to TherapieAgent
            therapie_runner = AgentRunner(self.therapie_agent, "Therapie", self.results["Diagnostik"])
            therapie_thread = threading.Thread(target=therapie_runner.run, name="TherapieThread")
            threads.append(therapie_thread)
            therapie_thread.start()

        # 4. Wait for all remaining threads
        for t in threads:
            if t.is_alive() and t is not diag_thread : # diag_thread already joined
                t.join()
        logger.info("All agent threads (Studien, Therapie) have completed.")

        # 5. Collect results and errors
        self.runtimes["Studien"] = studien_runner.runtime
        self.results["Studien"] = studien_runner.result
        if studien_runner.exception:
            self.errors["Studien"] = studien_runner.exception
            logger.warning("Studien Agent failed.", exc_info=studien_runner.exception)
            if "Studien" not in self.results or self.results["Studien"] is None: # Ensure it's an empty list on error
                 self.results["Studien"] = []


        if 'therapie_runner' in locals(): # Check if therapie_runner was initialized
            self.runtimes["Therapie"] = therapie_runner.runtime
            self.results["Therapie"] = therapie_runner.result
            if therapie_runner.exception:
                self.errors["Therapie"] = therapie_runner.exception
                logger.error("Therapie Agent failed.", exc_info=therapie_runner.exception)
        else: # This case if Diagnostik failed
            self.results["Therapie"] = None # Ensure it's None
            logger.info("Therapie agent was not started due to prior Diagnostik failure.")
        
        logger.info("Agent workflow execution finished.")
        # ReportAgent is available as self.report_agent if needed outside (e.g., in UI for download)