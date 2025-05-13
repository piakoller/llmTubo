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


    def _parse_diagnosis_for_studien_agent(self, full_patient_context: str) -> str:
        """
        Extracts the diagnosis search term for the StudienAgent from the full patient context,
        looking for a line starting with 'Diagnose:'.
        This function is adapted from your provided _parse_context.
        """
        diagnosis_search_term = ""
        context_lines = full_patient_context.strip().split('\n')
        for line in context_lines:
            line_stripped = line.strip() # Work with the stripped line for matching
            # Check for "Diagnose:" and also "Diagnose-Kürzel:" as a common variation
            if line_stripped.lower().startswith("diagnose:") or \
               line_stripped.lower().startswith("diagnose-kürzel:"):
                
                # Determine which prefix was found for accurate splitting
                prefix_to_split = "diagnose:" if line_stripped.lower().startswith("diagnose:") else "diagnose-kürzel:"
                
                try:
                    # Get the content after "Diagnose:" or "Diagnose-Kürzel:"
                    content_after_prefix = line_stripped.split(":", 1)[1].strip()
                    
                    # Check for text in parentheses: "CODE (TEXT)"
                    parts = content_after_prefix.split("(", 1)
                    if len(parts) > 1 and parts[1].endswith(")"): # Ensure closing parenthesis exists
                        # Extract text from "TEXT)"
                        text_in_parentheses = parts[1][:-1].strip() # Remove trailing ')' and strip
                        if text_in_parentheses:
                            diagnosis_search_term = text_in_parentheses
                            logger.info(f"Parsed diagnosis for StudienAgent (from text in parentheses): '{diagnosis_search_term}' from line: '{line_stripped}'")
                            break 
                        else: # Parentheses are empty like "CODE ()"
                            diagnosis_search_term = parts[0].strip() # Use the code part
                            logger.info(f"Parsed diagnosis for StudienAgent (from code, empty parentheses): '{diagnosis_search_term}' from line: '{line_stripped}'")
                            break
                    else: # No parentheses, or malformed
                        diagnosis_search_term = content_after_prefix # Use the whole content after "Diagnose:"
                        logger.info(f"Parsed diagnosis for StudienAgent (no parentheses): '{diagnosis_search_term}' from line: '{line_stripped}'")
                        break
                except IndexError:
                    logger.warning(f"Could not properly parse line starting with 'Diagnose:' or 'Diagnose-Kürzel:': '{line_stripped}'")
                    continue # Try next line
        
        if not diagnosis_search_term:
            logger.warning(f"Could not find a 'Diagnose:' or 'Diagnose-Kürzel:' line to parse for StudienAgent in context. Context snippet: {full_patient_context[:200]}...")
            # Fallback: use the 'main_diagnosis' field directly if parsing fails
            diagnosis_search_term = self.patient_data.get('main_diagnosis', '').strip()
            if diagnosis_search_term:
                logger.info(f"Falling back to 'main_diagnosis' field for StudienAgent: '{diagnosis_search_term}'")
            else:
                diagnosis_search_term = "tumor" # Last resort fallback
                logger.warning(f"Using generic 'tumor' for StudienAgent as all parsing attempts failed.")

        return diagnosis_search_term.strip()
    
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
        # The `_parse_diagnosis_for_studien_agent` function will now parse this `base_context`.
        studien_search_term = self._parse_diagnosis_for_studien_agent(base_context)
        
        # The context for StudienAgent is now the output of this parser.
        studien_context = studien_search_term 

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