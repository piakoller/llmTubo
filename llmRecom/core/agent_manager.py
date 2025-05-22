# core/agent_manager.py
import logging
import threading
import time
from datetime import datetime as dt
import os
import uuid
import json
from datetime import datetime as dt
from langchain_ollama import OllamaLLM

import config
from agents.base_agent import AgentRunner
from agents.diagnostik_agent import DiagnostikAgent
from agents.studien_agent import StudienAgent
from agents.therapie_agent import TherapieAgent
from agents.report_agent import ReportAgent
from services.geocoding_service import get_geopoint

logger = logging.getLogger(__name__)
human_eval_json_lock = threading.Lock()

def save_for_human_evaluation(
    patient_id: str,
    patient_context_summary: str,
    llm_full_recommendation: str,      
    llm_raw_response_with_think: str, 
    llm_think_block: str,
    guideline_used: str,
    llm_interaction_id: str,
    agent_name: str = "TherapieAgentOutput"
):
    # Ensure the directory for the JSON file exists
    eval_file_path = config.HUMAN_EVAL_JSON_FILE
    eval_dir = os.path.dirname(eval_file_path)
    if not os.path.exists(eval_dir):
        try:
            os.makedirs(eval_dir)
            logger.info(f"Created directory for human evaluation JSON file: {eval_dir}")
        except OSError as e:
            logger.error(f"OSError creating directory {eval_dir} for human evaluation JSON: {e}", exc_info=True)
            return None # Cannot proceed if directory can't be made

    # Generate a unique ID for this specific case being added to the evaluation set
    # This 'case_id_for_eval_tool' is for identifying this entry within the list of cases.
    unique_suffix = uuid.uuid4().hex[:8]
    case_id_for_eval_tool = f"evalcase_{patient_id.replace('_', '-')}_{agent_name}_{unique_suffix}"
    
    new_case_data = {
        "case_id_for_eval_tool": case_id_for_eval_tool,
        "llm_interaction_id": llm_interaction_id,
        "patient_id_original": patient_id,
        "patient_context_summary": patient_context_summary,
        "llm_raw_output_with_think": llm_raw_response_with_think, 
        "llm_think_block_extracted": llm_think_block,
        "llm_final_recommendation": llm_full_recommendation,
        "guideline_used": guideline_used,
        "agent_name_source": agent_name,
        "timestamp_generated_for_eval": dt.now().isoformat(), # Use imported datetime
        "evaluation_status": "pending"
    }

    with human_eval_json_lock: # Ensure thread-safe file access
        all_cases = []
        try:
            if os.path.exists(eval_file_path) and os.path.getsize(eval_file_path) > 0:
                with open(eval_file_path, 'r', encoding='utf-8') as f:
                    try:
                        all_cases = json.load(f)
                        if not isinstance(all_cases, list): # Ensure it's a list
                            logger.warning(f"Human evaluation file {eval_file_path} did not contain a list. Initializing as new list.")
                            all_cases = []
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON from {eval_file_path}. Will overwrite with new list.", exc_info=True)
                        all_cases = [] # Start fresh if file is corrupt
            
            all_cases.append(new_case_data) # Add the new case

            with open(eval_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_cases, f, indent=2, ensure_ascii=False)
            
            # logger.info(f"Appended case {case_id_for_eval_tool} for human evaluation to: {eval_file_path} (LLM interaction ID: {llm_interaction_id})")
            return eval_file_path # Or perhaps new_case_data for confirmation
            
        except IOError as e:
            logger.error(f"IOError during human evaluation save to {eval_file_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during human evaluation save to {eval_file_path}: {e}", exc_info=True)
        return None

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
    
    def _get_patient_context_summary_for_eval(self) -> str:
        """
        Constructs a concise patient context summary suitable for human evaluators.
        This should reflect what a human would quickly need to see.
        """
        # Customize this based on what's most important for the human evaluator
        # to see alongside the LLM's recommendation.
        summary_parts = [
            f"Patient ID: {self.patient_data.get('id', 'N/A')}",
            f"Guideline Specified: {self.patient_data.get('guideline', 'N/A')}",
            f"Diagnosis (Code): {self.patient_data.get('main_diagnosis', 'N/A')}",
            f"Diagnosis (Text Snippet): {self.patient_data.get('main_diagnosis_text', 'N/A')[:150]}...", # Snippet
            f"Stage: {self.patient_data.get('ann_arbor_stage', 'N/A')}"
        ]
        # Add other key fields you deem necessary for the evaluator
        if self.patient_data.get('clinical_info'):
            summary_parts.append(f"Clinical Info/Question: {self.patient_data.get('clinical_info')[:100]}...")
        
        return "\n".join(summary_parts)


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
            diag_final_resp, diag_raw_resp, diag_think, diag_id, diag_duration = diag_runner.result
            logger.info(f"LLM think response {diag_think}")            
            self.results["Diagnostik"] = diag_final_resp
            self.results["Diagnostik_raw_response"] = diag_raw_resp
            self.results["Diagnostik_think_block"] = diag_think
            self.results["Diagnostik_interaction_id"] = diag_id
            self.runtimes["Diagnostik_llm_invoke_s"] = diag_duration
            logger.info(f"Diagnostik Agent (ID: {diag_id}, LLM invoke time: {diag_duration:.2f}s) finished.")

            diag_context_for_therapie = diag_final_resp

            # 3. Therapie Agent (Sequential after Diagnostik)
            logger.info("Starting Therapie agent.")
            therapie_runner = AgentRunner(self.therapie_agent, "Therapie", diag_context_for_therapie) 
            therapie_thread = threading.Thread(target=therapie_runner.run, name="TherapieThread")
            threads.append(therapie_thread)
            therapie_thread.start()

        # 4. Wait for all remaining threads
        for t in threads:
            if t.is_alive() and t is not diag_thread : # diag_thread already joined
                t.join()
        logger.info("All agent threads (Studien, Therapie) have completed.")

        # 5. Collect results and errors for Studien
        self.runtimes["Studien"] = studien_runner.runtime
        self.results["Studien"] = studien_runner.result # StudienAgent result is likely a list, not a 5-tuple
        if studien_runner.exception:
            self.errors["Studien"] = studien_runner.exception
            logger.warning("Studien Agent failed.", exc_info=studien_runner.exception)
            if "Studien" not in self.results or self.results["Studien"] is None:
                 self.results["Studien"] = []

        # Collect results for Therapie
        if 'therapie_runner' in locals(): # Ensure therapie_runner was created
            self.runtimes["Therapie"] = therapie_runner.runtime
            if therapie_runner.exception:
                self.errors["Therapie"] = therapie_runner.exception
                logger.error(f"Therapie Agent failed: {therapie_runner.exception}", exc_info=therapie_runner.exception)
            else:
                # therapie_runner.result is (response_text, interaction_id)
                therapy_final_resp, therapy_raw_resp, therapy_think, therapy_id, therapy_duration = therapie_runner.result                
                self.results["Therapie"] = therapy_final_resp
                self.results["Therapie_raw_response"] = therapy_raw_resp
                self.results["Therapie_think_block"] = therapy_think
                self.results["Therapie_interaction_id"] = therapy_id
                self.runtimes["Therapie_llm_invoke_s"] = therapy_duration

                patient_context_summary_for_eval = self._get_patient_context_summary_for_eval()
                guideline_used = self.patient_data.get('guideline', 'Unknown Guideline')
                original_patient_id = self.patient_data.get('id', 'UnknownPatient')

                save_for_human_evaluation(
                    patient_id=original_patient_id,
                    patient_context_summary=patient_context_summary_for_eval,
                    llm_full_recommendation=therapy_final_resp,
                    llm_raw_response_with_think=therapy_raw_resp,
                    llm_think_block=therapy_think,
                    guideline_used=guideline_used,
                    llm_interaction_id=therapy_id,
                    agent_name="TherapieAgentOutput"
                )
                
        elif 'Therapie' in self.errors: # Check if an error was already recorded for Therapie
            logger.warning(f"Therapie Agent failed (error previously recorded), not saving for human evaluation. Error: {self.errors['Therapie']}")
        else: # Fallback if therapie_runner wasn't even created (e.g., Diagnostik failed)
             logger.info("No therapy recommendation produced (likely due to prior agent failure), not saving for human evaluation.")
        logger.info("Agent workflow execution finished.")