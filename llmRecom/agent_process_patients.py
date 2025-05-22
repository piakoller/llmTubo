import logging
import json
import os
import time
from datetime import datetime

# --- Project Imports ---
try:
    import config
    from utils.logging_setup import setup_logging
    from data_loader import load_patient_data
    from core.agent_manager import AgentWorkflowManager
    # ... (other agent imports if needed for type hinting) ...
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure this script is run from the project root or that 'llmRecom' is in your PYTHONPATH.")
    exit(1)

# --- Configuration for this Batch Script ---
# DEFAULT_LLM_MODEL_BATCH = "qwen3:32b"
DEFAULT_LLM_MODEL_BATCH = "llama3"
DEFAULT_GUIDELINE_PROVIDER_BATCH = "ESMO"
DEFAULT_STUDY_LOCATION_BATCH = "Bern, Switzerland"

EVAL_AGENT_DATA_DIR = "/home/pia/projects/llmTubo/tuboEval/data_for_evaluation/agent/test"

setup_logging()
logger = logging.getLogger("agent_batch_processor")

def get_batch_results_filename(llm_model: str, clinical_info_modified: bool) -> str: # Renamed parameter
    """
    Generates an output filename (without directory) that includes the LLM model
    and clinical_info_modified flag.
    """
    llm_safe = llm_model.replace(":", "_").replace("/", "_")
    return f"agent_{llm_safe}_clinical_info_modified_{clinical_info_modified}.json" # Updated filename pattern

def process_study_output(studien_results: list | None) -> list | None:
    if not studien_results or not isinstance(studien_results, list):
        return None
    processed_studies = []
    for study in studien_results:
        if isinstance(study, dict):
            processed_studies.append({
                "title": study.get("title", "N/A"),
                "nct_id": study.get("nct_id", "N/A")
            })
        else:
            logger.warning(f"Found non-dictionary item in studien_results: {study}")
    return processed_studies if processed_studies else None

def run_batch_processing(
    llm_model_override: str | None = None,
    guideline_override: str | None = None,
    study_location_override: str | None = None,
    output_filepath_override: str | None = None,
    clinical_info_modified_arg: bool = False, # Renamed from fragestellung_modified_arg
    patient_data_filepath_override: str | None = None
):
    # --- Determine effective clinical_info_modified state ---
    effective_clinical_info_modified = clinical_info_modified_arg # Start with arg
    
    # --- Determine effective patient data file path ---
    effective_patient_data_file = config.TUBO_EXCEL_FILE_PATH
    original_config_patient_file = config.TUBO_EXCEL_FILE_PATH

    if patient_data_filepath_override:
        effective_patient_data_file = patient_data_filepath_override
        logger.info(f"Using patient data file specified by argument: {effective_patient_data_file}")
        # If a patient data file is overridden, consider clinical_info to be 'modified' for this run
        if not effective_clinical_info_modified: # Only log if it wasn't already set by arg
            logger.info(f"Patient data file was overridden ('{os.path.basename(effective_patient_data_file)}'), "
                        "setting clinical_info_modified to True for this run.")
        effective_clinical_info_modified = True # Force to True if a specific patient data file is given
    
    if effective_patient_data_file != config.TUBO_EXCEL_FILE_PATH:
        config.TUBO_EXCEL_FILE_PATH = effective_patient_data_file
        logger.info(f"Temporarily set patient data file path in config to: {config.TUBO_EXCEL_FILE_PATH}")

    # --- LLM Model Configuration ---
    original_config_llm_model = config.LLM_MODEL
    effective_llm_model = original_config_llm_model

    if llm_model_override:
        effective_llm_model = llm_model_override
        config.LLM_MODEL = effective_llm_model
        logger.info(f"Overriding LLM model for batch processing to: {effective_llm_model}")
    elif DEFAULT_LLM_MODEL_BATCH != config.LLM_MODEL:
        effective_llm_model = DEFAULT_LLM_MODEL_BATCH
        config.LLM_MODEL = effective_llm_model
        logger.info(f"Using default batch LLM model: {effective_llm_model}")
    else:
        logger.info(f"Using LLM model from config.py: {effective_llm_model}")

    effective_guideline = guideline_override if guideline_override else DEFAULT_GUIDELINE_PROVIDER_BATCH
    effective_study_location = study_location_override if study_location_override else DEFAULT_STUDY_LOCATION_BATCH

    final_output_file: str
    if output_filepath_override:
        final_output_file = output_filepath_override
        logger.info(f"Using provided output file path: {final_output_file}")
    else:
        generated_filename = get_batch_results_filename(effective_llm_model, effective_clinical_info_modified)
        final_output_file = os.path.join(EVAL_AGENT_DATA_DIR, generated_filename)
        logger.info(f"Generated output file path: {final_output_file}")

    try:
        os.makedirs(EVAL_AGENT_DATA_DIR, exist_ok=True)
        logger.info(f"Ensured output directory exists: {EVAL_AGENT_DATA_DIR}")
    except OSError as e:
        logger.error(f"Could not create output directory {EVAL_AGENT_DATA_DIR}: {e}. Exiting.")
        if config.LLM_MODEL != original_config_llm_model: config.LLM_MODEL = original_config_llm_model
        if config.TUBO_EXCEL_FILE_PATH != original_config_patient_file: config.TUBO_EXCEL_FILE_PATH = original_config_patient_file
        return

    df_patients = load_patient_data(effective_patient_data_file)
    if df_patients is None or df_patients.empty:
        logger.error(f"No patient data loaded from {effective_patient_data_file}. Exiting batch processing.")
        if config.LLM_MODEL != original_config_llm_model: config.LLM_MODEL = original_config_llm_model
        if config.TUBO_EXCEL_FILE_PATH != original_config_patient_file: config.TUBO_EXCEL_FILE_PATH = original_config_patient_file
        return

    all_patient_results = []
    patient_ids = [pid for pid in df_patients["ID"].unique() if pid is not None and str(pid).strip() != ""]
    total_patients = len(patient_ids)
    logger.info(f"Found {total_patients} unique patient IDs to process from {os.path.basename(effective_patient_data_file)}.")

    for i, patient_id in enumerate(patient_ids):
        logger.info(f"Processing patient {i+1}/{total_patients}: {patient_id}")
        start_time_patient = time.perf_counter()
        patient_result_entry = {}

        try:
            patient_row = df_patients[df_patients["ID"] == patient_id].iloc[0]
            
            current_patient_data_for_agents = {
                "id": str(patient_id),
                "main_diagnosis_text": str(patient_row.get("main_diagnosis_text", "")),
                "secondary_diagnoses": str(patient_row.get("secondary_diagnoses", "")),
                "clinical_info": str(patient_row.get("clinical_info", "")), # This is the "Fragestellung"
                "pet_ct_report": str(patient_row.get("pet_ct_report", "")),
                "presentation_type": str(patient_row.get("presentation_type", "")),
                "main_diagnosis": str(patient_row.get("main_diagnosis", "")),
                "ann_arbor_stage": str(patient_row.get("ann_arbor_stage", "")),
                "accompanying_symptoms": str(patient_row.get("accompanying_symptoms", "")),
                "prognosis_score": str(patient_row.get("prognosis_score", "")),
                "guideline": effective_guideline,
                "location": effective_study_location
            }
            
            if effective_clinical_info_modified:
                logger.info(f"Clinical_info_modified is TRUE for patient {patient_id}. "
                            "The 'clinical_info' field itself is taken from the (potentially modified) patient data file. "
                            "This flag primarily affects output naming and metadata.")
                # If you need to *actually alter* the `clinical_info` field based on this flag,
                # beyond what's in the loaded data file, you would do it here.
                # For example:
                # current_patient_data_for_agents["clinical_info"] = "MODIFIED CONTEXT: " + current_patient_data_for_agents["clinical_info"]

            manager = AgentWorkflowManager(current_patient_data_for_agents)
            manager.run_workflow()

            def get_llm_duration(agent_key_prefix: str) -> float | None:
                return manager.runtimes.get(f"{agent_key_prefix}_llm_generation_s")

            patient_result_entry = {
                "patient_id_original": str(patient_id),
                "patient_data_source_file": os.path.basename(effective_patient_data_file),
                "timestamp_processed": datetime.now().isoformat(),
                "llm_model_used": effective_llm_model,
                "clinical_info_modified": effective_clinical_info_modified,
                "guideline_used": effective_guideline,
                "study_location_input": effective_study_location,
                
                # Diagnostik Agent Results
                "diagnostik_output_final": manager.results.get("Diagnostik"),
                "diagnostik_raw_response": manager.results.get("Diagnostik_raw_response"),   
                "diagnostik_think_block": manager.results.get("Diagnostik_think_block"),     
                "diagnostik_interaction_id": manager.results.get("Diagnostik_interaction_id"),
                
                # Studien Agent Results
                "studien_output": process_study_output(manager.results.get("Studien")),
                
                # Therapie Agent Results
                "therapie_output_final": manager.results.get("Therapie"),
                "therapie_raw_response": manager.results.get("Therapie_raw_response"),     
                "therapie_think_block": manager.results.get("Therapie_think_block"),       
                "therapie_interaction_id": manager.results.get("Therapie_interaction_id"),
                
                # Errors and Overall Runtimes
                "errors": manager.errors if manager.errors else None,
                "runtimes_overall_agents": {
                    # Store only overall agent runtimes here, LLM specific times are now separate
                    k: v for k, v in manager.runtimes.items() 
                    if not (k.endswith("_llm_invoke_s") or k.endswith("_llm_generation_s")) 
                },
                "patient_context_summary_for_eval": manager._get_patient_context_summary_for_eval() if hasattr(manager, '_get_patient_context_summary_for_eval') else "N/A"
            }
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing patient {patient_id}: {e}", exc_info=True)
            # Ensure all relevant fields are present in the error entry too
            patient_result_entry = {
                "patient_id_original": str(patient_id),
                "patient_data_source_file": os.path.basename(effective_patient_data_file),
                "timestamp_processed": datetime.now().isoformat(),
                "llm_model_used": effective_llm_model,
                "clinical_info_modified": effective_clinical_info_modified,
                "guideline_used": effective_guideline,
                "study_location_input": effective_study_location, # Added for consistency in error cases
                "status": "FAILED_CRITICAL_PROCESSING",
                "error_message": str(e)
            }
        finally:
            if patient_result_entry: # Ensure it's initialized
                all_patient_results.append(patient_result_entry)
            patient_processing_time = time.perf_counter() - start_time_patient
            status_ok = patient_result_entry.get('errors') is None and 'status' not in patient_result_entry
            logger.info(f"Finished processing patient {patient_id} in {patient_processing_time:.2f}s (Status: {'OK' if status_ok else 'ERROR'}).")
    try:
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_patient_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch processing complete. All results saved to: {final_output_file}")
    except Exception as e:
        logger.error(f"Failed to write batch results to {final_output_file}: {e}", exc_info=True)

    if config.LLM_MODEL != original_config_llm_model:
        config.LLM_MODEL = original_config_llm_model
        logger.info(f"Restored LLM model in config to: {config.LLM_MODEL}")
    if config.TUBO_EXCEL_FILE_PATH != original_config_patient_file:
        config.TUBO_EXCEL_FILE_PATH = original_config_patient_file
        logger.info(f"Restored patient data file path in config to: {config.TUBO_EXCEL_FILE_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch process patient data for therapy recommendations.")
    parser.add_argument(
        "--patient_data_file", type=str, default=None,
        help="Optional: Path to the patient data Excel file to use. Overrides config.TUBO_EXCEL_FILE_PATH for this run. If provided, 'clinical_info_modified' will be set to True."
    )
    parser.add_argument(
        "--llm_model", type=str, default=None,
        help=f"Override LLM model. Default: script's default ({DEFAULT_LLM_MODEL_BATCH}) or config.py."
    )
    parser.add_argument(
        "--guideline", type=str, default=DEFAULT_GUIDELINE_PROVIDER_BATCH,
        help=f"Guideline provider. Default: {DEFAULT_GUIDELINE_PROVIDER_BATCH}"
    )
    parser.add_argument(
        "--location", type=str, default=DEFAULT_STUDY_LOCATION_BATCH,
        help=f"Study search location. Default: {DEFAULT_STUDY_LOCATION_BATCH}"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Optional: Full path to save consolidated JSON results. If not set, generated into EVAL_AGENT_DATA_DIR."
    )
    parser.add_argument(
        "--clinical_info_modified", action="store_true", # Renamed argument
        help="Set if 'clinical_info' (Fragestellung) or other context was specially modified. This will be automatically set to True if --patient_data_file is used."
    )

    args = parser.parse_args()

    # Determine llm_to_log more safely
    if args.llm_model:
        llm_to_log = args.llm_model
    elif DEFAULT_LLM_MODEL_BATCH != getattr(config, 'LLM_MODEL', ''):
        llm_to_log = DEFAULT_LLM_MODEL_BATCH
    else:
        llm_to_log = getattr(config, 'LLM_MODEL', 'N/A (config missing LLM_MODEL)')

    # Determine initial clinical_info_modified state
    clinical_info_modified_for_run = args.clinical_info_modified # From command-line flag
    
    # If --patient_data_file is used, force clinical_info_modified to True
    if args.patient_data_file:
        if not clinical_info_modified_for_run: # Log if it's being changed by the file override
            logger.info(f"Patient data file ('{os.path.basename(args.patient_data_file)}') was specified, "
                        f"forcing 'clinical_info_modified' to True for this run.")
        clinical_info_modified_for_run = True
    
    patient_data_file_to_log = args.patient_data_file if args.patient_data_file else getattr(config, 'TUBO_EXCEL_FILE_PATH', 'N/A')

    logger.info(f"Running batch process with Patient Data File: {patient_data_file_to_log}, LLM: {llm_to_log}, "
                f"Guideline: {args.guideline}, Location: {args.location}, "
                f"Clinical Info Modified: {clinical_info_modified_for_run}")

    run_batch_processing(
        llm_model_override=args.llm_model,
        guideline_override=args.guideline,
        study_location_override=args.location,
        output_filepath_override=args.output_file,
        clinical_info_modified_arg=clinical_info_modified_for_run,
        patient_data_filepath_override=args.patient_data_file
    )