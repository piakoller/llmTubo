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
    # These imports might be needed if type hints in AgentWorkflowManager are strict
    # from agents.diagnostik_agent import DiagnostikAgent
    # from agents.studien_agent import StudienAgent
    # from agents.therapie_agent import TherapieAgent
    # from agents.report_agent import ReportAgent
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure this script is run from the project root or that 'llmRecom' is in your PYTHONPATH.")
    exit(1)

# --- Configuration for this Batch Script ---
DEFAULT_LLM_MODEL_BATCH = "qwen3:32b"
DEFAULT_GUIDELINE_PROVIDER_BATCH = "ESMO"
DEFAULT_STUDY_LOCATION_BATCH = "Bern, Switzerland"
BATCH_RESULTS_OUTPUT_FILE = "batch_processing_results.json"

setup_logging()
logger = logging.getLogger("batch_processor")

def process_study_output(studien_results: list | None) -> list | None:
    """
    Filters the studien_output to include only 'title' and 'nct_id' for each study.
    """
    if not studien_results or not isinstance(studien_results, list):
        return None # Or return [] if you prefer an empty list for no results
    
    processed_studies = []
    for study in studien_results:
        if isinstance(study, dict): # Ensure each item is a dictionary
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
    output_file: str = BATCH_RESULTS_OUTPUT_FILE,
    fragestellung_modified: bool = False # New parameter
):
    logger.info("Starting batch processing of patients...")

    original_config_llm_model = config.LLM_MODEL
    effective_llm_model = original_config_llm_model # Start with current config

    if llm_model_override:
        effective_llm_model = llm_model_override
        config.LLM_MODEL = effective_llm_model # Temporarily override global config
        logger.info(f"Overriding LLM model for batch processing to: {effective_llm_model}")
    elif DEFAULT_LLM_MODEL_BATCH != config.LLM_MODEL: # Only override if default script batch model is different
        effective_llm_model = DEFAULT_LLM_MODEL_BATCH
        config.LLM_MODEL = effective_llm_model
        logger.info(f"Using default batch LLM model: {effective_llm_model}")
    else:
        logger.info(f"Using LLM model from config.py: {effective_llm_model}")


    effective_guideline = guideline_override if guideline_override else DEFAULT_GUIDELINE_PROVIDER_BATCH
    effective_study_location = study_location_override if study_location_override else DEFAULT_STUDY_LOCATION_BATCH

    df_patients = load_patient_data(config.TUBO_EXCEL_FILE_PATH)
    if df_patients is None or df_patients.empty:
        logger.error("No patient data loaded. Exiting batch processing.")
        if config.LLM_MODEL != original_config_llm_model: config.LLM_MODEL = original_config_llm_model # Restore
        return

    all_patient_results = []
    patient_ids = [pid for pid in df_patients["ID"].unique() if pid is not None and str(pid).strip() != ""]
    total_patients = len(patient_ids)
    logger.info(f"Found {total_patients} unique patient IDs to process.")

    for i, patient_id in enumerate(patient_ids):
        logger.info(f"Processing patient {i+1}/{total_patients}: {patient_id}")
        start_time_patient = time.perf_counter()
        patient_result_entry = {} # Initialize here for the finally block

        try:
            patient_row = df_patients[df_patients["ID"] == patient_id].iloc[0]
            
            current_patient_data_for_agents = {
                "id": str(patient_id),
                "main_diagnosis_text": str(patient_row.get("main_diagnosis_text", "")),
                "secondary_diagnoses": str(patient_row.get("secondary_diagnoses", "")),
                "clinical_info": str(patient_row.get("clinical_info", "")),
                "pet_ct_report": str(patient_row.get("pet_ct_report", "")),
                "presentation_type": str(patient_row.get("presentation_type", "")),
                "main_diagnosis": str(patient_row.get("main_diagnosis", "")),
                "ann_arbor_stage": str(patient_row.get("ann_arbor_stage", "")),
                "accompanying_symptoms": str(patient_row.get("accompanying_symptoms", "")),
                "prognosis_score": str(patient_row.get("prognosis_score", "")),
                "guideline": effective_guideline,
                "location": effective_study_location
            }

            manager = AgentWorkflowManager(current_patient_data_for_agents)
            manager.run_workflow()

            patient_result_entry = {
                "patient_id_original": str(patient_id),
                "timestamp_processed": datetime.now().isoformat(),
                "llm_model_used": effective_llm_model, # Use the model active for this run
                "fragestellung_modified": fragestellung_modified, # Add the new attribute
                "guideline_used": effective_guideline,
                "study_location_input": effective_study_location,
                "diagnostik_output": manager.results.get("Diagnostik"),
                "diagnostik_interaction_id": manager.results.get("Diagnostik_interaction_id"),
                "studien_output": process_study_output(manager.results.get("Studien")), # Processed studies
                "therapie_output": manager.results.get("Therapie"),
                "therapie_interaction_id": manager.results.get("Therapie_interaction_id"),
                "errors": manager.errors if manager.errors else None,
                "runtimes": manager.runtimes,
                "patient_context_summary_for_eval": manager._get_patient_context_summary_for_eval() if hasattr(manager, '_get_patient_context_summary_for_eval') else "N/A"
            }
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR processing patient {patient_id}: {e}", exc_info=True)
            # Ensure basic info is logged even on critical failure before manager.run_workflow()
            patient_result_entry = {
                "patient_id_original": str(patient_id),
                "timestamp_processed": datetime.now().isoformat(),
                "llm_model_used": effective_llm_model,
                "fragestellung_modified": fragestellung_modified,
                "guideline_used": effective_guideline,
                "status": "FAILED_CRITICAL_PRE_WORKFLOW" if not patient_result_entry else "FAILED_PROCESSING",
                "error_message": str(e)
            }
        finally:
            if patient_result_entry: # Ensure it was initialized
                all_patient_results.append(patient_result_entry)
            patient_processing_time = time.perf_counter() - start_time_patient
            logger.info(f"Finished processing patient {patient_id} in {patient_processing_time:.2f}s (Status: {'OK' if patient_result_entry.get('errors') is None and 'status' not in patient_result_entry else 'ERROR'}).")


    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_patient_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch processing complete. All results saved to: {output_file}")
    except Exception as e: # Broader catch for file writing
        logger.error(f"Failed to write batch results to {output_file}: {e}", exc_info=True)

    if config.LLM_MODEL != original_config_llm_model: # Restore original config if changed
        config.LLM_MODEL = original_config_llm_model
        logger.info(f"Restored LLM model in config to: {config.LLM_MODEL}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch process patient data for therapy recommendations.")
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help=f"Override the LLM model to use (e.g., 'qwen3:7b'). Default uses script's default ({DEFAULT_LLM_MODEL_BATCH}) or config.py."
    )
    parser.add_argument(
        "--guideline",
        type=str,
        default=DEFAULT_GUIDELINE_PROVIDER_BATCH,
        help=f"Guideline provider. Default: {DEFAULT_GUIDELINE_PROVIDER_BATCH}"
    )
    parser.add_argument(
        "--location",
        type=str,
        default=DEFAULT_STUDY_LOCATION_BATCH,
        help=f"Study search location. Default: {DEFAULT_STUDY_LOCATION_BATCH}"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=BATCH_RESULTS_OUTPUT_FILE,
        help=f"Path to save the consolidated JSON results. Default: {BATCH_RESULTS_OUTPUT_FILE}"
    )
    parser.add_argument(
        "--fragestellung_modified",
        action="store_true", # Makes it a boolean flag, False by default
        help="Set to true if context were specially modified for this LLM run (for tracking experiments)."
    )

    args = parser.parse_args()

    # Determine effective LLM model for logging start message
    llm_to_log = args.llm_model if args.llm_model else (DEFAULT_LLM_MODEL_BATCH if DEFAULT_LLM_MODEL_BATCH != config.LLM_MODEL else config.LLM_MODEL)
    logger.info(f"Running batch process with LLM: {llm_to_log}, Guideline: {args.guideline}, Location: {args.location}, Fragestellung Modified: {args.fragestellung_modified}")

    run_batch_processing(
        llm_model_override=args.llm_model,
        guideline_override=args.guideline,
        study_location_override=args.location,
        output_file=args.output_file,
        fragestellung_modified=args.fragestellung_modified
    )