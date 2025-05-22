# utils_eval.py
import json
import os
import pandas as pd
from datetime import datetime
import logging
import glob

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# AGGREGATED_EXCEL_INPUT_FILE = "/home/pia/projects/llmTubo/tuboEval/expert_review_sheets/expert_evaluation_sheet_v2.xlsx"
# EVALUATION_RESULTS_SAVE_DIR = "/home/pia/projects/llmTubo/tuboEval/evaluations_completed_comparative/" # New dir for these specific evals
script_dir = os.path.dirname(os.path.abspath(__file__))

AGGREGATED_EXCEL_INPUT_FILE = os.path.join(script_dir, "expert_review_sheets", "expert_evaluation_sheet_v2.xlsx")
EVALUATION_RESULTS_SAVE_DIR = os.path.join(script_dir, "evaluations_completed_comparative")


try:
    os.makedirs(EVALUATION_RESULTS_SAVE_DIR, exist_ok=True)
except OSError as e:
    logger.error(f"Could not create directory {EVALUATION_RESULTS_SAVE_DIR}: {e}")

_df_all_cases_for_eval_cache: pd.DataFrame | None = None

def load_cases_from_aggregated_excel() -> pd.DataFrame | None:
    global _df_all_cases_for_eval_cache
    if _df_all_cases_for_eval_cache is not None:
        return _df_all_cases_for_eval_cache
    
    if not os.path.exists(AGGREGATED_EXCEL_INPUT_FILE):
        logger.error(f"Aggregated Excel input file not found: {AGGREGATED_EXCEL_INPUT_FILE}")
        return None
    try:
        logger.info(f"Loading cases for evaluation from: {AGGREGATED_EXCEL_INPUT_FILE}")
        df = pd.read_excel(AGGREGATED_EXCEL_INPUT_FILE, sheet_name="Expert Evaluation", dtype={"Patient ID": str})
        df = df.fillna('') 
        if "Patient ID" not in df.columns:
            logger.error("'Patient ID' column not found.")
            return None
        _df_all_cases_for_eval_cache = df
        logger.info(f"Successfully loaded {len(df)} patient cases.")
        return df
    except Exception as e:
        logger.error(f"Error loading cases from Excel {AGGREGATED_EXCEL_INPUT_FILE}: {e}", exc_info=True)
        return None

def get_patient_ids_for_selection() -> list[str]:
    df = load_cases_from_aggregated_excel()
    if df is not None and "Patient ID" in df.columns:
        return sorted(df["Patient ID"].unique().tolist())
    return []

def get_case_data_for_patient(patient_id: str) -> pd.Series | None:
    df = load_cases_from_aggregated_excel()
    if df is not None and "Patient ID" in df.columns:
        patient_rows = df[df["Patient ID"].astype(str) == str(patient_id)]
        if not patient_rows.empty:
            return patient_rows.iloc[0]
    return None

def get_available_llm_models_for_patient(case_data_series: pd.Series | None) -> list[str]:
    """
    Parses column names from the case_data_series to find unique LLM model names.
    Assumes column prefixes like "MultiAgent_Llama3_ModFalse - Final Recommendation".
    """
    if case_data_series is None:
        return []
    
    llm_models = set()
    # This parsing logic needs to be robust based on your aggregate_for_expert_review.py output columns
    for col_name in case_data_series.index:
        if isinstance(col_name, str) and (" - Final Recommendation" in col_name or " - Think Block" in col_name):
            prefix = col_name.split(" - ")[0] # e.g., "MultiAgent_Llama3_ModFalse"
            parts = prefix.split('_')
            # Example: MultiAgent_Llama3_ModFalse -> Llama3
            # Example: SinglePrompt_Qwen3_7B_ModTrue -> Qwen3_7B
            potential_llm_name = ""
            script_type_found = False
            if parts[0].lower() in ["multiagent", "singleprompt"]:
                script_type_found = True
                start_index = 1
            else: # If no script type prefix, assume first part could be LLM
                start_index = 0

            mod_indicator_index = -1
            for i in range(len(parts) -1, start_index -1, -1): # Search backwards for ModTrue/False
                if parts[i] in ["ModTrue", "ModFalse"]:
                    mod_indicator_index = i
                    break
            
            if mod_indicator_index != -1 and mod_indicator_index > start_index :
                potential_llm_name = "_".join(parts[start_index:mod_indicator_index])
            elif mod_indicator_index == -1 and len(parts) > start_index: # No Mod indicator
                potential_llm_name = "_".join(parts[start_index:])
            
            if potential_llm_name:
                 llm_models.add(potential_llm_name)

    return sorted(list(llm_models)) if llm_models else ["UnknownLLM"]


def save_comparative_evaluation(patient_id: str, llm_model_evaluated: str, evaluation_data: dict, expert_name: str) -> tuple[bool, str | None]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_patient_id = patient_id.replace('/', '_').replace('\\', '_')
    safe_expert_name = expert_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_llm_model = llm_model_evaluated.replace(':', '_').replace('/', '_')

    eval_filename = f"eval_{safe_patient_id}_llm_{safe_llm_model}_{safe_expert_name}_{timestamp}.json"
    filepath = os.path.join(EVALUATION_RESULTS_SAVE_DIR, eval_filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Comparative evaluation for P:{patient_id}, LLM:{llm_model_evaluated} by {expert_name} saved to: {filepath}")
        return True, eval_filename
    except Exception as e:
        logger.error(f"Error saving comparative evaluation for P:{patient_id}, LLM:{llm_model_evaluated}: {e}", exc_info=True)
        return False, None

def check_if_evaluated(patient_id: str, llm_model: str, expert_name: str) -> bool:
    """Checks if an evaluation file already exists for this patient-LLM-expert combo."""
    safe_patient_id = patient_id.replace('/', '_').replace('\\', '_')
    safe_expert_name = expert_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_llm_model = llm_model.replace(':', '_').replace('/', '_')
    
    # Search for files matching the pattern
    # This is a simple check; more robust would be to store a status in a DB or the Excel itself.
    pattern = f"eval_{safe_patient_id}_llm_{safe_llm_model}_{safe_expert_name}_*.json"
    matching_files = glob.glob(os.path.join(EVALUATION_RESULTS_SAVE_DIR, pattern))
    return bool(matching_files)