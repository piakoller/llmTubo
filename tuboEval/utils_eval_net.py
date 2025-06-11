# NET
# utils_eval_net.py
import json
import os
import pandas as pd
from datetime import datetime
import logging
import glob
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["llmTubo_eval_db"]
collection = db["expert_evaluations"]

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

script_dir = os.path.dirname(os.path.abspath(__file__))

#NET
# updated Patient Data
PATIENT_DATA = os.path.join(script_dir, "NET Tubo v2.xlsx")
AGGREGATED_LLM_DATA_JSON_FILE = os.path.join(script_dir, "expert_review_sheets", "net_expert_evaluation_data.json")
EVALUATION_RESULTS_SAVE_DIR = os.path.join(script_dir, "evaluations_completed_comparative")

try:
    os.makedirs(EVALUATION_RESULTS_SAVE_DIR, exist_ok=True)
except OSError as e:
    logger.error(f"Could not create directory {EVALUATION_RESULTS_SAVE_DIR}: {e}")

# --- Caching for loaded data ---
_patient_data_excel_cache: pd.DataFrame | None = None
_aggregated_llm_data_json_cache: dict | None = None

def load_aggregated_llm_data_from_json() -> dict | None:
    """
    Loads the aggregated LLM recommendation data from the JSON file.
    Caches the data to avoid re-reading for every request.
    """
    global _aggregated_llm_data_json_cache
    if _aggregated_llm_data_json_cache is not None:
        return _aggregated_llm_data_json_cache
    
    if not os.path.exists(AGGREGATED_LLM_DATA_JSON_FILE):
        logger.error(f"Aggregated LLM results JSON input file not found: {AGGREGATED_LLM_DATA_JSON_FILE}")
        return None
    
    try:
        logger.info(f"Loading aggregated LLM data from: {AGGREGATED_LLM_DATA_JSON_FILE}")
        with open(AGGREGATED_LLM_DATA_JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            logger.error(f"Expected a dictionary in {AGGREGATED_LLM_DATA_JSON_FILE}, but got {type(data)}. Cannot process.")
            return None
        
        _aggregated_llm_data_json_cache = data
        logger.info(f"Successfully loaded {len(data)} patient LLM results from JSON.")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {AGGREGATED_LLM_DATA_JSON_FILE}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error loading aggregated LLM data from JSON {AGGREGATED_LLM_DATA_JSON_FILE}: {e}", exc_info=True)
        return None

def get_patient_ids_for_selection() -> list[str]:
    """
    Retrieves a sorted list of all patient IDs available in the aggregated JSON data.
    """
    data = load_aggregated_llm_data_from_json()
    if data: # Check if data is not None and not empty
        return sorted(list(data.keys()))
    logger.warning("No patient IDs found in the aggregated LLM JSON data. Returning an empty list.")
    return []

def get_case_data_for_patient(patient_id: str) -> dict | None: # Changed return type to dict | None
    """
    Retrieves all aggregated data for a specific patient ID from the JSON.
    """
    data = load_aggregated_llm_data_from_json()
    if data and str(patient_id) in data:
        return data[str(patient_id)]
    logger.warning(f"Patient ID '{patient_id}' not found in aggregated JSON data.")
    return None

def parse_full_prefix_to_llm_and_variant(full_prefix: str) -> tuple[str | None, str | None, bool | None]:
    """
    Parses a full variant prefix (e.g., 'SinglePrompt_gemma3_27b_ModTrue')
    into its components: script_type, base_llm_name, and is_modified.
    """
    script_type = None
    base_llm_name = None
    is_modified = None

    parts = full_prefix.split('_')
    
    # Determine script_type
    if parts and parts[0].lower() in ["multiagent", "singleprompt"]:
        script_type = parts[0]
        temp_parts = parts[1:] # Remaining parts after script_type
    else:
        # This case should ideally not happen if prefixes are always well-formed
        logger.warning(f"Prefix '{full_prefix}' does not start with MultiAgent or SinglePrompt.")
        script_type = "UnknownScript"
        temp_parts = parts

    # Determine is_modified and extract base_llm_name
    if temp_parts and temp_parts[-1].lower() == "modtrue":
        is_modified = True
        llm_name_parts = temp_parts[:-1] # Parts without ModTrue
    elif temp_parts and temp_parts[-1].lower() == "modfalse":
        is_modified = False
        llm_name_parts = temp_parts[:-1] # Parts without ModFalse
    else:
        # Default to False if no Mod suffix, treat all remaining parts as base_llm_name
        is_modified = False
        llm_name_parts = temp_parts

    base_llm_name = "_".join(llm_name_parts)
    
    if not base_llm_name:
        logger.warning(f"Could not extract base LLM name from prefix: '{full_prefix}'. Using full prefix as fallback.")
        base_llm_name = full_prefix # Fallback in case parsing fails
        
    return script_type, base_llm_name, is_modified

def get_available_llm_models_for_patient(case_data: dict | pd.Series | None) -> list[str]:
    """
    Retrieves a sorted list of unique base LLM model names available for a patient.
    Accepts a dictionary (from get_case_data_for_patient) or a pd.Series (if converted).
    """
    if case_data is None:
        return []
    
    llm_base_models = set()
    
    # Access recommendation_variants directly from the dict, or from series if converted
    if isinstance(case_data, pd.Series):
        recommendation_variants_dict = case_data.get("recommendation_variants", {})
    elif isinstance(case_data, dict):
        recommendation_variants_dict = case_data.get("recommendation_variants", {})
    else:
        logger.warning(f"Unexpected type for case_data: {type(case_data)}. Expected dict or pd.Series.")
        return []

    if not isinstance(recommendation_variants_dict, dict):
        logger.error(f"Expected 'recommendation_variants' to be a dictionary, but got {type(recommendation_variants_dict)}. Cannot extract LLM models.")
        return []

    for full_prefix in recommendation_variants_dict.keys():
        _script_type, base_llm_name, _is_modified = parse_full_prefix_to_llm_and_variant(full_prefix)
        if base_llm_name:
            llm_base_models.add(base_llm_name)

    return sorted(list(llm_base_models)) if llm_base_models else []

def save_comparative_evaluation(patient_id: str, llm_model_evaluated: str, evaluation_data: dict, expert_name: str) -> tuple[bool, str | None]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_patient_id = patient_id.replace('/', '_').replace('\\', '_')
    safe_expert_name = expert_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_llm_model = llm_model_evaluated.replace(':', '_').replace('/', '_').replace('.', '_') # Added ._ for dot in model names

    eval_filename = f"eval_{safe_patient_id}_llm_{safe_llm_model}_{safe_expert_name}_{timestamp}.json"
    filepath = os.path.join(EVALUATION_RESULTS_SAVE_DIR, eval_filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Comparative evaluation saved locally to: {filepath}")
        
        # Only attempt MongoDB save if collection is available (connection successful)
        if client and db and collection:
            mongo_document = {
                "patient_id": patient_id,
                "llm_model": llm_model_evaluated,
                "expert_name": expert_name,
                "timestamp": timestamp,
                "evaluation_data": evaluation_data,
                "created_at": datetime.utcnow() # Add creation timestamp for MongoDB
            }
            collection.insert_one(mongo_document)
            logger.info(f"Comparative evaluation also saved to MongoDB (Cloud).")
        else:
            logger.warning("MongoDB connection not established; skipping cloud save.")


        return True, eval_filename

    except Exception as e:
        logger.error(f"Error saving comparative evaluation: {e}", exc_info=True)
        return False, None

def check_if_evaluated(patient_id: str, llm_model: str, expert_name: str) -> bool:
    """Checks if an evaluation file already exists for this patient-LLM-expert combo."""
    safe_patient_id = patient_id.replace('/', '_').replace('\\', '_')
    safe_expert_name = expert_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_llm_model = llm_model.replace(':', '_').replace('/', '_').replace('.', '_') # Added ._ for dot in model names
    
    # Check local files
    pattern = f"eval_{safe_patient_id}_llm_{safe_llm_model}_{safe_expert_name}_*.json"
    matching_files = glob.glob(os.path.join(EVALUATION_RESULTS_SAVE_DIR, pattern))
    if matching_files:
        logger.info(f"Found existing local evaluation for {patient_id} - {llm_model} - {expert_name}.")
        return True

    # Check MongoDB (if connected)
    if collection:
        try:
            mongo_query = {
                "patient_id": patient_id,
                "llm_model": llm_model,
                "expert_name": expert_name
            }
            if collection.find_one(mongo_query):
                logger.info(f"Found existing MongoDB evaluation for {patient_id} - {llm_model} - {expert_name}.")
                return True
        except Exception as e:
            logger.error(f"Error checking MongoDB for existing evaluation: {e}", exc_info=True)
    
    return False