# utils_eval.py
import json
import os
import glob
from datetime import datetime

EVAL_CASES_FILE = "/home/pia/projects/llmTubo/tuboEval/data_for_evaluation/all_evaluation_cases.json"
EVAL_RESULTS_DIR = "evaluations"

os.makedirs(DATA_DIR, exist_ok=True)

def load_all_pending_cases() -> list[dict]:
    """Loads all cases from the consolidated JSON file."""
    all_cases = []
    if os.path.exists(EVAL_CASES_FILE) and os.path.getsize(EVAL_CASES_FILE) > 0:
        try:
            with open(EVAL_CASES_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    all_cases = loaded_data
                else:
                    print(f"Warning: {EVAL_CASES_FILE} does not contain a list.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {EVAL_CASES_FILE}. Returning empty list.")
        except Exception as e:
            print(f"Error loading cases from {EVAL_CASES_FILE}: {e}")
    # Optionally filter for cases with "evaluation_status": "pending"
    # pending_cases = [case for case in all_cases if case.get("evaluation_status") == "pending"]
    # return pending_cases
    return all_cases # For now, return all; filtering can be added

def get_case_by_id_for_eval(case_id_for_eval_tool: str) -> dict | None:
    """Finds a specific case by its case_id_for_eval_tool from the list."""
    all_cases = load_all_pending_cases()
    for case in all_cases:
        if case.get("case_id_for_eval_tool") == case_id_for_eval_tool:
            return case
    return None

def save_expert_evaluation(case_id_for_eval_tool: str, evaluation_data: dict, expert_name: str):
    """Saves the expert's evaluation for a specific case."""
    # Save individual evaluation as before, or update status in the main file
    # For simplicity, let's keep saving individual evaluation files.
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use case_id_for_eval_tool for unique eval filenames
    eval_filename = f"{case_id_for_eval_tool}_result_{expert_name.replace(' ', '_')}_{timestamp}.json"
    filepath = os.path.join(EVAL_RESULTS_DIR, eval_filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        # Optional: Update the status of the case in the main all_evaluation_cases.json
        # This requires reading, modifying, and writing the whole file back (thread-safety needed if concurrent)
        # update_case_status_in_main_file(case_id_for_eval_tool, expert_name, "completed")
        
        return True, eval_filename
    except Exception as e:
        print(f"Error saving expert evaluation for {case_id_for_eval_tool}: {e}")
        return False, None

# # --- Example: How you might generate a case file (run this separately to create samples) ---
# def create_sample_case_file(case_id: str, patient_context: str, llm_recommendation: str, guideline_used: str):
#     case_data = {
#         "case_id": case_id,
#         "llm_interaction_id": llm_interaction_id,
#         "patient_context_summary": patient_context, # Keep this concise for display
#         "llm_full_recommendation": llm_recommendation, # The full text including Begründung
#         "guideline_used": guideline_used,
#         "timestamp_generated": datetime.now().isoformat()
#     }
#     filename = f"{case_id}.json"
#     filepath = os.path.join(DATA_DIR, filename)
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(case_data, f, indent=2, ensure_ascii=False)
#     print(f"Created sample case: {filename}")

# if __name__ == '__main__':
#     # Example: Create a couple of sample case files to test the evaluation tool
#     create_sample_case_file(
#         case_id="case_001",
#         patient_context="65 y/o male, DLBCL Stage IIA, IPI 1. Post 4 cycles R-CHOP. PET-CT shows Deauville 2.",
#         llm_recommendation="Therapieempfehlung:\nBeobachtung gemäß ESMO-Leitlinien.\n\nBegründung:\nDer Patient hat nach 4 Zyklen R-CHOP eine komplette metabolische Remission (Deauville 2) erreicht. Gemäß ESMO-Leitlinien für DLBCL mit niedrigem IPI und CR nach Induktion ist eine Beobachtung der Standardansatz. Weitere Therapie ist nicht indiziert.",
#         guideline_used="ESMO DLBCL 2023"
#     )
#     create_sample_case_file(
#         case_id="case_002",
#         patient_context="72 y/o female, Follicular Lymphoma Grade 3A, Stage III. Symptomatic. FLIPI 3.",
#         llm_recommendation="Therapieempfehlung:\nInitiierung einer Erstlinientherapie mit R-Bendamustin.\n\nBegründung:\nDie Patientin ist symptomatisch mit einem FLIPI-Score von 3, was eine Therapieindikation darstellt. R-Bendamustin ist eine Standardoption für fitte, ältere Patienten mit Follicular Lymphoma gemäß Onkopedia und internationalen Leitlinien und bietet ein gutes Nutzen-Risiko-Profil.",
#         guideline_used="Onkopedia Follicular Lymphoma"
#     )