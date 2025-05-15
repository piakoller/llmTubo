# utils_eval.py
import json
import os
import glob
from datetime import datetime

DATA_DIR = "data_for_evaluation"
EVAL_DIR = "evaluations"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def get_cases_for_evaluation():
    """Lists available .json case files from the data directory."""
    return sorted([os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, "*.json"))])

def load_case_data(case_filename: str) -> dict | None:
    """Loads data for a specific case."""
    filepath = os.path.join(DATA_DIR, case_filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading case {case_filename}: {e}")
        return None

def save_evaluation(case_filename: str, evaluation_data: dict, expert_name: str):
    """Saves the expert's evaluation."""
    base_name = os.path.splitext(case_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = f"{base_name}_eval_{expert_name.replace(' ', '_')}_{timestamp}.json"
    filepath = os.path.join(EVAL_DIR, eval_filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        return True, eval_filename
    except Exception as e:
        print(f"Error saving evaluation for {case_filename}: {e}")
        return False, None

# --- Example: How you might generate a case file (run this separately to create samples) ---
def create_sample_case_file(case_id: str, patient_context: str, llm_recommendation: str, guideline_used: str):
    case_data = {
        "case_id": case_id,
        "llm_interaction_id": llm_interaction_id,
        "patient_context_summary": patient_context, # Keep this concise for display
        "llm_full_recommendation": llm_recommendation, # The full text including Begründung
        "guideline_used": guideline_used,
        "timestamp_generated": datetime.now().isoformat()
    }
    filename = f"{case_id}.json"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(case_data, f, indent=2, ensure_ascii=False)
    print(f"Created sample case: {filename}")

if __name__ == '__main__':
    # Example: Create a couple of sample case files to test the evaluation tool
    create_sample_case_file(
        case_id="case_001",
        patient_context="65 y/o male, DLBCL Stage IIA, IPI 1. Post 4 cycles R-CHOP. PET-CT shows Deauville 2.",
        llm_recommendation="Therapieempfehlung:\nBeobachtung gemäß ESMO-Leitlinien.\n\nBegründung:\nDer Patient hat nach 4 Zyklen R-CHOP eine komplette metabolische Remission (Deauville 2) erreicht. Gemäß ESMO-Leitlinien für DLBCL mit niedrigem IPI und CR nach Induktion ist eine Beobachtung der Standardansatz. Weitere Therapie ist nicht indiziert.",
        guideline_used="ESMO DLBCL 2023"
    )
    create_sample_case_file(
        case_id="case_002",
        patient_context="72 y/o female, Follicular Lymphoma Grade 3A, Stage III. Symptomatic. FLIPI 3.",
        llm_recommendation="Therapieempfehlung:\nInitiierung einer Erstlinientherapie mit R-Bendamustin.\n\nBegründung:\nDie Patientin ist symptomatisch mit einem FLIPI-Score von 3, was eine Therapieindikation darstellt. R-Bendamustin ist eine Standardoption für fitte, ältere Patienten mit Follicular Lymphoma gemäß Onkopedia und internationalen Leitlinien und bietet ein gutes Nutzen-Risiko-Profil.",
        guideline_used="Onkopedia Follicular Lymphoma"
    )