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
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
except ImportError as e:
    print(f"Error importing project modules or Langchain: {e}")
    print("Ensure this script is run from the project root or that 'llmRecom' is in your PYTHONPATH,")
    print("and that Langchain components are installed.")
    exit(1)

# --- Configuration for this Script ---
DEFAULT_LLM_MODEL_SINGLE_PROMPT = "qwen3:32b"  # Or your preferred large context model
DEFAULT_GUIDELINE_PROVIDER_SINGLE_PROMPT = "ESMO"
OUTPUT_FILE_SINGLE_PROMPT = "single_prompt_recommendations.json"

# Define which patient data fields to include in the single prompt
# Adjust this list based on what's most critical for the LLM to see
PATIENT_FIELDS_FOR_PROMPT = [
    "id", # Patient ID
    "main_diagnosis", # Diagnose-Kürzel
    "main_diagnosis_text", # Hauptdiagnose (Text) - This can be long!
    "ann_arbor_stage", # Ann-Arbor Stadium
    "secondary_diagnoses", # Relevante Nebendiagnosen
    "clinical_info", # Klinische Angaben / Fragestellung
    "pet_ct_report", # PET-CT Bericht - This can also be very long!
    "accompanying_symptoms", # Begleitsymptome
    "prognosis_score" # Prognose-Score
    # Add or remove fields as needed. Be mindful of context window limits.
]


setup_logging()
logger = logging.getLogger("single_prompt_processor")

def format_patient_data_for_prompt(patient_row: dict, fields: list) -> str:
    """Formats selected patient data fields into a string for the prompt."""
    prompt_text = "Patienteninformationen:\n"
    for field in fields:
        value = patient_row.get(field)
        if value is not None and str(value).strip() != "":
            # Simple pretty printing for a field name
            field_name_pretty = field.replace("_", " ").title()
            prompt_text += f"- {field_name_pretty}: {str(value)}\n"
    return prompt_text.strip()

def generate_single_recommendation(
    patient_data_dict: dict, # Dictionary representing a single patient's relevant data
    guideline_name: str,
    llm: OllamaLLM
) -> tuple[str | None, str | None, float | None]:
    """
    Generates a therapy recommendation using a single comprehensive prompt.
    Returns: (recommendation_text, error_message, llm_duration)
    """
    
    # Construct the comprehensive prompt
    prompt_template_str = """
Du bist ein erfahrener Onkologie-Experte. Deine Aufgabe ist es, eine präzise und begründete Therapieempfehlung für den unten beschriebenen Patienten zu erstellen.
Stütze deine Empfehlung AUSSCHLIESSLICH auf die bereitgestellten Patienteninformationen und die angegebene medizinische Leitlinie.

**Zu berücksichtigende Leitlinie:** {guideline_name}

**{patient_data_string}**

**Anweisungen für die Therapieempfehlung:**
1.  Analysiere die oben genannten "Patienteninformationen" sorgfältig.
2.  Berücksichtige **ausschließlich** die angegebene Leitlinie: **{guideline_name}**.
3.  Formuliere **EINE EINZIGE, KONKRETE THERAPIEEMPFEHLUNG**.
4.  Gib eine klare und prägnante **Begründung** für die empfohlene Therapie. Die Begründung muss sich explizit auf die relevanten Punkte aus den Patienteninformationen und die Empfehlungen der Leitlinie beziehen.
5.  Wenn die Informationen für eine definitive Empfehlung unzureichend sind, gib dies an und schlage notwendige weitere diagnostische Schritte vor.
6.  Strukturiere deine Antwort wie folgt und füge keine zusätzlichen einleitenden oder abschließenden Sätze hinzu:

**Therapieempfehlung:**
[Hier deine konkrete Therapieempfehlung einfügen.]

**Begründung:**
[Hier deine detaillierte Begründung einfügen.]
    """

    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["guideline_name", "patient_data_string"]
    )

    chain = LLMChain(llm=llm, prompt=prompt) # Using simple LLMChain

    llm_duration = None
    start_time = time.perf_counter()
    try:
        logger.info(f"Generating single-prompt recommendation for Patient ID: {patient_data_dict.get('id', 'N/A')} using {guideline_name} guideline.")
        # The 'patient_data_string' is the formatted patient info
        response = chain.invoke({
            "guideline_name": guideline_name,
            "patient_data_string": format_patient_data_for_prompt(patient_data_dict, PATIENT_FIELDS_FOR_PROMPT)
        })
        llm_duration = time.perf_counter() - start_time
        
        recommendation_text = response.get('text', '').strip() # LLMChain returns a dict with 'text' key
        
        # Optional: Basic extraction of <think> block if model still produces it
        think_tag_end = "</think>"
        match = re.search(re.escape(think_tag_end), recommendation_text, re.IGNORECASE)
        if match:
            recommendation_text = recommendation_text[match.end():].strip()
            
        logger.info(f"LLM ({llm.model}) for patient {patient_data_dict.get('id', 'N/A')} took {llm_duration:.2f}s. Recommendation snippet: {recommendation_text[:100]}...")
        return recommendation_text, None, llm_duration

    except Exception as e:
        llm_duration = time.perf_counter() - start_time if start_time else None
        logger.error(f"Error generating single-prompt recommendation for Patient ID {patient_data_dict.get('id', 'N/A')}: {e}", exc_info=True)
        return None, str(e), llm_duration


def run_single_prompt_processing(
    llm_model_override: str | None = None,
    guideline_override: str | None = None,
    output_file: str = OUTPUT_FILE_SINGLE_PROMPT
):
    logger.info("Starting single-prompt processing for all patients...")

    # --- LLM Configuration ---
    effective_llm_model = llm_model_override if llm_model_override else DEFAULT_LLM_MODEL_SINGLE_PROMPT
    try:
        llm = OllamaLLM(model=effective_llm_model, temperature=config.LLM_TEMPERATURE if hasattr(config, 'LLM_TEMPERATURE') else 0.7)
        logger.info(f"Initialized LLM: {effective_llm_model}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM {effective_llm_model}: {e}", exc_info=True)
        return

    effective_guideline = guideline_override if guideline_override else DEFAULT_GUIDELINE_PROVIDER_SINGLE_PROMPT

    # --- Load Patient Data ---
    # Assuming config.TUBO_EXCEL_FILE_PATH is set correctly in your config.py
    df_patients = load_patient_data(config.TUBO_EXCEL_FILE_PATH)
    if df_patients is None or df_patients.empty:
        logger.error("No patient data loaded. Exiting.")
        return

    all_patient_recommendations = []
    patient_ids = [pid for pid in df_patients["ID"].unique() if pid is not None and str(pid).strip() != ""]
    total_patients = len(patient_ids)
    logger.info(f"Found {total_patients} unique patient IDs to process.")

    for i, patient_id in enumerate(patient_ids):
        logger.info(f"Processing patient {i+1}/{total_patients}: {patient_id}")
        
        patient_row_series = df_patients[df_patients["ID"] == patient_id].iloc[0]
        patient_data_dict = patient_row_series.to_dict() # Convert row to dict for easier access

        recommendation_text, error_msg, llm_duration = generate_single_recommendation(
            patient_data_dict,
            effective_guideline,
            llm
        )

        result_entry = {
            "patient_id_original": str(patient_id),
            "patient_data_source_file": os.path.basename(config.TUBO_EXCEL_FILE_PATH),
            "timestamp_processed": datetime.now().isoformat(),
            "llm_model_used": effective_llm_model,
            "guideline_used": effective_guideline,
            "single_prompt_recommendation": recommendation_text,
            "llm_generation_time_s": llm_duration,
            "error": error_msg
        }
        all_patient_recommendations.append(result_entry)

    # --- Save all results ---
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir): # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_patient_recommendations, f, indent=2, ensure_ascii=False)
        logger.info(f"Single-prompt processing complete. All recommendations saved to: {output_file}")
    except IOError as e:
        logger.error(f"Failed to write results to {output_file}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error writing results: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    import re # For <think> tag removal in generate_single_recommendation if needed

    parser = argparse.ArgumentParser(description="Generate therapy recommendations using a single prompt for all patients.")
    parser.add_argument(
        "--llm_model", type=str, default=DEFAULT_LLM_MODEL_SINGLE_PROMPT,
        help=f"LLM model to use. Default: {DEFAULT_LLM_MODEL_SINGLE_PROMPT}"
    )
    parser.add_argument(
        "--guideline", type=str, default=DEFAULT_GUIDELINE_PROVIDER_SINGLE_PROMPT,
        help=f"Guideline to reference. Default: {DEFAULT_GUIDELINE_PROVIDER_SINGLE_PROMPT}"
    )
    parser.add_argument(
        "--output_file", type=str, default=OUTPUT_FILE_SINGLE_PROMPT,
        help=f"Path to save the JSON results. Default: {OUTPUT_FILE_SINGLE_PROMPT}"
    )
    parser.add_argument(
        "--patient_data_file", type=str, default=None,
        help="Optional: Path to a specific patient data Excel file to use. Overrides config.TUBO_EXCEL_FILE_PATH for this run."
    )


    args = parser.parse_args()

    # Override patient data file in config if specified
    if args.patient_data_file:
        logger.info(f"Overriding patient data file path from config with: {args.patient_data_file}")
        config.TUBO_EXCEL_FILE_PATH = args.patient_data_file


    logger.info(f"Running single-prompt processing with LLM: {args.llm_model}, Guideline: {args.guideline}")

    run_single_prompt_processing(
        llm_model_override=args.llm_model,
        guideline_override=args.guideline,
        output_file=args.output_file
    )