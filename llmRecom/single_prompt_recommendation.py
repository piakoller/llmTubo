import logging
import json
import os
import time
from datetime import datetime
import re
import glob

# -- NET -- '
NET = True

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
DEFAULT_LLM_MODEL_SINGLE_PROMPT = "llama3" # Updated default
DEFAULT_GUIDELINE_PROVIDER_SINGLE_PROMPT = "ESMO"
EVAL_SINGLE_DATA_DIR = "/home/pia/projects/llmTubo/tuboEval/data_for_evaluation/single_prompt"
EVAL_SINGLE_NET_DATA_DIR = "/home/pia/projects/llmTubo/tuboEval/data_for_evaluation/single_prompt/net"


GUIDELINE_DATA_DIR_NET = "/home/pia/projects/llmTubo/llmRecom/data/NET/mds"
GUIDELINE_DATA_DIR_LMYPH = "/home/pia/projects/llmTubo/llmRecom/data/guidelines/mds"

PATIENT_FIELDS_FOR_PROMPT = [
    "id", "main_diagnosis", "main_diagnosis_text", "ann_arbor_stage", 
    "secondary_diagnoses", "clinical_info", "pet_ct_report", 
    "accompanying_symptoms", "prognosis_score"
]

setup_logging()
logger = logging.getLogger("single_prompt_processor")

def find_guideline_and_net_files(guideline_name: str, main_diagnosis: str, net_mode: bool) -> list[str]:
    """
    Sucht nach der passenden Guideline-Datei und – falls NET aktiv – nach NET-spezifischen Attachments.
    Gibt eine Liste aller zu attachenden Dateien zurück.
    """
    attachments = []
    if net_mode:
        guideline_dir = GUIDELINE_DATA_DIR_NET
        # Guideline-Datei suchen
        guideline_name_lower = guideline_name.lower()
        main_diag_lower = main_diagnosis.lower().replace(" ", "_")

        # NET Pressemittelung und NET Studie hinzufügen (Dateien mit "press" oder "study" im Namen)
        # Auch hier nur .mds-Dateien berücksichtigen
        net_guideline = glob.glob(os.path.join(guideline_dir, "*Guideline.md"))
        net_press = glob.glob(os.path.join(guideline_dir, "*press*.md"))
        net_study = glob.glob(os.path.join(guideline_dir, "*study*.md"))
        for f in net_guideline + net_press + net_study:
            if f not in attachments:
                attachments.append(f)
    else:
        guideline_dir = GUIDELINE_DATA_DIR_LMYPH
        guideline_name_lower = guideline_name.lower()
        main_diag_lower = main_diagnosis.lower().replace(" ", "_")
        pattern = f"*{guideline_name_lower}*{main_diag_lower}*"
        files = glob.glob(os.path.join(guideline_dir, pattern + ".mds")) 
        if files:
            attachments.append(files[0])
    return attachments

def get_results_filename(llm_model: str, clinical_info_modified: bool) -> str:
    llm_safe = llm_model.replace(":", "_").replace("/", "_").replace(".", "_")
    return f"single_prompt_{llm_safe}_clinical_info_modified_{clinical_info_modified}.json"

def format_patient_data_for_prompt(patient_row: dict, fields: list) -> str:
    prompt_text = "Patienteninformationen:\n"
    for field in fields:
        value = patient_row.get(field)
        if value is not None and str(value).strip() != "":
            field_name_pretty = field.replace("_", " ").title()
            prompt_text += f"- {field_name_pretty}: {str(value)}\n"
    return prompt_text.strip()

def load_markdown_content(filepath: str) -> str:
    """Lädt den Inhalt einer Markdown-Datei."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Fehler beim Laden der Markdown-Datei {filepath}: {e}")
        return f"Fehler beim Laden von {os.path.basename(filepath)}."

def generate_single_recommendation(
    patient_data_dict: dict,
    guideline_name: str,
    llm: OllamaLLM,
    clinical_info_modified_flag: bool
) -> tuple[str | None, str | None, str | None, float | None, dict]:
    patient_data_string = format_patient_data_for_prompt(patient_data_dict, PATIENT_FIELDS_FOR_PROMPT)
    if clinical_info_modified_flag:
        logger.info(f"Clinical info modified flag is TRUE for patient {patient_data_dict.get('id', 'N/A')}.")

    # --- Guideline-Dateien und ggf. NET-Anhänge suchen ---
    main_diagnosis = str(patient_data_dict.get("main_diagnosis", "")).strip()
    # attachments enthält nun nur Pfade zu .mds Dateien
    attachments = find_guideline_and_net_files(guideline_name, main_diagnosis, NET)
    
    attached_guideline_content = ""
    attached_file_names = [] # Zum Speichern der Namen der angehängten Dateien für das Logging/Debugging
    if attachments:
        logger.info(f"Attempting to load content from attachments: {attachments}")
        for filepath in attachments:
            content = load_markdown_content(filepath)
            if content:
                attached_guideline_content += f"\n\n--- Inhalt von {os.path.basename(filepath)} ---\n{content}\n---------------------------\n"
                attached_file_names.append(os.path.basename(filepath))
        if not attached_guideline_content:
            logger.warning(f"Kein lesbarer Inhalt aus den gefundenen Attachment-Dateien für {guideline_name} und Diagnose {main_diagnosis}")
    else:
        logger.warning(f"Keine Guideline-Datei (.mds) gefunden für {guideline_name} und Diagnose {main_diagnosis}")

    prompt_template_str = """
Du bist ein erfahrener Onkologie-Experte. Deine Aufgabe ist es, eine präzise und begründete Therapieempfehlung für den unten beschriebenen Patienten zu erstellen.
Stütze deine Empfehlung AUSSCHLIESSLICH auf die bereitgestellten Patienteninformationen und die angegebene medizinische Leitlinie sowie die angehängten Dokumente.
{modified_context_indicator}
**Zu berücksichtigende Leitlinie:** {guideline_name}

**Patienteninformationen:**
{patient_data_string}

{attached_guideline_content}

**Anweisungen für die Therapieempfehlung:**
1.  Analysiere die oben genannten "Patienteninformationen" sorgfältig.
2.  Berücksichtige **ausschließlich** die angegebene Leitlinie: **{guideline_name}**, die als Datei angehängten Dokumente und deren Inhalte.
3.  Formuliere **EINE EINZIGE, KONKRETE THERAPIEEMPFEHLUNG** auf Deutsch.
4.  Gib eine klare und prägnante **Begründung** für die empfohlene Therapie auf Deutsch. Die Begründung muss sich explizit auf die relevanten Punkte aus den Patienteninformationen und die Empfehlungen der Leitlinie/der angehängten Dokumente beziehen.
5.  Wenn die Informationen für eine definitive Empfehlung unzureichend sind, gib dies an und schlage notwendige weitere diagnostische Schritte vor.
6.  Strukturiere deine Antwort wie folgt und füge keine zusätzlichen einleitenden oder abschließenden Sätze hinzu (gib auch einen <think> Block aus, bevor du antwortest):

<think>
[Hier deine Denkprozesse vor der finalen Antwort einfügen. Dieser Block wird später separat gespeichert.]
</think>
**Therapieempfehlung:**
[Hier deine konkrete Therapieempfehlung einfügen.]

**Begründung:**
[Hier deine detaillierte Begründung einfügen.]
    """

    modified_context_indicator_text = ""
    if clinical_info_modified_flag:
        modified_context_indicator_text = "**Hinweis: Der für diese Anfrage verwendete klinische Kontext wurde als 'modifiziert' gekennzeichnet.**\n"

    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["guideline_name", "patient_data_string", "modified_context_indicator", "attached_guideline_content"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    llm_duration = None
    raw_llm_response = None
    final_recommendation_text = None
    start_time = time.perf_counter()
    
    # Prepare the input variables for the PromptTemplate
    template_input_vars = {
        "guideline_name": guideline_name,
        "patient_data_string": patient_data_string,
        "modified_context_indicator": modified_context_indicator_text,
        "attached_guideline_content": attached_guideline_content # Hinzugefügt
    }

    try:
        logger.info(f"Generating single-prompt recommendation for Patient ID: {patient_data_dict.get('id', 'N/A')} using {guideline_name} guideline. Modified: {clinical_info_modified_flag}")
        if attached_file_names:
            logger.info(f"Attaching content from markdown files: {', '.join(attached_file_names)}")
        
        llm_kwargs = {} 
        
        response_dict = chain.invoke(template_input_vars, **llm_kwargs) 

        # --- Save the input to the LLM ---
        llm_input = {
            "prompt_text": prompt.format(**template_input_vars), # Formatted Prompt mit allen Inhalten
            "attachments_used_for_content": attached_file_names, # Liste der Dateinamen, deren Inhalt angehängt wurde
            "llm_kwargs": {}
        }

        llm_duration = time.perf_counter() - start_time
        
        raw_llm_response = response_dict.get('text', '').strip()
        think_tag_end = "</think>"
        match = re.search(re.escape(think_tag_end), raw_llm_response, re.IGNORECASE)
        if match:
            final_recommendation_text = raw_llm_response[match.end():].strip()
        else:
            final_recommendation_text = raw_llm_response
            logger.warning(f"No <think> block found in LLM response for patient {patient_data_dict.get('id', 'N/A')}. Raw response used as final.")
            
        logger.info(f"LLM ({llm.model}) for patient {patient_data_dict.get('id', 'N/A')} took {llm_duration:.2f}s. Final recommendation snippet: {final_recommendation_text[:100]}...")
        return final_recommendation_text, raw_llm_response, None, llm_duration, llm_input

    except Exception as e:
        llm_duration = time.perf_counter() - start_time if start_time else None
        error_msg = str(e)
        logger.error(f"Error generating single-prompt recommendation for Patient ID {patient_data_dict.get('id', 'N/A')}: {error_msg}", exc_info=True)
        return None, raw_llm_response, error_msg, llm_duration, {} 

def run_single_prompt_processing(
    llm_model_override: str | None = None,
    guideline_override: str | None = None,
    output_filepath_override: str | None = None,
    clinical_info_modified_arg: bool = False,
    patient_data_filepath_override: str | None = None
):
    logger.info("Starting single-prompt processing for all patients...")

    effective_clinical_info_modified = clinical_info_modified_arg
    effective_patient_data_file = config.TUBO_EXCEL_FILE_PATH
    original_config_patient_file = config.TUBO_EXCEL_FILE_PATH

    if patient_data_filepath_override:
        effective_patient_data_file = patient_data_filepath_override
        logger.info(f"Using patient data file specified by argument: {effective_patient_data_file}")
        if not effective_clinical_info_modified:
            logger.info(f"Patient data file was overridden, setting clinical_info_modified to True.")
        effective_clinical_info_modified = True
    
    if effective_patient_data_file != config.TUBO_EXCEL_FILE_PATH:
        config.TUBO_EXCEL_FILE_PATH = effective_patient_data_file

    effective_llm_model = llm_model_override if llm_model_override else DEFAULT_LLM_MODEL_SINGLE_PROMPT
    try:
        llm = OllamaLLM(model=effective_llm_model, temperature=config.LLM_TEMPERATURE if hasattr(config, 'LLM_TEMPERATURE') else 0.7)
        logger.info(f"Initialized LLM: {effective_llm_model}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM {effective_llm_model}: {e}", exc_info=True)
        if effective_patient_data_file != original_config_patient_file: config.TUBO_EXCEL_FILE_PATH = original_config_patient_file
        return

    effective_guideline = guideline_override if guideline_override else DEFAULT_GUIDELINE_PROVIDER_SINGLE_PROMPT

    final_output_file: str
    if output_filepath_override:
        final_output_file = output_filepath_override
    else:
        generated_filename = get_results_filename(effective_llm_model, effective_clinical_info_modified)
        if NET:
            final_output_file = os.path.join(EVAL_SINGLE_NET_DATA_DIR, generated_filename)
        else:
            final_output_file = os.path.join(EVAL_SINGLE_DATA_DIR, generated_filename)
    logger.info(f"Output will be saved to: {final_output_file}")
    
    try:
        output_dir_for_final_file = os.path.dirname(final_output_file)
        if output_dir_for_final_file:
             os.makedirs(output_dir_for_final_file, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directory {os.path.dirname(final_output_file)}: {e}. Exiting.")
        if effective_patient_data_file != original_config_patient_file: config.TUBO_EXCEL_FILE_PATH = original_config_patient_file
        return

    df_patients = load_patient_data(effective_patient_data_file)
    if df_patients is None or df_patients.empty:
        logger.error(f"No patient data loaded from {effective_patient_data_file}. Exiting.")
        if effective_patient_data_file != original_config_patient_file: config.TUBO_EXCEL_FILE_PATH = original_config_patient_file
        return

    all_patient_recommendations = []
    patient_ids = [pid for pid in df_patients["ID"].unique() if pid is not None and str(pid).strip() != ""]
    total_patients = len(patient_ids)
    logger.info(f"Found {total_patients} unique patient IDs to process from {os.path.basename(effective_patient_data_file)}.")

    for i, patient_id in enumerate(patient_ids):
        logger.info(f"Processing patient {i+1}/{total_patients}: {patient_id}")
        
        patient_row_series = df_patients[df_patients["ID"] == patient_id].iloc[0]
        patient_data_dict = patient_row_series.to_dict()

        # Unpack the new 5-tuple
        final_recommendation, raw_response_with_think, error_msg, llm_duration, llm_input_data = generate_single_recommendation(
            patient_data_dict,
            effective_guideline,
            llm,
            effective_clinical_info_modified
        )

        result_entry = {
            "patient_id_original": str(patient_id),
            "patient_data_source_file": os.path.basename(effective_patient_data_file),
            "timestamp_processed": datetime.now().isoformat(),
            "llm_model_used": effective_llm_model,
            "clinical_info_modified": effective_clinical_info_modified,
            "guideline_used": effective_guideline,
            "llm_input": llm_input_data,
            "llm_raw_output_with_think": raw_response_with_think, # Save the raw output
            "single_prompt_recommendation_final": final_recommendation, # Save the processed final part
            "llm_generation_time_s": f"{llm_duration:.4f}" if llm_duration is not None else None,
            "error": error_msg
        }
        all_patient_recommendations.append(result_entry)

    try:
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_patient_recommendations, f, indent=2, ensure_ascii=False)
        logger.info(f"Single-prompt processing complete. All recommendations saved to: {final_output_file}")
    except Exception as e:
        logger.error(f"Failed to write results to {final_output_file}: {e}", exc_info=True)
    
    if effective_patient_data_file != original_config_patient_file:
        config.TUBO_EXCEL_FILE_PATH = original_config_patient_file
        logger.info(f"Restored patient data file path in config to: {config.TUBO_EXCEL_FILE_PATH}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate therapy recommendations using a single prompt for all patients.")
    # ... (argparse setup remains the same as your previous version) ...
    parser.add_argument(
        "--patient_data_file", type=str, default=None,
        help="Optional: Path to the patient data Excel file. If provided, 'clinical_info_modified' is set to True."
    )
    parser.add_argument(
        "--llm_model", type=str, default=DEFAULT_LLM_MODEL_SINGLE_PROMPT,
        help=f"LLM model to use. Default: {DEFAULT_LLM_MODEL_SINGLE_PROMPT}"
    )
    parser.add_argument(
        "--guideline", type=str, default=DEFAULT_GUIDELINE_PROVIDER_SINGLE_PROMPT,
        help=f"Guideline to reference. Default: {DEFAULT_GUIDELINE_PROVIDER_SINGLE_PROMPT}"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help=f"Path to save the JSON results. If not set, generated into EVAL_SINGLE_DATA_DIR."
    )
    parser.add_argument(
        "--clinical_info_modified", action="store_true",
        help="Set if 'clinical_info' or other context was specially modified. Automatically True if --patient_data_file is used."
    )

    args = parser.parse_args()

    clinical_info_modified_for_run = args.clinical_info_modified
    if args.patient_data_file:
        if not clinical_info_modified_for_run:
            logger.info(f"Patient data file ('{os.path.basename(args.patient_data_file)}') specified, "
                        f"forcing 'clinical_info_modified' to True for this run.")
        clinical_info_modified_for_run = True
    
    patient_data_file_to_log = args.patient_data_file if args.patient_data_file else getattr(config, 'TUBO_EXCEL_FILE_PATH', 'N/A')

    logger.info(f"Running single-prompt processing with Patient Data File: {patient_data_file_to_log}, "
                f"LLM: {args.llm_model}, Guideline: {args.guideline}, "
                f"Clinical Info Modified: {clinical_info_modified_for_run}")

    run_single_prompt_processing(
        llm_model_override=args.llm_model,
        guideline_override=args.guideline,
        output_filepath_override=args.output_file,
        clinical_info_modified_arg=clinical_info_modified_for_run,
        patient_data_filepath_override=args.patient_data_file
    )