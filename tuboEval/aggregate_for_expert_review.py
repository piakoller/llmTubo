import json
import os
import glob
import pandas as pd
from collections import defaultdict
import logging # Added for better feedback

# --- Configuration ---
# List of directories where your batch JSON output files are stored
RESULTS_DIRS = [
    "/home/pia/projects/llmTubo/tuboEval/data_for_evaluation/agent/",
    "/home/pia/projects/llmTubo/tuboEval/data_for_evaluation/single_prompt/"
]

# Output Excel file name
EXCEL_OUTPUT_FILE = "expert_evaluation_sheet_v2.xlsx" # Versioning the output
# Directory to save the Excel file
OUTPUT_DIR_FOR_EXCEL = "/home/pia/projects/llmTubo/tuboEval/expert_review_sheets/"

# Setup basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aggregator")


def extract_recommendation_details(filename: str, patient_run_entry: dict) -> tuple[str | None, str | None, str | None]:
    """
    Extracts the recommendation type, final recommendation text, and think block.
    Adjust logic based on the structure of your JSON files from different scripts.
    """
    rec_type = "UnknownType"
    final_rec_text = None
    think_block_text = None
    raw_response_text = None # To store the full raw response

    # Derive type from filename and/or content
    # This logic needs to be robust to differentiate your 4+ types correctly
    llm_model = patient_run_entry.get("llm_model_used", "UnknownLLM").replace(":", "_").replace("/", "_").replace(".", "_")
    modified = patient_run_entry.get("clinical_info_modified", False)
    
    if "agent_" in filename: # Heuristic for multi-agent results
        rec_type = f"MultiAgent_{llm_model}_Mod{modified}"
        final_rec_text = patient_run_entry.get("therapie_output_final") # From multi-agent output structure
        if final_rec_text is None: # Fallback for older multi-agent output structure
             final_rec_text = patient_run_entry.get("therapie_output")
        think_block_text = patient_run_entry.get("therapie_think_block")
        raw_response_text = patient_run_entry.get("therapie_raw_response")
    elif "single_prompt_" in filename: # Heuristic for single-prompt results
        rec_type = f"SinglePrompt_{llm_model}_Mod{modified}"
        final_rec_text = patient_run_entry.get("single_prompt_recommendation_final")
        # For single_prompt, the raw output contains the think block
        raw_response_text = patient_run_entry.get("llm_raw_output_with_think")
        # We need to extract the think block from the raw response if not already separated
        if raw_response_text and not think_block_text: # Only if not already provided as separate field
            think_tag_start_pattern = r"<think>"
            think_tag_end_pattern = r"</think>"
            try:
                import re # Local import for regex
                match = re.search(f"{think_tag_start_pattern}(.*?){think_tag_end_pattern}", 
                                  raw_response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    think_block_text = match.group(1).strip()
            except Exception:
                pass # Ignore if regex fails for some reason on raw text
    else:
        # Fallback if filename doesn't match known patterns, try to infer from content
        rec_type = f"Other_{llm_model}_Mod{modified}"
        # Try common keys for recommendation text
        final_rec_text = patient_run_entry.get("single_prompt_recommendation_final") or \
                         patient_run_entry.get("therapie_output_final") or \
                         patient_run_entry.get("therapie_output")
        think_block_text = patient_run_entry.get("therapie_think_block") # Or extract if needed

    # Ensure raw_response_text has a value if possible
    if not raw_response_text:
        raw_response_text = patient_run_entry.get("llm_raw_output_with_think") or \
                            patient_run_entry.get("therapie_raw_response") or \
                            patient_run_entry.get("diagnostik_raw_response") # Less ideal fallback

    return rec_type, final_rec_text, think_block_text, raw_response_text


def main():
    os.makedirs(OUTPUT_DIR_FOR_EXCEL, exist_ok=True)
    
    all_json_files_to_process = []
    for res_dir in RESULTS_DIRS:
        if not os.path.isdir(res_dir):
            logger.warning(f"Results directory not found: {res_dir}. Skipping.")
            continue
        all_json_files_to_process.extend(glob.glob(os.path.join(res_dir, "*.json")))
    
    if not all_json_files_to_process:
        logger.error(f"No JSON files found in any of the specified results directories: {RESULTS_DIRS}")
        return

    # Structure: { patient_id: { "metadata": {...}, "recommendations": { "type1": {"final_rec": "...", "think": "...", "raw": "..."}, ... } } }
    patient_aggregated_data = defaultdict(lambda: {"metadata": {}, "recommendations": {}})
    logger.info(f"Processing {len(all_json_files_to_process)} JSON files...")

    for json_file_path in all_json_files_to_process:
        filename_only = os.path.basename(json_file_path)
        logger.info(f"Reading file: {filename_only}")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data_from_file = json.load(f) 
                
                if not isinstance(data_from_file, list):
                    logger.warning(f"File {filename_only} does not contain a list of patient results. Skipping.")
                    continue

                for patient_run_entry in data_from_file:
                    patient_id = patient_run_entry.get("patient_id_original")
                    if not patient_id:
                        logger.warning(f"Entry missing 'patient_id_original' in {filename_only}. Skipping entry.")
                        continue

                    # Store patient summary/metadata only once if it's consistent
                    # Or, if metadata like data_source_file can vary per run for the same patient, store it differently
                    if not patient_aggregated_data[patient_id]["metadata"]:
                        patient_aggregated_data[patient_id]["metadata"] = {
                            "patient_data_summary": patient_run_entry.get("patient_context_summary_for_eval", f"Summary N/A for {patient_id}"),
                            "data_source_file": patient_run_entry.get("patient_data_source_file", "N/A"),
                            # Add any other consistent patient metadata here
                        }
                    
                    rec_type, final_rec, think_block, raw_response = extract_recommendation_details(filename_only, patient_run_entry)
                    
                    if final_rec or raw_response : # Only add if we have some recommendation text or raw output
                        patient_aggregated_data[patient_id]["recommendations"][rec_type] = {
                            "final_recommendation": final_rec if final_rec else "N/A",
                            "think_block": think_block if think_block else "", # Ensure it's a string
                            "raw_response_with_think": raw_response if raw_response else (think_block + "\n" + final_rec if think_block and final_rec else final_rec or "N/A")
                        }
                    else:
                         patient_aggregated_data[patient_id]["recommendations"][rec_type] = {
                            "final_recommendation": "ERROR or N/A in source file",
                            "think_block": "",
                            "raw_response_with_think": "ERROR or N/A in source file"
                         }

        except Exception as e:
            logger.error(f"Error processing file {filename_only}: {e}", exc_info=True)
            continue
            
    if not patient_aggregated_data:
        logger.error("No patient data aggregated. Exiting.")
        return

    # --- Prepare data for Pandas DataFrame ---
    records_for_df = []
    all_rec_types_found = sorted(list(set(
        rtype for data in patient_aggregated_data.values() for rtype in data["recommendations"].keys()
    )))

    if not all_rec_types_found:
        logger.error("No recommendation types found after processing. Cannot create Excel sheet.")
        return

    logger.info(f"Found recommendation types: {all_rec_types_found}")

    # Base columns + dynamically added recommendation and evaluation columns
    excel_columns = ["Patient ID", "Patient Data Summary", "Source File"]

    for rec_type in all_rec_types_found:
        excel_columns.append(f"{rec_type} - Final Recommendation")
        excel_columns.append(f"{rec_type} - Think Block")
        # excel_columns.append(f"{rec_type} - Full Raw Response") # Optional: if you want another column for this

    for rec_type in all_rec_types_found: # Separate loop for eval columns to group them
        excel_columns.append(f"Eval: {rec_type} - Adherence (1-5)")
        excel_columns.append(f"Eval: {rec_type} - Correctness (1-5)")
        excel_columns.append(f"Eval: {rec_type} - Clarity (1-5)")
        excel_columns.append(f"Eval: {rec_type} - Overall (1-5 Best)")
        excel_columns.append(f"Eval: {rec_type} - Comments")
    
    excel_columns.append("Overall Best Recommendation (Type Name)")
    excel_columns.append("General Comments for Patient")


    for patient_id, data in patient_aggregated_data.items():
        record = {
            "Patient ID": patient_id,
            "Patient Data Summary": data["metadata"].get("patient_data_summary", ""),
            "Source File": data["metadata"].get("data_source_file", "") 
        }
        for rec_type in all_rec_types_found:
            rec_data = data["recommendations"].get(rec_type, {})
            record[f"{rec_type} - Final Recommendation"] = rec_data.get("final_recommendation", "N/A")
            record[f"{rec_type} - Think Block"] = rec_data.get("think_block", "")
            # record[f"{rec_type} - Full Raw Response"] = rec_data.get("raw_response_with_think", "N/A")

            # Initialize empty evaluation fields
            record[f"Eval: {rec_type} - Adherence (1-5)"] = ""
            record[f"Eval: {rec_type} - Correctness (1-5)"] = ""
            record[f"Eval: {rec_type} - Clarity (1-5)"] = ""
            record[f"Eval: {rec_type} - Overall (1-5 Best)"] = ""
            record[f"Eval: {rec_type} - Comments"] = ""
        
        record["Overall Best Recommendation (Type Name)"] = ""
        record["General Comments for Patient"] = ""
        records_for_df.append(record)

    df = pd.DataFrame(records_for_df)
    
    if not df.empty:
        df = df[excel_columns] # Ensure desired column order and include all expected columns
    else:
        logger.warning("DataFrame is empty, cannot create Excel sheet with data.")
        # Create an empty Excel with headers if no data
        df = pd.DataFrame(columns=excel_columns)


    # --- Save to Excel ---
    excel_file_path = os.path.join(OUTPUT_DIR_FOR_EXCEL, EXCEL_OUTPUT_FILE)
    try:
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Expert Evaluation', index=False)
            
            instructions_data = {
                "Rating Scale": ["1=Poor/Strongly Disagree", "2=Fair/Disagree", "3=Neutral/Okay", "4=Good/Agree", "5=Excellent/Strongly Agree"],
                "Adherence": ["Guideline Adherence."], "Correctness": ["Clinical Correctness & Safety."],
                "Clarity": ["Clarity & Explainability of Justification."], "Overall (per rec)": ["Overall quality of this specific recommendation."],
                "Overall Best Rec (Type)": ["Enter the 'Type Name' column header of the recommendation you think is best (e.g., MultiAgent_Llama3_ModTrue)."]
            }
            instructions_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in instructions_data.items() ])) # Handle uneven lists
            instructions_df.to_excel(writer, sheet_name='Instructions', index=False)

        logger.info(f"Successfully created Excel sheet for expert evaluation: {excel_file_path}")
    except Exception as e:
        logger.error(f"Error writing Excel file: {e}", exc_info=True)

if __name__ == "__main__":
    main()