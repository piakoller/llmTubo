import json
import os
import glob
import pandas as pd
from collections import defaultdict
import logging
import re # Import the re module for regex

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIRS = [
    os.path.join(BASE_DIR, "data_for_evaluation", "agent", "net"),
    os.path.join(BASE_DIR, "data_for_evaluation", "single_prompt", "net")
]

# Output JSON file name
JSON_OUTPUT_FILE = "net_expert_evaluation_data.json"

# Directory to save the JSON file
OUTPUT_DIR_FOR_JSON = os.path.join(BASE_DIR, "expert_review_sheets")

# Setup basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d')
logger = logging.getLogger("aggregator")


def extract_recommendation_details(filename: str, patient_run_entry: dict) -> tuple[str, str, str, str, str]:
    """
    Extracts the recommendation type, final recommendation text, think block,
    raw response, and LLM input.
    """
    rec_type = "UnknownType"
    final_rec_text = None
    think_block_text = None
    raw_response_text = None
    llm_input_text = None # New variable for LLM input

    llm_model = patient_run_entry.get("llm_model_used", "UnknownLLM").replace(":", "_").replace("/", "_").replace(".", "_")
    modified = patient_run_entry.get("clinical_info_modified", False)
    
    if "agent_" in filename: # Heuristic for multi-agent results
        rec_type = f"MultiAgent_{llm_model}_Mod{modified}"
        final_rec_text = patient_run_entry.get("therapie_output_final")
        if final_rec_text is None: # Fallback for older multi-agent output structure
             final_rec_text = patient_run_entry.get("therapie_output")
        think_block_text = patient_run_entry.get("therapie_think_block")
        raw_response_text = patient_run_entry.get("therapie_raw_response")
        llm_input_text = patient_run_entry.get("llm_input", {}).get("prompt_text") # For multi-agent, LLM input is nested under 'llm_input'
    elif "single_prompt_" in filename: # Heuristic for single-prompt results
        rec_type = f"SinglePrompt_{llm_model}_Mod{modified}"
        final_rec_text = patient_run_entry.get("single_prompt_recommendation_final")
        raw_response_text = patient_run_entry.get("llm_raw_output_with_think")
        
        # Extract think block from raw response if not already separated
        if raw_response_text and not think_block_text:
            think_tag_start_pattern = r"<think>"
            think_tag_end_pattern = r"</think>"
            try:
                match = re.search(f"{think_tag_start_pattern}(.*?){think_tag_end_pattern}", 
                                   raw_response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    think_block_text = match.group(1).strip()
            except Exception:
                pass # Ignore if regex fails for some reason on raw text

        llm_input_data = patient_run_entry.get("llm_input")
        if llm_input_data:
            llm_input_text = llm_input_data.get("prompt_text") # For single-prompt, LLM input is directly under 'llm_input'

    else:
        rec_type = f"Other_{llm_model}_Mod{modified}"
        final_rec_text = patient_run_entry.get("single_prompt_recommendation_final") or \
                             patient_run_entry.get("therapie_output_final") or \
                             patient_run_entry.get("therapie_output")
        think_block_text = patient_run_entry.get("therapie_think_block") # Or extract if needed
        llm_input_text = patient_run_entry.get("llm_input", {}).get("prompt_text") # General fallback for LLM input

    # Ensure raw_response_text has a value if possible
    if not raw_response_text:
        raw_response_text = patient_run_entry.get("llm_raw_output_with_think") or \
                            patient_run_entry.get("therapie_raw_response") or \
                            patient_run_entry.get("diagnostik_raw_response")

    return rec_type, \
           final_rec_text if final_rec_text is not None else "", \
           think_block_text if think_block_text is not None else "", \
           raw_response_text if raw_response_text is not None else "", \
           llm_input_text if llm_input_text is not None else ""


# NEW: Function to extract sections from LLM input
def extract_llm_input_sections(llm_input_text: str) -> dict:
    """
    Extracts specific sections like system_instruction, context_info, patient_information,
    and attached_documents from the LLM input string using regex.
    """
    sections = {
        "system_instruction": "",
        "context_info": "",
        "patient_information": "",
        "attached_documents": ""
    }

    # Using '[\s\S]*?' to match any character including newlines and spaces, non-greedily
    patterns = {
        "system_instruction": r"<system_instruction>([\s\S]*?)</system_instruction>",
        "context_info": r"<context_info>([\s\S]*?)</context_info>",
        "patient_information": r"<patient_information>([\s\S]*?)</patient_information>",
        "attached_documents": r"<attached_documents>([\s\S]*?)</attached_documents>"
    }

    for section, pattern in patterns.items():
        match = re.search(pattern, llm_input_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_content = match.group(1).strip()
            
            # Special cleaning for attached_documents
            if section == "attached_documents":
                # Remove specific introductory line and surrounding whitespace
                extracted_content = re.sub(r'---\s*Inhalt von [^-\n]+\.md\s*---\s*', '', extracted_content, flags=re.IGNORECASE | re.MULTILINE)
                extracted_content = extracted_content.strip() # Re-strip for remaining whitespace
            
            sections[section] = extracted_content
    return sections


def main():
    # Create the output directory for JSON
    os.makedirs(OUTPUT_DIR_FOR_JSON, exist_ok=True)
    
    all_json_files_to_process = []
    for res_dir in RESULTS_DIRS:
        if not os.path.isdir(res_dir):
            logger.warning(f"Results directory not found: {res_dir}. Skipping.")
            continue
        all_json_files_to_process.extend(glob.glob(os.path.join(res_dir, "*.json")))
    
    if not all_json_files_to_process:
        logger.error(f"No JSON files found in any of the specified results directories: {RESULTS_DIRS}")
        return

    # Use a dictionary to store the final aggregated data
    aggregated_output_data = {} # Will hold the final structure for the JSON file

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

                    # Initialize patient entry if it doesn't exist
                    if patient_id not in aggregated_output_data:
                        aggregated_output_data[patient_id] = {
                            "patient_metadata": { # Renamed from "metadata" for clarity in output JSON
                                "patient_data_summary": patient_run_entry.get("patient_context_summary_for_eval", f"Summary N/A for {patient_id}"),
                                "data_source_file": patient_run_entry.get("patient_data_source_file", "N/A"),
                            },
                            "recommendation_variants": {}, # Renamed from "recommendations"
                            "evaluation_fields": { # New section for potential future evaluation inputs
                                "overall_best_recommendation": "",
                                "general_comments": ""
                            }
                        }
                    
                    rec_type, final_rec, think_block, raw_response, llm_input = extract_recommendation_details(filename_only, patient_run_entry)
                    
                    # Extract detailed LLM input sections
                    llm_input_parsed_sections = extract_llm_input_sections(llm_input)

                    # Only add if we have some recommendation text, raw output, or LLM input
                    if final_rec or raw_response or llm_input: 
                        aggregated_output_data[patient_id]["recommendation_variants"][rec_type] = {
                            "final_recommendation": final_rec,
                            "think_block": think_block,
                            "raw_response_with_think": raw_response,
                            "llm_input_full": llm_input, # Keep the full LLM input string
                            "llm_input_sections": llm_input_parsed_sections # Store parsed sections
                        }
                    else: # Fallback if no relevant data is found
                        aggregated_output_data[patient_id]["recommendation_variants"][rec_type] = {
                            "final_recommendation": "ERROR or N/A in source file",
                            "think_block": "",
                            "raw_response_with_think": "ERROR or N/A in source file",
                            "llm_input_full": "",
                            "llm_input_sections": {} # Default to empty dict if no sections
                        }

        except Exception as e:
            logger.error(f"Error processing file {filename_only}: {e}", exc_info=True)
            continue
            
    if not aggregated_output_data:
        logger.error("No patient data aggregated. Exiting.")
        return

    # --- Save to JSON ---
    json_file_path = os.path.join(OUTPUT_DIR_FOR_JSON, JSON_OUTPUT_FILE)
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_output_data, f, indent=4, ensure_ascii=False) # Use indent for readability
        logger.info(f"Successfully aggregated data and saved to JSON file: {json_file_path}")
    except Exception as e:
        logger.error(f"Error writing JSON file: {e}", exc_info=True)

if __name__ == "__main__":
    main()