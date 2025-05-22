# app_eval.py
import streamlit as st
from utils_eval import get_displayable_patient_case_options, get_full_case_data_for_patient_id, save_expert_evaluation_form_data
from datetime import datetime
import logging
import pandas as pd
import re # For parsing rec_type
from collections import defaultdict # If using defaultdict in get_recommendation_data_structured

# ... (logger setup, page config, title) ...
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="LLM Comparative Evaluation", layout="wide")
st.title("🧑‍⚕️ LLM Therapy Recommendation - Comparative Expert Evaluation")
st.markdown("Evaluate and compare different LLM-generated therapy recommendations for the same patient case.")
st.markdown("---")


# --- Helper function to parse rec_type_prefix and get structured data ---
# (Paste the parse_rec_type_prefix and get_recommendation_data_structured functions here)
def parse_rec_type_prefix(prefix: str) -> tuple[str | None, str | None, bool | None]:
    script_type = None; llm_model = None; is_modified = None
    parts = prefix.split('_')
    if not parts: return script_type, llm_model, is_modified

    # Try to identify script type (MultiAgent or SinglePrompt)
    if parts[0].lower() in ["multiagent", "singleprompt"]:
        script_type = parts[0]
        parts = parts[1:] # Consume the script type part
    
    # The rest could be LLM name, then ModTrue/ModFalse
    # Look for ModTrue/ModFalse from the end
    if parts and (parts[-1] == "ModTrue" or parts[-1] == "ModFalse"):
        is_modified = (parts[-1] == "ModTrue")
        llm_model = "_".join(parts[:-1]) # Everything before Mod part is LLM
    elif parts: # No Mod indicator, assume not modified, all parts are LLM
        llm_model = "_".join(parts)
        is_modified = False
    
    if not llm_model: # Fallback if parsing is difficult
        llm_model = prefix # Use full prefix if specific parsing fails
        logger.warning(f"Could not specifically parse LLM model from prefix '{prefix}'. Using full prefix.")

    # logger.debug(f"Parsed prefix '{prefix}': Script={script_type}, LLM={llm_model}, Modified={is_modified}")
    return script_type, llm_model, is_modified


def get_recommendation_data_structured(case_data_series: pd.Series) -> dict:
    structured_variants = defaultdict(lambda: {"Standard": None, "Modified": None})
    if case_data_series is None or not isinstance(case_data_series, pd.Series):
        return structured_variants

    rec_type_prefixes = set()
    for col_name in case_data_series.index:
        if isinstance(col_name, str):
            if " - Final Recommendation" in col_name:
                rec_type_prefixes.add(col_name.split(" - Final Recommendation")[0])
            elif " - Think Block" in col_name: # Ensure these suffixes are consistent
                rec_type_prefixes.add(col_name.split(" - Think Block")[0])
            elif " - Full Raw Response" in col_name:
                rec_type_prefixes.add(col_name.split(" - Full Raw Response")[0])
    
    if not rec_type_prefixes:
        logger.warning(f"No recommendation variant column prefixes found for Patient ID: {case_data_series.get('Patient ID', 'Unknown')}. Check Excel columns.")

    for prefix in sorted(list(rec_type_prefixes)):
        _script_type, llm_name_parsed, is_mod = parse_rec_type_prefix(prefix)

        if not llm_name_parsed:
            llm_name_parsed = f"UnparsedLLM_{prefix}" 
            logger.warning(f"Could not parse LLM name from prefix '{prefix}', using fallback.")

        variant_key = "Modified" if is_mod else "Standard"
        
        data_for_variant = {
            "final_text": str(case_data_series.get(f"{prefix} - Final Recommendation", "")),
            "think_block": str(case_data_series.get(f"{prefix} - Think Block", "")),
            "raw_response": str(case_data_series.get(f"{prefix} - Full Raw Response", "")),
            "full_prefix": prefix 
        }
        
        if llm_name_parsed not in structured_variants:
            structured_variants[llm_name_parsed] = {"Standard": None, "Modified": None} # Ensure both keys exist
        
        # Check if we are overwriting existing data, could indicate parsing issue or duplicate types
        if structured_variants[llm_name_parsed][variant_key] is not None:
            logger.warning(f"Potential overwrite for LLM '{llm_name_parsed}', Variant '{variant_key}'. "
                           f"Original prefix: {structured_variants[llm_name_parsed][variant_key]['full_prefix']}, "
                           f"New prefix: {prefix}. Check parsing logic or column uniqueness.")
        
        structured_variants[llm_name_parsed][variant_key] = data_for_variant
            
    return structured_variants


# --- Sidebar ---
st.sidebar.header("Case Selection & Evaluator")
if "expert_name" not in st.session_state: st.session_state.expert_name = ""
expert_name = st.sidebar.text_input("Your Name/Identifier:", value=st.session_state.expert_name, key="expert_name_input_key")
st.session_state.expert_name = expert_name

case_options = get_displayable_patient_case_options()
if not case_options:
    st.sidebar.error("No patient cases loaded. Check Excel file and path in utils_eval.py.")
    st.stop()

if "selected_patient_id" not in st.session_state or st.session_state.selected_patient_id not in case_options:
    st.session_state.selected_patient_id = list(case_options.keys())[0] if case_options else None
selected_patient_id = st.sidebar.selectbox(
    "Select Patient Case:", options=list(case_options.keys()),
    format_func=lambda x: case_options.get(x, x),
    index=list(case_options.keys()).index(st.session_state.selected_patient_id) if st.session_state.selected_patient_id in case_options else 0,
    key="patient_id_selector_key"
)
st.session_state.selected_patient_id = selected_patient_id

# --- Main Content ---
if selected_patient_id:
    case_data_series = get_full_case_data_for_patient_id(selected_patient_id)

    if case_data_series is not None:
        st.header(f"Evaluating Case for Patient ID: {case_data_series.get('Patient ID', selected_patient_id)}")
        # ... (other captions) ...
        st.markdown("---")

        st.subheader("📄 Patient Data Summary")
        patient_summary_text = str(case_data_series.get('Patient Data Summary', 'No summary provided.')) # Ensure it's a string
        st.text_area(
            "Patient Summary (Context for LLMs)",
            value=patient_summary_text, # Pass the full text
            height=300,  # Adjust height as desired; scrollbar will appear if text is longer
            disabled=True,
            key=f"summary_{selected_patient_id}"
        )
        st.markdown("---")

        # Get structured recommendation data: {LLM_Name: {"Standard": data, "Modified": data}}
        structured_recommendations = get_recommendation_data_structured(case_data_series)

        if not structured_recommendations:
            st.warning("No recommendation variants found for this patient. Check Excel column naming and parsing logic.")
        else:
            st.subheader("🤖 LLM Generated Recommendations for Comparison")

            # Iterate through each LLM group
            for llm_model_name, variants in structured_recommendations.items():
                st.markdown(f"### LLM: `{llm_model_name}`")
                
                # Use Streamlit columns to create a 2-column layout for Standard vs Modified
                col_std, col_mod = st.columns(2)

                with col_std:
                    st.markdown("**Standard Clinical Info Variant**")
                    std_data = variants.get("Standard")
                    if std_data and std_data.get("final_text", "N/A") != "N/A" and std_data.get("final_text"):
                        st.caption(f"Source Prefix: `{std_data['full_prefix']}`")
                        if std_data.get("think_block"):
                            with st.expander("Think Block (Standard)", expanded=False):
                                st.text_area(f"TB_Std_{llm_model_name}", value=std_data["think_block"], height=150, disabled=True, key=f"think_std_{llm_model_name}_{selected_patient_id}")
                        st.text_area(f"Final Rec (Standard)", value=std_data["final_text"], height=250, disabled=True, key=f"rec_std_{llm_model_name}_{selected_patient_id}")
                    else:
                        st.info("No 'Standard' variant found or recommendation is N/A.")
                
                with col_mod:
                    st.markdown("**Modified Clinical Info Variant**")
                    mod_data = variants.get("Modified")
                    if mod_data and mod_data.get("final_text", "N/A") != "N/A" and mod_data.get("final_text"):
                        st.caption(f"Source Prefix: `{mod_data['full_prefix']}`")
                        if mod_data.get("think_block"):
                            with st.expander("Think Block (Modified)", expanded=False):
                                st.text_area(f"TB_Mod_{llm_model_name}", value=mod_data["think_block"], height=150, disabled=True, key=f"think_mod_{llm_model_name}_{selected_patient_id}")
                        st.text_area(f"Final Rec (Modified)", value=mod_data["final_text"], height=250, disabled=True, key=f"rec_mod_{llm_model_name}_{selected_patient_id}")
                    else:
                        st.info("No 'Modified' variant found or recommendation is N/A.")
                st.markdown("---") # Separator after each LLM's 2x2 display

            # --- Evaluation Form ---
            if not st.session_state.expert_name:
                st.warning("Please enter your name/identifier in the sidebar to enable evaluation.")
            else:
                form_key = f"comp_eval_form_{selected_patient_id}_{st.session_state.expert_name.replace(' ', '_')}"
                with st.form(key=form_key):
                    st.subheader("📝 Your Comparative Evaluation")
                    evaluations_per_recommendation = {} # This will store {full_prefix: {eval_data}}

                    # Iterate through LLMs and their variants to create form fields
                    for llm_model_name, variants in structured_recommendations.items():
                        for variant_type, data in variants.items(): # variant_type is "Standard" or "Modified"
                            if data and data.get("final_text", "N/A") != "N/A" and data.get("final_text"): # Only create eval fields if rec exists
                                full_prefix = data["full_prefix"] # Use the original full prefix as key
                                evaluations_per_recommendation[full_prefix] = {}
                                
                                st.markdown(f"--- \n**Evaluating: `{full_prefix}` (LLM: `{llm_model_name}`, Type: `{variant_type}`)**")
                                
                                eval_options_adherence = ["N/A", "1-Not Adherent", "2-Partially (Major Issues)", "3-Partially (Minor Issues)", "4-Mostly Adherent", "5-Strongly Adherent"]
                                # ... (Define other eval_options lists as before) ...
                                eval_options_overall_single = ["N/A", "1-Poor", "2-Fair", "3-Good", "4-Excellent"]


                                evaluations_per_recommendation[full_prefix]["guideline_adherence"] = st.radio(
                                    f"A. Guideline Adherence", options=eval_options_adherence, index=0, 
                                    key=f"adh_{full_prefix}_{selected_patient_id}", horizontal=True
                                )
                                # ... Add other radio/slider inputs for B, C, D, E (overall_assessment_single_rec) ...
                                evaluations_per_recommendation[full_prefix]["clinical_correctness_safety"] = st.radio(
                                    f"B. Correctness/Safety", options=["N/A", "1-Incorrect/Unsafe", "2-Major Concerns", "3-Minor Concerns", "4-Correct & Safe"], 
                                    index=0, key=f"correct_{full_prefix}_{selected_patient_id}", horizontal=True)
                                evaluations_per_recommendation[full_prefix]["clarity_explainability"] = st.select_slider(
                                    f"C. Clarity (Justification)", options=["N/A", 1,2,3,4,5], value="N/A",
                                    key=f"clarity_{full_prefix}_{selected_patient_id}")
                                evaluations_per_recommendation[full_prefix]["completeness_justification"] = st.radio(
                                    f"D. Completeness (Justification)", options=["N/A", "Incomplete", "Mostly Complete", "Comprehensive"],
                                    index=0, key=f"complete_{full_prefix}_{selected_patient_id}", horizontal=True)
                                evaluations_per_recommendation[full_prefix]["overall_assessment_single_rec"] = st.select_slider(
                                    f"E. Overall (This Rec.)", options=eval_options_overall_single, value="N/A",
                                    key=f"overall_single_{full_prefix}_{selected_patient_id}")
                                
                                evaluations_per_recommendation[full_prefix]["comments_specific_to_rec"] = st.text_area(
                                    f"F. Comments for this specific recommendation (`{full_prefix}`)", height=75, 
                                    key=f"comments_single_{full_prefix}_{selected_patient_id}"
                                )
                    
                    st.markdown("--- \n**🏆 Overall Patient Case Evaluation**")
                    # The options for best rec should be the full_prefixes
                    displayable_prefixes = []
                    for llm_name, variants_map in structured_recommendations.items(): # variants_map is {"Standard": data, "Modified": data}
                        standard_variant = variants_map.get("Standard") # Get the dict for "Standard" variant
                        if standard_variant and standard_variant.get("full_prefix"): # Check if it exists and has 'full_prefix'
                            displayable_prefixes.append(standard_variant["full_prefix"])
                        
                        modified_variant = variants_map.get("Modified") # Get the dict for "Modified" variant
                        if modified_variant and modified_variant.get("full_prefix"): # Check if it exists and has 'full_prefix'
                            displayable_prefixes.append(modified_variant["full_prefix"])
                    # Simpler way to get prefixes if structured_recommendations is {llm: {variant_type: {data_with_full_prefix}}}
                    displayable_prefixes = []
                    for llm_name, variants_map in structured_recommendations.items():
                        if variants_map.get("Standard") and variants_map["Standard"].get("full_prefix"):
                            displayable_prefixes.append(variants_map["Standard"]["full_prefix"])
                        if variants_map.get("Modified") and variants_map["Modified"].get("full_prefix"):
                            displayable_prefixes.append(variants_map["Modified"]["full_prefix"])

                    overall_best_type_prefix = st.selectbox(
                        "Which recommendation (full prefix) do you consider BEST overall for this patient?",
                        options=["N/A"] + sorted(list(set(displayable_prefixes))), index=0, # Use the actual prefixes
                        key=f"best_rec_prefix_{selected_patient_id}"
                    )
                    general_comments = st.text_area(
                        "General comments for this patient case / overall comparison / why best was chosen:",
                        height=100, key=f"gen_comments_{selected_patient_id}"
                    )

                    submitted = st.form_submit_button("Submit Full Evaluation for this Patient")

                    if submitted:
                        full_evaluation_data_payload = {
                            "evaluations_per_recommendation_prefix": evaluations_per_recommendation, # Keyed by full_prefix
                            "overall_best_recommendation_full_prefix": overall_best_type_prefix,
                            "general_comments_for_patient": general_comments
                        }
                        success, saved_filename = save_expert_evaluation_form_data(
                            selected_patient_id, 
                            full_evaluation_data_payload, 
                            st.session_state.expert_name
                        )
                        if success:
                            st.success(f"Evaluation submitted successfully! Saved as: {saved_filename}")
                            st.balloons()
                        else:
                            st.error("Failed to save evaluation. Check console/logs.")
    else:
        st.error(f"Could not load data for selected patient ID: {selected_patient_id}")
else:
    st.info("Select a patient ID from the sidebar to begin evaluation.")