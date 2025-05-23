# app_eval.py
import streamlit as st
from utils_eval import (
    get_patient_ids_for_selection, 
    get_case_data_for_patient, 
    get_available_llm_models_for_patient,
    save_comparative_evaluation,
    check_if_evaluated,
    PATIENT_DATA
)
from datetime import datetime
import logging
import pandas as pd
from collections import defaultdict
from data_loader import load_patient_data

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="LLM Comparative Evaluation", page_icon="🧬", layout="wide")
st.title("LLM Tumorboard Recommendation - Comparative Expert Evaluation")
st.markdown("---")

if 'df_patients' not in st.session_state:
    st.session_state.df_patients = load_patient_data(PATIENT_DATA)

df_patients = st.session_state.df_patients

def display_patient_information(df_patients, patient_id):
    st.subheader("Patienteninformationen")
    patient_row = df_patients[df_patients["Patient ID"] == patient_id]
    if patient_row.empty:
        st.error("Keine Patientendaten gefunden.")
        return

    patient_data = patient_row.iloc[0]
    for col, val in patient_data.items():
        st.markdown(f"**{col}:** {val}")

def parse_rec_type_prefix(prefix: str) -> tuple[str | None, str | None, bool | None]:
    script_type = None; llm_model = None; is_modified = None
    parts = prefix.split('_')
    if not parts: return script_type, llm_model, is_modified
    if parts[0].lower() in ["multiagent", "singleprompt"]:
        script_type = parts[0]; parts = parts[1:]
    else: script_type = "UnknownScript" # Default if not explicitly SinglePrompt or MultiAgent

    if parts and (parts[-1] == "ModTrue" or parts[-1] == "ModFalse"):
        is_modified = (parts[-1] == "ModTrue")
        llm_model = "_".join(parts[:-1])
    elif parts:
        llm_model = "_".join(parts)
        is_modified = False
    if not llm_model: llm_model = prefix
    return script_type, llm_model, is_modified

def get_recommendation_data_structured(case_data_series: pd.Series | None) -> dict:
    structured_variants = defaultdict(lambda: {"Standard": None, "Modified": None})
    if case_data_series is None or not isinstance(case_data_series, pd.Series):
        return structured_variants
    rec_type_prefixes = set()
    for col_name in case_data_series.index:
        if isinstance(col_name, str):
            for suffix in [" - Final Recommendation", " - Think Block", " - Full Raw Response"]:
                if suffix in col_name:
                    rec_type_prefixes.add(col_name.split(suffix)[0])
                    break 
    for prefix in sorted(list(rec_type_prefixes)):
        _script_type, llm_name_parsed, is_mod = parse_rec_type_prefix(prefix)
        if not llm_name_parsed: llm_name_parsed = f"UnparsedLLM_{prefix}"
        variant_key = "Modified" if is_mod else "Standard"
        data_for_variant = {
            "final_text": str(case_data_series.get(f"{prefix} - Final Recommendation", "")),
            "think_block": str(case_data_series.get(f"{prefix} - Think Block", "")),
            "raw_response": str(case_data_series.get(f"{prefix} - Full Raw Response", "")),
            "full_prefix": prefix 
        }
        if llm_name_parsed not in structured_variants:
            structured_variants[llm_name_parsed] = {"Standard": None, "Modified": None}
        if structured_variants[llm_name_parsed][variant_key] is not None: # Check for overwrite
             logger.warning(f"Potential overwrite for LLM '{llm_name_parsed}', Variant '{variant_key}' from prefix '{prefix}'. "
                            f"Previous prefix: {structured_variants[llm_name_parsed][variant_key]['full_prefix']}")
        structured_variants[llm_name_parsed][variant_key] = data_for_variant
    return structured_variants

def render_evaluation_widgets(variant_prefix: str, patient_id: str, storage_dict: dict, form_key: str):
    key_suffix = f"{variant_prefix}_{patient_id}_{form_key}"
    st.markdown(f"**Evaluation for: `{variant_prefix}`**")
    eval_options_overall_single = ["N/A", "1-Poor", "2-Fair", "3-Good", "4-Excellent"]
    storage_dict[variant_prefix]["overall_assessment_single_rec"] = st.radio(
        f"Overall Assessment", 
        options=eval_options_overall_single, 
        index=0, 
        key=f"overall_s_{key_suffix}",
        horizontal=True
    )
    st.markdown("---")

# --- Helper Function to get recommendation data for a specific variant ---
def get_variant_data(case_data_series: pd.Series, llm_model: str, script_type: str, is_modified: bool) -> dict | None:
    """
    Constructs the expected column prefix and fetches data for a specific variant.
    """
    mod_suffix = "ModTrue" if is_modified else "ModFalse"
   
    full_prefix_found = None
    for col_name in case_data_series.index:
        if isinstance(col_name, str) and " - Final Recommendation" in col_name:
            prefix_candidate = col_name.split(" - Final Recommendation")[0]

            if llm_model in prefix_candidate and script_type.lower() in prefix_candidate.lower() and mod_suffix.lower() in prefix_candidate.lower():
                full_prefix_found = prefix_candidate
                break
    
    if not full_prefix_found:
        expected_prefix_pattern_start = f"{script_type}_{llm_model}"
        expected_prefix_pattern_end = f"{mod_suffix}"
        
        for col_prefix in case_data_series.index:
             if isinstance(col_prefix, str) and col_prefix.startswith(expected_prefix_pattern_start) and col_prefix.endswith(expected_prefix_pattern_end):
                  if " - Final Recommendation" in col_prefix: # Check it's a base prefix
                       full_prefix_found = col_prefix.split(" - Final Recommendation")[0]
                       break
        
        if not full_prefix_found:
            logger.warning(f"Could not find matching full_prefix for LLM:{llm_model}, Script:{script_type}, Mod:{is_modified}")
            return None

    data = {
        "full_prefix": full_prefix_found,
        "final_text": str(case_data_series.get(f"{full_prefix_found} - Final Recommendation", "")),
        "think_block": str(case_data_series.get(f"{full_prefix_found} - Think Block", "")),
        "raw_response": str(case_data_series.get(f"{full_prefix_found} - Full Raw Response", "")),
    }
    if not data["final_text"] and not data["raw_response"]: # No useful data
        return None
    return data

# --- Session State Initialization ---
if "expert_name" not in st.session_state: st.session_state.expert_name = ""
if "selected_patient_id" not in st.session_state: st.session_state.selected_patient_id = None
if "selected_llm_model" not in st.session_state: st.session_state.selected_llm_model = None

# --- Sidebar ---
st.sidebar.header("Controls")
st.session_state.expert_name = st.sidebar.text_input("Your Name/Identifier:", value=st.session_state.expert_name)

patient_id_options = get_patient_ids_for_selection()
if not patient_id_options:
    st.sidebar.error("No patient cases loaded from Excel.")
    st.stop()

# Initialize selected_patient_id if it's None or not in options
if st.session_state.selected_patient_id is None or st.session_state.selected_patient_id not in patient_id_options:
    st.session_state.selected_patient_id = patient_id_options[0]

selected_patient_id = st.sidebar.selectbox(
    "1. Select Patient Case:", options=patient_id_options,
    index=patient_id_options.index(st.session_state.selected_patient_id),
    key="patient_id_selector" # Give it a stable key
)
st.session_state.selected_patient_id = selected_patient_id # Update session state

# --- Load Case Data for Selected Patient ---
case_data_series = None
llm_model_options = []
if selected_patient_id:
    case_data_series = get_case_data_for_patient(selected_patient_id)
    if case_data_series is not None:
        llm_model_options = get_available_llm_models_for_patient(case_data_series)
        if not llm_model_options:
            st.sidebar.warning(f"No LLM variants found for patient {selected_patient_id}. Check Excel data and parsing.")
    else:
        st.sidebar.error(f"Could not load data for patient {selected_patient_id}.")
        st.stop()

if not llm_model_options:
    st.sidebar.info("Select a patient to see available LLM models.")
    st.stop()

# Initialize selected_LLM if it's None or not in new options
if st.session_state.selected_llm_model is None or st.session_state.selected_llm_model not in llm_model_options:
    st.session_state.selected_llm_model = llm_model_options[0] if llm_model_options else None

selected_llm_model = st.sidebar.selectbox(
    "2. Select LLM Model to Evaluate:", options=llm_model_options,
    index=llm_model_options.index(st.session_state.selected_llm_model) if st.session_state.selected_llm_model in llm_model_options else 0,
    key="llm_model_selector" # Stable key
)
st.session_state.selected_llm_model = selected_llm_model

# --- Display Already Evaluated Status ---
already_evaluated_this_llm = False
if selected_patient_id and selected_llm_model and st.session_state.expert_name:
    already_evaluated_this_llm = check_if_evaluated(selected_patient_id, selected_llm_model, st.session_state.expert_name)
    if already_evaluated_this_llm:
        st.sidebar.success(f"✅ You have already submitted an evaluation for Patient {selected_patient_id} with LLM {selected_llm_model}.")
    else:
        st.sidebar.info(f"➡️ Evaluate Patient {selected_patient_id} with LLM {selected_llm_model}.")


# --- Main Content Display & Evaluation Form ---
if selected_patient_id and selected_llm_model and case_data_series is not None:
    st.header(f"Patient ID: {selected_patient_id} - Evaluating LLM: `{selected_llm_model}`")
    st.subheader("📄 Patient Data Summary")
    # Patient Data
    patient_row = df_patients[df_patients["ID"] == selected_patient_id]
    if not patient_row.empty:
        patient_data = patient_row.iloc[0]
        patient_summary_text = "\n".join([f"{col}: {patient_data[col]}" for col in patient_row.columns])
    else:
        patient_summary_text = "Keine Patientendaten gefunden."

    st.text_area(
        "Patient Summary",
        value=patient_summary_text,
        height=300,
        disabled=False,
        key=f"summary_{selected_patient_id}"
    )
    st.markdown("---")

    # Clinical Info
    clinical_info_row = df_patients[df_patients["ID"] == selected_patient_id]
    if not clinical_info_row.empty:
        clinical_info = clinical_info_row.iloc[0]["clinical_info"]
    else:
        clinical_info = "Keine clinical_info Daten"

    st.text_area(
        "Clinical Info",
        value=clinical_info,
        height=100,
        disabled=False,
        key=f"clinical_info_{selected_patient_id}"
    )

    st.markdown("---")
    # Therapieempfehlung Ärzte
    therapy_doctor_row = df_patients[df_patients["ID"] == selected_patient_id]
    if not therapy_doctor_row.empty:
        therapy_doctor = therapy_doctor_row.iloc[0]["Beurteilung_und_Therapieempfehlung"]
    else:
        therapy_doctor = "Keine Beurteilung und Therapieempfehlung Daten"

    st.text_area(
        "Beurteilung und Therapieempfehlung",
        value=therapy_doctor,
        height=100,
        disabled=False,
        key=f"therapy_doctor_{selected_patient_id}"
    )

    st.markdown("---")
    sp_std_data = get_variant_data(case_data_series, selected_llm_model, "SinglePrompt", False)
    sp_mod_data = get_variant_data(case_data_series, selected_llm_model, "SinglePrompt", True)
    ma_std_data = get_variant_data(case_data_series, selected_llm_model, "MultiAgent", False)
    ma_mod_data = get_variant_data(case_data_series, selected_llm_model, "MultiAgent", True)

    # This dictionary will store the expert's input collected inside the form
    form_eval_data_storage = {} 

    # --- THE SINGLE FORM STARTS HERE ---
    form_key = f"eval_form_comparative_{selected_patient_id}_{selected_llm_model}_{st.session_state.expert_name.replace(' ', '_')}"
    with st.form(key=form_key):
        st.subheader(f"LLM: `{selected_llm_model}` - Recommendation Variants & Your Evaluation")

        # --- ROW 1: SINGLE-PROMPT ---
        st.markdown("#### Single-Prompt Approach")
        col_sp_std_disp, col_sp_mod_disp = st.columns(2) # Columns for displaying recommendations
        
        with col_sp_std_disp:
            st.markdown("**Standard Clinical Info**")
            if sp_std_data and sp_std_data.get("final_text"):
                st.caption(f"ID: `{sp_std_data['full_prefix']}`")
                with st.expander("Think Block", expanded=False): st.text_area("SP_Std_Think_Display", sp_std_data['think_block'], height=300, disabled=True, key=f"dsp_think_sp_std_{selected_patient_id}")
                st.text_area("SP_Std_Rec_Display", sp_std_data['final_text'], height=300, disabled=True, key=f"dsp_rec_sp_std_{selected_patient_id}")
                form_eval_data_storage[sp_std_data['full_prefix']] = {}
                render_evaluation_widgets(sp_std_data['full_prefix'], selected_patient_id, form_eval_data_storage, form_key)
            else: st.info("N/A or no recommendation text.")

        with col_sp_mod_disp:
            st.markdown("**Modified Clinical Info**")
            if sp_mod_data and sp_mod_data.get("final_text"):
                st.caption(f"ID: `{sp_mod_data['full_prefix']}`")
                with st.expander("Think Block", expanded=False): st.text_area("SP_Mod_Think_Display", sp_mod_data['think_block'], height=300, disabled=True, key=f"dsp_think_sp_mod_{selected_patient_id}")
                st.text_area("SP_Mod_Rec_Display", sp_mod_data['final_text'], height=300, disabled=True, key=f"dsp_rec_sp_mod_{selected_patient_id}")
                form_eval_data_storage[sp_mod_data['full_prefix']] = {}
                render_evaluation_widgets(sp_mod_data['full_prefix'], selected_patient_id, form_eval_data_storage, form_key)
            else: st.info("N/A or no recommendation text.")
        st.markdown("<br>", unsafe_allow_html=True)

        # --- ROW 2: MULTI-AGENT ---
        st.markdown("#### Multi-Agent Approach")
        col_ma_std_disp, col_ma_mod_disp = st.columns(2)

        with col_ma_std_disp:
            st.markdown("**Standard Clinical Info**")
            if ma_std_data and ma_std_data.get("final_text"):
                st.caption(f"ID: `{ma_std_data['full_prefix']}`")
                with st.expander("Think Block", expanded=False): st.text_area("MA_Std_Think_Display", ma_std_data['think_block'], height=300, disabled=True, key=f"dsp_think_ma_std_{selected_patient_id}")
                st.text_area("MA_Std_Rec_Display", ma_std_data['final_text'], height=300, disabled=True, key=f"dsp_rec_ma_std_{selected_patient_id}")
                form_eval_data_storage[ma_std_data['full_prefix']] = {}
                render_evaluation_widgets(ma_std_data['full_prefix'], selected_patient_id, form_eval_data_storage, form_key)
            else: st.info("N/A or no recommendation text.")

        with col_ma_mod_disp:
            st.markdown("**Modified Clinical Info**")
            if ma_mod_data and ma_mod_data.get("final_text"):
                st.caption(f"ID: `{ma_mod_data['full_prefix']}`")
                with st.expander("Think Block", expanded=False): st.text_area("MA_Mod_Think_Display", ma_mod_data['think_block'], height=300, disabled=True, key=f"dsp_think_ma_mod_{selected_patient_id}")
                st.text_area("MA_Mod_Rec_Display", ma_mod_data['final_text'], height=300, disabled=True, key=f"dsp_rec_ma_mod_{selected_patient_id}")
                form_eval_data_storage[ma_mod_data['full_prefix']] = {}
                render_evaluation_widgets(ma_mod_data['full_prefix'], selected_patient_id, form_eval_data_storage, form_key)
            else: st.info("N/A or no recommendation text.")
        
        st.markdown(f"\n**💬 Overall Comments for this LLM's ({selected_llm_model}) Performance on Patient {selected_patient_id}**")
        # This input widget must also be inside the form
        overall_comments_for_llm_patient_input = st.text_area(
            "General comments:",
            height=100, key=f"gen_comments_llm_pat_{selected_patient_id}_{selected_llm_model}"
        )

        # Submit button for the entire form
        submitted_form = st.form_submit_button(f"Submit All Evaluations for LLM: {selected_llm_model} on Patient: {selected_patient_id}")

        if submitted_form:
            # Check if expert name is provided (though the form wouldn't render if not, this is a double check)
            if not st.session_state.expert_name:
                 st.error("Please enter your Name/Identifier in the sidebar to submit an evaluation.")
            # Check if there was anything to evaluate (at least one variant was displayed and had widgets rendered)
            elif not form_eval_data_storage: 
                st.error("No recommendation variants were found or displayed for evaluation. Cannot submit.")
            else:
                evaluation_payload = {
                    # "patient_id_evaluated": selected_patient_id,
                    # "llm_model_evaluated": selected_llm_model,
                    # "expert_name": st.session_state.expert_name,
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "evaluations_per_variant": form_eval_data_storage, 
                    "overall_comments_for_llm_patient": overall_comments_for_llm_patient_input
                }
                success, saved_filename = save_comparative_evaluation(
                    selected_patient_id,
                    selected_llm_model, 
                    evaluation_payload, 
                    st.session_state.expert_name
                )
                if success:
                    st.success(f"Evaluation submitted successfully! Saved as: {saved_filename}")
                    st.balloons()
                    st.rerun() 
                else:
                    st.error("Failed to save evaluation. Check console/logs.")

else:
    st.info("Select a Patient ID and LLM Model from the sidebar to begin.")