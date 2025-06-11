import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict
import re # Importiere das re-Modul für reguläre Ausdrücke

# Importiere deine eigenen Module
from utils_eval_net import (
    get_patient_ids_for_selection,
    get_case_data_for_patient,
    get_available_llm_models_for_patient,
    save_comparative_evaluation,
    check_if_evaluated,
    PATIENT_DATA
)
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
if df_patients is not None:
    df_patients = df_patients.dropna(axis=1, how='all')
    df_patients = df_patients.loc[:, ~df_patients.columns.str.contains('^Unnamed')]

if df_patients is None:
    st.error("Patientendaten konnten nicht geladen werden. Bitte prüfe die Datenquelle.")
    st.stop()

def get_patient_summary_text(df_patients, patient_id):
    patient_row = df_patients[df_patients["ID"].astype(str) == str(patient_id)]
    if not patient_row.empty:
        patient_data = patient_row.iloc[0]
        return "\n".join([f"{col}: {patient_data[col]}" for col in patient_row.columns])
    else:
        return "Keine Patientendaten gefunden."

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
            for suffix in [" - Final Recommendation", " - Think Block", " - Full Raw Response", " - LLM Input"]:
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
            "llm_input": str(case_data_series.get(f"{prefix} - LLM Input", "")),
            "full_prefix": prefix 
        }
        if llm_name_parsed not in structured_variants:
            structured_variants[llm_name_parsed] = {"Standard": None, "Modified": None}
        if structured_variants[llm_name_parsed][variant_key] is not None: # Check for overwrite
             logger.warning(f"Potential overwrite for LLM '{llm_name_parsed}', Variant '{variant_key}' from prefix '{prefix}'. "
                            f"Previous prefix: {structured_variants[llm_name_parsed][variant_key]['full_prefix']}")
        structured_variants[llm_name_parsed][variant_key] = data_for_variant
    return structured_variants

def render_evaluation_widgets(variant_prefix: str, patient_id: str, storage_dict: dict, form_key: str): # form_key is passed in but not used here for element keys
    key_suffix = f"{variant_prefix}_{patient_id}_{st.session_state.expert_name.replace(' ', '_')}" # Use expert_name directly
    
    st.markdown(f"**Evaluation for: `{variant_prefix}`**")
    eval_options_overall_single = ["N/A", "1-Poor", "2-Fair", "3-Good", "4-Excellent"]
    storage_dict[variant_prefix]["overall_assessment_single_rec"] = st.radio(
        f"Overall Assessment", 
        options=eval_options_overall_single, 
        index=0, 
        key=f"overall_s_{key_suffix}", # This should now be unique
        horizontal=True
    )
    st.markdown("---")

def clean_newlines(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace("\\n", "\n").replace("\\\\n", "\n")

def get_variant_data(case_data: dict, base_llm_model_name: str, script_type: str, is_modified: bool) -> dict | None:
    """
    Konstruiert das erwartete Spaltenpräfix und holt Daten für eine spezifische Variante
    aus dem 'recommendation_variants'-Unter-Dictionary des Falles.
    """
    if case_data is None:
        logger.warning("case_data is None in get_variant_data.")
        return None

    recommendation_variants_data = case_data.get("recommendation_variants")
    if not isinstance(recommendation_variants_data, dict):
        logger.error(f"Expected 'recommendation_variants' to be a dictionary, but got {type(recommendation_variants_data)} for patient.")
        return None

    mod_suffix = "ModTrue" if is_modified else "ModFalse"
    
    expected_full_prefix = f"{script_type}_{base_llm_model_name}_{mod_suffix}"
    
    if expected_full_prefix in recommendation_variants_data:
        variant_content = recommendation_variants_data[expected_full_prefix]

        data = {
            "full_prefix": expected_full_prefix, # Behalte das volle Präfix bei
            "final_text": str(variant_content.get("final_recommendation", "")),
            "think_block": str(variant_content.get("think_block", "")),
            "raw_response": str(variant_content.get("raw_response_with_think", "")), # Beachte raw_response_with_think
            "llm_input": str(variant_content.get("llm_input_full", "")), # Beachte llm_input_full
        }
        
        if not data["final_text"] and not data["raw_response"]: # Keine nützlichen Daten
            logger.info(f"No useful data (final_text or raw_response) for variant {expected_full_prefix}.")
            return None
        return data
    else:
        logger.warning(f"Could not find matching full_prefix: '{expected_full_prefix}' for LLM:{base_llm_model_name}, Script:{script_type}, Mod:{is_modified}")
        return None

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

    # Regex patterns for each section
    patterns = {
        "system_instruction": r"<system_instruction>(.*?)</system_instruction>",
        "context_info": r"<context_info>(.*?)</context_info>",
        "patient_information": r"<patient_information>(.*?)</patient_information>",
        "attached_documents": r"<attached_documents>(.*?)</attached_documents>"
    }

    for section, pattern in patterns.items():
        match = re.search(pattern, llm_input_text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section] = match.group(1).strip()
    return sections


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
    case_data_for_patient_dict = get_case_data_for_patient(selected_patient_id)
    if case_data_for_patient_dict is not None:
        case_data_series = pd.Series(case_data_for_patient_dict)
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
if selected_patient_id and selected_llm_model and case_data_for_patient_dict is not None: # Use case_data_for_patient_dict here
    st.header(f"Patient ID: {selected_patient_id} - Evaluating LLM: `{selected_llm_model}`")
    st.subheader("📄 Patient Data Summary")
    # Patient Data
    patient_summary_text = get_patient_summary_text(df_patients, selected_patient_id)
    st.text_area(
        "Patient Summary",
        value=patient_summary_text,
        height=300,
        disabled=False,
        key=f"summary_{selected_patient_id}"
    )
    st.markdown("---")

    # Clinical Info
    clinical_info_row = df_patients[df_patients["ID"].astype(str).str.strip() == str(selected_patient_id).strip()]     
    if not clinical_info_row.empty:
        clinical_info = clinical_info_row.iloc[0]["Fragestellung"]
    else:
        clinical_info = "Keine Fragestellung"
    st.text_area(
        "Fragestellung",
        value=clinical_info,
        height=100,
        disabled=False,
        key=f"clinical_info_{selected_patient_id}"
    )

    st.markdown("---")
    # Therapieempfehlung Ärzte
    therapy_doctor_row = df_patients[df_patients["ID"].astype(str).str.strip() == selected_patient_id]
    if not therapy_doctor_row.empty:
        therapy_doctor = therapy_doctor_row.iloc[0]["Empfehlung"]
    else:
        therapy_doctor = "Keine Beurteilung und Therapieempfehlung Daten"

    st.text_area(
        "Beurteilung und Therapieempfehlung (wird LLM nicht mitgegeben)",
        value=therapy_doctor,
        height=100,
        disabled=False,
        key=f"therapy_doctor_{selected_patient_id}"
    )

    st.markdown("---")
    # Make sure to pass the dictionary, not the Series, to get_variant_data
    sp_std_data = get_variant_data(case_data_for_patient_dict, selected_llm_model, "SinglePrompt", False)
    sp_mod_data = get_variant_data(case_data_for_patient_dict, selected_llm_model, "SinglePrompt", True)
    ma_std_data = get_variant_data(case_data_for_patient_dict, selected_llm_model, "MultiAgent", False)
    ma_mod_data = get_variant_data(case_data_for_patient_dict, selected_llm_model, "MultiAgent", True)

    form_eval_data_storage = {} 

       # --- THE SINGLE FORM STARTS HERE ---
    form_key = f"eval_form_comparative_{selected_patient_id}_{selected_llm_model.replace(' ', '_')}_{st.session_state.expert_name.replace(' ', '_')}" # Ensure selected_llm_model is safe for key
    with st.form(key=form_key):
        st.subheader(f"LLM: `{selected_llm_model}` - Recommendation Variants & Your Evaluation")

        # --- ROW 1: SINGLE-PROMPT ---
        st.markdown("#### Single-Prompt Approach")
        col_sp_std_disp, col_sp_mod_disp = st.columns(2) # Columns for displaying recommendations
        
        with col_sp_std_disp:
            st.markdown(f"**Standard 'Fragestellung'**")
            if sp_std_data and (sp_std_data.get("final_text") or sp_std_data.get("think_block") or sp_std_data.get("llm_input")):
                st.caption(f"ID: `{sp_std_data['full_prefix']}`")
                
                # LLM Input
                if sp_std_data.get("llm_input"):
                    with st.expander("LLM Input", expanded=False): 
                        llm_input_sections = extract_llm_input_sections(sp_std_data['llm_input'])
                        
                        if llm_input_sections["system_instruction"]:
                            st.markdown("**System Instruction:**")
                            st.markdown(f"```xml\n{llm_input_sections['system_instruction']}\n```", unsafe_allow_html=True)
                        if llm_input_sections["context_info"]:
                            st.markdown("**Context Info:**")
                            st.markdown(f"```xml\n{llm_input_sections['context_info']}\n```", unsafe_allow_html=True)
                        if llm_input_sections["patient_information"]:
                            st.markdown("**Patient Information:**")
                            st.markdown(f"```xml\n{llm_input_sections['patient_information']}\n```", unsafe_allow_html=True)
                    if llm_input_sections["attached_documents"]:
                        with st.expander("Attached Documents", expanded=False): # NOT NESTED anymore
                            st.markdown(f"```xml\n{llm_input_sections['attached_documents']}\n```", unsafe_allow_html=True)
                    else:
                        st.info("No Attached Documents for this variant.")

                else:
                    st.info("No LLM Input for this variant.")

                # Think Block (formatted with HTML for better rendering of XML-like tags)
                if sp_std_data.get("think_block"):
                    with st.expander("Think Block", expanded=False):
                        # Use st.markdown with unsafe_allow_html=True for formatting with XML tags
                        st.markdown(f"```xml\n{sp_std_data['think_block']}\n```", unsafe_allow_html=True)
                else:
                    st.info("No Think Block for this variant.")

                st.text_area("SP_Std_Rec_Display", sp_std_data['final_text'], height=300, disabled=True, key=f"dsp_rec_sp_std_{selected_patient_id}")
                form_eval_data_storage[sp_std_data['full_prefix']] = {}
                render_evaluation_widgets(sp_std_data['full_prefix'], selected_patient_id, form_eval_data_storage, form_key)
            else: st.info("N/A or no recommendation text.")

        with col_sp_mod_disp:
            st.markdown(f"**Modified 'Fragestellung'**")
            if sp_mod_data and (sp_mod_data.get("final_text") or sp_mod_data.get("think_block") or sp_mod_data.get("llm_input")):
                st.caption(f"ID: `{sp_mod_data['full_prefix']}`")
                
                # LLM Input
                if sp_mod_data.get("llm_input"):
                    with st.expander("LLM Input", expanded=False):
                        llm_input_sections = extract_llm_input_sections(sp_mod_data['llm_input'])
                        
                        if llm_input_sections["system_instruction"]:
                            st.markdown("**System Instruction:**")
                            st.markdown(f"```xml\n{llm_input_sections['system_instruction']}\n```", unsafe_allow_html=True)
                        if llm_input_sections["context_info"]:
                            st.markdown("**Context Info:**")
                            st.markdown(f"```xml\n{llm_input_sections['context_info']}\n```", unsafe_allow_html=True)
                        if llm_input_sections["patient_information"]:
                            st.markdown("**Patient Information:**")
                            st.markdown(f"```xml\n{llm_input_sections['patient_information']}\n```", unsafe_allow_html=True)
                        
                    if llm_input_sections["attached_documents"]:
                        with st.expander("Attached Documents", expanded=False):
                            st.markdown(f"```xml\n{llm_input_sections['attached_documents']}\n```", unsafe_allow_html=True)
                    else:
                        st.info("No Attached Documents for this variant.")
                else:
                    st.info("No LLM Input for this variant.")

                # Think Block
                if sp_mod_data.get("think_block"):
                    with st.expander("Think Block", expanded=False):
                        st.markdown(f"```xml\n{sp_mod_data['think_block']}\n```", unsafe_allow_html=True)
                else:
                    st.info("No Think Block for this variant.")

                st.text_area("SP_Mod_Rec_Display", sp_mod_data['final_text'], height=300, disabled=True, key=f"dsp_rec_sp_mod_{selected_patient_id}")
                form_eval_data_storage[sp_mod_data['full_prefix']] = {}
                render_evaluation_widgets(sp_mod_data['full_prefix'], selected_patient_id, form_eval_data_storage, form_key)
            else: st.info("N/A or no recommendation text.")
        st.markdown("<br>", unsafe_allow_html=True)

        # --- ROW 2: MULTI-AGENT ---
        st.markdown("#### Multi-Agent Approach")
        col_ma_std_disp, col_ma_mod_disp = st.columns(2)

        with col_ma_std_disp:
            st.markdown(f"**Standard 'Fragestellung'**")
            if ma_std_data and (ma_std_data.get("final_text") or ma_std_data.get("think_block") or ma_std_data.get("llm_input")):
                st.caption(f"ID: `{ma_std_data['full_prefix']}`")
                
                # LLM Input
                if ma_std_data.get("llm_input"):
                    with st.expander("LLM Input", expanded=False):
                        llm_input_sections = extract_llm_input_sections(ma_std_data['llm_input'])

                        if llm_input_sections["system_instruction"]:
                            st.markdown("**System Instruction:**")
                            st.markdown(f"```xml\n{llm_input_sections['system_instruction']}\n```", unsafe_allow_html=True)
                        if llm_input_sections["context_info"]:
                            st.markdown("**Context Info:**")
                            st.markdown(f"```xml\n{llm_input_sections['context_info']}\n```", unsafe_allow_html=True)
                        if llm_input_sections["patient_information"]:
                            st.markdown("**Patient Information:**")
                            st.markdown(f"```xml\n{llm_input_sections['patient_information']}\n```", unsafe_allow_html=True)
                        
                    if llm_input_sections["attached_documents"]:
                        with st.expander("Attached Documents", expanded=False): # NOT NESTED anymore
                            st.markdown(f"```xml\n{llm_input_sections['attached_documents']}\n```", unsafe_allow_html=True)
                    else:
                        st.info("No Attached Documents for this variant.")
                else:
                    st.info("No LLM Input for this variant.")

                # Think Block
                if ma_std_data.get("think_block"):
                    with st.expander("Think Block", expanded=False):
                        st.markdown(f"```xml\n{ma_std_data['think_block']}\n```", unsafe_allow_html=True)
                else:
                    st.info("No Think Block for this variant.")

                st.text_area(
                    "MA_Std_Rec_Display",
                    clean_newlines(ma_std_data['final_text']),
                    height=300,
                    disabled=True,
                    key=f"dsp_rec_ma_std_{selected_patient_id}"
                )
                form_eval_data_storage[ma_std_data['full_prefix']] = {}
                render_evaluation_widgets(ma_std_data['full_prefix'], selected_patient_id, form_eval_data_storage, form_key)
            else: st.info("N/A or no recommendation text.")

        with col_ma_mod_disp:
            st.markdown(f"**Modified 'Fragestellung'**")
            if ma_mod_data and (ma_mod_data.get("final_text") or ma_mod_data.get("think_block") or ma_mod_data.get("llm_input")):
                st.caption(f"ID: `{ma_mod_data['full_prefix']}`")
                
                # LLM Input
                if ma_mod_data.get("llm_input"):
                    with st.expander("LLM Input", expanded=False):
                        llm_input_sections = extract_llm_input_sections(ma_mod_data['llm_input'])
                        
                        if llm_input_sections["system_instruction"]:
                            st.markdown("**System Instruction:**")
                            st.markdown(f"```xml\n{llm_input_sections['system_instruction']}\n```", unsafe_allow_html=True)
                        if llm_input_sections["context_info"]:
                            st.markdown("**Context Info:**")
                            st.markdown(f"```xml\n{llm_input_sections['context_info']}\n```", unsafe_allow_html=True)
                        if llm_input_sections["patient_information"]:
                            st.markdown("**Patient Information:**")
                            st.markdown(f"```xml\n{llm_input_sections['patient_information']}\n```", unsafe_allow_html=True)
                        
                    if llm_input_sections["attached_documents"]:
                        with st.expander("Attached Documents", expanded=False): # NOT NESTED anymore
                            st.markdown(f"```xml\n{llm_input_sections['attached_documents']}\n```", unsafe_allow_html=True)
                    else:
                        st.info("No Attached Documents for this variant.")
                else:
                    st.info("No LLM Input for this variant.")

                # Think Block
                if ma_mod_data.get("think_block"):
                    with st.expander("Think Block", expanded=False):
                        st.markdown(f"```xml\n{ma_mod_data['think_block']}\n```", unsafe_allow_html=True)
                else:
                    st.info("No Think Block for this variant.")

                st.text_area(
                    "MA_Mod_Rec_Display",
                    clean_newlines(ma_mod_data['final_text']),
                    height=300,
                    disabled=True,
                    key=f"dsp_rec_ma_mod_{selected_patient_id}"
                )
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
                    st.success(f"Evaluation submitted successfully! Saved as: **{saved_filename}**")
                    st.balloons()
                    st.rerun() 
                else:
                    st.error("Failed to save evaluation. Check console/logs.")

else:
    st.info("Select a Patient ID and LLM Model from the sidebar to begin.")