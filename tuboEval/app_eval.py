# app_eval.py
import streamlit as st
from utils_eval import get_cases_for_evaluation, load_case_data, save_evaluation

st.set_page_config(page_title="LLM Therapy Recommendation Evaluation", layout="wide")

st.title("Human Expert Evaluation Tool for LLM Therapy Recommendations")
st.markdown("---")

# --- Sidebar for Case Selection and Expert Info ---
st.sidebar.header("Case Selection & Evaluator")
expert_name = st.sidebar.text_input("Your Name/Identifier:", key="expert_name_input")

available_cases = get_cases_for_evaluation()
if not available_cases:
    st.sidebar.warning("No cases found in 'data_for_evaluation' directory.")
    st.stop()

selected_case_file = st.sidebar.selectbox(
    "Select Case to Evaluate:",
    options=available_cases,
    key="case_selector"
)

# --- Load and Display Case Data ---
if selected_case_file:
    case_data = load_case_data(selected_case_file)

    if case_data:
        st.header(f"Evaluating Case: {case_data.get('case_id_for_eval_tool', selected_case_file)}")
        st.caption(f"Linked LLM Interaction ID (for CSV): {case_data.get('llm_interaction_id', 'N/A')}") 
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patient Context (Summary provided to LLM)")
            st.text_area(
                "Patient Context",
                value=case_data.get('patient_context_summary', 'No context provided.'),
                height=250,
                disabled=True,
                key=f"context_{case_data.get('case_id')}"
            )

        with col2:
            st.subheader("LLM Generated Recommendation & Justification")
            st.text_area(
                "LLM Output",
                value=case_data.get('llm_full_recommendation', 'No recommendation provided.'),
                height=400, # Taller to see more text
                disabled=True,
                key=f"llm_rec_{case_data.get('case_id')}"
            )
        
        st.markdown("---")
        st.subheader("Your Evaluation")

        if not expert_name:
            st.warning("Please enter your name/identifier in the sidebar to enable evaluation.")
        else:
            # Unique key for the form based on case and expert to reset on case change
            form_key = f"eval_form_{case_data.get('case_id')}_{expert_name.replace(' ', '_')}"
            
            with st.form(key=form_key):
                eval_data = {} # To store expert's input

                st.markdown("**A. Guideline Adherence:**")
                eval_data["guideline_adherence"] = st.radio(
                    "How well does the recommendation adhere to the specified guideline?",
                    options=[
                        "Strongly Adherent", "Mostly Adherent", 
                        "Partially Adherent (Minor Deviations)", 
                        "Not Adherent (Significant Deviations/Incorrect)",
                        "Not Applicable / Unable to Determine"
                    ],
                    key=f"adh_{case_data.get('case_id')}",
                    horizontal=False # Easier to read stacked
                )

                st.markdown("**B. Clinical Correctness & Safety:**")
                eval_data["clinical_correctness_safety"] = st.radio(
                    "Assess the clinical correctness and safety of the recommendation.",
                    options=[
                        "Correct and Safe", "Mostly Correct, Minor Concerns",
                        "Potentially Unsafe / Significant Errors", "Incorrect"
                    ],
                    key=f"correct_{case_data.get('case_id')}",
                    horizontal=False
                )

                st.markdown("**C. Clarity & Explainability (of Justification):**")
                eval_data["clarity_explainability"] = st.slider(
                    "Rate the clarity and explainability of the LLM's justification.",
                    min_value=1, max_value=5, value=3, step=1,
                    help="1=Very Unclear, 3=Neutral, 5=Very Clear",
                    key=f"clarity_{case_data.get('case_id')}"
                )

                st.markdown("**D. Completeness (of Justification):**")
                eval_data["completeness_justification"] = st.radio(
                    "How complete is the justification regarding key patient factors?",
                    options=[
                        "Comprehensive", "Mostly Complete (Minor omissions)",
                        "Incomplete (Significant omissions)"
                    ],
                    key=f"complete_{case_data.get('case_id')}",
                    horizontal=False
                )

                st.markdown("**E. Overall Assessment:**")
                eval_data["overall_assessment"] = st.radio(
                    "What is your overall assessment of this LLM recommendation?",
                    options=[
                        "Excellent (Would use as is / with minimal touch-up)",
                        "Good (Minor edits needed for clinical use)",
                        "Fair (Significant edits needed, but concept is salvageable)",
                        "Poor (Not usable / Misleading)"
                    ],
                    key=f"overall_{case_data.get('case_id')}",
                    horizontal=False
                )

                st.markdown("**F. Free-text Comments:**")
                eval_data["comments"] = st.text_area(
                    "Provide any additional comments, specific errors, or suggestions.",
                    height=150,
                    key=f"comments_{case_data.get('case_id')}"
                )

                # Submit button for the form
                submitted = st.form_submit_button("Submit Evaluation")

                if submitted:
                    # Add metadata to the evaluation
                    eval_data["case_id_evaluated"] = case_data.get('case_id')
                    eval_data["case_filename_evaluated"] = selected_case_file
                    eval_data["expert_name"] = expert_name
                    eval_data["evaluation_timestamp"] = datetime.now().isoformat()
                    
                    success, saved_filename = save_evaluation(selected_case_file, eval_data, expert_name)
                    if success:
                        st.success(f"Evaluation submitted successfully! Saved as: {saved_filename}")
                        st.balloons()
                        # Optional: Could clear the form or move to next case
                    else:
                        st.error("Failed to save evaluation. Check console/logs.")
    else:
        st.error(f"Could not load data for case: {selected_case_file}")

else:
    st.info("Select a case from the sidebar to begin evaluation.")