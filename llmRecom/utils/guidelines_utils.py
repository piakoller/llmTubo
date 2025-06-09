# utils/guidelines_utils.py
import glob
import os
import config
import logging

logger = logging.getLogger(__name__)

# GUIDELINE_DATA_DIR_NET = "/home/pia/projects/llmTubo/llmRecom/data/NET/"
# GUIDELINE_DATA_DIR_LMYPH = "/home/pia/projects/llmTubo/llmRecom/data/guidelines/mds"

def find_guideline_and_net_files(guideline_name: str, main_diagnosis: str, net_mode: bool) -> list[str]:
    """
    Sucht nach der passenden Guideline-Datei(en) und – falls NET aktiv –
    nach NET-spezifischen Attachments (Pressemitteilung, Studie).
    Gibt eine Liste aller zu attachenden Dateien zurück.
    """
    attachments: List[str] = [] # Ensure it's a list
    guideline_name_lower = guideline_name.lower()
    main_diag_lower = main_diagnosis.lower().replace(" ", "_").replace("/", "_") # Handle slashes in diagnosis

    if net_mode:
        guideline_dir = config.GUIDELINE_DATA_DIR_NET
        
        pattern = f"*{guideline_name_lower}*{main_diag_lower}*"
        files = glob.glob(os.path.join(guideline_dir, pattern + ".*")) # Search for any extension

        if not files:
             # Fallback: Search just by guideline name if diagnosis part yielded no results
             pattern_fallback = f"*{guideline_name_lower}*"
             files = glob.glob(os.path.join(guideline_dir, pattern_fallback + ".*"))
             if files:
                 logger.warning(f"No specific guideline files found for pattern '{pattern}'. Using fallback pattern '{pattern_fallback}'. Found: {files}")
             else:
                 logger.warning(f"No guideline files found for NET mode using patterns '{pattern}' or '{pattern_fallback}' in {guideline_dir}.")

        # Add all found guideline files
        attachments.extend(files)

        # NET Pressemittelung und NET Studie hinzufügen (Dateien mit "press" oder "study" im Namen)
        # Search in the same NET directory
        net_press_files = glob.glob(os.path.join(guideline_dir, "*press*.*")) # Any extension
        net_study_files = glob.glob(os.path.join(guideline_dir, "*study*.*"))  # Any extension

        # Add unique press and study files
        for f in net_press_files + net_study_files:
            # Only add if it's not already in the list (e.g., if a file matched the guideline pattern AND press/study pattern)
            if f not in attachments:
                attachments.append(f)
                logger.info(f"Added NET specific file: {f}")

    else: # Not NET mode, use the default/Lymph directory
        guideline_dir = config.GUIDELINE_DATA_DIR_LMYPH
        logger.info(f"NET mode is FALSE. Searching in {guideline_dir}")
        # Suche nach relevanter Guideline-Datei(en) im Lymph-Verzeichnis
        pattern = f"*{guideline_name_lower}*{main_diag_lower}*"
        files = glob.glob(os.path.join(guideline_dir, pattern + ".*")) # Search for any extension

        if not files:
             # Fallback: Search just by guideline name
             pattern_fallback = f"*{guideline_name_lower}*"
             files = glob.glob(os.path.join(guideline_dir, pattern_fallback + ".*"))
             if files:
                 logger.warning(f"No specific guideline files found for pattern '{pattern}'. Using fallback pattern '{pattern_fallback}'. Found: {files}")
             else:
                 logger.warning(f"No guideline files found for non-NET mode using patterns '{pattern}' or '{pattern_fallback}' in {guideline_dir}.")

        # Add all found guideline files
        attachments.extend(files)

    # Filter out directories if any were matched by glob patterns
    attachments = [f for f in attachments if os.path.isfile(f)]

    # Optional: Log which files were found
    if attachments:
        logger.info(f"Found total attachments: {attachments}")
    else:
        logger.warning(f"No relevant files found for guideline '{guideline_name}', diagnosis '{main_diagnosis}', NET mode {net_mode}.")

    return attachments