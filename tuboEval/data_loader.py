# data_loader.py
import pandas as pd
import streamlit as st
import logging

logger = logging.getLogger(__name__)

@st.cache_data # Cache the loaded data
def load_patient_data(file_path: str) -> pd.DataFrame | None:
    """Loads patient data from the specified Excel file."""
    logger.info(f"Attempting to load patient data from: {file_path}")
    try:
        df = pd.read_excel(file_path, skiprows=8)
        if df.empty:
            logger.warning(f"Excel file loaded but is empty (or empty after skipping rows): {file_path}")
            st.warning("Die geladene Excel-Datei ist leer oder enthält keine Daten nach dem Überspringen der Kopfzeilen.")
            return None
        logger.info(f"Successfully loaded {len(df)} patient records from {file_path}")
        return df
    except FileNotFoundError:
        st.error(f"Fehler: Excel-Datei nicht gefunden unter {file_path}")
        logger.error(f"Excel file not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Fehler beim Laden der Excel-Datei: {e}")
        logger.error(f"Error loading Excel file: {e}", exc_info=True)
        return None