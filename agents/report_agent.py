# agents/report_agent.py
import logging
import urllib.parse
import requests
import os
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from .base_agent import Agent # Relative import
import config # Global config

logger = logging.getLogger(__name__)

class ReportAgent:
    """Generates and saves a structured medical report."""
    def __init__(self, llm: OllamaLLM, output_dir: str = "generated_report", file_type: str = "md"):
        if llm is None:
             raise ValueError("LLM instance must be provided to the ReportAgent.")
        self.llm = llm
        self.output_dir = output_dir
        # Ensure file type is supported (currently only md)
        self.file_type = file_type.lower()
        if self.file_type not in ["md", "markdown"]:
            logger.warning(f"Unsupported file type '{file_type}'. Defaulting to 'md'.")
            self.file_type = "md"
        logger.info(f"ReportAgent initialized. Output directory: '{output_dir}', File type: '{self.file_type}'")


    def generate_report_text(self, context: str, patient_data: dict, board_date: str) -> str:
        """Uses LLM to structure the context into a report format, incorporating patient data."""
        logger.info("Generating report text...")

        # Prepare patient info string for the prompt
        # Adjust keys based on your actual patient_data dictionary
        patient_info_str = f"""
            Patient: {patient_data.get('last_name', 'Nachname')}, {patient_data.get('first_name', 'Vorname')}
            Geburtsdatum: {patient_data.get('dob', 'XX.XX.XXXX')}
            PID: {patient_data.get('pid', 'XXXXXXXX')}
            """

        # The main diagnosis might be in patient_data or context, adjust as needed
        main_diagnosis_str = patient_data.get('main_diagnosis_text', 'Hauptdiagnose nicht spezifiziert')

        prompt = PromptTemplate(
            input_variables=["patient_info", "board_date", "main_diagnosis", "context"],
            template="""
            Du bist ein medizinischer Sekretär/Assistent, der Tumorboard-Protokolle im Stil des Inselspitals Bern (Schweiz) verfasst.
            Erstelle ein Protokoll für das Endokrine Tumorboard vom {board_date}.

            **Patientendaten:**
            {patient_info}

            **Hauptdiagnose (basierend auf vorliegenden Informationen):**
            1. {main_diagnosis}

            Der folgende `{context}` enthält eine Zusammenfassung der aktuellen diagnostischen Situation durch einen Experten und die daraus resultierende aktuelle Therapieempfehlung des Tumorboards.

            **Eingabeinformationen (Aktuelle Beurteilung & Empfehlung):**
            ---
            {context}
            ---

            **Deine Aufgabe:**
            Formatiere **nur** die Informationen aus dem `{context}` in ein medizinisches Protokoll im Markdown-Format.
            Orientiere dich am Stil des Inselspitals, insbesondere bei den Abschnitten 'Beurteilung' und 'Empfehlungen'.
            Fasse die diagnostische Einschätzung aus dem `{context}` unter der Überschrift '## Beurteilung' zusammen.
            Liste die aktuelle Therapieempfehlung(en) aus dem `{context}` unter der Überschrift '## Empfehlungen' auf, idealerweise als Bullet Points (`*` oder `•`).

            **WICHTIG:**
            *   Füge **keine** historischen Daten (frühere Scans, Behandlungen, Empfehlungen, Labore, Verläufe) hinzu, die nicht explizit im `{context}` stehen.
            *   Erfinde **keine** Patientendaten, Adressen, Board-Teilnehmer oder Abteilungsleiter.
            *   Generiere **keinen** Briefkopf oder Fusszeile.
            *   Konzentriere dich darauf, die *aktuelle Beurteilung* und die *neuen Empfehlungen* basierend auf dem `{context}` klar darzustellen.

            **Protokoll (Markdown):**

            ## Beurteilung
            [Hier die Zusammenfassung der diagnostischen Einschätzung aus dem context einfügen]

            ## Empfehlungen
            [Hier die aktuelle(n) Therapieempfehlung(en) aus dem context als Bullet Points auflisten]

            """
        )
        chain = prompt | self.llm
        try:
            report_text = chain.invoke({
                "patient_info": patient_info_str,
                "board_date": board_date,
                "main_diagnosis": main_diagnosis_str,
                "context": context
                })
            logger.info("Report text generated successfully.")
            # Prepend a basic header to the LLM output
            header = f"# Endokrines Tumorboard vom {board_date}\n\n"
            full_report = header + patient_info_str + f"\n## Diagnosen\n1. {main_diagnosis_str}\n\n" + report_text
            return full_report
        except Exception as e:
            logger.error(f"LLM invocation failed during report generation: {e}", exc_info=True)
            raise RuntimeError("LLM call failed during report generation") from e

    def save_report(self, report_text: str, filename_base: str) -> str:
        """Saves the generated report text to a file."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f"{filename_base}.{self.file_type}")
        logger.info(f"Attempting to save report to: {filepath}")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Report successfully saved to {filepath}")
            return filepath
        except OSError as e:
            logger.error(f"Failed to save report to {filepath}: {e}", exc_info=True)
            raise RuntimeError(f"Error saving report file: {e}") from e
        except Exception as e: # Catch any other unexpected errors during file writing
             logger.error(f"An unexpected error occurred while saving the report to {filepath}: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error saving report file: {e}") from e