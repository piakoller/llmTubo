# agents/diagnostik_agent.py
import logging
from langchain_ollama import OllamaLLM
from .base_agent import Agent
from typing import List, Tuple
import os

logger = logging.getLogger(__name__)

class DiagnostikAgent(Agent):
    """Analyzes diagnosis, symptoms, and clinical reports."""
    def __init__(self, llm: OllamaLLM):
        super().__init__("Diagnostik", "Analysiere Diagnose, Symptome und klinische Berichte und fasse die wichtigsten Punkte zusammen.", llm)

    def respond(self, context: str, attachments: List[str] | None = None) -> Tuple[str, str, str | None, str, float]:        
        """Uses LLM to analyze the provided context."""
        template = """
        Du bist ein medizinischer Experte für Diagnostik. Deine Rolle ist: {role}
        Analysiere den folgenden Patientenkontext und fasse die relevanten diagnostischen Informationen, das Stadium und wichtige klinische Punkte klar und prägnant auf Deutsch zusammen.

        {attachments_info}
        Kontext:
        ---
        {context}
        ---

        Falls angefügte Dokumente oder Informationen (siehe oben) vorhanden sind, berücksichtige diese bitte explizit in deiner Analyse. Ziehe sowohl die Leitlinie(n) als auch alle weiteren bereitgestellten Anhänge heran, um deine Zusammenfassung zu erstellen. 
        Gehe dabei auf Besonderheiten oder zusätzliche Hinweise aus den Anhängen ein, sofern sie für die Diagnostik relevant sind.

        Fasse die wichtigsten diagnostischen Erkenntnisse und klinischen Punkte basierend auf dem Kontext und den angefügten Dokumenten zusammen. Achte besonders auf die Hauptdiagnose, das Stadium und relevante klinische Details sowie die PET-CT-Ergebnisse.

        Zusammenfassung der Diagnostik:
        """
        
        attachments_info = "Folgende Dokumente stehen als Anhang zur Verfügung:\n" + "\n".join(f"- {os.path.basename(a)}" for a in attachments) if attachments else ""

        invoke_llm_result_tuple = self._invoke_llm(
            template,
            {
                "role": self.role_description,
                "context": context,
                "attachments_info": attachments_info # Pass attachments_info to template
            },
            attachments=attachments # Pass attachments to _invoke_llm
        )
        logger.info(f"Attachments provided: {attachments}")
        logger.debug(f"DiagnostikAgent _invoke_llm result: (response_snippet='{invoke_llm_result_tuple[0][:50]}...', interaction_id='{invoke_llm_result_tuple[3]}', duration={invoke_llm_result_tuple[4]:.2f}s)") # Updated tuple indexing
        
        return invoke_llm_result_tuple