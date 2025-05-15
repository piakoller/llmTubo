# agents/therapie_agent.py
import logging
import urllib.parse
import requests
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM

from .base_agent import Agent # Relative import
import config # Global config

logger = logging.getLogger(__name__)

class TherapieAgent(Agent):
    """Generates a therapy recommendation based on provided information."""
    def __init__(self, llm: OllamaLLM, guideline_provider: str):
        super().__init__(
            "Therapieempfehlung",
            "Erstelle eine detaillierte, leitlinienbasierte Therapieempfehlung basierend auf der Diagnoseanalyse.",
             llm
         )
        # Store the guideline provider
        self.guideline_provider = guideline_provider
        logger.info(f"TherapieAgent initialized with guideline: {self.guideline_provider}")

    def respond(self, context: str) -> str:
        """Uses LLM to generate therapy recommendations."""
        template = """
            # Therapieempfehlung für Tumorboard

            Du bist ein erfahrener Onkologie-Experte, spezialisiert auf Therapieplanung. Deine Aufgabe ist es, basierend auf der bereitgestellten diagnostischen Zusammenfassung und unter strikter Berücksichtigung der angegebenen Leitlinie, eine präzise und begründete Therapieempfehlung für den besprochenen Patienten zu erstellen.

            **Deine Rolle:** {role}

            **Leitlinie für diese Empfehlung:** {guideline_provider}

            **Patienten-Kontext und Diagnostische Zusammenfassung:**
            ---
            {context}
            ---

            **Anweisungen für die Erstellung der Therapieempfehlung:**
            1.  Analysiere die bereitgestellten Informationen im Abschnitt "Patienten-Kontext und Diagnostische Zusammenfassung" sorgfältig.
            2.  Berücksichtige **ausschließlich** die angegebene Leitlinie: **{guideline_provider}**. Ignoriere andere Leitlinien oder allgemeine Informationen, es sei denn, sie sind im Kontext als relevant aufgeführt.
            3.  Formuliere **EINE EINZIGE, KONKRETE THERAPIETHERAPIE**, die am besten zu den Fakten im Kontext und der angegebenen Leitlinie passt.
            4.  Sollte im Kontext bereits eine mögliche Therapie vorgeschlagen sein, bewerte diese kritisch im Licht der Leitlinie und integriere oder ersetze sie durch die am besten begründete Empfehlung.
            5.  Gib eine klare und prägnante Begründung für die empfohlene Therapie. Die Begründung muss sich explizit auf die relevanten Punkte aus dem Patienten-Kontext (Diagnose, Stadium, relevante Vorerkrankungen etc.) und die Empfehlungen der Leitlinie beziehen.
            6.  Sollten die im Kontext enthaltenen Informationen für eine definitive Therapieempfehlung unzureichend sein (z.B. fehlendes Staging, unklare Histologie), gib dies an und schlage notwendige weitere diagnostische Schritte vor.
            7.  Gib **ausschließlich** die Therapieempfehlung und die Begründung aus. Verwende dabei die unten angegebene Struktur der Überschriften. Füge keine zusätzlichen einleitenden oder abschliessenden Sätze hinzu, die nicht Teil der Empfehlung oder Begründung sind.

            **Deine Antwort (Im folgenden Markdown-Format):**

            **Therapieempfehlung:**
            [Hier deine einzige, konkrete Therapieempfehlung einfügen. Beginne direkt mit der Empfehlung.]

            **Begründung:**
            [Hier deine detaillierte Begründung einfügen. Beziehe dich auf den Kontext und die Leitlinie.]
            """
            # _invoke_llm now returns (final_response, interaction_id)
        final_response, interaction_id = self._invoke_llm(
            template,
            {
                "role": self.role_description,
                "guideline_provider": self.guideline_provider,
                "diagnostik_summary": diagnostik_summary_context
            }
        )
        logger.debug(f"TherapieAgent LLM interaction ID: {interaction_id}, response: {final_response[:50]}...")
        return final_response
