# agents/therapie_agent.py
import logging
import urllib.parse
import requests
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any, Tuple, Optional
import os

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

    def respond(self, diagnostik_summary_context: str, attachments: List[str] | None = None) -> Tuple[str, str, str | None, str, float]:
        """Uses LLM to generate therapy recommendations, using the diagnostik summary and optionally attached files."""
        template = """
            <think>
            [Hier deine Denkprozesse vor der finalen Antwort einfügen. Dieser Block wird später separat gespeichert.]
            </think>

            # Therapieempfehlung für Tumorboard

            Du bist ein erfahrener Onkologie-Experte, spezialisiert auf Therapieplanung. Deine Aufgabe ist es, basierend auf der bereitgestellten diagnostischen Zusammenfassung und unter **striktester Berücksichtigung der angefügten Dokumente**, insbesondere der darin enthaltenen Leitlinie '{guideline_provider}', eine präzise und begründete Therapieempfehlung für den besprochenen Patienten auf Deutsch zu erstellen.

            **Deine Rolle:** {role}

            **Leitlinie für diese Empfehlung:** {guideline_provider} (Siehe angefügte Dokumente)

            **Folgende Dokumente sind als Anhang verfügbar und müssen berücksichtigt werden:**
            {attachments_info}

            **Patienten-Kontext und Diagnostische Zusammenfassung (erstellt vom Diagnostik-Agent):**
            ---
            {diagnostik_summary_context}
            ---

            **Anweisungen für die Erstellung der Therapieempfehlung:**
            1.  Analysiere die bereitgestellten Informationen im Abschnitt "Patienten-Kontext und Diagnostische Zusammenfassung" sorgfältig.
            2.  **Konzentriere dich primär auf die angefügten Dokumente**, insbesondere die Leitlinie '{guideline_provider}'. Vergleiche den Patientenfall mit den dort beschriebenen Kriterien und Empfehlungen.
            3.  Sollten zusätzlich Pressemitteilungen oder Studien angefügt sein (im NET-Modus), bewerte deren Relevanz im Kontext der Leitlinie und der Patientendaten, um die bestmögliche Empfehlung abzuleiten.
            4.  Überprüfe wenn sinnvoll (im NET-Modus), ob die NETTER-2 Studie aus dem Anhang anwendbar ist
            5.  Formuliere **EINE EINZIGE, KONKRETE THERAPIEEMPFEHLUNG**, die am besten zu den Fakten im Kontext und den Empfehlungen in den **angefügten Dokumenten** passt.
            6.  Gib eine klare und prägnante Begründung für die empfohlene Therapie. Die Begründung muss sich explizit auf die relevanten Punkte aus dem Patienten-Kontext (Diagnose, Stadium, relevante Vorerkrankungen etc.) und die Empfehlungen aus den **angefügten Dokumenten (Leitlinie, Studie, PM)** beziehen.
            7.  Sollten die im Kontext enthaltenen Informationen *trotz der angefügten Dokumente* für eine definitive Therapieempfehlung unzureichend sein (z.B. fehlendes Staging, unklare Histologie), gib dies an und schlage notwendige weitere diagnostische Schritte vor.
            8.  Gib **ausschließlich** die Therapieempfehlung und die Begründung auf Deutsch aus. Verwende dabei die unten angegebene Struktur der Überschriften. Füge keine zusätzlichen einleitenden oder abschliessenden Sätze hinzu, die nicht Teil der Empfehlung oder Begründung sind.
            
            **Deine Antwort (Im folgenden Markdown-Format):**

            **Therapieempfehlung:**
            [Hier deine einzige, konkrete Therapieempfehlung auf Deutsch einfügen. Beginne direkt mit der Empfehlung.]

            **Begründung:**
            [Hier deine detaillierte Begründung einfügen. Beziehe dich auf den Kontext und die angefügten Dokumente.]
            """
        
        attachments_info = "Folgende Dokumente stehen als Anhang zur Verfügung:\n" + "\n".join(f"- {os.path.basename(a)}" for a in attachments) if attachments else ""

        invoke_llm_result_tuple = self._invoke_llm(
            template,
            {
                "role": self.role_description,
                "diagnostik_summary_context": diagnostik_summary_context,
                "guideline_provider": self.guideline_provider,
                "attachments_info": attachments_info
            },
            attachments=attachments
        )
        logger.info(f"Attachments provided: {attachments}")
        logger.debug(f"TherapieAgent _invoke_llm result: (response_snippet='{invoke_llm_result_tuple[0][:50]}...', interaction_id='{invoke_llm_result_tuple[3]}', duration={invoke_llm_result_tuple[4]:.2f}s)") # Updated tuple indexing
        
        return invoke_llm_result_tuple
    
