# agents/diagnostik_agent.py
import logging
from langchain_ollama import OllamaLLM
from .base_agent import Agent

logger = logging.getLogger(__name__)

class DiagnostikAgent(Agent):
    """Analyzes diagnosis, symptoms, and clinical reports."""
    def __init__(self, llm: OllamaLLM):
        super().__init__("Diagnostik", "Analysiere Diagnose, Symptome und klinische Berichte und fasse die wichtigsten Punkte zusammen.", llm)

    def respond(self, context: str) -> tuple[str, str]:
        """Uses LLM to analyze the provided context."""
        template = """
        Du bist ein medizinischer Experte für Diagnostik. Deine Rolle ist: {role}
        Analysiere den folgenden Patientenkontext und fasse die relevanten diagnostischen Informationen, das Stadium und wichtige klinische Punkte klar und prägnant auf Deutsch zusammen.

        Kontext:
        ---
        {context}
        ---

        Zusammenfassung der Diagnostik:
        """
        invoke_llm_result_tuple = self._invoke_llm(
            template,
            {"role": self.role_description, "context": context}
        )
        logger.debug(f"DiagnostikAgent _invoke_llm result: (response_snippet='{invoke_llm_result_tuple[0][:50]}...', interaction_id='{invoke_llm_result_tuple[1]}')")
        
        return invoke_llm_result_tuple
