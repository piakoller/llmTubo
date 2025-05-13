# agents/base_agent.py
import logging
import time
import re
from abc import ABC, abstractmethod
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class Agent(ABC):
    """Abstract base class for all agents."""
    def __init__(self, name: str, role_description: str, llm: OllamaLLM):
        self.name = name
        self.role_description = role_description
        if llm is None:
            raise ValueError("LLM instance must be provided to the agent.")
        self.llm = llm
        logger.info(f"Agent '{self.name}' initialized.")

    @abstractmethod
    def respond(self, context: str) -> any:
        """Process the given context and return a response."""
        pass

    def _extract_final_response(self, full_llm_output: str) -> str:
        """
        Extracts the content after the closing </think> tag.
        If no <think> block is found, returns the original output.
        """
        think_tag_end = "</think>"
        try:
            # Case-insensitive search for the closing tag
            match = re.search(re.escape(think_tag_end), full_llm_output, re.IGNORECASE)
            if match:
                # Get the content after the tag
                content_after_think = full_llm_output[match.end():]
                logger.debug(f"Extracted content after {think_tag_end}. Snippet: {content_after_think.strip()[:100]}...")
                return content_after_think.strip() # Remove leading/trailing whitespace
            else:
                # No <think> block found, return original (stripped)
                logger.debug("No <think> block found in LLM output. Returning original.")
                return full_llm_output.strip()
        except Exception as e:
            logger.error(f"Error during extraction of final response: {e}", exc_info=True)
            return full_llm_output.strip() # Fallback to original on error


    def _invoke_llm(self, prompt_template_str: str, context_vars: dict) -> str:
        """Helper method to invoke the LLM with a specific template."""
        prompt = PromptTemplate.from_template(prompt_template_str)
        chain = prompt | self.llm
        try:
            log_context_vars = {k: (str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in context_vars.items()}
            logger.debug(f"Invoking LLM for agent {self.name} with vars: {log_context_vars}")
            
            raw_response = chain.invoke(context_vars)
            if not isinstance(raw_response, str): # Ensure it's a string for processing
                raw_response = str(raw_response)

            logger.debug(f"LLM raw response snippet for {self.name}: {raw_response[:200]}...") # Log more of raw

            # Extract the final response after the <think> block
            final_response = self._extract_final_response(raw_response)
            
            return final_response # Return the processed response
        
        except Exception as e:
            logger.error(f"LLM invocation failed for agent {self.name}: {e}", exc_info=True)
            raise RuntimeError(f"LLM call failed for {self.name}") from e

class AgentRunner:
    """Handles the execution of an agent, capturing results, runtime, and exceptions."""
    def __init__(self, agent: Agent, name: str, context: str):
        self.agent = agent
        self.name = name
        self.context = context
        self.result: any = None
        self.runtime: float | None = None
        self.exception: Exception | None = None

    def run(self):
        """Executes the agent's respond method and records metrics."""
        context_snippet = self.context[:100] + '...' if len(self.context) > 100 else self.context
        logger.info(f"Agent '{self.name}' starting with context snippet: '{context_snippet}'")
        start_time = time.perf_counter()
        try:
            self.result = self.agent.respond(self.context)
        except Exception as e:
            logger.error(f"Agent '{self.name}' encountered an error: {e}", exc_info=True)
            self.exception = e
        finally:
            self.runtime = time.perf_counter() - start_time
            status = "failed" if self.exception else "completed"
            logger.info(f"Agent '{self.name}' {status} in {self.runtime:.2f}s.")