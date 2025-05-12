# agents/base_agent.py
import logging
import time
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

    def _invoke_llm(self, prompt_template_str: str, context_vars: dict) -> str:
        """Helper method to invoke the LLM with a specific template."""
        prompt = PromptTemplate.from_template(prompt_template_str)
        chain = prompt | self.llm
        try:
            # Log only keys or snippet of context for brevity if context is large
            log_context_vars = {k: (str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in context_vars.items()}
            logger.debug(f"Invoking LLM for agent {self.name} with vars: {log_context_vars}")
            response = chain.invoke(context_vars)
            logger.debug(f"LLM response snippet for {self.name}: {str(response)[:100]}...")
            return str(response) # Ensure string response
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