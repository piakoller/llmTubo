# agents/base_agent.py
import logging
import time
import re
import csv # For writing CSV files
import os # For checking file existence
import threading # For thread-safe CSV writing
from datetime import datetime # For timestamping interactions
from abc import ABC, abstractmethod
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

import config # Import your project's config file

# This logger will use the centralized configuration from utils/logging_setup.py
logger = logging.getLogger(__name__)

# Thread lock for CSV writing to prevent race conditions
csv_writer_lock = threading.Lock()

def log_llm_interaction_to_csv(agent_name: str, rendered_prompt: str, raw_response: str, final_response: str, error_message: str = ""):
    """
    Logs the LLM interaction details to a CSV file in a thread-safe manner.
    """
    file_path = config.LLM_INTERACTIONS_CSV_FILE
    file_exists = os.path.isfile(file_path)

    # Define field names for the CSV
    fieldnames = ['timestamp', 'agent_name', 'rendered_prompt', 'raw_response', 'final_response', 'error_message']
    
    interaction_data = {
        'timestamp': datetime.now().isoformat(),
        'agent_name': agent_name,
        'rendered_prompt': rendered_prompt,
        'raw_response': raw_response,
        'final_response': final_response,
        'error_message': error_message
    }

    with csv_writer_lock: # Acquire lock before writing
        try:
            with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(file_path) == 0: # Check if file is new or empty
                    writer.writeheader() # Write header only if file is new/empty
                writer.writerow(interaction_data)
            # logger.debug(f"Logged LLM interaction for agent {agent_name} to CSV.") # Optional: log CSV write
        except IOError as e:
            logger.error(f"IOError writing LLM interaction to CSV {file_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error writing LLM interaction to CSV {file_path}: {e}", exc_info=True)


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
            match = re.search(re.escape(think_tag_end), full_llm_output, re.IGNORECASE)
            if match:
                content_after_think = full_llm_output[match.end():]
                logger.debug(f"Extracted content after {think_tag_end}. Snippet: {content_after_think.strip()[:100]}...")
                return content_after_think.strip()
            else:
                logger.debug("No <think> block found in LLM output. Returning original.")
                return full_llm_output.strip()
        except Exception as e:
            logger.error(f"Error during extraction of final response: {e}", exc_info=True)
            return full_llm_output.strip()


    def _invoke_llm(self, prompt_template_str: str, context_vars: dict) -> str:
        """Helper method to invoke the LLM with a specific template and log interaction."""
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        
        # Render the prompt for logging
        rendered_prompt = ""
        try:
            # Ensure all context_vars keys are present in the prompt template's input_variables
            # This is a common source of errors if a key is missing.
            # Langchain's PromptTemplate.format_prompt might be safer or provide better error messages.
            # For simplicity, we'll try to render it directly if keys match.
            # A more robust way is to use prompt_template.format(**context_vars)
            # but ensure all required variables are in context_vars.
            
            # Let's use format_prompt for better safety and to get the string
            prompt_value = prompt_template.format_prompt(**context_vars)
            rendered_prompt = prompt_value.to_string()
        except KeyError as ke:
            logger.error(f"KeyError rendering prompt for agent {self.name}. Missing key: {ke}. Context_vars: {context_vars.keys()}")
            rendered_prompt = f"Error rendering prompt: Missing key {ke}"
        except Exception as e:
            logger.error(f"Error rendering prompt for agent {self.name}: {e}", exc_info=True)
            rendered_prompt = f"Error rendering prompt: {e}"

        chain = prompt_template | self.llm # Recreate chain with the template instance
        
        raw_response = ""
        final_response = ""
        error_message = ""

        try:
            log_context_vars = {k: (str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in context_vars.items()}
            logger.debug(f"Invoking LLM for agent {self.name} with vars: {log_context_vars}")
            
            llm_output = chain.invoke(context_vars) # Use the original context_vars dict
            
            if not isinstance(llm_output, str):
                raw_response = str(llm_output)
            else:
                raw_response = llm_output

            logger.info(f"LLM raw response snippet for {self.name}: {raw_response[:200]}...")
            final_response = self._extract_final_response(raw_response)
            
        except Exception as e:
            logger.error(f"LLM invocation failed for agent {self.name}: {e}", exc_info=True)
            error_message = str(e)
            # Log interaction even on failure
            log_llm_interaction_to_csv(self.name, rendered_prompt, raw_response, final_response, error_message)
            raise RuntimeError(f"LLM call failed for {self.name}") from e
        
        # Log interaction on success
        log_llm_interaction_to_csv(self.name, rendered_prompt, raw_response, final_response)
        return final_response


class AgentRunner:
    # ... (AgentRunner class remains the same as before) ...
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