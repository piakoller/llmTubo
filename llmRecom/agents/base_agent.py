# agents/base_agent.py
import logging
import time
import re
import csv
import os
import threading
from datetime import datetime
from abc import ABC, abstractmethod
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

import config # Import your project's config file

# This logger will use the centralized configuration from utils/logging_setup.py
logger = logging.getLogger(__name__)

# Thread lock for CSV writing
csv_writer_lock = threading.Lock()

def log_llm_interaction_to_csv(agent_name: str, rendered_prompt: str, raw_response: str,
                               think_block: str, final_response: str, error_message: str = ""):
    """
    Logs the LLM interaction details to a CSV file in a thread-safe manner.
    Includes a separate column for the <think> block.
    """
    file_path = config.LLM_INTERACTIONS_CSV_FILE
    file_exists = os.path.isfile(file_path)

    # Define field names for the CSV, now including 'think_block'
    fieldnames = ['timestamp', 'agent_name', 'rendered_prompt', 'raw_response', 'think_block', 'final_response', 'error_message']
    
    interaction_data = {
        'timestamp': datetime.now().isoformat(),
        'agent_name': agent_name,
        'rendered_prompt': rendered_prompt,
        'raw_response': raw_response,
        'think_block': think_block, # Add the think block
        'final_response': final_response,
        'error_message': error_message
    }

    with csv_writer_lock:
        try:
            with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(file_path) == 0:
                    writer.writeheader()
                writer.writerow(interaction_data)
        except IOError as e:
            logger.error(f"IOError writing LLM interaction to CSV {file_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error writing LLM interaction to CSV {file_path}: {e}", exc_info=True)


class Agent(ABC):
    def __init__(self, name: str, role_description: str, llm: OllamaLLM):
        self.name = name
        self.role_description = role_description
        if llm is None:
            raise ValueError("LLM instance must be provided to the agent.")
        self.llm = llm
        logger.info(f"Agent '{self.name}' initialized.")

    @abstractmethod
    def respond(self, context: str) -> any:
        pass

    def _extract_think_and_final_response(self, full_llm_output: str) -> tuple[str, str]:
        """
        Extracts the <think> block and the content after the closing </think> tag.
        Returns: (think_block_content, final_response_content)
        If no <think> block is found, think_block_content will be empty.
        """
        think_block_content = ""
        final_response_content = full_llm_output.strip() # Default to full output

        think_tag_start_pattern = r"<think>"
        think_tag_end_pattern = r"</think>"
        
        try:
            # Case-insensitive search for the full think block
            # Using re.DOTALL so '.' matches newlines within the think block
            match = re.search(f"{think_tag_start_pattern}(.*?){think_tag_end_pattern}", 
                              full_llm_output, re.IGNORECASE | re.DOTALL)
            
            if match:
                think_block_content = match.group(1).strip() # Content between <think> and </think>
                # The rest of the string after the matched </think> tag
                final_response_content = full_llm_output[match.end():].strip()
                logger.debug(f"Extracted think block. Snippet: {think_block_content[:100]}...")
                logger.debug(f"Extracted final response after think block. Snippet: {final_response_content[:100]}...")
            else:
                logger.debug("No <think> block found in LLM output. Final response is the original output.")
                # final_response_content is already set to full_llm_output.strip()
        except Exception as e:
            logger.error(f"Error during extraction of think/final response: {e}", exc_info=True)
            # Fallback: think_block remains empty, final_response is the original full output
            final_response_content = full_llm_output.strip()
            
        return think_block_content, final_response_content


    def _invoke_llm(self, prompt_template_str: str, context_vars: dict) -> str:
        """Helper method to invoke the LLM, log interaction, and separate think block."""
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        
        rendered_prompt = ""
        try:
            prompt_value = prompt_template.format_prompt(**context_vars)
            rendered_prompt = prompt_value.to_string()
        except KeyError as ke:
            logger.error(f"KeyError rendering prompt for agent {self.name}. Missing key: {ke}. Context_vars: {context_vars.keys()}")
            rendered_prompt = f"Error rendering prompt: Missing key {ke}"
        except Exception as e:
            logger.error(f"Error rendering prompt for agent {self.name}: {e}", exc_info=True)
            rendered_prompt = f"Error rendering prompt: {e}"

        chain = prompt_template | self.llm
        
        raw_response = ""
        think_block = ""
        final_response = ""
        error_message = ""

        try:
            log_context_vars = {k: (str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in context_vars.items()}
            logger.debug(f"Invoking LLM for agent {self.name} with vars: {log_context_vars}")
            
            llm_output = chain.invoke(context_vars)
            
            if not isinstance(llm_output, str):
                raw_response = str(llm_output)
            else:
                raw_response = llm_output

            logger.info(f"LLM raw response snippet for {self.name}: {raw_response[:200]}...")
            
            # Extract both think block and final response
            think_block, final_response = self._extract_think_and_final_response(raw_response)
            
        except Exception as e:
            logger.error(f"LLM invocation failed for agent {self.name}: {e}", exc_info=True)
            error_message = str(e)
            # Log interaction even on failure (think_block and final_response might be empty or partial)
            log_llm_interaction_to_csv(self.name, rendered_prompt, raw_response, think_block, final_response, error_message)
            raise RuntimeError(f"LLM call failed for {self.name}") from e
        
        # Log interaction on success
        log_llm_interaction_to_csv(self.name, rendered_prompt, raw_response, think_block, final_response)
        return final_response # The agent method still returns only the final_response


class AgentRunner:
    # ... (AgentRunner class remains the same) ...
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