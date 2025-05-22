# agents/base_agent.py
import logging
import time
import re
import csv
import os
import threading
import uuid
import html
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

import config

# This logger will use the centralized configuration from utils/logging_setup.py
logger = logging.getLogger(__name__)

# Thread lock for CSV writing
csv_writer_lock = threading.Lock()

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
        think_block_content = ""
        final_response_content = full_llm_output.strip()

        full_llm_output = html.unescape(full_llm_output)

        think_tag_start_pattern = r"<think>"
        think_tag_end_pattern = r"</think>"

        try:
            match = re.search(f"{think_tag_start_pattern}(.*?){think_tag_end_pattern}",
                            full_llm_output, re.IGNORECASE | re.DOTALL)
            if match:
                think_block_content = match.group(1).strip()
                final_response_content = full_llm_output[match.end():].strip()
                logger.debug(f"Extracted think block:\n{think_block_content}")
                logger.debug(f"Final response after think block:\n{final_response_content}")
            else:
                logger.warning("No <think> block found in output.")
        except Exception as e:
            logger.error(f"Regex extraction failed: {e}", exc_info=True)

        return think_block_content, final_response_content


    def _invoke_llm(self, prompt_template_str: str, context_vars: dict) -> tuple[str, str, str, str, Optional[float]]:
            """
            Invokes LLM, logs interaction, separates think block, and captures invoke duration.
            Returns: (final_response_str, raw_response_str, think_block_str, interaction_id_str, invoke_duration_float_sec)
            """
            interaction_id = uuid.uuid4().hex

            prompt_template = PromptTemplate.from_template(prompt_template_str)
            rendered_prompt = ""
            try:
                prompt_value = prompt_template.format_prompt(**context_vars)
                rendered_prompt = prompt_value.to_string()
            except Exception as e:
                logger.error(f"Error rendering prompt for agent {self.name}, ID {interaction_id}: {e}", exc_info=True)
                rendered_prompt = f"Error rendering prompt (ID: {interaction_id}): {e}"

            chain = prompt_template | self.llm
            
            raw_response_from_llm = ""
            think_block = ""
            final_response = ""
            error_message = ""
            invoke_duration: Optional[float] = None # To store time for chain.invoke()

            llm_call_start_time = time.perf_counter() # Start timer before invoke
            try:
                log_context_vars = {k: (str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in context_vars.items()}
                # logger.debug(f"Invoking LLM (ID: {interaction_id}) for agent {self.name} with vars: {log_context_vars}")
                
                llm_output = chain.invoke(context_vars)
                
                raw_response_from_llm = str(llm_output) if not isinstance(llm_output, str) else llm_output
                # print(raw_response_from_llm)
                
            except Exception as e:
                invoke_duration = time.perf_counter() - llm_call_start_time # Measure time even on error
                logger.error(f"LLM invocation (ID: {interaction_id}) failed for agent {self.name} after {invoke_duration:.4f}s: {e}", exc_info=True)
                error_message = str(e)
                # _extract_think_and_final_response might still be useful if raw_response_from_llm has partial data
                think_block = self._extract_think_and_final_response(raw_response_from_llm).think_block_content
                final_response = self._extract_think_and_final_response(raw_response_from_llm).final_response_content

                raise RuntimeError(f"LLM call (ID: {interaction_id}) failed for {self.name}") from e
            
            invoke_duration = time.perf_counter() - llm_call_start_time # Measure time on success
            # logger.info(f"LLM raw response (ID: {interaction_id}) for {self.name} (invoke took {invoke_duration:.4f}s): {llm_output[:200]}...")
            think_block, final_response = self._extract_think_and_final_response(raw_response_from_llm)
            logger.info(f"LLM raw response {think_block}")


            return final_response, raw_response_from_llm, think_block, interaction_id, invoke_duration

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
        # logger.info(f"Agent '{self.name}' starting with context snippet: '{context_snippet}'")
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