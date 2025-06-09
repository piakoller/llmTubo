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
from langchain.chains import LLMChain
import langfuse

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


    def _invoke_llm(
        self,
        template: str,
        variables: dict,
        attachments: list[str] | None = None,
        **kwargs
    ):
        """
        Invokes LLM, logs interaction, separates think block, and captures invoke duration.
        Returns: (final_response_str, raw_response_str, think_block_str, interaction_id_str, invoke_duration_float_sec)
        """
        interaction_id = uuid.uuid4().hex

        # Langfuse: Trace anlegen
        trace = langfuse.Trace(
            name=f"{self.name}_LLM_Call",
            metadata={
                "agent": self.name,
                "interaction_id": interaction_id,
                "attachments": attachments,
                "variables": {k: (str(v)[:200] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in variables.items()}
            }
        )

        # Prepare the prompt template
        prompt = PromptTemplate(
            template=template,
            input_variables=list(variables.keys())
        )
        chain = prompt | self.llm
        
        llm_kwargs = {}
        if attachments:
            llm_kwargs["attachments"] = attachments

        raw_response_from_llm = ""
        think_block = ""
        final_response = ""
        error_message = ""
        invoke_duration: Optional[float] = None

        llm_call_start_time = time.perf_counter()
        try:
            # Optionally log input variables (shortened for large strings)
            log_context_vars = {k: (str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in variables.items()}
            logger.debug(f"Invoking LLM for agent {self.name}, ID {interaction_id} with variables: {log_context_vars} and attachments: {attachments}")

            # Call the LLM with attachments if provided
            llm_output = chain.invoke(variables, **llm_kwargs)
            raw_response_from_llm = str(llm_output) if not isinstance(llm_output, str) else llm_output
            
            # Langfuse: Output loggen
            prompt_obs.end(
                output=raw_response_from_llm,
                metadata={"duration": time.perf_counter() - llm_call_start_time}
            )
            trace.end()

        except Exception as e:
            prompt_obs.end(
                output=str(e),
                metadata={"error": True}
            )
            trace.end()
            invoke_duration = time.perf_counter() - llm_call_start_time
            logger.error(f"LLM invocation (ID: {interaction_id}) failed for agent {self.name} after {invoke_duration:.4f}s: {e}", exc_info=True)
            error_message = str(e)
            # Try to extract think/final even from partial output
            think_block, final_response = self._extract_think_and_final_response(raw_response_from_llm)
            raise RuntimeError(f"LLM call (ID: {interaction_id}) failed for {self.name}") from e

        invoke_duration = time.perf_counter() - llm_call_start_time
        think_block, final_response = self._extract_think_and_final_response(raw_response_from_llm)
        logger.info(f"LLM raw response (ID: {interaction_id}) think block: {think_block}")

        return final_response, raw_response_from_llm, think_block, interaction_id, invoke_duration
        
class AgentRunner:
    """Handles the execution of an agent, capturing results, runtime, and exceptions."""
    def __init__(self, agent: Agent, name: str, context: str, attachments: List[str] | None = None):
        self.agent = agent
        self.name = name
        self.context = context
        self.attachments = attachments # Store attachments here
        self.result: Optional[Tuple[str, str, str | None, str, float] | List[Any]] = None # Updated return type hint
        self.exception: Optional[Exception] = None
        self.runtime: float = 0.0
        logger.debug(f"AgentRunner for '{self.name}' initialized.")

    def run(self):
        """Executes the agent's respond method."""
        start_time = time.perf_counter()
        try:
            # Pass attachments to the agent's respond method
            self.result = self.agent.respond(self.context, attachments=self.attachments)
            logger.debug(f"Agent '{self.name}' finished execution.")
        except Exception as e:
            logger.error(f"Agent '{self.name}' encountered an exception.", exc_info=True)
            self.exception = e
        finally:
            self.runtime = time.perf_counter() - start_time