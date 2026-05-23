import os
import json
import re
import base64
import getpass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.globals import set_debug, set_verbose

from experiments.utils import extract_json

# Global LangChain settings
set_debug(False)
set_verbose(False)

@dataclass
class DermAgentModels:
    lab_tech: Any
    triage: Any
    diagnose: Any
    critique: Any
    synthesizer: Any

class LLMManager:
    """
    Centralized manager for LLM configurations and common utilities.
    """
    GPT_MODEL = "gpt-5-mini"
    GEMINI_MODEL = "gemini-2.5-flash"

    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()
        self.models = self._initialize_models()

    def _initialize_models(self) -> DermAgentModels:
        """Initializes and returns agent-specific models based on provider."""
        if self.provider == "openai":
            return DermAgentModels(
                lab_tech=ChatOpenAI(model=self.GPT_MODEL, temperature=0.0, use_responses_api=True),
                triage=ChatOpenAI(model=self.GPT_MODEL, reasoning={"effort": "low"}, temperature=0.2, use_responses_api=True),
                diagnose=ChatOpenAI(model=self.GPT_MODEL, reasoning={"effort": "low"}, temperature=0.3, use_responses_api=True),
                critique=ChatOpenAI(model=self.GPT_MODEL, reasoning={"effort": "low"}, temperature=0.3, use_responses_api=True),
                synthesizer=ChatOpenAI(model=self.GPT_MODEL, reasoning={"effort": "low"}, temperature=0.3, use_responses_api=True)
            )
        elif self.provider == "gemini":
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
            
            common_params = {"max_output_tokens": 1800, "thinking_budget": 1000, "top_p": 0.95}
            return DermAgentModels(
                lab_tech=ChatGoogleGenerativeAI(model=self.GEMINI_MODEL, temperature=0.0, top_p=0.95),
                triage=ChatGoogleGenerativeAI(model=self.GEMINI_MODEL, temperature=0.2, **common_params),
                diagnose=ChatGoogleGenerativeAI(model=self.GEMINI_MODEL, temperature=0.3, **common_params, max_output_tokens=2400),
                critique=ChatGoogleGenerativeAI(model=self.GEMINI_MODEL, temperature=0.3, **common_params),
                synthesizer=ChatGoogleGenerativeAI(model=self.GEMINI_MODEL, temperature=0.3, **common_params)
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encodes an image to a base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    @staticmethod
    def add_trace(state: dict, *, agent: str, role: str, payload: Any):
        """Appends an interaction trace to the state."""
        if "trace" not in state: state["trace"] = []
        state["trace"].append({"agent": agent, "role": role, "payload": payload})

    @staticmethod
    def invoke_llm(model: Any, system_prompt: str, user_message: str) -> Dict[str, Any]:
        """Invokes a specific model and returns parsed JSON."""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        response = model.invoke(messages)
        return extract_json(response.content)
