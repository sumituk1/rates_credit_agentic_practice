"""Smoke test — requires Ollama running: ollama serve && ollama pull llama3.2"""
import pytest
from langchain_core.messages import HumanMessage

from models.llm import get_llm


@pytest.mark.integration
def test_llama_connection():
    """Verify Ollama is reachable and returns a non-empty response."""
    llm = get_llm()
    response = llm.invoke([HumanMessage(content="Return the word OK and nothing else.")])
    text = response.content if hasattr(response, "content") else str(response)
    assert "OK" in text.upper()