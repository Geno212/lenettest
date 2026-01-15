"""Shared LLM handles for non-class graph nodes.

LangGraph node functions in this repo are plain callables (not instantiated with the LLM).
To avoid circular imports (core graph importing assistants importing the graph), we keep
LLM handles in this small module and set them during graph build.
"""

from __future__ import annotations

from typing import Any, Optional


design_intent_llm: Optional[Any] = None
cross_phase_llm: Optional[Any] = None


def set_llms(*, design_intent: Any, cross_phase: Any) -> None:
    global design_intent_llm, cross_phase_llm
    design_intent_llm = design_intent
    cross_phase_llm = cross_phase
