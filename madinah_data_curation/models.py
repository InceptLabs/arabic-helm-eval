"""Pydantic response models for structured output in synthetic data generation."""

from typing import Literal

from pydantic import BaseModel


class MCQResponse(BaseModel):
    question: str
    options: dict[str, str]
    answer: Literal["أ", "ب", "ج", "د"]


class DialogueMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class DialogueResponse(BaseModel):
    messages: list[DialogueMessage]
