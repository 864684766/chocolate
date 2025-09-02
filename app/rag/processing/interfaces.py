from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Protocol


@dataclass
class RawSample:
    bytes: bytes
    meta: Dict[str, Any]


@dataclass
class ProcessedChunk:
    text: str
    meta: Dict[str, Any]


class MediaExtractor(Protocol):
    def extract(self, sample: RawSample) -> Dict[str, Any]:
        """返回 {"text": str, "meta": {...}}"""
        ...


class LanguageProcessor(Protocol):
    def clean(self, text: str) -> str: ...
    def chunk(self, text: str) -> List[str]: ...
    def extract_meta(self) -> Dict[str, Any]: ...


class QualityAssessor(Protocol):
    def score(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]: ...


