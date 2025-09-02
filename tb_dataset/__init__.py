"""
tb_dataset: Document ingestion and chunking pipeline

A Python package for ingesting real documents (PDF/DOCX/TXT), normalizing text,
smart-chunking to ≤512 tokens, and emitting JSONL samples.
"""

__version__ = "0.1.0"
__author__ = "DAWN System"

from .schema import DocumentSample
from .ingest import DocumentIngester
from .clean import TextCleaner
from .chunk import SmartChunker
from .generate import SyntheticGenerator, SyntheticSample, SemanticSmoother
from .business_rules import BusinessRuleEngine, DocumentClassifier
from .ood import OODGenerator
from .dedup import DuplicateDetector
from .split import GroupAwareSplitter
from .eda import DatasetAnalyzer
from .template_ingestor import TemplateIngestor, StyleProfile
from .style_influenced_generator import StyleInfluencedGenerator

__all__ = [
    "DocumentSample", "DocumentIngester", "TextCleaner", "SmartChunker",
    "SyntheticGenerator", "SyntheticSample", "SemanticSmoother",
    "BusinessRuleEngine", "DocumentClassifier", "OODGenerator",
    "DuplicateDetector", "GroupAwareSplitter", "DatasetAnalyzer",
    "TemplateIngestor", "StyleProfile", "StyleInfluencedGenerator"
] 