"""
Schema definitions for tb_dataset
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Literal
import json


SectionType = Literal["full", "header", "body", "footer"]


@dataclass
class DocumentSample:
    """
    Data structure for processed document samples.
    
    Represents a document section with its metadata, cleaned text,
    and chunked content ready for JSONL export.
    """
    source_path: str
    doc_id: str
    section: SectionType
    text: str
    char_len: int
    tok_len: int
    chunks: List[str]
    needs_chunking: bool
    template_hint: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict) -> "DocumentSample":
        """Create instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "DocumentSample":
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> bool:
        """
        Validate the sample data integrity.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not self.source_path:
            raise ValueError("source_path cannot be empty")
        
        if not self.doc_id:
            raise ValueError("doc_id cannot be empty")
        
        if self.section not in ["full", "header", "body", "footer"]:
            raise ValueError(f"Invalid section: {self.section}")
        
        if self.char_len != len(self.text):
            raise ValueError(f"char_len mismatch: {self.char_len} != {len(self.text)}")
        
        if self.tok_len <= 0:
            raise ValueError("tok_len must be positive")
        
        if not isinstance(self.chunks, list):
            raise ValueError("chunks must be a list")
        
        if self.needs_chunking and len(self.chunks) <= 1:
            raise ValueError("needs_chunking=True but only one chunk found")
        
        return True 