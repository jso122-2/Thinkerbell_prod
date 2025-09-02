"""
Text cleaning and normalization module
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import unicodedata


logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Handles text normalization and lightweight PII scrubbing.
    
    Provides methods to clean and normalize text content while preserving
    important formatting and structure.
    """
    
    def __init__(self, 
                 redact_emails: bool = True, 
                 redact_phones: bool = True,
                 redact_ssns: bool = False):
        """
        Initialize the text cleaner.
        
        Args:
            redact_emails: Whether to redact email addresses
            redact_phones: Whether to redact phone numbers
            redact_ssns: Whether to redact SSNs (disabled by default)
        """
        self.redact_emails = redact_emails
        self.redact_phones = redact_phones
        self.redact_ssns = redact_ssns
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Stats tracking
        self.stats = {
            'processed': 0,
            'emails_redacted': 0,
            'phones_redacted': 0,
            'ssns_redacted': 0,
            'whitespace_normalized': 0,
            'quotes_normalized': 0,
            'currency_normalized': 0
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning."""
        
        # Email pattern (basic but effective)
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone patterns (various formats)
        self.phone_patterns = [
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # 123-456-7890, 123.456.7890, 1234567890
            re.compile(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'),    # (123) 456-7890
            re.compile(r'\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'),  # +1-123-456-7890
        ]
        
        # SSN pattern
        self.ssn_pattern = re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b')
        
        # Currency patterns
        self.currency_patterns = [
            re.compile(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b'),  # $1,234.56
            re.compile(r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*dollars?\b', re.IGNORECASE),
            re.compile(r'USD\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b'),
        ]
        
        # Date patterns to preserve
        self.date_patterns = [
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),  # MM/DD/YYYY, MM-DD-YYYY
            re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),    # YYYY/MM/DD, YYYY-MM-DD
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
        ]
        
        # Quote normalization patterns
        self.quote_patterns = [
            (re.compile(r'[\u201C\u201D]'), '"'),  # Smart quotes to regular
            (re.compile(r'[\u2018\u2019]'), "'"),  # Smart apostrophes to regular
            (re.compile(r'\u2026'), '...'),   # Ellipsis normalization
        ]
        
        # Whitespace patterns
        self.whitespace_patterns = [
            (re.compile(r'\s+'), ' '),           # Multiple spaces to single
            (re.compile(r'\n\s*\n\s*\n+'), '\n\n'),  # Multiple newlines to double
            (re.compile(r'[ \t]+\n'), '\n'),     # Trailing whitespace
            (re.compile(r'\n[ \t]+'), '\n'),     # Leading whitespace after newline
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        self.stats['processed'] += 1
        cleaned = text
        
        # Unicode normalization
        cleaned = unicodedata.normalize('NFKC', cleaned)
        
        # Preserve important patterns first (dates, currency)
        preserved_items = []
        cleaned = self._preserve_patterns(cleaned, preserved_items)
        
        # PII redaction
        if self.redact_emails:
            cleaned, email_count = self._redact_pattern(cleaned, self.email_pattern, '[EMAIL]')
            self.stats['emails_redacted'] += email_count
        
        if self.redact_phones:
            phone_count = 0
            for pattern in self.phone_patterns:
                cleaned, count = self._redact_pattern(cleaned, pattern, '[PHONE]')
                phone_count += count
            self.stats['phones_redacted'] += phone_count
        
        if self.redact_ssns:
            cleaned, ssn_count = self._redact_pattern(cleaned, self.ssn_pattern, '[SSN]')
            self.stats['ssns_redacted'] += ssn_count
        
        # Quote normalization
        for pattern, replacement in self.quote_patterns:
            if pattern.search(cleaned):
                cleaned = pattern.sub(replacement, cleaned)
                self.stats['quotes_normalized'] += 1
        
        # Currency normalization
        for pattern in self.currency_patterns:
            matches = pattern.findall(cleaned)
            if matches:
                self.stats['currency_normalized'] += len(matches)
                # Normalize currency format to $X,XXX.XX
                cleaned = pattern.sub(lambda m: f"${m.group(1)}", cleaned)
        
        # Whitespace normalization
        for pattern, replacement in self.whitespace_patterns:
            if pattern.search(cleaned):
                cleaned = pattern.sub(replacement, cleaned)
                self.stats['whitespace_normalized'] += 1
        
        # Restore preserved patterns
        cleaned = self._restore_patterns(cleaned, preserved_items)
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _preserve_patterns(self, text: str, preserved_items: List[Tuple[str, str]]) -> str:
        """
        Preserve important patterns by replacing them with placeholders.
        
        Args:
            text: Input text
            preserved_items: List to store (placeholder, original) pairs
            
        Returns:
            Text with patterns replaced by placeholders
        """
        result = text
        
        # Preserve dates
        for i, pattern in enumerate(self.date_patterns):
            for match in pattern.finditer(text):
                placeholder = f"__DATE_{len(preserved_items)}__"
                preserved_items.append((placeholder, match.group(0)))
                result = result.replace(match.group(0), placeholder, 1)
        
        return result
    
    def _restore_patterns(self, text: str, preserved_items: List[Tuple[str, str]]) -> str:
        """
        Restore preserved patterns from placeholders.
        
        Args:
            text: Text with placeholders
            preserved_items: List of (placeholder, original) pairs
            
        Returns:
            Text with original patterns restored
        """
        result = text
        for placeholder, original in preserved_items:
            result = result.replace(placeholder, original)
        return result
    
    def _redact_pattern(self, text: str, pattern: re.Pattern, replacement: str) -> Tuple[str, int]:
        """
        Redact matches of a pattern with a replacement string.
        
        Args:
            text: Input text
            pattern: Compiled regex pattern
            replacement: String to replace matches with
            
        Returns:
            Tuple of (modified_text, count_of_replacements)
        """
        matches = pattern.findall(text)
        if matches:
            return pattern.sub(replacement, text), len(matches)
        return text, 0
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of text strings to clean
            
        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in texts]
    
    def get_stats(self) -> Dict[str, int]:
        """Get cleaning statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset cleaning statistics."""
        self.stats = {
            'processed': 0,
            'emails_redacted': 0,
            'phones_redacted': 0,
            'ssns_redacted': 0,
            'whitespace_normalized': 0,
            'quotes_normalized': 0,
            'currency_normalized': 0
        }
    
    def validate_text(self, text: str) -> bool:
        """
        Validate that text is suitable for processing.
        
        Args:
            text: Text to validate
            
        Returns:
            True if text is valid for processing
        """
        if not text or not isinstance(text, str):
            return False
        
        # Check for minimum length
        if len(text.strip()) < 10:
            return False
        
        # Check for excessive special characters (might be corrupted)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.5:
            return False
        
        return True
    
    def extract_metadata(self, text: str) -> Dict[str, any]:
        """
        Extract metadata from text during cleaning.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.splitlines()),
            'has_tables': '|' in text and text.count('|') > 5,
            'has_lists': bool(re.search(r'^\s*[•\-\*\d+\.]\s+', text, re.MULTILINE)),
            'language_hint': self._detect_language_hint(text),
            'pii_detected': {
                'emails': bool(self.email_pattern.search(text)),
                'phones': any(pattern.search(text) for pattern in self.phone_patterns),
                'ssns': bool(self.ssn_pattern.search(text)) if self.redact_ssns else False,
            }
        }
        
        return metadata
    
    def _detect_language_hint(self, text: str) -> str:
        """
        Simple language detection hint based on common words.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language hint ('en' for English, 'unknown' for others)
        """
        # Simple English detection
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) < 10:
            return 'unknown'
        
        english_count = sum(1 for word in words[:100] if word in english_words)
        english_ratio = english_count / min(len(words), 100)
        
        return 'en' if english_ratio > 0.1 else 'unknown' 