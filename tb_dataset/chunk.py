"""
Smart chunking module with sentence-aware splitting
"""

import re
import logging
import os
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

# Tokenizer for accurate token counting
from transformers import AutoTokenizer

# Environment-configurable model names
MODEL_NAME = os.getenv("TB_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOKENIZER_NAME = os.getenv("TB_TOKENIZER_MODEL", MODEL_NAME)

# Fail if they diverge to ensure consistency
if TOKENIZER_NAME != MODEL_NAME:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️  Model divergence detected: TB_EMBED_MODEL={MODEL_NAME}, TB_TOKENIZER_MODEL={TOKENIZER_NAME}")
    logger.warning("This configuration may cause embedding/tokenization mismatches")
    # Uncomment the next line to fail hard:
    # raise ValueError("TB_EMBED_MODEL and TB_TOKENIZER_MODEL must be the same")

# NLP libraries for sentence splitting
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None


logger = logging.getLogger(__name__)


class SmartChunker:
    """
    Intelligent text chunking with sentence awareness.
    
    Uses sentence-level splitting with fallback to clause-level when sentences
    are too large. Preserves numbers, dates, and currency amounts.
    """
    
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50, offline_mode: bool = False):
        """
        Initialize smart chunker with fallback capabilities.
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Token overlap between chunks
            offline_mode: Force offline-only mode (no downloads)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.offline_mode = offline_mode
        self.tokenizer = None
        self.nlp = None
        
        # Initialize tokenizer with fallback
        self._init_tokenizer(offline_mode)
        
        # Initialize spaCy model
        self._init_spacy()
        
        self.stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'avg_chunk_length': 0,
            'tokenizer_type': 'transformers' if self.tokenizer else 'fallback'
        }
        
        logger.info(f"SmartChunker initialized with {self.stats['tokenizer_type']} tokenizer")
    
    def _init_tokenizer(self, offline_mode: bool = False):
        """Initialize tokenizer with robust offline fallback."""
        try:
            # Try to get cached tokenizer with enhanced loading
            from .utils import get_cached_tokenizer, setup_offline_mode, check_model_availability
            
            # Check model availability first
            model_name = TOKENIZER_NAME
            if check_model_availability(model_name, "tokenizer"):
                logger.info(f"📋 Tokenizer {model_name} found in cache")
            else:
                logger.warning(f"📋 Tokenizer {model_name} not found in cache")
            
            # Setup offline mode based on parameter
            setup_offline_mode(strict=offline_mode)
            
            self.tokenizer = get_cached_tokenizer(model_name, offline_mode=offline_mode)
            
            if self.tokenizer is None:
                logger.warning("❌ Transformers tokenizer unavailable, using fallback word-based counting")
                if not offline_mode:
                    logger.info(f"💡 Try downloading first: python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{TOKENIZER_NAME}')\"")
            else:
                logger.info("✅ Tokenizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            logger.info("Falling back to simple word-based token counting")
            self.tokenizer = None
    
    def _init_spacy(self):
        """Initialize NLP processor with fallback options."""
        # Try spaCy first
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm") # Use a default model
                logger.info("Initialized spaCy model: en_core_web_sm")
                return
            except OSError:
                logger.warning(f"spaCy model 'en_core_web_sm' not found. Attempting download...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info(f"Downloaded and initialized spaCy model: en_core_web_sm")
                    return
                except Exception as e:
                    logger.warning(f"Failed to download spaCy model: {str(e)}")
        
        # Try NLTK as fallback
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not present
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                # Test NLTK sentence tokenizer
                sent_tokenize("Test sentence. Another sentence.")
                logger.info("Initialized NLTK sentence tokenizer")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK: {str(e)}")
        
        # If both fail, we'll use regex-based sentence splitting
        logger.warning("No NLP library available. Using regex-based sentence splitting.")
    
    def _compile_patterns(self):
        """Compile regex patterns for chunking logic."""
        
        # Patterns for preserving important units
        self.preserve_patterns = [
            # Numbers with currency symbols and commas
            re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?'),
            re.compile(r'\d{1,3}(?:,\d{3})+'),  # Large numbers with commas
            
            # Dates in various formats
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
            
            # Phone numbers
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            re.compile(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'),
            
            # Email addresses
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # URLs
            re.compile(r'https?://[^\s]+'),
            
            # Time expressions
            re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?\b'),
        ]
        
        # Clause splitting patterns (for when sentences are too long)
        self.clause_patterns = [
            re.compile(r';\s+'),     # Semicolons
            re.compile(r':\s+'),     # Colons
            re.compile(r'—\s+'),     # Em dashes
            re.compile(r'--\s+'),    # Double dashes
            re.compile(r',\s+and\s+'),  # Comma + and
            re.compile(r',\s+but\s+'),  # Comma + but
            re.compile(r',\s+or\s+'),   # Comma + or
        ]
        
        # Simple sentence splitting for regex fallback
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])'  # Split on sentence endings followed by capital letters
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text with fallback support.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self.tokenizer:
            try:
                # Use transformers tokenizer if available
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer failed, using fallback: {e}")
                return self._count_tokens_fallback(text)
        else:
            # Use fallback token counting
            return self._count_tokens_fallback(text)
    
    def _count_tokens_fallback(self, text: str) -> int:
        """
        Fallback token counting using simple heuristics.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        from .utils import calculate_simple_token_count
        return calculate_simple_token_count(text)
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using the best available method.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if not text.strip():
            return []
        
        # Use spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                return sentences
            except Exception as e:
                logger.warning(f"spaCy sentence splitting failed: {str(e)}")
        
        # Use NLTK if available
        if NLTK_AVAILABLE:
            try:
                sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
                return sentences
            except Exception as e:
                logger.warning(f"NLTK sentence splitting failed: {str(e)}")
        
        # Fallback to regex
        sentences = [s.strip() for s in self.sentence_pattern.split(text) if s.strip()]
        return sentences
    
    def split_clauses(self, sentence: str) -> List[str]:
        """
        Split a long sentence into clauses.
        
        Args:
            sentence: Sentence to split
            
        Returns:
            List of clauses
        """
        clauses = [sentence]
        
        for pattern in self.clause_patterns:
            new_clauses = []
            for clause in clauses:
                parts = pattern.split(clause)
                if len(parts) > 1:
                    # Keep the delimiter with the first part
                    for i, part in enumerate(parts[:-1]):
                        delimiter_match = pattern.search(clause[len(''.join(parts[:i+1])):])
                        if delimiter_match:
                            new_clauses.append(part + delimiter_match.group())
                        else:
                            new_clauses.append(part)
                    new_clauses.append(parts[-1])
                else:
                    new_clauses.append(clause)
            clauses = new_clauses
        
        # Clean and filter clauses
        clauses = [c.strip() for c in clauses if c.strip()]
        
        # Ensure no clause is too small to be meaningful
        filtered_clauses = []
        for clause in clauses:
            if len(clause.split()) >= 3:  # At least 3 words
                filtered_clauses.append(clause)
            elif filtered_clauses:  # Merge small clauses with previous
                filtered_clauses[-1] += " " + clause
            else:
                filtered_clauses.append(clause)
        
        return filtered_clauses
    
    def preserve_units(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Preserve important units by replacing with placeholders.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (text_with_placeholders, [(placeholder, original)])
        """
        preserved = []
        result = text
        
        for pattern in self.preserve_patterns:
            matches = list(pattern.finditer(result))
            for match in reversed(matches):  # Process in reverse to maintain positions
                placeholder = f"__PRESERVE_{len(preserved)}__"
                preserved.append((placeholder, match.group(0)))
                result = result[:match.start()] + placeholder + result[match.end():]
        
        if preserved:
            return result, preserved
        else:
            return text, []
    
    def restore_units(self, text: str, preserved: List[Tuple[str, str]]) -> str:
        """
        Restore preserved units from placeholders.
        
        Args:
            text: Text with placeholders
            preserved: List of (placeholder, original) pairs
            
        Returns:
            Text with restored units
        """
        result = text
        for placeholder, original in preserved:
            result = result.replace(placeholder, original)
        return result
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into token-limited pieces with smart sentence/clause awareness.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        self.stats['documents_processed'] += 1
        
        # First check if the entire text fits in one chunk
        total_tokens = self.count_tokens(text)
        if total_tokens <= self.max_tokens:
            self.stats['total_chunks_created'] += 1
            return [text.strip()]
        
        # Preserve important units
        preserved_text, preserved_units = self.preserve_units(text)
        
        # Split into sentences
        sentences = self.split_sentences(preserved_text)
        if not sentences:
            return [text.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If sentence is too long, split into clauses
            if sentence_tokens > self.max_tokens:
                clauses = self.split_clauses(sentence)
                
                for clause in clauses:
                    clause_tokens = self.count_tokens(clause)
                    
                    # If clause is still too long, force split by words
                    if clause_tokens > self.max_tokens:
                        words = clause.split()
                        word_chunk = []
                        word_tokens = 0
                        
                        for word in words:
                            word_token_count = self.count_tokens(word)
                            if word_tokens + word_token_count > self.max_tokens and word_chunk:
                                # Save current word chunk
                                chunk_text = ' '.join(word_chunk)
                                chunk_text = self.restore_units(chunk_text, preserved_units)
                                chunks.append(chunk_text.strip())
                                self.stats['total_chunks_created'] += 1
                                
                                word_chunk = [word]
                                word_tokens = word_token_count
                            else:
                                word_chunk.append(word)
                                word_tokens += word_token_count
                        
                        # Add remaining words as a clause
                        if word_chunk:
                            clause = ' '.join(word_chunk)
                            clause_tokens = self.count_tokens(clause)
                    
                    # Try to add clause to current chunk
                    if current_tokens + clause_tokens > self.max_tokens and current_chunk:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk)
                        chunk_text = self.restore_units(chunk_text, preserved_units)
                        chunks.append(chunk_text.strip())
                        self.stats['total_chunks_created'] += 1
                        
                        current_chunk = [clause]
                        current_tokens = clause_tokens
                    else:
                        current_chunk.append(clause)
                        current_tokens += clause_tokens
            else:
                # Regular sentence processing
                if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_text = self.restore_units(chunk_text, preserved_units)
                    chunks.append(chunk_text.strip())
                    self.stats['total_chunks_created'] += 1
                    
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_text = self.restore_units(chunk_text, preserved_units)
            chunks.append(chunk_text.strip())
            self.stats['total_chunks_created'] += 1
        
        # Update average chunk length
        if chunks:
            total_chunk_length = sum(len(chunk) for chunk in chunks)
            self.stats['avg_chunk_length'] = total_chunk_length / len(chunks)
        
        return chunks
    
    def needs_chunking(self, text: str) -> bool:
        """
        Determine if text needs to be chunked.
        
        Args:
            text: Text to check
            
        Returns:
            True if text exceeds max_tokens
        """
        return self.count_tokens(text) > self.max_tokens
    
    def get_stats(self) -> Dict[str, any]:
        """Get chunking statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset chunking statistics."""
        self.stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'avg_chunk_length': 0,
            'tokenizer_type': 'transformers' if self.tokenizer else 'fallback'
        }
    
    def validate_chunks(self, chunks: List[str]) -> bool:
        """
        Validate that chunks meet token limits and quality requirements.
        
        Args:
            chunks: List of text chunks to validate
            
        Returns:
            True if all chunks are valid
        """
        for i, chunk in enumerate(chunks):
            token_count = self.count_tokens(chunk)
            if token_count > self.max_tokens:
                logger.warning(f"Chunk {i} exceeds max_tokens: {token_count} > {self.max_tokens}")
                return False
            
            if len(chunk.strip()) < 10:
                logger.warning(f"Chunk {i} is too short: {len(chunk)} characters")
                return False
        
        return True 