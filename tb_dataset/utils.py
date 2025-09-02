"""
Utility functions for tb_dataset package.

Provides helper functions for file operations, caching, and model loading
with offline fallback capabilities.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib

logger = logging.getLogger(__name__)

# Environment-configurable model names
MODEL_NAME = os.getenv("TB_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOKENIZER_NAME = os.getenv("TB_TOKENIZER_MODEL", MODEL_NAME)

# Fail if they diverge to ensure consistency
if TOKENIZER_NAME != MODEL_NAME:
    logger.warning(f"⚠️  Model divergence detected: TB_EMBED_MODEL={MODEL_NAME}, TB_TOKENIZER_MODEL={TOKENIZER_NAME}")
    logger.warning("This configuration may cause embedding/tokenization mismatches")
    # Uncomment the next line to fail hard:
    # raise ValueError("TB_EMBED_MODEL and TB_TOKENIZER_MODEL must be the same")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cached_sentence_transformer(model_name: str = None, device: str = None, offline_mode: bool = False):
    """
    Get sentence transformer with robust offline loading and caching.
    
    Args:
        model_name: Name of the sentence transformer model (uses MODEL_NAME if None)
        device: Device to use ('cpu', 'cuda', etc.) or None for auto-detection
        offline_mode: Force offline-only mode (no downloads)
        
    Returns:
        SentenceTransformer model or None if unavailable
    """
    if model_name is None:
        model_name = MODEL_NAME
    try:
        from sentence_transformers import SentenceTransformer
        from .device_utils import get_sentence_transformer_device
        
        # Get device for GPU acceleration
        model_device = get_sentence_transformer_device(device)
        
        # Try different loading strategies
        model = None
        cache_dir = Path("./hf_cache")
        
        # Strategy 1: Try local cache first
        if cache_dir.exists():
            try:
                model = SentenceTransformer(model_name, cache_folder=str(cache_dir), device=model_device)
                logger.info(f"✅ Loaded SentenceTransformer from cache: {model_name} on device: {model_device}")
                logger.info(f"📁 Model path: {cache_dir / model_name}")
                return model
            except Exception as e:
                logger.debug(f"Failed to load from cache: {e}")
        
        # Strategy 2: Try local files only (no download)
        if not offline_mode:
            try:
                # Set up cache directory
                ensure_dir(cache_dir)
                
                model = SentenceTransformer(
                    model_name, 
                    cache_folder=str(cache_dir),
                    device=model_device
                )
                logger.info(f"✅ Loaded/Downloaded SentenceTransformer: {model_name} on device: {model_device}")
                logger.info(f"📁 Model cached to: {cache_dir / model_name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load/download SentenceTransformer {model_name}: {e}")
        
        # Strategy 3: Force offline mode
        try:
            # Check if model exists in standard cache locations
            possible_paths = [
                cache_dir / model_name,
                Path.home() / ".cache" / "huggingface" / "transformers" / model_name,
                Path.home() / ".cache" / "torch" / "sentence_transformers" / model_name.replace("/", "_"),
            ]
            
            for path in possible_paths:
                if path.exists() and any(path.glob("*.json")):  # Basic check for model files
                    try:
                        model = SentenceTransformer(str(path), device=model_device)
                        logger.info(f"✅ Loaded SentenceTransformer from local path: {path} on device: {model_device}")
                        return model
                    except Exception as e:
                        logger.debug(f"Failed to load from {path}: {e}")
            
            # Last resort: try with local_files_only
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            model = SentenceTransformer(model_name, device=model_device)
            logger.info(f"✅ Loaded SentenceTransformer (offline mode): {model_name} on device: {model_device}")
            return model
            
        except Exception as e:
            logger.debug(f"Offline loading failed: {e}")
        
        logger.error(f"❌ Failed to load SentenceTransformer {model_name}")
        logger.error(f"💡 To download models, run: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{MODEL_NAME}')\"")
        return None
        
    except ImportError as e:
        logger.error(f"sentence-transformers not installed: {e}")
        logger.info("Install with: pip install sentence-transformers")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading SentenceTransformer: {e}")
        return None


def get_cached_tokenizer(model_name: str = None, offline_mode: bool = False):
    """
    Get tokenizer with robust offline loading and caching.
    
    Args:
        model_name: Name of the tokenizer model (uses TOKENIZER_NAME if None)
        offline_mode: Force offline-only mode (no downloads)
        
    Returns:
        Tokenizer or None if unavailable
    """
    if model_name is None:
        model_name = TOKENIZER_NAME
    try:
        from transformers import AutoTokenizer
        
        cache_dir = Path("./hf_cache")
        
        # Strategy 1: Try local cache with cache_dir
        if cache_dir.exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    local_files_only=True
                )
                logger.info(f"✅ Loaded cached tokenizer: {model_name}")
                logger.info(f"📁 Tokenizer path: {cache_dir}")
                return tokenizer
            except Exception as e:
                logger.debug(f"Failed to load tokenizer from cache: {e}")
        
        # Strategy 2: Try download if not in offline mode
        if not offline_mode:
            try:
                ensure_dir(cache_dir)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir)
                )
                logger.info(f"✅ Loaded/Downloaded tokenizer: {model_name}")
                logger.info(f"📁 Tokenizer cached to: {cache_dir}")
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load/download tokenizer {model_name}: {e}")
        
        # Strategy 3: Try standard cache locations
        try:
            # Force offline mode
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )
            logger.info(f"✅ Loaded tokenizer (offline mode): {model_name}")
            return tokenizer
        except Exception as e:
            logger.debug(f"Offline tokenizer loading failed: {e}")
        
        logger.error(f"❌ Failed to load tokenizer {model_name}")
        logger.error(f"💡 To download tokenizer, run: python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{TOKENIZER_NAME}')\"")
        return None
        
    except ImportError as e:
        logger.error(f"transformers not installed: {e}")
        logger.info("Install with: pip install transformers")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading tokenizer: {e}")
        return None


def check_model_availability(model_name: str, model_type: str = "sentence_transformer") -> bool:
    """
    Check if a model is available locally without loading it.
    
    Args:
        model_name: Name of the model
        model_type: Type of model ('sentence_transformer' or 'tokenizer')
        
    Returns:
        True if model is available locally
    """
    cache_dir = Path("./hf_cache")
    
    # Check common cache locations
    possible_paths = [
        cache_dir / model_name,
        Path.home() / ".cache" / "huggingface" / "transformers",
        Path.home() / ".cache" / "torch" / "sentence_transformers",
    ]
    
    for path in possible_paths:
        if path.exists():
            # Look for model files
            if model_type == "sentence_transformer":
                if any(path.glob("**/config.json")) or any(path.glob("**/pytorch_model.bin")):
                    return True
            elif model_type == "tokenizer":
                if any(path.glob("**/tokenizer_config.json")) or any(path.glob("**/vocab.txt")):
                    return True
    
    return False


def setup_offline_mode(strict: bool = False):
    """
    Setup environment for offline operation.
    
    Args:
        strict: If True, prevent all network access attempts
    """
    if strict:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
        logger.info("🔒 Configured for STRICT offline operation (no network access)")
    else:
        # More lenient - allow cache access but prefer local files
        os.environ["HF_HUB_OFFLINE"] = "0"  # Allow hub access if needed
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logger.info("🌐 Configured for offline-preferred operation (cache first)")


def get_fallback_tokenizer():
    """
    Get a simple fallback tokenizer that works without downloads.
    
    Returns:
        Simple tokenizer function
    """
    def simple_tokenize(text: str) -> list:
        """Simple word-based tokenization."""
        # Basic tokenization - split on whitespace and punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    return simple_tokenize


def calculate_simple_token_count(text: str) -> int:
    """
    Calculate token count using simple heuristics (offline fallback).
    
    Uses approximation: 1 token ≈ 0.75 words for English text.
    """
    if not text:
        return 0
    
    # Simple word count
    words = len(text.split())
    
    # Approximate token count (transformers typically use ~0.75 words per token)
    estimated_tokens = int(words * 1.33)  # 1/0.75 = 1.33
    
    return estimated_tokens


def save_json(data: Any, filepath: Path, indent: int = 2) -> None:
    """Save data as JSON file."""
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_file_info(path) -> tuple:
    """
    Get normalized path and modification time for a file.
    
    Args:
        path: File path (string or Path object)
        
    Returns:
        Tuple of (normalized_path, mtime)
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    normalized_path = str(path_obj.resolve())
    mtime = path_obj.stat().st_mtime
    
    return normalized_path, mtime


def hash_id(path, mtime: float) -> str:
    """
    Generate a stable hash ID based on file path and modification time.
    
    Args:
        path: File path (string or Path object)
        mtime: File modification time (timestamp)
        
    Returns:
        Hexadecimal hash string (first 16 characters)
    """
    path_str = str(Path(path).resolve())
    content = f"{path_str}:{mtime}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def clear_model_cache():
    """Clear any cached models to free memory."""
    # This is a placeholder for backward compatibility
    logger.info("Model cache clearing requested (using direct instantiation now)") 