"""
Device detection and management utilities for CUDA acceleration.

This module provides centralized device detection for SentenceTransformers and FAISS
to enable GPU acceleration when available.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Cache for device detection
_detected_device: Optional[str] = None
_torch_available: Optional[bool] = None
_cuda_available: Optional[bool] = None
_faiss_gpu_available: Optional[bool] = None


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    global _torch_available
    if _torch_available is None:
        try:
            import torch
            _torch_available = True
        except ImportError:
            _torch_available = False
            logger.warning("PyTorch not available - falling back to CPU")
    return _torch_available


def is_cuda_available() -> bool:
    """Check if CUDA is available in PyTorch."""
    global _cuda_available
    if _cuda_available is None:
        if not is_torch_available():
            _cuda_available = False
        else:
            try:
                import torch
                _cuda_available = torch.cuda.is_available()
                if _cuda_available:
                    logger.info(f"CUDA available with {torch.cuda.device_count()} device(s)")
                    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("CUDA not available - using CPU")
            except Exception as e:
                logger.warning(f"Error checking CUDA availability: {e}")
                _cuda_available = False
    return _cuda_available


def is_faiss_gpu_available() -> bool:
    """Check if FAISS GPU support is available."""
    global _faiss_gpu_available
    if _faiss_gpu_available is None:
        try:
            import faiss
            # Try to create a GPU resource - this will fail if GPU not available
            res = faiss.StandardGpuResources()
            _faiss_gpu_available = True
            logger.info("FAISS GPU support available")
        except Exception as e:
            _faiss_gpu_available = False
            logger.info(f"FAISS GPU not available: {e}")
    return _faiss_gpu_available


def get_device(force_device: Optional[str] = None) -> str:
    """
    Get the appropriate device for computation.
    
    Args:
        force_device: Force a specific device ('cpu', 'cuda', 'cuda:0', etc.)
                     If None, auto-detect best available device
    
    Returns:
        Device string ('cpu', 'cuda', 'cuda:0', etc.)
    """
    global _detected_device
    
    # Return cached result if no force_device specified
    if force_device is None and _detected_device is not None:
        return _detected_device
    
    # Handle forced device
    if force_device is not None:
        if force_device.startswith('cuda') and not is_cuda_available():
            logger.warning(f"Forced device {force_device} not available, falling back to CPU")
            return 'cpu'
        logger.info(f"Using forced device: {force_device}")
        return force_device
    
    # Auto-detect best device
    if is_cuda_available():
        device = 'cuda'
        logger.info("Auto-detected device: cuda")
    else:
        device = 'cpu'
        logger.info("Auto-detected device: cpu")
    
    # Cache the result
    _detected_device = device
    return device


def get_sentence_transformer_device(force_device: Optional[str] = None) -> str:
    """
    Get device string suitable for SentenceTransformer initialization.
    
    Args:
        force_device: Override device selection
    
    Returns:
        Device string for SentenceTransformer
    """
    device = get_device(force_device)
    logger.info(f"SentenceTransformer will use device: {device}")
    return device


def setup_faiss_gpu(cpu_index, device_id: int = 0):
    """
    Convert a CPU FAISS index to GPU if possible.
    
    Args:
        cpu_index: CPU FAISS index
        device_id: GPU device ID (default: 0)
    
    Returns:
        GPU index if available, otherwise original CPU index
    """
    if not is_faiss_gpu_available():
        logger.info("FAISS GPU not available, using CPU index")
        return cpu_index
    
    try:
        import faiss
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, device_id, cpu_index)
        logger.info(f"Successfully moved FAISS index to GPU device {device_id}")
        return gpu_index
    except Exception as e:
        logger.warning(f"Failed to move FAISS index to GPU: {e}")
        return cpu_index


def log_device_info():
    """Log comprehensive device information."""
    logger.info("=== Device Configuration ===")
    logger.info(f"PyTorch available: {is_torch_available()}")
    
    if is_torch_available():
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {is_cuda_available()}")
        
        if is_cuda_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
    logger.info(f"FAISS GPU available: {is_faiss_gpu_available()}")
    logger.info(f"Selected device: {get_device()}")
    logger.info("=== End Device Configuration ===")


def get_device_from_env() -> Optional[str]:
    """Get device from environment variable THINKERBELL_DEVICE."""
    return os.getenv('THINKERBELL_DEVICE', None) 