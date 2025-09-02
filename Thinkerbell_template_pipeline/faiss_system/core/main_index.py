#!/usr/bin/env python3
"""
Main FAISS Index Implementation
Production-ready main index with error handling and graceful degradation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from ..config.index_config import IndexConfig
from ..utils.error_handling import (
    ErrorHandler, ErrorCategory, ErrorSeverity, handle_errors, ErrorContext
)

# Import the original implementation
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "thinkerbell"))
from thinkerbell.core.contract_template_index import (
    ContractTemplateIndex as BaseContractTemplateIndex,
    SearchResult, HealthMetrics, IndexMetadata
)

logger = logging.getLogger(__name__)

class ContractTemplateIndex(BaseContractTemplateIndex):
    """Enhanced contract template index with error handling and configuration"""
    
    def __init__(self, config: IndexConfig, error_handler: Optional[ErrorHandler] = None):
        """
        Initialize contract template index with configuration.
        
        Args:
            config: Index configuration
            error_handler: Optional error handler instance
        """
        self.config = config
        self.error_handler = error_handler or ErrorHandler()
        
        # Initialize base class with config values
        base_config = {
            "hnsw": {
                "M": config.hnsw.M,
                "efConstruction": config.hnsw.efConstruction,
                "efSearch": config.hnsw.efSearch_base,
                "max_ef_search": config.hnsw.efSearch_max,
                "min_ef_search": config.hnsw.efSearch_min
            },
            "performance": {
                "target_latency_ms": config.health.latency_p95_threshold_ms,
                "memory_limit_mb": self._calculate_memory_limit(),
                "rebuild_threshold": config.health.rebuild_drift_threshold,
                "recall_threshold": config.health.recall_threshold
            },
            "cache": {
                "enabled": config.performance.cache_enabled,
                "max_size": config.performance.cache_size,
                "ttl_seconds": config.performance.cache_ttl_seconds
            },
            "health": {
                "check_interval_seconds": config.health.monitoring_interval_seconds,
                "performance_window": config.health.performance_window_size
            }
        }
        
        super().__init__(dim=config.dimension, config=base_config)
        
        # Configuration-specific initialization
        self._apply_configuration_settings()
        
    def _calculate_memory_limit(self) -> int:
        """Calculate memory limit based on system and configuration"""
        try:
            import psutil
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            return int(total_memory_mb * self.config.health.memory_limit_percent / 100)
        except ImportError:
            logger.warning("psutil not available, using default memory limit")
            return 800  # Default 800MB
            
    def _apply_configuration_settings(self):
        """Apply configuration-specific settings"""
        # Adaptive efSearch setting
        if hasattr(self, 'index') and self.index and self.config.hnsw.efSearch_adaptive:
            self.index.hnsw.efSearch = self.config.hnsw.efSearch_base
            
    @handle_errors(
        category=ErrorCategory.INDEX_CORRUPTION,
        severity=ErrorSeverity.HIGH,
        auto_recover=True
    )
    def build_index(
        self, 
        vectors: np.ndarray, 
        metadata: List[Dict],
        incremental: bool = False
    ) -> bool:
        """
        Build index with error handling and recovery.
        
        Args:
            vectors: Document vectors
            metadata: Document metadata
            incremental: Whether to add to existing index
            
        Returns:
            bool: Success status
        """
        with ErrorContext(
            self.error_handler,
            category=ErrorCategory.INDEX_CORRUPTION,
            severity=ErrorSeverity.HIGH,
            component="main_index"
        ) as ctx:
            ctx.add_context("operation", "build_index")
            ctx.add_context("vector_count", len(vectors))
            ctx.add_context("incremental", incremental)
            
            # Validate inputs
            self._validate_build_inputs(vectors, metadata)
            
            # Check memory requirements
            estimated_memory = self._estimate_build_memory_requirement(vectors)
            if estimated_memory > self._calculate_memory_limit():
                raise MemoryError(f"Estimated memory requirement ({estimated_memory}MB) exceeds limit")
                
            # Perform build
            return super().build_index(vectors, metadata, incremental)
            
    def _validate_build_inputs(self, vectors: np.ndarray, metadata: List[Dict]):
        """Validate build inputs"""
        if vectors.shape[0] != len(metadata):
            raise ValueError(f"Vector count {vectors.shape[0]} != metadata count {len(metadata)}")
            
        if vectors.shape[1] != self.config.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != expected {self.config.dimension}")
            
        if vectors.shape[0] == 0:
            raise ValueError("Cannot build index with empty vectors")
            
        # Check for NaN or infinite values
        if np.any(~np.isfinite(vectors)):
            raise ValueError("Vectors contain NaN or infinite values")
            
    def _estimate_build_memory_requirement(self, vectors: np.ndarray) -> int:
        """Estimate memory requirement for building index"""
        # Vector storage (float32)
        vector_memory = vectors.shape[0] * vectors.shape[1] * 4
        
        # HNSW graph (estimated)
        graph_memory = vectors.shape[0] * self.config.hnsw.M * 8
        
        # Metadata and overhead (estimated)
        overhead_memory = vectors.shape[0] * 1024  # 1KB per document
        
        total_bytes = vector_memory + graph_memory + overhead_memory
        return int(total_bytes / (1024 * 1024))  # Convert to MB
        
    @handle_errors(
        category=ErrorCategory.SEARCH_FAILURE,
        severity=ErrorSeverity.MEDIUM,
        auto_recover=True,
        fallback_return=[]
    )
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 5,
        filters: Optional[Dict] = None,
        adaptive_ef: bool = True
    ) -> List[SearchResult]:
        """
        Search with error handling and fallback.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filters: Optional metadata filters
            adaptive_ef: Whether to use adaptive efSearch
            
        Returns:
            List of search results
        """
        with ErrorContext(
            self.error_handler,
            category=ErrorCategory.SEARCH_FAILURE,
            severity=ErrorSeverity.MEDIUM,
            component="main_index"
        ) as ctx:
            ctx.add_context("operation", "search")
            ctx.add_context("k", k)
            ctx.add_context("adaptive_ef", adaptive_ef)
            ctx.add_context("main_index", self)
            
            # Validate search inputs
            self._validate_search_inputs(query_vector, k)
            
            # Apply adaptive efSearch if enabled
            if adaptive_ef and self.config.hnsw.efSearch_adaptive:
                self._apply_adaptive_ef_search()
                
            # Perform search
            return super().search(query_vector, k, filters, adaptive_ef)
            
    def _validate_search_inputs(self, query_vector: np.ndarray, k: int):
        """Validate search inputs"""
        if query_vector.shape[0] != self.config.dimension:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} != expected {self.config.dimension}")
            
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
            
        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Query vector contains NaN or infinite values")
            
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
            
    def _apply_adaptive_ef_search(self):
        """Apply adaptive efSearch based on current conditions"""
        try:
            if not self.index:
                return
                
            # Get current system conditions
            memory_pressure = self._get_memory_pressure()
            recent_latency = self._get_recent_average_latency()
            
            # Calculate optimal efSearch
            base_ef = self.config.hnsw.efSearch_base
            optimal_ef = base_ef
            
            # Adjust based on memory pressure
            if memory_pressure > 0.8:
                optimal_ef = max(self.config.hnsw.efSearch_min, int(optimal_ef * 0.7))
            elif memory_pressure < 0.5:
                optimal_ef = min(self.config.hnsw.efSearch_max, int(optimal_ef * 1.2))
                
            # Adjust based on latency
            target_latency = self.config.health.latency_p95_threshold_ms
            if recent_latency > target_latency * 1.5:
                optimal_ef = max(self.config.hnsw.efSearch_min, int(optimal_ef * 0.8))
            elif recent_latency < target_latency * 0.5:
                optimal_ef = min(self.config.hnsw.efSearch_max, int(optimal_ef * 1.1))
                
            # Apply if significantly different
            if abs(optimal_ef - self.index.hnsw.efSearch) >= 4:
                self.index.hnsw.efSearch = optimal_ef
                logger.debug(f"Adaptive efSearch updated to {optimal_ef}")
                
        except Exception as e:
            logger.warning(f"Adaptive efSearch failed: {e}")
            
    def _get_memory_pressure(self) -> float:
        """Get current memory pressure"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.5  # Conservative estimate
            
    def _get_recent_average_latency(self) -> float:
        """Get recent average query latency"""
        if len(self.query_times) >= 10:
            return np.mean(self.query_times[-10:])
        return 0.0
        
    @handle_errors(
        category=ErrorCategory.PERSISTENCE_FAILURE,
        severity=ErrorSeverity.HIGH,
        auto_recover=True
    )
    def save_versioned(self, base_path: str) -> str:
        """
        Save with error handling and backup.
        
        Args:
            base_path: Base directory for saving
            
        Returns:
            str: Version ID of saved index
        """
        with ErrorContext(
            self.error_handler,
            category=ErrorCategory.PERSISTENCE_FAILURE,
            severity=ErrorSeverity.HIGH,
            component="main_index"
        ) as ctx:
            ctx.add_context("operation", "save_versioned")
            ctx.add_context("base_path", base_path)
            
            # Ensure directory exists and is writable
            self._validate_save_path(base_path)
            
            # Perform save
            version_id = super().save_versioned(base_path)
            
            # Cleanup old versions based on config
            self._cleanup_old_versions(base_path)
            
            return version_id
            
    def _validate_save_path(self, base_path: str):
        """Validate save path is accessible"""
        from pathlib import Path
        import os
        
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = path / "test_write"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"Cannot write to {base_path}: {e}")
            
    def _cleanup_old_versions(self, base_path: str):
        """Cleanup old versions based on configuration"""
        try:
            from pathlib import Path
            import shutil
            
            base_dir = Path(base_path)
            if not base_dir.exists():
                return
                
            # Find version directories
            version_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("v_")]
            version_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the specified number of versions
            max_versions = self.config.persistence.backup_versions
            if len(version_dirs) > max_versions:
                for old_dir in version_dirs[max_versions:]:
                    shutil.rmtree(old_dir)
                    logger.info(f"Cleaned up old version: {old_dir.name}")
                    
        except Exception as e:
            logger.warning(f"Version cleanup failed: {e}")
            
    @handle_errors(
        category=ErrorCategory.PERSISTENCE_FAILURE,
        severity=ErrorSeverity.HIGH,
        auto_recover=True
    )
    def load_latest(self, base_path: str) -> bool:
        """
        Load with error handling and validation.
        
        Args:
            base_path: Base directory containing versions
            
        Returns:
            bool: Success status
        """
        with ErrorContext(
            self.error_handler,
            category=ErrorCategory.PERSISTENCE_FAILURE,
            severity=ErrorSeverity.HIGH,
            component="main_index"
        ) as ctx:
            ctx.add_context("operation", "load_latest")
            ctx.add_context("base_path", base_path)
            
            # Validate load path
            self._validate_load_path(base_path)
            
            # Attempt load
            success = super().load_latest(base_path)
            
            if success:
                # Validate loaded index
                self._validate_loaded_index()
                
            return success
            
    def _validate_load_path(self, base_path: str):
        """Validate load path exists and is readable"""
        from pathlib import Path
        
        path = Path(base_path)
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {base_path}")
            
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {base_path}")
            
    def _validate_loaded_index(self):
        """Validate loaded index is consistent"""
        if not self.is_built:
            raise RuntimeError("Index loaded but not marked as built")
            
        if self.index and self.index.ntotal == 0:
            raise RuntimeError("Index loaded but contains no vectors")
            
        # Validate dimension consistency
        if hasattr(self, 'dim') and self.dim != self.config.dimension:
            raise ValueError(f"Loaded index dimension {self.dim} != config dimension {self.config.dimension}")
            
    def get_enhanced_health_check(self) -> Dict[str, Any]:
        """Get enhanced health check with configuration context"""
        base_health = self.health_check()
        
        # Add configuration-specific health checks
        config_health = {
            "configuration": {
                "environment": self.config.environment,
                "hnsw_parameters": {
                    "M": self.config.hnsw.M,
                    "efConstruction": self.config.hnsw.efConstruction,
                    "efSearch_current": self.index.hnsw.efSearch if self.index else None,
                    "efSearch_adaptive": self.config.hnsw.efSearch_adaptive
                },
                "memory_configuration": {
                    "limit_percent": self.config.health.memory_limit_percent,
                    "calculated_limit_mb": self._calculate_memory_limit(),
                    "current_usage_mb": self._estimate_memory_usage()
                },
                "performance_targets": {
                    "recall_threshold": self.config.health.recall_threshold,
                    "latency_threshold_ms": self.config.health.latency_p95_threshold_ms
                }
            },
            "error_statistics": self.error_handler.get_error_statistics() if self.error_handler else {}
        }
        
        # Merge with base health check
        if isinstance(base_health, dict):
            base_health.update(config_health)
            return base_health
        else:
            return config_health
            
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            "dimension": self.config.dimension,
            "environment": self.config.environment,
            "hnsw": {
                "M": self.config.hnsw.M,
                "efConstruction": self.config.hnsw.efConstruction,
                "efSearch_base": self.config.hnsw.efSearch_base,
                "efSearch_adaptive": self.config.hnsw.efSearch_adaptive,
                "efSearch_range": [self.config.hnsw.efSearch_min, self.config.hnsw.efSearch_max]
            },
            "health": {
                "recall_threshold": self.config.health.recall_threshold,
                "memory_limit_percent": self.config.health.memory_limit_percent,
                "latency_threshold_ms": self.config.health.latency_p95_threshold_ms
            },
            "performance": {
                "cache_enabled": self.config.performance.cache_enabled,
                "cache_size": self.config.performance.cache_size,
                "target_qps": self.config.performance.target_qps
            }
        } 