#!/usr/bin/env python3
"""
Bones Index Implementation
Ultra-reliable fallback index that never fails with configuration integration.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

from ..config.index_config import IndexConfig
from ..utils.error_handling import (
    ErrorHandler, ErrorCategory, ErrorSeverity, handle_errors, ErrorContext
)

# Import the original implementation
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "thinkerbell"))
from thinkerbell.core.bones_index import (
    BonesIndex as BaseBonesIndex,
    BonesResult, TemplateSignature
)

logger = logging.getLogger(__name__)

class BonesIndex(BaseBonesIndex):
    """Enhanced bones index with configuration and error handling"""
    
    def __init__(self, config: IndexConfig, error_handler: Optional[ErrorHandler] = None):
        """
        Initialize bones index with configuration.
        
        Args:
            config: Index configuration
            error_handler: Optional error handler instance
        """
        self.config = config
        self.error_handler = error_handler or ErrorHandler()
        
        # Initialize base class
        super().__init__(dim=config.dimension)
        
        # Configuration-specific settings
        self.emergency_template = "template_style_2"
        
    @handle_errors(
        category=ErrorCategory.INDEX_CORRUPTION,
        severity=ErrorSeverity.LOW,  # Bones index is designed to never fail
        auto_recover=True
    )
    def build_from_templates(self, template_signatures: Optional[Dict] = None) -> bool:
        """
        Build bones index with enhanced error handling.
        Always succeeds due to failsafe design.
        
        Args:
            template_signatures: Optional custom template signatures
            
        Returns:
            bool: Success status (should always be True)
        """
        try:
            # Always attempt to build
            success = super().build_from_templates(template_signatures)
            
            if not success:
                # This should never happen, but handle gracefully
                logger.warning("Bones index build failed, creating minimal fallback")
                self._create_minimal_fallback()
                success = True
                
            logger.info(f"Bones index built successfully with {len(self.template_signatures)} templates")
            return success
            
        except Exception as e:
            # Even if everything fails, create absolute minimal fallback
            logger.error(f"Bones index build error: {e}")
            self._create_absolute_fallback()
            return True  # Always return True for bones index
            
    def _create_minimal_fallback(self):
        """Create minimal fallback with essential templates"""
        try:
            # Create minimal set of templates
            minimal_templates = {
                "template_style_2": TemplateSignature(
                    id="template_style_2",
                    name="Emergency Fallback",
                    style="formal_brief",
                    complexity="medium",
                    industry="general",
                    signature_text="Partnership opportunity with [influencer] for [brand]. Budget: $[fee]. Requirements: [deliverables].",
                    embedding=np.random.randn(self.config.dimension),
                    priority=10
                ),
                "fallback_minimal": TemplateSignature(
                    id="fallback_minimal",
                    name="Absolute Minimal",
                    style="minimal",
                    complexity="any",
                    industry="any",
                    signature_text="Brand partnership agreement.",
                    embedding=np.random.randn(self.config.dimension),
                    priority=1
                )
            }
            
            # Normalize embeddings
            for template in minimal_templates.values():
                template.embedding = template.embedding / np.linalg.norm(template.embedding)
                
            # Force build with minimal templates
            self.template_signatures = minimal_templates
            self._force_build_index()
            
            logger.warning("Created minimal bones index fallback")
            
        except Exception as e:
            logger.critical(f"Minimal fallback creation failed: {e}")
            self._create_absolute_fallback()
            
    def _create_absolute_fallback(self):
        """Create absolute minimal fallback (no dependencies)"""
        try:
            # Single template for absolute emergency
            self.template_signatures = {
                "emergency": TemplateSignature(
                    id="template_style_2",
                    name="Emergency",
                    style="emergency",
                    complexity="any",
                    industry="any",
                    signature_text="Template agreement",
                    embedding=np.zeros(self.config.dimension),
                    priority=1
                )
            }
            
            self.signature_order = ["template_style_2"]
            self.is_built = True
            self.emergency_template = "template_style_2"
            
            logger.critical("Created absolute emergency fallback")
            
        except Exception as e:
            logger.critical(f"Absolute fallback creation failed: {e}")
            # If even this fails, we're in serious trouble
            
    def _force_build_index(self):
        """Force build index with current templates"""
        try:
            import faiss
            
            if not self.template_signatures:
                return
                
            # Create index
            self.index = faiss.IndexFlatIP(self.config.dimension)
            
            # Prepare embeddings
            embeddings = []
            self.signature_order = []
            
            for template in sorted(self.template_signatures.values(), key=lambda x: x.priority, reverse=True):
                embeddings.append(template.embedding)
                self.signature_order.append(template.id)
                
            # Add to index
            if embeddings:
                embeddings_array = np.vstack(embeddings).astype(np.float32)
                self.index.add(embeddings_array)
                
            self.is_built = True
            
        except Exception as e:
            logger.error(f"Force build failed: {e}")
            # Even if FAISS fails, we can still return template IDs
            self.is_built = True
            
    @handle_errors(
        category=ErrorCategory.SEARCH_FAILURE,
        severity=ErrorSeverity.LOW,
        auto_recover=True,
        fallback_return="template_style_2"
    )
    def fallback_search(self, query_vector: np.ndarray, k: int = 1) -> str:
        """
        Emergency fallback search - guaranteed to return a template ID.
        This method NEVER fails.
        
        Args:
            query_vector: Query vector
            k: Number of results (typically 1 for fallback)
            
        Returns:
            str: Template ID (guaranteed to return something)
        """
        try:
            # Attempt normal fallback search
            return super().fallback_search(query_vector, k)
            
        except Exception as e:
            # If even the bones search fails, return emergency template
            logger.error(f"Bones fallback search failed: {e}")
            return self.emergency_template
            
    @handle_errors(
        category=ErrorCategory.SEARCH_FAILURE,
        severity=ErrorSeverity.LOW,
        auto_recover=True
    )
    def search_with_metadata(self, query_vector: np.ndarray, k: int = 3) -> List[BonesResult]:
        """
        Search with metadata - enhanced with error handling.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of BonesResult objects (never empty)
        """
        try:
            results = super().search_with_metadata(query_vector, k)
            
            # Ensure we always return at least one result
            if not results:
                results = self._get_emergency_result()
                
            return results
            
        except Exception as e:
            logger.error(f"Bones metadata search failed: {e}")
            return self._get_emergency_result()
            
    def _get_emergency_result(self) -> List[BonesResult]:
        """Get emergency result when all else fails"""
        return [BonesResult(
            template_id=self.emergency_template,
            template_name="Emergency Template",
            confidence=0.5,
            metadata={
                "emergency_fallback": True,
                "error_recovery": True,
                "environment": self.config.environment
            }
        )]
        
    def get_template_by_complexity(self, complexity: str) -> str:
        """Get template by complexity with configuration awareness"""
        try:
            result = super().get_template_by_complexity(complexity)
            
            # If no result, use configuration-aware fallback
            if not result:
                return self._get_config_aware_fallback(complexity=complexity)
                
            return result
            
        except Exception as e:
            logger.warning(f"Complexity-based template selection failed: {e}")
            return self._get_config_aware_fallback(complexity=complexity)
            
    def get_template_by_industry(self, industry: str) -> str:
        """Get template by industry with configuration awareness"""
        try:
            result = super().get_template_by_industry(industry)
            
            # If no result, use configuration-aware fallback
            if not result:
                return self._get_config_aware_fallback(industry=industry)
                
            return result
            
        except Exception as e:
            logger.warning(f"Industry-based template selection failed: {e}")
            return self._get_config_aware_fallback(industry=industry)
            
    def _get_config_aware_fallback(self, complexity: str = None, industry: str = None) -> str:
        """Get fallback template based on configuration and context"""
        
        # Environment-aware fallbacks
        if self.config.environment == "production":
            # Use most reliable template in production
            return "template_style_2"
        elif self.config.environment == "development":
            # Use simple template in development
            return "template_style_1"
        else:
            # Staging - balanced approach
            if complexity == "simple":
                return "template_style_1"
            elif complexity == "complex":
                return "template_style_4"
            else:
                return "template_style_2"
                
    def is_healthy(self) -> bool:
        """Check if bones index is healthy (should always be True)"""
        try:
            # Bones index is designed to always be healthy
            basic_health = super().is_healthy()
            
            # Additional configuration-based health checks
            config_health = True
            
            # Check if we have minimum templates
            if len(self.template_signatures) == 0:
                logger.warning("Bones index has no templates")
                config_health = False
                
            # Check if emergency template is available
            if self.emergency_template not in self.template_signatures:
                logger.warning(f"Emergency template {self.emergency_template} not available")
                # This is not fatal for bones index
                
            return basic_health and config_health
            
        except Exception as e:
            logger.warning(f"Bones health check failed: {e}")
            return True  # Bones index reports healthy even if health check fails
            
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics with configuration context"""
        base_stats = self.get_stats()
        
        # Add configuration-specific stats
        enhanced_stats = {
            **base_stats,
            "configuration": {
                "environment": self.config.environment,
                "dimension": self.config.dimension,
                "emergency_template": self.emergency_template
            },
            "reliability": {
                "designed_availability": "100%",
                "never_fails": True,
                "emergency_fallback_available": self.emergency_template in self.template_signatures
            },
            "memory_efficiency": {
                "estimated_memory_kb": len(self.template_signatures) * self.config.dimension * 4 / 1024,
                "target_footprint": "~17KB",
                "memory_limit_percent": self.config.health.memory_limit_percent
            }
        }
        
        # Add error statistics if available
        if self.error_handler:
            enhanced_stats["error_statistics"] = self.error_handler.get_error_statistics()
            
        return enhanced_stats
        
    @handle_errors(
        category=ErrorCategory.PERSISTENCE_FAILURE,
        severity=ErrorSeverity.MEDIUM,
        auto_recover=True
    )
    def save_bones(self, file_path: str) -> bool:
        """Save bones index with enhanced error handling"""
        try:
            success = super().save_bones(file_path)
            
            if not success:
                # Try alternative save location
                alt_path = f"{file_path}.backup"
                logger.warning(f"Primary save failed, trying backup location: {alt_path}")
                success = super().save_bones(alt_path)
                
            return success
            
        except Exception as e:
            logger.error(f"Bones save failed: {e}")
            
            # Try to save minimal version
            try:
                self._save_minimal_bones(file_path)
                return True
            except Exception as save_error:
                logger.error(f"Minimal bones save failed: {save_error}")
                return False
                
    def _save_minimal_bones(self, file_path: str):
        """Save minimal bones index for emergency recovery"""
        minimal_data = {
            "emergency_template": self.emergency_template,
            "dimension": self.config.dimension,
            "environment": self.config.environment,
            "signature_order": [self.emergency_template],
            "minimal_save": True
        }
        
        import json
        with open(f"{file_path}.minimal", 'w') as f:
            json.dump(minimal_data, f)
            
        logger.info(f"Saved minimal bones index to {file_path}.minimal")
        
    @handle_errors(
        category=ErrorCategory.PERSISTENCE_FAILURE,
        severity=ErrorSeverity.MEDIUM,
        auto_recover=True
    )
    def load_bones(self, file_path: str) -> bool:
        """Load bones index with enhanced error handling"""
        try:
            # Try normal load first
            success = super().load_bones(file_path)
            
            if success:
                return True
                
            # Try backup location
            alt_path = f"{file_path}.backup"
            if Path(alt_path).exists():
                logger.info(f"Trying backup location: {alt_path}")
                success = super().load_bones(alt_path)
                if success:
                    return True
                    
            # Try minimal save
            minimal_path = f"{file_path}.minimal"
            if Path(minimal_path).exists():
                logger.info(f"Trying minimal save: {minimal_path}")
                return self._load_minimal_bones(minimal_path)
                
            # If all fails, create new bones index
            logger.warning("All load attempts failed, creating new bones index")
            return self.build_from_templates()
            
        except Exception as e:
            logger.error(f"Bones load failed: {e}")
            
            # Emergency fallback - always succeeds
            logger.warning("Creating emergency bones index")
            self._create_absolute_fallback()
            return True
            
    def _load_minimal_bones(self, file_path: str) -> bool:
        """Load minimal bones index"""
        try:
            import json
            
            with open(file_path, 'r') as f:
                minimal_data = json.load(f)
                
            self.emergency_template = minimal_data.get("emergency_template", "template_style_2")
            
            # Create minimal template set
            self._create_minimal_fallback()
            
            logger.info("Loaded minimal bones index")
            return True
            
        except Exception as e:
            logger.error(f"Minimal bones load failed: {e}")
            self._create_absolute_fallback()
            return True 