#!/usr/bin/env python3
"""
FAISS Template Index Setup Script
Production setup and configuration for the FAISS template matching system.
"""

import os
import sys
import json
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'faiss-cpu',
        'sentence-transformers', 
        'scikit-learn',
        'numpy',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss-cpu':
                import faiss
            elif package == 'sentence-transformers':
                import sentence_transformers
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
                
        except ImportError:
            missing_packages.append(package)
            
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
        
    logger.info("✓ All required dependencies found")
    return True

def create_directory_structure(base_dir: Path):
    """Create necessary directory structure"""
    directories = [
        base_dir / "faiss_indices",
        base_dir / "faiss_indices" / "versions",
        base_dir / "faiss_indices" / "logs",
        base_dir / "thinkerbell" / "core"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
        
def create_config_files(base_dir: Path):
    """Create configuration files"""
    
    # Main configuration
    config = {
        "faiss_system": {
            "model_path": "sentence-transformers/all-MiniLM-L6-v2",
            "index_dir": "faiss_indices",
            "dimension": 384,
            "hnsw_parameters": {
                "M": 32,
                "efConstruction": 200,
                "efSearch": 64
            },
            "memory_limit_mb": 1000,
            "performance_targets": {
                "p95_latency_ms": 200,
                "recall_threshold": 0.95,
                "max_fallback_rate": 0.3
            }
        },
        "operational": {
            "health_monitoring": {
                "enabled": True,
                "interval_seconds": 60,
                "max_history_days": 30
            },
            "persistence": {
                "max_versions": 5,
                "auto_backup": True,
                "cache_ttl_days": 180
            },
            "adaptive_optimization": {
                "enabled": True,
                "auto_rebuild_threshold": 0.5,
                "recall_degradation_threshold": 0.05
            }
        },
        "integration": {
            "template_hierarchy": {
                "simple": ["template_style_1"],
                "medium": ["template_style_2", "template_style_3"], 
                "complex": ["template_style_4", "template_style_5"]
            },
            "fallback_behavior": {
                "use_bones_index": True,
                "emergency_template": "template_style_2",
                "confidence_threshold": 0.3
            }
        }
    }
    
    config_file = base_dir / "faiss_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Created configuration file: {config_file}")
    
    # Template mapping configuration
    template_mapping = {
        "core_templates": {
            "template_style_1": {
                "name": "Simple Casual Email",
                "complexity": "simple",
                "style": "casual_email",
                "industries": ["general", "fashion", "beauty"]
            },
            "template_style_2": {
                "name": "Medium Formal Brief", 
                "complexity": "medium",
                "style": "formal_brief",
                "industries": ["general", "tech", "finance"]
            },
            "template_style_3": {
                "name": "Medium Professional",
                "complexity": "medium", 
                "style": "professional_memo",
                "industries": ["tech", "automotive", "home"]
            },
            "template_style_4": {
                "name": "Complex Detailed Agreement",
                "complexity": "complex",
                "style": "detailed_agreement", 
                "industries": ["beauty", "fashion", "tech"]
            },
            "template_style_5": {
                "name": "Complex Enterprise",
                "complexity": "complex",
                "style": "enterprise_contract",
                "industries": ["finance", "automotive", "tech"]
            }
        },
        "template_features": {
            "simple": {
                "max_fields": 6,
                "typical_length": 100,
                "delivery_formats": ["email", "chat", "sms"]
            },
            "medium": {
                "max_fields": 10,
                "typical_length": 250,
                "delivery_formats": ["email", "document", "proposal"]
            },
            "complex": {
                "max_fields": 15,
                "typical_length": 500,
                "delivery_formats": ["legal_document", "contract", "agreement"]
            }
        }
    }
    
    template_file = base_dir / "template_mapping.json"
    with open(template_file, 'w') as f:
        json.dump(template_mapping, f, indent=2)
        
    logger.info(f"Created template mapping: {template_file}")

def create_integration_wrapper(base_dir: Path):
    """Create integration wrapper for existing pipeline"""
    
    wrapper_code = '''#!/usr/bin/env python3
"""
FAISS Template Integration Wrapper
Integrates FAISS template index with existing Thinkerbell pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import existing components
try:
    from thinkerbell.core.template_manager import TemplateManager as LegacyTemplateManager
except ImportError:
    LegacyTemplateManager = None

# Import FAISS components
from thinkerbell.core.faiss_template_manager import FAISSTemplateManager

class EnhancedTemplateManager:
    """Enhanced template manager with FAISS integration"""
    
    def __init__(self, config_path: Optional[str] = None, use_faiss: bool = True):
        self.use_faiss = use_faiss
        self.config_path = config_path or "faiss_config.json"
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize FAISS manager
        if self.use_faiss:
            self.faiss_manager = FAISSTemplateManager(
                index_dir=self.config['faiss_system']['index_dir'],
                model_path=self.config['faiss_system']['model_path']
            )
            self.faiss_manager.initialize()
        else:
            self.faiss_manager = None
            
        # Initialize legacy manager as fallback
        if LegacyTemplateManager:
            self.legacy_manager = LegacyTemplateManager()
        else:
            self.legacy_manager = None
            
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return {
                'faiss_system': {
                    'index_dir': 'faiss_indices',
                    'model_path': 'sentence-transformers/all-MiniLM-L6-v2'
                },
                'integration': {
                    'fallback_behavior': {
                        'emergency_template': 'template_style_2',
                        'confidence_threshold': 0.3
                    }
                }
            }
            
    def find_best_template(self, extracted_info: Dict[str, Any]) -> tuple[str, float]:
        """Find best template with FAISS or legacy fallback"""
        
        if self.use_faiss and self.faiss_manager:
            try:
                # Try FAISS-based matching
                result = self.faiss_manager.get_template_recommendations(extracted_info)
                return result['best_template_match'], result['match_confidence']
                
            except Exception as e:
                # Fallback to legacy manager
                if self.legacy_manager:
                    return self.legacy_manager.find_best_template(extracted_info)
                else:
                    # Emergency fallback
                    emergency_template = self.config['integration']['fallback_behavior']['emergency_template']
                    return emergency_template, 0.5
                    
        elif self.legacy_manager:
            return self.legacy_manager.find_best_template(extracted_info)
        else:
            # Emergency fallback
            emergency_template = self.config['integration']['fallback_behavior']['emergency_template']
            return emergency_template, 0.5
            
    def load_dataset(self, dataset_path: str) -> bool:
        """Load dataset into FAISS index"""
        if self.faiss_manager:
            return self.faiss_manager.load_contract_dataset(dataset_path)
        return False
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the template system"""
        if self.faiss_manager:
            return self.faiss_manager.get_performance_metrics()
        return {'legacy_mode': True}
'''
    
    wrapper_file = base_dir / "thinkerbell" / "core" / "enhanced_template_manager.py"
    with open(wrapper_file, 'w') as f:
        f.write(wrapper_code)
        
    logger.info(f"Created integration wrapper: {wrapper_file}")

def create_requirements_file(base_dir: Path):
    """Create requirements file for FAISS system"""
    
    requirements = [
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.0.0",
        "scikit-learn>=1.0.0", 
        "numpy>=1.20.0",
        "psutil>=5.8.0",
        "transformers>=4.20.0",
        "torch>=1.10.0"
    ]
    
    requirements_file = base_dir / "requirements_faiss.txt"
    with open(requirements_file, 'w') as f:
        f.write("\\n".join(requirements))
        
    logger.info(f"Created requirements file: {requirements_file}")

def setup_faiss_system(base_dir: str, load_dataset: bool = False, dataset_path: str = None):
    """Main setup function for FAISS system"""
    
    base_path = Path(base_dir)
    logger.info(f"Setting up FAISS Template Index System in: {base_path}")
    
    # Check dependencies
    if not check_dependencies():
        return False
        
    # Create directory structure
    create_directory_structure(base_path)
    
    # Create configuration files
    create_config_files(base_path)
    
    # Create integration wrapper
    create_integration_wrapper(base_path)
    
    # Create requirements file
    create_requirements_file(base_path)
    
    # Initialize FAISS system if requested
    if load_dataset and dataset_path:
        logger.info("Initializing FAISS system with dataset...")
        try:
            sys.path.append(str(base_path))
            from thinkerbell.core.faiss_template_manager import FAISSTemplateManager
            
            manager = FAISSTemplateManager(
                index_dir=str(base_path / "faiss_indices")
            )
            
            if manager.initialize():
                if manager.load_contract_dataset(dataset_path):
                    logger.info("✓ Dataset loaded successfully")
                else:
                    logger.warning("Failed to load dataset")
            else:
                logger.warning("Failed to initialize FAISS manager")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS system: {e}")
            
    logger.info("✓ FAISS Template Index System setup completed")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Install requirements: pip install -r requirements_faiss.txt")
    logger.info("2. Load your dataset: python -c \"from thinkerbell.core.enhanced_template_manager import EnhancedTemplateManager; mgr = EnhancedTemplateManager(); mgr.load_dataset('path/to/dataset')\"")
    logger.info("3. Test the system: python faiss_integration_demo.py")
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Setup FAISS Template Index System")
    parser.add_argument("--base-dir", default=".", help="Base directory for setup")
    parser.add_argument("--load-dataset", action="store_true", help="Load dataset during setup")
    parser.add_argument("--dataset-path", help="Path to dataset for initial loading")
    
    args = parser.parse_args()
    
    success = setup_faiss_system(
        base_dir=args.base_dir,
        load_dataset=args.load_dataset,
        dataset_path=args.dataset_path
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 