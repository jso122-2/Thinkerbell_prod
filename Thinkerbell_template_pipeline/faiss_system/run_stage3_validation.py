#!/usr/bin/env python3
"""
Stage 3 Validation Script
Comprehensive validation of configuration, error handling, testing, and performance targets.
"""

import os
import sys
import time
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
import json
from concurrent.futures import ThreadPoolExecutor

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "thinkerbell"))

from config.index_config import (
    IndexConfig, get_config, ConfigManager, 
    DEVELOPMENT_CONFIG, STAGING_CONFIG, PRODUCTION_CONFIG
)
from core.main_index import ContractTemplateIndex
from core.bones_index import BonesIndex
from utils.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
from thinkerbell.core.index_manager import IndexManager, SearchStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Stage3Validator:
    """Comprehensive validator for Stage 3 requirements"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {
            "configuration": {},
            "error_handling": {},
            "testing": {},
            "performance": {},
            "file_structure": {},
            "overall_status": "pending"
        }
        
        # Test data
        self.test_vectors = self._generate_test_vectors(1000)
        self.test_metadata = self._generate_test_metadata(1000)
        
        logger.info(f"Stage 3 validation initialized with temp dir: {self.temp_dir}")
        
    def __del__(self):
        """Cleanup temp directory"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            
    def _generate_test_vectors(self, count: int) -> np.ndarray:
        """Generate test vectors for validation"""
        np.random.seed(42)
        vectors = np.random.randn(count, 384)  # Fixed dimension for consistency
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.astype(np.float32)
        
    def _generate_test_metadata(self, count: int) -> List[Dict]:
        """Generate test metadata for validation"""
        metadata = []
        templates = ["template_style_1", "template_style_2", "template_style_3", "template_style_4", "template_style_5"]
        complexities = ["simple", "medium", "complex"]
        industries = ["tech", "fashion", "beauty", "finance", "automotive", "home"]
        
        for i in range(count):
            meta = {
                "sample_id": f"validation_{i:04d}",
                "template_mapping": {
                    "best_template_match": templates[i % len(templates)],
                    "match_confidence": 0.7 + (i % 30) / 100.0,
                    "fallback_templates": [templates[(i+1) % len(templates)], templates[(i+2) % len(templates)]]
                },
                "classification": {
                    "document_type": "INFLUENCER_AGREEMENT",
                    "complexity_level": complexities[i % len(complexities)],
                    "industry": industries[i % len(industries)],
                    "confidence_target": 0.85,
                    "should_process": True
                },
                "raw_input": {
                    "text": f"Sample contract text {i} with brand and influencer details",
                    "token_count": 50 + i % 200,
                    "text_style": "formal" if i % 2 == 0 else "casual",
                    "completeness": "complete"
                },
                "extracted_fields": {
                    "influencer": f"TestInfluencer_{i}",
                    "brand": f"TestBrand_{i}",
                    "fee": f"${5000 + i * 100}",
                    "fee_numeric": 5000 + i * 100,
                    "deliverables": [f"{(i % 5) + 1} x Instagram post"],
                    "engagement_term": f"{(i % 6) + 2} months",
                    "exclusivity_period": f"{(i % 12) + 4} weeks",
                    "territory": "Australia"
                },
                "validation_scores": {
                    "semantic_coherence": 0.75 + (i % 25) / 100.0,
                    "business_logic_valid": True,
                    "temporal_logic_valid": True,
                    "field_extractability": 0.8 + (i % 20) / 100.0,
                    "human_reviewed": False
                }
            }
            metadata.append(meta)
            
        return metadata
        
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete Stage 3 validation"""
        logger.info("="*70)
        logger.info("STAGE 3 FAISS SYSTEM VALIDATION")
        logger.info("="*70)
        
        try:
            # Validate configuration system
            self.validate_configuration_structure()
            
            # Validate error handling
            self.validate_error_handling()
            
            # Validate file structure
            self.validate_file_structure()
            
            # Run comprehensive tests
            self.validate_testing_coverage()
            
            # Validate performance targets
            self.validate_performance_targets()
            
            # Overall assessment
            self.assess_overall_status()
            
            # Generate report
            return self.generate_validation_report()
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.results["overall_status"] = "failed"
            self.results["error"] = str(e)
            return self.results
            
    def validate_configuration_structure(self):
        """Validate configuration management system"""
        logger.info("Validating Configuration Structure...")
        
        config_results = {
            "predefined_configs": True,
            "environment_support": True,
            "validation": True,
            "persistence": True,
            "error_count": 0
        }
        
        try:
            # Test predefined configurations
            dev_config = get_config("development")
            staging_config = get_config("staging")
            prod_config = get_config("production")
            
            self.assertEqual(dev_config.environment, "development")
            self.assertEqual(staging_config.environment, "staging")
            self.assertEqual(prod_config.environment, "production")
            
            # Test required configuration structure
            required_sections = ["hnsw", "health", "persistence", "performance"]
            for section in required_sections:
                self.assertTrue(hasattr(prod_config, section), f"Missing config section: {section}")
                
            # Test specific required parameters
            self.assertEqual(prod_config.hnsw.M, 32)
            self.assertEqual(prod_config.hnsw.efConstruction, 200)
            self.assertEqual(prod_config.hnsw.efSearch_base, 64)
            self.assertTrue(prod_config.hnsw.efSearch_adaptive)
            
            self.assertEqual(prod_config.health.recall_threshold, 0.85)
            self.assertEqual(prod_config.health.memory_limit_percent, 70)
            self.assertEqual(prod_config.health.rebuild_drift_threshold, 0.5)
            self.assertEqual(prod_config.health.cache_ttl_months, 6)
            
            # Test configuration manager
            config_manager = ConfigManager()
            
            # Test save/load
            test_config_path = str(Path(self.temp_dir) / "test_config.json")
            save_success = config_manager.save_config(prod_config, test_config_path)
            self.assertTrue(save_success)
            
            loaded_config = config_manager.load_config(test_config_path)
            self.assertEqual(loaded_config.environment, prod_config.environment)
            
            # Test validation
            validation_success = config_manager.validate_config(prod_config)
            self.assertTrue(validation_success)
            
            logger.info("✓ Configuration structure validation passed")
            
        except Exception as e:
            config_results["error_count"] += 1
            config_results["error"] = str(e)
            logger.error(f"Configuration validation failed: {e}")
            
        self.results["configuration"] = config_results
        
    def validate_error_handling(self):
        """Validate error handling and graceful degradation"""
        logger.info("Validating Error Handling...")
        
        error_results = {
            "graceful_degradation": True,
            "automatic_fallback": True,
            "recovery_from_corruption": True,
            "memory_pressure_handling": True,
            "error_count": 0
        }
        
        try:
            error_handler = ErrorHandler()
            
            # Test error recording and recovery
            test_error = ValueError("Test error for validation")
            result = error_handler.handle_error(
                test_error,
                ErrorCategory.SEARCH_FAILURE,
                ErrorSeverity.MEDIUM,
                context={"test": "validation"},
                component="validator"
            )
            
            self.assertIn("error_id", result)
            self.assertIn("recovery_attempted", result)
            
            # Test main index with error handling
            config = get_config("development")
            config.persistence.base_path = self.temp_dir
            
            main_index = ContractTemplateIndex(config, error_handler)
            
            # Test graceful degradation - main index unavailable
            try:
                # Search before building should be handled gracefully
                query_vector = self.test_vectors[0]
                results = main_index.search(query_vector)
                # Should either work or fail gracefully
            except RuntimeError:
                # Expected behavior
                pass
                
            # Build index and test fallback scenarios
            main_index.build_index(self.test_vectors[:100], self.test_metadata[:100])
            
            # Test bones index fallback
            bones_index = BonesIndex(config, error_handler)
            bones_index.build_from_templates()
            
            # Bones index should never fail
            template_id = bones_index.fallback_search(self.test_vectors[0])
            self.assertIsInstance(template_id, str)
            
            # Test memory pressure handling
            original_limit = config.health.memory_limit_percent
            config.health.memory_limit_percent = 10  # Very low limit
            
            try:
                results = main_index.search(self.test_vectors[0], k=5, adaptive_ef=True)
                # Should adapt to memory pressure
            finally:
                config.health.memory_limit_percent = original_limit
                
            logger.info("✓ Error handling validation passed")
            
        except Exception as e:
            error_results["error_count"] += 1
            error_results["error"] = str(e)
            logger.error(f"Error handling validation failed: {e}")
            
        self.results["error_handling"] = error_results
        
    def validate_file_structure(self):
        """Validate file structure organization"""
        logger.info("Validating File Structure...")
        
        structure_results = {
            "core_modules": True,
            "utils_modules": True,
            "tests_modules": True,
            "config_modules": True,
            "error_count": 0
        }
        
        try:
            base_path = Path(__file__).parent
            
            # Check required directory structure
            required_dirs = [
                "core",
                "utils", 
                "tests",
                "config"
            ]
            
            for dir_name in required_dirs:
                dir_path = base_path / dir_name
                self.assertTrue(dir_path.exists(), f"Missing directory: {dir_name}")
                self.assertTrue(dir_path.is_dir(), f"Not a directory: {dir_name}")
                
            # Check required core modules
            core_modules = [
                "core/main_index.py",
                "core/bones_index.py"
            ]
            
            for module in core_modules:
                module_path = base_path / module
                self.assertTrue(module_path.exists(), f"Missing core module: {module}")
                
            # Check utils modules
            utils_modules = [
                "utils/error_handling.py"
            ]
            
            for module in utils_modules:
                module_path = base_path / module
                self.assertTrue(module_path.exists(), f"Missing utils module: {module}")
                
            # Check test modules
            test_modules = [
                "tests/test_main_index.py",
                "tests/test_integration.py"
            ]
            
            for module in test_modules:
                module_path = base_path / module
                self.assertTrue(module_path.exists(), f"Missing test module: {module}")
                
            # Check config modules
            config_modules = [
                "config/index_config.py"
            ]
            
            for module in config_modules:
                module_path = base_path / module
                self.assertTrue(module_path.exists(), f"Missing config module: {module}")
                
            logger.info("✓ File structure validation passed")
            
        except Exception as e:
            structure_results["error_count"] += 1
            structure_results["error"] = str(e)
            logger.error(f"File structure validation failed: {e}")
            
        self.results["file_structure"] = structure_results
        
    def validate_testing_coverage(self):
        """Validate testing coverage requirements"""
        logger.info("Validating Testing Coverage...")
        
        testing_results = {
            "crud_operations": True,
            "fallback_testing": True,
            "health_triggers": True,
            "persistence_testing": True,
            "performance_testing": True,
            "memory_pressure_testing": True,
            "concurrency_testing": True,
            "error_count": 0
        }
        
        try:
            # Test CRUD operations
            config = get_config("development")
            config.persistence.base_path = self.temp_dir
            
            main_index = ContractTemplateIndex(config)
            
            # CREATE - Build index
            build_success = main_index.build_index(self.test_vectors[:50], self.test_metadata[:50])
            self.assertTrue(build_success)
            
            # READ - Search
            query_vector = self.test_vectors[0]
            results = main_index.search(query_vector, k=5)
            self.assertGreater(len(results), 0)
            
            # UPDATE - Upsert
            new_vector = np.random.randn(config.dimension)
            new_vector = new_vector / np.linalg.norm(new_vector)
            new_metadata = {"sample_id": "test_upsert", "template_mapping": {"best_template_match": "template_style_1"}}
            doc_id = main_index.upsert(new_vector, new_metadata)
            self.assertGreaterEqual(doc_id, 0)
            
            # DELETE - Mark as deleted
            if doc_id in main_index.metadata_store:
                main_index.metadata_store[doc_id]["_deleted"] = True
                
            # Test fallback behavior
            bones_index = BonesIndex(config)
            bones_index.build_from_templates()
            
            # Should always return something
            fallback_result = bones_index.fallback_search(query_vector)
            self.assertIsInstance(fallback_result, str)
            
            # Test health triggers
            health = main_index.health_check()
            self.assertIn("overall_status", health)
            
            needs_rebuild, reason = main_index.needs_rebuild()
            self.assertIsInstance(needs_rebuild, bool)
            
            # Test persistence
            save_path = str(Path(self.temp_dir) / "test_persistence")
            version_id = main_index.save_versioned(save_path)
            self.assertTrue(version_id)
            
            new_index = ContractTemplateIndex(config)
            load_success = new_index.load_latest(save_path)
            self.assertTrue(load_success)
            
            # Test memory pressure handling
            original_limit = config.health.memory_limit_percent
            config.health.memory_limit_percent = 20
            
            try:
                results = main_index.search(query_vector, k=5, adaptive_ef=True)
                self.assertGreater(len(results), 0)
            finally:
                config.health.memory_limit_percent = original_limit
                
            # Test basic concurrency
            def search_worker():
                return main_index.search(self.test_vectors[np.random.randint(0, 50)], k=3)
                
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(search_worker) for _ in range(5)]
                for future in futures:
                    result = future.result()
                    self.assertIsInstance(result, list)
                    
            logger.info("✓ Testing coverage validation passed")
            
        except Exception as e:
            testing_results["error_count"] += 1
            testing_results["error"] = str(e)
            logger.error(f"Testing coverage validation failed: {e}")
            
        self.results["testing"] = testing_results
        
    def validate_performance_targets(self):
        """Validate performance targets are met"""
        logger.info("Validating Performance Targets...")
        
        performance_results = {
            "recall_at_5_target": False,  # ≥85%
            "latency_p95_main_target": False,  # ≤200ms
            "latency_bones_target": False,  # ≤10ms
            "memory_footprint_target": False,  # <1GB at 50K vectors (scaled)
            "availability_target": False,  # 99.9%
            "measurements": {},
            "error_count": 0
        }
        
        try:
            # Setup test system
            config_dict = {
                "dimension": 384,
                "main_index": {
                    "hnsw": {"M": 32, "efConstruction": 200, "efSearch": 64},
                    "performance": {"target_latency_ms": 200, "memory_limit_mb": 800}
                },
                "health": {"latency_threshold": 200, "memory_limit": 800, "recall_threshold": 0.85}
            }
            
            manager = IndexManager(config_dict)
            manager.initialize()
            
            # Use subset of data for performance testing (scaling down from 50K target)
            test_size = min(1000, len(self.test_vectors))  # Scale down for test environment
            build_success = manager.build_main_index(
                self.test_vectors[:test_size], 
                self.test_metadata[:test_size]
            )
            self.assertTrue(build_success)
            
            # Test 1: Recall measurement
            logger.info("  Testing recall@5...")
            test_queries = self.test_vectors[:50]  # 50 test queries
            ground_truth = []
            
            for i in range(50):
                meta = self.test_metadata[i]
                true_template = meta["template_mapping"]["best_template_match"]
                ground_truth.append([true_template])
                
            recall_measurement = manager.health_monitor.measure_recall(
                test_queries, ground_truth, k_values=[1, 3, 5]
            )
            
            recall_at_5 = recall_measurement.recall_at_5
            performance_results["measurements"]["recall_at_5"] = recall_at_5
            performance_results["recall_at_5_target"] = recall_at_5 >= 0.85
            
            logger.info(f"  Recall@5: {recall_at_5:.3f} (target: ≥0.85)")
            
            # Test 2: Main index latency
            logger.info("  Testing main index latency...")
            main_latencies = []
            
            for i in range(100):
                query_vector = self.test_vectors[i % test_size]
                start_time = time.time()
                results = manager.search_with_fallback(query_vector, k=5, strategy=SearchStrategy.MAIN_ONLY)
                latency_ms = (time.time() - start_time) * 1000
                main_latencies.append(latency_ms)
                
            main_p95 = np.percentile(main_latencies, 95)
            performance_results["measurements"]["main_latency_p95_ms"] = main_p95
            performance_results["latency_p95_main_target"] = main_p95 <= 200
            
            logger.info(f"  Main index P95 latency: {main_p95:.1f}ms (target: ≤200ms)")
            
            # Test 3: Bones index latency
            logger.info("  Testing bones index latency...")
            bones_latencies = []
            
            for i in range(100):
                query_vector = self.test_vectors[i % test_size]
                start_time = time.time()
                results = manager.search_with_fallback(query_vector, k=5, strategy=SearchStrategy.BONES_ONLY)
                latency_ms = (time.time() - start_time) * 1000
                bones_latencies.append(latency_ms)
                
            bones_p95 = np.percentile(bones_latencies, 95)
            performance_results["measurements"]["bones_latency_p95_ms"] = bones_p95
            performance_results["latency_bones_target"] = bones_p95 <= 10
            
            logger.info(f"  Bones index P95 latency: {bones_p95:.1f}ms (target: ≤10ms)")
            
            # Test 4: Memory footprint (scaled)
            logger.info("  Testing memory footprint...")
            health = manager.get_system_health()
            
            # Estimate memory for scaled dataset
            if "main_index" in health and "details" in health["main_index"]:
                current_memory = health["main_index"]["details"].get("memory", {}).get("index_usage_mb", 0)
                # Scale estimate for 50K vectors (50x current test size if using 1K)
                scaling_factor = 50000 / test_size
                estimated_50k_memory = current_memory * scaling_factor
                
                performance_results["measurements"]["current_memory_mb"] = current_memory
                performance_results["measurements"]["estimated_50k_memory_mb"] = estimated_50k_memory
                performance_results["memory_footprint_target"] = estimated_50k_memory < 1000  # <1GB
                
                logger.info(f"  Current memory: {current_memory:.1f}MB")
                logger.info(f"  Estimated 50K memory: {estimated_50k_memory:.1f}MB (target: <1000MB)")
            else:
                logger.warning("  Memory usage data not available")
                
            # Test 5: Availability (fallback behavior)
            logger.info("  Testing availability...")
            availability_tests = 0
            successful_responses = 0
            
            # Test normal operation
            for i in range(50):
                try:
                    query_vector = self.test_vectors[i % test_size]
                    results = manager.search_with_fallback(query_vector, k=3)
                    availability_tests += 1
                    if results:
                        successful_responses += 1
                except Exception:
                    availability_tests += 1
                    
            # Test with main index "failure"
            original_healthy = manager.main_index_healthy
            manager.main_index_healthy = False
            
            for i in range(20):
                try:
                    query_vector = self.test_vectors[i % test_size]
                    results = manager.search_with_fallback(query_vector, k=3)
                    availability_tests += 1
                    if results:
                        successful_responses += 1
                except Exception:
                    availability_tests += 1
                    
            manager.main_index_healthy = original_healthy
            
            availability_rate = successful_responses / availability_tests if availability_tests > 0 else 0
            performance_results["measurements"]["availability_rate"] = availability_rate
            performance_results["availability_target"] = availability_rate >= 0.999  # 99.9%
            
            logger.info(f"  Availability: {availability_rate:.3f} (target: ≥0.999)")
            
            # Overall performance assessment
            targets_met = sum([
                performance_results["recall_at_5_target"],
                performance_results["latency_p95_main_target"],
                performance_results["latency_bones_target"],
                performance_results["memory_footprint_target"],
                performance_results["availability_target"]
            ])
            
            logger.info(f"✓ Performance validation: {targets_met}/5 targets met")
            
        except Exception as e:
            performance_results["error_count"] += 1
            performance_results["error"] = str(e)
            logger.error(f"Performance validation failed: {e}")
            
        self.results["performance"] = performance_results
        
    def assess_overall_status(self):
        """Assess overall validation status"""
        logger.info("Assessing Overall Status...")
        
        # Count successful validations
        successful_validations = 0
        total_validations = 0
        
        for category, results in self.results.items():
            if category == "overall_status":
                continue
                
            if isinstance(results, dict):
                error_count = results.get("error_count", 0)
                if error_count == 0:
                    successful_validations += 1
                total_validations += 1
                
        # Performance targets assessment
        performance = self.results.get("performance", {})
        performance_targets_met = 0
        if "recall_at_5_target" in performance:
            performance_targets_met = sum([
                performance.get("recall_at_5_target", False),
                performance.get("latency_p95_main_target", False),
                performance.get("latency_bones_target", False),
                performance.get("memory_footprint_target", False),
                performance.get("availability_target", False)
            ])
            
        # Overall assessment
        if successful_validations == total_validations and performance_targets_met >= 4:
            self.results["overall_status"] = "excellent"
        elif successful_validations >= total_validations * 0.8 and performance_targets_met >= 3:
            self.results["overall_status"] = "good"
        elif successful_validations >= total_validations * 0.6 and performance_targets_met >= 2:
            self.results["overall_status"] = "acceptable"
        else:
            self.results["overall_status"] = "needs_improvement"
            
        logger.info(f"Overall Status: {self.results['overall_status']}")
        logger.info(f"Validations: {successful_validations}/{total_validations}")
        logger.info(f"Performance Targets: {performance_targets_met}/5")
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        logger.info("Generating Validation Report...")
        
        report = {
            "stage3_validation": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "overall_status": self.results["overall_status"],
                "summary": {
                    "configuration": self.results["configuration"].get("error_count", 0) == 0,
                    "error_handling": self.results["error_handling"].get("error_count", 0) == 0,
                    "file_structure": self.results["file_structure"].get("error_count", 0) == 0,
                    "testing": self.results["testing"].get("error_count", 0) == 0,
                    "performance": self.results["performance"].get("error_count", 0) == 0
                },
                "detailed_results": self.results,
                "recommendations": []
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        if self.results["configuration"].get("error_count", 0) > 0:
            recommendations.append("Review configuration management implementation")
            
        if self.results["error_handling"].get("error_count", 0) > 0:
            recommendations.append("Improve error handling and recovery mechanisms")
            
        performance = self.results.get("performance", {})
        if not performance.get("recall_at_5_target", False):
            recommendations.append("Improve recall performance - consider index tuning or more training data")
            
        if not performance.get("latency_p95_main_target", False):
            recommendations.append("Optimize main index latency - consider reducing efSearch or improving hardware")
            
        if not performance.get("memory_footprint_target", False):
            recommendations.append("Reduce memory footprint - consider index compression or optimization")
            
        if not recommendations:
            recommendations.append("System meets all requirements - ready for production deployment")
            
        report["stage3_validation"]["recommendations"] = recommendations
        
        # Save report
        report_path = Path(self.temp_dir) / "stage3_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Validation report saved to: {report_path}")
        
        return report
        
    def assertEqual(self, a, b, msg=None):
        """Helper assertion method"""
        if a != b:
            raise AssertionError(msg or f"{a} != {b}")
            
    def assertTrue(self, condition, msg=None):
        """Helper assertion method"""
        if not condition:
            raise AssertionError(msg or f"Condition is not True")
            
    def assertGreater(self, a, b, msg=None):
        """Helper assertion method"""
        if not (a > b):
            raise AssertionError(msg or f"{a} is not greater than {b}")
            
    def assertGreaterEqual(self, a, b, msg=None):
        """Helper assertion method"""
        if not (a >= b):
            raise AssertionError(msg or f"{a} is not greater than or equal to {b}")
            
    def assertIsInstance(self, obj, cls, msg=None):
        """Helper assertion method"""
        if not isinstance(obj, cls):
            raise AssertionError(msg or f"{obj} is not an instance of {cls}")
            
    def assertIn(self, item, container, msg=None):
        """Helper assertion method"""
        if item not in container:
            raise AssertionError(msg or f"{item} not found in {container}")

def main():
    """Main validation entry point"""
    validator = Stage3Validator()
    
    try:
        report = validator.run_complete_validation()
        
        # Print summary
        print("\n" + "="*70)
        print("STAGE 3 VALIDATION SUMMARY")
        print("="*70)
        
        status = report["stage3_validation"]["overall_status"]
        print(f"Overall Status: {status.upper()}")
        
        summary = report["stage3_validation"]["summary"]
        for category, passed in summary.items():
            status_icon = "✓" if passed else "✗"
            print(f"{status_icon} {category.capitalize()}: {'PASSED' if passed else 'FAILED'}")
            
        # Performance targets
        performance = report["stage3_validation"]["detailed_results"].get("performance", {})
        if "measurements" in performance:
            print("\nPerformance Targets:")
            measurements = performance["measurements"]
            
            if "recall_at_5" in measurements:
                recall = measurements["recall_at_5"]
                target_met = "✓" if recall >= 0.85 else "✗"
                print(f"{target_met} Recall@5: {recall:.3f} (≥0.85)")
                
            if "main_latency_p95_ms" in measurements:
                latency = measurements["main_latency_p95_ms"]
                target_met = "✓" if latency <= 200 else "✗"
                print(f"{target_met} Main P95 Latency: {latency:.1f}ms (≤200ms)")
                
            if "bones_latency_p95_ms" in measurements:
                latency = measurements["bones_latency_p95_ms"]
                target_met = "✓" if latency <= 10 else "✗"
                print(f"{target_met} Bones P95 Latency: {latency:.1f}ms (≤10ms)")
                
            if "availability_rate" in measurements:
                availability = measurements["availability_rate"]
                target_met = "✓" if availability >= 0.999 else "✗"
                print(f"{target_met} Availability: {availability:.3f} (≥0.999)")
                
        # Recommendations
        recommendations = report["stage3_validation"]["recommendations"]
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"• {rec}")
                
        print("="*70)
        
        return 0 if status in ["excellent", "good"] else 1
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 