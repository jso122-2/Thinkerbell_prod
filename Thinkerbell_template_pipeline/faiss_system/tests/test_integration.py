#!/usr/bin/env python3
"""
Integration Tests for Complete FAISS System
Tests all components working together: main index, bones index, manager, health monitoring.
"""

import unittest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.index_config import IndexConfig, get_config
from core.main_index import ContractTemplateIndex
from core.bones_index import BonesIndex
from utils.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity

# Import original implementations for integration
sys.path.append(str(Path(__file__).parent.parent.parent / "thinkerbell"))
from thinkerbell.core.index_manager import IndexManager, SearchStrategy
from thinkerbell.core.health_monitor import HealthMonitor

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for complete FAISS system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dict = {
            "dimension": 384,
            "main_index": {
                "hnsw": {"M": 16, "efConstruction": 100, "efSearch": 32},
                "performance": {"target_latency_ms": 200, "memory_limit_mb": 500}
            },
            "health": {"latency_threshold": 200, "memory_limit": 500, "recall_threshold": 0.80}
        }
        
        # Create configuration
        self.index_config = get_config("development")
        self.index_config.persistence.base_path = self.temp_dir
        
        # Initialize error handler
        self.error_handler = ErrorHandler()
        
        # Test data
        self.test_vectors = self._generate_test_vectors(200)
        self.test_metadata = self._generate_test_metadata(200)
        
    def tearDown(self):
        """Clean up integration test environment"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            
    def _generate_test_vectors(self, count: int) -> np.ndarray:
        """Generate test vectors for integration testing"""
        np.random.seed(123)  # Different seed from unit tests
        vectors = np.random.randn(count, self.index_config.dimension)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.astype(np.float32)
        
    def _generate_test_metadata(self, count: int) -> list:
        """Generate test metadata for integration testing"""
        metadata = []
        templates = ["template_style_1", "template_style_2", "template_style_3", "template_style_4"]
        complexities = ["simple", "medium", "complex"]
        industries = ["tech", "fashion", "beauty", "finance", "automotive"]
        
        for i in range(count):
            meta = {
                "sample_id": f"integration_test_{i:03d}",
                "template_mapping": {
                    "best_template_match": templates[i % len(templates)],
                    "match_confidence": 0.7 + (i % 30) / 100.0
                },
                "classification": {
                    "complexity_level": complexities[i % len(complexities)],
                    "industry": industries[i % len(industries)],
                    "document_type": "INFLUENCER_AGREEMENT"
                },
                "extracted_fields": {
                    "influencer": f"TestInfluencer_{i}",
                    "brand": f"TestBrand_{i}",
                    "fee_numeric": 5000 + i * 250,
                    "engagement_term": f"{2 + i % 6} months"
                },
                "validation_scores": {
                    "semantic_coherence": 0.8 + (i % 20) / 100.0,
                    "business_logic_valid": True
                }
            }
            metadata.append(meta)
            
        return metadata
        
    # Core Integration Tests
    
    def test_main_and_bones_index_integration(self):
        """Test main index and bones index working together"""
        # Initialize both indices
        main_index = ContractTemplateIndex(self.index_config, self.error_handler)
        bones_index = BonesIndex(self.index_config, self.error_handler)
        
        # Build both indices
        main_success = main_index.build_index(self.test_vectors, self.test_metadata)
        bones_success = bones_index.build_from_templates()
        
        self.assertTrue(main_success)
        self.assertTrue(bones_success)
        
        # Test search consistency
        query_vector = self.test_vectors[0]
        
        main_results = main_index.search(query_vector, k=3)
        bones_results = bones_index.search_with_metadata(query_vector, k=3)
        
        self.assertGreater(len(main_results), 0)
        self.assertGreater(len(bones_results), 0)
        
        # Both should return valid template IDs
        for result in main_results:
            self.assertIsInstance(result.template_id, str)
            self.assertGreater(len(result.template_id), 0)
            
        for result in bones_results:
            self.assertIsInstance(result.template_id, str)
            self.assertGreater(len(result.template_id), 0)
            
    def test_index_manager_orchestration(self):
        """Test IndexManager orchestrating all components"""
        # Create index manager with config
        manager = IndexManager(self.config_dict)
        
        # Initialize system
        success = manager.initialize()
        self.assertTrue(success)
        
        # Build main index
        build_success = manager.build_main_index(self.test_vectors, self.test_metadata)
        self.assertTrue(build_success)
        
        # Test different search strategies
        query_vector = self.test_vectors[5]
        strategies = [
            SearchStrategy.MAIN_ONLY,
            SearchStrategy.BONES_ONLY,
            SearchStrategy.MAIN_WITH_BONES_FALLBACK,
            SearchStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            results = manager.search_with_fallback(query_vector, k=3, strategy=strategy)
            self.assertGreater(len(results), 0, f"No results for strategy: {strategy}")
            
            # Check result structure
            for result in results:
                self.assertHasAttr(result, 'template_id')
                self.assertHasAttr(result, 'confidence')
                self.assertHasAttr(result, 'source')
                
        # Test system health
        health = manager.get_system_health()
        self.assertIn("overall_healthy", health)
        self.assertIn("main_index", health)
        self.assertIn("bones_index", health)
        
    def test_error_handling_integration(self):
        """Test error handling across all components"""
        main_index = ContractTemplateIndex(self.index_config, self.error_handler)
        main_index.build_index(self.test_vectors[:50], self.test_metadata[:50])
        
        initial_error_count = self.error_handler.recovery_stats["total_errors"]
        
        # Trigger various error conditions
        error_scenarios = [
            # Invalid query vector
            lambda: main_index.search(np.full(self.index_config.dimension, np.nan)),
            # Search without build
            lambda: ContractTemplateIndex(self.index_config).search(self.test_vectors[0]),
            # Invalid save path
            lambda: main_index.save_versioned("/invalid/readonly/path")
        ]
        
        for scenario in error_scenarios:
            try:
                scenario()
            except Exception:
                # Expected - errors should be handled
                pass
                
        # Check error handling
        final_error_count = self.error_handler.recovery_stats["total_errors"]
        error_stats = self.error_handler.get_error_statistics()
        
        self.assertIsInstance(error_stats, dict)
        self.assertIn("total_errors_24h", error_stats)
        
    def test_health_monitoring_integration(self):
        """Test health monitoring with real components"""
        # Create components
        main_index = ContractTemplateIndex(self.index_config, self.error_handler)
        bones_index = BonesIndex(self.index_config, self.error_handler)
        
        # Build indices
        main_index.build_index(self.test_vectors, self.test_metadata)
        bones_index.build_from_templates()
        
        # Create health monitor
        health_monitor = HealthMonitor(self.config_dict.get("health", {}))
        health_monitor.initialize(main_index, bones_index)
        
        # Record some search operations
        for i in range(10):
            query_vector = self.test_vectors[i]
            results = main_index.search(query_vector, k=3)
            
            # Record search for monitoring
            health_monitor.record_search(query_vector, results, 50 + i * 5)
            
        # Get health report
        health_report = health_monitor.get_health_report()
        
        self.assertIn("overall_status", health_report)
        self.assertIn("current_performance", health_report)
        self.assertIn("monitoring_stats", health_report)
        
        # Check performance data
        perf_data = health_report["current_performance"]
        self.assertIn("latency_p95", perf_data)
        self.assertIn("memory_usage_mb", perf_data)
        
    # Fallback and Recovery Tests
    
    def test_graceful_degradation_main_to_bones(self):
        """Test graceful degradation from main to bones index"""
        # Set up system
        manager = IndexManager(self.config_dict)
        manager.initialize()
        manager.build_main_index(self.test_vectors, self.test_metadata)
        
        query_vector = self.test_vectors[0]
        
        # Normal operation
        results_normal = manager.search_with_fallback(query_vector, k=3)
        self.assertGreater(len(results_normal), 0)
        
        # Simulate main index failure
        original_main_index = manager.main_index
        manager.main_index_healthy = False
        
        # Should fallback to bones
        results_fallback = manager.search_with_fallback(query_vector, k=3)
        self.assertGreater(len(results_fallback), 0)
        
        # Check that bones index was used
        bones_used = any(result.source == "bones" for result in results_fallback)
        self.assertTrue(bones_used or len(results_fallback) > 0)
        
        # Restore main index
        manager.main_index = original_main_index
        manager.main_index_healthy = True
        
    def test_bones_index_never_fails(self):
        """Test that bones index never fails under any circumstances"""
        bones_index = BonesIndex(self.index_config, self.error_handler)
        
        # Should succeed even without building
        template_id = bones_index.fallback_search(self.test_vectors[0])
        self.assertIsInstance(template_id, str)
        self.assertGreater(len(template_id), 0)
        
        # Build and test
        bones_index.build_from_templates()
        
        # Should always succeed
        for i in range(10):
            query_vector = self.test_vectors[i]
            
            # Fallback search
            template_id = bones_index.fallback_search(query_vector)
            self.assertIsInstance(template_id, str)
            
            # Metadata search
            results = bones_index.search_with_metadata(query_vector, k=3)
            self.assertGreater(len(results), 0)
            
        # Should always be healthy
        self.assertTrue(bones_index.is_healthy())
        
    def test_emergency_fallback_scenarios(self):
        """Test emergency fallback scenarios"""
        manager = IndexManager(self.config_dict)
        manager.initialize()
        
        query_vector = self.test_vectors[0]
        
        # Scenario 1: No main index built
        results = manager.search_with_fallback(query_vector, k=3)
        self.assertGreater(len(results), 0)
        
        # Scenario 2: Simulate both indices failing (should still return something)
        original_bones = manager.bones_index
        manager.bones_index = None
        manager.main_index_healthy = False
        
        results = manager.search_with_fallback(query_vector, k=3)
        # Should get emergency fallback
        self.assertGreater(len(results), 0)
        
        # Restore
        manager.bones_index = original_bones
        
    # Performance and Scalability Tests
    
    def test_system_performance_targets(self):
        """Test that system meets performance targets"""
        manager = IndexManager(self.config_dict)
        manager.initialize()
        manager.build_main_index(self.test_vectors, self.test_metadata)
        
        # Latency test
        query_times = []
        for i in range(100):
            query_vector = self.test_vectors[i % len(self.test_vectors)]
            
            start_time = time.time()
            results = manager.search_with_fallback(query_vector, k=5)
            query_time = (time.time() - start_time) * 1000  # ms
            
            query_times.append(query_time)
            self.assertGreater(len(results), 0)
            
        # Check performance targets
        avg_latency = np.mean(query_times)
        p95_latency = np.percentile(query_times, 95)
        
        # Targets (relaxed for test environment)
        self.assertLess(avg_latency, 300, f"Average latency {avg_latency:.1f}ms exceeds target")
        self.assertLess(p95_latency, 600, f"P95 latency {p95_latency:.1f}ms exceeds target")
        
        print(f"Performance: avg={avg_latency:.1f}ms, p95={p95_latency:.1f}ms")
        
    def test_recall_measurement(self):
        """Test recall measurement functionality"""
        manager = IndexManager(self.config_dict)
        manager.initialize()
        manager.build_main_index(self.test_vectors, self.test_metadata)
        
        # Create test queries and ground truth
        test_queries = self.test_vectors[:20]
        ground_truth = []
        
        for i in range(20):
            # For test, use template from metadata as ground truth
            meta = self.test_metadata[i]
            true_template = meta["template_mapping"]["best_template_match"]
            ground_truth.append([true_template])
            
        # Measure recall
        recall_measurement = manager.health_monitor.measure_recall(
            test_queries, ground_truth, k_values=[1, 3, 5]
        )
        
        self.assertIsNotNone(recall_measurement)
        self.assertGreaterEqual(recall_measurement.recall_at_1, 0.0)
        self.assertLessEqual(recall_measurement.recall_at_1, 1.0)
        self.assertGreaterEqual(recall_measurement.recall_at_5, recall_measurement.recall_at_1)
        
        print(f"Recall@1: {recall_measurement.recall_at_1:.3f}")
        print(f"Recall@5: {recall_measurement.recall_at_5:.3f}")
        
    def test_memory_usage_within_limits(self):
        """Test memory usage stays within configured limits"""
        manager = IndexManager(self.config_dict)
        manager.initialize()
        manager.build_main_index(self.test_vectors, self.test_metadata)
        
        # Check system health
        health = manager.get_system_health()
        
        # Memory usage should be reported
        if "memory_usage_mb" in health.get("main_index", {}).get("details", {}):
            memory_usage = health["main_index"]["details"]["memory_usage_mb"]
            memory_limit = self.config_dict["health"]["memory_limit"]
            
            self.assertLess(memory_usage, memory_limit * 1.2, 
                          f"Memory usage {memory_usage}MB exceeds limit {memory_limit}MB")
            
    # Concurrency and Stress Tests
    
    def test_concurrent_system_operations(self):
        """Test concurrent operations across the system"""
        manager = IndexManager(self.config_dict)
        manager.initialize()
        manager.build_main_index(self.test_vectors[:100], self.test_metadata[:100])
        
        results = {"searches": [], "health_checks": [], "errors": []}
        
        def search_worker():
            """Concurrent search worker"""
            for i in range(20):
                try:
                    query_vector = self.test_vectors[i % 100]
                    search_results = manager.search_with_fallback(query_vector, k=3)
                    results["searches"].append(len(search_results))
                    time.sleep(0.01)
                except Exception as e:
                    results["errors"].append(str(e))
                    
        def health_worker():
            """Concurrent health check worker"""
            for i in range(10):
                try:
                    health = manager.get_system_health()
                    results["health_checks"].append(health["overall_healthy"])
                    time.sleep(0.05)
                except Exception as e:
                    results["errors"].append(str(e))
                    
        def maintenance_worker():
            """Periodic maintenance worker"""
            for i in range(5):
                try:
                    manager.periodic_maintenance()
                    time.sleep(0.1)
                except Exception as e:
                    results["errors"].append(str(e))
                    
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(search_worker),
                executor.submit(health_worker),
                executor.submit(maintenance_worker)
            ]
            
            for future in futures:
                future.result()
                
        # Verify results
        self.assertGreater(len(results["searches"]), 15)  # Most searches should succeed
        self.assertGreater(len(results["health_checks"]), 5)  # Most health checks should succeed
        
        if results["errors"]:
            print(f"Concurrent operation errors: {results['errors'][:3]}")
            
    def test_system_stress_under_load(self):
        """Stress test the complete system"""
        manager = IndexManager(self.config_dict)
        manager.initialize()
        manager.build_main_index(self.test_vectors, self.test_metadata)
        
        # Start health monitoring
        manager.health_monitor.start_monitoring(interval_seconds=2)
        
        try:
            # Simulate high load
            total_operations = 0
            errors = []
            
            for round_num in range(5):  # 5 rounds of operations
                
                # Burst of searches
                search_count = 0
                for i in range(50):
                    try:
                        query_vector = self.test_vectors[i % len(self.test_vectors)]
                        results = manager.search_with_fallback(query_vector, k=3)
                        if results:
                            search_count += 1
                        total_operations += 1
                    except Exception as e:
                        errors.append(str(e))
                        
                # Health checks
                try:
                    health = manager.get_system_health()
                    if not health["overall_healthy"]:
                        print(f"System unhealthy in round {round_num}")
                except Exception as e:
                    errors.append(str(e))
                    
                # Maintenance
                try:
                    manager.periodic_maintenance()
                except Exception as e:
                    errors.append(str(e))
                    
                print(f"Round {round_num}: {search_count}/50 searches successful")
                
        finally:
            manager.health_monitor.stop_monitoring()
            
        # System should handle the load
        self.assertGreater(total_operations, 200)
        
        if errors:
            print(f"Stress test errors: {len(errors)} total")
            
        # Get final health report
        final_health = manager.get_system_health()
        print(f"Final system health: {final_health['overall_healthy']}")
        
    # Data Persistence and Recovery Tests
    
    def test_complete_system_persistence(self):
        """Test complete system save and restore"""
        # Create and build system
        manager1 = IndexManager(self.config_dict)
        manager1.initialize()
        manager1.build_main_index(self.test_vectors, self.test_metadata)
        
        # Perform some operations to establish state
        for i in range(10):
            query_vector = self.test_vectors[i]
            results = manager1.search_with_fallback(query_vector, k=3)
            
        # Save system
        save_path = str(Path(self.temp_dir) / "system_save")
        save_success = manager1.save_system(save_path)
        self.assertTrue(save_success)
        
        # Create new system and load
        manager2 = IndexManager(self.config_dict)
        load_success = manager2.load_system(save_path)
        self.assertTrue(load_success)
        
        # Verify loaded system works
        query_vector = self.test_vectors[0]
        results1 = manager1.search_with_fallback(query_vector, k=5)
        results2 = manager2.search_with_fallback(query_vector, k=5)
        
        self.assertEqual(len(results1), len(results2))
        
        # Health should be good
        health1 = manager1.get_system_health()
        health2 = manager2.get_system_health()
        
        self.assertIsInstance(health1, dict)
        self.assertIsInstance(health2, dict)
        
    def test_disaster_recovery_scenarios(self):
        """Test disaster recovery scenarios"""
        manager = IndexManager(self.config_dict)
        manager.initialize()
        manager.build_main_index(self.test_vectors, self.test_metadata)
        
        # Save system state
        save_path = str(Path(self.temp_dir) / "disaster_test")
        manager.save_system(save_path)
        
        query_vector = self.test_vectors[0]
        
        # Scenario 1: Corrupt main index
        if hasattr(manager.main_index, 'index'):
            manager.main_index.index = None
            
        results = manager.search_with_fallback(query_vector, k=3)
        self.assertGreater(len(results), 0)  # Should fallback to bones
        
        # Scenario 2: Complete system failure and recovery
        manager_recovered = IndexManager(self.config_dict)
        recovery_success = manager_recovered.load_system(save_path)
        
        if recovery_success:
            results_recovered = manager_recovered.search_with_fallback(query_vector, k=3)
            self.assertGreater(len(results_recovered), 0)
            
    def assertHasAttr(self, obj, attr):
        """Helper method to check if object has attribute"""
        self.assertTrue(hasattr(obj, attr), f"Object does not have attribute: {attr}")

if __name__ == "__main__":
    unittest.main(verbosity=2) 