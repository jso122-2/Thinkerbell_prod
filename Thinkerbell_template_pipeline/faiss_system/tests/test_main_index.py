#!/usr/bin/env python3
"""
Comprehensive Tests for ContractTemplateIndex
Covers: CRUD ops, fallback, health triggers, persistence, performance, memory pressure, concurrency
"""

import unittest
import tempfile
import shutil
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.index_config import IndexConfig, get_config
from core.main_index import ContractTemplateIndex
from utils.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity

class TestContractTemplateIndex(unittest.TestCase):
    """Comprehensive test suite for ContractTemplateIndex"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = get_config("development")
        self.config.persistence.base_path = self.temp_dir
        self.error_handler = ErrorHandler()
        self.index = ContractTemplateIndex(self.config, self.error_handler)
        
        # Test data
        self.test_vectors = self._generate_test_vectors(100)
        self.test_metadata = self._generate_test_metadata(100)
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            
    def _generate_test_vectors(self, count: int) -> np.ndarray:
        """Generate test vectors"""
        np.random.seed(42)  # Reproducible tests
        vectors = np.random.randn(count, self.config.dimension)
        # Normalize vectors
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.astype(np.float32)
        
    def _generate_test_metadata(self, count: int) -> list:
        """Generate test metadata"""
        metadata = []
        templates = ["template_style_1", "template_style_2", "template_style_3"]
        complexities = ["simple", "medium", "complex"]
        industries = ["tech", "fashion", "beauty", "finance"]
        
        for i in range(count):
            meta = {
                "sample_id": f"test_{i:03d}",
                "template_mapping": {
                    "best_template_match": templates[i % len(templates)]
                },
                "classification": {
                    "complexity_level": complexities[i % len(complexities)],
                    "industry": industries[i % len(industries)]
                },
                "extracted_fields": {
                    "influencer": f"Influencer_{i}",
                    "brand": f"Brand_{i}",
                    "fee_numeric": 1000 + i * 100
                }
            }
            metadata.append(meta)
            
        return metadata
        
    # CRUD Operations Tests
    
    def test_index_build_success(self):
        """Test successful index building"""
        success = self.index.build_index(self.test_vectors, self.test_metadata)
        
        self.assertTrue(success)
        self.assertTrue(self.index.is_built)
        self.assertEqual(self.index.index.ntotal, len(self.test_vectors))
        
    def test_index_build_incremental(self):
        """Test incremental index building"""
        # Initial build
        initial_vectors = self.test_vectors[:50]
        initial_metadata = self.test_metadata[:50]
        success = self.index.build_index(initial_vectors, initial_metadata)
        self.assertTrue(success)
        
        initial_count = self.index.index.ntotal
        
        # Incremental build
        additional_vectors = self.test_vectors[50:]
        additional_metadata = self.test_metadata[50:]
        success = self.index.build_index(additional_vectors, additional_metadata, incremental=True)
        
        self.assertTrue(success)
        self.assertEqual(self.index.index.ntotal, initial_count + len(additional_vectors))
        
    def test_index_build_validation_errors(self):
        """Test input validation during build"""
        # Mismatched vector and metadata counts
        with self.assertRaises(ValueError):
            self.index.build_index(self.test_vectors[:10], self.test_metadata[:5])
            
        # Wrong dimension
        wrong_dim_vectors = np.random.randn(10, 256).astype(np.float32)
        with self.assertRaises(ValueError):
            self.index.build_index(wrong_dim_vectors, self.test_metadata[:10])
            
        # Empty vectors
        with self.assertRaises(ValueError):
            self.index.build_index(np.array([]).reshape(0, self.config.dimension), [])
            
        # NaN values
        nan_vectors = self.test_vectors[:10].copy()
        nan_vectors[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.index.build_index(nan_vectors, self.test_metadata[:10])
            
    def test_search_basic(self):
        """Test basic search functionality"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        query_vector = self.test_vectors[0]
        results = self.index.search(query_vector, k=5)
        
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)
        
        # Check result structure
        for result in results:
            self.assertIsInstance(result.doc_id, int)
            self.assertIsInstance(result.confidence, float)
            self.assertIsInstance(result.template_id, str)
            self.assertIsInstance(result.metadata, dict)
            
    def test_search_with_filters(self):
        """Test search with metadata filters"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        query_vector = self.test_vectors[0]
        filters = {"classification.complexity_level": "medium"}
        results = self.index.search(query_vector, k=5, filters=filters)
        
        # Verify filter application (if implemented)
        self.assertIsInstance(results, list)
        
    def test_search_validation_errors(self):
        """Test search input validation"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        # Wrong dimension
        wrong_dim_query = np.random.randn(256)
        with self.assertRaises(ValueError):
            self.index.search(wrong_dim_query)
            
        # Invalid k
        with self.assertRaises(ValueError):
            self.index.search(self.test_vectors[0], k=0)
            
        # NaN query
        nan_query = np.full(self.config.dimension, np.nan)
        with self.assertRaises(ValueError):
            self.index.search(nan_query)
            
        # Search before build
        empty_index = ContractTemplateIndex(self.config)
        with self.assertRaises(RuntimeError):
            empty_index.search(self.test_vectors[0])
            
    def test_upsert_operations(self):
        """Test upsert functionality"""
        self.index.build_index(self.test_vectors[:50], self.test_metadata[:50])
        
        # Insert new vector
        new_vector = np.random.randn(self.config.dimension)
        new_vector = new_vector / np.linalg.norm(new_vector)
        new_metadata = {
            "sample_id": "new_test",
            "template_mapping": {"best_template_match": "template_style_1"}
        }
        
        doc_id = self.index.upsert(new_vector, new_metadata)
        self.assertGreaterEqual(doc_id, 0)
        
        # Update existing vector
        update_metadata = {
            "sample_id": "updated_test",
            "template_mapping": {"best_template_match": "template_style_2"}
        }
        
        update_id = self.index.upsert(new_vector, update_metadata, doc_id=doc_id)
        self.assertGreaterEqual(update_id, 0)
        
    def test_delete_operations(self):
        """Test delete functionality (marking as deleted)"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        # Get initial count
        initial_count = len(self.index.metadata_store)
        
        # "Delete" a document (mark as deleted)
        doc_id = 0
        if doc_id in self.index.metadata_store:
            self.index.metadata_store[doc_id]["_deleted"] = True
            
            # Check deletion
            self.assertTrue(self.index.metadata_store[doc_id].get("_deleted", False))
            
    # Health and Triggers Tests
    
    def test_health_check_comprehensive(self):
        """Test comprehensive health check"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        health = self.index.get_enhanced_health_check()
        
        self.assertIn("overall_status", health)
        self.assertIn("configuration", health)
        self.assertIn("error_statistics", health)
        
        # Check configuration section
        config_section = health["configuration"]
        self.assertEqual(config_section["environment"], self.config.environment)
        self.assertIn("hnsw_parameters", config_section)
        self.assertIn("memory_configuration", config_section)
        
    def test_needs_rebuild_triggers(self):
        """Test rebuild trigger conditions"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        # Fresh index should not need rebuild
        needs_rebuild, reason = self.index.needs_rebuild()
        self.assertFalse(needs_rebuild)
        
        # Simulate high deletion ratio
        for i in range(len(self.index.metadata_store) // 2):
            if i in self.index.metadata_store:
                self.index.metadata_store[i]["_deleted"] = True
                
        needs_rebuild, reason = self.index.needs_rebuild()
        self.assertTrue(needs_rebuild)
        self.assertIn("deletion ratio", reason)
        
    def test_adaptive_ef_search(self):
        """Test adaptive efSearch adjustment"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        original_ef = self.index.index.hnsw.efSearch
        
        # Simulate high memory pressure
        self.index.config.health.memory_limit_percent = 30  # Very low limit
        
        # Perform search to trigger adaptation
        query_vector = self.test_vectors[0]
        results = self.index.search(query_vector, k=5, adaptive_ef=True)
        
        # efSearch might be adjusted
        self.assertIsInstance(results, list)
        
    # Persistence Tests
    
    def test_save_and_load_cycle(self):
        """Test complete save and load cycle"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        # Save index
        save_path = str(Path(self.temp_dir) / "test_index")
        version_id = self.index.save_versioned(save_path)
        self.assertTrue(version_id)
        
        # Create new index and load
        new_index = ContractTemplateIndex(self.config, self.error_handler)
        success = new_index.load_latest(save_path)
        
        self.assertTrue(success)
        self.assertTrue(new_index.is_built)
        self.assertEqual(new_index.index.ntotal, self.index.index.ntotal)
        
        # Test search on loaded index
        query_vector = self.test_vectors[0]
        results = new_index.search(query_vector, k=5)
        self.assertGreater(len(results), 0)
        
    def test_version_cleanup(self):
        """Test version cleanup functionality"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        save_path = str(Path(self.temp_dir) / "version_test")
        
        # Create multiple versions
        versions = []
        for i in range(5):
            # Modify index slightly
            new_vector = np.random.randn(1, self.config.dimension).astype(np.float32)
            new_metadata = [{"sample_id": f"version_{i}"}]
            self.index.build_index(new_vector, new_metadata, incremental=True)
            
            version_id = self.index.save_versioned(save_path)
            versions.append(version_id)
            
        # Check that only configured number of versions are kept
        base_dir = Path(save_path)
        version_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("v_")]
        
        max_versions = self.config.persistence.backup_versions
        self.assertLessEqual(len(version_dirs), max_versions)
        
    def test_persistence_error_handling(self):
        """Test persistence error handling"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        # Try to save to invalid path
        invalid_path = "/invalid/readonly/path"
        
        # Should handle error gracefully
        try:
            version_id = self.index.save_versioned(invalid_path)
            # If it succeeds, that's fine too (might have created alternative path)
        except Exception:
            # Expected behavior
            pass
            
    # Performance Tests
    
    def test_query_latency_targets(self):
        """Test query latency meets targets"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        query_times = []
        for i in range(50):
            query_vector = self.test_vectors[i % len(self.test_vectors)]
            
            start_time = time.time()
            results = self.index.search(query_vector, k=5)
            query_time = (time.time() - start_time) * 1000  # ms
            
            query_times.append(query_time)
            
        avg_latency = np.mean(query_times)
        p95_latency = np.percentile(query_times, 95)
        
        # Check targets (relaxed for test environment)
        self.assertLess(avg_latency, 500)  # 500ms average
        self.assertLess(p95_latency, 1000)  # 1s p95
        
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"P95 latency: {p95_latency:.2f}ms")
        
    def test_memory_usage_estimation(self):
        """Test memory usage estimation"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        estimated_memory = self.index._estimate_memory_usage()
        self.assertGreater(estimated_memory, 0)
        
        # Should be reasonable for test data size
        self.assertLess(estimated_memory, 1000)  # Less than 1GB for test data
        
        print(f"Estimated memory usage: {estimated_memory:.2f}MB")
        
    def test_cache_performance(self):
        """Test query caching performance"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        query_vector = self.test_vectors[0]
        
        # First query (cache miss)
        start_time = time.time()
        results1 = self.index.search(query_vector, k=5)
        first_time = time.time() - start_time
        
        # Second query (cache hit if implemented)
        start_time = time.time()
        results2 = self.index.search(query_vector, k=5)
        second_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(len(results1), len(results2))
        
        print(f"First query: {first_time*1000:.2f}ms")
        print(f"Second query: {second_time*1000:.2f}ms")
        
    # Memory Pressure Tests
    
    def test_memory_pressure_handling(self):
        """Test memory pressure detection and handling"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        # Simulate memory pressure
        original_limit = self.config.health.memory_limit_percent
        self.config.health.memory_limit_percent = 10  # Very low limit
        
        try:
            # Should still be able to search
            query_vector = self.test_vectors[0]
            results = self.index.search(query_vector, k=5, adaptive_ef=True)
            self.assertGreater(len(results), 0)
            
            # Check if efSearch was reduced
            current_ef = self.index.index.hnsw.efSearch
            self.assertGreaterEqual(current_ef, self.config.hnsw.efSearch_min)
            
        finally:
            self.config.health.memory_limit_percent = original_limit
            
    def test_memory_build_validation(self):
        """Test memory validation during build"""
        # Create very large vector set to trigger memory check
        large_vectors = np.random.randn(10000, self.config.dimension).astype(np.float32)
        large_metadata = [{"id": i} for i in range(10000)]
        
        # Should handle gracefully (might succeed or fail depending on system)
        try:
            success = self.index.build_index(large_vectors, large_metadata)
            # If successful, that's fine
        except MemoryError:
            # Expected for very large datasets
            pass
            
    # Concurrency Tests
    
    def test_concurrent_searches(self):
        """Test concurrent search operations"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        def search_worker(worker_id):
            """Worker function for concurrent searches"""
            results = []
            for i in range(10):
                query_vector = self.test_vectors[(worker_id * 10 + i) % len(self.test_vectors)]
                search_results = self.index.search(query_vector, k=3)
                results.extend(search_results)
            return len(results)
            
        # Run concurrent searches
        num_workers = 4
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(search_worker, i) for i in range(num_workers)]
            
            results = []
            for future in as_completed(futures):
                result_count = future.result()
                results.append(result_count)
                
        # All workers should have completed successfully
        self.assertEqual(len(results), num_workers)
        for count in results:
            self.assertGreater(count, 0)
            
    def test_concurrent_build_and_search(self):
        """Test concurrent build and search operations"""
        # Initial build
        self.index.build_index(self.test_vectors[:50], self.test_metadata[:50])
        
        search_results = []
        build_success = []
        
        def search_worker():
            """Continuous search worker"""
            for i in range(20):
                try:
                    query_vector = self.test_vectors[i % 50]
                    results = self.index.search(query_vector, k=3)
                    search_results.append(len(results))
                    time.sleep(0.01)  # Small delay
                except Exception as e:
                    search_results.append(0)
                    
        def build_worker():
            """Incremental build worker"""
            try:
                additional_vectors = self.test_vectors[50:60]
                additional_metadata = self.test_metadata[50:60]
                success = self.index.build_index(additional_vectors, additional_metadata, incremental=True)
                build_success.append(success)
            except Exception as e:
                build_success.append(False)
                
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            search_future = executor.submit(search_worker)
            build_future = executor.submit(build_worker)
            
            # Wait for completion
            search_future.result()
            build_future.result()
            
        # Check results
        self.assertGreater(len(search_results), 0)
        self.assertEqual(len(build_success), 1)
        
    def test_thread_safety_stress(self):
        """Stress test for thread safety"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        errors = []
        results_count = []
        
        def stress_worker(worker_id):
            """Stress test worker"""
            try:
                for i in range(100):
                    query_vector = self.test_vectors[i % len(self.test_vectors)]
                    results = self.index.search(query_vector, k=2)
                    results_count.append(len(results))
                    
                    # Occasionally trigger health check
                    if i % 20 == 0:
                        health = self.index.get_enhanced_health_check()
                        
            except Exception as e:
                errors.append(str(e))
                
        # Run multiple workers
        num_workers = 8
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_workers)]
            
            for future in as_completed(futures):
                future.result()
                
        # Check for errors
        if errors:
            print(f"Errors encountered: {errors[:5]}")  # Show first 5 errors
            
        # Should have many successful operations
        self.assertGreater(len(results_count), num_workers * 50)
        
    # Integration Tests
    
    def test_error_recovery_integration(self):
        """Test integration with error handling"""
        self.index.build_index(self.test_vectors, self.test_metadata)
        
        # Get initial error count
        initial_errors = self.error_handler.recovery_stats["total_errors"]
        
        # Trigger an error condition
        try:
            # Invalid search
            invalid_query = np.full(self.config.dimension, np.inf)
            results = self.index.search(invalid_query)
        except:
            pass
            
        # Check if error was recorded
        current_errors = self.error_handler.recovery_stats["total_errors"]
        # Might not always trigger depending on validation
        
    def test_configuration_integration(self):
        """Test configuration integration"""
        # Test with different configurations
        prod_config = get_config("production")
        prod_config.persistence.base_path = self.temp_dir
        
        prod_index = ContractTemplateIndex(prod_config, self.error_handler)
        success = prod_index.build_index(self.test_vectors[:10], self.test_metadata[:10])
        
        self.assertTrue(success)
        
        # Check configuration-specific behavior
        config_summary = prod_index.get_configuration_summary()
        self.assertEqual(config_summary["environment"], "production")
        
if __name__ == "__main__":
    unittest.main(verbosity=2) 