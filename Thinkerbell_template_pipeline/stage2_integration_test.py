#!/usr/bin/env python3
"""
Stage 2 Integration Test - Comprehensive testing of all FAISS classes
Tests ContractTemplateIndex, BonesIndex, IndexManager, and HealthMonitor together.
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add thinkerbell modules to path
sys.path.append(str(Path(__file__).parent / "thinkerbell"))

from thinkerbell.core.contract_template_index import ContractTemplateIndex, SearchResult
from thinkerbell.core.bones_index import BonesIndex, BonesResult
from thinkerbell.core.index_manager import IndexManager, SearchStrategy
from thinkerbell.core.health_monitor import HealthMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Stage2IntegrationTest:
    """Comprehensive integration test for Stage 2 classes"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.test_data_dir = self.base_dir / "complete_pipeline_5000"
        self.test_output_dir = self.base_dir / "stage2_test_output"
        
        # Create output directory
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.config = {
            "dimension": 384,
            "main_index": {
                "hnsw": {
                    "M": 32,
                    "efConstruction": 200,
                    "efSearch": 64
                },
                "performance": {
                    "target_latency_ms": 200,
                    "memory_limit_mb": 800,
                    "rebuild_threshold": 0.5
                }
            },
            "health": {
                "latency_threshold": 200,
                "memory_limit": 800,
                "recall_threshold": 0.95
            }
        }
        
        # Component instances
        self.contract_index = None
        self.bones_index = None
        self.index_manager = None
        self.health_monitor = None
        
        # Test data
        self.test_vectors = None
        self.test_metadata = None
        self.test_queries = None
        
    def run_complete_test(self):
        """Run complete integration test suite"""
        logger.info("=" * 70)
        logger.info("STAGE 2 FAISS INTEGRATION TEST")
        logger.info("=" * 70)
        
        try:
            # Test 1: Individual component tests
            self.test_contract_template_index()
            self.test_bones_index()
            self.test_health_monitor()
            
            # Test 2: Integrated system test
            self.test_index_manager()
            
            # Test 3: Performance and stress tests
            self.test_performance_characteristics()
            
            # Test 4: Failure scenarios
            self.test_failure_scenarios()
            
            # Test 5: Production simulation
            self.test_production_simulation()
            
            logger.info("=" * 70)
            logger.info("✓ ALL STAGE 2 TESTS PASSED")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            raise
            
    def test_contract_template_index(self):
        """Test ContractTemplateIndex class"""
        logger.info("Test 1: ContractTemplateIndex")
        
        # Initialize index
        self.contract_index = ContractTemplateIndex(
            dim=self.config["dimension"],
            config=self.config["main_index"]
        )
        
        # Load test data
        test_vectors, test_metadata = self._load_test_data(sample_size=1000)
        
        # Test build_index
        logger.info("  Testing index building...")
        start_time = time.time()
        success = self.contract_index.build_index(test_vectors, test_metadata)
        build_time = time.time() - start_time
        
        assert success, "Index build failed"
        assert self.contract_index.is_built, "Index not marked as built"
        logger.info(f"  ✓ Index built successfully in {build_time:.2f}s")
        
        # Test search
        logger.info("  Testing search functionality...")
        query_vector = test_vectors[0]  # Use first vector as query
        results = self.contract_index.search(query_vector, k=5)
        
        assert len(results) > 0, "No search results returned"
        assert all(isinstance(r, SearchResult) for r in results), "Invalid result types"
        assert all(r.confidence >= 0 for r in results), "Invalid confidence scores"
        logger.info(f"  ✓ Search returned {len(results)} results")
        
        # Test upsert
        logger.info("  Testing upsert functionality...")
        new_vector = np.random.randn(self.config["dimension"])
        new_metadata = {"test": "upsert", "template_mapping": {"best_template_match": "test_template"}}
        
        doc_id = self.contract_index.upsert(new_vector, new_metadata)
        assert doc_id >= 0, "Upsert failed"
        logger.info(f"  ✓ Upsert successful, doc_id: {doc_id}")
        
        # Test health_check
        logger.info("  Testing health check...")
        health = self.contract_index.health_check()
        assert "overall_status" in health, "Health check missing status"
        logger.info(f"  ✓ Health check status: {health['overall_status']}")
        
        # Test needs_rebuild
        needs_rebuild, reason = self.contract_index.needs_rebuild()
        logger.info(f"  ✓ Rebuild check: {needs_rebuild} ({reason})")
        
        # Test save/load
        logger.info("  Testing save/load...")
        version_id = self.contract_index.save_versioned(str(self.test_output_dir / "contract_index"))
        assert version_id, "Save failed"
        
        # Create new index and load
        new_index = ContractTemplateIndex(dim=self.config["dimension"])
        load_success = new_index.load_latest(str(self.test_output_dir / "contract_index"))
        assert load_success, "Load failed"
        assert new_index.is_built, "Loaded index not built"
        logger.info(f"  ✓ Save/load successful, version: {version_id}")
        
        logger.info("✓ ContractTemplateIndex tests passed\n")
        
    def test_bones_index(self):
        """Test BonesIndex class"""
        logger.info("Test 2: BonesIndex")
        
        # Initialize bones index
        self.bones_index = BonesIndex(dim=self.config["dimension"])
        
        # Test build_from_templates
        logger.info("  Testing bones index building...")
        success = self.bones_index.build_from_templates()
        assert success, "Bones index build failed"
        assert self.bones_index.is_built, "Bones index not marked as built"
        logger.info(f"  ✓ Bones index built with {len(self.bones_index.template_signatures)} templates")
        
        # Test fallback_search
        logger.info("  Testing fallback search...")
        query_vector = np.random.randn(self.config["dimension"])
        template_id = self.bones_index.fallback_search(query_vector)
        assert template_id, "Fallback search returned empty"
        logger.info(f"  ✓ Fallback search returned: {template_id}")
        
        # Test search_with_metadata
        logger.info("  Testing search with metadata...")
        bones_results = self.bones_index.search_with_metadata(query_vector, k=3)
        assert len(bones_results) > 0, "No bones results returned"
        assert all(isinstance(r, BonesResult) for r in bones_results), "Invalid bones result types"
        logger.info(f"  ✓ Bones search returned {len(bones_results)} results")
        
        # Test get_template_by_complexity
        for complexity in ["simple", "medium", "complex"]:
            template_id = self.bones_index.get_template_by_complexity(complexity)
            assert template_id, f"No template for complexity: {complexity}"
            logger.info(f"  ✓ {complexity} template: {template_id}")
            
        # Test save/load
        logger.info("  Testing bones save/load...")
        save_path = str(self.test_output_dir / "bones_index.json")
        save_success = self.bones_index.save_bones(save_path)
        assert save_success, "Bones save failed"
        
        new_bones = BonesIndex(dim=self.config["dimension"])
        load_success = new_bones.load_bones(save_path)
        assert load_success, "Bones load failed"
        assert new_bones.is_built, "Loaded bones index not built"
        logger.info("  ✓ Bones save/load successful")
        
        # Test health
        assert self.bones_index.is_healthy(), "Bones index not healthy"
        stats = self.bones_index.get_stats()
        logger.info(f"  ✓ Bones stats: {stats['total_templates']} templates, {stats['memory_estimate_kb']:.1f}KB")
        
        logger.info("✓ BonesIndex tests passed\n")
        
    def test_health_monitor(self):
        """Test HealthMonitor class"""
        logger.info("Test 3: HealthMonitor")
        
        # Initialize health monitor
        self.health_monitor = HealthMonitor(self.config["health"])
        
        # Initialize with indices
        if not self.contract_index:
            self.contract_index = ContractTemplateIndex(dim=self.config["dimension"])
        if not self.bones_index:
            self.bones_index = BonesIndex(dim=self.config["dimension"])
            self.bones_index.build_from_templates()
            
        self.health_monitor.initialize(self.contract_index, self.bones_index)
        
        # Test memory pressure check
        memory_pressure = self.health_monitor.check_memory_pressure()
        assert 0 <= memory_pressure <= 1, "Invalid memory pressure value"
        logger.info(f"  ✓ Memory pressure: {memory_pressure:.2f}")
        
        # Test CPU usage
        cpu_usage = self.health_monitor.get_cpu_usage()
        assert cpu_usage >= 0, "Invalid CPU usage"
        logger.info(f"  ✓ CPU usage: {cpu_usage:.1f}%")
        
        # Test search recording
        logger.info("  Testing search recording...")
        for i in range(10):
            query_vector = np.random.randn(self.config["dimension"])
            mock_results = [type('MockResult', (), {
                'confidence': 0.8,
                'source': 'test',
                'fallback_used': i % 3 == 0
            })()]
            self.health_monitor.record_search(query_vector, mock_results, 50 + i * 10)
            
        logger.info("  ✓ Search recording successful")
        
        # Test query pattern analysis
        logger.info("  Testing query pattern analysis...")
        patterns = self.health_monitor.analyze_query_patterns()
        logger.info(f"  ✓ Found {len(patterns)} query patterns")
        
        # Test adaptive efSearch
        logger.info("  Testing adaptive efSearch...")
        optimal_ef = self.health_monitor.adaptive_ef_search(base_ef=64)
        assert 16 <= optimal_ef <= 128, "efSearch out of bounds"
        logger.info(f"  ✓ Adaptive efSearch: {optimal_ef}")
        
        # Test health report
        logger.info("  Testing health report...")
        health_report = self.health_monitor.get_health_report()
        assert "overall_status" in health_report, "Health report missing status"
        assert "current_performance" in health_report, "Health report missing performance"
        logger.info(f"  ✓ Health report status: {health_report['overall_status']}")
        
        logger.info("✓ HealthMonitor tests passed\n")
        
    def test_index_manager(self):
        """Test IndexManager orchestration"""
        logger.info("Test 4: IndexManager Integration")
        
        # Initialize index manager
        self.index_manager = IndexManager(self.config)
        
        # Test initialization
        logger.info("  Testing IndexManager initialization...")
        success = self.index_manager.initialize()
        assert success, "IndexManager initialization failed"
        logger.info("  ✓ IndexManager initialized")
        
        # Build main index with test data
        logger.info("  Building main index...")
        test_vectors, test_metadata = self._load_test_data(sample_size=500)
        build_success = self.index_manager.build_main_index(test_vectors, test_metadata)
        assert build_success, "Main index build failed"
        logger.info("  ✓ Main index built")
        
        # Test different search strategies
        strategies = [
            SearchStrategy.MAIN_ONLY,
            SearchStrategy.BONES_ONLY,
            SearchStrategy.MAIN_WITH_BONES_FALLBACK,
            SearchStrategy.PARALLEL_SEARCH,
            SearchStrategy.ADAPTIVE
        ]
        
        query_vector = test_vectors[0]
        
        for strategy in strategies:
            logger.info(f"  Testing {strategy.value} strategy...")
            start_time = time.time()
            results = self.index_manager.search_with_fallback(
                query_vector, k=5, strategy=strategy
            )
            search_time = (time.time() - start_time) * 1000
            
            assert len(results) > 0, f"No results for {strategy.value}"
            assert all(hasattr(r, 'template_id') for r in results), "Invalid result format"
            logger.info(f"    ✓ {strategy.value}: {len(results)} results in {search_time:.1f}ms")
            
        # Test system health
        logger.info("  Testing system health...")
        health = self.index_manager.get_system_health()
        assert "overall_healthy" in health, "Health missing overall status"
        logger.info(f"  ✓ System health: {'healthy' if health['overall_healthy'] else 'unhealthy'}")
        
        # Test periodic maintenance
        logger.info("  Testing periodic maintenance...")
        maintenance_success = self.index_manager.periodic_maintenance()
        assert maintenance_success, "Periodic maintenance failed"
        logger.info("  ✓ Periodic maintenance successful")
        
        # Test cache burn cycle
        logger.info("  Testing cache burn cycle...")
        burn_success = self.index_manager.burn_cache_cycle()
        assert burn_success, "Cache burn cycle failed"
        logger.info("  ✓ Cache burn cycle successful")
        
        # Test save/load system
        logger.info("  Testing system save/load...")
        save_path = str(self.test_output_dir / "index_manager")
        save_success = self.index_manager.save_system(save_path)
        assert save_success, "System save failed"
        
        # Create new manager and load
        new_manager = IndexManager(self.config)
        load_success = new_manager.load_system(save_path)
        assert load_success, "System load failed"
        logger.info("  ✓ System save/load successful")
        
        logger.info("✓ IndexManager tests passed\n")
        
    def test_performance_characteristics(self):
        """Test performance characteristics and benchmarks"""
        logger.info("Test 5: Performance Characteristics")
        
        if not self.index_manager:
            self._setup_index_manager()
            
        # Latency benchmark
        logger.info("  Running latency benchmark...")
        query_times = []
        test_vectors, _ = self._load_test_data(sample_size=100)
        
        for i in range(50):
            query_vector = test_vectors[i % len(test_vectors)]
            start_time = time.time()
            results = self.index_manager.search_with_fallback(query_vector, k=5)
            query_time = (time.time() - start_time) * 1000
            query_times.append(query_time)
            
        avg_latency = np.mean(query_times)
        p95_latency = np.percentile(query_times, 95)
        p99_latency = np.percentile(query_times, 99)
        
        logger.info(f"  ✓ Latency benchmark:")
        logger.info(f"    Average: {avg_latency:.2f}ms")
        logger.info(f"    P95: {p95_latency:.2f}ms")
        logger.info(f"    P99: {p99_latency:.2f}ms")
        
        # Check performance targets
        target_p95 = self.config["main_index"]["performance"]["target_latency_ms"]
        assert p95_latency <= target_p95 * 2, f"P95 latency too high: {p95_latency:.1f}ms > {target_p95 * 2}ms"
        
        # Memory usage test
        logger.info("  Testing memory usage...")
        health = self.index_manager.get_system_health()
        memory_usage = health["system"]["search_performance"].get("memory_usage", 0)
        memory_limit = self.config["health"]["memory_limit"]
        
        logger.info(f"  ✓ Memory usage: {memory_usage:.1f}MB (limit: {memory_limit}MB)")
        
        # Throughput test
        logger.info("  Testing throughput...")
        concurrent_queries = 10
        start_time = time.time()
        
        for _ in range(concurrent_queries):
            query_vector = np.random.randn(self.config["dimension"])
            self.index_manager.search_with_fallback(query_vector, k=3)
            
        total_time = time.time() - start_time
        qps = concurrent_queries / total_time
        
        logger.info(f"  ✓ Throughput: {qps:.1f} QPS")
        
        logger.info("✓ Performance tests passed\n")
        
    def test_failure_scenarios(self):
        """Test failure scenarios and recovery"""
        logger.info("Test 6: Failure Scenarios")
        
        if not self.index_manager:
            self._setup_index_manager()
            
        # Test main index failure simulation
        logger.info("  Testing main index failure...")
        original_main_index = self.index_manager.main_index
        self.index_manager.main_index.is_built = False  # Simulate failure
        
        query_vector = np.random.randn(self.config["dimension"])
        results = self.index_manager.search_with_fallback(query_vector, k=5)
        
        assert len(results) > 0, "No fallback results"
        assert all(r.source in ['bones', 'emergency'] for r in results), "Fallback not used"
        logger.info(f"  ✓ Main index failure handled, {len(results)} fallback results")
        
        # Restore main index
        self.index_manager.main_index = original_main_index
        
        # Test bones index failure simulation
        logger.info("  Testing bones index failure...")
        original_bones = self.index_manager.bones_index
        broken_bones = BonesIndex(dim=self.config["dimension"])
        # Don't build it - leave it broken
        self.index_manager.bones_index = broken_bones
        
        results = self.index_manager.search_with_fallback(query_vector, k=5)
        assert len(results) > 0, "No emergency results"
        logger.info(f"  ✓ Bones index failure handled, {len(results)} emergency results")
        
        # Restore bones index
        self.index_manager.bones_index = original_bones
        
        # Test memory pressure simulation
        logger.info("  Testing memory pressure response...")
        # Force high memory pressure
        original_check = self.index_manager.health_monitor.check_memory_pressure
        self.index_manager.health_monitor.check_memory_pressure = lambda: 0.95
        
        # Run adaptive optimization
        old_ef = self.index_manager.health_monitor.current_ef_search
        new_ef = self.index_manager.health_monitor.adaptive_ef_search()
        
        assert new_ef <= old_ef, "efSearch not reduced under memory pressure"
        logger.info(f"  ✓ Memory pressure response: efSearch {old_ef} → {new_ef}")
        
        # Restore original function
        self.index_manager.health_monitor.check_memory_pressure = original_check
        
        logger.info("✓ Failure scenario tests passed\n")
        
    def test_production_simulation(self):
        """Simulate production workload"""
        logger.info("Test 7: Production Simulation")
        
        if not self.index_manager:
            self._setup_index_manager()
            
        # Start health monitoring
        self.index_manager.health_monitor.start_monitoring(interval_seconds=5)
        
        # Simulate realistic query patterns
        logger.info("  Simulating production workload...")
        
        query_patterns = [
            {"complexity": "simple", "weight": 0.4},
            {"complexity": "medium", "weight": 0.4},
            {"complexity": "complex", "weight": 0.2}
        ]
        
        total_queries = 100
        query_times = []
        fallback_count = 0
        
        for i in range(total_queries):
            # Select pattern based on weights
            pattern = np.random.choice(
                [p["complexity"] for p in query_patterns],
                p=[p["weight"] for p in query_patterns]
            )
            
            # Generate query vector (simulate different complexities)
            if pattern == "simple":
                query_vector = np.random.randn(self.config["dimension"]) * 0.5
            elif pattern == "medium":
                query_vector = np.random.randn(self.config["dimension"])
            else:  # complex
                query_vector = np.random.randn(self.config["dimension"]) * 1.5
                
            # Execute query
            start_time = time.time()
            results = self.index_manager.search_with_fallback(
                query_vector, 
                k=5,
                strategy=SearchStrategy.ADAPTIVE
            )
            query_time = (time.time() - start_time) * 1000
            query_times.append(query_time)
            
            # Check for fallbacks
            if any(getattr(r, 'fallback_used', False) for r in results):
                fallback_count += 1
                
            # Occasional maintenance (simulate real conditions)
            if i % 50 == 0 and i > 0:
                self.index_manager.periodic_maintenance()
                
        # Stop monitoring
        self.index_manager.health_monitor.stop_monitoring()
        
        # Analyze results
        avg_latency = np.mean(query_times)
        p95_latency = np.percentile(query_times, 95)
        fallback_rate = fallback_count / total_queries
        
        logger.info(f"  ✓ Production simulation results:")
        logger.info(f"    Total queries: {total_queries}")
        logger.info(f"    Average latency: {avg_latency:.2f}ms")
        logger.info(f"    P95 latency: {p95_latency:.2f}ms")
        logger.info(f"    Fallback rate: {fallback_rate:.1%}")
        
        # Validate production requirements
        assert avg_latency < 300, f"Average latency too high: {avg_latency:.1f}ms"
        assert p95_latency < 600, f"P95 latency too high: {p95_latency:.1f}ms"
        assert fallback_rate < 0.5, f"Fallback rate too high: {fallback_rate:.1%}"
        
        # Get final health report
        final_health = self.index_manager.get_system_health()
        logger.info(f"  ✓ Final system health: {'healthy' if final_health['overall_healthy'] else 'needs attention'}")
        
        logger.info("✓ Production simulation passed\n")
        
    # Helper methods
    
    def _load_test_data(self, sample_size: int = 1000) -> tuple:
        """Load test data for index building"""
        if self.test_vectors is not None and len(self.test_vectors) >= sample_size:
            return self.test_vectors[:sample_size], self.test_metadata[:sample_size]
            
        logger.info(f"  Loading test data (sample_size={sample_size})...")
        
        vectors = []
        metadata = []
        
        # Check if we have actual data
        if self.test_data_dir.exists():
            # Load from actual contract data
            sample_files = list((self.test_data_dir / "samples").rglob("*.json"))
            
            for i, sample_file in enumerate(sample_files[:sample_size]):
                if sample_file.name == "batch_metadata.json":
                    continue
                    
                try:
                    with open(sample_file, 'r') as f:
                        sample_data = json.load(f)
                        
                    # Create embedding from text (simplified)
                    text = sample_data.get("raw_input", {}).get("text", "")
                    if text:
                        # Simple hash-based vector (in production, use real embeddings)
                        vector = self._text_to_vector(text)
                        vectors.append(vector)
                        metadata.append(sample_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to load {sample_file}: {e}")
                    
                if len(vectors) >= sample_size:
                    break
                    
        # Fill remaining with synthetic data if needed
        while len(vectors) < sample_size:
            vector = np.random.randn(self.config["dimension"])
            vector = vector / np.linalg.norm(vector)  # Normalize
            
            meta = {
                "sample_id": f"synthetic_{len(vectors)}",
                "template_mapping": {
                    "best_template_match": f"template_style_{np.random.randint(1, 6)}"
                },
                "classification": {
                    "complexity_level": np.random.choice(["simple", "medium", "complex"]),
                    "industry": np.random.choice(["tech", "fashion", "beauty", "finance"])
                }
            }
            
            vectors.append(vector)
            metadata.append(meta)
            
        self.test_vectors = np.array(vectors)
        self.test_metadata = metadata
        
        logger.info(f"  ✓ Loaded {len(vectors)} test samples")
        return self.test_vectors[:sample_size], self.test_metadata[:sample_size]
        
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector (simplified hash-based approach)"""
        # Simple hash-based vector generation for testing
        hash_value = hash(text)
        np.random.seed(abs(hash_value) % (2**32))
        vector = np.random.randn(self.config["dimension"])
        vector = vector / np.linalg.norm(vector)
        return vector
        
    def _setup_index_manager(self):
        """Setup index manager for testing"""
        self.index_manager = IndexManager(self.config)
        self.index_manager.initialize()
        
        # Build with test data
        test_vectors, test_metadata = self._load_test_data(sample_size=500)
        self.index_manager.build_main_index(test_vectors, test_metadata)

def main():
    """Main test entry point"""
    test = Stage2IntegrationTest()
    test.run_complete_test()

if __name__ == "__main__":
    main() 