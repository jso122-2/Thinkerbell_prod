#!/usr/bin/env python3
"""
FAISS Template Index Integration Demo
Demonstrates integration with existing Thinkerbell pipeline and usage patterns.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add thinkerbell modules to path
sys.path.append(str(Path(__file__).parent / "thinkerbell"))

from thinkerbell.core.faiss_template_index import FAISSTemplateIndex
from thinkerbell.core.faiss_template_manager import FAISSTemplateManager
from thinkerbell.core.faiss_operations import FAISSHealthMonitor, FAISSPersistenceManager, AdaptiveSearchOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FAISSIntegrationDemo:
    """Demonstration of FAISS integration with Thinkerbell pipeline"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.faiss_dir = self.base_dir / "faiss_indices"
        self.data_dir = self.base_dir / "complete_pipeline_5000"
        
        # Initialize components
        self.template_manager = None
        self.health_monitor = None
        self.persistence_manager = None
        self.search_optimizer = None
        
    def run_complete_demo(self):
        """Run complete integration demonstration"""
        logger.info("=" * 60)
        logger.info("FAISS Template Index Integration Demo")
        logger.info("=" * 60)
        
        try:
            # Step 1: Initialize system
            self.initialize_system()
            
            # Step 2: Load contract dataset
            self.load_contract_data()
            
            # Step 3: Demonstrate template matching
            self.demonstrate_template_matching()
            
            # Step 4: Show operational features
            self.demonstrate_operational_features()
            
            # Step 5: Performance testing
            self.run_performance_tests()
            
            # Step 6: Integration examples
            self.show_integration_examples()
            
            logger.info("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
            
        finally:
            self.cleanup()
            
    def initialize_system(self):
        """Initialize FAISS system components"""
        logger.info("Step 1: Initializing FAISS Template System")
        
        # Create template manager
        self.template_manager = FAISSTemplateManager(
            index_dir=str(self.faiss_dir),
            model_path="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        if not self.template_manager.initialize():
            raise RuntimeError("Failed to initialize template manager")
            
        # Initialize operational components
        self.health_monitor = FAISSHealthMonitor(self.template_manager)
        self.persistence_manager = FAISSPersistenceManager(str(self.faiss_dir / "versions"))
        self.search_optimizer = AdaptiveSearchOptimizer(self.template_manager.faiss_index)
        
        logger.info("✓ FAISS system initialized")
        
    def load_contract_data(self):
        """Load contract dataset into FAISS index"""
        logger.info("Step 2: Loading Contract Dataset")
        
        if not self.data_dir.exists():
            logger.warning(f"Dataset directory not found: {self.data_dir}")
            logger.info("Creating minimal test dataset...")
            self._create_test_dataset()
        else:
            logger.info(f"Loading from: {self.data_dir}")
            
        success = self.template_manager.load_contract_dataset(str(self.data_dir))
        
        if success:
            # Save initial version
            version_id = self.persistence_manager.save_version(
                str(self.faiss_dir), 
                "initial_load"
            )
            logger.info(f"✓ Dataset loaded and saved as version: {version_id}")
        else:
            logger.error("Failed to load dataset")
            
    def _create_test_dataset(self):
        """Create minimal test dataset for demo"""
        test_samples = [
            {
                "sample_id": "test_001",
                "classification": {
                    "document_type": "INFLUENCER_AGREEMENT",
                    "complexity_level": "simple",
                    "industry": "fashion"
                },
                "raw_input": {
                    "text": "Hi! We'd love to work with Sarah Chen on a Myer campaign. Budget is around $5,000 and we need 2 Instagram posts. Looking at 3 months engagement with 4 weeks exclusivity.",
                    "text_style": "casual"
                },
                "extracted_fields": {
                    "influencer": "Sarah Chen",
                    "brand": "Myer",
                    "fee": "$5,000",
                    "fee_numeric": 5000,
                    "deliverables": ["2 x Instagram posts"],
                    "engagement_term": "3 months",
                    "exclusivity_period": "4 weeks"
                },
                "template_mapping": {
                    "best_template_match": "template_style_1",
                    "match_confidence": 0.8
                }
            },
            {
                "sample_id": "test_002", 
                "classification": {
                    "document_type": "INFLUENCER_AGREEMENT",
                    "complexity_level": "complex",
                    "industry": "tech"
                },
                "raw_input": {
                    "text": "Comprehensive Influencer Agreement: Microsoft and Alex Johnson partnership. Total compensation: $25,000. Detailed content requirements: 5 blog posts, 10 social media posts, 1 webinar. Full engagement term: 6 months. Exclusivity provisions: 12 weeks within category. Usage rights: 24 months. Territory: Australia.",
                    "text_style": "formal"
                },
                "extracted_fields": {
                    "influencer": "Alex Johnson",
                    "brand": "Microsoft", 
                    "fee": "$25,000",
                    "fee_numeric": 25000,
                    "deliverables": ["5 x blog posts", "10 x social media posts", "1 x webinar"],
                    "engagement_term": "6 months",
                    "exclusivity_period": "12 weeks",
                    "usage_term": "24 months"
                },
                "template_mapping": {
                    "best_template_match": "template_style_5",
                    "match_confidence": 0.9
                }
            }
        ]
        
        # Create test directory structure
        test_dir = self.data_dir / "samples" / "batch_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        for sample in test_samples:
            sample_file = test_dir / f"{sample['sample_id']}.json"
            with open(sample_file, 'w') as f:
                json.dump(sample, f, indent=2)
                
        logger.info(f"Created test dataset with {len(test_samples)} samples")
        
    def demonstrate_template_matching(self):
        """Demonstrate template matching capabilities"""
        logger.info("Step 3: Demonstrating Template Matching")
        
        test_queries = [
            {
                "text": "Need Quinn Richardson for KFC campaign. Budget around $6,751, looking for 1 x 15-second TikTok.",
                "expected_complexity": "medium"
            },
            {
                "text": "Hi! Quick collab with Emma for Cotton On. $3k budget, need 2 Instagram posts.",
                "expected_complexity": "simple"
            },
            {
                "text": "Strategic Partnership Agreement: Telstra enterprise collaboration with David Wilson. Investment structure: $50,000 total package. Comprehensive deliverable suite: 10 blog posts, 20 social posts, 3 webinars. Partnership duration: 12 months.",
                "expected_complexity": "complex"
            }
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\nQuery {i}: {query['text'][:50]}...")
            
            start_time = time.time()
            result = self.template_manager.find_best_template(
                contract_text=query['text'],
                return_alternatives=True
            )
            query_time = (time.time() - start_time) * 1000
            
            logger.info(f"  → Template: {result['best_template_match']}")
            logger.info(f"  → Confidence: {result['match_confidence']:.3f}")
            logger.info(f"  → Source: {result['source_index']}")
            logger.info(f"  → Latency: {query_time:.1f}ms")
            
            if 'fallback_templates' in result:
                logger.info(f"  → Alternatives: {', '.join(result['fallback_templates'])}")
                
    def demonstrate_operational_features(self):
        """Demonstrate operational features"""
        logger.info("Step 4: Demonstrating Operational Features")
        
        # Start health monitoring
        logger.info("Starting health monitoring...")
        self.health_monitor.start_monitoring(interval_seconds=5)
        
        # Wait a bit for monitoring data
        time.sleep(10)
        
        # Get health report
        health_report = self.health_monitor.get_health_report()
        logger.info(f"Overall health status: {health_report['overall_status']}")
        
        # Show performance metrics
        metrics = self.template_manager.get_performance_metrics()
        logger.info(f"Index health: {metrics['index_health']['healthy']}")
        logger.info(f"Query statistics: {metrics['query_statistics']['total_queries']} total queries")
        
        # Demonstrate adaptive optimization
        logger.info("Running adaptive search optimization...")
        self.search_optimizer.optimize_search_parameters()
        
        opt_report = self.search_optimizer.get_optimization_report()
        if 'current_ef_search' in opt_report:
            logger.info(f"Current efSearch: {opt_report['current_ef_search']}")
            
        # Stop monitoring
        self.health_monitor.stop_monitoring()
        
    def run_performance_tests(self):
        """Run performance tests"""
        logger.info("Step 5: Running Performance Tests")
        
        # Test latency with various query sizes
        test_queries = [
            "Short query",
            "Medium length query with more details about influencer campaign",
            "Very long and detailed query that includes comprehensive information about the influencer partnership agreement including all deliverables, compensation details, exclusivity terms, usage rights, and other contractual provisions that would typically be found in a complex business agreement"
        ]
        
        latencies = []
        
        for query in test_queries:
            times = []
            for _ in range(10):  # 10 iterations per query
                start_time = time.time()
                self.template_manager.find_best_template(query)
                query_time = (time.time() - start_time) * 1000
                times.append(query_time)
                
            avg_latency = sum(times) / len(times)
            p95_latency = sorted(times)[int(0.95 * len(times))]
            latencies.append((len(query), avg_latency, p95_latency))
            
            logger.info(f"Query length {len(query):3d}: avg={avg_latency:5.1f}ms, p95={p95_latency:5.1f}ms")
            
        # Check if meeting performance targets
        overall_p95 = max(lat[2] for lat in latencies)
        target_met = overall_p95 <= 200  # Target: p95 ≤ 200ms
        
        logger.info(f"Performance target (p95 ≤ 200ms): {'✓ MET' if target_met else '✗ FAILED'}")
        logger.info(f"Actual p95: {overall_p95:.1f}ms")
        
    def show_integration_examples(self):
        """Show integration examples with existing pipeline"""
        logger.info("Step 6: Integration Examples")
        
        # Example 1: Direct integration with template manager
        logger.info("\nExample 1: Direct Template Manager Integration")
        
        extracted_fields = {
            "influencer": "Jamie Wilson",
            "brand": "Qantas",
            "fee_numeric": 15000,
            "deliverables": ["3 x Instagram posts", "1 x YouTube video"],
            "engagement_term": "4 months",
            "exclusivity_period": "8 weeks",
            "metadata": {"industry": "travel"}
        }
        
        recommendation = self.template_manager.get_template_recommendations(extracted_fields)
        logger.info(f"  Template recommendation: {recommendation['best_template_match']}")
        logger.info(f"  Analysis: {recommendation.get('analysis', {})}")
        
        # Example 2: Fallback behavior
        logger.info("\nExample 2: Fallback Behavior Testing")
        
        # Simulate index failure
        original_main_index = self.template_manager.faiss_index.main_index
        self.template_manager.faiss_index.main_index = None
        
        result = self.template_manager.find_best_template("Emergency test query")
        logger.info(f"  Fallback result: {result['best_template_match']} (source: {result['source_index']})")
        
        # Restore index
        self.template_manager.faiss_index.main_index = original_main_index
        
        # Example 3: Version management
        logger.info("\nExample 3: Version Management")
        
        versions = self.persistence_manager.list_versions()
        logger.info(f"  Available versions: {len(versions)}")
        for version in versions[:3]:  # Show first 3
            logger.info(f"    {version['version_id']} ({version.get('tag', 'no tag')})")
            
    def cleanup(self):
        """Cleanup resources"""
        if self.health_monitor:
            self.health_monitor.stop_monitoring()

def main():
    """Main demo entry point"""
    demo = FAISSIntegrationDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main() 