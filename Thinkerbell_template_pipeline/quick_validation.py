#!/usr/bin/env python3
"""
Quick Validation Script for Thinkerbell Synthetic Dataset Generation
Tests all components before full generation to catch issues early
"""

import json
import random
from thinkerbell.data.synthetic_dataset_generator import SyntheticDatasetGenerator
from thinkerbell.data.dataset_validation import DatasetValidator

def test_small_batch_generation():
    """Test generation of a small batch"""
    
    print("🧪 Testing Small Batch Generation")
    print("=" * 50)
    
    # Test with minimal features first
    generator = SyntheticDatasetGenerator(
        use_semantic_smoothing=False,  # Start simple
        use_text_preprocessing=False,
        use_ood_contamination=False
    )
    
    print("🔄 Generating 10 basic samples...")
    basic_dataset = generator.generate_dataset(num_samples=10)
    
    print(f"✅ Generated {len(basic_dataset)} basic samples")
    
    # Test with all features
    generator_full = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=True,
        contamination_ratio=0.2
    )
    
    print("🔄 Generating 20 samples with all features...")
    full_dataset = generator_full.generate_dataset(num_samples=20)
    
    print(f"✅ Generated {len(full_dataset)} samples with all features")
    
    return basic_dataset, full_dataset

def test_validation_components():
    """Test validation components"""
    
    print("\n🧪 Testing Validation Components")
    print("=" * 50)
    
    # Generate test dataset
    generator = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=True,
        contamination_ratio=0.2
    )
    
    test_dataset = generator.generate_dataset(num_samples=50)
    
    # Test validator
    validator = DatasetValidator()
    
    print("📊 Testing distribution balance...")
    distribution_scores = validator.validate_distribution_balance(test_dataset)
    for metric, score in distribution_scores.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n👤 Testing human extractability...")
    extraction_accuracy, extraction_issues = validator.validate_human_extractability(test_dataset, sample_size=10)
    print(f"  Accuracy: {extraction_accuracy:.3f}")
    print(f"  Issues: {len(extraction_issues)}")
    
    print("\n🧠 Testing semantic coherence...")
    coherence_score, coherence_issues = validator.validate_semantic_coherence(test_dataset, sample_size=10)
    print(f"  Coherence: {coherence_score:.3f}")
    print(f"  Issues: {len(coherence_issues)}")
    
    print("\n📏 Testing token lengths...")
    length_score, length_issues = validator.validate_token_lengths(test_dataset)
    print(f"  Length compliance: {length_score:.3f}")
    print(f"  Issues: {len(length_issues)}")
    
    return test_dataset

def test_benchmark_generation():
    """Test benchmark test set generation"""
    
    print("\n🧪 Testing Benchmark Generation")
    print("=" * 50)
    
    # Generate small dataset
    generator = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=True,
        contamination_ratio=0.2
    )
    
    dataset = generator.generate_dataset(num_samples=100)
    
    # Test benchmark generation
    validator = DatasetValidator()
    train_set, test_set = validator.generate_benchmark_test_set(dataset, test_ratio=0.2)
    
    print(f"✅ Generated benchmark split:")
    print(f"  Train set: {len(train_set)} samples")
    print(f"  Test set: {len(test_set)} samples")
    
    # Check that test set has messy samples
    messy_count = sum(1 for item in test_set if item.get("metadata", {}).get("test_set_messy", False))
    print(f"  Messy samples in test set: {messy_count}")
    
    return train_set, test_set

def test_metadata_generation():
    """Test metadata generation"""
    
    print("\n🧪 Testing Metadata Generation")
    print("=" * 50)
    
    # Generate small dataset
    generator = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=True,
        contamination_ratio=0.2
    )
    
    dataset = generator.generate_dataset(num_samples=50)
    
    # Test metadata generation
    validator = DatasetValidator()
    generator_config = {
        "use_semantic_smoothing": True,
        "use_text_preprocessing": True,
        "use_ood_contamination": True,
        "contamination_ratio": 0.2
    }
    
    metadata = validator.create_dataset_metadata(dataset, generator_config)
    
    print("✅ Generated metadata:")
    print(f"  Version: {metadata.get('version', 'N/A')}")
    print(f"  Total samples: {metadata.get('total_samples', 0)}")
    print(f"  Generation date: {metadata.get('generation_date', 'N/A')}")
    
    # Check statistics
    stats = metadata.get('statistics', {})
    if stats:
        print(f"  Classification distribution: {stats.get('classification_distribution', {})}")
        print(f"  Complexity distribution: {stats.get('complexity_distribution', {})}")
    
    return metadata

def run_comprehensive_test():
    """Run comprehensive test of all components"""
    
    print("🧪 Comprehensive Pipeline Test")
    print("=" * 60)
    
    try:
        # Test 1: Small batch generation
        print("\n📊 Test 1: Small batch generation...")
        basic_dataset, full_dataset = test_small_batch_generation()
        
        # Test 2: Validation components
        print("\n📊 Test 2: Validation components...")
        test_dataset = test_validation_components()
        
        # Test 3: Benchmark generation
        print("\n📊 Test 3: Benchmark generation...")
        train_set, test_set = test_benchmark_generation()
        
        # Test 4: Metadata generation
        print("\n📊 Test 4: Metadata generation...")
        metadata = test_metadata_generation()
        
        # Test 5: Comprehensive validation
        print("\n📊 Test 5: Comprehensive validation...")
        validator = DatasetValidator()
        generator_config = {
            "use_semantic_smoothing": True,
            "use_text_preprocessing": True,
            "use_ood_contamination": True,
            "contamination_ratio": 0.2
        }
        
        result = validator.comprehensive_validation(full_dataset, generator_config)
        
        print(f"\n📊 Comprehensive validation result:")
        print(f"  Passed: {result.passed}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Issues: {len(result.issues)}")
        
        if result.passed:
            print("\n✅ All tests passed! Pipeline is ready for full generation.")
            print("🚀 Run: python generate_training_dataset.py")
        else:
            print("\n❌ Some tests failed. Please review issues before full generation.")
            for issue in result.issues[:5]:
                print(f"  - {issue}")
        
        return result.passed
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick validation"""
    
    print("🔍 Quick Validation for Thinkerbell Dataset Generation")
    print("=" * 60)
    
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 Quick validation completed successfully!")
        print("📋 All components are working correctly.")
        print("🚀 Ready for full dataset generation.")
    else:
        print("\n❌ Quick validation failed!")
        print("🔧 Please fix issues before proceeding with full generation.")

if __name__ == "__main__":
    main() 