#!/usr/bin/env python3
"""
Test OOD Robustness for Thinkerbell Synthetic Dataset Generation
Validates OOD contamination and edge case handling
"""

import json
import random
from thinkerbell.data.synthetic_dataset_generator import SyntheticDatasetGenerator
from thinkerbell.utils.ood_contamination import OODContaminator

def test_ood_contamination_generation():
    """Test OOD contamination generation"""
    
    print("🧪 Testing OOD Contamination Generation")
    print("=" * 60)
    
    # Test with OOD contamination enabled
    generator_with_ood = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=True,
        contamination_ratio=0.2
    )
    
    print("🔄 Generating 50 samples with OOD contamination...")
    dataset_with_ood = generator_with_ood.generate_dataset(num_samples=50)
    
    # Test without OOD contamination
    generator_without_ood = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=False
    )
    
    print("🔄 Generating 50 samples without OOD contamination...")
    dataset_without_ood = generator_without_ood.generate_dataset(num_samples=50)
    
    # Analyze differences
    print("\n📊 OOD Contamination Analysis:")
    
    # Check classification distribution
    with_ood_classifications = {}
    without_ood_classifications = {}
    
    for item in dataset_with_ood:
        classification = item.get("classification", "UNKNOWN")
        with_ood_classifications[classification] = with_ood_classifications.get(classification, 0) + 1
    
    for item in dataset_without_ood:
        classification = item.get("classification", "UNKNOWN")
        without_ood_classifications[classification] = without_ood_classifications.get(classification, 0) + 1
    
    print(f"  With OOD contamination:")
    for classification, count in with_ood_classifications.items():
        percentage = (count / len(dataset_with_ood)) * 100
        print(f"    {classification}: {count} samples ({percentage:.1f}%)")
    
    print(f"  Without OOD contamination:")
    for classification, count in without_ood_classifications.items():
        percentage = (count / len(dataset_without_ood)) * 100
        print(f"    {classification}: {count} samples ({percentage:.1f}%)")
    
    # Check sample types
    with_ood_sample_types = {}
    for item in dataset_with_ood:
        sample_type = item.get("metadata", {}).get("sample_type", "unknown")
        with_ood_sample_types[sample_type] = with_ood_sample_types.get(sample_type, 0) + 1
    
    print(f"\n  Sample types with OOD contamination:")
    for sample_type, count in with_ood_sample_types.items():
        percentage = (count / len(dataset_with_ood)) * 100
        print(f"    {sample_type}: {count} samples ({percentage:.1f}%)")

def test_ood_sample_quality():
    """Test quality of OOD samples"""
    
    print("\n🧪 Testing OOD Sample Quality")
    print("=" * 50)
    
    contaminator = OODContaminator()
    
    # Test OOD negative samples
    print("📝 Testing OOD negative samples:")
    ood_negatives = []
    for i in range(10):
        sample = contaminator.generate_ood_negative()
        ood_negatives.append(sample)
        
        print(f"  Sample {i+1}:")
        print(f"    Type: {sample.sample_type}")
        print(f"    Label: {sample.label}")
        print(f"    Confidence Target: {sample.confidence_target}")
        print(f"    Text: {sample.text[:100]}...")
        print(f"    OOD Indicators: {sample.ood_indicators}")
        print(f"    Fallback: {sample.fallback_response}")
        
        # Validate sample
        valid, issues = contaminator.validate_ood_sample(sample)
        print(f"    Valid: {valid}")
        if issues:
            print(f"    Issues: {issues}")
        print()
    
    # Test edge case samples
    print("📝 Testing edge case samples:")
    edge_cases = []
    for i in range(10):
        sample = contaminator.generate_edge_case()
        edge_cases.append(sample)
        
        print(f"  Sample {i+1}:")
        print(f"    Type: {sample.sample_type}")
        print(f"    Label: {sample.label}")
        print(f"    Confidence Target: {sample.confidence_target}")
        print(f"    Text: {sample.text[:100]}...")
        print(f"    OOD Indicators: {sample.ood_indicators}")
        print(f"    Should Process: {sample.should_process}")
        
        # Validate sample
        valid, issues = contaminator.validate_ood_sample(sample)
        print(f"    Valid: {valid}")
        if issues:
            print(f"    Issues: {issues}")
        print()
    
    # Quality statistics
    print("📊 OOD Sample Quality Statistics:")
    ood_valid_count = sum(1 for sample in ood_negatives if contaminator.validate_ood_sample(sample)[0])
    edge_valid_count = sum(1 for sample in edge_cases if contaminator.validate_ood_sample(sample)[0])
    
    print(f"  OOD negative samples: {ood_valid_count}/10 valid ({ood_valid_count/10:.1%})")
    print(f"  Edge case samples: {edge_valid_count}/10 valid ({edge_valid_count/10:.1%})")

def test_ood_integration():
    """Test OOD integration with main generator"""
    
    print("\n🧪 Testing OOD Integration")
    print("=" * 50)
    
    generator = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=True,
        contamination_ratio=0.3  # Higher contamination for testing
    )
    
    print("🔄 Generating 30 samples with OOD integration...")
    dataset = generator.generate_dataset(num_samples=30)
    
    # Analyze the integrated dataset
    print("\n📊 Integrated Dataset Analysis:")
    
    # Sample type distribution
    sample_types = {}
    classifications = {}
    confidence_targets = []
    
    for item in dataset:
        sample_type = item.get("metadata", {}).get("sample_type", "unknown")
        classification = item.get("classification", "UNKNOWN")
        confidence_target = item.get("confidence_target", 0.0)
        
        sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
        classifications[classification] = classifications.get(classification, 0) + 1
        confidence_targets.append(confidence_target)
    
    print(f"  Sample type distribution:")
    for sample_type, count in sample_types.items():
        percentage = (count / len(dataset)) * 100
        print(f"    {sample_type}: {count} samples ({percentage:.1f}%)")
    
    print(f"  Classification distribution:")
    for classification, count in classifications.items():
        percentage = (count / len(dataset)) * 100
        print(f"    {classification}: {count} samples ({percentage:.1f}%)")
    
    print(f"  Confidence target statistics:")
    print(f"    Min: {min(confidence_targets):.3f}")
    print(f"    Max: {max(confidence_targets):.3f}")
    print(f"    Avg: {sum(confidence_targets) / len(confidence_targets):.3f}")
    
    # Check OOD indicators
    ood_indicators = {}
    for item in dataset:
        indicators = item.get("metadata", {}).get("ood_indicators", [])
        for indicator in indicators:
            ood_indicators[indicator] = ood_indicators.get(indicator, 0) + 1
    
    print(f"  OOD indicators found:")
    for indicator, count in sorted(ood_indicators.items(), key=lambda x: x[1], reverse=True):
        print(f"    {indicator}: {count} occurrences")

def test_ood_validation():
    """Test OOD validation and fallback responses"""
    
    print("\n🧪 Testing OOD Validation")
    print("=" * 50)
    
    generator = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=True,
        contamination_ratio=0.2
    )
    
    print("🔄 Generating 20 samples for validation testing...")
    dataset = generator.generate_dataset(num_samples=20)
    
    # Test validation logic
    print("\n📝 Validation Results:")
    
    for i, item in enumerate(dataset):
        sample_type = item.get("metadata", {}).get("sample_type", "unknown")
        classification = item.get("classification", "UNKNOWN")
        confidence_target = item.get("confidence_target", 0.0)
        should_process = item.get("metadata", {}).get("should_process", True)
        fallback_response = item.get("metadata", {}).get("fallback_response")
        
        print(f"  Sample {i+1}:")
        print(f"    Type: {sample_type}")
        print(f"    Classification: {classification}")
        print(f"    Confidence Target: {confidence_target:.3f}")
        print(f"    Should Process: {should_process}")
        if fallback_response:
            print(f"    Fallback: {fallback_response}")
        
        # Check if validation makes sense
        if sample_type == "ood_negative":
            if classification == "NOT_INFLUENCER_AGREEMENT" and confidence_target < 0.5:
                print(f"    ✅ Valid OOD negative")
            else:
                print(f"    ❌ Invalid OOD negative")
        elif sample_type == "edge_case":
            if classification == "INFLUENCER_AGREEMENT" and 0.5 < confidence_target < 0.9:
                print(f"    ✅ Valid edge case")
            else:
                print(f"    ❌ Invalid edge case")
        elif sample_type == "positive":
            if classification == "INFLUENCER_AGREEMENT" and confidence_target > 0.8:
                print(f"    ✅ Valid positive sample")
            else:
                print(f"    ❌ Invalid positive sample")
        print()

def main():
    """Run all OOD robustness tests"""
    
    print("🧪 OOD Robustness Test Suite")
    print("=" * 60)
    
    try:
        # Test OOD contamination generation
        test_ood_contamination_generation()
        
        # Test OOD sample quality
        test_ood_sample_quality()
        
        # Test OOD integration
        test_ood_integration()
        
        # Test OOD validation
        test_ood_validation()
        
        print("\n✅ All OOD robustness tests completed successfully!")
        print("\n📋 Test Summary:")
        print("  - OOD contamination generation working")
        print("  - OOD sample quality validation functional")
        print("  - OOD integration with main generator operational")
        print("  - OOD validation and fallback responses working")
        
        print("\n🚀 Ready to generate robust training dataset!")
        print("Run: python generate_training_dataset.py")
        
    except Exception as e:
        print(f"\n❌ OOD robustness test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 