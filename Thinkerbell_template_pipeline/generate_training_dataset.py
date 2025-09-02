#!/usr/bin/env python3
"""
Generate Training Dataset for Thinkerbell AI Document Formatter
Complete pipeline to generate synthetic data and prepare for training
"""

import os
import sys
import json
from thinkerbell.data.synthetic_dataset_generator import SyntheticDatasetGenerator
from thinkerbell.utils.training_data_preparation import TrainingDataPreparation
from thinkerbell.data.dataset_validation import DatasetValidator

def generate_validation_batch(generator: SyntheticDatasetGenerator, size: int = 100) -> list:
    """Generate a small batch for manual validation first"""
    
    print(f"\n🔍 Step 1.5: Generating validation batch ({size} samples)...")
    
    # Generate validation batch
    validation_dataset = generator.generate_dataset(num_samples=size)
    
    # Save validation batch for manual review
    validation_file = "validation_batch.json"
    generator.save_dataset(validation_dataset, validation_file)
    
    print(f"✅ Validation batch saved to {validation_file}")
    print(f"📝 Please manually review {size} samples before proceeding")
    print(f"🔍 Check for:")
    print(f"  - Realistic business scenarios")
    print(f"  - Proper field extraction")
    print(f"  - Appropriate OOD samples")
    print(f"  - Token length compliance")
    
    return validation_dataset

def run_comprehensive_validation(dataset: list, generator_config: dict) -> bool:
    """Run comprehensive validation on the dataset"""
    
    print(f"\n🔍 Step 1.6: Running comprehensive validation...")
    
    validator = DatasetValidator()
    result = validator.comprehensive_validation(dataset, generator_config)
    
    print(f"\n📊 Validation Results:")
    print(f"  ✅ Passed: {result.passed}")
    print(f"  📈 Overall Score: {result.score:.3f}")
    print(f"  ⚠️  Issues Found: {len(result.issues)}")
    print(f"  💡 Recommendations: {len(result.recommendations)}")
    
    if not result.passed:
        print(f"\n❌ Validation failed! Issues:")
        for issue in result.issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations[:5]:  # Show first 5 recommendations
            print(f"  - {rec}")
        
        return False
    
    print(f"\n✅ Validation passed! Dataset quality is production-ready.")
    return True

def generate_benchmark_test_set(dataset: list, test_ratio: float = 0.2) -> tuple:
    """Generate benchmark test set with extra messy samples"""
    
    print(f"\n🔍 Step 1.7: Generating benchmark test set...")
    
    validator = DatasetValidator()
    train_set, test_set = validator.generate_benchmark_test_set(dataset, test_ratio)
    
    print(f"  📊 Train set: {len(train_set)} samples")
    print(f"  📊 Test set: {len(test_set)} samples")
    print(f"  🧪 Test set includes extra messy/realistic samples")
    
    return train_set, test_set

def create_versioned_metadata(dataset: list, generator_config: dict) -> dict:
    """Create versioned metadata for the dataset"""
    
    print(f"\n📋 Step 1.8: Creating versioned metadata...")
    
    validator = DatasetValidator()
    metadata = validator.create_dataset_metadata(dataset, generator_config)
    
    # Add version control info
    metadata["version"] = "v1.0"
    metadata["generation_pipeline"] = {
        "semantic_smoothing": True,
        "text_preprocessing": True,
        "ood_contamination": True,
        "contamination_ratio": 0.2,
        "validation_enabled": True
    }
    
    return metadata

def main():
    """Run the complete training dataset generation pipeline with validation"""
    
    print("🎯 Thinkerbell AI Document Formatter - Training Dataset Generation")
    print("=" * 80)
    
    # Step 1: Generate synthetic dataset with all robustness features
    print("\n📊 Step 1: Generating synthetic influencer agreements with full robustness...")
    
    # Initialize generator with all features enabled
    generator_config = {
        "use_semantic_smoothing": True,
        "use_text_preprocessing": True,
        "use_ood_contamination": True,
        "contamination_ratio": 0.2
    }
    
    generator = SyntheticDatasetGenerator(**generator_config)
    
    # Generate 1000 samples with balanced complexity distribution
    complexity_distribution = {
        "simple": 0.25,    # 25% simple cases
        "medium": 0.55,    # 55% medium complexity  
        "complex": 0.20    # 20% complex cases
    }
    
    print("🔄 Generating dataset with business logic validation, text preprocessing, and OOD contamination...")
    dataset = generator.generate_dataset(
        num_samples=1000,
        complexity_distribution=complexity_distribution
    )
    
    # Step 1.5: Generate validation batch for manual review
    validation_batch = generate_validation_batch(generator, size=100)
    
    # Step 1.6: Run comprehensive validation
    validation_passed = run_comprehensive_validation(dataset, generator_config)
    
    if not validation_passed:
        print(f"\n❌ Dataset validation failed! Please fix issues and regenerate.")
        print(f"💡 Common fixes:")
        print(f"  - Adjust generator parameters")
        print(f"  - Improve semantic smoother logic")
        print(f"  - Fix text preprocessing")
        print(f"  - Review OOD contamination")
        return
    
    # Step 1.7: Generate benchmark test set
    train_set, test_set = generate_benchmark_test_set(dataset, test_ratio=0.2)
    
    # Step 1.8: Create versioned metadata
    metadata = create_versioned_metadata(dataset, generator_config)
    
    # Save the complete dataset with metadata
    output_file = "synthetic_influencer_agreements.json"
    generator.save_dataset(dataset, output_file)
    
    # Save metadata separately
    with open("dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Generated {len(dataset)} synthetic agreements with full robustness features")
    
    # Step 2: Prepare training data
    print("\n🔄 Step 2: Preparing training data...")
    prep = TrainingDataPreparation(output_file)
    
    # Load the generated dataset
    loaded_dataset = prep.load_dataset()
    if not loaded_dataset:
        print("❌ Failed to load dataset")
        return
    
    # Prepare data for all tasks
    print("📊 Preparing classification data...")
    prep.prepare_classification_data()
    
    print("📊 Preparing extraction data...")
    prep.prepare_extraction_data()
    
    print("📊 Preparing template matching data...")
    prep.prepare_template_matching_data()
    
    # Analyze data quality
    prep.analyze_data_quality()
    
    # Save training data
    prep.save_training_data("training_data")
    
    # Generate final summary
    summary = prep.generate_training_summary()
    print(f"\n📈 Final Summary:")
    print(f"Total synthetic agreements: {summary['dataset_size']}")
    for task_name, task_info in summary['tasks'].items():
        print(f"  {task_name}: {task_info['total_size']} training examples")
    
    # Print semantic smoother quality metrics if available
    if hasattr(generator, 'semantic_smoother') and generator.semantic_smoother:
        print(f"\n🧠 Semantic Smoother Quality Metrics:")
        metrics = generator.semantic_smoother.get_quality_metrics()
        print(f"  Generation efficiency: {metrics['generation_efficiency']:.2%}")
        print(f"  Average semantic coherence: {metrics['semantic_coherence_avg']:.3f}")
    
    # Print OOD contamination statistics
    if hasattr(generator, 'quality_metrics') and 'ood_stats' in generator.quality_metrics:
        print(f"\n🛡️ OOD Contamination Statistics:")
        ood_stats = generator.quality_metrics['ood_stats']
        print(f"  Positive samples: {ood_stats['positive_samples']}")
        print(f"  OOD negative samples: {ood_stats['ood_negative_samples']}")
        print(f"  Edge case samples: {ood_stats['edge_case_samples']}")
        print(f"  Contamination ratio: {ood_stats['contamination_ratio']:.1%}")
    
    # Print classification distribution
    classification_dist = {}
    for item in dataset:
        classification = item.get("classification", "UNKNOWN")
        classification_dist[classification] = classification_dist.get(classification, 0) + 1
    
    print(f"\n📊 Classification Distribution:")
    for classification, count in classification_dist.items():
        percentage = (count / len(dataset)) * 100
        print(f"  {classification}: {count} samples ({percentage:.1f}%)")
    
    # Print distribution balance
    validator = DatasetValidator()
    distribution_scores = validator.validate_distribution_balance(dataset)
    print(f"\n📊 Distribution Balance Scores:")
    for metric, score in distribution_scores.items():
        print(f"  {metric}: {score:.3f}")
    
    print(f"\n🎉 Training dataset generation complete!")
    print(f"📁 Files created:")
    print(f"  - {output_file} (raw synthetic data with full robustness)")
    print(f"  - validation_batch.json (manual review sample)")
    print(f"  - dataset_metadata.json (versioned metadata)")
    print(f"  - training_data/ (prepared training datasets)")
    print(f"  - training_data/metadata.json (dataset metadata)")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Review validation_batch.json manually")
    print(f"  2. Check training_data/ directory for prepared datasets")
    print(f"  3. Use the data to train your sentence encoder pipeline")
    print(f"  4. Test with real influencer agreements from data/ directory")
    print(f"  5. Validate semantic coherence against real business scenarios")
    print(f"  6. Test OOD detection on non-influencer agreement documents")
    print(f"  7. Use benchmark test set for final performance evaluation")
    
    print(f"\n🛡️ Robustness Features Enabled:")
    print(f"  ✅ Semantic smoothing for business logic coherence")
    print(f"  ✅ Text preprocessing for sentence encoder optimization")
    print(f"  ✅ OOD contamination for production robustness")
    print(f"  ✅ Token length management for transformer models")
    print(f"  ✅ Quality metrics tracking and validation")
    print(f"  ✅ Distribution balance tracking")
    print(f"  ✅ Human validation samples")
    print(f"  ✅ Benchmark test set generation")
    print(f"  ✅ Version control and metadata")

if __name__ == "__main__":
    main() 