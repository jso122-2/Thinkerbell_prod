#!/usr/bin/env python3
"""
Generate Individual Files for Thinkerbell Synthetic Dataset
Main script for generating structured dataset with individual JSON files
"""

import json
import random
from pathlib import Path
from thinkerbell.data.individual_file_generator import IndividualFileGenerator
from thinkerbell.data.dataset_loader import DatasetLoader
from thinkerbell.data.dataset_validation import DatasetValidator

def main():
    """Generate synthetic dataset using individual file approach"""
    
    print("🎯 Thinkerbell Individual File Dataset Generation")
    print("=" * 60)
    
    # Configuration
    output_dir = "synthetic_dataset"
    total_samples = 1000
    batch_size = 100
    validation_batch_size = 50
    
    # Initialize generator
    generator_config = {
        "use_semantic_smoothing": True,
        "use_text_preprocessing": True,
        "use_ood_contamination": True,
        "contamination_ratio": 0.2
    }
    
    print("🔄 Initializing individual file generator...")
    generator = IndividualFileGenerator(output_dir, generator_config)
    
    # Step 1: Generate validation batch for manual review
    print(f"\n📝 Step 1: Generating validation batch ({validation_batch_size} samples)...")
    validation_samples = generator.generate_validation_batch(size=validation_batch_size)
    
    print(f"✅ Validation batch saved to {generator.output_dir}/validation/manual_review_queue/")
    print(f"📝 Please review {validation_batch_size} samples before proceeding")
    print(f"🔍 Check for:")
    print(f"  - Realistic business scenarios")
    print(f"  - Proper field extraction")
    print(f"  - Appropriate OOD samples")
    print(f"  - Token length compliance")
    
    # Step 2: Generate main dataset in batches
    print(f"\n📊 Step 2: Generating main dataset ({total_samples} samples in {total_samples//batch_size} batches)...")
    
    batches = []
    for batch_num in range(1, (total_samples // batch_size) + 1):
        print(f"\n🔄 Generating batch {batch_num} ({batch_size} samples)...")
        
        batch_metadata = generator.generate_batch(batch_size=batch_size)
        batches.append(batch_metadata)
        
        print(f"✅ Batch {batch_num} complete")
        print(f"  Samples: {len(batch_metadata.samples_generated)}")
        print(f"  Quality metrics: {batch_metadata.quality_metrics}")
    
    # Step 3: Create train/test split
    print(f"\n🔀 Step 3: Creating train/test split...")
    train_count, test_count = generator.create_train_test_split(test_ratio=0.2)
    
    # Step 4: Save comprehensive metadata
    print(f"\n📋 Step 4: Saving comprehensive metadata...")
    generator.save_dataset_metadata()
    
    # Step 5: Load and validate dataset
    print(f"\n🔍 Step 5: Loading and validating dataset...")
    loader = DatasetLoader(output_dir)
    samples = loader.load_dataset(include_ood=True, quality_threshold=0.0)
    
    # Print comprehensive statistics
    loader.print_stats()
    
    # Step 6: Quality analysis
    print(f"\n📊 Step 6: Quality analysis...")
    quality_analysis = loader.analyze_sample_quality()
    
    print(f"Quality Distribution:")
    for category, count in quality_analysis["quality_distribution"].items():
        percentage = (count / quality_analysis["total_samples"]) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    if quality_analysis["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in quality_analysis["recommendations"]:
            print(f"  - {rec}")
    
    # Step 7: Export for training
    print(f"\n📤 Step 7: Exporting for training...")
    training_file = loader.export_for_training("training_dataset.json")
    
    # Step 8: Final summary
    print(f"\n🎉 Individual file generation complete!")
    print(f"📁 Output directory: {generator.output_dir}")
    print(f"📊 Generated {len(samples)} samples")
    print(f"📦 Batches: {len(batches)}")
    print(f"🔀 Train/Test split: {train_count}/{test_count}")
    
    print(f"\n📁 Directory structure:")
    print(f"  {generator.output_dir}/")
    print(f"  ├── metadata/")
    print(f"  │   ├── dataset_metadata.json")
    print(f"  │   └── generation_config.json")
    print(f"  ├── samples/")
    print(f"  │   ├── batch_001/")
    print(f"  │   │   ├── sample_001_simple_fashion_inf.json")
    print(f"  │   │   ├── sample_002_medium_food_inf.json")
    print(f"  │   │   └── batch_metadata.json")
    print(f"  │   └── batch_002/")
    print(f"  ├── validation/")
    print(f"  │   ├── manual_review_queue/")
    print(f"  │   └── approved_samples/")
    print(f"  └── test_set/")
    print(f"      └── holdout_samples/")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Review validation samples in validation/manual_review_queue/")
    print(f"  2. Move approved samples to validation/approved_samples/")
    print(f"  3. Use training_dataset.json for model training")
    print(f"  4. Test with holdout samples in test_set/holdout_samples/")
    print(f"  5. Analyze quality with dataset_loader.py")
    
    print(f"\n🛡️ Features enabled:")
    print(f"  ✅ Individual file generation for version control")
    print(f"  ✅ Batch metadata tracking")
    print(f"  ✅ Train/test split at file level")
    print(f"  ✅ Quality analysis and filtering")
    print(f"  ✅ Manual review queue system")
    print(f"  ✅ Comprehensive metadata tracking")

def test_individual_file_generation():
    """Test the individual file generation with a small dataset"""
    
    print("🧪 Testing Individual File Generation")
    print("=" * 50)
    
    # Test with smaller numbers
    output_dir = "test_synthetic_dataset"
    total_samples = 50
    batch_size = 10
    
    # Initialize generator
    generator_config = {
        "use_semantic_smoothing": True,
        "use_text_preprocessing": True,
        "use_ood_contamination": True,
        "contamination_ratio": 0.2
    }
    
    generator = IndividualFileGenerator(output_dir, generator_config)
    
    # Generate validation batch
    print("📝 Generating validation batch...")
    validation_samples = generator.generate_validation_batch(size=10)
    
    # Generate main batches
    print("📊 Generating main batches...")
    batches = []
    for batch_num in range(1, (total_samples // batch_size) + 1):
        batch_metadata = generator.generate_batch(batch_size=batch_size)
        batches.append(batch_metadata)
    
    # Create train/test split
    print("🔀 Creating train/test split...")
    train_count, test_count = generator.create_train_test_split(test_ratio=0.2)
    
    # Save metadata
    generator.save_dataset_metadata()
    
    # Test loading
    print("📂 Testing dataset loading...")
    loader = DatasetLoader(output_dir)
    samples = loader.load_dataset(include_ood=True)
    
    # Print stats
    loader.print_stats()
    
    # Test export
    training_file = loader.export_for_training("test_training_dataset.json")
    
    print(f"\n✅ Individual file generation test complete!")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Samples: {len(samples)}")
    print(f"📦 Batches: {len(batches)}")
    print(f"🔀 Train/Test: {train_count}/{test_count}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_individual_file_generation()
    else:
        main() 