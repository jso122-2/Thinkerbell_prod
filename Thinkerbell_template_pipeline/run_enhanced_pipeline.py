#!/usr/bin/env python3
"""
Enhanced Data Pipeline Runner for Thinkerbell
Orchestrates the complete enhanced labeling and splitting process to address overfitting
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_labeling_system import EnhancedLabelingSystem
from enhanced_split_chunks import EnhancedChunkSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_enhanced_pipeline(
    input_dir: str = "synthetic_dataset/splits",
    output_dir: str = "synthetic_dataset/enhanced_pipeline",
    distractor_ratio: float = 0.15,
    multi_label_ratio: float = 0.25,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = "complexity",
    seed: int = 42
):
    """
    Run the complete enhanced data pipeline
    
    Args:
        input_dir: Directory containing original split data
        output_dir: Output directory for enhanced pipeline
        distractor_ratio: Proportion of distractor samples to generate
        multi_label_ratio: Proportion of multi-label samples to generate
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        stratify_by: Stratification method for splitting
        seed: Random seed for reproducibility
    """
    
    logger.info("=" * 60)
    logger.info("ENHANCED THINKERBELL DATA PIPELINE")
    logger.info("=" * 60)
    logger.info("Addressing overfitting by:")
    logger.info("1. Expanding label space with sub-labels")
    logger.info("2. Creating multi-label samples")  
    logger.info("3. Adding 'none of the above' distractor class")
    logger.info("4. Maintaining proportional distribution")
    logger.info("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Enhanced Labeling
    logger.info("\n🔧 STEP 1: ENHANCED LABELING SYSTEM")
    logger.info("-" * 40)
    
    labeling_system = EnhancedLabelingSystem(seed=seed)
    
    # Process original dataset with enhanced labels
    enhanced_stats = labeling_system.process_dataset(
        input_dir=input_dir,
        output_dir=str(output_path),
        distractor_ratio=distractor_ratio,
        multi_label_ratio=multi_label_ratio
    )
    
    # Log enhanced labeling results
    logger.info(f"\n✅ Enhanced labeling complete!")
    logger.info(f"   Original samples: {enhanced_stats.get('sample_types', {}).get('enhanced_original', 0)}")
    logger.info(f"   Multi-label samples: {enhanced_stats.get('sample_types', {}).get('multi_label', 0)}")
    logger.info(f"   Distractor samples: {enhanced_stats.get('sample_types', {}).get('distractor', 0)}")
    logger.info(f"   Total samples: {enhanced_stats.get('total_samples', 0)}")
    logger.info(f"   Unique combinations: {enhanced_stats.get('num_unique_combinations', 0)}")
    
    # Step 2: Enhanced Splitting
    logger.info(f"\n🔀 STEP 2: ENHANCED SPLITTING")
    logger.info("-" * 40)
    
    # Setup paths for splitting
    enhanced_samples_file = output_path / "enhanced_splits" / "all_enhanced_samples.json"
    split_output_dir = output_path / "enhanced_splits"
    
    if not enhanced_samples_file.exists():
        logger.error(f"Enhanced samples file not found: {enhanced_samples_file}")
        logger.error("Please check that Step 1 completed successfully.")
        return None
    
    # Initialize enhanced splitter
    splitter = EnhancedChunkSplitter(
        input_file=str(enhanced_samples_file),
        output_dir=str(split_output_dir),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_by=stratify_by
    )
    
    # Process enhanced splits
    split_stats = splitter.process_enhanced_splits()
    
    # Log splitting results
    logger.info(f"\n✅ Enhanced splitting complete!")
    logger.info(f"   Train samples: {split_stats.get('train_samples', 0)}")
    logger.info(f"   Val samples: {split_stats.get('val_samples', 0)}")
    logger.info(f"   Test samples: {split_stats.get('test_samples', 0)}")
    logger.info(f"   Unique labels: {len(split_stats.get('label_distribution', {}))}")
    logger.info(f"   Unique combinations: {len(split_stats.get('label_combination_distribution', {}))}")
    
    # Step 3: Validation and Summary
    logger.info(f"\n📊 STEP 3: VALIDATION & SUMMARY")
    logger.info("-" * 40)
    
    # Validate improvements
    improvements = validate_improvements(enhanced_stats, split_stats)
    
    # Generate summary report
    summary_report = generate_summary_report(
        enhanced_stats, split_stats, improvements, 
        input_dir, output_dir, 
        distractor_ratio, multi_label_ratio
    )
    
    # Save summary report
    summary_file = output_path / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Complete summary saved to: {summary_file}")
    
    # Final success message
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ENHANCED PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info("Key improvements achieved:")
    for improvement in improvements['improvements']:
        logger.info(f"✓ {improvement}")
    logger.info("\nNext steps:")
    logger.info("1. Use enhanced_splits/train,val,test directories for training")
    logger.info("2. Update training scripts to handle enhanced_labels field")
    logger.info("3. Monitor for overfitting reduction during training")
    logger.info("=" * 60)
    
    return summary_report

def validate_improvements(enhanced_stats: dict, split_stats: dict) -> dict:
    """Validate that improvements address the original overfitting issues"""
    
    improvements = []
    warnings = []
    
    # Check label space expansion
    unique_combinations = enhanced_stats.get('num_unique_combinations', 0)
    if unique_combinations >= 50:
        improvements.append(f"Expanded to {unique_combinations} unique label combinations (target: 50+)")
    else:
        warnings.append(f"Only {unique_combinations} unique combinations (target: 50+)")
    
    # Check multi-label complexity
    sample_types = enhanced_stats.get('sample_types', {})
    multi_label_count = sample_types.get('multi_label', 0)
    total_samples = enhanced_stats.get('total_samples', 1)
    multi_label_ratio = multi_label_count / total_samples
    
    if multi_label_ratio >= 0.20:
        improvements.append(f"Added {multi_label_count} multi-label samples ({multi_label_ratio:.1%})")
    else:
        warnings.append(f"Low multi-label ratio: {multi_label_ratio:.1%}")
    
    # Check distractor samples
    distractor_count = sample_types.get('distractor', 0)
    distractor_ratio = distractor_count / total_samples
    
    if distractor_ratio >= 0.10:
        improvements.append(f"Added {distractor_count} distractor samples ({distractor_ratio:.1%})")
    else:
        warnings.append(f"Low distractor ratio: {distractor_ratio:.1%}")
    
    # Check label distribution balance
    label_dist = split_stats.get('label_distribution', {})
    balanced_labels = 0
    total_labels = len(label_dist)
    
    for label, dist in label_dist.items():
        if dist.get('total', 0) > 0:
            train_ratio = dist.get('train', 0) / dist.get('total', 1)
            if 0.6 <= train_ratio <= 0.8:  # Reasonable train ratio
                balanced_labels += 1
    
    if total_labels > 0:
        balance_ratio = balanced_labels / total_labels
        if balance_ratio >= 0.8:
            improvements.append(f"Maintained balanced distribution ({balance_ratio:.1%} of labels)")
        else:
            warnings.append(f"Some label distribution imbalance ({balance_ratio:.1%} balanced)")
    
    # Check complexity distribution
    complexity_dist = split_stats.get('complexity_distribution', {})
    if len(complexity_dist) >= 3:
        improvements.append(f"Created {len(complexity_dist)} complexity levels for varied difficulty")
    
    return {
        'improvements': improvements,
        'warnings': warnings,
        'metrics': {
            'unique_combinations': unique_combinations,
            'multi_label_ratio': multi_label_ratio,
            'distractor_ratio': distractor_ratio,
            'balanced_labels_ratio': balance_ratio if total_labels > 0 else 0
        }
    }

def generate_summary_report(enhanced_stats: dict, split_stats: dict, improvements: dict,
                          input_dir: str, output_dir: str, 
                          distractor_ratio: float, multi_label_ratio: float) -> dict:
    """Generate comprehensive summary report"""
    
    return {
        'pipeline_info': {
            'input_directory': input_dir,
            'output_directory': output_dir,
            'distractor_ratio': distractor_ratio,
            'multi_label_ratio': multi_label_ratio,
            'timestamp': str(Path().cwd()),  # placeholder for actual timestamp
        },
        'original_problem': {
            'description': "Overfitting due to trivial classification task",
            'issues': [
                "Only 5 label combinations for 1,387 chunks",
                "Strong, repetitive text patterns",
                "Model can easily memorize patterns"
            ]
        },
        'solution_implemented': {
            'label_space_expansion': {
                'sub_labels': "Added granular sub-labels for each base label",
                'content_specific': "Added platform and content-type specific labels",
                'complexity_levels': "Created 4 complexity levels (minimal, moderate, complex, comprehensive)"
            },
            'multi_label_samples': {
                'description': "Created samples with 2-3 overlapping label combinations",
                'count': enhanced_stats.get('sample_types', {}).get('multi_label', 0),
                'ratio': enhanced_stats.get('sample_types', {}).get('multi_label', 0) / enhanced_stats.get('total_samples', 1)
            },
            'distractor_class': {
                'description': "Added realistic 'none of the above' samples from different domains",
                'categories': ["employment", "real_estate", "financial", "legal_general", "academic", "medical", "technical"],
                'count': enhanced_stats.get('sample_types', {}).get('distractor', 0),
                'ratio': enhanced_stats.get('sample_types', {}).get('distractor', 0) / enhanced_stats.get('total_samples', 1)
            }
        },
        'results': {
            'enhanced_labeling': enhanced_stats,
            'splitting': {
                'total_samples': split_stats.get('total_samples', 0),
                'train_samples': split_stats.get('train_samples', 0),
                'val_samples': split_stats.get('val_samples', 0),
                'test_samples': split_stats.get('test_samples', 0),
                'unique_labels': len(split_stats.get('label_distribution', {})),
                'unique_combinations': len(split_stats.get('label_combination_distribution', {}))
            },
            'improvements': improvements
        },
        'training_recommendations': [
            "Use enhanced_labels field instead of original labels",
            "Monitor validation accuracy - should not reach 100% as easily",
            "Implement early stopping if validation accuracy plateaus",
            "Consider using label smoothing or mixup augmentation",
            "Track per-complexity-level performance to ensure balanced learning"
        ],
        'file_locations': {
            'enhanced_samples': f"{output_dir}/enhanced_splits/all_enhanced_samples.json",
            'train_split': f"{output_dir}/enhanced_splits/train/",
            'val_split': f"{output_dir}/enhanced_splits/val/",
            'test_split': f"{output_dir}/enhanced_splits/test/",
            'statistics': f"{output_dir}/enhanced_splits/enhanced_split_stats.json"
        }
    }

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced Data Pipeline to Address Overfitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults
    python run_enhanced_pipeline.py
    
    # Custom ratios
    python run_enhanced_pipeline.py --distractor-ratio 0.2 --multi-label-ratio 0.3
    
    # Custom splitting
    python run_enhanced_pipeline.py --stratify-by sample_type --train-ratio 0.8
        """
    )
    
    parser.add_argument("--input-dir", type=str, default="synthetic_dataset/splits",
                       help="Input directory with original splits")
    parser.add_argument("--output-dir", type=str, default="synthetic_dataset/enhanced_pipeline", 
                       help="Output directory for enhanced pipeline")
    parser.add_argument("--distractor-ratio", type=float, default=0.15,
                       help="Ratio of distractor samples to generate (default: 0.15)")
    parser.add_argument("--multi-label-ratio", type=float, default=0.25,
                       help="Ratio of multi-label samples to generate (default: 0.25)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test set ratio (default: 0.15)")
    parser.add_argument("--stratify-by", type=str, default="complexity",
                       choices=["complexity", "label_count", "sample_type"],
                       help="Stratification method (default: complexity)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 0.001:
        logger.error("Train, val, and test ratios must sum to 1.0")
        return 1
    
    if not (0.0 <= args.distractor_ratio <= 0.5):
        logger.error("Distractor ratio must be between 0.0 and 0.5")
        return 1
        
    if not (0.0 <= args.multi_label_ratio <= 0.5):
        logger.error("Multi-label ratio must be between 0.0 and 0.5")
        return 1
    
    # Run pipeline
    try:
        summary = run_enhanced_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            distractor_ratio=args.distractor_ratio,
            multi_label_ratio=args.multi_label_ratio,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify_by=args.stratify_by,
            seed=args.seed
        )
        
        if summary:
            logger.info("Pipeline completed successfully!")
            return 0
        else:
            logger.error("Pipeline failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 