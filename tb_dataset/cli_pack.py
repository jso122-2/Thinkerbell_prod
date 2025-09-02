"""
CLI interface for dataset packaging with deduplication, splitting, and EDA.

Usage:
    python -m tb_dataset.cli_pack --in data/synth/samples --out data/final --val 0.15 --test 0.15 --sim-thresh 0.95
    tb-pack --help

Examples:
    # Full pipeline with deduplication and EDA
    tb-pack --in synthetic_samples/ --out final_dataset/ --val 0.15 --test 0.15
    
    # Custom similarity threshold for near-duplicates
    tb-pack --in samples/ --out dataset/ --val 0.2 --test 0.2 --sim-thresh 0.98
    
    # Skip EDA for faster processing
    tb-pack --in samples/ --out dataset/ --no-eda --val 0.15 --test 0.15
    
    # With custom random seed
    tb-pack --in samples/ --out dataset/ --seed 123 --val 0.15 --test 0.15
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm function
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .dedup import DuplicateDetector
from .split import GroupAwareSplitter
from .eda import DatasetAnalyzer
from .utils import ensure_dir


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def copy_samples_to_split_dirs(samples: Dict[str, Dict[str, Any]], 
                              splits: Dict[str, List[str]], 
                              output_dir: Path) -> Dict[str, Path]:
    """
    Copy samples to split-specific directories.
    
    Args:
        samples: Dictionary of sample_id -> sample_data
        splits: Split assignments
        output_dir: Output directory
        
    Returns:
        Dictionary mapping split -> directory path
    """
    split_dirs = {}
    
    for split_name, sample_ids in splits.items():
        split_dir = output_dir / split_name
        ensure_dir(split_dir)
        split_dirs[split_name] = split_dir
        
        # Copy samples to split directory
        for sample_id in sample_ids:
            if sample_id in samples:
                sample_data = samples[sample_id]
                sample_file = split_dir / f"{sample_id}.json"
                
                with open(sample_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    return split_dirs


def create_flattened_jsonl(samples: Dict[str, Dict[str, Any]], 
                          splits: Dict[str, List[str]], 
                          output_dir: Path):
    """
    Create flattened JSONL files for easy loading.
    
    Args:
        samples: Dictionary of sample_id -> sample_data
        splits: Split assignments  
        output_dir: Output directory
    """
    for split_name, sample_ids in splits.items():
        jsonl_file = output_dir / f"{split_name}.jsonl"
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for sample_id in sample_ids:
                if sample_id in samples:
                    sample_data = samples[sample_id]
                    f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
        
        logger.info(f"Created flattened JSONL: {jsonl_file} ({len(sample_ids)} samples)")


def create_split_statistics(samples: Dict[str, Dict[str, Any]], 
                           splits: Dict[str, List[str]], 
                           splitter_stats: Dict[str, Any],
                           dedup_stats: Dict[str, Any],
                           output_dir: Path):
    """
    Create comprehensive split statistics.
    
    Args:
        samples: Dictionary of sample_id -> sample_data
        splits: Split assignments
        splitter_stats: Statistics from GroupAwareSplitter
        dedup_stats: Statistics from DuplicateDetector
        output_dir: Output directory
    """
    # Calculate detailed statistics
    split_stats = {
        'generation_info': {
            'total_samples_before_dedup': dedup_stats.get('total_samples', 0),
            'samples_removed_in_dedup': dedup_stats.get('samples_removed', 0),
            'final_sample_count': len(samples),
            'exact_duplicates_found': dedup_stats.get('exact_duplicates_found', 0),
            'near_duplicates_found': dedup_stats.get('near_duplicates_found', 0),
        },
        'split_distribution': {},
        'label_distributions': {},
        'length_statistics': {},
        'quality_metrics': {}
    }
    
    # Split distribution
    for split_name, sample_ids in splits.items():
        split_count = len(sample_ids)
        split_stats['split_distribution'][split_name] = {
            'count': split_count,
            'percentage': split_count / len(samples) * 100 if len(samples) > 0 else 0
        }
    
    # Label distributions per split
    for split_name, sample_ids in splits.items():
        split_stats['label_distributions'][split_name] = {
            'document_types': {},
            'complexities': {},
            'industries': {},
            'ood_count': 0,
            'style_distribution': {}
        }
        
        for sample_id in sample_ids:
            if sample_id not in samples:
                continue
                
            sample_data = samples[sample_id]
            classification = sample_data.get('classification', {})
            
            # Document types
            doc_type = classification.get('document_type', 'unknown')
            split_stats['label_distributions'][split_name]['document_types'][doc_type] = \
                split_stats['label_distributions'][split_name]['document_types'].get(doc_type, 0) + 1
            
            # Complexities
            complexity = classification.get('complexity', 'unknown')
            split_stats['label_distributions'][split_name]['complexities'][complexity] = \
                split_stats['label_distributions'][split_name]['complexities'].get(complexity, 0) + 1
            
            # Industries
            industry = classification.get('industry', 'unknown')
            split_stats['label_distributions'][split_name]['industries'][industry] = \
                split_stats['label_distributions'][split_name]['industries'].get(industry, 0) + 1
            
            # OOD count
            if sample_data.get('is_ood', False):
                split_stats['label_distributions'][split_name]['ood_count'] += 1
            
            # Style distribution
            style = sample_data.get('raw_input', {}).get('style', 'unknown')
            split_stats['label_distributions'][split_name]['style_distribution'][style] = \
                split_stats['label_distributions'][split_name]['style_distribution'].get(style, 0) + 1
    
    # Length statistics per split
    for split_name, sample_ids in splits.items():
        char_lengths = []
        token_lengths = []
        
        for sample_id in sample_ids:
            if sample_id not in samples:
                continue
                
            sample_data = samples[sample_id]
            raw_input = sample_data.get('raw_input', {})
            
            # Character length
            text = raw_input.get('text', '')
            char_lengths.append(len(text))
            
            # Token length
            token_count = raw_input.get('token_count', 0)
            token_lengths.append(token_count)
        
        if char_lengths:
            import numpy as np
            split_stats['length_statistics'][split_name] = {
                'character_lengths': {
                    'mean': float(np.mean(char_lengths)),
                    'median': float(np.median(char_lengths)),
                    'std': float(np.std(char_lengths)),
                    'min': int(np.min(char_lengths)),
                    'max': int(np.max(char_lengths))
                },
                'token_lengths': {
                    'mean': float(np.mean(token_lengths)),
                    'median': float(np.median(token_lengths)),
                    'std': float(np.std(token_lengths)),
                    'min': int(np.min(token_lengths)),
                    'max': int(np.max(token_lengths))
                }
            }
    
    # Quality metrics
    total_coherence_scores = []
    business_ok_count = 0
    temporal_ok_count = 0
    
    for sample_data in samples.values():
        validation = sample_data.get('validation', {})
        
        coherence = validation.get('semantic_coherence', 0)
        total_coherence_scores.append(coherence)
        
        if validation.get('business_ok', False):
            business_ok_count += 1
        
        if validation.get('temporal_ok', False):
            temporal_ok_count += 1
    
    if total_coherence_scores:
        import numpy as np
        split_stats['quality_metrics'] = {
            'coherence_scores': {
                'mean': float(np.mean(total_coherence_scores)),
                'median': float(np.median(total_coherence_scores)),
                'std': float(np.std(total_coherence_scores)),
                'min': float(np.min(total_coherence_scores)),
                'max': float(np.max(total_coherence_scores))
            },
            'business_validation': {
                'passed': business_ok_count,
                'percentage': business_ok_count / len(samples) * 100 if len(samples) > 0 else 0
            },
            'temporal_validation': {
                'passed': temporal_ok_count,
                'percentage': temporal_ok_count / len(samples) * 100 if len(samples) > 0 else 0
            }
        }
    
    # Add splitter statistics
    split_stats['group_aware_splitting'] = splitter_stats
    
    # Add deduplication statistics
    split_stats['deduplication'] = dedup_stats
    
    # Save statistics
    stats_file = output_dir / 'split_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved split statistics to {stats_file}")
    
    return split_stats


def print_red_flag_summary(split_stats: Dict[str, Any], red_flags: List[str]):
    """Print red flag summary for potential issues."""
    
    print("\n" + "🚩" * 20 + " RED FLAG SUMMARY " + "🚩" * 20)
    
    if not red_flags:
        print("✅ No major issues detected in the dataset!")
    else:
        print(f"⚠️  {len(red_flags)} potential issues found:")
        for i, flag in enumerate(red_flags, 1):
            print(f"  {i}. {flag}")
    
    print("\n" + "📊" * 15 + " DATASET OVERVIEW " + "📊" * 15)
    
    # Overall statistics
    gen_info = split_stats.get('generation_info', {})
    print(f"Total samples after dedup: {gen_info.get('final_sample_count', 0):,}")
    print(f"Samples removed in dedup: {gen_info.get('samples_removed_in_dedup', 0):,}")
    
    # Split distribution
    split_dist = split_stats.get('split_distribution', {})
    for split_name, stats in split_dist.items():
        print(f"{split_name.capitalize()}: {stats['count']:,} samples ({stats['percentage']:.1f}%)")
    
    # Quality metrics
    quality = split_stats.get('quality_metrics', {})
    if quality:
        coherence = quality.get('coherence_scores', {})
        business = quality.get('business_validation', {})
        temporal = quality.get('temporal_validation', {})
        
        print(f"\nQuality Metrics:")
        print(f"  Mean coherence: {coherence.get('mean', 0):.3f}")
        print(f"  Business validation: {business.get('percentage', 0):.1f}% passed")
        print(f"  Temporal validation: {temporal.get('percentage', 0):.1f}% passed")
    
    print("🚩" * 60)


def main():
    """Main CLI entry point for dataset packaging."""
    parser = argparse.ArgumentParser(
        description="Package synthetic samples with deduplication, splitting, and EDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--in', '--input', dest='input_dir', required=True,
        help='Input directory containing synthetic sample batch directories'
    )
    parser.add_argument(
        '--out', '--output', dest='output_dir', required=True,
        help='Output directory for final packaged dataset'
    )
    
    # Split parameters
    parser.add_argument(
        '--val', '--validation', type=float, default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test', type=float, default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    
    # Deduplication parameters
    parser.add_argument(
        '--sim-thresh', '--similarity-threshold', type=float, default=0.95,
        help='Cosine similarity threshold for near-duplicates (default: 0.95)'
    )
    parser.add_argument(
        '--no-dedup', action='store_true',
        help='Skip deduplication step'
    )
    
    # EDA parameters
    parser.add_argument(
        '--no-eda', action='store_true',
        help='Skip EDA report generation'
    )
    parser.add_argument(
        '--eda-dir', type=str,
        help='Custom directory for EDA outputs (default: output_dir/eda)'
    )
    
    # Processing options
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing output directory'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be processed without creating files'
    )
    
    # Output options  
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose and not args.quiet)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        if not args.dry_run:
            logger.error(f"Output directory exists: {output_dir}. Use --overwrite to replace.")
            sys.exit(1)
    
    # Validate split ratios
    train_ratio = 1.0 - args.val - args.test
    if train_ratio <= 0:
        logger.error(f"Invalid split ratios: train={train_ratio:.2f}, val={args.val}, test={args.test}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Processing samples from: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Split ratios: train={train_ratio:.2f}, val={args.val}, test={args.test}")
        if not args.no_dedup:
            print(f"Similarity threshold: {args.sim_thresh}")
    
    if args.dry_run:
        print("\nDRY RUN - No files will be created")
        return 0
    
    # Create output directory
    if not args.dry_run:
        ensure_dir(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    start_time = time.time()
    
    # Step 1: Deduplication
    if not args.no_dedup:
        if not args.quiet:
            print("\n📄 Step 1: Deduplication")
        
        logger.info("Starting deduplication process...")
        detector = DuplicateDetector(similarity_threshold=args.sim_thresh)
        
        samples, samples_to_remove, removal_reasons = detector.process_directory(input_dir)
        clean_samples = detector.get_clean_samples(samples, samples_to_remove)
        dedup_stats = detector.get_stats()
        
        # Save deduplication report
        if not args.dry_run:
            dedup_report_dir = output_dir / 'deduplication_reports'
            detector.save_deduplication_report(dedup_report_dir, samples_to_remove, removal_reasons)
        
        if not args.quiet:
            print(f"   Processed {len(samples):,} samples")
            print(f"   Removed {len(samples_to_remove):,} duplicates")
            print(f"   Final clean dataset: {len(clean_samples):,} samples")
    
    else:
        if not args.quiet:
            print("\n📄 Step 1: Loading samples (skipping deduplication)")
        
        # Load samples without deduplication
        detector = DuplicateDetector()
        samples = detector.load_samples_from_directory(input_dir)
        clean_samples = samples
        dedup_stats = {'total_samples': len(samples), 'samples_removed': 0}
        
        if not args.quiet:
            print(f"   Loaded {len(samples):,} samples")
    
    if not clean_samples:
        logger.error("No samples to process after deduplication")
        sys.exit(1)
    
    # Step 2: Group-aware splitting
    if not args.quiet:
        print("\n📊 Step 2: Group-aware splitting")
    
    logger.info("Starting group-aware splitting...")
    splitter = GroupAwareSplitter(
        train_ratio=train_ratio,
        val_ratio=args.val, 
        test_ratio=args.test,
        random_seed=args.seed
    )
    
    splits = splitter.split_samples(clean_samples)
    splitter_stats = splitter.get_split_statistics()
    
    if not args.quiet:
        print(f"   Train: {len(splits['train']):,} samples")
        print(f"   Validation: {len(splits['val']):,} samples") 
        print(f"   Test: {len(splits['test']):,} samples")
    
    # Step 3: Package dataset
    if not args.quiet:
        print("\n📦 Step 3: Packaging dataset")
    
    logger.info("Packaging dataset files...")
    
    if not args.dry_run:
        # Copy samples to split directories
        split_dirs = copy_samples_to_split_dirs(clean_samples, splits, output_dir)
        
        # Create flattened JSONL files
        create_flattened_jsonl(clean_samples, splits, output_dir)
        
        # Create split statistics
        split_stats = create_split_statistics(
            clean_samples, splits, splitter_stats, dedup_stats, output_dir
        )
        
        # Save split metadata
        splitter.save_split_metadata(output_dir / 'metadata', splits, clean_samples)
    
    if not args.quiet:
        for split_name in splits.keys():
            print(f"   Created {split_name}/ directory and {split_name}.jsonl")
    
    # Step 4: EDA Analysis
    if not args.no_eda:
        if not args.quiet:
            print("\n📈 Step 4: EDA Analysis")
        
        logger.info("Generating EDA analysis...")
        
        eda_dir = Path(args.eda_dir) if args.eda_dir else output_dir / 'eda'
        analyzer = DatasetAnalyzer(eda_dir)
        
        if not args.dry_run:
            analysis_results = analyzer.analyze_dataset(clean_samples, splits)
            red_flags = analyzer.generate_red_flag_summary(analysis_results)
        else:
            red_flags = []
        
        if not args.quiet:
            print(f"   Generated charts and analysis in {eda_dir}")
    else:
        if not args.quiet:
            print("\n📈 Step 4: EDA Analysis (skipped)")
        red_flags = []
    
    # Processing complete
    end_time = time.time()
    processing_time = end_time - start_time
    
    if not args.quiet:
        print(f"\n✅ Packaging complete in {processing_time:.2f} seconds!")
    
    # Print summary and red flags
    if not args.dry_run and not args.quiet:
        if not args.no_eda:
            print_red_flag_summary(split_stats, red_flags)
        else:
            # Print basic summary without EDA red flags
            print(f"\nFinal dataset: {len(clean_samples):,} samples")
            print(f"Train: {len(splits['train']):,}, Val: {len(splits['val']):,}, Test: {len(splits['test']):,}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 