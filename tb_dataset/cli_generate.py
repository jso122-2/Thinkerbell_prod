"""
CLI interface for synthetic data generation using semantic smoother and OOD contamination.

Usage:
    python -m tb_dataset.cli_generate --in data/ingested --out data/synth --n 5000 --ood 0.2
    tb-generate --help

Examples:
    # Generate 1000 samples with 20% OOD
    tb-generate --in ingested_jsonl/ --out synthetic_output/ --n 1000 --ood 0.2
    
    # Quick test generation
    tb-generate --in data/ingested --out test_output --n 100 --verbose
    
    # Large batch with custom settings
    tb-generate --in processed/ --out training_data/ --n 10000 --ood 0.15 --batch-size 50
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm function
    def tqdm(iterable, *args, **kwargs):
        return iterable
import random

from .generate import SyntheticGenerator
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


def create_batch_structure(output_dir: Path) -> Dict[str, Path]:
    """
    Create directory structure for batch output.
    
    Args:
        output_dir: Root output directory
        
    Returns:
        Dictionary with paths to created directories
    """
    paths = {
        'samples': output_dir / 'samples',
        'metadata': output_dir / 'metadata',
        'validation': output_dir / 'validation'
    }
    
    for path in paths.values():
        ensure_dir(path)
    
    return paths


def write_sample_batch(samples: List[Dict[str, Any]], batch_dir: Path, batch_num: int) -> Path:
    """
    Write a batch of samples to individual JSON files.
    
    Args:
        samples: List of sample dictionaries
        batch_dir: Directory for batch files
        batch_num: Batch number for naming
        
    Returns:
        Path to the batch directory
    """
    batch_path = batch_dir / f"batch_{batch_num:03d}"
    ensure_dir(batch_path)
    
    for i, sample in enumerate(samples):
        sample_file = batch_path / f"sample_{i:06d}.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
    
    return batch_path


def write_batch_metadata(samples: List[Dict[str, Any]], metadata_path: Path, 
                        batch_num: int, generator_stats: Dict[str, Any]):
    """
    Write metadata for a batch of samples.
    
    Args:
        samples: List of sample dictionaries
        metadata_path: Path to metadata directory
        batch_num: Batch number
        generator_stats: Generation statistics
    """
    metadata = {
        'batch_number': batch_num,
        'sample_count': len(samples),
        'generation_timestamp': time.time(),
        'generator_version': samples[0]['generator_version'] if samples else 'unknown',
        'statistics': {
            'document_types': {},
            'complexities': {},
            'industries': {},
            'ood_count': 0,
            'coherent_count': 0,
            'fee_distribution': {},
            'token_statistics': {
                'mean': 0,
                'min': 0,
                'max': 0,
                'std': 0
            }
        },
        'generator_stats': generator_stats
    }
    
    # Calculate batch statistics
    token_counts = []
    fee_amounts = []
    
    for sample in samples:
        # Document type distribution
        doc_type = sample['classification']['document_type']
        metadata['statistics']['document_types'][doc_type] = \
            metadata['statistics']['document_types'].get(doc_type, 0) + 1
        
        # Complexity distribution
        complexity = sample['classification']['complexity']
        metadata['statistics']['complexities'][complexity] = \
            metadata['statistics']['complexities'].get(complexity, 0) + 1
        
        # Industry distribution
        industry = sample['classification']['industry']
        metadata['statistics']['industries'][industry] = \
            metadata['statistics']['industries'].get(industry, 0) + 1
        
        # OOD count
        if sample.get('is_ood', False):
            metadata['statistics']['ood_count'] += 1
        
        # Coherence count
        if sample['validation'].get('semantic_coherence', 0) >= 0.1:
            metadata['statistics']['coherent_count'] += 1
        
        # Token statistics
        token_count = sample['raw_input']['token_count']
        token_counts.append(token_count)
        
        # Fee distribution
        fee_numeric = sample['extracted_fields'].get('fee_numeric', 0)
        if fee_numeric > 0:
            fee_amounts.append(fee_numeric)
            fee_bracket = f"{fee_numeric//1000}k" if fee_numeric >= 1000 else "<1k"
            metadata['statistics']['fee_distribution'][fee_bracket] = \
                metadata['statistics']['fee_distribution'].get(fee_bracket, 0) + 1
    
    # Calculate token statistics
    if token_counts:
        import numpy as np
        metadata['statistics']['token_statistics'] = {
            'mean': float(np.mean(token_counts)),
            'min': int(np.min(token_counts)),
            'max': int(np.max(token_counts)),
            'std': float(np.std(token_counts))
        }
    
    # Write metadata file
    metadata_file = metadata_path / f"batch_{batch_num:03d}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def generate_quality_report(all_samples: List[Dict[str, Any]], 
                          generation_stats: Dict[str, Any],
                          generation_time: float) -> str:
    """
    Generate a quality report for the generated samples.
    
    Args:
        all_samples: All generated samples
        generation_stats: Generation statistics
        generation_time: Total generation time
        
    Returns:
        Formatted quality report string
    """
    if not all_samples:
        return "No samples generated."
    
    # Calculate statistics
    total_samples = len(all_samples)
    ood_samples = sum(1 for s in all_samples if s.get('is_ood', False))
    coherent_samples = sum(1 for s in all_samples if s['validation'].get('semantic_coherence', 0) >= 0.1)
    
    # Document type distribution
    doc_types = {}
    for sample in all_samples:
        doc_type = sample['classification']['document_type']
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    # Industry distribution
    industries = {}
    for sample in all_samples:
        industry = sample['classification']['industry']
        industries[industry] = industries.get(industry, 0) + 1
    
    # Fee distribution
    fee_amounts = []
    for sample in all_samples:
        fee_numeric = sample['extracted_fields'].get('fee_numeric', 0)
        if fee_numeric > 0:
            fee_amounts.append(fee_numeric)
    
    # Generate report
    report_lines = [
        "=" * 60,
        "SYNTHETIC DATA GENERATION QUALITY REPORT",
        "=" * 60,
        f"Total samples generated: {total_samples:,}",
        f"Generation time: {generation_time:.2f} seconds",
        f"Average time per sample: {generation_time/total_samples:.3f}s",
        "",
        "SAMPLE DISTRIBUTION:",
        f"  Coherent samples: {coherent_samples:,} ({coherent_samples/total_samples*100:.1f}%)",
        f"  OOD samples: {ood_samples:,} ({ood_samples/total_samples*100:.1f}%)",
        f"  Regular samples: {total_samples-ood_samples:,} ({(total_samples-ood_samples)/total_samples*100:.1f}%)",
        "",
        "DOCUMENT TYPES:",
    ]
    
    for doc_type, count in sorted(doc_types.items()):
        percentage = count / total_samples * 100
        report_lines.append(f"  {doc_type}: {count:,} ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        "TOP INDUSTRIES:",
    ])
    
    # Sort industries by count and show top 5
    top_industries = sorted(industries.items(), key=lambda x: x[1], reverse=True)[:5]
    for industry, count in top_industries:
        percentage = count / total_samples * 100
        report_lines.append(f"  {industry}: {count:,} ({percentage:.1f}%)")
    
    if fee_amounts:
        import numpy as np
        report_lines.extend([
            "",
            "FEE STATISTICS:",
            f"  Mean fee: ${np.mean(fee_amounts):,.0f}",
            f"  Median fee: ${np.median(fee_amounts):,.0f}",
            f"  Fee range: ${np.min(fee_amounts):,.0f} - ${np.max(fee_amounts):,.0f}",
        ])
    
    # Generation statistics
    if generation_stats:
        smoother_stats = generation_stats.get('smoother_stats', {})
        business_stats = generation_stats.get('business_rules_stats', {})
        ood_stats = generation_stats.get('ood_stats', {})
        
        report_lines.extend([
            "",
            "GENERATION STATISTICS:",
            f"  Coherence failures: {generation_stats.get('coherence_failures', 0)}",
            f"  Business rule failures: {generation_stats.get('business_rule_failures', 0)}",
            f"  Token limit failures: {generation_stats.get('token_limit_failures', 0)}",
        ])
        
        if smoother_stats.get('coherence_scores'):
            mean_coherence = smoother_stats.get('mean_coherence', 0)
            report_lines.append(f"  Average coherence score: {mean_coherence:.3f}")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


def main():
    """Main CLI entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic influencer agreement samples with OOD contamination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--in', '--input', dest='input_dir', required=True,
        help='Input directory containing ingested JSONL files'
    )
    parser.add_argument(
        '--out', '--output', dest='output_dir', required=True,
        help='Output directory for synthetic samples'
    )
    
    # Generation parameters
    parser.add_argument(
        '--n', '--samples', type=int, default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--ood', '--ood-ratio', type=float, default=0.2,
        help='Fraction of samples that should be OOD (default: 0.2)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=50,
        help='Number of samples per batch directory (default: 50)'
    )
    
    # Quality parameters
    parser.add_argument(
        '--max-tokens', type=int, default=512,
        help='Maximum tokens per sample (default: 512)'
    )
    parser.add_argument(
        '--coherence-threshold', type=float, default=0.1,
        help='Minimum coherence score required (default: 0.1)'
    )
    parser.add_argument(
        '--target-words', type=str, default='300,600',
        help='Target word count range as min,max (default: 300,600)'
    )
    
    # Generation options
    parser.add_argument(
        '--seed', type=int,
        help='Random seed for reproducible generation'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing output directory'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be generated without creating files'
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
    parser.add_argument(
        '--report', action='store_true',
        help='Generate detailed quality report'
    )
    
    # Device options
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for computation (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose and not args.quiet)
    logger = logging.getLogger(__name__)
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    
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
    
    # Parse target word range
    try:
        word_range = [int(x.strip()) for x in args.target_words.split(',')]
        if len(word_range) != 2 or word_range[0] >= word_range[1]:
            raise ValueError("Invalid word range")
        target_word_range = tuple(word_range)
    except ValueError:
        logger.error("Invalid target word range. Use format: min,max (e.g., 300,600)")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Generating {args.n:,} synthetic samples...")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"OOD ratio: {args.ood:.1%}")
        print(f"Batch size: {args.batch_size}")
    
    if args.dry_run:
        print("\nDRY RUN - No files will be created")
        return 0
    
    # Create output directory structure
    if not args.dry_run:
        paths = create_batch_structure(output_dir)
        logger.info(f"Created output structure in {output_dir}")
    
    # Initialize generator with device support
    device = None if args.device == 'auto' else args.device
    logger.info(f"Initializing synthetic generator with device: {args.device}")
    
    # Log device information
    from .device_utils import log_device_info
    log_device_info()
    
    generator = SyntheticGenerator(
        max_tokens=args.max_tokens,
        target_word_range=target_word_range,
        ood_ratio=args.ood,
        coherence_threshold=args.coherence_threshold,
        device=device
    )
    
    # Load ingested data
    logger.info(f"Loading ingested data from {input_dir}")
    ingested_samples = generator.load_ingested_data(input_dir)
    
    if not ingested_samples:
        logger.error("No ingested samples found. Run ingestion first.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(ingested_samples)} ingested samples")
    
    # Generate samples
    start_time = time.time()
    all_samples = []
    failed_generations = 0
    
    # Calculate batches
    num_batches = (args.n + args.batch_size - 1) // args.batch_size
    
    for batch_num in range(num_batches):
        batch_start = batch_num * args.batch_size
        batch_end = min(batch_start + args.batch_size, args.n)
        batch_size = batch_end - batch_start
        
        if not args.quiet:
            print(f"\nGenerating batch {batch_num + 1}/{num_batches} ({batch_size} samples)...")
        
        batch_samples = []
        
        for i in range(batch_size):
            sample_id = f"sample_{batch_start + i:06d}"
            
            # Show progress
            if not args.quiet and (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{batch_size} samples...", end='\r')
            
            # Generate sample
            sample = generator.generate_sample(sample_id)
            
            if sample:
                batch_samples.append(sample.to_dict())
            else:
                failed_generations += 1
                logger.warning(f"Failed to generate sample {sample_id}")
        
        if not args.quiet:
            print(f"  Generated {len(batch_samples)}/{batch_size} samples")
        
        # Write batch to files
        if batch_samples and not args.dry_run:
            batch_path = write_sample_batch(batch_samples, paths['samples'], batch_num + 1)
            
            # Write batch metadata
            generator_stats = generator.get_stats()
            write_batch_metadata(batch_samples, paths['metadata'], batch_num + 1, generator_stats)
            
            logger.info(f"Wrote batch {batch_num + 1} to {batch_path}")
        
        all_samples.extend(batch_samples)
    
    # Generation complete
    end_time = time.time()
    generation_time = end_time - start_time
    
    if not args.quiet:
        print(f"\nGeneration completed!")
        print(f"Successfully generated: {len(all_samples):,} samples")
        print(f"Failed generations: {failed_generations}")
        print(f"Total time: {generation_time:.2f} seconds")
    
    # Write overall metadata
    if all_samples and not args.dry_run:
        overall_metadata = {
            'generation_summary': {
                'total_samples': len(all_samples),
                'failed_generations': failed_generations,
                'generation_time': generation_time,
                'samples_per_second': len(all_samples) / generation_time,
                'parameters': {
                    'n_samples': args.n,
                    'ood_ratio': args.ood,
                    'max_tokens': args.max_tokens,
                    'coherence_threshold': args.coherence_threshold,
                    'target_word_range': target_word_range,
                    'batch_size': args.batch_size,
                    'seed': args.seed
                }
            },
            'generator_stats': generator.get_stats()
        }
        
        summary_file = paths['metadata'] / 'generation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Wrote generation summary to {summary_file}")
    
    # Generate quality report
    if args.report or args.verbose:
        if all_samples:
            report = generate_quality_report(all_samples, generator.get_stats(), generation_time)
            print("\n" + report)
            
            # Save report to file
            if not args.dry_run:
                report_file = output_dir / 'quality_report.txt'
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Saved quality report to {report_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 