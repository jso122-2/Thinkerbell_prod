"""
CLI interface for synthetic preamble generation

Usage:
    python -m tb_dataset.cli_synthetic --count 100 --out synthetic_preambles/
    tb-synthetic --count 500 --industries fashion food tech --complexity medium
    tb-synthetic --help

Examples:
    # Generate 100 mixed preambles
    tb-synthetic --count 100 --out training_data/

    # Generate specific industries
    tb-synthetic --count 200 --industries fashion beauty --out preambles/

    # Control complexity distribution
    tb-synthetic --count 500 --simple 0.2 --medium 0.6 --complex 0.2

    # With processing pipeline
    tb-synthetic --count 300 --out processed/ --clean --chunk --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time

from .synthetic_preamble import SyntheticPreambleGenerator
from .schema import DocumentSample
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


def validate_complexity_distribution(simple: float, medium: float, complex: float) -> Dict[str, float]:
    """Validate and normalize complexity distribution."""
    total = simple + medium + complex
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Complexity distribution must sum to 1.0, got {total}")
    
    return {
        "simple": simple / total,
        "medium": medium / total, 
        "complex": complex / total
    }


def generate_synthetic_preambles(args) -> None:
    """Generate synthetic business preambles."""
    
    logger = logging.getLogger(__name__)
    
    # Setup output directory
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    
    # Initialize generator
    logger.info("Initializing synthetic preamble generator...")
    generator = SyntheticPreambleGenerator()
    
    # Validate complexity distribution
    complexity_dist = validate_complexity_distribution(
        args.simple, args.medium, args.complex
    )
    
    logger.info(f"Complexity distribution: {complexity_dist}")
    
    # Validate industries
    available_industries = list(generator.brands_by_industry.keys())
    if args.industries:
        invalid_industries = set(args.industries) - set(available_industries)
        if invalid_industries:
            raise ValueError(f"Invalid industries: {invalid_industries}. "
                           f"Available: {available_industries}")
        industries = args.industries
    else:
        industries = available_industries
    
    logger.info(f"Using industries: {industries}")
    
    # Generate synthetic documents
    start_time = time.time()
    logger.info(f"Generating {args.count} synthetic preambles...")
    
    synthetic_docs = generator.generate_synthetic_documents(
        count=args.count,
        complexity_distribution=complexity_dist,
        industries=industries
    )
    
    generation_time = time.time() - start_time
    logger.info(f"Generated {len(synthetic_docs)} documents in {generation_time:.2f}s")
    
    # Save raw synthetic documents
    if args.save_raw:
        raw_file = output_dir / "raw_synthetic_documents.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(synthetic_docs, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved raw documents to {raw_file}")
    
    # Create DocumentSample objects with processing
    start_time = time.time()
    logger.info("Creating DocumentSample objects and processing...")
    
    samples = generator.create_document_samples(
        synthetic_docs=synthetic_docs,
        output_dir=output_dir,
        clean=args.clean,
        chunk=args.chunk
    )
    
    processing_time = time.time() - start_time
    logger.info(f"Processed {len(samples)} samples in {processing_time:.2f}s")
    
    # Export to JSONL
    start_time = time.time()
    output_file = output_dir / args.output_file
    generator.export_to_jsonl(samples, output_file)
    
    export_time = time.time() - start_time
    logger.info(f"Exported to {output_file} in {export_time:.2f}s")
    
    # Generate statistics
    stats = generate_statistics(synthetic_docs, samples)
    
    # Save statistics
    stats_file = output_dir / "generation_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print_summary(stats, output_file)


def generate_statistics(synthetic_docs: List[Dict], samples: List[DocumentSample]) -> Dict:
    """Generate comprehensive statistics about the generated data."""
    
    # Basic counts
    stats = {
        "generation": {
            "total_documents": len(synthetic_docs),
            "total_samples": len(samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Complexity distribution
    complexity_counts = {}
    for doc in synthetic_docs:
        complexity = doc["complexity"]
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    stats["complexity_distribution"] = {
        comp: {"count": count, "percentage": count / len(synthetic_docs) * 100}
        for comp, count in complexity_counts.items()
    }
    
    # Industry distribution
    industry_counts = {}
    for doc in synthetic_docs:
        industry = doc["industry"]
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    
    stats["industry_distribution"] = {
        ind: {"count": count, "percentage": count / len(synthetic_docs) * 100}
        for ind, count in industry_counts.items()
    }
    
    # Brand distribution
    brand_counts = {}
    for doc in synthetic_docs:
        brand = doc["brand"]
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    stats["brand_distribution"] = {
        brand: count for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
    }
    
    # Text statistics
    char_counts = [doc["char_count"] for doc in synthetic_docs]
    word_counts = [doc["word_count"] for doc in synthetic_docs]
    
    stats["text_statistics"] = {
        "characters": {
            "min": min(char_counts),
            "max": max(char_counts),
            "avg": sum(char_counts) / len(char_counts),
            "total": sum(char_counts)
        },
        "words": {
            "min": min(word_counts),
            "max": max(word_counts),
            "avg": sum(word_counts) / len(word_counts),
            "total": sum(word_counts)
        }
    }
    
    # Chunking statistics
    chunk_counts = [len(sample.chunks) for sample in samples]
    chunked_samples = [sample for sample in samples if sample.needs_chunking]
    
    stats["chunking_statistics"] = {
        "total_chunks": sum(chunk_counts),
        "samples_requiring_chunking": len(chunked_samples),
        "chunking_rate": len(chunked_samples) / len(samples) * 100,
        "avg_chunks_per_sample": sum(chunk_counts) / len(samples),
        "chunk_distribution": {
            "1_chunk": sum(1 for c in chunk_counts if c == 1),
            "2_chunks": sum(1 for c in chunk_counts if c == 2),
            "3_plus_chunks": sum(1 for c in chunk_counts if c >= 3)
        }
    }
    
    return stats


def print_summary(stats: Dict, output_file: Path) -> None:
    """Print generation summary."""
    
    print("\n" + "="*60)
    print("🎯 SYNTHETIC PREAMBLE GENERATION COMPLETE")
    print("="*60)
    
    gen_stats = stats["generation"]
    print(f"📊 Generated: {gen_stats['total_documents']} documents → {gen_stats['total_samples']} samples")
    print(f"📅 Timestamp: {gen_stats['timestamp']}")
    print(f"💾 Output: {output_file}")
    
    print(f"\n📈 COMPLEXITY DISTRIBUTION:")
    for comp, data in stats["complexity_distribution"].items():
        print(f"   {comp.capitalize()}: {data['count']} ({data['percentage']:.1f}%)")
    
    print(f"\n🏭 INDUSTRY DISTRIBUTION:")
    for ind, data in stats["industry_distribution"].items():
        print(f"   {ind.capitalize()}: {data['count']} ({data['percentage']:.1f}%)")
    
    text_stats = stats["text_statistics"]
    print(f"\n📝 TEXT STATISTICS:")
    print(f"   Characters: {text_stats['characters']['min']}-{text_stats['characters']['max']} "
          f"(avg: {text_stats['characters']['avg']:.0f})")
    print(f"   Words: {text_stats['words']['min']}-{text_stats['words']['max']} "
          f"(avg: {text_stats['words']['avg']:.0f})")
    
    chunk_stats = stats["chunking_statistics"]
    print(f"\n🔄 CHUNKING STATISTICS:")
    print(f"   Total chunks: {chunk_stats['total_chunks']}")
    print(f"   Chunking rate: {chunk_stats['chunking_rate']:.1f}%")
    print(f"   Avg chunks/sample: {chunk_stats['avg_chunks_per_sample']:.1f}")
    
    chunk_dist = chunk_stats["chunk_distribution"]
    print(f"   Distribution: 1 chunk: {chunk_dist['1_chunk']}, "
          f"2 chunks: {chunk_dist['2_chunks']}, "
          f"3+ chunks: {chunk_dist['3_plus_chunks']}")
    
    print(f"\n✅ Generation complete! Use the output for training business document understanding.")


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic business preambles for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Basic options
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=100,
        help="Number of preambles to generate (default: 100)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="synthetic_preambles",
        help="Output directory (default: synthetic_preambles)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="preambles.jsonl",
        help="Output JSONL filename (default: preambles.jsonl)"
    )
    
    # Industry selection
    available_industries = ["fashion", "food", "tech", "home", "beauty", "travel"]
    parser.add_argument(
        "--industries",
        nargs="+",
        choices=available_industries,
        help=f"Industries to include (default: all). Choices: {available_industries}"
    )
    
    # Complexity distribution
    parser.add_argument(
        "--simple",
        type=float,
        default=0.3,
        help="Proportion of simple preambles (default: 0.3)"
    )
    
    parser.add_argument(
        "--medium",
        type=float,
        default=0.5,
        help="Proportion of medium preambles (default: 0.5)"
    )
    
    parser.add_argument(
        "--complex",
        type=float,
        default=0.2,
        help="Proportion of complex preambles (default: 0.2)"
    )
    
    # Processing options
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Apply text cleaning to generated preambles"
    )
    
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Apply chunking to generated preambles"
    )
    
    # Output options
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw synthetic documents (before processing)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        generate_synthetic_preambles(args)
    except Exception as e:
        logging.error(f"Generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 