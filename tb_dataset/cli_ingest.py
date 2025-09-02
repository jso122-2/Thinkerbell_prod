"""
CLI interface for tb_dataset document ingestion and processing

Usage:
    python -m tb_dataset.cli_ingest --in data/raw --out data/ingested
    tb-ingest --in data/raw --out data/ingested --recursive --clean --chunk
    tb-ingest --help

Examples:
    # Basic ingestion
    tb-ingest --in documents/ --out processed/
    
    # With cleaning and chunking
    tb-ingest --in contracts/ --out jsonl_output/ --clean --chunk
    
    # Process single file
    tb-ingest --in contract.pdf --out output/ --clean --chunk --verbose
    
    # Custom chunking limits
    tb-ingest --in docs/ --out chunks/ --chunk --max-tokens 400 --model-name gpt-4
    
    # PII redaction options
    tb-ingest --in sensitive/ --out cleaned/ --clean --redact-emails --redact-phones
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm function
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .ingest import DocumentIngester
from .clean import TextCleaner
from .chunk import SmartChunker
from .schema import DocumentSample
from .utils import ensure_dir, get_file_info, hash_id


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


def create_document_sample(ingested_doc: Dict, section_key: str, 
                          cleaned_text: str, chunks: List[str], 
                          tok_len: int, needs_chunking: bool) -> DocumentSample:
    """
    Create a DocumentSample from processed data.
    
    Args:
        ingested_doc: Dictionary from DocumentIngester
        section_key: Section name ('full', 'header', 'body', 'footer')
        cleaned_text: Cleaned text content
        chunks: List of text chunks
        tok_len: Token count
        needs_chunking: Whether text needed chunking
        
    Returns:
        DocumentSample instance
    """
    return DocumentSample(
        source_path=ingested_doc['source_path'],
        doc_id=ingested_doc['doc_id'],
        section=section_key,
        text=cleaned_text,
        char_len=len(cleaned_text),
        tok_len=tok_len,
        chunks=chunks,
        needs_chunking=needs_chunking,
        template_hint=None  # Leave null for now as requested
    )


def process_document(file_path: Path, ingester: DocumentIngester, 
                    cleaner: Optional[TextCleaner], 
                    chunker: Optional[SmartChunker],
                    output_dir: Path) -> List[DocumentSample]:
    """
    Process a single document through the full pipeline.
    
    Args:
        file_path: Path to document file
        ingester: Document ingester instance
        cleaner: Text cleaner instance (optional)
        chunker: Smart chunker instance (optional)
        output_dir: Output directory for JSONL files
        
    Returns:
        List of DocumentSample objects created
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Ingest document
        logger.info(f"Ingesting {file_path.name}")
        ingested_doc = ingester.ingest_file(file_path)
        
        samples = []
        
        # Process each section
        for section_key, section_text in ingested_doc['sections'].items():
            if not section_text.strip():
                continue
            
            # Step 2: Clean text (optional)
            if cleaner:
                cleaned_text = cleaner.clean_text(section_text)
                if not cleaner.validate_text(cleaned_text):
                    logger.warning(f"Skipping invalid text in {section_key} section of {file_path.name}")
                    continue
            else:
                cleaned_text = section_text.strip()
            
            # Step 3: Count tokens and chunk if needed
            if chunker:
                tok_len = chunker.count_tokens(cleaned_text)
                needs_chunking = chunker.needs_chunking(cleaned_text)
                
                if needs_chunking:
                    chunks = chunker.chunk_text(cleaned_text)
                    if not chunker.validate_chunks(chunks):
                        logger.warning(f"Chunk validation failed for {section_key} section of {file_path.name}")
                        # Use single chunk as fallback
                        chunks = [cleaned_text]
                        needs_chunking = False
                else:
                    chunks = [cleaned_text]
            else:
                # Simple word-based token estimation
                tok_len = len(cleaned_text.split())
                needs_chunking = False
                chunks = [cleaned_text]
            
            # Create sample
            sample = create_document_sample(
                ingested_doc=ingested_doc,
                section_key=section_key,
                cleaned_text=cleaned_text,
                chunks=chunks,
                tok_len=tok_len,
                needs_chunking=needs_chunking
            )
            
            # Validate sample
            try:
                sample.validate()
                samples.append(sample)
            except ValueError as e:
                logger.error(f"Sample validation failed for {file_path.name} {section_key}: {str(e)}")
                continue
        
        # Step 4: Write JSONL output
        if samples:
            output_file = output_dir / f"{file_path.stem}_{ingested_doc['doc_id']}.jsonl"
            write_jsonl(samples, output_file)
            logger.info(f"Created {output_file} with {len(samples)} samples")
        
        return samples
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return []


def write_jsonl(samples: List[DocumentSample], output_file: Path):
    """
    Write DocumentSample objects to JSONL file.
    
    Args:
        samples: List of DocumentSample objects
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample.to_json() + '\n')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents and create JSONL training samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--in', '--input', dest='input_path', required=True,
        help='Input file or directory path'
    )
    parser.add_argument(
        '--out', '--output', dest='output_path', required=True,
        help='Output directory for JSONL files'
    )
    
    # Processing options
    parser.add_argument(
        '--recursive', '-r', action='store_true',
        help='Recursively search subdirectories (default: False)'
    )
    parser.add_argument(
        '--clean', action='store_true',
        help='Enable text cleaning and normalization'
    )
    parser.add_argument(
        '--chunk', action='store_true',
        help='Enable smart chunking'
    )
    
    # Chunking options
    parser.add_argument(
        '--max-tokens', type=int, default=480,
        help='Maximum tokens per chunk (default: 480)'
    )
    parser.add_argument(
        '--model-name', default='sentence-transformers/all-mpnet-base-v2',
        help='Tokenizer model name for token counting'
    )
    parser.add_argument(
        '--spacy-model', default='en_core_web_sm',
        help='spaCy model for sentence splitting'
    )
    
    # Cleaning options
    parser.add_argument(
        '--redact-emails', action='store_true',
        help='Redact email addresses with [EMAIL] (default when --clean used)'
    )
    parser.add_argument(
        '--redact-phones', action='store_true',
        help='Redact phone numbers with [PHONE] (default when --clean used)'
    )
    parser.add_argument(
        '--redact-ssns', action='store_true',
        help='Redact SSNs with [SSN] (disabled by default)'
    )
    parser.add_argument(
        '--no-redact-emails', action='store_true',
        help='Disable email redaction even when cleaning'
    )
    parser.add_argument(
        '--no-redact-phones', action='store_true',
        help='Disable phone redaction even when cleaning'
    )
    
    # Logging and output options
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--stats', action='store_true',
        help='Print processing statistics at the end'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Process files but do not write output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_path)
    if not args.dry_run:
        ensure_dir(output_path)
        logger.info(f"Output directory: {output_path}")
    
    # Initialize components
    logger.info("Initializing processing components...")
    
    # Document ingester
    ingester = DocumentIngester()
    
    # Text cleaner (optional)
    cleaner = None
    if args.clean:
        # Determine redaction settings
        redact_emails = args.redact_emails and not args.no_redact_emails
        redact_phones = args.redact_phones and not args.no_redact_phones
        redact_ssns = args.redact_ssns
        
        # Default to redacting emails and phones when cleaning unless explicitly disabled
        if not args.no_redact_emails and not args.redact_emails:
            redact_emails = True
        if not args.no_redact_phones and not args.redact_phones:
            redact_phones = True
        
        cleaner = TextCleaner(
            redact_emails=redact_emails,
            redact_phones=redact_phones,
            redact_ssns=redact_ssns
        )
        logger.info(f"Text cleaning enabled (emails={redact_emails}, phones={redact_phones}, ssns={redact_ssns})")
    
    # Smart chunker (optional)
    chunker = None
    if args.chunk:
        chunker = SmartChunker(
            max_tokens=args.max_tokens,
            model_name=args.model_name,
            spacy_model=args.spacy_model
        )
        logger.info(f"Smart chunking enabled (max_tokens={args.max_tokens}, model={args.model_name})")
    
    # Collect files to process
    files_to_process = []
    if input_path.is_file():
        if input_path.suffix.lower() in ingester.SUPPORTED_EXTENSIONS:
            files_to_process = [input_path]
        else:
            logger.error(f"Unsupported file type: {input_path.suffix}")
            sys.exit(1)
    else:
        files_to_process = []
        pattern = "**/*" if args.recursive else "*"
        for file_path in input_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in ingester.SUPPORTED_EXTENSIONS:
                files_to_process.append(file_path)
        
        if not files_to_process:
            logger.error(f"No supported files found in {input_path}")
            sys.exit(1)
    
    logger.info(f"Found {len(files_to_process)} file(s) to process")
    
    # Process files with progress bar
    start_time = time.time()
    all_samples = []
    
    file_progress = tqdm(files_to_process, desc="Processing files", unit="file") if TQDM_AVAILABLE else files_to_process
    
    for file_path in file_progress:
        if args.dry_run:
            logger.info(f"DRY RUN: Would process {file_path}")
            continue
        
        if TQDM_AVAILABLE:
            file_progress.set_description(f"Processing {file_path.name[:30]}")
        
        samples = process_document(
            file_path=file_path,
            ingester=ingester,
            cleaner=cleaner,
            chunker=chunker,
            output_dir=output_path
        )
        all_samples.extend(samples)
        
        if TQDM_AVAILABLE:
            file_progress.set_description(f"Processed {len(all_samples)} samples")
    
    # Processing complete
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"Processing complete: {len(all_samples)} samples from {len(files_to_process)} files in {processing_time:.2f}s")
    
    # Print statistics if requested
    if args.stats:
        print("\n" + "="*50)
        print("PROCESSING STATISTICS")
        print("="*50)
        
        # Ingester stats
        ing_stats = ingester.get_stats()
        print(f"Documents processed: {ing_stats['processed']}")
        print(f"Ingestion errors: {ing_stats['errors']}")
        print(f"Files by type: {ing_stats['by_type']}")
        
        # Cleaner stats
        if cleaner:
            clean_stats = cleaner.get_stats()
            print(f"\nText cleaning:")
            print(f"  Texts processed: {clean_stats['processed']}")
            print(f"  Emails redacted: {clean_stats['emails_redacted']}")
            print(f"  Phones redacted: {clean_stats['phones_redacted']}")
            print(f"  SSNs redacted: {clean_stats['ssns_redacted']}")
            print(f"  Whitespace normalized: {clean_stats['whitespace_normalized']}")
            print(f"  Quotes normalized: {clean_stats['quotes_normalized']}")
            print(f"  Currency normalized: {clean_stats['currency_normalized']}")
        
        # Chunker stats
        if chunker:
            chunk_stats = chunker.get_stats()
            print(f"\nSmart chunking:")
            print(f"  Documents processed: {chunk_stats['documents_processed']}")
            print(f"  Chunks created: {chunk_stats['chunks_created']}")
            print(f"  Sentences split: {chunk_stats['sentences_split']}")
            print(f"  Clauses split: {chunk_stats['clauses_split']}")
            print(f"  Units preserved: {chunk_stats['preserved_units']}")
            print(f"  Avg tokens per chunk: {chunk_stats['avg_tokens_per_chunk']:.1f}")
            print(f"  NLP method: {chunk_stats['nlp_method']}")
        
        # Sample stats
        if all_samples:
            sections_count = {}
            chunks_needing_split = 0
            total_chunks = 0
            
            for sample in all_samples:
                sections_count[sample.section] = sections_count.get(sample.section, 0) + 1
                if sample.needs_chunking:
                    chunks_needing_split += 1
                total_chunks += len(sample.chunks)
            
            print(f"\nSample statistics:")
            print(f"  Total samples: {len(all_samples)}")
            print(f"  Samples by section: {sections_count}")
            print(f"  Samples needing chunking: {chunks_needing_split}")
            print(f"  Total chunks created: {total_chunks}")
            print(f"  Avg chunks per sample: {total_chunks/len(all_samples):.1f}")
        
        print("="*50)


if __name__ == "__main__":
    main() 