#!/usr/bin/env python3
"""
Example usage of tb_dataset package

This script demonstrates how to use the document ingestion and chunking pipeline.
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path for local import
sys.path.insert(0, str(Path(__file__).parent.parent))

from tb_dataset import DocumentIngester, TextCleaner, SmartChunker
from tb_dataset.schema import DocumentSample


def create_sample_document():
    """Create a sample text document for demonstration."""
    content = """
CONFIDENTIAL AGREEMENT

This Agreement is entered into on January 15, 2024, between Company A and Company B.

TERMS AND CONDITIONS

1. Payment Terms
   - Total amount: $125,000.00
   - Due date: March 1, 2024
   - Late fee: $500.00 per day

2. Contact Information
   - Primary contact: John Smith (555) 123-4567
   - Email: john.smith@company-a.com
   - Secondary contact: Jane Doe (555) 987-6543

3. Deliverables
   This agreement covers the following deliverables: software development, 
   testing, documentation, and training materials. The work must be completed
   according to the specifications outlined in Appendix A.

CONFIDENTIALITY CLAUSE

Both parties agree to maintain strict confidentiality regarding all information
shared during the course of this agreement. This includes technical specifications,
business strategies, and financial information.

Signed on this day, January 15, 2024.
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content.strip())
        return Path(f.name)


def main():
    """Demonstrate the tb_dataset pipeline."""
    print("TB_DATASET EXAMPLE")
    print("=" * 50)
    
    # Create sample document
    sample_file = create_sample_document()
    print(f"Created sample document: {sample_file}")
    
    try:
        # Initialize components
        print("\n1. Initializing components...")
        ingester = DocumentIngester()
        cleaner = TextCleaner(redact_emails=True, redact_phones=True, redact_ssns=False)
        chunker = SmartChunker(max_tokens=100)  # Small chunks for demo
        
        # Step 1: Ingest document
        print("\n2. Ingesting document...")
        doc_data = ingester.ingest_file(sample_file)
        print(f"   Document ID: {doc_data['doc_id']}")
        print(f"   Original length: {len(doc_data['text_content'])} characters")
        print(f"   Sections found: {list(doc_data['sections'].keys())}")
        
        # Step 2: Clean text
        print("\n3. Cleaning text...")
        original_text = doc_data['sections']['full']
        cleaned_text = cleaner.clean_text(original_text)
        print(f"   Cleaned length: {len(cleaned_text)} characters")
        
        # Show cleaning effects
        if '[EMAIL]' in cleaned_text:
            print("   ✓ Email addresses redacted")
        if '[PHONE]' in cleaned_text:
            print("   ✓ Phone numbers redacted")
        
        # Step 3: Chunk text
        print("\n4. Chunking text...")
        token_count = chunker.count_tokens(cleaned_text)
        needs_chunking = chunker.needs_chunking(cleaned_text)
        print(f"   Total tokens: {token_count}")
        print(f"   Needs chunking: {needs_chunking}")
        
        if needs_chunking:
            chunks = chunker.chunk_text(cleaned_text)
            print(f"   Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                chunk_tokens = chunker.count_tokens(chunk)
                print(f"   Chunk {i+1}: {chunk_tokens} tokens")
        else:
            chunks = [cleaned_text]
            print("   No chunking needed")
        
        # Step 4: Create sample
        print("\n5. Creating DocumentSample...")
        sample = DocumentSample(
            source_path=doc_data['source_path'],
            doc_id=doc_data['doc_id'],
            section="full",
            text=cleaned_text,
            char_len=len(cleaned_text),
            tok_len=token_count,
            chunks=chunks,
            needs_chunking=needs_chunking
        )
        
        # Validate sample
        sample.validate()
        print("   ✓ Sample validation passed")
        
        # Step 5: Show output
        print("\n6. Sample output (first 200 characters):")
        json_output = sample.to_json()
        print(json_output[:200] + "...")
        
        # Statistics
        print("\n7. Statistics:")
        ing_stats = ingester.get_stats()
        clean_stats = cleaner.get_stats()
        chunk_stats = chunker.get_stats()
        
        print(f"   Ingester: {ing_stats['processed']} docs processed")
        print(f"   Cleaner: {clean_stats['emails_redacted']} emails, {clean_stats['phones_redacted']} phones redacted")
        print(f"   Chunker: {chunk_stats['chunks_created']} chunks, avg {chunk_stats['avg_tokens_per_chunk']:.1f} tokens")
        
    finally:
        # Clean up
        sample_file.unlink()
        print(f"\nCleaned up temporary file: {sample_file}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nTo use the CLI:")
    print("tb-ingest --in /path/to/documents --out /path/to/output --clean --chunk --stats")


if __name__ == "__main__":
    main() 