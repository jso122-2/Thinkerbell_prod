#!/usr/bin/env python3
"""
Example: Synthetic Business Preamble Generation with tb_dataset

Demonstrates how to use the SyntheticPreambleGenerator to create
training data for business document understanding.
"""

from pathlib import Path
from tb_dataset import SyntheticPreambleGenerator, TextCleaner, SmartChunker

def main():
    """Example usage of synthetic preamble generation"""
    
    print("🎯 Example: Synthetic Business Preamble Generation")
    print("=" * 60)
    
    # Initialize the generator
    generator = SyntheticPreambleGenerator()
    
    # Example 1: Generate a single preamble
    print("\n📝 Example 1: Single Preamble Generation")
    
    preamble = generator.generate_preamble(
        brand="Koala",
        industry="home", 
        complexity="medium"
    )
    
    print(f"Generated preamble:")
    print(f"  {preamble}")
    print(f"  Length: {len(preamble)} chars, {len(preamble.split())} words")
    
    # Example 2: Generate batch of synthetic documents
    print(f"\n📊 Example 2: Batch Generation")
    
    synthetic_docs = generator.generate_synthetic_documents(
        count=25,
        complexity_distribution={"simple": 0.2, "medium": 0.6, "complex": 0.2},
        industries=["fashion", "food", "tech"]
    )
    
    print(f"Generated {len(synthetic_docs)} synthetic documents:")
    
    # Show distribution
    complexity_counts = {}
    industry_counts = {}
    
    for doc in synthetic_docs:
        complexity = doc["complexity"]
        industry = doc["industry"]
        
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    
    print(f"  Complexity: {complexity_counts}")
    print(f"  Industries: {industry_counts}")
    
    # Show sample documents
    print(f"\n  Sample documents:")
    for i, doc in enumerate(synthetic_docs[:3]):
        print(f"    {i+1}. {doc['brand']} ({doc['industry']}, {doc['complexity']})")
        print(f"       {doc['text'][:80]}...")
    
    # Example 3: Process through tb_dataset pipeline
    print(f"\n🔄 Example 3: tb_dataset Pipeline Processing")
    
    output_dir = Path("example_output")
    
    # Create DocumentSample objects with cleaning and chunking
    samples = generator.create_document_samples(
        synthetic_docs=synthetic_docs,
        output_dir=output_dir,
        clean=True,   # Apply text cleaning
        chunk=True    # Apply smart chunking
    )
    
    print(f"Created {len(samples)} DocumentSample objects")
    
    # Analyze processing results
    chunked_samples = [s for s in samples if s.needs_chunking]
    total_chunks = sum(len(s.chunks) for s in samples)
    
    print(f"  Samples requiring chunking: {len(chunked_samples)}")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Average chunks per sample: {total_chunks / len(samples):.1f}")
    
    # Example 4: Export to JSONL
    print(f"\n💾 Example 4: Export to JSONL")
    
    output_file = output_dir / "business_preambles.jsonl"
    generator.export_to_jsonl(samples, output_file)
    
    if output_file.exists():
        file_size = output_file.stat().st_size
        print(f"✅ Exported to {output_file}")
        print(f"   File size: {file_size:,} bytes")
        
        # Count lines
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        print(f"   Samples: {line_count}")
    
    # Example 5: Use with existing tb_dataset tools
    print(f"\n🛠️ Example 5: Integration with tb_dataset Tools")
    
    # Example of using tb_dataset components directly
    cleaner = TextCleaner()
    chunker = SmartChunker(max_tokens=400)  # Smaller chunks for this example
    
    # Process a sample preamble
    sample_text = generator.generate_preamble("Bunnings", "home", "complex")
    
    print(f"Original text ({len(sample_text)} chars):")
    print(f"  {sample_text}")
    
    # Clean the text
    cleaned_text = cleaner.clean_text(sample_text)
    print(f"\nCleaned text ({len(cleaned_text)} chars):")
    print(f"  {cleaned_text}")
    
    # Check if chunking is needed
    needs_chunking = chunker.needs_chunking(cleaned_text)
    print(f"\nNeeds chunking: {needs_chunking}")
    
    if needs_chunking:
        chunks = chunker.chunk_text(cleaned_text)
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {chunk[:60]}...")
    else:
        print("Text fits in single chunk")
    
    print(f"\n✅ Example complete!")
    print(f"💡 Key benefits:")
    print(f"   - Creates realistic business context for training")
    print(f"   - Integrates seamlessly with tb_dataset pipeline")
    print(f"   - Generates varied, high-quality synthetic data")
    print(f"   - Supports multiple industries and complexity levels")
    print(f"   - Compatible with existing text processing tools")

if __name__ == "__main__":
    main() 