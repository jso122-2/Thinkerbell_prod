#!/usr/bin/env python3
"""
Example usage of tb_dataset synthetic generation pipeline

This script demonstrates how to use the semantic smoother, business rules,
and OOD contamination to generate coherent synthetic training samples.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add tb_dataset to Python path for local import
sys.path.insert(0, str(Path(__file__).parent.parent))

from tb_dataset import SyntheticGenerator, DocumentSample


def create_sample_ingested_data(temp_dir: Path):
    """Create sample ingested JSONL data for demonstration."""
    
    # Sample documents that would come from the ingestion pipeline
    sample_docs = [
        {
            "source_path": "/data/contract1.pdf",
            "doc_id": "abc123def456",
            "section": "full",
            "text": "INFLUENCER COLLABORATION AGREEMENT This agreement is between FashionBrand and the content creator for Instagram campaign featuring summer collection with posts and stories showcasing outfits and styling tips.",
            "char_len": 200,
            "tok_len": 45,
            "chunks": ["INFLUENCER COLLABORATION AGREEMENT This agreement is between FashionBrand and the content creator for Instagram campaign featuring summer collection with posts and stories showcasing outfits and styling tips."],
            "needs_chunking": False,
            "template_hint": None
        },
        {
            "source_path": "/data/agreement2.docx",
            "doc_id": "def456ghi789",
            "section": "full", 
            "text": "Brand Partnership for Food Campaign We are excited to collaborate with you on our healthy meal delivery service promotion including recipe videos, cooking tutorials, and Instagram story content featuring our fresh ingredients.",
            "char_len": 195,
            "tok_len": 42,
            "chunks": ["Brand Partnership for Food Campaign We are excited to collaborate with you on our healthy meal delivery service promotion including recipe videos, cooking tutorials, and Instagram story content featuring our fresh ingredients."],
            "needs_chunking": False,
            "template_hint": None
        }
    ]
    
    # Write JSONL files
    for i, doc in enumerate(sample_docs):
        jsonl_file = temp_dir / f"sample_doc_{i+1}.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"Created {len(sample_docs)} sample ingested documents in {temp_dir}")
    return len(sample_docs)


def demonstrate_business_rules():
    """Demonstrate business rules engine capabilities."""
    print("\n" + "="*50)
    print("BUSINESS RULES ENGINE DEMONSTRATION")
    print("="*50)
    
    from tb_dataset.business_rules import BusinessRuleEngine, DocumentClassifier
    
    # Initialize business rules engine
    engine = BusinessRuleEngine()
    
    # Generate agreement content
    print("Generating realistic influencer agreement...")
    content = engine.generate_agreement_content()
    
    if content:
        print(f"\nGenerated Agreement:")
        print(f"Industry: {content['industry']}")
        print(f"Complexity: {content['complexity']}")
        print(f"Fee: {content['fields']['fee']}")
        print(f"Brand: {content['fields']['brand']}")
        print(f"Campaign: {content['fields']['campaign']}")
        print(f"Deliverables: {content['fields']['deliverables']}")
        
        print(f"\nAgreement Text (first 200 chars):")
        print(content['text'][:200] + "...")
        
        # Validate agreement
        validation = engine.validate_agreement(content)
        print(f"\nValidation Results:")
        print(f"Business Rules OK: {validation['business_ok']}")
        print(f"Temporal Consistency OK: {validation['temporal_ok']}")
        
        # Classify document
        classifier = DocumentClassifier()
        classification = classifier.classify_document(content['text'])
        print(f"\nDocument Classification:")
        print(f"Type: {classification['document_type']}")
        print(f"Complexity: {classification['complexity']}")
        print(f"Industry: {classification['industry']}")


def demonstrate_ood_generation():
    """Demonstrate OOD sample generation."""
    print("\n" + "="*50)
    print("OOD CONTAMINATION DEMONSTRATION")
    print("="*50)
    
    from tb_dataset.ood import OODGenerator
    
    generator = OODGenerator()
    
    # Generate different types of OOD samples
    print("Generating negative sample (non-influencer document)...")
    negative_sample = generator._generate_negative_sample()
    if negative_sample:
        print(f"Type: {negative_sample['ood_type']}")
        print(f"Classification: {negative_sample['classification']['document_type']}")
        print(f"Text preview: {negative_sample['text'][:150]}...")
    
    print("\nGenerating edge case sample (minimal info influencer)...")
    edge_sample = generator._generate_edge_case_sample()
    if edge_sample:
        print(f"Type: {edge_sample['ood_type']}")
        print(f"Classification: {edge_sample['classification']['document_type']}")
        print(f"Text preview: {edge_sample['text'][:150]}...")


def demonstrate_semantic_smoothing():
    """Demonstrate semantic coherence checking."""
    print("\n" + "="*50)
    print("SEMANTIC SMOOTHING DEMONSTRATION")
    print("="*50)
    
    from tb_dataset.generate import SemanticSmoother
    
    smoother = SemanticSmoother()
    
    # Test coherent vs incoherent scenarios
    test_scenarios = [
        ("Coherent", "Nike athletic wear collaboration featuring workout videos and fitness content for Instagram"),
        ("Incoherent", "Banking app legal terms with pet makeup tutorial for construction equipment"),
        ("Mixed", "Fashion brand partnership with cooking videos and financial advice content"),
    ]
    
    print("Testing semantic coherence scores:")
    for label, text in test_scenarios:
        score = smoother.calculate_coherence_score(text)
        is_coherent = smoother.is_coherent(text, threshold=0.1)
        print(f"{label:>10}: {score:.3f} ({'✓' if is_coherent else '✗'})")


def demonstrate_full_generation():
    """Demonstrate the complete generation pipeline."""
    print("\n" + "="*50)
    print("FULL GENERATION PIPELINE DEMONSTRATION")
    print("="*50)
    
    # Create temporary directory with sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample ingested data
        print("Setting up sample ingested data...")
        num_docs = create_sample_ingested_data(temp_path)
        
        # Initialize generator
        print("\nInitializing synthetic generator...")
        generator = SyntheticGenerator(
            max_tokens=300,  # Shorter for demo
            ood_ratio=0.3,   # 30% OOD samples
            coherence_threshold=0.1
        )
        
        # Load ingested data
        print("Loading ingested data...")
        ingested_samples = generator.load_ingested_data(temp_path)
        print(f"Loaded {len(ingested_samples)} ingested samples")
        
        # Generate synthetic samples
        print("\nGenerating synthetic samples...")
        samples = []
        
        for i in range(5):  # Generate 5 samples for demo
            sample_id = f"demo_sample_{i+1:03d}"
            synthetic_sample = generator.generate_sample(sample_id)
            
            if synthetic_sample:
                samples.append(synthetic_sample)
                print(f"Generated {sample_id}:")
                print(f"  Type: {synthetic_sample.classification['document_type']}")
                print(f"  Industry: {synthetic_sample.classification['industry']}")
                print(f"  OOD: {synthetic_sample.is_ood}")
                print(f"  Tokens: {synthetic_sample.raw_input['token_count']}")
                print(f"  Coherence: {synthetic_sample.validation['semantic_coherence']:.3f}")
                
                if synthetic_sample.is_ood:
                    print(f"  OOD Type: {synthetic_sample.extracted_fields.get('ood_type', 'unknown')}")
                else:
                    print(f"  Fee: {synthetic_sample.extracted_fields.get('fee', 'N/A')}")
            else:
                print(f"Failed to generate {sample_id}")
        
        print(f"\nSuccessfully generated {len(samples)} samples")
        
        # Show generation statistics
        stats = generator.get_stats()
        print(f"\nGeneration Statistics:")
        print(f"  Total generated: {stats['samples_generated']}")
        print(f"  OOD samples: {stats['ood_samples']}")
        print(f"  Coherence failures: {stats['coherence_failures']}")
        print(f"  Business rule failures: {stats['business_rule_failures']}")
        
        # Show one complete sample
        if samples:
            print(f"\n" + "-"*40)
            print("SAMPLE OUTPUT (first 300 characters):")
            print("-"*40)
            sample_json = samples[0].to_json()
            print(sample_json[:300] + "...")


def main():
    """Run the complete demonstration."""
    print("TB_DATASET SYNTHETIC GENERATION DEMONSTRATION")
    print("=" * 60)
    
    try:
        demonstrate_business_rules()
        demonstrate_ood_generation()
        demonstrate_semantic_smoothing()
        demonstrate_full_generation()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nTo use the generation CLI:")
        print("tb-generate --in data/ingested --out data/synth --n 5000 --ood 0.2")
        print("\nOr use the Python API:")
        print("from tb_dataset import SyntheticGenerator")
        print("generator = SyntheticGenerator()")
        print("sample = generator.generate_sample('sample_001')")
        
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 