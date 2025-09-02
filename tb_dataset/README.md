# tb_dataset: Document Ingestion and Chunking Pipeline

A Python package for ingesting real documents (PDF/DOCX/TXT), normalizing text, smart-chunking to ≤512 tokens, and emitting JSONL samples for training data preparation.

## Features

### Document Ingestion & Processing
- **Multi-format support**: PDF, DOCX, TXT, and Markdown files
- **Smart text cleaning**: Normalization, whitespace cleanup, optional PII redaction
- **Intelligent chunking**: Sentence-aware splitting with clause-level fallback
- **Token-accurate counting**: Uses transformer tokenizers for precise token limits

### Synthetic Data Generation
- **Semantic smoothing**: Coherence validation using sentence transformers
- **Business rule validation**: Realistic fee brackets, deliverable bundles, temporal consistency
- **OOD contamination**: Negative samples and edge cases for model robustness
- **Style transfer**: Formal, casual, and structured writing styles
- **Industry-specific generation**: Fashion, food, tech, beauty, home, and B2B content

### System Design
- **Flexible pipeline**: Modular components can be used independently
- **Robust error handling**: Graceful fallbacks when dependencies are missing
- **Comprehensive CLI**: Easy-to-use command-line interfaces for all workflows

## Installation

```bash
# Basic installation
pip install tb-dataset

# With all dependencies
pip install tb-dataset[all]

# Development installation
pip install tb-dataset[dev]
```

## Quick Start

### Command Line Interface

#### Document Ingestion
```bash
# Basic usage
tb-ingest --in documents/ --out processed/

# With cleaning and chunking
tb-ingest --in contracts/ --out jsonl_output/ --clean --chunk

# Process single file with verbose output
tb-ingest --in contract.pdf --out output/ --clean --chunk --verbose --stats

# Custom chunking parameters
tb-ingest --in docs/ --out chunks/ --chunk --max-tokens 400

# PII redaction options
tb-ingest --in sensitive/ --out cleaned/ --clean --redact-emails --redact-phones
```

#### Synthetic Data Generation
```bash
# Generate 5000 samples with 20% OOD contamination
tb-generate --in data/ingested --out data/synth --n 5000 --ood 0.2

# Quick test generation
tb-generate --in ingested_jsonl/ --out test_output/ --n 100 --verbose

# Large batch with custom settings
tb-generate --in processed/ --out training_data/ --n 10000 --ood 0.15 --batch-size 50

# With quality report and custom parameters
tb-generate --in data/ingested --out synthetic/ --n 2000 --ood 0.2 --report --max-tokens 400
```

### Python API

#### Document Processing Pipeline
```python
from tb_dataset import DocumentIngester, TextCleaner, SmartChunker
from tb_dataset.schema import DocumentSample

# Initialize components
ingester = DocumentIngester()
cleaner = TextCleaner(redact_emails=True, redact_phones=True)
chunker = SmartChunker(max_tokens=480)

# Process a document
doc_data = ingester.ingest_file("contract.pdf")
cleaned_text = cleaner.clean_text(doc_data['text_content'])
chunks = chunker.chunk_text(cleaned_text)

# Create training sample
sample = DocumentSample(
    source_path=doc_data['source_path'],
    doc_id=doc_data['doc_id'],
    section="full",
    text=cleaned_text,
    char_len=len(cleaned_text),
    tok_len=chunker.count_tokens(cleaned_text),
    chunks=chunks,
    needs_chunking=len(chunks) > 1
)

# Export to JSON
print(sample.to_json())
```

#### Synthetic Data Generation
```python
from tb_dataset import SyntheticGenerator, BusinessRuleEngine, OODGenerator
from pathlib import Path

# Initialize generator with custom settings
generator = SyntheticGenerator(
    max_tokens=512,
    ood_ratio=0.2,
    coherence_threshold=0.1
)

# Load ingested data
ingested_samples = generator.load_ingested_data(Path("data/ingested"))

# Generate individual samples
synthetic_sample = generator.generate_sample("sample_001")
if synthetic_sample:
    print(f"Generated {synthetic_sample.sample_id}")
    print(f"Type: {synthetic_sample.classification['document_type']}")
    print(f"Industry: {synthetic_sample.classification['industry']}")
    print(f"Fee: {synthetic_sample.extracted_fields.get('fee', 'N/A')}")
    print(synthetic_sample.to_json())

# Business rules engine for custom generation
business_engine = BusinessRuleEngine()
agreement_content = business_engine.generate_agreement_content()
validation = business_engine.validate_agreement(agreement_content)

# OOD generator for negative samples
ood_generator = OODGenerator()
negative_sample = ood_generator.generate_ood_sample()
```

## Output Formats

### Ingested Document Format
Each processed document generates a JSONL file with records like:

```json
{
  "source_path": "/path/to/document.pdf",
  "doc_id": "a1b2c3d4e5f6g7h8",
  "section": "full",
  "text": "Cleaned and normalized text content...",
  "char_len": 1234,
  "tok_len": 487,
  "chunks": ["First chunk...", "Second chunk..."],
  "needs_chunking": true,
  "template_hint": null
}
```

### Synthetic Sample Format
Generated synthetic samples follow this structure:

```json
{
  "sample_id": "sample_000123",
  "generator_version": "v1",
  "raw_input": {
    "text": "INFLUENCER COLLABORATION AGREEMENT...",
    "token_count": 372,
    "style": "formal"
  },
  "classification": {
    "document_type": "INFLUENCER_AGREEMENT",
    "complexity": "medium",
    "industry": "fashion"
  },
  "extracted_fields": {
    "client": "Jordan Smith",
    "brand": "Fashion Nova",
    "campaign": "Summer Collection",
    "fee": "$8,500",
    "fee_numeric": 8500,
    "deliverables": ["2 Instagram posts", "4 Instagram stories", "1 reel"],
    "exclusivity_period": "6 weeks",
    "exclusivity_scope": ["competing fashion brands"],
    "engagement_term": "8 weeks",
    "usage_term": "12 months",
    "territory": "Australia"
  },
  "validation": {
    "semantic_coherence": 0.85,
    "business_ok": true,
    "temporal_ok": true,
    "dedup_hash": "a1b2c3d4e5f6g7h8"
  },
  "template_hint": null,
  "is_ood": false
}
```

## Dependencies

### Required
- `transformers>=4.20.0` - For accurate token counting
- `regex>=2022.0.0` - Enhanced regex support

### Optional
- `PyMuPDF>=1.20.0` - PDF parsing (enables .pdf support)
- `python-docx>=0.8.11` - DOCX parsing (enables .docx support)
- `spacy>=3.4.0` - Advanced sentence splitting
- `nltk>=3.7` - Fallback sentence splitting

## Configuration

### Text Cleaning Options
- Email redaction: `[EMAIL]`
- Phone redaction: `[PHONE]`
- SSN redaction: `[SSN]` (disabled by default)
- Smart quote normalization
- Currency format standardization
- Whitespace normalization

### Chunking Options
- Maximum tokens per chunk (default: 480)
- Sentence-aware splitting
- Clause-level fallback for long sentences
- Preservation of numbers, dates, and currency
- Multiple tokenizer models supported

## Development

```bash
# Clone and install in development mode
git clone https://github.com/dawn-system/tb-dataset
cd tb-dataset
pip install -e .[dev]

# Run tests
pytest

# Format code
black tb_dataset/
isort tb_dataset/

# Type checking
mypy tb_dataset/
```

## Requirements

- Python 3.10+
- Basic dependencies work without external libraries
- Optional dependencies add PDF/DOCX support and better NLP

## License

MIT License. See LICENSE file for details. 