"""
normalize_docs_to_md.py

Converts .pdf and .docx influencer agreements into clean, section-tagged markdown.
Prepares tb_dataset-compatible chunks for preamble generator.
"""

import os
import sys
from pathlib import Path
import re
import json
from datetime import datetime
from typing import Dict, List, Optional

try:
    import mammoth  # for .docx
except ImportError:
    print("⚠️  mammoth not installed. Run: pip install mammoth")
    mammoth = None

try:
    import pdfplumber  # for .pdf
except ImportError:
    print("⚠️  pdfplumber not installed. Run: pip install pdfplumber")
    pdfplumber = None

# Define input/output dirs
DATA_DIR = Path("thinkerbell/data")
OUTPUT_DIR = Path("thinkerbell/normalized_markdown")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = [".docx", ".pdf"]

def clean_text(text: str) -> str:
    """Basic normalization + spacing fixes."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n+', '\n', text)  # collapse multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)  # collapse spaces/tabs
    text = re.sub(r'\r', '', text)  # remove carriage returns
    
    # Clean up common formatting artifacts
    text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)  # join broken words
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # normalize paragraph breaks
    
    return text.strip()

def extract_docx(path: Path) -> str:
    """Extract text from DOCX file."""
    if mammoth is None:
        raise ImportError("mammoth required for DOCX processing")
    
    try:
        with open(path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return clean_text(result.value)
    except Exception as e:
        print(f"❌ Error extracting DOCX {path.name}: {e}")
        return ""

def extract_pdf(path: Path) -> str:
    """Extract text from PDF file."""
    if pdfplumber is None:
        raise ImportError("pdfplumber required for PDF processing")
    
    try:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    print(f"⚠️  No text extracted from page {page_num} of {path.name}")
        
        return clean_text("\n".join(text_parts))
    except Exception as e:
        print(f"❌ Error extracting PDF {path.name}: {e}")
        return ""

def tag_sections(text: str, filename: str) -> str:
    """
    Apply markdown headings to main sections:
    - Preamble
    - Commercial Terms
    - Clauses
    - Signatures
    """
    if not text:
        return text
    
    tagged = text
    
    # Add document header
    doc_title = filename.replace('_', ' ').replace('.md', '')
    tagged = f"# {doc_title}\n\n{tagged}"
    
    # Tag preamble - look for common agreement openings
    preamble_patterns = [
        r'(?i)(THIS AGREEMENT)',
        r'(?i)(THIS INFLUENCER AGREEMENT)',
        r'(?i)(THIS TALENT AGREEMENT)',
        r'(?i)(INFLUENCER AGREEMENT)',
        r'(?i)(TALENT AGREEMENT)',
        r'(?i)(Agreement made)',
        r'(?i)(This Agreement is made)'
    ]
    
    for pattern in preamble_patterns:
        if re.search(pattern, tagged):
            tagged = re.sub(pattern, r"## PREAMBLE\n\n\1", tagged, count=1)
            break
    
    # Tag known section triggers
    section_map = {
        r'(?i)\b(COMMERCIAL TERMS?)\b': r'## COMMERCIAL TERMS\n\n\1',
        r'(?i)\b(SPECIAL CONDITIONS?)\b': r'## SPECIAL CONDITIONS\n\n\1',
        r'(?i)\b(ADDITIONAL TERMS?)\b': r'## ADDITIONAL TERMS\n\n\1',
        r'(?i)\b(PAYMENT TERMS?)\b': r'## PAYMENT TERMS\n\n\1',
        r'(?i)\b(DELIVERABLES?)\b': r'## DELIVERABLES\n\n\1',
        r'(?i)\b(CONTENT REQUIREMENTS?)\b': r'## CONTENT REQUIREMENTS\n\n\1',
        r'(?i)\b(USAGE RIGHTS?)\b': r'## USAGE RIGHTS\n\n\1',
        r'(?i)\b(EXCLUSIVITY)\b': r'## EXCLUSIVITY\n\n\1',
        r'(?i)\b(CONFIDENTIALITY)\b': r'## CONFIDENTIALITY\n\n\1',
        r'(?i)\b(TERMINATION)\b': r'## TERMINATION\n\n\1',
        r'(?i)\b(SIGNATURES?)\b': r'## SIGNATURES\n\n\1',
        r'(?i)\b(EXECUTION)\b': r'## EXECUTION\n\n\1'
    }
    
    for pattern, replacement in section_map.items():
        tagged = re.sub(pattern, replacement, tagged)
    
    # Clean up multiple section headers
    tagged = re.sub(r'(## [^\n]+)\n\n(## [^\n]+)', r'\1\n\2', tagged)
    
    return tagged

def extract_metadata(text: str, filename: str) -> Dict:
    """Extract metadata from the document."""
    metadata = {
        "filename": filename,
        "extraction_date": datetime.now().isoformat(),
        "word_count": len(text.split()) if text else 0,
        "char_count": len(text) if text else 0,
        "sections_found": []
    }
    
    # Find sections
    section_headers = re.findall(r'## ([^\n]+)', text)
    metadata["sections_found"] = section_headers
    
    # Try to identify brand/talent from filename
    if "AMAZON" in filename.upper():
        metadata["brand"] = "Amazon"
    elif "DOVE" in filename.upper():
        metadata["brand"] = "Dove"
    elif "GOLDEN GAYTIME" in filename.upper():
        metadata["brand"] = "Golden Gaytime"
    elif "KOALA" in filename.upper():
        metadata["brand"] = "Koala"
    elif "MATTEL" in filename.upper():
        metadata["brand"] = "Mattel"
    elif "QUEEN" in filename.upper():
        metadata["brand"] = "Queen"
    elif "REXONA" in filename.upper():
        metadata["brand"] = "Rexona"
    
    # Detect document type
    if any(term in text.upper() for term in ["INFLUENCER", "TALENT", "AGREEMENT"]):
        metadata["document_type"] = "INFLUENCER_AGREEMENT"
    else:
        metadata["document_type"] = "UNKNOWN"
    
    return metadata

def convert_file(path: Path) -> bool:
    """Convert a single file to markdown."""
    try:
        print(f"🔄 Processing: {path.name}")
        
        # Extract text based on file type
        if path.suffix.lower() == ".docx":
            text = extract_docx(path)
        elif path.suffix.lower() == ".pdf":
            text = extract_pdf(path)
        else:
            print(f"❌ Unsupported file type: {path.suffix}")
            return False
        
        if not text:
            print(f"❌ No text extracted from {path.name}")
            return False
        
        # Generate clean filename for output
        clean_name = re.sub(r'[^\w\s-]', '', path.stem)
        clean_name = re.sub(r'[-\s]+', '_', clean_name)
        
        # Tag sections
        tagged_text = tag_sections(text, clean_name)
        
        # Extract metadata
        metadata = extract_metadata(tagged_text, path.name)
        
        # Save markdown file
        out_path = OUTPUT_DIR / f"{clean_name}.md"
        out_path.write_text(tagged_text, encoding="utf-8")
        
        # Save metadata file
        metadata_path = OUTPUT_DIR / f"{clean_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Converted: {path.name} → {out_path.name}")
        print(f"   Words: {metadata['word_count']}, Sections: {len(metadata['sections_found'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting {path.name}: {e}")
        return False

def main():
    """Convert all supported documents in the data directory."""
    print("📄 Thinkerbell Document Normalizer")
    print("=" * 50)
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return
    
    # Check dependencies
    missing_deps = []
    if mammoth is None:
        missing_deps.append("mammoth")
    if pdfplumber is None:
        missing_deps.append("pdfplumber")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return
    
    # Find all supported files
    supported_files = []
    for ext in SUPPORTED_EXTS:
        supported_files.extend(DATA_DIR.glob(f"*{ext}"))
    
    if not supported_files:
        print(f"❌ No supported files found in {DATA_DIR}")
        print(f"Looking for: {', '.join(SUPPORTED_EXTS)}")
        return
    
    print(f"Found {len(supported_files)} files to process:")
    for f in supported_files:
        print(f"  - {f.name}")
    print()
    
    # Convert files
    successful = 0
    failed = 0
    
    for file_path in supported_files:
        if convert_file(file_path):
            successful += 1
        else:
            failed += 1
    
    print()
    print("📊 Conversion Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Output directory: {OUTPUT_DIR}")
    
    if successful > 0:
        print("\n🎯 Next steps:")
        print("1. Review the normalized markdown files")
        print("2. Update preamble_generator.py to use these real examples")
        print("3. Extract preamble sections for training data")

if __name__ == "__main__":
    main() 