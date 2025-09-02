#!/usr/bin/env python3
"""
Chunk Generator for Thinkerbell Formatter Project
Reads synthetic dataset and creates semantic chunks for sentence encoder training
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Import existing Thinkerbell utilities
try:
    from thinkerbell.utils.text_preprocessor import TextPreprocessor
    HAS_TEXT_PREPROCESSOR = True
except ImportError:
    HAS_TEXT_PREPROCESSOR = False
    print("⚠️ TextPreprocessor not available - using fallback implementation")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("⚠️ NLTK not available - using regex sentence splitting")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ Transformers not available - using character-based token estimation")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ChunkGenerator:
    """
    Generates semantic chunks from synthetic dataset for sentence encoder training
    """
    
    def __init__(self, 
                 input_dir: str = "synthetic_dataset/samples",
                 output_dir: str = "synthetic_dataset/chunks",
                 max_tokens: int = 480,
                 min_tokens: int = 20,
                 tokenizer_model: str = "sentence-transformers/all-mpnet-base-v2",
                 debug: bool = False):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tokenizer_model = tokenizer_model
        self.debug = debug
        
        # Fixed label set
        self.valid_labels = {
            "client", "brand", "campaign", "fee", "deliverables", 
            "exclusivity_period", "exclusivity_scope", "engagement_term", 
            "usage_term", "territory"
        }
        
        # Initialize tokenizer
        self.tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
                logger.info(f"✅ Loaded tokenizer: {tokenizer_model}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        
        # Initialize text preprocessor if available
        self.text_preprocessor = None
        if HAS_TEXT_PREPROCESSOR:
            try:
                self.text_preprocessor = TextPreprocessor(
                    model_name=tokenizer_model, 
                    max_tokens=max_tokens
                )
                logger.info("✅ Using existing TextPreprocessor")
            except Exception as e:
                logger.warning(f"Failed to initialize TextPreprocessor: {e}")
        
        # Stats tracking
        self.stats = {
            "total_samples_processed": 0,
            "total_chunks_generated": 0,
            "samples_with_errors": 0,
            "empty_samples": 0,
            "label_distribution": {},
            "chunks_per_sample": []
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using available tokenizer or fallback"""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            except:
                pass
        
        # Fallback: estimate tokens as ~4 characters per token
        return len(text) // 4
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK or regex fallback"""
        if HAS_NLTK:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Regex fallback
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into semantic chunks of ~400-480 tokens each
        Uses existing TextPreprocessor if available, otherwise fallback implementation
        """
        if self.text_preprocessor:
            try:
                return self.text_preprocessor.smart_chunk_strategy(text, self.max_tokens)
            except Exception as e:
                logger.warning(f"TextPreprocessor failed, using fallback: {e}")
        
        # Fallback chunking implementation
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            token_count = self.count_tokens(test_chunk)
            
            if token_count <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if self.count_tokens(chunk) >= self.min_tokens]
        
        return chunks
    
    def extract_labels_for_chunk(self, chunk_text: str, extracted_fields: Dict) -> List[str]:
        """
        Extract labels for a chunk based on which extracted fields appear in it
        Returns list of field names that have values appearing in the chunk
        """
        labels = []
        chunk_lower = chunk_text.lower()
        
        for field_name, field_value in extracted_fields.items():
            # Skip if field not in valid labels
            if field_name not in self.valid_labels:
                continue
            
            # Skip if no value
            if not field_value:
                continue
            
            # Handle different value types
            if isinstance(field_value, str):
                values_to_check = [field_value]
            elif isinstance(field_value, list):
                values_to_check = field_value
            else:
                values_to_check = [str(field_value)]
            
            # Check if any value appears in chunk (case-insensitive)
            for value in values_to_check:
                value_str = str(value).lower()
                
                # Direct substring match
                if value_str in chunk_lower:
                    labels.append(field_name)
                    break
                
                # Extract key phrases for more sophisticated matching
                key_phrases = self._extract_key_phrases(value_str)
                for phrase in key_phrases:
                    if phrase in chunk_lower:
                        labels.append(field_name)
                        break
                
                if field_name in labels:
                    break
        
        # Update label stats
        for label in labels:
            self.stats["label_distribution"][label] = self.stats["label_distribution"].get(label, 0) + 1
        
        return sorted(list(set(labels)))  # Remove duplicates and sort
    
    def _extract_key_phrases(self, value: str) -> List[str]:
        """Extract key phrases from field values for matching"""
        phrases = []
        
        # Clean the value
        clean_value = re.sub(r'[^\w\s]', ' ', value)
        words = clean_value.split()
        
        # Add individual words (if meaningful)
        for word in words:
            if len(word) > 2 and not word.isdigit():
                phrases.append(word.lower())
        
        # Add the full cleaned phrase
        if len(words) > 1:
            phrases.append(' '.join(words).lower())
        
        # Special handling for common patterns
        if 'instagram' in value.lower():
            phrases.extend(['instagram', 'ig'])
        if 'facebook' in value.lower():
            phrases.extend(['facebook', 'fb'])
        if '$' in value:
            # Extract numeric part for fee matching
            numbers = re.findall(r'\d+', value)
            phrases.extend(numbers)
        
        return phrases
    
    def process_sample(self, sample_path: Path, batch_dir: Path) -> List[Dict]:
        """Process a single sample file and return list of chunks"""
        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load sample {sample_path}: {e}")
            self.stats["samples_with_errors"] += 1
            return []
        
        # Extract required fields
        sample_id = sample_data.get('sample_id', sample_path.stem)
        raw_input = sample_data.get('raw_input', {})
        text = raw_input.get('text', '')
        extracted_fields = sample_data.get('extracted_fields', {})
        
        if not text:
            logger.warning(f"No text found in sample {sample_id}")
            self.stats["empty_samples"] += 1
            return []
        
        # Create chunks
        chunks = self.create_chunks(text)
        
        # Process each chunk
        chunk_data = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{sample_id}_c{i+1}"
            token_count = self.count_tokens(chunk_text)
            labels = self.extract_labels_for_chunk(chunk_text, extracted_fields)
            
            chunk_info = {
                "sample_id": sample_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": token_count,
                "labels": labels
            }
            
            chunk_data.append(chunk_info)
            
            # Save individual chunk file
            self._save_chunk_file(chunk_info, batch_dir, f"{chunk_id}.json")
        
        # Update stats
        self.stats["total_samples_processed"] += 1
        self.stats["total_chunks_generated"] += len(chunk_data)
        self.stats["chunks_per_sample"].append(len(chunk_data))
        
        # Debug output for first 3 samples
        if self.debug and self.stats["total_samples_processed"] <= 3:
            logger.info(f"\n--- Debug: Sample {sample_id} ---")
            logger.info(f"Text length: {len(text)} chars, {self.count_tokens(text)} tokens")
            logger.info(f"Generated {len(chunk_data)} chunks")
            for i, chunk in enumerate(chunk_data):
                logger.info(f"  Chunk {i+1}: {chunk['token_count']} tokens, labels: {chunk['labels']}")
                if i == 0:  # Show text of first chunk only
                    logger.info(f"  Text preview: {chunk['text'][:100]}...")
        
        return chunk_data
    
    def _save_chunk_file(self, chunk_data: Dict, batch_dir: Path, filename: str):
        """Save a single chunk to a JSON file"""
        chunk_dir = self.output_dir / batch_dir.name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = chunk_dir / filename
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save chunk to {output_file}: {e}")
    
    def process_batch(self, batch_dir: Path) -> None:
        """Process all samples in a batch directory"""
        sample_files = list(batch_dir.glob("sample_*.json"))
        logger.info(f"Processing {len(sample_files)} samples in {batch_dir.name}")
        
        for sample_file in sample_files:
            self.process_sample(sample_file, batch_dir)
    
    def generate_chunks_for_dataset(self):
        """Process entire synthetic dataset and generate chunks"""
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            return
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all batch directories
        batch_dirs = [d for d in self.input_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
        batch_dirs.sort()
        
        if not batch_dirs:
            logger.error(f"No batch directories found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(batch_dirs)} batch directories")
        
        # Process each batch
        for batch_dir in batch_dirs:
            self.process_batch(batch_dir)
        
        # Generate summary
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of chunk generation"""
        total_samples = self.stats["total_samples_processed"]
        total_chunks = self.stats["total_chunks_generated"]
        avg_chunks = sum(self.stats["chunks_per_sample"]) / len(self.stats["chunks_per_sample"]) if self.stats["chunks_per_sample"] else 0
        
        logger.info("\n" + "="*50)
        logger.info("CHUNK GENERATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Total chunks generated: {total_chunks}")
        logger.info(f"Avg chunks per sample: {avg_chunks:.2f}")
        logger.info(f"Samples with errors: {self.stats['samples_with_errors']}")
        logger.info(f"Empty samples: {self.stats['empty_samples']}")
        
        # Label distribution
        if self.stats["label_distribution"]:
            logger.info("\nLabel distribution:")
            for label, count in sorted(self.stats["label_distribution"].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {label}: {count} chunks ({count/total_chunks*100:.1f}%)")
        
        logger.info("\n🎉 Chunk generation complete!")


def main():
    """Main function for running chunk generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate semantic chunks from synthetic dataset")
    parser.add_argument("--input-dir", default="synthetic_dataset/samples", 
                       help="Path to synthetic dataset samples directory")
    parser.add_argument("--output-dir", default="synthetic_dataset/chunks", 
                       help="Output directory for chunk files")
    parser.add_argument("--max-tokens", type=int, default=480, 
                       help="Maximum tokens per chunk")
    parser.add_argument("--min-tokens", type=int, default=20,
                       help="Minimum tokens per chunk (drop smaller chunks)")
    parser.add_argument("--debug", action="store_true",
                       help="Print debug information for first 3 processed samples")
    parser.add_argument("--tokenizer-model", default="sentence-transformers/all-mpnet-base-v2",
                       help="Tokenizer model to use")
    
    args = parser.parse_args()
    
    # Create chunk generator
    generator = ChunkGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        tokenizer_model=args.tokenizer_model,
        debug=args.debug
    )
    
    # Generate chunks
    generator.generate_chunks_for_dataset()


if __name__ == "__main__":
    main() 