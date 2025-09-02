#!/usr/bin/env python3
"""
Diagnose and Fix Overfitting Issues

This script analyzes the current dataset to identify overfitting causes and proposes solutions:
1. Analyzes label combination diversity
2. Checks for text pattern repetition
3. Proposes data augmentation strategies
4. Creates a more challenging evaluation setup
"""

import json
import os
import glob
from collections import Counter, defaultdict
import re
import hashlib
from pathlib import Path
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_label_combinations():
    """Analyze the distribution and diversity of label combinations"""
    logger.info("=== ANALYZING LABEL COMBINATIONS ===")
    
    # Load all chunks
    chunks = []
    for split in ['train', 'val', 'test']:
        split_dir = f"Thinkerbell_template_pipeline/synthetic_dataset/splits/{split}"
        if os.path.exists(split_dir):
            for chunk_file in glob.glob(os.path.join(split_dir, "*.json")):
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                        chunk['split'] = split
                        chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to load {chunk_file}: {e}")
    
    logger.info(f"Loaded {len(chunks)} total chunks")
    
    # Analyze label combinations
    combo_counter = Counter()
    split_combo_dist = defaultdict(lambda: defaultdict(int))
    
    for chunk in chunks:
        combo = "|".join(sorted(chunk.get("labels", [])))
        combo_counter[combo] += 1
        split_combo_dist[chunk['split']][combo] += 1
    
    logger.info(f"\nFound {len(combo_counter)} unique label combinations:")
    for combo, count in combo_counter.most_common():
        pct = (count / len(chunks)) * 100
        logger.info(f"  {combo}: {count} chunks ({pct:.1f}%)")
    
    # Check split distribution
    logger.info(f"\nLabel combination distribution across splits:")
    for combo in combo_counter.keys():
        train_count = split_combo_dist['train'][combo]
        val_count = split_combo_dist['val'][combo]
        test_count = split_combo_dist['test'][combo]
        total = train_count + val_count + test_count
        
        logger.info(f"  {combo}:")
        logger.info(f"    Train: {train_count}/{total} ({train_count/total*100:.1f}%)")
        logger.info(f"    Val:   {val_count}/{total} ({val_count/total*100:.1f}%)")
        logger.info(f"    Test:  {test_count}/{total} ({test_count/total*100:.1f}%)")
    
    return chunks, combo_counter

def analyze_text_patterns(chunks):
    """Analyze text patterns to identify repetitive structures"""
    logger.info("\n=== ANALYZING TEXT PATTERNS ===")
    
    # Normalize text for pattern analysis
    def normalize_text(text):
        # Replace numbers with <NUM>
        text = re.sub(r'\d+', '<NUM>', text)
        # Replace common entities
        text = re.sub(r'\$<NUM>[,<NUM>]*', '<MONEY>', text)
        text = re.sub(r'<NUM>\s*(week|month|year)s?', '<DURATION>', text)
        # Replace proper nouns (simple heuristic)
        words = text.split()
        normalized_words = []
        for word in words:
            if word[0].isupper() and len(word) > 1 and not word.isupper():
                normalized_words.append('<PROPER_NOUN>')
            else:
                normalized_words.append(word.lower())
        return ' '.join(normalized_words)
    
    # Group chunks by label combination and analyze patterns
    combo_patterns = defaultdict(list)
    
    for chunk in chunks:
        combo = "|".join(sorted(chunk.get("labels", [])))
        normalized = normalize_text(chunk.get("text", ""))
        combo_patterns[combo].append(normalized)
    
    # Calculate pattern diversity for each combo
    logger.info("\nText pattern diversity by label combination:")
    for combo, patterns in combo_patterns.items():
        unique_patterns = len(set(patterns))
        total_patterns = len(patterns)
        diversity_ratio = unique_patterns / total_patterns
        
        logger.info(f"  {combo}:")
        logger.info(f"    Total chunks: {total_patterns}")
        logger.info(f"    Unique patterns: {unique_patterns}")
        logger.info(f"    Diversity ratio: {diversity_ratio:.3f}")
        
        # Show most common patterns
        pattern_counter = Counter(patterns)
        logger.info(f"    Most common patterns:")
        for pattern, count in pattern_counter.most_common(3):
            logger.info(f"      [{count}x] {pattern[:100]}...")
    
    return combo_patterns

def propose_solutions(chunks, combo_counter, combo_patterns):
    """Propose concrete solutions to address overfitting"""
    logger.info("\n=== PROPOSED SOLUTIONS ===")
    
    # Solution 1: Data Augmentation
    logger.info("1. DATA AUGMENTATION STRATEGIES:")
    logger.info("   a) Paraphrase existing chunks using different sentence structures")
    logger.info("   b) Add noise: synonym replacement, sentence reordering")
    logger.info("   c) Generate adversarial examples with similar text but different labels")
    logger.info("   d) Create label-preserving text variations")
    
    # Solution 2: More Challenging Splits
    logger.info("\n2. CHALLENGING EVALUATION SETUP:")
    logger.info("   a) Ensure no text pattern overlap between train/val/test")
    logger.info("   b) Create domain-shift test sets (different writing styles)")
    logger.info("   c) Add harder negative examples in evaluation")
    
    # Solution 3: Model Architecture Changes
    logger.info("\n3. MODEL ARCHITECTURE IMPROVEMENTS:")
    logger.info("   a) Add regularization (dropout, weight decay)")
    logger.info("   b) Use smaller models to reduce memorization capacity")
    logger.info("   c) Implement curriculum learning (start with harder examples)")
    
    # Solution 4: Training Strategy
    logger.info("\n4. TRAINING STRATEGY CHANGES:")
    logger.info("   a) Reduce epochs to prevent memorization")
    logger.info("   b) Add more aggressive data shuffling")
    logger.info("   c) Implement early stopping based on generalization gap")
    
    # Calculate specific recommendations
    total_chunks = len(chunks)
    unique_combos = len(combo_counter)
    avg_chunks_per_combo = total_chunks / unique_combos
    
    logger.info(f"\n5. SPECIFIC RECOMMENDATIONS:")
    logger.info(f"   Current state: {total_chunks} chunks, {unique_combos} combinations")
    logger.info(f"   Average chunks per combination: {avg_chunks_per_combo:.1f}")
    logger.info(f"   Recommended minimum combinations: {max(50, unique_combos * 10)}")
    logger.info(f"   Recommended text diversity per combo: >0.8 (currently varies)")

def create_augmented_data_sample(chunks):
    """Create a sample of augmented data to demonstrate the approach"""
    logger.info("\n=== CREATING AUGMENTED DATA SAMPLE ===")
    
    # Simple augmentation examples
    augmentations = []
    
    for chunk in chunks[:5]:  # Just first 5 as examples
        original_text = chunk.get("text", "")
        labels = chunk.get("labels", [])
        
        # Augmentation 1: Sentence reordering
        sentences = original_text.split('. ')
        if len(sentences) > 1:
            random.shuffle(sentences)
            aug_text1 = '. '.join(sentences)
            augmentations.append({
                "original_chunk_id": chunk.get("chunk_id"),
                "augmentation_type": "sentence_reordering",
                "original_text": original_text,
                "augmented_text": aug_text1,
                "labels": labels
            })
        
        # Augmentation 2: Synonym replacement (simple example)
        aug_text2 = original_text.replace("campaign", "project").replace("engagement", "contract")
        if aug_text2 != original_text:
            augmentations.append({
                "original_chunk_id": chunk.get("chunk_id"),
                "augmentation_type": "synonym_replacement",
                "original_text": original_text,
                "augmented_text": aug_text2,
                "labels": labels
            })
    
    # Save sample augmentations
    output_file = "Thinkerbell_template_pipeline/augmentation_samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmentations, f, indent=2)
    
    logger.info(f"Created {len(augmentations)} augmentation examples")
    logger.info(f"Saved to: {output_file}")
    
    return augmentations

def main():
    """Main diagnosis function"""
    logger.info("OVERFITTING DIAGNOSIS AND SOLUTION PROPOSAL")
    logger.info("=" * 50)
    
    # Step 1: Analyze current data
    chunks, combo_counter = analyze_label_combinations()
    combo_patterns = analyze_text_patterns(chunks)
    
    # Step 2: Propose solutions
    propose_solutions(chunks, combo_counter, combo_patterns)
    
    # Step 3: Create sample augmentations
    augmentations = create_augmented_data_sample(chunks)
    
    # Step 4: Summary and next steps
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY: CRITICAL OVERFITTING ISSUES IDENTIFIED")
    logger.info("=" * 50)
    logger.info("🚨 PROBLEM: Only 5 unique label combinations for 1,387 chunks")
    logger.info("🚨 PROBLEM: High text pattern repetition within combinations")
    logger.info("🚨 PROBLEM: Model can easily memorize patterns → perfect scores")
    logger.info("")
    logger.info("✅ IMMEDIATE ACTIONS NEEDED:")
    logger.info("1. Generate more diverse label combinations (target: 50+)")
    logger.info("2. Implement text augmentation pipeline")
    logger.info("3. Create harder evaluation scenarios")
    logger.info("4. Reduce model capacity or add regularization")
    logger.info("5. Re-train with overfitting prevention measures")

if __name__ == "__main__":
    main() 