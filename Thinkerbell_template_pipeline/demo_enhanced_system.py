#!/usr/bin/env python3
"""
Demo: Enhanced Labeling System
Shows how the enhanced system expands label space and creates complex samples
"""

import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_labeling_system import EnhancedLabelingSystem

def demo_enhanced_labeling():
    """Demonstrate enhanced labeling on sample data"""
    
    print("=" * 60)
    print("DEMO: ENHANCED LABELING SYSTEM")
    print("=" * 60)
    
    # Sample original chunk
    original_chunk = {
        "sample_id": "demo_001",
        "chunk_id": "demo_001_c1",
        "text": "Need James Wilson for Grill'd campaign. Budget around $19,094, looking for 3 x Lifestyle photography, 1 x Instagram post, 2 x Instagram story, 2 x Facebook story. 5 week exclusivity from other similar brands. Campaign runs October 2025. Usage rights for 9 months. Engagement period: 6 months.",
        "token_count": 72,
        "labels": [
            "brand",
            "campaign", 
            "client",
            "deliverables",
            "engagement_term",
            "exclusivity_scope",
            "fee",
            "usage_term"
        ]
    }
    
    print("ORIGINAL CHUNK:")
    print(f"Text: {original_chunk['text'][:100]}...")
    print(f"Original labels ({len(original_chunk['labels'])}): {original_chunk['labels']}")
    print()
    
    # Initialize enhanced labeling system
    labeling_system = EnhancedLabelingSystem(seed=42)
    
    # Generate enhanced labels for different complexity levels
    complexity_levels = ["minimal", "moderate", "complex", "comprehensive"]
    
    for complexity in complexity_levels:
        print(f"ENHANCED LABELS ({complexity.upper()}):")
        enhanced_chunk = labeling_system.generate_enhanced_labels(original_chunk, complexity)
        
        enhanced_labels = enhanced_chunk.get("enhanced_labels", [])
        print(f"  Label count: {len(enhanced_labels)}")
        print(f"  Labels: {enhanced_labels}")
        print(f"  Complexity level: {enhanced_chunk.get('complexity_level')}")
        print()
    
    # Demo multi-label samples
    print("MULTI-LABEL SAMPLE CREATION:")
    chunks = [enhanced_chunk for enhanced_chunk in [
        labeling_system.generate_enhanced_labels(original_chunk, "moderate") 
        for _ in range(3)
    ]]
    
    multi_label_samples = labeling_system.create_multi_label_samples(chunks, 1.0)
    
    if multi_label_samples:
        sample = multi_label_samples[0]
        print(f"Combined text: {sample['text'][:100]}...")
        print(f"Combined labels ({len(sample['labels'])}): {sample['labels']}")
        print(f"Source samples: {sample.get('source_samples', [])}")
        print()
    
    # Demo distractor samples
    print("DISTRACTOR SAMPLES:")
    distractor_samples = labeling_system.generate_distractor_samples(3)
    
    for i, sample in enumerate(distractor_samples):
        print(f"Distractor {i+1}:")
        print(f"  Category: {sample.get('distractor_category')}")
        print(f"  Text: {sample['text'][:80]}...")
        print(f"  Labels: {sample['labels']}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY OF IMPROVEMENTS:")
    print("=" * 60)
    
    # Calculate improvements
    original_combo = "|".join(sorted(original_chunk['labels']))
    all_enhanced_combos = set()
    
    for complexity in complexity_levels:
        enhanced = labeling_system.generate_enhanced_labels(original_chunk, complexity)
        combo = "|".join(sorted(enhanced.get("enhanced_labels", [])))
        all_enhanced_combos.add(combo)
    
    print(f"✓ Original combination: 1 unique")
    print(f"✓ Enhanced combinations: {len(all_enhanced_combos)} unique")
    print(f"✓ Multi-label capability: {len(multi_label_samples)} samples created")
    print(f"✓ Distractor samples: {len(distractor_samples)} across {len(set(s.get('distractor_category') for s in distractor_samples))} domains")
    print(f"✓ Complexity levels: {len(complexity_levels)} different difficulty tiers")
    
    improvement_ratio = len(all_enhanced_combos) / 1
    print(f"✓ Label combination expansion: {improvement_ratio}x increase")
    
    print("\nThis addresses overfitting by:")
    print("1. Expanding label space from 1 to multiple unique combinations")
    print("2. Adding complexity variation within samples")  
    print("3. Creating multi-label samples that require broader associations")
    print("4. Including realistic negative examples (distractors)")
    print("5. Maintaining proportional distribution across complexity levels")

def demo_label_analysis():
    """Analyze label patterns in enhanced system"""
    
    print("\n" + "=" * 60)
    print("LABEL PATTERN ANALYSIS")
    print("=" * 60)
    
    labeling_system = EnhancedLabelingSystem(seed=42)
    
    # Show sub-label structure
    print("SUB-LABEL STRUCTURE:")
    for base_label, sub_labels in labeling_system.sub_labels.items():
        print(f"  {base_label}: {len(sub_labels)} sub-labels")
        print(f"    → {sub_labels}")
        print()
    
    # Show content-specific labels
    print("CONTENT-SPECIFIC LABELS:")
    for category, labels in labeling_system.content_specific_labels.items():
        print(f"  {category}: {len(labels)} labels")
        print(f"    → {labels}")
        print()
    
    # Show distractor categories
    print("DISTRACTOR CATEGORIES:")
    for category, labels in labeling_system.distractor_categories.items():
        print(f"  {category}: {len(labels)} labels")
        print(f"    → {labels}")
        print()
    
    # Calculate total possible combinations
    total_base_labels = len(labeling_system.base_labels)
    total_sub_labels = sum(len(subs) for subs in labeling_system.sub_labels.values())
    total_content_labels = sum(len(labels) for labels in labeling_system.content_specific_labels.values())
    total_distractor_labels = sum(len(labels) for labels in labeling_system.distractor_categories.values())
    
    total_labels = total_base_labels + total_sub_labels + total_content_labels + total_distractor_labels
    
    print("LABEL SPACE STATISTICS:")
    print(f"  Base labels: {total_base_labels}")
    print(f"  Sub-labels: {total_sub_labels}")
    print(f"  Content-specific: {total_content_labels}")
    print(f"  Distractor labels: {total_distractor_labels}")
    print(f"  Total label space: {total_labels}")
    print(f"  Theoretical combinations: exponential complexity")

if __name__ == "__main__":
    demo_enhanced_labeling()
    demo_label_analysis() 