#!/usr/bin/env python3
"""
Enhanced Split Chunks for Thinkerbell Formatter Project
Handles expanded label space with sub-labels, multi-label samples, and distractor classes
Maintains label distribution while preserving document integrity
"""

import argparse
import json
import os
import random
import shutil
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class EnhancedChunkSplitter:
    """
    Enhanced splitter for complex label space with:
    - Sub-labels and multi-labels
    - Distractor samples
    - Proportional distribution maintenance
    - Stratified sampling for balanced splits
    """
    
    def __init__(self, 
                 input_file: str = "synthetic_dataset/enhanced_splits/all_enhanced_samples.json",
                 output_dir: str = "synthetic_dataset/enhanced_splits",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42,
                 stratify_by: str = "complexity"):
        
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.stratify_by = stratify_by  # "complexity", "label_count", or "sample_type"
        
        # Verify ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        if not 0.999 <= total_ratio <= 1.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize statistics
        self.stats = {
            "total_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "label_distribution": defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0}),
            "complexity_distribution": defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0}),
            "sample_type_distribution": defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0}),
            "label_combination_distribution": defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0}),
            "label_count_distribution": defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})
        }

    def load_enhanced_samples(self) -> List[Dict]:
        """Load all enhanced samples from input file"""
        logger.info(f"Loading enhanced samples from {self.input_file}")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        logger.info(f"Loaded {len(samples)} enhanced samples")
        self.stats["total_samples"] = len(samples)
        
        return samples

    def create_stratification_groups(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Create stratification groups based on specified strategy"""
        
        groups = defaultdict(list)
        
        for sample in samples:
            if self.stratify_by == "complexity":
                key = sample.get("complexity_level", "unknown")
            elif self.stratify_by == "label_count":
                label_count = sample.get("label_count", len(sample.get("enhanced_labels", [])))
                # Group by label count ranges
                if label_count <= 2:
                    key = "low_complexity"
                elif label_count <= 5:
                    key = "medium_complexity"
                elif label_count <= 8:
                    key = "high_complexity"
                else:
                    key = "very_high_complexity"
            elif self.stratify_by == "sample_type":
                if sample.get("is_distractor", False):
                    key = "distractor"
                elif sample.get("is_multi_label", False):
                    key = "multi_label"
                else:
                    key = "enhanced_original"
            else:
                # Default: use primary label as stratification key
                labels = sample.get("enhanced_labels", [])
                key = labels[0] if labels else "unlabeled"
            
            groups[key].append(sample)
        
        logger.info(f"Created {len(groups)} stratification groups:")
        for group_name, group_samples in groups.items():
            logger.info(f"  {group_name}: {len(group_samples)} samples")
        
        return dict(groups)

    def stratified_split(self, groups: Dict[str, List[Dict]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Perform stratified split to maintain group proportions across splits"""
        
        train_samples, val_samples, test_samples = [], [], []
        
        for group_name, group_samples in groups.items():
            if len(group_samples) == 0:
                continue
            
            # Shuffle group samples
            shuffled_samples = group_samples.copy()
            random.shuffle(shuffled_samples)
            
            if len(shuffled_samples) == 1:
                # Single sample - assign to train
                train_samples.extend(shuffled_samples)
                logger.info(f"  {group_name}: 1 sample → train")
                continue
            
            if len(shuffled_samples) == 2:
                # Two samples - one to train, one to val/test
                train_samples.append(shuffled_samples[0])
                val_samples.append(shuffled_samples[1])
                logger.info(f"  {group_name}: 2 samples → 1 train, 1 val")
                continue
            
            # Calculate split sizes for this group
            n_samples = len(shuffled_samples)
            n_train = max(1, int(n_samples * self.train_ratio))
            n_val = max(1, int(n_samples * self.val_ratio))
            n_test = n_samples - n_train - n_val
            
            # Ensure we have at least one sample in test if possible
            if n_test <= 0 and n_samples >= 3:
                n_test = 1
                n_val = max(1, n_samples - n_train - n_test)
            
            # Split the group
            train_end = n_train
            val_end = n_train + n_val
            
            group_train = shuffled_samples[:train_end]
            group_val = shuffled_samples[train_end:val_end]
            group_test = shuffled_samples[val_end:]
            
            train_samples.extend(group_train)
            val_samples.extend(group_val)
            test_samples.extend(group_test)
            
            logger.info(f"  {group_name}: {n_samples} samples → {len(group_train)} train, {len(group_val)} val, {len(group_test)} test")
        
        logger.info(f"Final split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        return train_samples, val_samples, test_samples

    def update_statistics(self, train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
        """Update detailed statistics for all splits"""
        
        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        }
        
        for split_name, samples in splits.items():
            self.stats[f"{split_name}_samples"] = len(samples)
            
            for sample in samples:
                labels = sample.get("enhanced_labels", [])
                complexity = sample.get("complexity_level", "unknown")
                label_count = sample.get("label_count", len(labels))
                
                # Sample type
                if sample.get("is_distractor", False):
                    sample_type = "distractor"
                elif sample.get("is_multi_label", False):
                    sample_type = "multi_label"
                else:
                    sample_type = "enhanced_original"
                
                # Update label distribution
                for label in labels:
                    self.stats["label_distribution"][label][split_name] += 1
                    self.stats["label_distribution"][label]["total"] += 1
                
                # Update complexity distribution
                self.stats["complexity_distribution"][complexity][split_name] += 1
                self.stats["complexity_distribution"][complexity]["total"] += 1
                
                # Update sample type distribution
                self.stats["sample_type_distribution"][sample_type][split_name] += 1
                self.stats["sample_type_distribution"][sample_type]["total"] += 1
                
                # Update label combination distribution
                combo = "|".join(sorted(labels))
                self.stats["label_combination_distribution"][combo][split_name] += 1
                self.stats["label_combination_distribution"][combo]["total"] += 1
                
                # Update label count distribution
                count_range = self._get_label_count_range(label_count)
                self.stats["label_count_distribution"][count_range][split_name] += 1
                self.stats["label_count_distribution"][count_range]["total"] += 1

    def _get_label_count_range(self, count: int) -> str:
        """Get label count range string"""
        if count <= 2:
            return "1-2_labels"
        elif count <= 4:
            return "3-4_labels"
        elif count <= 6:
            return "5-6_labels"
        elif count <= 8:
            return "7-8_labels"
        else:
            return "9+_labels"

    def save_splits(self, train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
        """Save split samples to separate directories"""
        
        # Create output directories
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        }
        
        for split_name, samples in splits.items():
            split_dir = self.output_dir / split_name
            
            logger.info(f"Saving {len(samples)} samples to {split_dir}")
            
            for sample in samples:
                # Create filename
                sample_id = sample.get("sample_id", f"unknown_{random.randint(1000, 9999)}")
                chunk_id = sample.get("chunk_id", f"{sample_id}_c1")
                filename = f"{chunk_id}.json"
                
                # Save sample
                output_file = split_dir / filename
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(sample, f, indent=2, ensure_ascii=False)

    def save_statistics(self):
        """Save detailed statistics"""
        stats_file = self.output_dir / "enhanced_split_stats.json"
        
        # Convert defaultdicts to regular dicts for JSON serialization
        serializable_stats = {}
        for key, value in self.stats.items():
            if isinstance(value, defaultdict):
                serializable_stats[key] = dict(value)
            else:
                serializable_stats[key] = value
        
        # Add summary statistics
        serializable_stats["summary"] = {
            "total_samples": self.stats["total_samples"],
            "train_ratio_actual": self.stats["train_samples"] / self.stats["total_samples"],
            "val_ratio_actual": self.stats["val_samples"] / self.stats["total_samples"],
            "test_ratio_actual": self.stats["test_samples"] / self.stats["total_samples"],
            "unique_labels": len(self.stats["label_distribution"]),
            "unique_combinations": len(self.stats["label_combination_distribution"]),
            "stratification_method": self.stratify_by
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Statistics saved to {stats_file}")

    def validate_split_balance(self):
        """Validate that splits maintain reasonable label distribution balance"""
        
        logger.info("\n=== SPLIT BALANCE VALIDATION ===")
        
        # Check label distribution balance
        total_samples = self.stats["total_samples"]
        imbalanced_labels = []
        
        for label, distribution in self.stats["label_distribution"].items():
            train_ratio = distribution["train"] / distribution["total"] if distribution["total"] > 0 else 0
            val_ratio = distribution["val"] / distribution["total"] if distribution["total"] > 0 else 0
            test_ratio = distribution["test"] / distribution["total"] if distribution["total"] > 0 else 0
            
            # Check if ratios are too far from target
            target_train, target_val, target_test = self.train_ratio, self.val_ratio, self.test_ratio
            
            if (abs(train_ratio - target_train) > 0.15 or 
                abs(val_ratio - target_val) > 0.15 or 
                abs(test_ratio - target_test) > 0.15):
                imbalanced_labels.append({
                    "label": label,
                    "total": distribution["total"],
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "test_ratio": test_ratio
                })
        
        if imbalanced_labels:
            logger.warning(f"Found {len(imbalanced_labels)} labels with imbalanced distribution:")
            for item in imbalanced_labels[:5]:  # Show top 5
                logger.warning(f"  {item['label']}: train={item['train_ratio']:.2f}, val={item['val_ratio']:.2f}, test={item['test_ratio']:.2f}")
        else:
            logger.info("✓ All labels have balanced distribution across splits")
        
        # Check complexity distribution
        logger.info("\nComplexity distribution across splits:")
        for complexity, distribution in self.stats["complexity_distribution"].items():
            train_ratio = distribution["train"] / distribution["total"] if distribution["total"] > 0 else 0
            val_ratio = distribution["val"] / distribution["total"] if distribution["total"] > 0 else 0
            test_ratio = distribution["test"] / distribution["total"] if distribution["total"] > 0 else 0
            
            logger.info(f"  {complexity}: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")

    def process_enhanced_splits(self) -> Dict:
        """Main processing function for enhanced splits"""
        
        logger.info("=== ENHANCED CHUNK SPLITTING ===")
        
        # Load samples
        samples = self.load_enhanced_samples()
        
        # Create stratification groups
        groups = self.create_stratification_groups(samples)
        
        # Perform stratified split
        train_samples, val_samples, test_samples = self.stratified_split(groups)
        
        # Update statistics
        self.update_statistics(train_samples, val_samples, test_samples)
        
        # Save splits
        self.save_splits(train_samples, val_samples, test_samples)
        
        # Save statistics
        self.save_statistics()
        
        # Validate balance
        self.validate_split_balance()
        
        logger.info(f"\n✓ Enhanced splitting complete!")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Train: {len(train_samples)} samples")
        logger.info(f"  Val: {len(val_samples)} samples")
        logger.info(f"  Test: {len(test_samples)} samples")
        
        return dict(self.stats)

def main():
    """Main function for enhanced chunk splitting"""
    parser = argparse.ArgumentParser(description="Enhanced chunk splitting with expanded label space")
    
    parser.add_argument("--input-file", type=str, 
                       default="synthetic_dataset/enhanced_splits/all_enhanced_samples.json",
                       help="Input file with enhanced samples")
    parser.add_argument("--output-dir", type=str,
                       default="synthetic_dataset/enhanced_splits",
                       help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Ratio for training set")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Ratio for validation set")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Ratio for test set")
    parser.add_argument("--stratify-by", type=str, default="complexity",
                       choices=["complexity", "label_count", "sample_type"],
                       help="Stratification strategy")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize splitter
    splitter = EnhancedChunkSplitter(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by=args.stratify_by
    )
    
    # Process splits
    stats = splitter.process_enhanced_splits()
    
    # Print summary
    logger.info("\n=== FINAL SUMMARY ===")
    logger.info(f"Total samples processed: {stats['total_samples']}")
    logger.info(f"Unique labels: {len(stats['label_distribution'])}")
    logger.info(f"Unique combinations: {len(stats['label_combination_distribution'])}")
    logger.info(f"Stratification method: {args.stratify_by}")

if __name__ == "__main__":
    main() 