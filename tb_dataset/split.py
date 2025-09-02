"""
Group-aware dataset splitting module.

Provides stratified splitting that ensures no group leakage across train/val/test
sets while maintaining class distribution balance.
"""

import hashlib
import logging
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import json


logger = logging.getLogger(__name__)


class GroupAwareSplitter:
    """
    Handles group-aware dataset splitting to prevent data leakage.
    
    Groups samples by template_family or falls back to client+industry hash.
    Ensures no group appears in multiple splits while maintaining stratification.
    """
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        Initialize group-aware splitter.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            random_seed: Random seed for reproducible splits
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Set random seed
        random.seed(random_seed)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'total_groups': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'train_groups': 0,
            'val_groups': 0,
            'test_groups': 0,
            'complexity_distribution': {
                'train': {},
                'val': {},
                'test': {}
            },
            'industry_distribution': {
                'train': {},
                'val': {},
                'test': {}
            },
            'document_type_distribution': {
                'train': {},
                'val': {},
                'test': {}
            }
        }
    
    def compute_group_id(self, sample_data: Dict[str, Any]) -> str:
        """
        Compute group ID for a sample.
        
        Uses template_hint if available, otherwise falls back to
        sha1(industry|client) for grouping.
        
        Args:
            sample_data: Sample data dictionary
            
        Returns:
            Group ID string
        """
        # Try template_hint first
        template_hint = sample_data.get('template_hint')
        if template_hint:
            return f"template_{template_hint}"
        
        # Fallback to industry|client hash
        classification = sample_data.get('classification', {})
        extracted_fields = sample_data.get('extracted_fields', {})
        
        industry = classification.get('industry', 'unknown')
        client = extracted_fields.get('client', 'unknown')
        
        # Create hash of industry|client combination
        group_key = f"{industry}|{client}"
        group_hash = hashlib.sha1(group_key.encode('utf-8')).hexdigest()[:8]
        
        return f"industry_client_{group_hash}"
    
    def group_samples(self, samples: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Group samples by their computed group IDs.
        
        Args:
            samples: Dictionary of sample_id -> sample_data
            
        Returns:
            Dictionary mapping group_id -> list of sample_ids
        """
        groups = defaultdict(list)
        
        for sample_id, sample_data in samples.items():
            group_id = self.compute_group_id(sample_data)
            groups[group_id].append(sample_id)
        
        logger.info(f"Created {len(groups)} groups from {len(samples)} samples")
        
        # Log group size distribution
        group_sizes = [len(sample_ids) for sample_ids in groups.values()]
        avg_group_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0
        max_group_size = max(group_sizes) if group_sizes else 0
        min_group_size = min(group_sizes) if group_sizes else 0
        
        logger.info(f"Group sizes: min={min_group_size}, max={max_group_size}, avg={avg_group_size:.1f}")
        
        return dict(groups)
    
    def stratify_groups(self, 
                       groups: Dict[str, List[str]], 
                       samples: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Stratify groups by complexity to maintain balanced distribution.
        
        Args:
            groups: Dictionary mapping group_id -> list of sample_ids
            samples: Dictionary of sample_id -> sample_data
            
        Returns:
            Dictionary mapping complexity -> list of group_ids
        """
        complexity_groups = defaultdict(list)
        
        for group_id, sample_ids in groups.items():
            # Determine group complexity by majority vote
            complexities = []
            for sample_id in sample_ids:
                complexity = samples[sample_id].get('classification', {}).get('complexity', 'medium')
                complexities.append(complexity)
            
            # Use most common complexity for the group
            group_complexity = Counter(complexities).most_common(1)[0][0]
            complexity_groups[group_complexity].append(group_id)
        
        logger.info(f"Complexity stratification: {dict((k, len(v)) for k, v in complexity_groups.items())}")
        
        return dict(complexity_groups)
    
    def split_groups(self, 
                    complexity_groups: Dict[str, List[str]], 
                    groups: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
        """
        Split groups across train/val/test while maintaining complexity stratification.
        
        Args:
            complexity_groups: Dictionary mapping complexity -> list of group_ids
            groups: Dictionary mapping group_id -> list of sample_ids
            
        Returns:
            Tuple of (train_groups, val_groups, test_groups)
        """
        train_groups = []
        val_groups = []
        test_groups = []
        
        # Split each complexity stratum independently
        for complexity, group_ids in complexity_groups.items():
            # Shuffle groups for random assignment
            shuffled_groups = group_ids.copy()
            random.shuffle(shuffled_groups)
            
            # Calculate split points based on number of samples in groups
            total_samples_in_complexity = sum(len(groups[gid]) for gid in group_ids)
            
            train_target = int(total_samples_in_complexity * self.train_ratio)
            val_target = int(total_samples_in_complexity * self.val_ratio)
            
            # Assign groups to splits
            current_train_samples = 0
            current_val_samples = 0
            
            for group_id in shuffled_groups:
                group_size = len(groups[group_id])
                
                # Decide which split this group goes to
                if current_train_samples < train_target:
                    train_groups.append(group_id)
                    current_train_samples += group_size
                elif current_val_samples < val_target:
                    val_groups.append(group_id)
                    current_val_samples += group_size
                else:
                    test_groups.append(group_id)
            
            logger.info(f"Complexity {complexity}: train={len([g for g in train_groups if g in group_ids])}, "
                       f"val={len([g for g in val_groups if g in group_ids])}, "
                       f"test={len([g for g in test_groups if g in group_ids])} groups")
        
        return train_groups, val_groups, test_groups
    
    def create_sample_splits(self, 
                           train_groups: List[str],
                           val_groups: List[str], 
                           test_groups: List[str],
                           groups: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
        """
        Convert group splits to sample splits.
        
        Args:
            train_groups: List of group IDs for training
            val_groups: List of group IDs for validation  
            test_groups: List of group IDs for testing
            groups: Dictionary mapping group_id -> list of sample_ids
            
        Returns:
            Tuple of (train_sample_ids, val_sample_ids, test_sample_ids)
        """
        train_samples = []
        val_samples = []
        test_samples = []
        
        for group_id in train_groups:
            train_samples.extend(groups[group_id])
        
        for group_id in val_groups:
            val_samples.extend(groups[group_id])
        
        for group_id in test_groups:
            test_samples.extend(groups[group_id])
        
        return train_samples, val_samples, test_samples
    
    def calculate_statistics(self, 
                           train_samples: List[str],
                           val_samples: List[str],
                           test_samples: List[str],
                           samples: Dict[str, Dict[str, Any]],
                           train_groups: List[str],
                           val_groups: List[str],
                           test_groups: List[str]) -> Dict[str, Any]:
        """
        Calculate detailed statistics for the splits.
        
        Args:
            train_samples: Training sample IDs
            val_samples: Validation sample IDs
            test_samples: Test sample IDs
            samples: All sample data
            train_groups: Training group IDs
            val_groups: Validation group IDs
            test_groups: Test group IDs
            
        Returns:
            Statistics dictionary
        """
        def analyze_split(sample_ids: List[str], split_name: str):
            """Analyze a single split."""
            split_stats = {
                'complexity': defaultdict(int),
                'industry': defaultdict(int),
                'document_type': defaultdict(int)
            }
            
            for sample_id in sample_ids:
                sample_data = samples[sample_id]
                classification = sample_data.get('classification', {})
                
                complexity = classification.get('complexity', 'unknown')
                industry = classification.get('industry', 'unknown')
                doc_type = classification.get('document_type', 'unknown')
                
                split_stats['complexity'][complexity] += 1
                split_stats['industry'][industry] += 1
                split_stats['document_type'][doc_type] += 1
            
            return dict(split_stats['complexity']), dict(split_stats['industry']), dict(split_stats['document_type'])
        
        # Analyze each split
        train_complexity, train_industry, train_doc_type = analyze_split(train_samples, 'train')
        val_complexity, val_industry, val_doc_type = analyze_split(val_samples, 'val')
        test_complexity, test_industry, test_doc_type = analyze_split(test_samples, 'test')
        
        # Update stats
        self.stats.update({
            'total_samples': len(samples),
            'total_groups': len(set(train_groups + val_groups + test_groups)),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'train_groups': len(train_groups),
            'val_groups': len(val_groups),
            'test_groups': len(test_groups),
            'complexity_distribution': {
                'train': train_complexity,
                'val': val_complexity,
                'test': test_complexity
            },
            'industry_distribution': {
                'train': train_industry,
                'val': val_industry,
                'test': test_industry
            },
            'document_type_distribution': {
                'train': train_doc_type,
                'val': val_doc_type,
                'test': test_doc_type
            }
        })
        
        return self.stats
    
    def validate_splits(self, 
                       train_groups: List[str],
                       val_groups: List[str],
                       test_groups: List[str]) -> bool:
        """
        Validate that splits have no group leakage.
        
        Args:
            train_groups: Training group IDs
            val_groups: Validation group IDs
            test_groups: Test group IDs
            
        Returns:
            True if splits are valid (no overlap)
        """
        train_set = set(train_groups)
        val_set = set(val_groups)
        test_set = set(test_groups)
        
        # Check for overlaps
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap:
            logger.error(f"Train/Val group overlap: {train_val_overlap}")
            return False
        
        if train_test_overlap:
            logger.error(f"Train/Test group overlap: {train_test_overlap}")
            return False
        
        if val_test_overlap:
            logger.error(f"Val/Test group overlap: {val_test_overlap}")
            return False
        
        logger.info("Split validation passed: no group leakage detected")
        return True
    
    def split_samples(self, samples: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Perform complete group-aware splitting of samples.
        
        Args:
            samples: Dictionary of sample_id -> sample_data
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys mapping to sample ID lists
        """
        logger.info(f"Starting group-aware split of {len(samples)} samples")
        
        # Group samples
        groups = self.group_samples(samples)
        
        # Stratify by complexity
        complexity_groups = self.stratify_groups(groups, samples)
        
        # Split groups
        train_groups, val_groups, test_groups = self.split_groups(complexity_groups, groups)
        
        # Validate splits
        if not self.validate_splits(train_groups, val_groups, test_groups):
            raise ValueError("Split validation failed: group leakage detected")
        
        # Convert to sample splits
        train_samples, val_samples, test_samples = self.create_sample_splits(
            train_groups, val_groups, test_groups, groups
        )
        
        # Calculate statistics
        self.calculate_statistics(
            train_samples, val_samples, test_samples, samples,
            train_groups, val_groups, test_groups
        )
        
        logger.info(f"Split complete: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
    def get_split_statistics(self) -> Dict[str, Any]:
        """Get detailed split statistics."""
        return self.stats.copy()
    
    def save_split_metadata(self, output_dir: Path, splits: Dict[str, List[str]], samples: Dict[str, Dict[str, Any]]):
        """
        Save split metadata and statistics.
        
        Args:
            output_dir: Directory to save metadata
            splits: Dictionary with split assignments
            samples: All sample data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save split assignments
        split_assignments_file = output_dir / 'split_assignments.json'
        with open(split_assignments_file, 'w', encoding='utf-8') as f:
            json.dump(splits, f, ensure_ascii=False, indent=2)
        
        # Save detailed statistics
        split_stats_file = output_dir / 'split_stats.json'
        with open(split_stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # Save group mappings for debugging
        groups = self.group_samples(samples)
        group_mappings_file = output_dir / 'group_mappings.json'
        
        group_data = {}
        for group_id, sample_ids in groups.items():
            group_data[group_id] = {
                'sample_ids': sample_ids,
                'size': len(sample_ids),
                'complexity': samples[sample_ids[0]].get('classification', {}).get('complexity', 'unknown'),
                'industry': samples[sample_ids[0]].get('classification', {}).get('industry', 'unknown')
            }
        
        with open(group_mappings_file, 'w', encoding='utf-8') as f:
            json.dump(group_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved split metadata to {output_dir}")
    
    def print_split_summary(self):
        """Print a summary of the split results."""
        if not self.stats['total_samples']:
            print("No split statistics available")
            return
        
        print("\n" + "="*60)
        print("DATASET SPLIT SUMMARY")
        print("="*60)
        
        # Overall statistics
        print(f"Total samples: {self.stats['total_samples']:,}")
        print(f"Total groups: {self.stats['total_groups']:,}")
        print()
        
        # Split sizes
        print("Split Distribution:")
        train_pct = self.stats['train_samples'] / self.stats['total_samples'] * 100
        val_pct = self.stats['val_samples'] / self.stats['total_samples'] * 100
        test_pct = self.stats['test_samples'] / self.stats['total_samples'] * 100
        
        print(f"  Train: {self.stats['train_samples']:,} samples ({train_pct:.1f}%) from {self.stats['train_groups']} groups")
        print(f"  Val:   {self.stats['val_samples']:,} samples ({val_pct:.1f}%) from {self.stats['val_groups']} groups")
        print(f"  Test:  {self.stats['test_samples']:,} samples ({test_pct:.1f}%) from {self.stats['test_groups']} groups")
        print()
        
        # Complexity distribution
        print("Complexity Distribution:")
        for complexity in ['simple', 'medium', 'complex']:
            train_count = self.stats['complexity_distribution']['train'].get(complexity, 0)
            val_count = self.stats['complexity_distribution']['val'].get(complexity, 0)
            test_count = self.stats['complexity_distribution']['test'].get(complexity, 0)
            total_count = train_count + val_count + test_count
            
            if total_count > 0:
                print(f"  {complexity.capitalize()}: Train={train_count}, Val={val_count}, Test={test_count} (Total={total_count})")
        
        print()
        
        # Document type distribution
        print("Document Type Distribution:")
        doc_types = set()
        for split in ['train', 'val', 'test']:
            doc_types.update(self.stats['document_type_distribution'][split].keys())
        
        for doc_type in sorted(doc_types):
            train_count = self.stats['document_type_distribution']['train'].get(doc_type, 0)
            val_count = self.stats['document_type_distribution']['val'].get(doc_type, 0)
            test_count = self.stats['document_type_distribution']['test'].get(doc_type, 0)
            total_count = train_count + val_count + test_count
            
            if total_count > 0:
                print(f"  {doc_type}: Train={train_count}, Val={val_count}, Test={test_count} (Total={total_count})")
        
        print("="*60) 