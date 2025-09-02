#!/usr/bin/env python3
"""
Split Chunks for Thinkerbell Formatter Project
Splits chunks into train/val/test sets while maintaining label distribution
and ensuring all chunks from the same document stay in the same split
"""

import argparse
import json
import os
import random
import shutil
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ChunkSplitter:
    """
    Splits chunks into train/val/test sets while preserving label distribution
    and ensuring document integrity (all chunks from one document in the same split)
    """
    
    def __init__(self, 
                 input_dir: str = "synthetic_dataset/chunks",
                 output_dir: str = "synthetic_dataset/splits",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 copy_mode: bool = True,
                 seed: int = 42):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.copy_mode = copy_mode
        self.seed = seed
        
        # Verify ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        if not 0.999 <= total_ratio <= 1.001:  # Allow for small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Stats tracking
        self.stats = {
            "total_chunks": 0,
            "total_documents": 0,
            "train_chunks": 0,
            "val_chunks": 0,
            "test_chunks": 0,
            "train_documents": 0,
            "val_documents": 0,
            "test_documents": 0,
            "label_distribution": defaultdict(lambda: {"total": 0, "train": 0, "val": 0, "test": 0}),
            "label_combo_distribution": {}
        }
    
    def extract_document_id(self, chunk_id):
        """Extract document ID from chunk ID (e.g., 'sample_123_c1' -> 'sample_123')"""
        # Default pattern: remove _c\d+ suffix
        match = re.match(r'(.*?)_c\d+$', chunk_id)
        if match:
            return match.group(1)
        return chunk_id  # If no match, return the original chunk_id

    def load_chunks(self) -> Dict[str, List[Dict]]:
        """
        Load all chunk files from input directory
        Returns dictionary mapping document_id -> list of chunk data
        """
        logger.info(f"Loading chunks from {self.input_dir}")
        
        all_chunks = []
        doc_to_chunks = defaultdict(list)
        chunk_files = list(self.input_dir.glob("**/*.json"))
        
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self.input_dir}")
        
        logger.info(f"Found {len(chunk_files)} chunk files")
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    # Extract document ID from chunk ID
                    chunk_id = chunk_data.get('chunk_id', chunk_file.stem)
                    document_id = chunk_data.get('document_id', None)
                    if document_id is None:
                        document_id = self.extract_document_id(chunk_id)
                    
                    # Add file path for later processing
                    chunk_data['file_path'] = chunk_file
                    chunk_data['document_id'] = document_id
                    
                    all_chunks.append(chunk_data)
                    doc_to_chunks[document_id].append(chunk_data)
            except Exception as e:
                logger.error(f"Error loading {chunk_file}: {e}")
        
        self.stats["total_chunks"] = len(all_chunks)
        self.stats["total_documents"] = len(doc_to_chunks)
        logger.info(f"Loaded {len(all_chunks)} chunks from {len(doc_to_chunks)} documents")
        
        # Update label distribution stats for all chunks
        for chunk in all_chunks:
            for label in chunk.get('labels', []):
                self.stats["label_distribution"][label]["total"] += 1
        
        return doc_to_chunks

    def group_documents_by_label_profile(self, doc_to_chunks: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
        """
        Group documents based on their label profile
        Returns dict mapping label-profile -> list of document_ids
        """
        # Calculate label profile for each document
        doc_label_profiles = {}
        
        for doc_id, chunks in doc_to_chunks.items():
            # Count labels in this document
            label_counts = defaultdict(int)
            for chunk in chunks:
                for label in chunk.get('labels', []):
                    label_counts[label] += 1
                    
            # Normalize to get label profile
            total = sum(label_counts.values())
            if total > 0:
                label_profile = {label: count/total for label, count in label_counts.items()}
                
                # Convert to string for grouping
                profile_key = "|".join(f"{l}:{label_profile[l]:.2f}" for l in sorted(label_profile.keys()))
                doc_label_profiles[doc_id] = profile_key
        
        # Group documents by profile
        profile_to_docs = defaultdict(list)
        for doc_id, profile in doc_label_profiles.items():
            profile_to_docs[profile].append(doc_id)
            
        # Log profile distribution
        logger.info(f"Found {len(profile_to_docs)} unique document label profiles")
        
        return profile_to_docs

    def stratified_document_split(self, 
                                doc_to_chunks: Dict[str, List[Dict]]) -> Tuple[List[str], List[str], List[str]]:
        """
        Split documents into train/val/test preserving label distribution
        Returns (train_doc_ids, val_doc_ids, test_doc_ids)
        """
        profile_to_docs = self.group_documents_by_label_profile(doc_to_chunks)
        
        # Prepare split assignments
        train_docs, val_docs, test_docs = [], [], []
        
        # For each profile group, split proportionally
        for profile, doc_ids in profile_to_docs.items():
            # Shuffle documents within each profile group
            random.shuffle(doc_ids)
            
            # Calculate split points
            n = len(doc_ids)
            train_idx = int(n * self.train_ratio)
            val_idx = train_idx + int(n * self.val_ratio)
            
            # Assign documents to splits
            train_docs.extend(doc_ids[:train_idx])
            val_docs.extend(doc_ids[train_idx:val_idx])
            test_docs.extend(doc_ids[val_idx:])
        
        return train_docs, val_docs, test_docs

    def process_splits(self) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        Process all chunks and split into train/val/test sets
        Returns (train_chunks_dict, val_chunks_dict, test_chunks_dict)
        """
        # Load all chunks grouped by document
        doc_to_chunks = self.load_chunks()
        
        # Split documents while maintaining label distribution
        train_docs, val_docs, test_docs = self.stratified_document_split(doc_to_chunks)
        
        # Assign all chunks based on document assignment
        train_chunks, val_chunks, test_chunks = {}, {}, {}
        
        for doc_id in train_docs:
            train_chunks[doc_id] = doc_to_chunks[doc_id]
            self.stats["train_chunks"] += len(doc_to_chunks[doc_id])
            # Update label stats
            for chunk in doc_to_chunks[doc_id]:
                for label in chunk.get('labels', []):
                    self.stats["label_distribution"][label]["train"] += 1
        
        for doc_id in val_docs:
            val_chunks[doc_id] = doc_to_chunks[doc_id]
            self.stats["val_chunks"] += len(doc_to_chunks[doc_id])
            # Update label stats
            for chunk in doc_to_chunks[doc_id]:
                for label in chunk.get('labels', []):
                    self.stats["label_distribution"][label]["val"] += 1
        
        for doc_id in test_docs:
            test_chunks[doc_id] = doc_to_chunks[doc_id]
            self.stats["test_chunks"] += len(doc_to_chunks[doc_id])
            # Update label stats
            for chunk in doc_to_chunks[doc_id]:
                for label in chunk.get('labels', []):
                    self.stats["label_distribution"][label]["test"] += 1
        
        # Update document counts
        self.stats["train_documents"] = len(train_docs)
        self.stats["val_documents"] = len(val_docs)
        self.stats["test_documents"] = len(test_docs)
        
        logger.info(f"Split into: {len(train_docs)} train documents ({self.stats['train_chunks']} chunks), "
                    f"{len(val_docs)} val documents ({self.stats['val_chunks']} chunks), "
                    f"{len(test_docs)} test documents ({self.stats['test_chunks']} chunks)")
        
        return train_chunks, val_chunks, test_chunks

    def save_splits(self, train_chunks: Dict[str, List[Dict]], val_chunks: Dict[str, List[Dict]], 
                    test_chunks: Dict[str, List[Dict]]):
        """
        Save splits to output directories
        """
        # Create output directories
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        test_dir = self.output_dir / "test"
        
        for directory in [train_dir, val_dir, test_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Helper function to save chunks
        def save_chunks_from_dict(chunks_dict: Dict[str, List[Dict]], target_dir: Path):
            for doc_id, chunks in chunks_dict.items():
                for chunk in chunks:
                    src_path = chunk['file_path']
                    target_path = target_dir / src_path.name
                    
                    if self.copy_mode:
                        shutil.copy2(src_path, target_path)
                    else:
                        shutil.move(src_path, target_path)
        
        # Save each split
        save_chunks_from_dict(train_chunks, train_dir)
        save_chunks_from_dict(val_chunks, val_dir)
        save_chunks_from_dict(test_chunks, test_dir)
        
        logger.info(f"Saved splits to {self.output_dir}")
        logger.info(f"  Train: {train_dir}")
        logger.info(f"  Val: {val_dir}")
        logger.info(f"  Test: {test_dir}")

    def generate_stats_file(self):
        """
        Generate and save statistics about the splits
        """
        stats_path = self.output_dir / "split_stats.json"
        
        # Calculate percentages for label distribution
        for label, counts in self.stats["label_distribution"].items():
            total = counts["total"]
            for split in ["train", "val", "test"]:
                if total > 0:
                    counts[f"{split}_pct"] = round(counts[split] / total * 100, 1)
                else:
                    counts[f"{split}_pct"] = 0
        
        # Add document distribution info
        self.stats["document_distribution"] = {
            "train": self.stats["train_documents"],
            "val": self.stats["val_documents"],
            "test": self.stats["test_documents"],
            "train_pct": round(self.stats["train_documents"] / self.stats["total_documents"] * 100, 1),
            "val_pct": round(self.stats["val_documents"] / self.stats["total_documents"] * 100, 1),
            "test_pct": round(self.stats["test_documents"] / self.stats["total_documents"] * 100, 1)
        }
        
        # Add chunk distribution info
        self.stats["chunk_distribution"] = {
            "train": self.stats["train_chunks"],
            "val": self.stats["val_chunks"],
            "test": self.stats["test_chunks"],
            "train_pct": round(self.stats["train_chunks"] / self.stats["total_chunks"] * 100, 1),
            "val_pct": round(self.stats["val_chunks"] / self.stats["total_chunks"] * 100, 1),
            "test_pct": round(self.stats["test_chunks"] / self.stats["total_chunks"] * 100, 1)
        }
        
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)
            logger.info(f"Saved statistics to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")

    def run(self):
        """
        Run the entire splitting process
        """
        logger.info(f"Starting document-based split process with seed {self.seed}")
        logger.info(f"Split ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        logger.info(f"Ensuring all chunks from the same document stay in the same split")
        
        train_chunks, val_chunks, test_chunks = self.process_splits()
        self.save_splits(train_chunks, val_chunks, test_chunks)
        self.generate_stats_file()
        
        # Print label distribution summary
        logger.info("\nLabel distribution summary:")
        logger.info(f"{'Label':<20} {'Total':<8} {'Train %':<8} {'Val %':<8} {'Test %':<8}")
        logger.info("-" * 60)
        
        for label, counts in sorted(self.stats["label_distribution"].items()):
            logger.info(f"{label:<20} {counts['total']:<8} {counts['train_pct']:<8} {counts['val_pct']:<8} {counts['test_pct']:<8}")
        
        # Print document distribution
        logger.info("\nDocument distribution:")
        logger.info(f"Train: {self.stats['train_documents']} documents ({self.stats['document_distribution']['train_pct']}%)")
        logger.info(f"Val:   {self.stats['val_documents']} documents ({self.stats['document_distribution']['val_pct']}%)")
        logger.info(f"Test:  {self.stats['test_documents']} documents ({self.stats['document_distribution']['test_pct']}%)")
        
        # Print chunk distribution
        logger.info("\nChunk distribution:")
        logger.info(f"Train: {self.stats['train_chunks']} chunks ({self.stats['chunk_distribution']['train_pct']}%)")
        logger.info(f"Val:   {self.stats['val_chunks']} chunks ({self.stats['chunk_distribution']['val_pct']}%)")
        logger.info(f"Test:  {self.stats['test_chunks']} chunks ({self.stats['chunk_distribution']['test_pct']}%)")
        
        logger.info("\n🎉 Document-based split generation complete!")


def main():
    """Main function for running chunk splitting"""
    parser = argparse.ArgumentParser(description="Split chunks into train/val/test sets while preserving label distribution")
    
    parser.add_argument("--input", default="synthetic_dataset/chunks", 
                       help="Path to directory containing chunk files")
    parser.add_argument("--output", default="synthetic_dataset/splits", 
                       help="Output directory for split chunks")
    parser.add_argument("--train", type=float, default=0.7, 
                       help="Ratio for training set")
    parser.add_argument("--val", type=float, default=0.15,
                       help="Ratio for validation set")
    parser.add_argument("--test", type=float, default=0.15,
                       help="Ratio for test set")
    parser.add_argument("--move", action="store_true",
                       help="Move files instead of copying them")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create splitter
    splitter = ChunkSplitter(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        copy_mode=not args.move,
        seed=args.seed
    )
    
    # Run splitting process
    splitter.run()


if __name__ == "__main__":
    main() 