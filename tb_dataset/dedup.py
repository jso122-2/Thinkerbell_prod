"""
Deduplication module for detecting exact and near-duplicate samples.

Provides MD5-based exact deduplication and cosine similarity-based near-duplicate
detection using sentence transformers. Handles removal favoring newer samples
and higher coherence scores.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import json
import numpy as np
from collections import defaultdict

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm function
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Sentence transformers for near-duplicate detection
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Cosine similarity
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None


logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Detects exact and near-duplicate samples across datasets.
    
    Uses MD5 hashing for exact duplicates and sentence transformer embeddings
    with cosine similarity for near-duplicates.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.95,
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 device: str = None):
        """
        Initialize duplicate detector.
        
        Args:
            similarity_threshold: Cosine similarity threshold for near-duplicates
            model_name: Sentence transformer model name
            batch_size: Batch size for embedding generation
            device: Device to use for computation ('cpu', 'cuda', etc.)
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        
        # Initialize model
        self.model = None
        self._init_model()
        
        # Tracking structures
        self.exact_duplicates = {}  # hash -> [sample_ids]
        self.near_duplicates = []   # [(sample_id1, sample_id2, similarity)]
        self.processed_samples = {}  # sample_id -> sample_data
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'exact_duplicates_found': 0,
            'near_duplicates_found': 0,
            'samples_removed': 0,
            'exact_duplicate_groups': 0,
            'near_duplicate_pairs': 0
        }
    
    def _init_model(self, offline_mode: bool = False):
        """Initialize the sentence transformer model with robust offline support."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available. Near-duplicate detection disabled.")
            return
        
        from .utils import get_cached_sentence_transformer, check_model_availability
        
        # Check model availability first
        if check_model_availability(self.model_name, "sentence_transformer"):
            logger.info(f"📋 SentenceTransformer {self.model_name} found in cache")
        else:
            logger.warning(f"📋 SentenceTransformer {self.model_name} not found in cache")
        
        self.model = get_cached_sentence_transformer(
            model_name=self.model_name, 
            device=self.device,
            offline_mode=offline_mode
        )
        if self.model is None:
            logger.warning("❌ Near-duplicate detection will be skipped due to model failure")
            if not offline_mode:
                logger.info("💡 Try downloading first: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\"")
        else:
            logger.info("✅ Near-duplicate detection model initialized successfully")
    
    def load_samples_from_directory(self, samples_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load all samples from a directory structure.
        
        Args:
            samples_dir: Directory containing batch subdirectories with JSON samples
            
        Returns:
            Dictionary mapping sample_id to sample data
        """
        samples = {}
        
        # Find all JSON files in batch directories
        all_sample_files = []
        for batch_dir in samples_dir.glob("batch_*"):
            if not batch_dir.is_dir():
                continue
            all_sample_files.extend(batch_dir.glob("*.json"))
        
        # Add progress bar for loading samples
        file_progress = tqdm(all_sample_files, desc="Loading samples", unit="file") if TQDM_AVAILABLE else all_sample_files
        
        for sample_file in file_progress:
            try:
                if TQDM_AVAILABLE:
                    file_progress.set_description(f"Loading {sample_file.name}")
                
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                
                sample_id = sample_data.get('sample_id')
                if sample_id:
                    samples[sample_id] = sample_data
                    samples[sample_id]['_file_path'] = str(sample_file)
                else:
                    logger.warning(f"Sample missing sample_id: {sample_file}")
                    
            except Exception as e:
                logger.error(f"Failed to load sample {sample_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(samples)} samples from {samples_dir}")
        return samples
    
    def detect_exact_duplicates(self, samples: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Detect exact duplicates using MD5 hash of raw text.
        
        Args:
            samples: Dictionary of sample_id -> sample_data
            
        Returns:
            Dictionary mapping hash -> list of sample_ids with that hash
        """
        hash_to_samples = defaultdict(list)
        
        for sample_id, sample_data in samples.items():
            # Get raw text
            raw_text = sample_data.get('raw_input', {}).get('text', '')
            if not raw_text:
                logger.warning(f"Sample {sample_id} missing raw text")
                continue
            
            # Calculate MD5 hash
            text_hash = hashlib.md5(raw_text.encode('utf-8')).hexdigest()
            hash_to_samples[text_hash].append(sample_id)
        
        # Filter to only duplicates (groups with > 1 sample)
        exact_duplicates = {h: ids for h, ids in hash_to_samples.items() if len(ids) > 1}
        
        self.stats['exact_duplicate_groups'] = len(exact_duplicates)
        self.stats['exact_duplicates_found'] = sum(len(ids) - 1 for ids in exact_duplicates.values())
        
        logger.info(f"Found {len(exact_duplicates)} exact duplicate groups affecting {self.stats['exact_duplicates_found']} samples")
        
        return exact_duplicates
    
    def detect_near_duplicates(self, samples: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, float]]:
        """
        Detect near-duplicates using sentence transformer embeddings.
        
        Args:
            samples: Dictionary of sample_id -> sample_data
            
        Returns:
            List of (sample_id1, sample_id2, similarity) tuples
        """
        if not self.model:
            logger.warning("Sentence transformer not available. Skipping near-duplicate detection.")
            return []
        
        # Extract texts and sample IDs
        sample_ids = []
        texts = []
        
        for sample_id, sample_data in samples.items():
            raw_text = sample_data.get('raw_input', {}).get('text', '')
            if raw_text:
                sample_ids.append(sample_id)
                texts.append(raw_text)
        
        if len(texts) < 2:
            logger.info("Not enough samples for near-duplicate detection")
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} samples...")
        
        # Generate embeddings in batches with progress bar
        embeddings = []
        batch_ranges = list(range(0, len(texts), self.batch_size))
        batch_progress = tqdm(batch_ranges, desc="Generating embeddings", unit="batch") if TQDM_AVAILABLE else batch_ranges
        
        for i in batch_progress:
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            
            if TQDM_AVAILABLE:
                batch_progress.set_description(f"Embeddings: {len(embeddings)}/{len(texts)}")
        
        embeddings = np.array(embeddings)
        
        # Compute pairwise similarities
        logger.info("Computing pairwise similarities...")
        similarities = cosine_similarity(embeddings)
        
        # Find near-duplicates with progress bar
        near_duplicates = []
        total_comparisons = (len(sample_ids) * (len(sample_ids) - 1)) // 2
        
        comparison_progress = tqdm(total=total_comparisons, desc="Finding near-duplicates", unit="comparison") if TQDM_AVAILABLE else None
        
        for i in range(len(sample_ids)):
            for j in range(i + 1, len(sample_ids)):
                similarity = similarities[i, j]
                
                if similarity >= self.similarity_threshold:
                    near_duplicates.append((sample_ids[i], sample_ids[j], similarity))
                
                if comparison_progress:
                    comparison_progress.update(1)
                    if len(near_duplicates) > 0:
                        comparison_progress.set_description(f"Found {len(near_duplicates)} near-duplicates")
        
        if comparison_progress:
            comparison_progress.close()
        
        self.stats['near_duplicate_pairs'] = len(near_duplicates)
        self.stats['near_duplicates_found'] = len(set().union(*[(id1, id2) for id1, id2, _ in near_duplicates]))
        
        logger.info(f"Found {len(near_duplicates)} near-duplicate pairs affecting {self.stats['near_duplicates_found']} samples")
        
        return near_duplicates
    
    def resolve_duplicates(self, 
                          samples: Dict[str, Dict[str, Any]],
                          exact_duplicates: Dict[str, List[str]],
                          near_duplicates: List[Tuple[str, str, float]]) -> Tuple[Set[str], Dict[str, str]]:
        """
        Resolve duplicates by selecting the best sample from each duplicate group.
        
        Args:
            samples: Dictionary of sample_id -> sample_data
            exact_duplicates: Exact duplicate groups
            near_duplicates: Near-duplicate pairs
            
        Returns:
            Tuple of (samples_to_remove, removal_reasons)
        """
        samples_to_remove = set()
        removal_reasons = {}
        
        # Resolve exact duplicates
        for hash_key, duplicate_ids in exact_duplicates.items():
            if len(duplicate_ids) <= 1:
                continue
            
            # Select best sample based on criteria
            best_sample_id = self._select_best_sample(samples, duplicate_ids)
            
            # Mark others for removal
            for sample_id in duplicate_ids:
                if sample_id != best_sample_id:
                    samples_to_remove.add(sample_id)
                    removal_reasons[sample_id] = f"exact_duplicate_of_{best_sample_id}"
        
        # Build near-duplicate groups
        near_duplicate_groups = self._build_near_duplicate_groups(near_duplicates)
        
        # Resolve near-duplicate groups
        for group in near_duplicate_groups:
            if len(group) <= 1:
                continue
            
            # Filter out already removed samples
            remaining_group = [sid for sid in group if sid not in samples_to_remove]
            
            if len(remaining_group) <= 1:
                continue
            
            # Select best sample
            best_sample_id = self._select_best_sample(samples, remaining_group)
            
            # Mark others for removal
            for sample_id in remaining_group:
                if sample_id != best_sample_id:
                    samples_to_remove.add(sample_id)
                    removal_reasons[sample_id] = f"near_duplicate_of_{best_sample_id}"
        
        self.stats['samples_removed'] = len(samples_to_remove)
        
        logger.info(f"Selected {len(samples_to_remove)} samples for removal")
        
        return samples_to_remove, removal_reasons
    
    def _select_best_sample(self, samples: Dict[str, Dict[str, Any]], candidate_ids: List[str]) -> str:
        """
        Select the best sample from a group of duplicates.
        
        Selection criteria (in order):
        1. Higher semantic coherence score
        2. Not OOD (prefer in-distribution samples)
        3. Newer sample (lexicographically later sample_id)
        
        Args:
            samples: Dictionary of sample_id -> sample_data
            candidate_ids: List of candidate sample IDs
            
        Returns:
            ID of the best sample
        """
        def score_sample(sample_id: str) -> Tuple[float, bool, str]:
            sample_data = samples[sample_id]
            
            # Coherence score (higher is better)
            coherence = sample_data.get('validation', {}).get('semantic_coherence', 0.0)
            
            # Prefer non-OOD samples (False < True, so negate is_ood)
            is_ood = sample_data.get('is_ood', False)
            
            # Sample ID as tiebreaker (later is better)
            sample_id_str = sample_id
            
            return (coherence, not is_ood, sample_id_str)
        
        # Sort by score and return the best
        best_sample_id = max(candidate_ids, key=score_sample)
        return best_sample_id
    
    def _build_near_duplicate_groups(self, near_duplicates: List[Tuple[str, str, float]]) -> List[List[str]]:
        """
        Build connected components from near-duplicate pairs.
        
        Args:
            near_duplicates: List of (sample_id1, sample_id2, similarity) tuples
            
        Returns:
            List of groups, where each group is a list of connected sample IDs
        """
        if not near_duplicates:
            return []
        
        # Build adjacency list
        adjacency = defaultdict(set)
        for id1, id2, _ in near_duplicates:
            adjacency[id1].add(id2)
            adjacency[id2].add(id1)
        
        # Find connected components using DFS
        visited = set()
        groups = []
        
        def dfs(node: str, current_group: List[str]):
            if node in visited:
                return
            visited.add(node)
            current_group.append(node)
            
            for neighbor in adjacency[node]:
                dfs(neighbor, current_group)
        
        for node in adjacency:
            if node not in visited:
                group = []
                dfs(node, group)
                if len(group) > 1:
                    groups.append(group)
        
        return groups
    
    def process_directory(self, samples_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Set[str], Dict[str, str]]:
        """
        Complete deduplication process for a directory of samples.
        
        Args:
            samples_dir: Directory containing sample batch subdirectories
            
        Returns:
            Tuple of (all_samples, samples_to_remove, removal_reasons)
        """
        logger.info(f"Starting deduplication process for {samples_dir}")
        
        # Load all samples
        samples = self.load_samples_from_directory(samples_dir)
        self.stats['total_samples'] = len(samples)
        
        if not samples:
            logger.warning("No samples found to process")
            return {}, set(), {}
        
        # Detect exact duplicates
        logger.info("Detecting exact duplicates...")
        exact_duplicates = self.detect_exact_duplicates(samples)
        
        # Detect near duplicates
        logger.info("Detecting near duplicates...")
        near_duplicates = self.detect_near_duplicates(samples)
        
        # Resolve duplicates
        logger.info("Resolving duplicates...")
        samples_to_remove, removal_reasons = self.resolve_duplicates(samples, exact_duplicates, near_duplicates)
        
        # Store results
        self.exact_duplicates = exact_duplicates
        self.near_duplicates = near_duplicates
        self.processed_samples = samples
        
        logger.info(f"Deduplication complete. Removing {len(samples_to_remove)} of {len(samples)} samples")
        
        return samples, samples_to_remove, removal_reasons
    
    def save_deduplication_report(self, output_dir: Path, samples_to_remove: Set[str], removal_reasons: Dict[str, str]):
        """
        Save detailed deduplication report.
        
        Args:
            output_dir: Directory to save report
            samples_to_remove: Set of sample IDs to remove
            removal_reasons: Dictionary mapping sample_id to removal reason
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save exact duplicates report
        exact_dupes_file = output_dir / 'exact_duplicates.json'
        with open(exact_dupes_file, 'w', encoding='utf-8') as f:
            json.dump(self.exact_duplicates, f, ensure_ascii=False, indent=2)
        
        # Save near duplicates report
        near_dupes_file = output_dir / 'near_duplicates.json'
        near_dupes_data = [
            {
                'sample_id1': id1,
                'sample_id2': id2,
                'similarity': float(sim)
            }
            for id1, id2, sim in self.near_duplicates
        ]
        with open(near_dupes_file, 'w', encoding='utf-8') as f:
            json.dump(near_dupes_data, f, ensure_ascii=False, indent=2)
        
        # Save removal decisions
        removals_file = output_dir / 'removal_decisions.json'
        removal_data = {
            'samples_to_remove': list(samples_to_remove),
            'removal_reasons': removal_reasons,
            'statistics': self.stats
        }
        with open(removals_file, 'w', encoding='utf-8') as f:
            json.dump(removal_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved deduplication reports to {output_dir}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return self.stats.copy()
    
    def get_clean_samples(self, samples: Dict[str, Dict[str, Any]], samples_to_remove: Set[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get clean samples with duplicates removed.
        
        Args:
            samples: All samples
            samples_to_remove: Sample IDs to remove
            
        Returns:
            Clean samples dictionary
        """
        return {
            sample_id: sample_data 
            for sample_id, sample_data in samples.items() 
            if sample_id not in samples_to_remove
        } 