#!/usr/bin/env python3
"""
Thinkerbell Sentence Encoder Training Pipeline

Complete pipeline for training a fresh sentence encoder model with dataset preparation,
training, evaluation, and archival.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random

# ML/Training imports
try:
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import numpy as np
    from sklearn.metrics import classification_report, accuracy_score
    from collections import Counter, defaultdict
    HAS_ML_DEPS = True
except ImportError as e:
    HAS_ML_DEPS = False
    print(f"⚠️ ML dependencies not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for Thinkerbell sentence encoder"""
    
    def __init__(self, 
                 dataset_dir: str = "synthetic_dataset/chunks",
                 models_dir: str = "models",
                 archive_dir: str = "archive",
                 target_samples: int = 6000,
                 model_name: str = "all-MiniLM-L6-v2",
                 epochs: int = 3,
                 batch_size: int = 64,
                 learning_rate: float = 2e-5,
                 balanced: bool = False,
                 save_best: bool = True):
        
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path(models_dir)
        self.archive_dir = Path(archive_dir)
        self.target_samples = target_samples
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.balanced = balanced
        self.save_best = save_best
        
        # Training configuration
        self.config = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "balanced_sampling": balanced,
            "target_samples": target_samples,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "timestamp": datetime.now().isoformat()
        }
        
        # Results tracking
        self.results = {
            "training_config": self.config,
            "dataset_stats": {},
            "training_metrics": {},
            "evaluation_results": {},
            "best_checkpoint": None
        }
        
        # Paths
        self.splits_dir = Path("synthetic_dataset/splits")
        self.best_model_dir = self.models_dir / "thinkerbell-encoder-best"
        
    def purge_old_artifacts(self) -> None:
        """Step 1: Purge old training artifacts"""
        logger.info("🗑️ Step 1: Purging old artifacts...")
        
        # Create archive directory if needed
        archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_archive_dir = self.archive_dir / f"run_{archive_timestamp}"
        run_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Archive existing models
        if self.models_dir.exists():
            models_to_archive = [d for d in self.models_dir.iterdir() 
                               if d.is_dir() and d.name != ".gitkeep"]
            
            if models_to_archive:
                models_archive_dir = run_archive_dir / "models"
                models_archive_dir.mkdir(exist_ok=True)
                
                for model_dir in models_to_archive:
                    archive_path = models_archive_dir / model_dir.name
                    logger.info(f"  Archiving {model_dir} -> {archive_path}")
                    shutil.move(str(model_dir), str(archive_path))
            
            # Clean models directory but keep .gitkeep
            for item in self.models_dir.iterdir():
                if item.name != ".gitkeep":
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
        else:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean cache and output directories
        cache_dirs = [".cache", "output", "checkpoints"]
        for cache_dir in cache_dirs:
            if Path(cache_dir).exists():
                logger.info(f"  Cleaning {cache_dir}/")
                shutil.rmtree(cache_dir)
        
        # Archive old split stats
        old_split_stats = self.splits_dir / "split_stats.json"
        if old_split_stats.exists():
            archive_path = run_archive_dir / "split_stats.json"
            logger.info(f"  Archiving split_stats.json -> {archive_path}")
            shutil.copy2(old_split_stats, archive_path)
            old_split_stats.unlink()
        
        logger.info("✅ Artifact purging complete")
        self.results["archive_dir"] = str(run_archive_dir)
    
    def prepare_dataset(self) -> None:
        """Step 2: Prepare and validate dataset"""
        logger.info("📊 Step 2: Preparing dataset...")
        
        # Validate dataset exists and has correct number of samples
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        chunk_files = list(self.dataset_dir.glob("**/*.json"))
        actual_count = len(chunk_files)
        
        logger.info(f"  Found {actual_count} chunk files")
        
        if actual_count != self.target_samples:
            logger.warning(f"  Expected {self.target_samples} samples, found {actual_count}")
            if actual_count < self.target_samples:
                logger.info(f"  Will work with {actual_count} samples (less than target)")
            else:
                logger.info(f"  Will use first {self.target_samples} samples")
                # Note: We could implement sampling logic here if needed
        
        # Run split_chunks.py to create train/val/test splits
        logger.info("  Running split_chunks.py...")
        split_cmd = [
            sys.executable, "split_chunks.py",
            "--input", str(self.dataset_dir),
            "--output", str(self.splits_dir),
            "--train", str(self.config["train_ratio"]),
            "--val", str(self.config["val_ratio"]),
            "--test", str(self.config["test_ratio"])
        ]
        
        result = subprocess.run(split_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Split command failed: {result.stderr}")
            raise RuntimeError("Dataset splitting failed")
        
        logger.info("  Dataset splitting complete")
        
        # Load split statistics
        split_stats_path = self.splits_dir / "split_stats.json"
        if split_stats_path.exists():
            with open(split_stats_path, 'r') as f:
                split_stats = json.load(f)
            self.results["dataset_stats"] = split_stats
            
            logger.info(f"  Train: {split_stats['train_chunks']} chunks")
            logger.info(f"  Val: {split_stats['val_chunks']} chunks") 
            logger.info(f"  Test: {split_stats['test_chunks']} chunks")
        
        logger.info("✅ Dataset preparation complete")
    
    def load_chunks(self, split: str) -> List[Dict]:
        """Load chunks from a specific split"""
        split_dir = self.splits_dir / split
        chunk_files = list(split_dir.glob("*.json"))
        
        chunks = []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to load {chunk_file}: {e}")
        
        return chunks
    
    def create_training_examples(self, chunks: List[Dict]) -> List[InputExample]:
        """Convert chunks to sentence transformer training examples"""
        examples = []
        
        # Group chunks by labels for positive/negative sampling
        label_groups = defaultdict(list)
        for chunk in chunks:
            labels_key = tuple(sorted(chunk.get('labels', [])))
            label_groups[labels_key].append(chunk)
        
        # Create positive pairs (same labels) and negative pairs (different labels)
        for label_combo, label_chunks in label_groups.items():
            if len(label_chunks) < 2:
                continue
                
            # Create positive pairs within same label group
            for i in range(len(label_chunks)):
                for j in range(i + 1, min(i + 3, len(label_chunks))):  # Limit pairs per chunk
                    examples.append(InputExample(
                        texts=[label_chunks[i]['text'], label_chunks[j]['text']],
                        label=1.0
                    ))
            
            # Create negative pairs with different label groups
            other_labels = [k for k in label_groups.keys() if k != label_combo]
            if other_labels:
                selected_other = random.choice(other_labels)
                other_chunks = label_groups[selected_other]
                
                for chunk in label_chunks[:3]:  # Limit negative pairs
                    neg_chunk = random.choice(other_chunks)
                    examples.append(InputExample(
                        texts=[chunk['text'], neg_chunk['text']],
                        label=0.0
                    ))
        
        logger.info(f"  Created {len(examples)} training examples")
        return examples
    
    def create_balanced_sampler(self, examples: List[InputExample]) -> Optional[WeightedRandomSampler]:
        """Create balanced sampler for training"""
        if not self.balanced:
            return None
        
        # Count label distribution
        labels = [example.label for example in examples]
        label_counts = Counter(labels)
        
        # Calculate weights (inverse frequency)
        weights = []
        for label in labels:
            weight = 1.0 / label_counts[label]
            weights.append(weight)
        
        return WeightedRandomSampler(weights, len(weights))
    
    def train_model(self) -> None:
        """Step 3: Train the sentence encoder model"""
        logger.info("🚀 Step 3: Training sentence encoder...")
        
        if not HAS_ML_DEPS:
            raise RuntimeError("ML dependencies not available for training")
        
        # Load training data
        train_chunks = self.load_chunks("train")
        val_chunks = self.load_chunks("val")
        
        logger.info(f"  Loaded {len(train_chunks)} training chunks")
        logger.info(f"  Loaded {len(val_chunks)} validation chunks")
        
        # Create training examples
        train_examples = self.create_training_examples(train_chunks)
        
        # Initialize model
        logger.info(f"  Loading base model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Configure loss function
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # Set up validation evaluator
        val_examples = self.create_training_examples(val_chunks)
        val_texts = [example.texts for example in val_examples[:100]]  # Limit for speed
        val_labels = [example.label for example in val_examples[:100]]
        
        # Create corpus and queries for IR evaluation
        corpus = {str(i): text[1] for i, text in enumerate(val_texts)}
        queries = {str(i): text[0] for i, text in enumerate(val_texts)}
        relevant_docs = {str(i): [str(i)] for i in range(len(val_texts))}
        
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="validation"
        )
        
        # Training configuration
        warmup_steps = int(len(train_dataloader) * 0.1)
        output_path = str(self.best_model_dir) if self.save_best else None
        
        logger.info(f"  Training for {self.epochs} epochs...")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        
        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=len(train_dataloader) // 2,
            output_path=output_path,
            save_best_model=self.save_best,
            show_progress_bar=True
        )
        
        # Save training results
        self.results["training_metrics"] = {
            "total_examples": len(train_examples),
            "epochs_completed": self.epochs,
            "warmup_steps": warmup_steps,
            "model_saved_to": str(self.best_model_dir) if output_path else None
        }
        
        logger.info("✅ Model training complete")
    
    def evaluate_model(self) -> None:
        """Step 4: Evaluate the trained model"""
        logger.info("📈 Step 4: Evaluating trained model...")
        
        if not HAS_ML_DEPS:
            logger.warning("ML dependencies not available, skipping evaluation")
            return
        
        if not self.best_model_dir.exists():
            logger.warning("Best model directory not found, skipping evaluation")
            return
        
        # Load the trained model
        model = SentenceTransformer(str(self.best_model_dir))
        
        # Load test data
        test_chunks = self.load_chunks("test")
        logger.info(f"  Loaded {len(test_chunks)} test chunks")
        
        # Create test examples
        test_examples = self.create_training_examples(test_chunks)
        
        # Compute embeddings for evaluation
        texts = [example.texts[0] for example in test_examples[:500]]  # Limit for speed
        embeddings = model.encode(texts, convert_to_tensor=True)
        
        # Basic similarity evaluation
        similarities = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1), 
            embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Compute metrics
        recall_at_3 = self._compute_recall_at_k(similarities, k=3)
        recall_at_5 = self._compute_recall_at_k(similarities, k=5)
        mrr = self._compute_mrr(similarities)
        
        eval_results = {
            "test_samples": len(test_chunks),
            "test_examples_evaluated": len(texts),
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "mrr": mrr,
            "model_dimension": model.get_sentence_embedding_dimension()
        }
        
        # Save evaluation results
        eval_results_path = self.best_model_dir / "eval_results.json"
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        self.results["evaluation_results"] = eval_results
        
        logger.info(f"  Recall@3: {recall_at_3:.4f}")
        logger.info(f"  Recall@5: {recall_at_5:.4f}")
        logger.info(f"  MRR: {mrr:.4f}")
        logger.info(f"  Model dimension: {eval_results['model_dimension']}")
        logger.info(f"  Results saved to: {eval_results_path}")
        
        logger.info("✅ Model evaluation complete")
    
    def _compute_recall_at_k(self, similarities: torch.Tensor, k: int) -> float:
        """Compute Recall@K metric"""
        # For each query, get top-k most similar documents
        _, top_k_indices = torch.topk(similarities, k, dim=1)
        
        # Check if the correct document (same index) is in top-k
        correct_predictions = 0
        for i in range(similarities.size(0)):
            if i in top_k_indices[i]:
                correct_predictions += 1
        
        return correct_predictions / similarities.size(0)
    
    def _compute_mrr(self, similarities: torch.Tensor) -> float:
        """Compute Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for i in range(similarities.size(0)):
            # Get ranking of correct document
            _, ranked_indices = torch.sort(similarities[i], descending=True)
            correct_rank = (ranked_indices == i).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / correct_rank)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    def archive_dataset(self) -> None:
        """Step 5: Archive the dataset and training configuration"""
        logger.info("📦 Step 5: Archiving dataset and configuration...")
        
        # Create dataset archive directory
        archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_archive_dir = self.archive_dir / "datasets" / f"run_{archive_timestamp}"
        dataset_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Archive dataset splits
        if self.splits_dir.exists():
            archive_splits_dir = dataset_archive_dir / "splits"
            shutil.copytree(self.splits_dir, archive_splits_dir)
            logger.info(f"  Archived dataset splits to: {archive_splits_dir}")
        
        # Save training log
        training_log_path = self.best_model_dir / "training_log.json"
        if self.best_model_dir.exists():
            with open(training_log_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"  Saved training log to: {training_log_path}")
        
        # Archive training configuration
        config_archive_path = dataset_archive_dir / "training_config.json"
        with open(config_archive_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"  Archived training config to: {config_archive_path}")
        
        logger.info("✅ Archival complete")
    
    def run_pipeline(self) -> None:
        """Run the complete training pipeline"""
        logger.info("🎯 Starting Thinkerbell Sentence Encoder Training Pipeline")
        logger.info("=" * 80)
        
        try:
            # Step 1: Purge old artifacts
            self.purge_old_artifacts()
            
            # Step 2: Prepare dataset
            self.prepare_dataset()
            
            # Step 3: Train model
            self.train_model()
            
            # Step 4: Evaluate model
            self.evaluate_model()
            
            # Step 5: Archive everything
            self.archive_dataset()
            
            logger.info("=" * 80)
            logger.info("🎉 Training pipeline completed successfully!")
            
            # Print summary
            if self.results["evaluation_results"]:
                eval_results = self.results["evaluation_results"]
                logger.info(f"📊 Final Results:")
                logger.info(f"  Model: {self.model_name}")
                logger.info(f"  Test samples: {eval_results.get('test_samples', 'N/A')}")
                logger.info(f"  Recall@3: {eval_results.get('recall_at_3', 'N/A'):.4f}")
                logger.info(f"  Recall@5: {eval_results.get('recall_at_5', 'N/A'):.4f}")
                logger.info(f"  MRR: {eval_results.get('mrr', 'N/A'):.4f}")
                logger.info(f"  Model saved to: {self.best_model_dir}")
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(
        description="Thinkerbell Sentence Encoder Training Pipeline"
    )
    parser.add_argument(
        "--dataset", 
        default="synthetic_dataset/chunks",
        help="Path to dataset chunks directory"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--model", 
        default="all-MiniLM-L6-v2",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--target-samples", 
        type=int, 
        default=6000,
        help="Expected number of training samples"
    )
    parser.add_argument(
        "--balanced", 
        action="store_true",
        help="Use balanced sampling to upsample rare label combinations"
    )
    parser.add_argument(
        "--save-best", 
        action="store_true", 
        default=True,
        help="Save the best model checkpoint"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        dataset_dir=args.dataset,
        target_samples=args.target_samples,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        balanced=args.balanced,
        save_best=args.save_best
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 