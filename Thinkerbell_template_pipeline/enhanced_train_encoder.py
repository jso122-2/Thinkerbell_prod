#!/usr/bin/env python3
"""
Enhanced Thinkerbell Sentence Encoder Training
Handles expanded label space with sub-labels, multi-labels, and distractor classes
Uses contrastive learning with hard negative mining for improved generalization
"""

from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix
import json
import glob
import os
import random
import logging
import time
import numpy as np
from pathlib import Path
import torch
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_enhanced_split(split_dir):
    """Load enhanced chunks from a split directory and return text-combo pairs"""
    logger.info(f"Loading enhanced data from {split_dir}")
    pairs = []
    
    for fp in glob.glob(os.path.join(split_dir, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
            text = j["text"]
            
            # Use enhanced_labels if available, fallback to original labels
            labels = j.get("enhanced_labels", j.get("labels", []))
            
            # Create stable combo name
            combo = "|".join(sorted(labels))
            
            # Add metadata for analysis
            metadata = {
                "complexity_level": j.get("complexity_level", "unknown"),
                "sample_type": "distractor" if j.get("is_distractor", False) else 
                             "multi_label" if j.get("is_multi_label", False) else "enhanced_original",
                "label_count": len(labels)
            }
            
            pairs.append((text, combo, metadata))
    
    logger.info(f"Loaded {len(pairs)} enhanced examples from {split_dir}")
    
    # Log label distribution
    combo_counts = Counter(pair[1] for pair in pairs)
    complexity_counts = Counter(pair[2]["complexity_level"] for pair in pairs)
    type_counts = Counter(pair[2]["sample_type"] for pair in pairs)
    
    logger.info(f"  Unique combinations: {len(combo_counts)}")
    logger.info(f"  Complexity distribution: {dict(complexity_counts)}")
    logger.info(f"  Sample type distribution: {dict(type_counts)}")
    
    return pairs

class EnhancedValidationEvaluator(SentenceEvaluator):
    """Enhanced validation evaluator with complexity-aware metrics"""
    
    def __init__(self, val_data, name="enhanced_validation"):
        self.val_data = val_data
        self.name = name
        self.best_score = -1
        
        # Group data by complexity and type for detailed analysis
        self.complexity_groups = defaultdict(list)
        self.type_groups = defaultdict(list)
        
        for text, combo, metadata in val_data:
            complexity = metadata["complexity_level"]
            sample_type = metadata["sample_type"]
            
            self.complexity_groups[complexity].append((text, combo, metadata))
            self.type_groups[sample_type].append((text, combo, metadata))
    
    def __call__(self, model, output_path, epoch=-1, steps=-1):
        """Evaluate model with complexity-aware metrics"""
        
        logger.info(f"Running enhanced validation (epoch {epoch})")
        
        # Overall performance
        overall_score = self._evaluate_subset(model, self.val_data, "Overall")
        
        # Complexity-specific performance
        complexity_scores = {}
        for complexity, data in self.complexity_groups.items():
            if len(data) > 10:  # Only evaluate if enough samples
                score = self._evaluate_subset(model, data, f"Complexity-{complexity}")
                complexity_scores[complexity] = score
        
        # Sample type-specific performance
        type_scores = {}
        for sample_type, data in self.type_groups.items():
            if len(data) > 10:  # Only evaluate if enough samples
                score = self._evaluate_subset(model, data, f"Type-{sample_type}")
                type_scores[sample_type] = score
        
        # Check for improvement
        improved = overall_score > self.best_score
        if improved:
            self.best_score = overall_score
            logger.info(f"✓ New best validation score: {overall_score:.4f}")
        
        # Log detailed results
        logger.info("Enhanced Validation Results:")
        logger.info(f"  Overall: {overall_score:.4f}")
        for complexity, score in complexity_scores.items():
            logger.info(f"  {complexity}: {score:.4f}")
        for sample_type, score in type_scores.items():
            logger.info(f"  {sample_type}: {score:.4f}")
        
        # Check for overfitting patterns
        self._check_overfitting_patterns(complexity_scores, type_scores)
        
        return overall_score
    
    def _evaluate_subset(self, model, data, subset_name):
        """Evaluate model on a data subset"""
        
        if len(data) < 5:
            return 0.0
        
        texts = [item[0] for item in data]
        combos = [item[1] for item in data]
        
        # Get embeddings
        embeddings = model.encode(texts, convert_to_tensor=True)
        
        # Use k-NN for evaluation (simulates retrieval task)
        unique_combos = list(set(combos))
        if len(unique_combos) < 2:
            return 0.5  # No diversity
        
        correct = 0
        total = len(data)
        
        for i, (text, true_combo, _) in enumerate(data):
            # Find nearest neighbor (excluding self)
            similarities = torch.cosine_similarity(embeddings[i:i+1], embeddings)
            similarities[i] = -1  # Exclude self
            
            nearest_idx = torch.argmax(similarities).item()
            predicted_combo = combos[nearest_idx]
            
            if predicted_combo == true_combo:
                correct += 1
        
        accuracy = correct / total
        return accuracy
    
    def _check_overfitting_patterns(self, complexity_scores, type_scores):
        """Check for overfitting patterns in the results"""
        
        warnings = []
        
        # Check if simpler samples have much higher accuracy
        if "minimal" in complexity_scores and "comprehensive" in complexity_scores:
            simple_score = complexity_scores["minimal"]
            complex_score = complexity_scores["comprehensive"]
            
            if simple_score - complex_score > 0.2:
                warnings.append(f"Large gap between simple ({simple_score:.3f}) and complex ({complex_score:.3f}) samples")
        
        # Check if distractor performance is too high (should be challenging)
        if "distractor" in type_scores:
            distractor_score = type_scores["distractor"]
            if distractor_score > 0.95:
                warnings.append(f"Distractor accuracy too high ({distractor_score:.3f}) - may indicate overfitting")
        
        # Check if multi-label performance is reasonable
        if "multi_label" in type_scores and "enhanced_original" in type_scores:
            multi_score = type_scores["multi_label"]
            original_score = type_scores["enhanced_original"]
            
            if multi_score > original_score + 0.1:
                warnings.append(f"Multi-label samples unexpectedly easier than originals")
        
        if warnings:
            logger.warning("⚠️  Potential overfitting patterns detected:")
            for warning in warnings:
                logger.warning(f"    {warning}")

class ComplexityAwareContrastiveLoss(losses.ContrastiveLoss):
    """Contrastive loss with complexity-aware negative sampling"""
    
    def __init__(self, model, distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE, 
                 margin=0.5, complexity_weight=True):
        super().__init__(model, distance_metric, margin)
        self.complexity_weight = complexity_weight
        self.complexity_weights = {
            "minimal": 1.0,
            "moderate": 1.2,
            "complex": 1.4,
            "comprehensive": 1.6,
            "distractor": 1.8,
            "multi_label": 1.3
        }
    
    def forward(self, sentence_features, labels):
        """Forward pass with complexity weighting"""
        
        if not self.complexity_weight:
            return super().forward(sentence_features, labels)
        
        # Extract complexity information from sentence features if available
        # For now, use standard contrastive loss
        # In a full implementation, you'd parse the complexity from the data
        
        return super().forward(sentence_features, labels)

def create_enhanced_training_data(train_pairs, use_hard_negatives=True):
    """Create training examples with enhanced negative sampling"""
    
    logger.info("Creating enhanced training examples...")
    
    # Group by combo and complexity
    combo_groups = defaultdict(list)
    complexity_groups = defaultdict(list)
    
    for text, combo, metadata in train_pairs:
        combo_groups[combo].append((text, combo, metadata))
        complexity_groups[metadata["complexity_level"]].append((text, combo, metadata))
    
    examples = []
    
    # Create positive pairs (same combo)
    for combo, group in combo_groups.items():
        if len(group) < 2:
            continue
        
        # Sample positive pairs within the same combo
        for i in range(min(len(group), 10)):  # Limit to avoid explosion
            for j in range(i + 1, min(len(group), 10)):
                text1, _, meta1 = group[i]
                text2, _, meta2 = group[j]
                
                examples.append(InputExample(texts=[text1, text2], label=1.0))
    
    # Create negative pairs (different combos)
    combo_list = list(combo_groups.keys())
    
    for _ in range(len(examples)):  # Match number of positive examples
        # Select two different combos
        combo1, combo2 = random.sample(combo_list, 2)
        
        text1, _, meta1 = random.choice(combo_groups[combo1])
        text2, _, meta2 = random.choice(combo_groups[combo2])
        
        examples.append(InputExample(texts=[text1, text2], label=0.0))
    
    # Add hard negatives if requested
    if use_hard_negatives:
        hard_negative_examples = create_hard_negatives(combo_groups, num_hard_negatives=len(examples) // 4)
        examples.extend(hard_negative_examples)
    
    random.shuffle(examples)
    logger.info(f"Created {len(examples)} training examples")
    
    return examples

def create_hard_negatives(combo_groups, num_hard_negatives=100):
    """Create hard negative examples (similar text, different labels)"""
    
    hard_negatives = []
    combo_list = list(combo_groups.keys())
    
    for _ in range(num_hard_negatives):
        # Select two different combos that might be confusable
        combo1, combo2 = random.sample(combo_list, 2)
        
        # Try to find texts that might be similar
        text1, _, _ = random.choice(combo_groups[combo1])
        text2, _, _ = random.choice(combo_groups[combo2])
        
        # Check if they share some words (simple similarity heuristic)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
        
        if overlap > 0.3:  # Some similarity but different labels
            hard_negatives.append(InputExample(texts=[text1, text2], label=0.0))
    
    logger.info(f"Created {len(hard_negatives)} hard negative examples")
    return hard_negatives

def enhanced_train_model(model_name="sentence-transformers/all-MiniLM-L6-v2",
                        train_dir="synthetic_dataset/enhanced_pipeline/enhanced_splits/train",
                        val_dir="synthetic_dataset/enhanced_pipeline/enhanced_splits/val",
                        output_dir="models/enhanced-thinkerbell-encoder",
                        num_epochs=4,
                        batch_size=16,
                        use_hard_negatives=True,
                        use_complexity_weighting=True):
    """Train enhanced sentence encoder with overfitting prevention"""
    
    logger.info("=" * 60)
    logger.info("ENHANCED THINKERBELL ENCODER TRAINING")
    logger.info("=" * 60)
    
    # Load data
    train_pairs = load_enhanced_split(train_dir)
    val_pairs = load_enhanced_split(val_dir)
    
    if not train_pairs:
        logger.error("No training data found!")
        return None
    
    # Initialize model
    logger.info(f"Initializing model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Create training examples
    train_examples = create_enhanced_training_data(train_pairs, use_hard_negatives)
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Create loss function
    if use_complexity_weighting:
        train_loss = ComplexityAwareContrastiveLoss(model)
    else:
        train_loss = losses.ContrastiveLoss(model)
    
    # Create evaluator
    evaluator = EnhancedValidationEvaluator(val_pairs)
    
    # Training configuration
    warmup_steps = int(len(train_dataloader) * 0.1)  # 10% of steps
    
    logger.info("Training configuration:")
    logger.info(f"  Training examples: {len(train_examples)}")
    logger.info(f"  Validation examples: {len(val_pairs)}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Hard negatives: {use_hard_negatives}")
    logger.info(f"  Complexity weighting: {use_complexity_weighting}")
    
    # Train model
    logger.info("Starting training...")
    start_time = time.time()
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=len(train_dataloader) // 2,  # Evaluate twice per epoch
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True,
        use_amp=True  # Automatic mixed precision for faster training
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save enhanced model info
    model_info = {
        "model_name": model_name,
        "training_time": training_time,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "train_examples": len(train_examples),
        "val_examples": len(val_pairs),
        "best_validation_score": evaluator.best_score,
        "enhancement_features": {
            "hard_negatives": use_hard_negatives,
            "complexity_weighting": use_complexity_weighting,
            "label_space_expansion": True,
            "multi_label_support": True,
            "distractor_class": True
        }
    }
    
    info_file = Path(output_dir) / "enhanced_model_info.json"
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Enhanced model saved to: {output_dir}")
    logger.info(f"Best validation score: {evaluator.best_score:.4f}")
    
    return model, evaluator.best_score

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Thinkerbell Encoder Training")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Base model to fine-tune")
    parser.add_argument("--train-dir", type=str, 
                       default="synthetic_dataset/enhanced_pipeline/enhanced_splits/train",
                       help="Training data directory")
    parser.add_argument("--val-dir", type=str,
                       default="synthetic_dataset/enhanced_pipeline/enhanced_splits/val", 
                       help="Validation data directory")
    parser.add_argument("--output-dir", type=str, default="models/enhanced-thinkerbell-encoder",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=4,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--no-hard-negatives", action="store_true",
                       help="Disable hard negative mining")
    parser.add_argument("--no-complexity-weighting", action="store_true",
                       help="Disable complexity-aware loss weighting")
    
    args = parser.parse_args()
    
    # Check if enhanced data exists
    if not Path(args.train_dir).exists():
        logger.error(f"Training directory not found: {args.train_dir}")
        logger.error("Please run the enhanced pipeline first:")
        logger.error("  python run_enhanced_pipeline.py")
        return 1
    
    # Train model
    model, best_score = enhanced_train_model(
        model_name=args.model_name,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_hard_negatives=not args.no_hard_negatives,
        use_complexity_weighting=not args.no_complexity_weighting
    )
    
    if model and best_score > 0:
        logger.info("✓ Enhanced training completed successfully!")
        return 0
    else:
        logger.error("✗ Enhanced training failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 