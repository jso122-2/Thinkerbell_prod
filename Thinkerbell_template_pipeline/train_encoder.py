#!/usr/bin/env python3
"""
Thinkerbell Sentence Encoder Training
Uses contrastive learning to train an encoder that maps chunks to their label combinations
Enhanced with semantic improvements to prevent surface pattern memorization
"""

from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
import json
import glob
import os
import random
import re
import logging
import time
import numpy as np
from pathlib import Path
import torch
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenDropoutProcessor:
    """
    Applies token-level dropout or span masking to input text to prevent surface pattern memorization
    """
    def __init__(self, dropout_rate=0.125, mask_token="[MASK]"):
        self.dropout_rate = dropout_rate
        self.mask_token = mask_token
        
    def apply_token_dropout(self, text):
        """Apply random token dropout to text"""
        # Simple tokenization on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        if len(tokens) <= 2:  # Don't mask very short sequences
            return text
            
        # Determine number of tokens to mask
        num_to_mask = max(1, int(len(tokens) * self.dropout_rate))
        
        # Randomly select positions to mask
        mask_positions = random.sample(range(len(tokens)), min(num_to_mask, len(tokens)))
        
        # Apply masking
        masked_tokens = []
        for i, token in enumerate(tokens):
            if i in mask_positions:
                masked_tokens.append(self.mask_token)
            else:
                masked_tokens.append(token)
        
        # Reconstruct text (simple approach - may not preserve exact spacing)
        result = ""
        for i, token in enumerate(masked_tokens):
            if i > 0 and re.match(r'\w', token) and re.match(r'\w', masked_tokens[i-1]):
                result += " "
            result += token
            
        return result

class GroupAwareSampler(Sampler):
    """
    Custom sampler that ensures no two samples from the same document/template appear in the same batch
    """
    def __init__(self, dataset, batch_size, group_key_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_key_fn = group_key_fn
        
        # Group samples by their group key (sample_id)
        self.groups = defaultdict(list)
        for idx in range(len(dataset)):
            group_key = self.group_key_fn(dataset[idx])
            self.groups[group_key].append(idx)
        
        self.group_keys = list(self.groups.keys())
        logger.info(f"GroupAwareSampler: {len(self.group_keys)} unique groups, {len(dataset)} total samples")
        
    def __iter__(self):
        """Generate batches ensuring no group overlap within batch"""
        shuffled_groups = self.group_keys.copy()
        random.shuffle(shuffled_groups)
        
        batch = []
        used_groups = set()
        
        for group_key in shuffled_groups:
            if group_key in used_groups:
                continue
                
            # Add one sample from this group
            group_samples = self.groups[group_key]
            sample_idx = random.choice(group_samples)
            batch.append(sample_idx)
            used_groups.add(group_key)
            
            # If batch is full, yield it and start new batch
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
                used_groups = set()
        
        # Yield remaining batch if any
        if batch:
            yield batch
    
    def __len__(self):
        return len(self.group_keys) // self.batch_size

def load_split(split_dir):
    """Load all chunks from a split directory and return text-combo pairs with metadata"""
    logger.info(f"Loading data from {split_dir}")
    pairs = []
    for fp in glob.glob(os.path.join(split_dir, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
            text = j["text"]                       # chunk text
            labels = j["labels"]                   # list like ["brand","client",...]
            combo = "|".join(sorted(labels))       # stable combo name
            sample_id = j.get("sample_id", "unknown")  # for group-aware batching
            pairs.append((text, combo, sample_id))
    
    logger.info(f"Loaded {len(pairs)} examples from {split_dir}")
    return pairs

class RecallAtKEvaluator(SentenceEvaluator):
    """
    Enhanced evaluator that tracks Recall@3 for early stopping and ranking quality assessment
    """
    def __init__(self, val_pairs, model_name, best_model_path, recall_k=3):
        self.val_pairs = val_pairs
        self.model_name = model_name
        self.best_model_path = best_model_path
        self.recall_k = recall_k
        self.best_recall_at_k = -1.0
        self.best_accuracy = -1.0
        
        # Extract unique combos and texts
        self.texts = [t for t, _, _ in val_pairs]
        self.gold_combos = [c for _, c, _ in val_pairs]
        self.unique_combos = sorted(list(set(self.gold_combos)))
        
        logger.info(f"Created Recall@{recall_k} evaluator with {len(val_pairs)} examples and {len(self.unique_combos)} unique combos")
    
    def __call__(self, model, output_path=None, epoch=None, steps=None):
        """
        Evaluate the model on validation data with both accuracy and Recall@K
        """
        # Encode validation texts
        text_embeddings = model.encode(self.texts, convert_to_tensor=True, show_progress_bar=False)
        
        # Encode all unique label combinations
        combo_embeddings = model.encode(self.unique_combos, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate cosine similarities
        cos_sim = torch.nn.functional.cosine_similarity(
            text_embeddings.unsqueeze(1).expand(-1, len(self.unique_combos), -1),
            combo_embeddings.unsqueeze(0).expand(len(self.texts), -1, -1),
            dim=2
        )
        
        # Calculate accuracy (top-1)
        pred_indices = torch.argmax(cos_sim, dim=1).cpu().numpy()
        pred_combos = [self.unique_combos[idx] for idx in pred_indices]
        accuracy = sum(1 for pred, gold in zip(pred_combos, self.gold_combos) if pred == gold) / len(self.gold_combos)
        
        # Calculate Recall@K
        _, top_k_indices = torch.topk(cos_sim, k=min(self.recall_k, len(self.unique_combos)), dim=1)
        top_k_indices = top_k_indices.cpu().numpy()
        
        recall_at_k_hits = 0
        for i, gold_combo in enumerate(self.gold_combos):
            gold_idx = self.unique_combos.index(gold_combo)
            if gold_idx in top_k_indices[i]:
                recall_at_k_hits += 1
        
        recall_at_k = recall_at_k_hits / len(self.gold_combos)
        
        logger.info(f"Validation metrics (epoch={epoch}, steps={steps}): "
                   f"Accuracy={accuracy:.4f}, Recall@{self.recall_k}={recall_at_k:.4f}")
        
        # Save best model based on Recall@K (prioritizing ranking quality)
        if recall_at_k > self.best_recall_at_k:
            self.best_recall_at_k = recall_at_k
            self.best_accuracy = accuracy
            logger.info(f"New best Recall@{self.recall_k}: {recall_at_k:.4f} - Saving model to {self.best_model_path}")
            model.save(self.best_model_path)
            
            # Save a record of the best scores
            with open(os.path.join(self.best_model_path, "best_score.txt"), "w") as f:
                f.write(f"Best Recall@{self.recall_k}: {recall_at_k:.4f}\n"
                       f"Accuracy at best Recall@{self.recall_k}: {accuracy:.4f}\n"
                       f"Epoch: {epoch}\nSteps: {steps}")
        
        # Return Recall@K as primary metric for early stopping
        return recall_at_k

class ValidationEvaluator(SentenceEvaluator):
    """
    Legacy validation evaluator - kept for compatibility but RecallAtKEvaluator is preferred
    """
    def __init__(self, val_pairs, model_name, best_model_path):
        self.val_pairs = val_pairs
        self.model_name = model_name
        self.best_model_path = best_model_path
        self.best_score = -1.0
        
        # Extract unique combos - adjust for new tuple format
        self.texts = [t for t, _, _ in val_pairs]
        self.gold_combos = [c for _, c, _ in val_pairs]
        self.unique_combos = sorted(list(set(self.gold_combos)))
        
        logger.info(f"Created validation evaluator with {len(val_pairs)} examples and {len(self.unique_combos)} unique combos")
    
    def __call__(self, model, output_path=None, epoch=None, steps=None):
        """
        Evaluate the model on validation data
        """
        # Encode validation texts
        text_embeddings = model.encode(self.texts, convert_to_tensor=True, show_progress_bar=False)
        
        # Encode all unique label combinations
        combo_embeddings = model.encode(self.unique_combos, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate cosine similarities
        cos_sim = torch.nn.functional.cosine_similarity(
            text_embeddings.unsqueeze(1).expand(-1, len(self.unique_combos), -1),
            combo_embeddings.unsqueeze(0).expand(len(self.texts), -1, -1),
            dim=2
        )
        
        # Get predicted combos (highest similarity)
        pred_indices = torch.argmax(cos_sim, dim=1).cpu().numpy()
        pred_combos = [self.unique_combos[idx] for idx in pred_indices]
        
        # Calculate accuracy
        correct = sum(1 for pred, gold in zip(pred_combos, self.gold_combos) if pred == gold)
        accuracy = correct / len(self.gold_combos)
        
        logger.info(f"Validation accuracy (epoch={epoch}, steps={steps}): {accuracy:.4f}")
        
        # Save best model
        if accuracy > self.best_score:
            self.best_score = accuracy
            logger.info(f"New best score: {accuracy:.4f} - Saving model to {self.best_model_path}")
            model.save(self.best_model_path)
            
            # Save a record of the best score
            with open(os.path.join(self.best_model_path, "best_score.txt"), "w") as f:
                f.write(f"Best validation accuracy: {accuracy:.4f}\nEpoch: {epoch}\nSteps: {steps}")
        
        return accuracy

def create_weighted_sampler(train_pairs):
    """
    Create a WeightedRandomSampler based on inverse combo frequency
    This upsamples rare label combinations to balance the training
    """
    # Count frequency of each label combination
    combos = [combo for _, combo in train_pairs]
    combo_counter = Counter(combos)
    
    logger.info("Label combination distribution:")
    for combo, count in sorted(combo_counter.items(), key=lambda x: x[1]):
        logger.info(f"  {combo}: {count} instances")
    
    # Calculate inverse frequency for each combo (with smoothing)
    total_samples = len(combos)
    combo_weights = {combo: total_samples / (count * len(combo_counter)) for combo, count in combo_counter.items()}
    
    # Assign weights to each sample based on its combo
    weights = [combo_weights[combo] for _, combo in train_pairs]
    
    # Convert to tensor
    weights_tensor = torch.DoubleTensor(weights)
    
    # Create sampler
    sampler = WeightedRandomSampler(weights=weights_tensor, 
                                   num_samples=len(weights), 
                                   replacement=True)
    
    logger.info("Created weighted sampler with inverse frequency weights:")
    for combo, weight in sorted(combo_weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {combo}: weight={weight:.4f} (upsampling factor: {weight:.2f}x)")
    
    return sampler

class EnhancedHardNegativeMiner:
    """
    Enhanced hard negative miner that prioritizes template-aware and semantically similar negatives
    """
    def __init__(self, train_pairs, model, num_negatives=3):
        self.train_pairs = train_pairs
        self.model = model
        self.num_negatives = num_negatives
        
        # Extract texts, combos, and sample_ids
        self.texts = [t for t, _, _ in train_pairs]
        self.combos = [c for _, c, _ in train_pairs]
        self.sample_ids = [s for _, _, s in train_pairs]
        self.unique_combos = list(set(self.combos))
        
        # Group examples by combo and by sample_id (template)
        self.combo_to_indices = defaultdict(list)
        self.sample_to_indices = defaultdict(list)
        self.sample_to_combo = {}
        
        for idx, (combo, sample_id) in enumerate(zip(self.combos, self.sample_ids)):
            self.combo_to_indices[combo].append(idx)
            self.sample_to_indices[sample_id].append(idx)
            self.sample_to_combo[sample_id] = combo
            
        # Compute initial embeddings
        logger.info(f"Computing initial embeddings for {len(self.texts)} texts...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Initialize nearest neighbors model
        logger.info("Initializing NearestNeighbors model...")
        self.nn_model = NearestNeighbors(n_neighbors=min(50, len(self.texts)), metric='cosine')
        self.nn_model.fit(self.embeddings)
        
        # Mine initial hard negatives
        logger.info("Mining enhanced hard negatives...")
        self.hard_negatives = self._mine_enhanced_hard_negatives()
        logger.info(f"Found {sum(len(negs) for negs in self.hard_negatives)} enhanced hard negatives")

    def _mine_enhanced_hard_negatives(self):
        """
        Mine enhanced hard negatives prioritizing:
        1. Same template structure but different labels (structural similarity)
        2. Similar embeddings but different labels (semantic similarity)
        """
        hard_negatives = [[] for _ in range(len(self.texts))]
        
        # Find nearest neighbors for each example
        distances, indices = self.nn_model.kneighbors(self.embeddings)
        
        # For each example, find hard negatives
        for i in range(len(self.texts)):
            current_combo = self.combos[i]
            current_sample_id = self.sample_ids[i]
            neg_indices = []
            
            # Priority 1: Template-aware negatives (same template structure, different labels)
            # Find other samples with different combos but similar templates
            template_negatives = []
            for other_sample_id, other_combo in self.sample_to_combo.items():
                if (other_sample_id != current_sample_id and 
                    other_combo != current_combo and
                    len(self.sample_to_indices[other_sample_id]) > 0):
                    # Add one example from this template with different labels
                    template_negatives.extend(self.sample_to_indices[other_sample_id])
            
            # Add some template-aware negatives (up to half of requested negatives)
            if template_negatives:
                random.shuffle(template_negatives)
                max_template_negs = max(1, self.num_negatives // 2)
                neg_indices.extend(template_negatives[:max_template_negs])
            
            # Priority 2: Embedding-based hard negatives (semantically similar, different labels)
            for j in indices[i]:
                if (j != i and 
                    self.combos[j] != current_combo and 
                    j not in neg_indices):
                    neg_indices.append(j)
                    if len(neg_indices) >= self.num_negatives:
                        break
            
            # Ensure we have enough negatives by fallback to any different combo
            if len(neg_indices) < self.num_negatives:
                for j in range(len(self.texts)):
                    if (j != i and 
                        self.combos[j] != current_combo and 
                        j not in neg_indices):
                        neg_indices.append(j)
                        if len(neg_indices) >= self.num_negatives:
                            break
            
            # Store the hard negative indices
            hard_negatives[i] = neg_indices[:self.num_negatives]
            
        return hard_negatives
    
    def update_embeddings(self):
        """
        Update embeddings and recompute hard negatives
        """
        logger.info("Updating embeddings and mining new hard negatives...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=False, convert_to_numpy=True)
        self.nn_model.fit(self.embeddings)
        self.hard_negatives = self._mine_enhanced_hard_negatives()
        logger.info(f"Updated hard negatives: {sum(len(negs) for negs in self.hard_negatives)} total")
    
    def get_train_examples_with_hard_negatives(self, mix_ratio=0.5, token_dropout_processor=None):
        """
        Generate training examples with a mix of hard negatives and the original positive pairs
        mix_ratio: proportion of hard negatives to include (0.5 means 50% hard, 50% original)
        token_dropout_processor: optional processor to apply token dropout
        """
        train_examples = []
        
        for i, (text, combo, sample_id) in enumerate(self.train_pairs):
            # Apply token dropout if processor is provided
            processed_text = text
            if token_dropout_processor and random.random() < 0.5:  # 50% chance to apply dropout
                processed_text = token_dropout_processor.apply_token_dropout(text)
            
            if random.random() < mix_ratio and len(self.hard_negatives[i]) > 0:
                # Create example with text and its combo as positive, but include hard negatives
                negatives = [self.texts[idx] for idx in self.hard_negatives[i]]
                train_examples.append(InputExample(texts=[processed_text, combo], label=1.0))
                
                # Add hard negative examples (negative pairs)
                for neg_text in negatives:
                    train_examples.append(InputExample(texts=[processed_text, neg_text], label=0.0))
            else:
                # Just the original positive pair
                train_examples.append(InputExample(texts=[processed_text, combo], label=1.0))
                
        return train_examples

class HardNegativeContrastiveLoss(losses.ContrastiveLoss):
    """
    Extension of ContrastiveLoss with hard negative mining
    """
    def __init__(self, model, hard_negative_miner, margin=0.5):
        super().__init__(model=model, margin=margin)
        self.hard_negative_miner = hard_negative_miner
        
    def update_hard_negatives(self, update_freq=5):
        """Update hard negatives periodically"""
        self.hard_negative_miner.update_embeddings()
        
def main():
    # Load data splits
    train = load_split("synthetic_dataset/splits/train")
    val = load_split("synthetic_dataset/splits/val")
    test = load_split("synthetic_dataset/splits/test")

    # De-dup trivial repeats in training
    train_dedup = list({(t,c,s) for (t,c,s) in train})
    logger.info(f"Removed {len(train) - len(train_dedup)} duplicate pairs from training")
    train = train_dedup

    # Print unique label combinations
    unique_combos = set(combo for _, combo, _ in train)
    logger.info(f"Found {len(unique_combos)} unique label combinations:")
    for combo in sorted(unique_combos):
        count = sum(1 for _, c, _ in train if c == combo)
        logger.info(f"  - {combo}: {count} instances")

    # Model backbone
    model_name = "sentence-transformers/all-mpnet-base-v2"
    logger.info(f"Initializing model: {model_name}")
    model = SentenceTransformer(model_name)

    # Initialize token dropout processor for semantic robustness
    token_dropout = TokenDropoutProcessor(dropout_rate=0.125)
    logger.info("Initialized token dropout processor with 12.5% dropout rate")

    # Initialize enhanced hard negative miner
    hard_negative_miner = EnhancedHardNegativeMiner(train, model, num_negatives=5)
    
    # Get training examples with hard negatives and token dropout
    train_data = hard_negative_miner.get_train_examples_with_hard_negatives(
        mix_ratio=0.6, 
        token_dropout_processor=token_dropout
    )
    logger.info(f"Created {len(train_data)} training examples with enhanced hard negatives and token dropout")
    
    # Configure dataloader with group-aware batching
    batch_size = 32  # Reduced batch size for better group diversity
    
    # Create mapping from training examples back to sample_ids for group-aware batching
    example_to_sample_id = {}
    for i, (text, combo, sample_id) in enumerate(train):
        # Find corresponding training examples by matching text
        for j, example in enumerate(train_data):
            if text in example.texts[0]:  # Match by text content
                example_to_sample_id[j] = sample_id
                break
    
    # Group-aware dataset wrapper
    class GroupAwareDataset:
        def __init__(self, data, example_to_sample_id):
            self.data = data
            self.example_to_sample_id = example_to_sample_id
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]
            
        def get_sample_id(self, idx):
            return self.example_to_sample_id.get(idx, f"unknown_{idx}")
    
    group_aware_dataset = GroupAwareDataset(train_data, example_to_sample_id)
    
    # Create group-aware sampler
    def get_sample_id_fn(example_or_idx):
        if isinstance(example_or_idx, int):
            return group_aware_dataset.get_sample_id(example_or_idx)
        return f"hash_{hash(str(example_or_idx)) % 1000}"
    
    group_sampler = GroupAwareSampler(
        dataset=range(len(train_data)),  # Use indices 
        batch_size=batch_size,
        group_key_fn=lambda idx: group_aware_dataset.get_sample_id(idx)
    )
    
    # Create custom data loader with group-aware batching
    def collate_fn(indices):
        return [train_data[i] for i in indices]
    
    train_loader = DataLoader(
        range(len(train_data)),
        batch_sampler=group_sampler,
        collate_fn=collate_fn
    )
    logger.info(f"Created group-aware dataloader with {len(train_data)} examples, batch size {batch_size}")
    logger.info(f"Group-aware sampler will prevent same-document samples in batches")

    # MultipleNegativesRankingLoss with hard negatives
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    logger.info("Using MultipleNegativesRankingLoss with enhanced hard negatives")

    # Create output directory and best model path
    output_path = "models/thinkerbell-encoder-semantic"  # New path for semantic improvements
    best_model_path = "models/thinkerbell-encoder-semantic-best"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(best_model_path).mkdir(parents=True, exist_ok=True)
    
    # Create Recall@3 evaluator for early stopping on ranking quality
    evaluator = RecallAtKEvaluator(val, model_name, best_model_path, recall_k=3)
    logger.info("Using Recall@3 evaluator for early stopping on ranking quality")
    
    # Training parameters
    epochs = 6  # Increased epochs for better convergence
    warmup_steps = int(0.1 * len(train_loader))
    evaluation_steps = len(train_loader)  # Evaluate after each epoch
    
    logger.info(f"Training for {epochs} epochs with {warmup_steps} warmup steps")
    logger.info(f"Will evaluate and save best model every {evaluation_steps} steps")
    
    # Track training time
    start_time = time.time()
    
    # Add a callback to update hard negatives periodically
    class HardNegativeUpdateCallback:
        def __init__(self, miner, update_frequency=2000):
            self.miner = miner
            self.update_frequency = update_frequency
            self.steps = 0
            
        def __call__(self, score, epoch, steps):
            self.steps += 1
            if self.steps % self.update_frequency == 0:
                logger.info(f"Updating hard negatives at epoch {epoch}, step {steps}")
                self.miner.update_embeddings()
    
    hard_negative_callback = HardNegativeUpdateCallback(hard_negative_miner, update_frequency=evaluation_steps)
    
    # Train with evaluation
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        callback=hard_negative_callback,
        output_path=output_path,
        show_progress_bar=True
    )
    
    # Report training time
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    # Save model info for both final and best models
    model_info = {
        "base_model": model_name,
        "training_examples": len(train_data),
        "epochs": epochs,
        "batch_size": batch_size,
        "unique_label_combinations": list(sorted(unique_combos)),
        "training_time_seconds": train_time,
        "best_recall_at_3": evaluator.best_recall_at_k,
        "best_accuracy_at_best_recall": evaluator.best_accuracy,
        "semantic_improvements": {
            "token_dropout_rate": token_dropout.dropout_rate,
            "group_aware_batching": True,
            "enhanced_hard_negatives": True,
            "template_aware_negatives": True,
            "recall_at_k_early_stopping": True
        },
        "hard_negative_ratio": 0.6,
        "hard_negatives_per_example": hard_negative_miner.num_negatives
    }
    
    with open(os.path.join(output_path, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)
        
    with open(os.path.join(best_model_path, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Final model saved to: {output_path}")
    logger.info(f"Best model saved to: {best_model_path} (Recall@3: {evaluator.best_recall_at_k:.4f}, Accuracy: {evaluator.best_accuracy:.4f})")
    logger.info(f"Semantic improvements applied: token dropout, group-aware batching, template-aware hard negatives, Recall@3 early stopping")

if __name__ == "__main__":
    main() 