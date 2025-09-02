#!/usr/bin/env python3
"""
Holdout Evaluation for Thinkerbell Encoder
Evaluates the model on entirely new templates never seen in train/val to test generalization
"""

from sentence_transformers import SentenceTransformer
import json
import glob
import os
import logging
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_holdout_samples(holdout_dir):
    """Load holdout samples that use entirely new templates"""
    logger.info(f"Loading holdout data from {holdout_dir}")
    samples = []
    
    for fp in glob.glob(os.path.join(holdout_dir, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
            
            # Extract template information
            template_id = j.get("template_mapping", {}).get("best_template_match", "unknown")
            base_template = j.get("generation_metadata", {}).get("base_template_used", "unknown")
            
            # For chunked data, load the chunk text and labels
            if "raw_input" in j:
                text = j["raw_input"]["text"]
                # Extract labels from extracted_fields
                labels = []
                fields = j.get("extracted_fields", {})
                
                # Map fields to standardized labels
                label_mapping = {
                    "influencer": "influencer",
                    "client": "client", 
                    "brand": "brand",
                    "campaign": "campaign",
                    "fee": "fee",
                    "deliverables": "deliverables",
                    "exclusivity_period": "exclusivity_period",
                    "engagement_term": "engagement_term", 
                    "usage_term": "usage_term",
                    "territory": "territory",
                    "start_date": "start_date"
                }
                
                for field, value in fields.items():
                    if field in label_mapping and value:
                        labels.append(label_mapping[field])
                
                combo = "|".join(sorted(labels))
                sample_id = j.get("sample_id", "unknown")
                
                samples.append({
                    "text": text,
                    "labels": labels,
                    "combo": combo,
                    "sample_id": sample_id,
                    "template_id": template_id,
                    "base_template": base_template
                })
    
    logger.info(f"Loaded {len(samples)} holdout samples")
    return samples

def analyze_template_overlap(train_templates, holdout_templates):
    """Analyze template overlap between train and holdout sets"""
    train_set = set(train_templates)
    holdout_set = set(holdout_templates) 
    
    overlap = train_set.intersection(holdout_set)
    unique_holdout = holdout_set - train_set
    
    logger.info(f"Template analysis:")
    logger.info(f"  Training templates: {len(train_set)}")
    logger.info(f"  Holdout templates: {len(holdout_set)}")
    logger.info(f"  Overlapping templates: {len(overlap)} ({overlap})")
    logger.info(f"  Unique holdout templates: {len(unique_holdout)} ({unique_holdout})")
    
    return len(overlap) == 0  # True if no overlap (good for evaluation)

def evaluate_model_on_holdout(model_path, holdout_samples, train_combos):
    """Evaluate trained model on holdout samples"""
    logger.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Extract texts and combos
    texts = [s["text"] for s in holdout_samples]
    gold_combos = [s["combo"] for s in holdout_samples]
    
    # Use all unique combos from training + any new ones from holdout
    all_combos = list(set(train_combos + gold_combos))
    logger.info(f"Evaluating with {len(all_combos)} total label combinations")
    
    # Encode texts and combos
    logger.info("Encoding holdout texts...")
    text_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    logger.info("Encoding label combinations...")
    combo_embeddings = model.encode(all_combos, convert_to_tensor=True, show_progress_bar=True)
    
    # Calculate similarities
    cos_sim = torch.nn.functional.cosine_similarity(
        text_embeddings.unsqueeze(1).expand(-1, len(all_combos), -1),
        combo_embeddings.unsqueeze(0).expand(len(texts), -1, -1),
        dim=2
    )
    
    # Calculate metrics
    results = {}
    
    # Top-1 accuracy
    pred_indices = torch.argmax(cos_sim, dim=1).cpu().numpy()
    pred_combos = [all_combos[idx] for idx in pred_indices]
    top1_acc = sum(1 for pred, gold in zip(pred_combos, gold_combos) if pred == gold) / len(gold_combos)
    results["top1_accuracy"] = top1_acc
    
    # Recall@K for K=1,3,5
    for k in [1, 3, 5]:
        if k <= len(all_combos):
            _, top_k_indices = torch.topk(cos_sim, k=k, dim=1)
            top_k_indices = top_k_indices.cpu().numpy()
            
            recall_at_k_hits = 0
            for i, gold_combo in enumerate(gold_combos):
                if gold_combo in all_combos:
                    gold_idx = all_combos.index(gold_combo)
                    if gold_idx in top_k_indices[i]:
                        recall_at_k_hits += 1
            
            recall_at_k = recall_at_k_hits / len(gold_combos)
            results[f"recall_at_{k}"] = recall_at_k
    
    # Mean Reciprocal Rank (MRR)
    mrr_scores = []
    for i, gold_combo in enumerate(gold_combos):
        if gold_combo in all_combos:
            gold_idx = all_combos.index(gold_combo)
            # Find rank of gold combo (1-indexed)
            sorted_indices = torch.argsort(cos_sim[i], descending=True).cpu().numpy()
            rank = np.where(sorted_indices == gold_idx)[0][0] + 1
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)  # Not found
    
    results["mrr"] = np.mean(mrr_scores)
    
    return results, pred_combos, gold_combos

def analyze_error_patterns(holdout_samples, pred_combos, gold_combos):
    """Analyze error patterns by template type"""
    template_errors = defaultdict(list)
    
    for sample, pred, gold in zip(holdout_samples, pred_combos, gold_combos):
        template_id = sample["template_id"]
        is_correct = pred == gold
        
        template_errors[template_id].append({
            "correct": is_correct,
            "predicted": pred,
            "gold": gold,
            "text": sample["text"][:100] + "..." if len(sample["text"]) > 100 else sample["text"]
        })
    
    # Summarize by template
    logger.info("\nError analysis by template:")
    for template_id, errors in template_errors.items():
        correct_count = sum(1 for e in errors if e["correct"])
        total_count = len(errors)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        logger.info(f"  {template_id}: {correct_count}/{total_count} correct ({accuracy:.3f})")
        
        # Show some examples of errors
        error_examples = [e for e in errors if not e["correct"]][:3]
        for example in error_examples:
            logger.info(f"    ERROR: '{example['text']}' -> predicted: {example['predicted']}, gold: {example['gold']}")

def main():
    # Load holdout data (entirely new templates)
    holdout_samples = load_holdout_samples("complete_pipeline_5000/test_set/holdout_samples")
    
    if not holdout_samples:
        logger.error("No holdout samples found! Please check the path.")
        return
    
    # Load training data to get template information and label combinations
    train_samples = []
    train_files = glob.glob("synthetic_dataset/splits/train/*.json")[:100]  # Sample for template analysis
    for fp in train_files:
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
            train_samples.append(j)
    
    # Extract template info from training samples (for overlap analysis)
    train_templates = [s.get("template_id", "unknown") for s in train_samples]
    holdout_templates = [s["template_id"] for s in holdout_samples]
    
    # Analyze template overlap
    no_overlap = analyze_template_overlap(train_templates, holdout_templates)
    if not no_overlap:
        logger.warning("Template overlap detected! This evaluation may not test true generalization.")
    
    # Get training label combinations
    train_combos = []
    train_split_files = glob.glob("synthetic_dataset/splits/train/*.json")
    for fp in train_split_files[:200]:  # Sample to get label variety
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
            labels = j.get("labels", [])
            combo = "|".join(sorted(labels))
            train_combos.append(combo)
    
    train_combos = list(set(train_combos))  # Unique combos
    logger.info(f"Found {len(train_combos)} unique label combinations in training data")
    
    # Evaluate different model checkpoints
    model_paths = [
        "models/thinkerbell-encoder-semantic-best",
        "models/thinkerbell-encoder-hn-best", 
        "models/thinkerbell-encoder-best"
    ]
    
    all_results = {}
    for model_path in model_paths:
        if Path(model_path).exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating model: {model_path}")
            logger.info(f"{'='*60}")
            
            try:
                results, pred_combos, gold_combos = evaluate_model_on_holdout(
                    model_path, holdout_samples, train_combos
                )
                
                all_results[model_path] = results
                
                logger.info(f"\nHoldout evaluation results for {model_path}:")
                for metric, value in results.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
                # Error analysis
                analyze_error_patterns(holdout_samples, pred_combos, gold_combos)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path}: {e}")
        else:
            logger.warning(f"Model path does not exist: {model_path}")
    
    # Compare models
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("Model Comparison on Holdout Set:")
        logger.info(f"{'='*60}")
        
        for metric in ["top1_accuracy", "recall_at_3", "mrr"]:
            logger.info(f"\n{metric.upper()}:")
            for model_path, results in all_results.items():
                if metric in results:
                    logger.info(f"  {model_path}: {results[metric]:.4f}")
    
    # Save detailed results
    output_file = "holdout_evaluation_results.json"
    detailed_results = {
        "holdout_samples_count": len(holdout_samples),
        "template_overlap_analysis": {
            "no_overlap": no_overlap,
            "train_templates": len(set(train_templates)), 
            "holdout_templates": len(set(holdout_templates))
        },
        "model_results": all_results,
        "evaluation_timestamp": str(np.datetime64('now'))
    }
    
    with open(output_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main() 