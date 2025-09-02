#!/usr/bin/env python3
"""
Thinkerbell Sentence Encoder Evaluation
Evaluates the performance of the trained encoder on validation and test sets
Includes both combo-level and item-level evaluation to check for memorization
"""

from sentence_transformers import SentenceTransformer
import json
import glob
import os
import numpy as np
import logging
from sklearn.metrics import classification_report
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_pairs(split_dir):
    """Load text-combo pairs from a split directory"""
    pairs = []
    combos = set()
    
    logger.info(f"Loading pairs from {split_dir}")
    for fp in glob.glob(os.path.join(split_dir, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            j = json.load(f)
            t = j["text"]
            c = "|".join(sorted(j["labels"]))
            # Add chunk_id and file_path for item-level evaluation
            chunk_id = j.get("chunk_id", os.path.basename(fp).split('.')[0])
            pairs.append((t, c, chunk_id, fp))
            combos.add(c)
    
    logger.info(f"Loaded {len(pairs)} pairs with {len(combos)} unique combinations")
    return pairs, sorted(list(combos))

def evaluate_combo_rankings(pairs, model, combos, combo_vecs, k_values=[1, 3, 5]):
    """
    Evaluate using LABEL COMBINATIONS as retrieval targets
    Calculates ranking-based metrics for combo-level retrieval
    """
    logger.info("\nEvaluating COMBO-LEVEL retrieval (texts match to label combinations)...")
    
    texts = [t for t, _, _, _ in pairs]
    gold = [c for _, c, _, _ in pairs]
    gold_indices = [combos.index(g) for g in gold]
    
    logger.info(f"Encoding {len(texts)} texts")
    text_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    # Calculate similarities and sort to get rankings
    logger.info("Computing similarities and rankings")
    sims = text_vecs @ combo_vecs.T
    
    # For each example, get the indices of combos sorted by similarity (descending)
    rankings = np.argsort(-sims, axis=1)
    
    # Calculate metrics
    metrics = {}
    
    # Calculate Recall@k
    for k in k_values:
        # For each example, check if the gold combo is in the top k predictions
        top_k_correct = 0
        for i, gold_idx in enumerate(gold_indices):
            if gold_idx in rankings[i, :k]:
                top_k_correct += 1
        
        recall_k = top_k_correct / len(pairs)
        metrics[f'recall@{k}'] = recall_k
        logger.info(f"Recall@{k}: {recall_k:.4f}")
    
    # Calculate Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for i, gold_idx in enumerate(gold_indices):
        # Find the rank of the gold combo (add 1 because ranks are 1-based)
        rank = np.where(rankings[i] == gold_idx)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)
    
    mrr = np.mean(reciprocal_ranks)
    metrics['mrr'] = mrr
    logger.info(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    
    # Top-1 predictions (same as Recall@1, but we need the predicted combos)
    top1_preds = [combos[idx] for idx in rankings[:, 0]]
    
    return metrics, top1_preds, gold

def evaluate_item_rankings(pairs, model, k_values=[1, 3, 5]):
    """
    Evaluate using ITEMS (chunks) as retrieval targets
    Calculates ranking-based metrics for item-level self-retrieval
    """
    logger.info("\nEvaluating ITEM-LEVEL retrieval (each chunk should retrieve itself)...")
    
    texts = [t for t, _, _, _ in pairs]
    chunk_ids = [chunk_id for _, _, chunk_id, _ in pairs]
    label_combos = [c for _, c, _, _ in pairs]
    
    # Group chunk IDs by label combo for finding duplicates
    combo_to_chunks = {}
    for i, (_, combo, chunk_id, _) in enumerate(pairs):
        if combo not in combo_to_chunks:
            combo_to_chunks[combo] = []
        combo_to_chunks[combo].append(i)
    
    logger.info(f"Encoding {len(texts)} texts")
    text_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    # Use FAISS for efficient similarity search
    logger.info("Building FAISS index for item-level retrieval")
    
    # Set up FAISS index
    d = text_vecs.shape[1]  # Embedding dimension
    index = faiss.IndexFlatIP(d)  # Inner product (cosine when vectors are normalized)
    index.add(text_vecs)
    
    # Perform retrieval evaluation
    metrics = {}
    total_queries = len(texts)
    
    # Initialize metrics counters
    recall_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []
    
    logger.info(f"Evaluating item-level retrieval for {total_queries} queries")
    
    for query_idx in range(total_queries):
        query_vec = text_vecs[query_idx:query_idx+1]  # Keep as 2D array
        query_combo = label_combos[query_idx]
        
        # Find all chunks that share the same label combo (including self)
        correct_indices = set(combo_to_chunks[query_combo])
        
        # Search with k+1 to account for self-match
        k_search = max(k_values) + 1
        sims, indices = index.search(query_vec, k_search)
        
        # Remove self from results (first result should be the query itself)
        retrieved_indices = indices[0].tolist()
        if query_idx in retrieved_indices:
            retrieved_indices.remove(query_idx)
        else:
            # If self not found (shouldn't happen with cosine), just remove first
            retrieved_indices = retrieved_indices[1:]
        
        # Calculate recall@k
        for k in k_values:
            top_k_indices = set(retrieved_indices[:k])
            # Check if any correct index (besides self) is in top k
            if any(idx in correct_indices for idx in top_k_indices if idx != query_idx):
                recall_counts[k] += 1
        
        # Calculate reciprocal rank
        rr = 0.0
        for rank, idx in enumerate(retrieved_indices):
            if idx in correct_indices and idx != query_idx:
                rr = 1.0 / (rank + 1)  # +1 because rank is 0-indexed
                break
        reciprocal_ranks.append(rr)
    
    # Compute final metrics
    for k in k_values:
        recall_k = recall_counts[k] / total_queries
        metrics[f'recall@{k}'] = recall_k
        logger.info(f"Recall@{k}: {recall_k:.4f}")
    
    mrr = np.mean(reciprocal_ranks)
    metrics['mrr'] = mrr
    logger.info(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    
    return metrics

def detailed_metrics(pred_combos, gold_combos, combos):
    """Calculate detailed metrics per label combination"""
    logger.info("Computing detailed metrics per label combination")
    
    # Convert to binary classification for each combo
    y_true = np.zeros((len(gold_combos), len(combos)))
    y_pred = np.zeros((len(pred_combos), len(combos)))
    
    for i, gold in enumerate(gold_combos):
        combo_idx = combos.index(gold)
        y_true[i, combo_idx] = 1
        
    for i, pred in enumerate(pred_combos):
        combo_idx = combos.index(pred)
        y_pred[i, combo_idx] = 1
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=combos, output_dict=True)
    
    return report

def compare_metrics(combo_metrics, item_metrics):
    """Compare combo-level and item-level metrics to detect memorization"""
    logger.info("\n===== MEMORIZATION ANALYSIS =====")
    logger.info("Comparing combo-level vs item-level retrieval metrics:")
    
    metrics_diff = {}
    
    for metric in ['recall@1', 'recall@3', 'recall@5', 'mrr']:
        combo_val = combo_metrics.get(metric, 0)
        item_val = item_metrics.get(metric, 0)
        diff = combo_val - item_val
        metrics_diff[metric] = diff
        
        logger.info(f"{metric.upper()}: combo={combo_val:.4f}, item={item_val:.4f}, diff={diff:.4f}")
    
    # Interpretation
    if metrics_diff['recall@1'] > 0.2 and metrics_diff['mrr'] > 0.2:
        logger.info("\nPOTENTIAL MEMORIZATION DETECTED: Combo-level metrics significantly higher than item-level")
        logger.info("This suggests the model may be memorizing label combinations rather than understanding semantics")
    elif abs(metrics_diff['recall@1']) < 0.1 and abs(metrics_diff['mrr']) < 0.1:
        logger.info("\nBALANCED PERFORMANCE: Similar metrics for combo and item level")
        logger.info("This suggests the model is capturing true semantic relationships, not just memorizing")
    elif metrics_diff['recall@1'] < -0.1:
        logger.info("\nBETTER ITEM RETRIEVAL: Item-level metrics higher than combo-level")
        logger.info("This suggests the model may be overfitting to specific examples rather than generalizing to concepts")
    
    return metrics_diff

def main():
    """Main evaluation function"""
    # Load the trained model
    model_path = "models/thinkerbell-encoder-best"  # Changed to use the best model
    logger.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Load validation and test data
    val_pairs, val_combos = load_pairs("synthetic_dataset/splits/val")
    test_pairs, test_combos = load_pairs("synthetic_dataset/splits/test")
    
    # Define k values for Recall@k
    k_values = [1, 3, 5]
    
    # Combine all unique combos from both sets
    all_combos = sorted(list(set(val_combos + test_combos)))
    logger.info(f"Found {len(all_combos)} unique label combinations")
    
    # Embed label combos once
    logger.info("Encoding label combinations")
    combo_vecs = model.encode(all_combos, normalize_embeddings=True)
    
    # 1. COMBO-LEVEL EVALUATION: Text -> Label Combos
    # -----------------------------------------------
    
    # Calculate combo-level metrics for validation set
    logger.info("\nEvaluating validation set...")
    val_combo_metrics, val_preds, val_gold = evaluate_combo_rankings(val_pairs, model, all_combos, combo_vecs, k_values)
    
    # Calculate combo-level metrics for test set
    logger.info("\nEvaluating test set...")
    test_combo_metrics, test_preds, test_gold = evaluate_combo_rankings(test_pairs, model, all_combos, combo_vecs, k_values)
    
    # Calculate and display detailed metrics for test set
    test_report = detailed_metrics(test_preds, test_gold, all_combos)
    
    # 2. ITEM-LEVEL EVALUATION: Text -> Text (self-retrieval)
    # -------------------------------------------------------
    
    # Calculate item-level metrics for validation set
    val_item_metrics = evaluate_item_rankings(val_pairs, model, k_values)
    
    # Calculate item-level metrics for test set
    test_item_metrics = evaluate_item_rankings(test_pairs, model, k_values)
    
    # 3. COMPARE METRICS: Check for memorization
    # ------------------------------------------
    
    # Compare validation metrics
    logger.info("\n----- VALIDATION SET -----")
    val_diff = compare_metrics(val_combo_metrics, val_item_metrics)
    
    # Compare test metrics
    logger.info("\n----- TEST SET -----")
    test_diff = compare_metrics(test_combo_metrics, test_item_metrics)
    
    # Display macro avg results for combo-level evaluation
    logger.info("\nDetailed Test Metrics (macro avg):")
    logger.info(f"Precision: {test_report['macro avg']['precision']:.4f}")
    logger.info(f"Recall: {test_report['macro avg']['recall']:.4f}")
    logger.info(f"F1-score: {test_report['macro avg']['f1-score']:.4f}")
    
    # Display per-class results
    logger.info("\nPer-combo performance (F1 scores):")
    for combo in all_combos:
        if combo in test_report:
            f1 = test_report[combo]['f1-score']
            support = test_report[combo]['support']
            logger.info(f"{combo[:50] + '...' if len(combo) > 50 else combo}: {f1:.4f} (support: {int(support)})")
    
    # Save results to file with expanded metrics
    results = {
        # Validation metrics
        "validation_metrics": {
            "combo_level": val_combo_metrics,
            "item_level": val_item_metrics,
            "diff": val_diff
        },
        
        # Test metrics
        "test_metrics": {
            "combo_level": test_combo_metrics,
            "item_level": test_item_metrics,
            "diff": test_diff
        },
        
        # For backwards compatibility
        "val_accuracy": val_combo_metrics['recall@1'],
        "test_accuracy": test_combo_metrics['recall@1'],
        
        # Detailed classification metrics
        "detailed_metrics": test_report
    }
    
    with open(os.path.join(model_path, "eval_results_with_memorization_check.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Summary of key metrics
    logger.info("\n==== Summary of Key Metrics ====")
    logger.info("Combo-Level Retrieval (text -> label combinations):")
    logger.info(f"Validation: Accuracy: {val_combo_metrics['recall@1']:.4f}, Recall@3: {val_combo_metrics['recall@3']:.4f}, Recall@5: {val_combo_metrics['recall@5']:.4f}, MRR: {val_combo_metrics['mrr']:.4f}")
    logger.info(f"Test: Accuracy: {test_combo_metrics['recall@1']:.4f}, Recall@3: {test_combo_metrics['recall@3']:.4f}, Recall@5: {test_combo_metrics['recall@5']:.4f}, MRR: {test_combo_metrics['mrr']:.4f}")
    
    logger.info("\nItem-Level Retrieval (text -> similar texts):")
    logger.info(f"Validation: Recall@1: {val_item_metrics['recall@1']:.4f}, Recall@3: {val_item_metrics['recall@3']:.4f}, Recall@5: {val_item_metrics['recall@5']:.4f}, MRR: {val_item_metrics['mrr']:.4f}")
    logger.info(f"Test: Recall@1: {test_item_metrics['recall@1']:.4f}, Recall@3: {test_item_metrics['recall@3']:.4f}, Recall@5: {test_item_metrics['recall@5']:.4f}, MRR: {test_item_metrics['mrr']:.4f}")
    
    logger.info("\nEvaluation complete! Results saved to eval_results_with_memorization_check.json")

if __name__ == "__main__":
    main() 