#!/usr/bin/env python3
"""
Semantic Training Pipeline Runner
Runs the enhanced training with semantic improvements and evaluates on holdout set
"""

import subprocess
import sys
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    """Run the enhanced semantic training"""
    logger.info("="*60)
    logger.info("STARTING SEMANTIC TRAINING")
    logger.info("="*60)
    
    try:
        result = subprocess.run([sys.executable, "train_encoder.py"], 
                              capture_output=False, text=True, check=True)
        logger.info("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return False

def run_holdout_evaluation():
    """Run holdout evaluation on new templates"""
    logger.info("="*60)
    logger.info("STARTING HOLDOUT EVALUATION")
    logger.info("="*60)
    
    try:
        result = subprocess.run([sys.executable, "evaluate_holdout.py"], 
                              capture_output=False, text=True, check=True)
        logger.info("Holdout evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Holdout evaluation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Holdout evaluation failed with error: {e}")
        return False

def check_dependencies():
    """Check if required files exist"""
    required_files = [
        "train_encoder.py",
        "evaluate_holdout.py",
        "synthetic_dataset/splits/train",
        "synthetic_dataset/splits/val",
        "synthetic_dataset/splits/test"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required files/directories:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    return True

def main():
    """Main pipeline runner"""
    logger.info("Semantic Training Pipeline")
    logger.info("Includes: token dropout, group-aware batching, template-aware hard negatives, Recall@3 early stopping")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependencies check failed. Please ensure all required files exist.")
        sys.exit(1)
    
    # Run training
    training_success = run_training()
    if not training_success:
        logger.error("Training failed. Skipping evaluation.")
        sys.exit(1)
    
    # Check if the semantic model was created
    semantic_model_path = "models/thinkerbell-encoder-semantic-best"
    if not Path(semantic_model_path).exists():
        logger.warning(f"Expected model not found at {semantic_model_path}")
        logger.warning("Evaluation may not find the trained model")
    
    # Run holdout evaluation
    evaluation_success = run_holdout_evaluation()
    if not evaluation_success:
        logger.error("Holdout evaluation failed.")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info("Check the following outputs:")
    logger.info(f"  - Trained model: {semantic_model_path}")
    logger.info("  - Evaluation results: holdout_evaluation_results.json")
    logger.info("  - Logs contain detailed training and evaluation metrics")

if __name__ == "__main__":
    main() 