#!/usr/bin/env python3
"""
Enhanced Thinkerbell Training Pipeline with MinIO and Redis Integration

Extends the existing training pipeline with:
- Model artifact upload to MinIO after each epoch
- Metrics caching in Redis
- Training resume from MinIO checkpoints
- Enhanced logging and monitoring
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random

# Import existing training pipeline
from train_pipeline import TrainingPipeline

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

# Import Thinkerbell utilities
try:
    # Add thinkerbell to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "thinkerbell"))
    from utils.minio_client import get_minio_client, upload_file, download_file, list_objects
    from utils.redis_client import get_redis_client, cache_set, cache_get
    from config.settings import Settings
    HAS_THINKERBELL_UTILS = True
except ImportError as e:
    HAS_THINKERBELL_UTILS = False
    print(f"⚠️ Thinkerbell utilities not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_training_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


class EnhancedTrainingPipeline(TrainingPipeline):
    """
    Enhanced training pipeline with MinIO and Redis integration.
    
    Extends the base TrainingPipeline with:
    - Automatic model upload to MinIO after each epoch
    - Metrics caching in Redis
    - Training resume from MinIO checkpoints
    - Enhanced monitoring and logging
    """
    
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
                 save_best: bool = True,
                 enable_minio: bool = True,
                 enable_redis: bool = True,
                 resume_from_minio: bool = True):
        
        # Initialize base pipeline
        super().__init__(
            dataset_dir=dataset_dir,
            models_dir=models_dir,
            archive_dir=archive_dir,
            target_samples=target_samples,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            balanced=balanced,
            save_best=save_best
        )
        
        # Enhanced features
        self.enable_minio = enable_minio and HAS_THINKERBELL_UTILS
        self.enable_redis = enable_redis and HAS_THINKERBELL_UTILS
        self.resume_from_minio = resume_from_minio and self.enable_minio
        
        # Training session ID for uniqueness
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_run_id = f"training_{self.session_id}"
        
        # Enhanced paths
        self.checkpoint_dir = self.models_dir / "checkpoints" / self.training_run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # MinIO and Redis clients
        self.minio_client = get_minio_client() if self.enable_minio else None
        self.redis_client = get_redis_client() if self.enable_redis else None
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.epoch_metrics = []
        
        logger.info(f"Enhanced training pipeline initialized:")
        logger.info(f"  Session ID: {self.session_id}")
        logger.info(f"  MinIO enabled: {self.enable_minio}")
        logger.info(f"  Redis enabled: {self.enable_redis}")
        logger.info(f"  Resume from MinIO: {self.resume_from_minio}")
    
    def test_integrations(self) -> bool:
        """Test MinIO and Redis connections."""
        logger.info("🔧 Testing integrations...")
        
        success = True
        
        # Test MinIO
        if self.enable_minio:
            try:
                minio_status = self.minio_client.test_connection()
                if minio_status:
                    logger.info("✅ MinIO connection successful")
                    # Ensure bucket exists
                    bucket_created = self.minio_client.create_bucket(Settings.MINIO_BUCKET)
                    if bucket_created:
                        logger.info("✅ MinIO bucket ready")
                    else:
                        logger.warning("⚠️ Failed to create MinIO bucket")
                        success = False
                else:
                    logger.warning("⚠️ MinIO connection failed")
                    success = False
            except Exception as e:
                logger.error(f"❌ MinIO test failed: {e}")
                success = False
        
        # Test Redis
        if self.enable_redis:
            try:
                redis_status = self.redis_client.test_connection()
                if redis_status:
                    logger.info("✅ Redis connection successful")
                else:
                    logger.warning("⚠️ Redis connection failed")
                    success = False
            except Exception as e:
                logger.error(f"❌ Redis test failed: {e}")
                success = False
        
        return success
    
    def check_for_existing_checkpoints(self) -> Optional[str]:
        """Check MinIO for existing checkpoints to resume from."""
        if not self.enable_minio or not self.resume_from_minio:
            return None
        
        logger.info("🔍 Checking for existing checkpoints in MinIO...")
        
        try:
            # List checkpoints for this model
            checkpoint_prefix = f"checkpoints/{self.model_name.replace('/', '_')}"
            checkpoint_objects = self.minio_client.list_objects(Settings.MINIO_BUCKET, prefix=checkpoint_prefix)
            
            if not checkpoint_objects:
                logger.info("No existing checkpoints found")
                return None
            
            # Find the latest checkpoint
            model_checkpoints = [obj for obj in checkpoint_objects if obj.endswith('/pytorch_model.bin')]
            if model_checkpoints:
                latest_checkpoint = max(model_checkpoints)
                logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                return latest_checkpoint
            
        except Exception as e:
            logger.warning(f"Failed to check for checkpoints: {e}")
        
        return None
    
    def download_checkpoint(self, checkpoint_path: str) -> Optional[str]:
        """Download checkpoint from MinIO."""
        if not self.enable_minio:
            return None
        
        try:
            # Create local checkpoint directory
            local_checkpoint_dir = self.checkpoint_dir / "resume"
            local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model files
            checkpoint_base = checkpoint_path.rsplit('/', 1)[0]
            model_files = ['pytorch_model.bin', 'config.json', 'config_sentence_transformers.json']
            
            downloaded_files = []
            for model_file in model_files:
                remote_path = f"{checkpoint_base}/{model_file}"
                local_path = local_checkpoint_dir / model_file
                
                if self.minio_client.object_exists(Settings.MINIO_BUCKET, remote_path):
                    success = download_file(Settings.MINIO_BUCKET, remote_path, local_path)
                    if success:
                        downloaded_files.append(str(local_path))
                        logger.info(f"Downloaded: {model_file}")
            
            if downloaded_files:
                logger.info(f"✅ Downloaded checkpoint to: {local_checkpoint_dir}")
                return str(local_checkpoint_dir)
            
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
        
        return None
    
    def save_epoch_metrics_to_redis(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save epoch metrics to Redis."""
        if not self.enable_redis:
            return
        
        try:
            # Save individual epoch metrics
            epoch_key = f"metrics:epoch:{self.training_run_id}:{epoch}"
            cache_set(epoch_key, metrics, expire_seconds=86400)  # 24 hours
            
            # Update training run summary
            summary_key = f"training:summary:{self.training_run_id}"
            summary = cache_get(summary_key) or {
                "training_run_id": self.training_run_id,
                "model_name": self.model_name,
                "start_time": datetime.now().isoformat(),
                "total_epochs": self.epochs,
                "completed_epochs": 0,
                "best_score": 0.0,
                "epoch_metrics": []
            }
            
            summary["completed_epochs"] = epoch
            summary["best_score"] = max(summary["best_score"], metrics.get("validation_score", 0.0))
            summary["epoch_metrics"].append({
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                **metrics
            })
            
            cache_set(summary_key, summary, expire_seconds=86400 * 7)  # 7 days
            
            logger.info(f"📊 Cached metrics for epoch {epoch}")
            
        except Exception as e:
            logger.warning(f"Failed to save metrics to Redis: {e}")
    
    def upload_epoch_checkpoint_to_minio(self, epoch: int, model_path: str, metrics: Dict[str, Any]) -> None:
        """Upload epoch checkpoint to MinIO."""
        if not self.enable_minio:
            return
        
        try:
            # Create checkpoint path in MinIO
            model_safe_name = self.model_name.replace('/', '_')
            checkpoint_base = f"checkpoints/{model_safe_name}/{self.training_run_id}/epoch_{epoch}"
            
            # Upload model files
            model_path = Path(model_path)
            if model_path.is_dir():
                # Upload all files in the model directory
                for model_file in model_path.iterdir():
                    if model_file.is_file():
                        remote_path = f"{checkpoint_base}/{model_file.name}"
                        success = upload_file(Settings.MINIO_BUCKET, remote_path, model_file)
                        if success:
                            logger.info(f"📤 Uploaded {model_file.name} to MinIO")
            
            # Upload epoch metadata
            metadata = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "training_run_id": self.training_run_id,
                "model_name": self.model_name,
                "metrics": metrics,
                "config": self.config
            }
            
            metadata_file = self.checkpoint_dir / f"epoch_{epoch}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            metadata_remote_path = f"{checkpoint_base}/metadata.json"
            upload_file(Settings.MINIO_BUCKET, metadata_remote_path, metadata_file)
            
            logger.info(f"✅ Uploaded epoch {epoch} checkpoint to MinIO")
            
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to MinIO: {e}")
    
    def create_custom_evaluator(self, val_chunks: List[Dict]) -> 'EnhancedEvaluator':
        """Create custom evaluator with MinIO/Redis integration."""
        return EnhancedEvaluator(
            val_chunks=val_chunks,
            pipeline=self,
            name="enhanced_validation"
        )
    
    def train_model(self) -> None:
        """Enhanced training method with MinIO and Redis integration."""
        logger.info("🚀 Step 3: Enhanced training with MinIO/Redis integration...")
        
        if not HAS_ML_DEPS:
            raise RuntimeError("ML dependencies not available for training")
        
        # Test integrations
        if not self.test_integrations():
            logger.warning("⚠️ Some integrations failed, continuing with available features")
        
        # Check for existing checkpoints
        existing_checkpoint = self.check_for_existing_checkpoints()
        start_epoch = 0
        
        if existing_checkpoint:
            logger.info(f"🔄 Found existing checkpoint: {existing_checkpoint}")
            choice = input("Resume from checkpoint? (y/n): ").lower().strip()
            if choice == 'y':
                local_checkpoint = self.download_checkpoint(existing_checkpoint)
                if local_checkpoint:
                    logger.info(f"✅ Resuming from: {local_checkpoint}")
                    # Extract epoch number from checkpoint path
                    try:
                        start_epoch = int(existing_checkpoint.split('epoch_')[1].split('/')[0]) + 1
                        logger.info(f"Resuming from epoch {start_epoch}")
                    except:
                        logger.warning("Could not determine epoch, starting from 0")
        
        # Load training data
        train_chunks = self.load_chunks("train")
        val_chunks = self.load_chunks("val")
        
        logger.info(f"  Loaded {len(train_chunks)} training chunks")
        logger.info(f"  Loaded {len(val_chunks)} validation chunks")
        
        # Create training examples
        train_examples = self.create_training_examples(train_chunks)
        
        # Initialize model
        logger.info(f"  Loading base model: {self.model_name}")
        if existing_checkpoint and start_epoch > 0:
            # Load from checkpoint
            local_checkpoint = self.download_checkpoint(existing_checkpoint)
            if local_checkpoint:
                model = SentenceTransformer(local_checkpoint)
                logger.info("✅ Loaded model from checkpoint")
            else:
                model = SentenceTransformer(self.model_name)
        else:
            model = SentenceTransformer(self.model_name)
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Configure loss function
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # Create enhanced evaluator
        evaluator = self.create_custom_evaluator(val_chunks)
        
        # Training configuration
        warmup_steps = int(len(train_dataloader) * 0.1)
        
        logger.info(f"  Training for {self.epochs} epochs (starting from {start_epoch})...")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        
        # Manual training loop for better control
        self.manual_training_loop(
            model=model,
            train_dataloader=train_dataloader,
            train_loss=train_loss,
            evaluator=evaluator,
            warmup_steps=warmup_steps,
            start_epoch=start_epoch
        )
        
        # Save final results
        self.results["training_metrics"] = {
            "total_examples": len(train_examples),
            "epochs_completed": self.epochs,
            "warmup_steps": warmup_steps,
            "model_saved_to": str(self.best_model_dir),
            "best_validation_score": self.best_score,
            "training_run_id": self.training_run_id
        }
        
        logger.info("✅ Enhanced training complete")
    
    def manual_training_loop(self, model, train_dataloader, train_loss, evaluator, warmup_steps, start_epoch=0):
        """Manual training loop with epoch-by-epoch hooks."""
        from sentence_transformers import SentenceTransformer
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        
        total_steps = len(train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        model.train()
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            logger.info(f"📚 Starting epoch {epoch + 1}/{self.epochs}")
            
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop for this epoch
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                loss_value = train_loss(batch, None)
                epoch_loss += loss_value.item()
                num_batches += 1
                
                # Backward pass
                loss_value.backward()
                optimizer.step()
                scheduler.step()
                
                if num_batches % 100 == 0:
                    logger.info(f"  Batch {num_batches}/{len(train_dataloader)}, Loss: {loss_value.item():.4f}")
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / num_batches
            
            # Evaluate model
            logger.info(f"🔍 Evaluating epoch {epoch + 1}...")
            eval_score = evaluator(model, epoch=epoch)
            
            # Update best score
            if eval_score > self.best_score:
                self.best_score = eval_score
                logger.info(f"🎯 New best score: {self.best_score:.4f}")
                
                # Save best model
                if self.save_best:
                    model.save(str(self.best_model_dir))
                    logger.info(f"💾 Saved best model to: {self.best_model_dir}")
            
            # Collect epoch metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "validation_score": eval_score,
                "epoch_time": epoch_time,
                "learning_rate": scheduler.get_last_lr()[0],
                "timestamp": datetime.now().isoformat()
            }
            
            self.epoch_metrics.append(epoch_metrics)
            
            # Save epoch checkpoint
            epoch_checkpoint_dir = self.checkpoint_dir / f"epoch_{epoch + 1}"
            epoch_checkpoint_dir.mkdir(exist_ok=True)
            model.save(str(epoch_checkpoint_dir))
            
            # Enhanced integrations - upload to MinIO and cache in Redis
            self.save_epoch_metrics_to_redis(epoch + 1, epoch_metrics)
            self.upload_epoch_checkpoint_to_minio(epoch + 1, str(epoch_checkpoint_dir), epoch_metrics)
            
            logger.info(f"✅ Epoch {epoch + 1} complete - Loss: {avg_loss:.4f}, Score: {eval_score:.4f}, Time: {epoch_time:.1f}s")


class EnhancedEvaluator:
    """Custom evaluator with enhanced integration."""
    
    def __init__(self, val_chunks: List[Dict], pipeline: EnhancedTrainingPipeline, name: str = "validation"):
        self.val_chunks = val_chunks
        self.pipeline = pipeline
        self.name = name
        self.best_score = 0.0
    
    def __call__(self, model, epoch: int = None) -> float:
        """Evaluate the model and return score."""
        try:
            # Create evaluation examples
            val_examples = self.pipeline.create_training_examples(self.val_chunks[:100])  # Limit for speed
            
            if not val_examples:
                return 0.0
            
            # Simple evaluation - compute embeddings and similarity
            texts1 = [ex.texts[0] for ex in val_examples[:50]]
            texts2 = [ex.texts[1] for ex in val_examples[:50]]
            labels = [ex.label for ex in val_examples[:50]]
            
            embeddings1 = model.encode(texts1, convert_to_tensor=True)
            embeddings2 = model.encode(texts2, convert_to_tensor=True)
            
            # Compute cosine similarity
            import torch.nn.functional as F
            similarities = F.cosine_similarity(embeddings1, embeddings2)
            
            # Simple accuracy based on threshold
            predictions = (similarities > 0.5).float()
            accuracy = (predictions == torch.tensor(labels)).float().mean().item()
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0


def main():
    """Main function for enhanced training pipeline."""
    parser = argparse.ArgumentParser(
        description="Enhanced Thinkerbell Training Pipeline with MinIO/Redis"
    )
    parser.add_argument("--dataset", default="synthetic_dataset/chunks", help="Path to dataset chunks directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Base model to fine-tune")
    parser.add_argument("--target-samples", type=int, default=6000, help="Expected number of training samples")
    parser.add_argument("--balanced", action="store_true", help="Use balanced sampling")
    parser.add_argument("--save-best", action="store_true", default=True, help="Save the best model checkpoint")
    parser.add_argument("--disable-minio", action="store_true", help="Disable MinIO integration")
    parser.add_argument("--disable-redis", action="store_true", help="Disable Redis integration")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing checkpoints")
    
    args = parser.parse_args()
    
    # Create and run enhanced pipeline
    pipeline = EnhancedTrainingPipeline(
        dataset_dir=args.dataset,
        target_samples=args.target_samples,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        balanced=args.balanced,
        save_best=args.save_best,
        enable_minio=not args.disable_minio,
        enable_redis=not args.disable_redis,
        resume_from_minio=not args.no_resume
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()



