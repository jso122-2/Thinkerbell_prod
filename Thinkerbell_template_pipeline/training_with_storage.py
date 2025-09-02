#!/usr/bin/env python3
"""
Enhanced Training Pipeline with Unified Storage Integration

This module demonstrates how to integrate the unified storage utilities
with the existing training pipeline for automatic model checkpointing,
metrics caching, and artifact management.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import existing training pipeline
try:
    from train_pipeline import TrainingPipeline
    HAS_BASE_PIPELINE = True
except ImportError:
    TrainingPipeline = object
    HAS_BASE_PIPELINE = False

# Import storage utilities
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.storage_utils import (
        cache_training_metrics, get_training_metrics, store_model_checkpoint,
        store_object, retrieve_object, redis_set, redis_get,
        upload_file, download_file, ensure_bucket
    )
    HAS_STORAGE_UTILS = True
except ImportError as e:
    print(f"Storage utils not available: {e}")
    HAS_STORAGE_UTILS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StorageEnhancedTrainingPipeline(TrainingPipeline):
    """
    Training pipeline enhanced with unified storage capabilities.
    
    Adds automatic:
    - Model checkpoint storage in MinIO
    - Metrics caching in Redis
    - Training state persistence
    - Artifact management
    """
    
    def __init__(self, *args, **kwargs):
        # Extract storage-specific parameters
        self.enable_storage = kwargs.pop('enable_storage', True) and HAS_STORAGE_UTILS
        self.storage_bucket = kwargs.pop('storage_bucket', 'thinkerbell-training')
        self.cache_metrics = kwargs.pop('cache_metrics', True)
        self.checkpoint_every = kwargs.pop('checkpoint_every', 1)  # Every N epochs
        
        # Initialize base pipeline
        if HAS_BASE_PIPELINE:
            super().__init__(*args, **kwargs)
        else:
            # Mock initialization for demonstration
            self.model_name = kwargs.get('model_name', 'test-model')
            self.epochs = kwargs.get('epochs', 3)
            self.best_model_dir = Path('models/test-model-best')
        
        # Storage-specific initialization
        if self.enable_storage:
            self.training_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.storage_prefix = f"training/{self.model_name.replace('/', '_')}/{self.training_run_id}"
            
            # Ensure bucket exists
            try:
                ensure_bucket(self.storage_bucket)
                logger.info(f"✅ Storage bucket ready: {self.storage_bucket}")
            except Exception as e:
                logger.warning(f"⚠️ Storage bucket setup failed: {e}")
                self.enable_storage = False
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        self.best_score = 0.0
        
        logger.info(f"Enhanced training pipeline initialized:")
        logger.info(f"  Run ID: {getattr(self, 'training_run_id', 'N/A')}")
        logger.info(f"  Storage enabled: {self.enable_storage}")
        logger.info(f"  Metrics caching: {self.cache_metrics}")
    
    def save_training_state(self) -> bool:
        """Save current training state to storage."""
        if not self.enable_storage:
            return False
        
        try:
            state = {
                "training_run_id": self.training_run_id,
                "model_name": self.model_name,
                "current_epoch": self.current_epoch,
                "best_score": self.best_score,
                "training_history": self.training_history,
                "config": getattr(self, 'config', {}),
                "timestamp": datetime.now().isoformat()
            }
            
            state_key = f"training_state:{self.training_run_id}"
            success = store_object(state_key, state, storage_type="redis", expire=86400*7)  # 7 days
            
            if success:
                logger.info(f"💾 Saved training state for epoch {self.current_epoch}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")
            return False
    
    def load_training_state(self, run_id: str) -> bool:
        """Load training state from storage."""
        if not self.enable_storage:
            return False
        
        try:
            state_key = f"training_state:{run_id}"
            state = retrieve_object(state_key)
            
            if state:
                self.training_run_id = state.get("training_run_id", run_id)
                self.current_epoch = state.get("current_epoch", 0)
                self.best_score = state.get("best_score", 0.0)
                self.training_history = state.get("training_history", [])
                
                logger.info(f"📂 Loaded training state from epoch {self.current_epoch}")
                logger.info(f"   Best score: {self.best_score}")
                logger.info(f"   History entries: {len(self.training_history)}")
                
                return True
            else:
                logger.warning(f"No training state found for run: {run_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            return False
    
    def cache_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]) -> bool:
        """Cache metrics for a specific epoch."""
        if not self.enable_storage or not self.cache_metrics:
            return False
        
        try:
            # Add metadata
            enhanced_metrics = {
                **metrics,
                "epoch": epoch,
                "training_run_id": self.training_run_id,
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache in Redis
            success = cache_training_metrics(self.training_run_id, epoch, enhanced_metrics)
            
            if success:
                logger.info(f"📊 Cached metrics for epoch {epoch}")
                
                # Also add to training history
                self.training_history.append(enhanced_metrics)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache epoch metrics: {e}")
            return False
    
    def store_epoch_checkpoint(self, epoch: int, model_path: Path, metrics: Dict[str, Any]) -> bool:
        """Store model checkpoint for a specific epoch."""
        if not self.enable_storage:
            return False
        
        try:
            # Store model files
            success = store_model_checkpoint(
                self.training_run_id, 
                epoch, 
                model_path, 
                self.storage_bucket
            )
            
            if success:
                # Also store epoch metadata
                metadata = {
                    "epoch": epoch,
                    "model_path": str(model_path),
                    "metrics": metrics,
                    "storage_path": f"checkpoints/{self.training_run_id}/epoch_{epoch}",
                    "timestamp": datetime.now().isoformat()
                }
                
                metadata_key = f"checkpoint_meta:{self.training_run_id}:epoch_{epoch}"
                store_object(metadata_key, metadata, storage_type="redis", expire=86400*30)  # 30 days
                
                logger.info(f"💾 Stored checkpoint for epoch {epoch}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store epoch checkpoint: {e}")
            return False
    
    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints for this training run."""
        if not self.enable_storage:
            return []
        
        try:
            # Get checkpoint metadata from Redis
            checkpoints = []
            
            # This would need to be implemented based on Redis key patterns
            # For now, return cached training history
            for entry in self.training_history:
                if 'epoch' in entry:
                    checkpoints.append({
                        "epoch": entry['epoch'],
                        "metrics": {k: v for k, v in entry.items() if k not in ['epoch', 'timestamp', 'training_run_id', 'model_name']},
                        "timestamp": entry.get('timestamp')
                    })
            
            return sorted(checkpoints, key=lambda x: x['epoch'])
            
        except Exception as e:
            logger.error(f"Failed to get available checkpoints: {e}")
            return []
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch."""
        self.current_epoch = epoch
        logger.info(f"🚀 Starting epoch {epoch}/{self.epochs}")
        
        # Save training state
        if self.enable_storage:
            self.save_training_state()
    
    def on_epoch_end(self, epoch: int, model_path: Path, metrics: Dict[str, Any]):
        """Called at the end of each epoch."""
        logger.info(f"✅ Completed epoch {epoch}")
        
        # Update best score
        current_score = metrics.get('validation_score', metrics.get('accuracy', 0))
        if current_score > self.best_score:
            self.best_score = current_score
            logger.info(f"🎯 New best score: {self.best_score:.4f}")
        
        # Cache metrics
        if self.cache_metrics:
            self.cache_epoch_metrics(epoch, metrics)
        
        # Store checkpoint if needed
        if self.enable_storage and (epoch % self.checkpoint_every == 0):
            self.store_epoch_checkpoint(epoch, model_path, metrics)
        
        # Save updated training state
        if self.enable_storage:
            self.save_training_state()
    
    def on_training_complete(self):
        """Called when training is complete."""
        logger.info("🎉 Training completed!")
        
        if self.enable_storage:
            # Store final summary
            summary = {
                "training_run_id": self.training_run_id,
                "model_name": self.model_name,
                "total_epochs": self.epochs,
                "best_score": self.best_score,
                "total_training_time": sum(entry.get('epoch_time', 0) for entry in self.training_history),
                "final_metrics": self.training_history[-1] if self.training_history else {},
                "completed_at": datetime.now().isoformat(),
                "status": "completed"
            }
            
            summary_key = f"training_summary:{self.training_run_id}"
            store_object(summary_key, summary, storage_type="redis", expire=86400*90)  # 90 days
            
            # Also store as "latest" reference
            latest_key = f"latest_training:{self.model_name.replace('/', '_')}"
            redis_set(latest_key, self.training_run_id, expire=86400*30)
            
            logger.info(f"📋 Stored training summary: {summary_key}")
    
    def train_model_with_storage(self):
        """Enhanced training method with storage integration."""
        logger.info("🚀 Starting enhanced training with storage integration...")
        
        # Mock training loop for demonstration
        for epoch in range(1, self.epochs + 1):
            self.on_epoch_start(epoch)
            
            # Simulate training
            epoch_start = time.time()
            time.sleep(0.5)  # Simulate training time
            epoch_time = time.time() - epoch_start
            
            # Mock metrics
            loss = max(0.1, 2.0 - (epoch * 0.3))
            accuracy = min(0.95, 0.5 + (epoch * 0.08))
            val_loss = loss * 1.1
            val_accuracy = accuracy * 0.95
            
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "validation_score": val_accuracy,
                "epoch_time": epoch_time,
                "learning_rate": 2e-5
            }
            
            logger.info(f"  Epoch {epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Mock model saving
            epoch_model_path = self.best_model_dir / f"epoch_{epoch}"
            epoch_model_path.mkdir(parents=True, exist_ok=True)
            
            # Create mock model files
            (epoch_model_path / "config.json").write_text(json.dumps({"epoch": epoch}))
            (epoch_model_path / "model.bin").write_text(f"mock_model_epoch_{epoch}")
            
            self.on_epoch_end(epoch, epoch_model_path, metrics)
        
        self.on_training_complete()


def demonstrate_storage_integration():
    """Demonstrate the storage-enhanced training pipeline."""
    logger.info("🧪 Demonstrating Storage-Enhanced Training Pipeline")
    logger.info("=" * 60)
    
    if not HAS_STORAGE_UTILS:
        logger.error("❌ Storage utils not available")
        return
    
    # Create enhanced pipeline
    pipeline = StorageEnhancedTrainingPipeline(
        model_name="demo-storage-model",
        epochs=3,
        enable_storage=True,
        cache_metrics=True,
        checkpoint_every=1,
        storage_bucket="thinkerbell-demo"
    )
    
    # Run training
    pipeline.train_model_with_storage()
    
    # Demonstrate retrieval
    logger.info("\n📊 Demonstrating metrics retrieval...")
    
    # Get all metrics for this run
    all_metrics = get_training_metrics(pipeline.training_run_id)
    if all_metrics:
        logger.info(f"Retrieved {len(all_metrics)} epochs of cached metrics:")
        for metrics in all_metrics:
            epoch = metrics['epoch']
            acc = metrics['accuracy']
            loss = metrics['loss']
            logger.info(f"  Epoch {epoch}: Accuracy={acc:.4f}, Loss={loss:.4f}")
    
    # Get available checkpoints
    checkpoints = pipeline.get_available_checkpoints()
    logger.info(f"\n💾 Available checkpoints: {len(checkpoints)}")
    for checkpoint in checkpoints:
        epoch = checkpoint['epoch']
        score = checkpoint['metrics'].get('validation_score', 0)
        logger.info(f"  Checkpoint epoch {epoch}: Score={score:.4f}")
    
    # Cleanup demo files
    import shutil
    shutil.rmtree(pipeline.best_model_dir, ignore_errors=True)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Storage-Enhanced Training Pipeline Demo")
    parser.add_argument("--model-name", default="demo-model", help="Model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--disable-storage", action="store_true", help="Disable storage features")
    parser.add_argument("--storage-bucket", default="thinkerbell-training", help="Storage bucket name")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint frequency")
    parser.add_argument("--demo", action="store_true", help="Run demonstration mode")
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_storage_integration()
    else:
        # Create and run pipeline
        pipeline = StorageEnhancedTrainingPipeline(
            model_name=args.model_name,
            epochs=args.epochs,
            enable_storage=not args.disable_storage,
            storage_bucket=args.storage_bucket,
            checkpoint_every=args.checkpoint_every
        )
        
        pipeline.train_model_with_storage()


if __name__ == "__main__":
    main()



