#!/usr/bin/env python3
"""
Training Pipeline Integration Patch

This script patches the existing training pipeline to add MinIO and Redis integration
without requiring a complete rewrite. It monkey-patches the TrainingPipeline class
to add epoch-level hooks for model upload and metrics caching.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

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

logger = logging.getLogger(__name__)


class TrainingIntegrationMixin:
    """
    Mixin class to add MinIO and Redis integration to existing training pipelines.
    
    This can be mixed into existing TrainingPipeline classes to add:
    - Model upload to MinIO after each epoch
    - Metrics caching in Redis
    - Training resume capabilities
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Integration settings
        self.enable_minio = kwargs.get('enable_minio', True) and HAS_THINKERBELL_UTILS
        self.enable_redis = kwargs.get('enable_redis', True) and HAS_THINKERBELL_UTILS
        self.resume_from_minio = kwargs.get('resume_from_minio', True) and self.enable_minio
        
        # Training session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_run_id = f"training_{self.session_id}"
        
        # Clients
        self.minio_client = get_minio_client() if self.enable_minio else None
        self.redis_client = get_redis_client() if self.enable_redis else None
        
        # Tracking
        self.current_epoch = 0
        self.epoch_metrics = []
        
        logger.info(f"Training integration enabled - MinIO: {self.enable_minio}, Redis: {self.enable_redis}")
    
    def save_epoch_metrics_to_redis(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save epoch metrics to Redis."""
        if not self.enable_redis:
            return
        
        try:
            # Cache individual epoch metrics
            epoch_key = f"metrics:epoch:{self.training_run_id}:{epoch}"
            cache_set(epoch_key, metrics, expire_seconds=86400)  # 24 hours
            
            # Update training summary
            summary_key = f"training:summary:{self.training_run_id}"
            summary = cache_get(summary_key) or {
                "training_run_id": self.training_run_id,
                "model_name": getattr(self, 'model_name', 'unknown'),
                "start_time": datetime.now().isoformat(),
                "total_epochs": getattr(self, 'epochs', 0),
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
            logger.info(f"📊 Cached metrics for epoch {epoch} in Redis")
            
        except Exception as e:
            logger.warning(f"Failed to save metrics to Redis: {e}")
    
    def upload_epoch_checkpoint_to_minio(self, epoch: int, model_path: str, metrics: Dict[str, Any]) -> None:
        """Upload epoch checkpoint to MinIO."""
        if not self.enable_minio:
            return
        
        try:
            # Create checkpoint path
            model_safe_name = getattr(self, 'model_name', 'unknown').replace('/', '_')
            checkpoint_base = f"checkpoints/{model_safe_name}/{self.training_run_id}/epoch_{epoch}"
            
            # Upload model files
            model_path = Path(model_path)
            uploaded_files = []
            
            if model_path.is_dir():
                for model_file in model_path.iterdir():
                    if model_file.is_file():
                        remote_path = f"{checkpoint_base}/{model_file.name}"
                        success = upload_file(Settings.MINIO_BUCKET, remote_path, model_file)
                        if success:
                            uploaded_files.append(model_file.name)
            elif model_path.is_file():
                remote_path = f"{checkpoint_base}/{model_path.name}"
                success = upload_file(Settings.MINIO_BUCKET, remote_path, model_path)
                if success:
                    uploaded_files.append(model_path.name)
            
            # Upload metadata
            metadata = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "training_run_id": self.training_run_id,
                "model_name": getattr(self, 'model_name', 'unknown'),
                "metrics": metrics,
                "uploaded_files": uploaded_files
            }
            
            # Save metadata locally first
            checkpoint_dir = getattr(self, 'checkpoint_dir', Path("checkpoints"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = checkpoint_dir / f"epoch_{epoch}_metadata.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Upload metadata
            metadata_remote_path = f"{checkpoint_base}/metadata.json"
            upload_file(Settings.MINIO_BUCKET, metadata_remote_path, metadata_file)
            
            logger.info(f"📤 Uploaded epoch {epoch} checkpoint to MinIO ({len(uploaded_files)} files)")
            
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to MinIO: {e}")
    
    def check_for_resume_checkpoint(self) -> Optional[str]:
        """Check for existing checkpoints to resume from."""
        if not self.enable_minio or not self.resume_from_minio:
            return None
        
        try:
            model_safe_name = getattr(self, 'model_name', 'unknown').replace('/', '_')
            checkpoint_prefix = f"checkpoints/{model_safe_name}"
            
            checkpoint_objects = self.minio_client.list_objects(Settings.MINIO_BUCKET, prefix=checkpoint_prefix)
            
            if checkpoint_objects:
                # Find latest checkpoint
                model_checkpoints = [obj for obj in checkpoint_objects if 'pytorch_model.bin' in obj]
                if model_checkpoints:
                    latest = max(model_checkpoints)
                    logger.info(f"Found potential resume checkpoint: {latest}")
                    return latest
            
        except Exception as e:
            logger.warning(f"Failed to check for resume checkpoints: {e}")
        
        return None
    
    def on_epoch_complete(self, epoch: int, model_path: str, metrics: Dict[str, Any]) -> None:
        """Hook called after each epoch completes."""
        self.current_epoch = epoch
        self.epoch_metrics.append(metrics)
        
        # Save to Redis and upload to MinIO
        self.save_epoch_metrics_to_redis(epoch, metrics)
        self.upload_epoch_checkpoint_to_minio(epoch, model_path, metrics)
        
        logger.info(f"✅ Epoch {epoch} integration hooks completed")


def patch_training_pipeline(pipeline_class):
    """
    Decorator to patch an existing training pipeline class with MinIO/Redis integration.
    
    Usage:
        @patch_training_pipeline
        class MyTrainingPipeline(TrainingPipeline):
            # existing code...
    """
    
    # Create new class that inherits from both TrainingIntegrationMixin and the original class
    class EnhancedPipeline(TrainingIntegrationMixin, pipeline_class):
        
        def train_model(self, *args, **kwargs):
            """Enhanced train_model with integration hooks."""
            logger.info("🔧 Starting training with enhanced integration...")
            
            # Check for resume checkpoint
            resume_checkpoint = self.check_for_resume_checkpoint()
            if resume_checkpoint:
                logger.info(f"Found checkpoint for potential resume: {resume_checkpoint}")
                # Note: Actual resume logic would need to be implemented based on specific training framework
            
            # Call original train_model
            result = super().train_model(*args, **kwargs)
            
            # Final summary
            if self.enable_redis:
                summary_key = f"training:summary:{self.training_run_id}"
                final_summary = cache_get(summary_key)
                if final_summary:
                    final_summary["completed_at"] = datetime.now().isoformat()
                    final_summary["status"] = "completed"
                    cache_set(summary_key, final_summary, expire_seconds=86400 * 30)  # 30 days
            
            logger.info(f"🎉 Training completed with integration - Run ID: {self.training_run_id}")
            return result
    
    return EnhancedPipeline


def create_simple_epoch_hook():
    """
    Create a simple function that can be called after each epoch to upload models and cache metrics.
    
    This is for cases where you can't modify the training class but can add a callback.
    """
    
    # Initialize clients
    minio_client = get_minio_client() if HAS_THINKERBELL_UTILS else None
    redis_client = get_redis_client() if HAS_THINKERBELL_UTILS else None
    
    # Session tracking
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_run_id = f"training_{session_id}"
    
    def epoch_hook(epoch: int, model_path: str, metrics: Dict[str, Any], model_name: str = "unknown"):
        """
        Simple epoch hook function.
        
        Args:
            epoch: Current epoch number
            model_path: Path to saved model
            metrics: Dictionary of epoch metrics
            model_name: Name of the model being trained
        """
        
        # Save to Redis
        if redis_client:
            try:
                epoch_key = f"metrics:epoch:{training_run_id}:{epoch}"
                cache_set(epoch_key, metrics, expire_seconds=86400)
                logger.info(f"📊 Cached epoch {epoch} metrics in Redis")
            except Exception as e:
                logger.warning(f"Failed to cache metrics: {e}")
        
        # Upload to MinIO
        if minio_client:
            try:
                model_safe_name = model_name.replace('/', '_')
                checkpoint_base = f"checkpoints/{model_safe_name}/{training_run_id}/epoch_{epoch}"
                
                model_path = Path(model_path)
                if model_path.exists():
                    if model_path.is_dir():
                        for model_file in model_path.iterdir():
                            if model_file.is_file():
                                remote_path = f"{checkpoint_base}/{model_file.name}"
                                upload_file(Settings.MINIO_BUCKET, remote_path, model_file)
                    else:
                        remote_path = f"{checkpoint_base}/{model_path.name}"
                        upload_file(Settings.MINIO_BUCKET, remote_path, model_path)
                    
                    logger.info(f"📤 Uploaded epoch {epoch} model to MinIO")
                
            except Exception as e:
                logger.warning(f"Failed to upload model: {e}")
    
    logger.info(f"Created epoch hook for training run: {training_run_id}")
    return epoch_hook


# Example usage
if __name__ == "__main__":
    # Example 1: Using the patch decorator
    print("Example of how to use the training integration patch:")
    print()
    print("# Method 1: Patch existing class")
    print("from training_integration_patch import patch_training_pipeline")
    print("from train_pipeline import TrainingPipeline")
    print()
    print("@patch_training_pipeline")
    print("class EnhancedTrainingPipeline(TrainingPipeline):")
    print("    pass")
    print()
    print("pipeline = EnhancedTrainingPipeline(")
    print("    enable_minio=True,")
    print("    enable_redis=True,")
    print("    resume_from_minio=True")
    print(")")
    print("pipeline.run_pipeline()")
    print()
    
    # Example 2: Using the simple hook
    print("# Method 2: Simple epoch hook")
    print("from training_integration_patch import create_simple_epoch_hook")
    print()
    print("epoch_hook = create_simple_epoch_hook()")
    print()
    print("# In your training loop:")
    print("for epoch in range(num_epochs):")
    print("    # ... training code ...")
    print("    model.save(f'model_epoch_{epoch}')")
    print("    metrics = {'loss': loss, 'accuracy': acc}")
    print("    epoch_hook(epoch, f'model_epoch_{epoch}', metrics, 'my-model')")
    print()
    
    # Test integration if available
    if HAS_THINKERBELL_UTILS:
        print("Testing integrations...")
        
        # Test Redis
        try:
            redis_client = get_redis_client()
            if redis_client.test_connection():
                print("✅ Redis connection successful")
            else:
                print("❌ Redis connection failed")
        except Exception as e:
            print(f"❌ Redis test error: {e}")
        
        # Test MinIO
        try:
            minio_client = get_minio_client()
            if minio_client.test_connection():
                print("✅ MinIO connection successful")
            else:
                print("❌ MinIO connection failed")
        except Exception as e:
            print(f"❌ MinIO test error: {e}")
    else:
        print("⚠️ Thinkerbell utilities not available for testing")



