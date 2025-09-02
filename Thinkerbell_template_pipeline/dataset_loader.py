#!/usr/bin/env python3
"""
Dataset Loader for Thinkerbell Synthetic Dataset
Loads individual files back into training format with filtering and analysis
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DatasetStats:
    """Statistics for loaded dataset"""
    total_samples: int
    classification_distribution: Dict[str, int]
    complexity_distribution: Dict[str, int]
    industry_distribution: Dict[str, int]
    quality_metrics: Dict[str, float]
    file_count: int
    batch_count: int

class DatasetLoader:
    """Load and analyze individual file dataset"""
    
    def __init__(self, dataset_dir: str = "synthetic_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.loaded_samples = []
        self.stats = None
    
    def load_dataset(self, include_ood: bool = True, 
                    include_validation: bool = False,
                    complexity_filter: Optional[List[str]] = None,
                    industry_filter: Optional[List[str]] = None,
                    quality_threshold: float = 0.0) -> List[Dict]:
        """Load dataset with filtering options"""
        
        print(f"📂 Loading dataset from {self.dataset_dir}")
        
        samples = []
        
        # Load from batch directories
        batch_dirs = list((self.dataset_dir / "samples").glob("batch_*"))
        print(f"  Found {len(batch_dirs)} batch directories")
        
        for batch_dir in batch_dirs:
            sample_files = list(batch_dir.glob("sample_*.json"))
            print(f"  Loading {len(sample_files)} samples from {batch_dir.name}")
            
            for sample_file in sample_files:
                try:
                    with open(sample_file) as f:
                        sample = json.load(f)
                    
                    # Apply filters
                    if not self._passes_filters(sample, include_ood, complexity_filter, 
                                              industry_filter, quality_threshold):
                        continue
                    
                    samples.append(sample)
                    
                except Exception as e:
                    print(f"  ⚠️  Error loading {sample_file}: {e}")
        
        # Load from validation if requested
        if include_validation:
            validation_dir = self.dataset_dir / "validation" / "approved_samples"
            if validation_dir.exists():
                validation_files = list(validation_dir.glob("*.json"))
                print(f"  Loading {len(validation_files)} validation samples")
                
                for validation_file in validation_files:
                    try:
                        with open(validation_file) as f:
                            sample = json.load(f)
                        
                        if self._passes_filters(sample, include_ood, complexity_filter,
                                              industry_filter, quality_threshold):
                            samples.append(sample)
                    
                    except Exception as e:
                        print(f"  ⚠️  Error loading {validation_file}: {e}")
        
        self.loaded_samples = samples
        self.stats = self._calculate_stats(samples)
        
        print(f"✅ Loaded {len(samples)} samples")
        return samples
    
    def _passes_filters(self, sample: Dict, include_ood: bool,
                       complexity_filter: Optional[List[str]],
                       industry_filter: Optional[List[str]],
                       quality_threshold: float) -> bool:
        """Check if sample passes all filters"""
        
        # OOD filter
        if not include_ood and sample.get('classification', {}).get('document_type') != "INFLUENCER_AGREEMENT":
            return False
        
        # Complexity filter
        if complexity_filter:
            complexity = sample.get('classification', {}).get('complexity_level', '')
            if complexity not in complexity_filter:
                return False
        
        # Industry filter
        if industry_filter:
            industry = sample.get('classification', {}).get('industry', '')
            if industry not in industry_filter:
                return False
        
        # Quality threshold
        if quality_threshold > 0:
            validation_score = sample.get('validation_scores', {}).get('semantic_coherence', 0)
            if validation_score < quality_threshold:
                return False
        
        return True
    
    def _calculate_stats(self, samples: List[Dict]) -> DatasetStats:
        """Calculate comprehensive statistics for the dataset"""
        
        if not samples:
            return DatasetStats(0, {}, {}, {}, {}, 0, 0)
        
        # Basic counts
        total_samples = len(samples)
        
        # Classification distribution
        classification_dist = Counter()
        complexity_dist = Counter()
        industry_dist = Counter()
        
        # Quality metrics
        quality_scores = defaultdict(list)
        
        # Batch tracking
        batch_ids = set()
        
        for sample in samples:
            # Classification stats
            doc_type = sample.get('classification', {}).get('document_type', 'UNKNOWN')
            classification_dist[doc_type] += 1
            
            # Complexity stats
            complexity = sample.get('classification', {}).get('complexity_level', 'UNKNOWN')
            complexity_dist[complexity] += 1
            
            # Industry stats
            industry = sample.get('classification', {}).get('industry', 'UNKNOWN')
            industry_dist[industry] += 1
            
            # Quality stats
            validation_scores = sample.get('validation_scores', {})
            for metric, value in validation_scores.items():
                if isinstance(value, (int, float)):
                    quality_scores[metric].append(value)
            
            # Batch tracking
            batch_id = sample.get('batch_id', 'unknown')
            batch_ids.add(batch_id)
        
        # Calculate average quality metrics
        avg_quality_metrics = {}
        for metric, values in quality_scores.items():
            if values:
                avg_quality_metrics[f"avg_{metric}"] = sum(values) / len(values)
                avg_quality_metrics[f"min_{metric}"] = min(values)
                avg_quality_metrics[f"max_{metric}"] = max(values)
        
        return DatasetStats(
            total_samples=total_samples,
            classification_distribution=dict(classification_dist),
            complexity_distribution=dict(complexity_dist),
            industry_distribution=dict(industry_dist),
            quality_metrics=avg_quality_metrics,
            file_count=total_samples,
            batch_count=len(batch_ids)
        )
    
    def print_stats(self):
        """Print comprehensive dataset statistics"""
        
        if not self.stats:
            print("❌ No dataset loaded. Call load_dataset() first.")
            return
        
        print("\n📊 Dataset Statistics")
        print("=" * 50)
        print(f"Total samples: {self.stats.total_samples}")
        print(f"Batches: {self.stats.batch_count}")
        
        print(f"\n📋 Classification Distribution:")
        for classification, count in self.stats.classification_distribution.items():
            percentage = (count / self.stats.total_samples) * 100
            print(f"  {classification}: {count} ({percentage:.1f}%)")
        
        print(f"\n📊 Complexity Distribution:")
        for complexity, count in self.stats.complexity_distribution.items():
            percentage = (count / self.stats.total_samples) * 100
            print(f"  {complexity}: {count} ({percentage:.1f}%)")
        
        print(f"\n🏭 Industry Distribution:")
        for industry, count in self.stats.industry_distribution.items():
            percentage = (count / self.stats.total_samples) * 100
            print(f"  {industry}: {count} ({percentage:.1f}%)")
        
        print(f"\n📈 Quality Metrics:")
        for metric, value in self.stats.quality_metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    def create_train_test_split(self, test_ratio: float = 0.2, 
                               random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """Create train/test split from loaded samples"""
        
        if not self.loaded_samples:
            print("❌ No samples loaded. Call load_dataset() first.")
            return [], []
        
        print(f"🔀 Creating train/test split (test ratio: {test_ratio:.1%})...")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Shuffle samples
        shuffled_samples = self.loaded_samples.copy()
        random.shuffle(shuffled_samples)
        
        # Split
        split_point = int(len(shuffled_samples) * (1 - test_ratio))
        train_samples = shuffled_samples[:split_point]
        test_samples = shuffled_samples[split_point:]
        
        print(f"✅ Split complete:")
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Test samples: {len(test_samples)}")
        
        return train_samples, test_samples
    
    def export_for_training(self, output_file: str = "training_dataset.json",
                           include_metadata: bool = True) -> str:
        """Export loaded samples in training format"""
        
        if not self.loaded_samples:
            print("❌ No samples loaded. Call load_dataset() first.")
            return ""
        
        # Prepare training format
        training_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_samples": len(self.loaded_samples),
                "stats": {
                    "classification_distribution": self.stats.classification_distribution,
                    "complexity_distribution": self.stats.complexity_distribution,
                    "industry_distribution": self.stats.industry_distribution,
                    "quality_metrics": self.stats.quality_metrics
                }
            } if include_metadata else {},
            "samples": self.loaded_samples
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(self.loaded_samples)} samples to {output_file}")
        return output_file
    
    def find_problematic_samples(self, quality_threshold: float = 0.6) -> List[Dict]:
        """Find samples with quality issues"""
        
        if not self.loaded_samples:
            print("❌ No samples loaded. Call load_dataset() first.")
            return []
        
        problematic_samples = []
        
        for sample in self.loaded_samples:
            validation_scores = sample.get('validation_scores', {})
            
            # Check for quality issues
            issues = []
            
            semantic_coherence = validation_scores.get('semantic_coherence', 1.0)
            if semantic_coherence < quality_threshold:
                issues.append(f"Low semantic coherence: {semantic_coherence:.3f}")
            
            field_extractability = validation_scores.get('field_extractability', 1.0)
            if field_extractability < quality_threshold:
                issues.append(f"Low field extractability: {field_extractability:.3f}")
            
            if not validation_scores.get('business_logic_valid', True):
                issues.append("Business logic violation")
            
            if not validation_scores.get('temporal_logic_valid', True):
                issues.append("Temporal logic violation")
            
            if issues:
                problematic_samples.append({
                    "sample_id": sample.get('sample_id', 'unknown'),
                    "issues": issues,
                    "validation_scores": validation_scores,
                    "classification": sample.get('classification', {}),
                    "raw_input_preview": sample.get('raw_input', {}).get('text', '')[:100] + "..."
                })
        
        print(f"🔍 Found {len(problematic_samples)} problematic samples")
        return problematic_samples
    
    def analyze_sample_quality(self) -> Dict:
        """Analyze overall sample quality"""
        
        if not self.loaded_samples:
            print("❌ No samples loaded. Call load_dataset() first.")
            return {}
        
        quality_analysis = {
            "total_samples": len(self.loaded_samples),
            "quality_distribution": {
                "excellent": 0,  # > 0.9
                "good": 0,       # 0.7-0.9
                "fair": 0,       # 0.5-0.7
                "poor": 0        # < 0.5
            },
            "issue_types": Counter(),
            "recommendations": []
        }
        
        for sample in self.loaded_samples:
            validation_scores = sample.get('validation_scores', {})
            semantic_coherence = validation_scores.get('semantic_coherence', 0)
            
            # Categorize quality
            if semantic_coherence > 0.9:
                quality_analysis["quality_distribution"]["excellent"] += 1
            elif semantic_coherence > 0.7:
                quality_analysis["quality_distribution"]["good"] += 1
            elif semantic_coherence > 0.5:
                quality_analysis["quality_distribution"]["fair"] += 1
            else:
                quality_analysis["quality_distribution"]["poor"] += 1
            
            # Track issues
            if not validation_scores.get('business_logic_valid', True):
                quality_analysis["issue_types"]["business_logic"] += 1
            if not validation_scores.get('temporal_logic_valid', True):
                quality_analysis["issue_types"]["temporal_logic"] += 1
        
        # Generate recommendations
        poor_count = quality_analysis["quality_distribution"]["poor"]
        if poor_count > len(self.loaded_samples) * 0.1:
            quality_analysis["recommendations"].append("High number of poor quality samples - review generator logic")
        
        if quality_analysis["issue_types"]["business_logic"] > 0:
            quality_analysis["recommendations"].append("Business logic violations detected - improve semantic smoother")
        
        return quality_analysis
    
    def get_sample_by_id(self, sample_id: str) -> Optional[Dict]:
        """Get a specific sample by ID"""
        
        for sample in self.loaded_samples:
            if sample.get('sample_id') == sample_id:
                return sample
        
        return None
    
    def get_samples_by_batch(self, batch_id: str) -> List[Dict]:
        """Get all samples from a specific batch"""
        
        return [sample for sample in self.loaded_samples if sample.get('batch_id') == batch_id]

def test_dataset_loader():
    """Test the dataset loader functionality"""
    
    print("🧪 Testing Dataset Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = DatasetLoader("test_synthetic_dataset")
    
    # Try to load dataset
    try:
        samples = loader.load_dataset(include_ood=True, quality_threshold=0.0)
        
        if samples:
            # Print stats
            loader.print_stats()
            
            # Test train/test split
            train_samples, test_samples = loader.create_train_test_split(test_ratio=0.2)
            
            # Test export
            output_file = loader.export_for_training("test_training_dataset.json")
            
            # Test quality analysis
            quality_analysis = loader.analyze_sample_quality()
            print(f"\n📊 Quality Analysis:")
            for category, count in quality_analysis["quality_distribution"].items():
                print(f"  {category}: {count}")
            
            # Test problematic sample detection
            problematic = loader.find_problematic_samples(quality_threshold=0.6)
            if problematic:
                print(f"\n⚠️  Found {len(problematic)} problematic samples")
                for sample in problematic[:3]:  # Show first 3
                    print(f"  {sample['sample_id']}: {', '.join(sample['issues'])}")
            
            print(f"\n✅ Dataset loader test complete!")
        else:
            print("❌ No samples found in test directory")
    
    except Exception as e:
        print(f"❌ Dataset loader test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_loader() 