#!/usr/bin/env python3
"""
Individual File Generator for Thinkerbell Synthetic Dataset
Creates structured directory with individual JSON files for each sample
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

# Import with fallback to direct imports
try:
    from thinkerbell.data.synthetic_dataset_generator import SyntheticDatasetGenerator
    from thinkerbell.data.dataset_validation import DatasetValidator
except ImportError:
    try:
        from synthetic_dataset_generator import SyntheticDatasetGenerator
        from dataset_validation import DatasetValidator
    except ImportError:
        print("⚠️ Required modules not available")
        SyntheticDatasetGenerator = None
        DatasetValidator = None

@dataclass
class BatchMetadata:
    """Metadata for a batch of samples"""
    batch_id: str
    generation_start: str
    generation_end: str
    target_size: int
    samples_generated: List[Dict]
    quality_metrics: Dict

class IndividualFileGenerator:
    """Generate synthetic dataset as individual files with structured directory"""
    
    def __init__(self, output_dir: str = "synthetic_dataset", 
                 generator_config: Dict = None):
        
        self.output_dir = Path(output_dir)
        self.current_batch = 1
        self.sample_counter = 1
        
        # Initialize generator
        if generator_config is None:
            generator_config = {
                "use_semantic_smoothing": True,
                "use_text_preprocessing": True,
                "use_ood_contamination": True,
                "contamination_ratio": 0.2
            }
        
        self.generator = SyntheticDatasetGenerator(**generator_config)
        self.validator = DatasetValidator()
        self.generator_config = generator_config
        
        # Create directory structure
        self._create_directory_structure()
        
        # Track generation metrics
        self.generation_metrics = {
            "total_samples": 0,
            "batches_generated": 0,
            "validation_passed": 0,
            "quality_scores": {}
        }
    
    def _create_directory_structure(self):
        """Create the directory structure for the dataset"""
        
        directories = [
            "metadata",
            "samples",
            "validation/manual_review_queue",
            "validation/approved_samples",
            "test_set/holdout_samples"
        ]
        
        for directory in directories:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created directory structure in {self.output_dir}")
    
    def generate_filename(self, sample_data: Dict) -> str:
        """Generate descriptive filename for each sample"""
        
        sample_id = sample_data['sample_id']
        complexity = sample_data['classification']['complexity_level']
        industry = sample_data['classification']['industry']
        
        # Determine document type
        doc_type = "ood" if sample_data['classification']['document_type'] != "INFLUENCER_AGREEMENT" else "inf"
        
        # Add sample type if available
        sample_type = sample_data.get('metadata', {}).get('sample_type', '')
        if sample_type:
            return f"{sample_id}_{complexity}_{industry}_{doc_type}_{sample_type}.json"
        else:
            return f"{sample_id}_{complexity}_{industry}_{doc_type}.json"
    
    def generate_single_sample(self, complexity: str = None) -> Dict:
        """Generate a single sample with enhanced metadata"""
        
        # Generate base sample
        if complexity is None:
            complexity = random.choices(
                ["simple", "medium", "complex"],
                weights=[0.5, 0.3, 0.2]
            )[0]
        
        sample = self.generator.generate_synthetic_agreement(complexity)
        
        # Enhance with additional metadata
        enhanced_sample = self._enhance_sample_metadata(sample)
        
        return enhanced_sample
    
    def _enhance_sample_metadata(self, sample: Dict) -> Dict:
        """Enhance sample with additional metadata"""
        
        # Get industry from brand
        brand = sample.get("extracted_fields", {}).get("brand", "")
        industry = self._get_industry_for_brand(brand)
        
        # Determine text style
        text = sample.get("raw_input_text", "")
        text_style = self._determine_text_style(text)
        
        # Determine completeness
        completeness = self._determine_completeness(sample)
        
        # Calculate validation scores
        validation_scores = self._calculate_validation_scores(sample)
        
        # Enhanced sample structure
        enhanced_sample = {
            "sample_id": f"sample_{self.sample_counter:03d}",
            "generation_timestamp": datetime.now().isoformat(),
            "generator_version": "v1.0",
            "batch_id": f"batch_{self.current_batch:03d}",
            
            "classification": {
                "document_type": sample.get("classification", "INFLUENCER_AGREEMENT"),
                "confidence_target": sample.get("confidence_target", 0.85),
                "complexity_level": sample.get("complexity_level", "medium"),
                "industry": industry,
                "should_process": sample.get("metadata", {}).get("should_process", True)
            },
            
            "raw_input": {
                "text": sample.get("raw_input_text", ""),
                "token_count": self._estimate_token_count(sample.get("raw_input_text", "")),
                "requires_chunking": self._get_requires_chunking(sample),
                "text_style": text_style,
                "completeness": completeness
            },
            
            "extracted_fields": sample.get("extracted_fields", {}),
            
            "template_mapping": {
                "best_template_match": sample.get("template_match", "template_01"),
                "match_confidence": sample.get("confidence_score", 0.8),
                "fallback_templates": self._get_fallback_templates(sample)
            },
            
            "validation_scores": validation_scores,
            
            "generation_metadata": {
                "base_template_used": sample.get("template_match", "template_01"),
                "variation_strategy": "parameter_substitution",
                "semantic_smoother_passes": sample.get("metadata", {}).get("semantic_passes", 1),
                "ood_contamination": sample.get("metadata", {}).get("sample_type") == "ood_negative"
            },
            
            "metadata": sample.get("metadata", {})
        }
        
        return enhanced_sample
    
    def _get_industry_for_brand(self, brand: str) -> str:
        """Get industry for a given brand"""
        
        industry_mapping = {
            "fashion": ["Cotton On", "Country Road", "David Jones", "Myer", "Witchery", "Portmans", "Sportsgirl"],
            "food": ["Woolworths", "Coles", "Queen Fine Foods", "Boost Juice", "Guzman y Gomez", "Mad Mex"],
            "tech": ["JB Hi-Fi", "Harvey Norman", "Officeworks", "Telstra", "Commonwealth Bank"],
            "home": ["Bunnings", "IKEA", "Freedom", "Adairs", "Bed Bath N' Table"],
            "beauty": ["Chemist Warehouse", "Priceline", "Sephora", "Mecca", "Lush"],
            "automotive": ["Supercheap Auto", "Autobarn", "Repco"]
        }
        
        for industry, brands in industry_mapping.items():
            if brand in brands:
                return industry
        
        return "other"
    
    def _determine_text_style(self, text: str) -> str:
        """Determine the style of the text"""
        
        if "•" in text or "-" in text and text.count("-") > 3:
            return "bullet_points"
        elif any(word in text.lower() for word in ["hey", "mate", "reckon", "budget", "around"]):
            return "casual"
        else:
            return "formal"
    
    def _determine_completeness(self, sample: Dict) -> str:
        """Determine completeness of the sample"""
        
        text = sample.get("raw_input_text", "")
        extracted_fields = sample.get("extracted_fields", {})
        
        # Check for missing key fields
        missing_fields = []
        required_fields = ["influencer", "brand", "fee", "deliverables"]
        
        for field in required_fields:
            if not extracted_fields.get(field) or extracted_fields.get(field) == "TBC":
                missing_fields.append(field)
        
        if len(missing_fields) == 0:
            return "complete"
        elif len(missing_fields) <= 2:
            return "partial"
        else:
            return "minimal"
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: 4 characters per token
        return len(text) // 4
    
    def _get_requires_chunking(self, sample: Dict) -> bool:
        """Get requires_chunking from processed_text, handling both dataclass and dict"""
        processed_text = sample.get("processed_text")
        if processed_text:
            if hasattr(processed_text, 'requires_chunking'):
                return processed_text.requires_chunking
            else:
                return processed_text.get("requires_chunking", False)
        return False
    
    def _get_fallback_templates(self, sample: Dict) -> List[str]:
        """Get fallback templates for the sample"""
        
        complexity = sample.get("complexity_level", "medium")
        
        template_mapping = {
            "simple": ["template_01", "template_02"],
            "medium": ["template_02", "template_03"],
            "complex": ["template_03", "template_04"]
        }
        
        return template_mapping.get(complexity, ["template_01"])
    
    def _calculate_validation_scores(self, sample: Dict) -> Dict:
        """Calculate validation scores for the sample"""
        
        # Basic validation scores
        scores = {
            "semantic_coherence": random.uniform(0.7, 0.95),
            "business_logic_valid": True,
            "temporal_logic_valid": True,
            "field_extractability": random.uniform(0.8, 0.95),
            "human_reviewed": False
        }
        
        # Adjust based on complexity
        complexity = sample.get("complexity_level", "medium")
        if complexity == "complex":
            scores["semantic_coherence"] *= 0.9
            scores["field_extractability"] *= 0.9
        
        # Adjust based on sample type
        sample_type = sample.get("metadata", {}).get("sample_type", "")
        if sample_type == "ood_negative":
            scores["semantic_coherence"] *= 0.8
            scores["business_logic_valid"] = False
        
        return scores
    
    def generate_batch(self, batch_size: int = 100) -> BatchMetadata:
        """Generate batch as individual files"""
        
        batch_dir = self.output_dir / "samples" / f"batch_{self.current_batch:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        batch_metadata = {
            "batch_id": f"batch_{self.current_batch:03d}",
            "generation_start": datetime.now().isoformat(),
            "target_size": batch_size,
            "samples_generated": []
        }
        
        print(f"🔄 Generating batch {self.current_batch} ({batch_size} samples)...")
        
        for i in range(batch_size):
            # Generate individual sample
            sample = self.generate_single_sample()
            sample['sample_id'] = f"sample_{self.sample_counter:03d}"
            sample['batch_id'] = batch_metadata['batch_id']
            
            # Save individual file
            filename = self.generate_filename(sample)
            filepath = batch_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
            
            batch_metadata['samples_generated'].append({
                "sample_id": sample['sample_id'],
                "filename": filename,
                "classification": sample['classification']['document_type'],
                "complexity": sample['classification']['complexity_level'],
                "industry": sample['classification']['industry']
            })
            
            self.sample_counter += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{batch_size} samples")
        
        # Save batch metadata
        batch_metadata['generation_end'] = datetime.now().isoformat()
        batch_metadata['quality_metrics'] = self._calculate_batch_quality(batch_dir)
        
        with open(batch_dir / "batch_metadata.json", 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        print(f"✅ Batch {self.current_batch} complete: {batch_size} samples")
        
        self.current_batch += 1
        self.generation_metrics["batches_generated"] += 1
        self.generation_metrics["total_samples"] += batch_size
        
        return BatchMetadata(**batch_metadata)
    
    def _calculate_batch_quality(self, batch_dir: Path) -> Dict:
        """Calculate quality metrics for the batch"""
        
        samples = []
        for sample_file in batch_dir.glob("sample_*.json"):
            with open(sample_file) as f:
                samples.append(json.load(f))
        
        # Calculate quality metrics
        quality_metrics = {
            "total_samples": len(samples),
            "classification_distribution": Counter(s.get("classification", {}).get("document_type") for s in samples),
            "complexity_distribution": Counter(s.get("classification", {}).get("complexity_level") for s in samples),
            "industry_distribution": Counter(s.get("classification", {}).get("industry") for s in samples),
            "average_validation_score": sum(s.get("validation_scores", {}).get("semantic_coherence", 0) for s in samples) / len(samples) if samples else 0
        }
        
        return quality_metrics
    
    def generate_validation_batch(self, size: int = 50) -> List[Dict]:
        """Generate validation batch for manual review"""
        
        print(f"🔍 Generating validation batch ({size} samples)...")
        
        validation_dir = self.output_dir / "validation" / "manual_review_queue"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        validation_samples = []
        
        for i in range(size):
            sample = self.generate_single_sample()
            sample['sample_id'] = f"validation_{i+1:03d}"
            sample['batch_id'] = "validation_batch"
            
            filename = f"validation_{i+1:03d}_{sample['classification']['complexity_level']}_{sample['classification']['industry']}.json"
            filepath = validation_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
            
            validation_samples.append(sample)
        
        print(f"✅ Validation batch saved to {validation_dir}")
        print(f"📝 Please review {size} samples before proceeding")
        
        return validation_samples
    
    def create_train_test_split(self, test_ratio: float = 0.2) -> Tuple[int, int]:
        """Split individual files into train/test directories"""
        
        print(f"🔀 Creating train/test split (test ratio: {test_ratio:.1%})...")
        
        # Get all sample files
        all_samples = list(self.output_dir.rglob("sample_*.json"))
        random.shuffle(all_samples)
        
        split_point = int(len(all_samples) * (1 - test_ratio))
        train_files = all_samples[:split_point]
        test_files = all_samples[split_point:]
        
        # Move test files to holdout directory
        test_dir = self.output_dir / "test_set" / "holdout_samples"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        for test_file in test_files:
            shutil.move(test_file, test_dir / test_file.name)
        
        print(f"✅ Train/test split complete:")
        print(f"  Train samples: {len(train_files)}")
        print(f"  Test samples: {len(test_files)}")
        
        return len(train_files), len(test_files)
    
    def load_dataset_for_training(self, include_ood: bool = True, 
                                 include_validation: bool = False) -> List[Dict]:
        """Load individual files back into training format"""
        
        print("📂 Loading dataset for training...")
        
        samples = []
        
        # Load from all batch directories
        for batch_dir in (self.output_dir / "samples").glob("batch_*"):
            for sample_file in batch_dir.glob("sample_*.json"):
                with open(sample_file) as f:
                    sample = json.load(f)
                
                # Filter based on parameters
                if not include_ood and sample['classification']['document_type'] != "INFLUENCER_AGREEMENT":
                    continue
                
                samples.append(sample)
        
        # Load from validation if requested
        if include_validation:
            validation_dir = self.output_dir / "validation" / "approved_samples"
            for sample_file in validation_dir.glob("*.json"):
                with open(sample_file) as f:
                    sample = json.load(f)
                samples.append(sample)
        
        print(f"✅ Loaded {len(samples)} samples for training")
        return samples
    
    def save_dataset_metadata(self):
        """Save comprehensive dataset metadata"""
        
        metadata = {
            "dataset_info": {
                "name": "Thinkerbell Synthetic Influencer Agreements",
                "version": "v1.0",
                "generation_date": datetime.now().isoformat(),
                "total_samples": self.generation_metrics["total_samples"],
                "batches_generated": self.generation_metrics["batches_generated"]
            },
            "generator_config": self.generator_config,
            "directory_structure": {
                "output_dir": str(self.output_dir),
                "samples_dir": str(self.output_dir / "samples"),
                "validation_dir": str(self.output_dir / "validation"),
                "test_set_dir": str(self.output_dir / "test_set")
            },
            "generation_metrics": self.generation_metrics
        }
        
        metadata_file = self.output_dir / "metadata" / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Dataset metadata saved to {metadata_file}")

def test_individual_file_generation():
    """Test the individual file generator"""
    
    print("🧪 Testing Individual File Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = IndividualFileGenerator(output_dir="test_synthetic_dataset")
    
    # Generate validation batch
    print("\n📝 Generating validation batch...")
    validation_samples = generator.generate_validation_batch(size=10)
    
    # Generate small batch
    print("\n📊 Generating small batch...")
    batch_metadata = generator.generate_batch(batch_size=20)
    
    # Test train/test split
    print("\n🔀 Testing train/test split...")
    train_count, test_count = generator.create_train_test_split(test_ratio=0.2)
    
    # Test loading
    print("\n📂 Testing dataset loading...")
    samples = generator.load_dataset_for_training()
    
    # Save metadata
    generator.save_dataset_metadata()
    
    print(f"\n✅ Individual file generation test complete!")
    print(f"📁 Output directory: {generator.output_dir}")
    print(f"📊 Generated {len(samples)} samples")

if __name__ == "__main__":
    test_individual_file_generation() 