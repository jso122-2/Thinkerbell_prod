# Thinkerbell Synthetic Data Generation Pipeline

A unified, production-ready synthetic data generation system for training AI document formatters with robust quality control and individual file management.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Check system status
python synthetic_data_launcher.py --status

# Test all components
python synthetic_data_launcher.py --test

# Generate validation batch (50 samples)
python synthetic_data_launcher.py --validation-batch 50

# Generate full dataset (individual files)
python synthetic_data_launcher.py --mode individual --samples 1000

# Generate full dataset (monolithic)
python synthetic_data_launcher.py --mode monolithic --samples 1000

# Run quality analysis on existing dataset
python synthetic_data_launcher.py --quality-analysis synthetic_dataset/
```

## 📁 Project Structure

```
Thinkerbell_template_pipeline/
├── synthetic_data_launcher.py      # 🎯 Main unified launcher
├── requirements.txt                 # 📦 All dependencies
├── README.md                       # 📖 This file
│
├── Core Generators
├── synthetic_dataset_generator.py   # 🔧 Main dataset generator
├── individual_file_generator.py     # 📄 Individual file generator
├── ood_contamination.py           # 🛡️ OOD contamination system
│
├── Quality Control
├── dataset_validation.py           # ✅ Comprehensive validation
├── dataset_loader.py               # 📂 Dataset loading utilities
├── semantic_smoother.py           # 🧠 Semantic coherence
├── text_preprocessor.py           # 📝 Text preprocessing
│
├── Testing & Validation
├── test_ood_robustness.py         # 🧪 OOD robustness tests
├── quick_validation.py            # ⚡ Quick validation tests
│
└── Legacy (Alternative Approaches)
├── generate_training_dataset.py    # 📊 Original monolithic approach
├── generate_individual_files.py    # 📄 Individual file approach
└── training_data_preparation.py   # 🎯 Training data preparation
```

## 🎯 Features

### ✅ **Unified Launcher**
- **Single entry point** for all synthetic data operations
- **Centralized imports** with graceful fallbacks
- **Comprehensive configuration** management
- **Component testing** and system status reporting

### ✅ **Individual File Generation**
- **Version control friendly** - Git can track individual changes
- **Debugging support** - Isolate problematic samples instantly
- **Manual review system** - Queue samples for human validation
- **Incremental generation** - Add samples without rebuilding
- **Parallel processing** - Process samples across multiple cores

### ✅ **Quality Control**
- **Distribution balance tracking** - Ensure realistic data distribution
- **Human extractability validation** - Verify humans can extract fields
- **Semantic coherence checking** - Ensure business logic makes sense
- **Token length management** - Optimize for transformer models
- **OOD contamination** - 20% non-agreement samples for robustness

### ✅ **Production Features**
- **Train/test splitting** at file level
- **Quality analysis** and problematic sample detection
- **Comprehensive metadata** tracking
- **Export for training** in various formats
- **Error handling** and graceful degradation

## 🛠️ Usage Examples

### Basic Generation
```bash
# Generate 1000 samples with individual files
python synthetic_data_launcher.py --mode individual --samples 1000

# Generate 500 samples as single file
python synthetic_data_launcher.py --mode monolithic --samples 500
```

### Quality Control
```bash
# Generate validation batch for manual review
python synthetic_data_launcher.py --validation-batch 100

# Run quality analysis on existing dataset
python synthetic_data_launcher.py --quality-analysis synthetic_dataset/

# Test all components
python synthetic_data_launcher.py --test
```

### Advanced Configuration
```python
# Custom configuration
launcher = SyntheticDataLauncher()
launcher.config["generation"]["total_samples"] = 2000
launcher.config["features"]["contamination_ratio"] = 0.3
launcher.run_complete_pipeline(mode="individual")
```

## 📊 Output Structure

### Individual File Mode
```
synthetic_dataset/
├── metadata/
│   ├── dataset_metadata.json
│   └── generation_config.json
├── samples/
│   ├── batch_001/
│   │   ├── sample_001_simple_fashion_inf.json
│   │   ├── sample_002_medium_food_inf.json
│   │   ├── sample_003_ood_employment_ood_negative.json
│   │   └── batch_metadata.json
│   └── batch_002/
├── validation/
│   ├── manual_review_queue/
│   └── approved_samples/
└── test_set/
    └── holdout_samples/
```

### Sample File Format
```json
{
  "sample_id": "sample_001",
  "generation_timestamp": "2025-01-20T10:30:00Z",
  "classification": {
    "document_type": "INFLUENCER_AGREEMENT",
    "confidence_target": 0.85,
    "complexity_level": "simple",
    "industry": "fashion"
  },
  "raw_input": {
    "text": "Need Sarah Chen for Cotton On spring campaign...",
    "token_count": 287,
    "text_style": "casual"
  },
  "extracted_fields": {
    "influencer": "Sarah Chen",
    "client": "Cotton On",
    "fee": "$8,500",
    "deliverables": ["3 x Instagram posts", "5 x Instagram stories"]
  },
  "validation_scores": {
    "semantic_coherence": 0.87,
    "field_extractability": 0.91
  }
}
```

## 🔧 Configuration

### Generation Settings
```python
"generation": {
    "total_samples": 1000,        # Total samples to generate
    "batch_size": 100,            # Samples per batch
    "validation_batch_size": 50,  # Validation samples
    "test_ratio": 0.2            # Test set percentage
}
```

### Feature Flags
```python
"features": {
    "use_semantic_smoothing": True,    # Business logic validation
    "use_text_preprocessing": True,    # Token length optimization
    "use_ood_contamination": True,    # Out-of-distribution samples
    "contamination_ratio": 0.2        # OOD sample percentage
}
```

### Quality Thresholds
```python
"quality": {
    "human_extraction_accuracy": 0.85,  # Human extractability
    "semantic_coherence_min": 0.6,      # Business logic coherence
    "token_length_max": 512,            # Transformer token limit
    "distribution_balance_min": 0.7     # Data distribution balance
}
```

## 🧪 Testing

### Component Testing
```bash
# Test all components
python synthetic_data_launcher.py --test

# Check system status
python synthetic_data_launcher.py --status
```

### Quality Validation
```python
from dataset_loader import DatasetLoader

# Load and analyze dataset
loader = DatasetLoader("synthetic_dataset/")
samples = loader.load_dataset(include_ood=True)
loader.print_stats()

# Find problematic samples
problematic = loader.find_problematic_samples(quality_threshold=0.6)
print(f"Found {len(problematic)} problematic samples")
```

## 📈 Quality Metrics

### Distribution Balance
- **Fee ranges**: 40% low ($1.5K-8K), 40% mid ($8K-18K), 20% premium ($18K-35K)
- **Complexity**: 50% simple, 30% medium, 20% complex
- **Industries**: Balanced across fashion, food, tech, home, beauty, automotive
- **Classification**: 80% influencer agreements, 20% OOD samples

### Quality Scores
- **Semantic coherence**: >0.6 (business logic makes sense)
- **Human extractability**: >0.85 (humans can extract fields)
- **Token length compliance**: >90% within 512 tokens
- **Distribution balance**: >0.7 (realistic data distribution)

## 🚀 Production Deployment

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Validate System
```bash
python synthetic_data_launcher.py --status
python synthetic_data_launcher.py --test
```

### 3. Generate Validation Batch
```bash
python synthetic_data_launcher.py --validation-batch 100
# Review samples in synthetic_dataset/validation/manual_review_queue/
```

### 4. Generate Full Dataset
```bash
python synthetic_data_launcher.py --mode individual --samples 1000
```

### 5. Quality Analysis
```bash
python synthetic_data_launcher.py --quality-analysis synthetic_dataset/
```

## 🔍 Troubleshooting

### Common Issues

**Import Errors**: Some components may not be available
```bash
# Check what's available
python synthetic_data_launcher.py --status
```

**Quality Issues**: Poor semantic coherence or extractability
```bash
# Run quality analysis
python synthetic_data_launcher.py --quality-analysis synthetic_dataset/
```

**Memory Issues**: Large datasets causing memory problems
```bash
# Use smaller batch sizes
python synthetic_data_launcher.py --mode individual --samples 500 --batch-size 50
```

### Performance Optimization

- **Parallel processing**: Use individual file mode for large datasets
- **Batch processing**: Adjust batch size based on memory
- **Quality filtering**: Use quality thresholds to filter samples
- **Incremental generation**: Add samples without rebuilding

## 📚 API Reference

### SyntheticDataLauncher
Main unified launcher class with all features.

```python
launcher = SyntheticDataLauncher()

# Generate validation batch
validation_samples = launcher.generate_validation_batch(size=50)

# Generate individual files dataset
result = launcher.generate_individual_files(total_samples=1000)

# Generate monolithic dataset
result = launcher.generate_monolithic_dataset(total_samples=1000)

# Run quality analysis
analysis = launcher.run_quality_analysis("synthetic_dataset/")

# Test components
success = launcher.test_all_components()
```

### DatasetLoader
Load and analyze individual file datasets.

```python
loader = DatasetLoader("synthetic_dataset/")

# Load with filters
samples = loader.load_dataset(
    include_ood=True,
    complexity_filter=["simple", "medium"],
    quality_threshold=0.7
)

# Create train/test split
train_samples, test_samples = loader.create_train_test_split(test_ratio=0.2)

# Export for training
training_file = loader.export_for_training("training_dataset.json")
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests** for new functionality
4. **Run quality checks** with `python synthetic_data_launcher.py --test`
5. **Submit a pull request**

## 📄 License

This project is part of the Thinkerbell AI Document Formatter system.

---

**🎯 Ready to generate production-quality synthetic data for your AI document formatter!** 