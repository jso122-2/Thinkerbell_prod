# Synthetic Influencer Agreement Dataset Generator

## Overview

This pipeline generates realistic synthetic variations of influencer agreements for training Thinkerbell's AI document formatter. The system creates ~1000 synthetic examples with systematic parameter variations and **semantic smoothing** to ensure business logic coherence.

### Key Features

1. **Content Classification** - Distinguish influencer agreements from other content
2. **Information Extraction** - Extract key fields (influencer, client, fee, deliverables, etc.)
3. **Template Matching** - Match content to appropriate template styles
4. **🧠 Semantic Smoothing** - Ensures business logic coherence and realistic scenarios

## Quick Start

### 1. Install Dependencies

```bash
pip install -r training_requirements.txt
```

### 2. Generate Dataset

```bash
python generate_training_dataset.py
```

This will create:
- `synthetic_influencer_agreements.json` - Raw synthetic dataset with semantic smoothing
- `training_data/` - Prepared training datasets for each task

### 3. Test Semantic Smoother

```bash
python test_semantic_smoother.py
```

## 🧠 Semantic Smoother Architecture

### Problem Solved
Random dictionary combinations create unrealistic business scenarios:
- $25K budget for 1 Instagram story
- McDonald's doing luxury skincare campaigns  
- 2-week campaigns with 12-month exclusivity
- Tech influencers promoting baby food

### Solution Components

#### 1. Business Logic Constraints

**Fee-Deliverable Coherence:**
```python
FEE_LOGIC = {
    "simple": {  # 1-2 posts, basic content
        "fee_range": (1500, 8000),
        "deliverables": ["1-2 posts", "stories", "single video"]
    },
    "medium": {  # Multi-platform, some exclusivity
        "fee_range": (8000, 18000), 
        "deliverables": ["3-5 posts", "interviews", "events", "multi-platform"]
    },
    "premium": { # Major campaigns, high exclusivity
        "fee_range": (18000, 35000),
        "deliverables": ["shoots", "TV appearances", "long exclusivity", "major events"]
    }
}
```

**Industry-Brand Alignment:**
```python
INDUSTRY_COHERENCE = {
    "fashion": {
        "brands": ["Cotton On", "Country Road", "David Jones"],
        "products": ["spring collection", "winter range", "accessories"],
        "exclusivity_scope": ["fashion brands", "retail competitors"],
        "typical_influencers": ["fashion", "lifestyle", "beauty"]
    },
    "food": {
        "brands": ["Woolworths", "Coles", "Queen Fine Foods"],
        "products": ["new product launch", "seasonal campaign", "recipe content"],
        "exclusivity_scope": ["grocery competitors", "food brands"],
        "typical_influencers": ["food", "cooking", "family lifestyle"]
    }
}
```

#### 2. Temporal Logic Validator

**Campaign-Exclusivity Coherence:**
```python
def validate_temporal_logic(engagement_term, exclusivity_period, usage_term):
    """Ensure realistic timing relationships"""
    
    # Exclusivity shouldn't be much longer than engagement for small campaigns
    if engagement_term <= 2 and exclusivity_period > 6:
        return False, "Short campaign with excessive exclusivity"
    
    # Usage terms should align with campaign value
    if fee < 5000 and usage_term > 12:
        return False, "Low fee with extended usage rights"
        
    return True, "Valid timing"
```

#### 3. Semantic Embedding Validator

**Use sentence-transformer to check "business sense":**
```python
def semantic_coherence_check(brand, product, influencer_type, deliverables):
    """Check if combination makes business sense"""
    
    # Create business scenario description
    scenario = f"{brand} {product} campaign with {influencer_type} influencer doing {deliverables}"
    
    # Compare against known good/bad combinations
    good_examples = [
        "Woolworths grocery campaign with food influencer doing recipe videos",
        "Cotton On fashion campaign with style influencer doing outfit posts"
    ]
    
    bad_examples = [
        "McDonald's luxury skincare campaign with tech influencer doing dance videos",
        "Banking app campaign with pet influencer doing makeup tutorials"
    ]
    
    # Semantic similarity scoring
    coherence_score = good_similarity - bad_similarity
    return coherence_score > 0.1, coherence_score
```

#### 4. Quality Metrics Dashboard

Track semantic smoother effectiveness:
```python
QUALITY_METRICS = {
    "business_logic_pass_rate": 0.85,    # 85% pass business rules
    "semantic_coherence_avg": 0.73,      # Average coherence score  
    "manual_review_quality": 0.91,       # Human validation score
    "generation_efficiency": 0.68        # Samples that pass all filters
}
```

## Dataset Structure

### Raw Synthetic Data Format

Each synthetic agreement contains:

```json
{
  "id": "synth_abc12345",
  "raw_input_text": "Need Sarah Chen for Woolworths spring campaign...",
  "extracted_fields": {
    "influencer": "Sarah Chen",
    "client": "Woolworths",
    "brand": "Woolworths", 
    "campaign": "Spring Campaign",
    "fee": "$8,000",
    "fee_numeric": 8000,
    "deliverables": ["2 x Instagram posts", "3 x Instagram stories"],
    "exclusivity_period": "4 weeks",
    "exclusivity_scope": "grocery brands",
    "engagement_term": "2 months",
    "usage_term": "6 months",
    "territory": "Australia",
    "start_date": "March 2025"
  },
  "template_match": "template_style_2",
  "complexity_level": "medium",
  "confidence_score": 0.85,
  "metadata": {
    "fee_tier": "mid",
    "num_deliverables": 2,
    "campaign_type": "Spring Collection Launch",
    "product_category": "Food & Beverage",
    "industry": "food"
  }
}
```

### Training Data Structure

The pipeline creates separate datasets for each task:

#### Classification Task
- `training_data/classification_train.json`
- `training_data/classification_validation.json` 
- `training_data/classification_test.json`

Format:
```json
{
  "text": "raw input text",
  "label": 1,  // 1 = influencer agreement, 0 = other content
  "confidence": 0.85,
  "complexity": "medium"
}
```

#### Extraction Task
- `training_data/extraction_train.json`
- `training_data/extraction_validation.json`
- `training_data/extraction_test.json`

Format:
```json
{
  "text": "raw input text", 
  "field": "influencer",
  "value": "Sarah Chen",
  "confidence": 0.85
}
```

#### Template Matching Task
- `training_data/template_matching_train.json`
- `training_data/template_matching_validation.json`
- `training_data/template_matching_test.json`

Format:
```json
{
  "text": "raw input text",
  "template": "template_style_2", 
  "confidence": 0.85,
  "complexity": "medium"
}
```

## Parameter Variations

### Australian Brands/Clients (Organized by Industry)
- **Fashion**: Cotton On, Country Road, David Jones, Myer, Witchery, Portmans, Sportsgirl
- **Food**: Woolworths, Coles, Queen Fine Foods, Boost Juice, Guzman y Gomez, Mad Mex
- **Tech**: JB Hi-Fi, Harvey Norman, Officeworks, Telstra, Commonwealth Bank, Qantas
- **Home**: Bunnings, IKEA, Freedom, Adairs, Bed Bath N' Table, Pillow Talk
- **Beauty**: Chemist Warehouse, Priceline, Sephora, Mecca, Lush
- **Automotive**: Supercheap Auto, Autobarn, Repco

### Fee Tiers (Semantic Constraints)
- **Micro**: $1,500 - $5,000 AUD (simple campaigns)
- **Mid**: $5,000 - $15,000 AUD (medium complexity)
- **Premium**: $15,000 - $35,000 AUD (major campaigns)

### Deliverables (Complexity-Based)
- **Simple**: Instagram posts, stories, basic content
- **Medium**: Multi-platform content, interviews, photography
- **Premium**: Shoots, TV appearances, brand ambassador content

### Campaign Types (Industry-Specific)
- **Fashion**: Spring Collection Launch, Winter Range, Limited Edition
- **Food**: Product Launch, Seasonal Campaign, Recipe Series
- **Tech**: Product Launch, Tech Review, Digital Services
- **Home**: Product Launch, DIY Series, Home Tips
- **Beauty**: Product Launch, Beauty Tips, Skincare Routine
- **Automotive**: Product Launch, Travel Series, Adventure Content

### Complexity Levels
- **Simple**: Clean, structured text with clear information
- **Medium**: Some noise, incomplete information, variations
- **Complex**: Typos, abbreviations, mixed formatting, Australian colloquialisms

## Usage Examples

### Generate Custom Dataset with Semantic Smoothing

```python
from synthetic_dataset_generator import SyntheticDatasetGenerator

# Create generator with semantic smoothing
generator = SyntheticDatasetGenerator(use_semantic_smoothing=True)

# Generate with custom parameters
dataset = generator.generate_dataset(
    num_samples=500,
    complexity_distribution={
        "simple": 0.3,
        "medium": 0.5, 
        "complex": 0.2
    }
)

# Save dataset
generator.save_dataset(dataset, "my_custom_dataset.json")
```

### Test Semantic Smoother

```python
from semantic_smoother import SemanticSmoother, BusinessLogicValidator

# Test business logic validation
validator = BusinessLogicValidator()
valid, msg = validator.validate_fee_deliverable_coherence(
    fee=8000, 
    deliverables=["2 x Instagram posts"], 
    complexity="medium"
)
print(f"Valid: {valid} - {msg}")

# Test semantic coherence
smoother = SemanticSmoother()
coherent, score = smoother.semantic_checker.semantic_coherence_check(scenario)
print(f"Coherent: {coherent} (score: {score:.3f})")
```

### Prepare Training Data

```python
from training_data_preparation import TrainingDataPreparation

# Load and prepare data
prep = TrainingDataPreparation("synthetic_influencer_agreements.json")
prep.load_dataset()

# Prepare for specific tasks
classification_data = prep.prepare_classification_data()
extraction_data = prep.prepare_extraction_data()
template_data = prep.prepare_template_matching_data()

# Save prepared data
prep.save_training_data("my_training_data")
```

## Training Integration

### For Sentence Encoder Training

```python
import json
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader

# Load training data
with open("training_data/classification_train.json", "r") as f:
    train_data = json.load(f)

# Create training examples
train_examples = []
for item in train_data:
    train_examples.append(InputExample(
        texts=[item["text"]], 
        label=item["label"]
    ))

# Train model
model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
```

### For Information Extraction

```python
# Load extraction data
with open("training_data/extraction_train.json", "r") as f:
    extraction_data = json.load(f)

# Group by field type
field_groups = {}
for item in extraction_data:
    field = item["field"]
    if field not in field_groups:
        field_groups[field] = []
    field_groups[field].append(item)

# Train separate models for each field
for field, examples in field_groups.items():
    # Train field-specific extraction model
    print(f"Training {field} extraction model with {len(examples)} examples")
```

## Quality Metrics

The pipeline includes built-in quality analysis:

- **Complexity Distribution**: Ensures balanced representation across simple/medium/complex
- **Fee Statistics**: Tracks fee ranges and distributions
- **Template Distribution**: Monitors template style variety
- **Field Coverage**: Ensures all extraction fields have sufficient examples
- **🧠 Semantic Coherence**: Validates business logic realism
- **Generation Efficiency**: Tracks successful vs failed generation attempts

## Customization

### Adding New Brands

```python
generator = SyntheticDatasetGenerator()
generator.australian_brands["new_industry"].extend([
    "New Brand 1", "New Brand 2", "New Brand 3"
])
```

### Adding New Deliverables

```python
generator.deliverables["new_complexity"].extend([
    "New deliverable 1", "New deliverable 2"
])
```

### Custom Business Logic Rules

```python
def custom_business_validation(scenario):
    # Add your custom business logic
    return is_valid, validation_message

validator = BusinessLogicValidator()
validator.custom_validation = custom_business_validation
```

## Validation

### Test with Real Data

After training, validate against real influencer agreements:

```python
# Load real agreements from data/ directory
real_agreements = load_real_agreements("data/")

# Test classification accuracy
for agreement in real_agreements:
    prediction = model.predict(agreement.text)
    print(f"Real agreement classified as: {prediction}")
```

### Performance Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Extraction**: Field-wise accuracy, entity recognition metrics
- **Template Matching**: Template selection accuracy
- **🧠 Semantic Coherence**: Business logic validation scores

## Troubleshooting

### Common Issues

1. **Dataset too small**: Increase `num_samples` in generator
2. **Poor variety**: Check parameter pools in `SyntheticDatasetGenerator`
3. **Training data imbalance**: Adjust `complexity_distribution`
4. **Missing dependencies**: Install from `training_requirements.txt`
5. **Low semantic coherence**: Check business logic rules in `semantic_smoother.py`

### Performance Tips

- Use GPU acceleration for sentence transformer training
- Batch process large datasets
- Cache embeddings for repeated training runs
- Use mixed precision training for faster training
- Enable semantic smoothing for better data quality

## Next Steps

1. **Review Generated Data**: Check `synthetic_influencer_agreements.json`
2. **Train Models**: Use prepared datasets in `training_data/`
3. **Validate**: Test against real agreements from `data/`
4. **Deploy**: Integrate trained models into Thinkerbell pipeline
5. **Iterate**: Refine based on real-world performance
6. **🧠 Monitor Quality**: Track semantic coherence metrics

## Files Created

- `synthetic_dataset_generator.py` - Main generation logic with semantic smoothing
- `semantic_smoother.py` - Business logic validation and semantic coherence
- `training_data_preparation.py` - Data preparation pipeline  
- `generate_training_dataset.py` - Complete pipeline runner
- `test_semantic_smoother.py` - Validation and testing suite
- `training_requirements.txt` - Dependencies
- `synthetic_influencer_agreements.json` - Generated dataset with semantic smoothing
- `training_data/` - Prepared training datasets
- `SYNTHETIC_DATASET_README.md` - This documentation 