# FAISS Contract Template Index System

Production-ready FAISS indexing system for contract template matching with sentence-transformer retrieval.

## 🎯 System Overview

This system provides fast, scalable template matching for contract documents using:

- **Main Index**: IndexHNSWFlat (M=32, efConstruction=200, efSearch=64)
- **Bones Index**: IndexFlatIP (11 core template signatures, never fails)
- **Target Performance**: p95 ≤ 200ms for k ≤ 5 queries
- **Memory Footprint**: <1GB total
- **Fallback Strategy**: Multi-level fallback ensuring 100% availability

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FAISS Template Index                     │
├─────────────────────────────────────────────────────────────┤
│  Enhanced Template Manager                                  │
│  ├── FAISS Template Manager                                 │
│  ├── Health Monitor                                         │
│  ├── Persistence Manager                                    │
│  └── Adaptive Search Optimizer                             │
├─────────────────────────────────────────────────────────────┤
│  Core FAISS Indices                                        │
│  ├── Main HNSW Index (Production queries)                  │
│  └── Bones FlatIP Index (Never-fail fallback)             │
├─────────────────────────────────────────────────────────────┤
│  Sentence Transformer Model                                │
│  └── all-MiniLM-L6-v2 (384D embeddings)                  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Setup System

```bash
# Run setup script
python setup_faiss_system.py --base-dir . --load-dataset --dataset-path complete_pipeline_5000

# Install dependencies
pip install -r requirements_faiss.txt
```

### 2. Basic Usage

```python
from thinkerbell.core.enhanced_template_manager import EnhancedTemplateManager

# Initialize manager
manager = EnhancedTemplateManager()

# Find template for contract text
result = manager.find_best_template({
    "influencer": "Sarah Chen",
    "brand": "Myer", 
    "fee_numeric": 5000,
    "deliverables": ["2 x Instagram posts"],
    "engagement_term": "3 months"
})

print(f"Template: {result[0]}, Confidence: {result[1]}")
```

### 3. Run Demo

```bash
python faiss_integration_demo.py
```

## 📊 Core Components

### FAISSTemplateIndex
- **Purpose**: Core FAISS indexing with HNSW and FlatIP indices
- **Features**: Adaptive efSearch, memory monitoring, emergency fallback
- **Guarantees**: Never fails (bones index always available)

### FAISSTemplateManager  
- **Purpose**: High-level template operations and metadata management
- **Features**: Template recommendations, complexity analysis, auto-rebuild
- **Integration**: Seamless integration with existing pipeline

### FAISSHealthMonitor
- **Purpose**: Continuous health monitoring and alerting
- **Metrics**: Latency, memory usage, recall, fallback rates
- **Alerts**: Real-time alerts for performance degradation

### FAISSPersistenceManager
- **Purpose**: Versioned persistence with rollback capabilities
- **Features**: Version management, automatic backups, rollback safety
- **Lifecycle**: 6-month cache with auto-cleanup

### AdaptiveSearchOptimizer
- **Purpose**: Runtime optimization of search parameters
- **Strategy**: Memory-aware efSearch adjustment
- **Goals**: Balance latency vs quality based on system state

## 🎛️ Configuration

### Core Settings (`faiss_config.json`)

```json
{
  "faiss_system": {
    "model_path": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "hnsw_parameters": {
      "M": 32,
      "efConstruction": 200, 
      "efSearch": 64
    },
    "memory_limit_mb": 1000,
    "performance_targets": {
      "p95_latency_ms": 200,
      "recall_threshold": 0.95,
      "max_fallback_rate": 0.3
    }
  }
}
```

### Template Hierarchy

- **Simple**: `template_style_1` (casual email, bullet points)
- **Medium**: `template_style_2`, `template_style_3` (formal brief, professional)
- **Complex**: `template_style_4`, `template_style_5` (detailed agreement, enterprise)

## 🔧 Operations

### Health Monitoring

```python
from thinkerbell.core.faiss_operations import FAISSHealthMonitor

monitor = FAISSHealthMonitor(template_manager)
monitor.start_monitoring(interval_seconds=60)

# Get health report
health = monitor.get_health_report()
print(f"Status: {health['overall_status']}")
```

### Version Management

```python  
from thinkerbell.core.faiss_operations import FAISSPersistenceManager

persistence = FAISSPersistenceManager("faiss_indices/versions")

# Save version
version_id = persistence.save_version("faiss_indices", "production_v1")

# List versions
versions = persistence.list_versions()

# Rollback if needed
persistence.rollback_to_version(version_id, "faiss_indices")
```

### Performance Optimization

```python
from thinkerbell.core.faiss_operations import AdaptiveSearchOptimizer

optimizer = AdaptiveSearchOptimizer(faiss_index)
optimizer.optimize_search_parameters()

# Get optimization report
report = optimizer.get_optimization_report()
```

## 📈 Performance Characteristics

### Latency Targets
- **p95 ≤ 200ms** for k ≤ 5 queries
- **p99 ≤ 500ms** under normal load
- **Bones fallback ≤ 10ms** (guaranteed)

### Memory Usage
- **Total footprint**: <1GB 
- **Main index**: ~800MB (5K vectors + HNSW graph)
- **Bones index**: ~17KB (11 core templates)
- **Metadata**: <50MB

### Throughput
- **Target**: <10 QPS (internal tool)
- **Sustainable**: 50+ QPS with current architecture
- **Burst capacity**: 100+ QPS for short periods

## 🛡️ Fallback Strategy

1. **Primary**: Main HNSW index search
2. **Secondary**: Bones FlatIP index (if confidence low)
3. **Tertiary**: Emergency template (system failure)

### Fallback Triggers
- Main index unavailable
- Query confidence < threshold (0.3)
- Memory pressure > 80%
- Latency > 500ms

## 🔍 Template Matching Logic

### Input Processing
1. Extract text from `raw_input.text`
2. Generate 384D sentence embedding
3. Normalize for cosine similarity (inner product)

### Semantic Search
1. Search main HNSW index with metadata filtering
2. Apply confidence thresholding
3. Fallback to bones index if needed
4. Return ranked template candidates

### Post-Processing
1. Map FAISS results to standard template IDs
2. Apply business logic validation
3. Generate confidence scores
4. Provide alternative recommendations

## 📋 Operational Procedures

### Daily Operations
- Monitor health dashboard
- Check query latency trends
- Review fallback rates
- Validate memory usage

### Weekly Operations  
- Analyze performance metrics
- Review optimization recommendations
- Check version backup status
- Plan capacity adjustments

### Monthly Operations
- Evaluate recall performance
- Consider index rebuilds
- Update performance baselines
- Archive old versions

### Rebuild Triggers
- **Data growth**: >50% new documents
- **Recall drop**: >5% degradation 
- **Cache expiry**: >6 months old
- **Performance**: Consistent latency issues

## 🚨 Troubleshooting

### Common Issues

#### High Latency
```bash
# Check efSearch setting
# Reduce if > 64 for memory-constrained systems
# Increase if < 32 for quality-focused systems
```

#### High Fallback Rate
```bash
# Check main index health
# Consider index rebuild
# Verify data quality
```

#### Memory Pressure
```bash
# Monitor memory usage trends
# Reduce efSearch temporarily  
# Plan index optimization
```

#### Index Corruption
```bash
# Use version rollback
# Rebuild from source data
# Verify data integrity
```

### Emergency Procedures

#### Complete System Failure
1. System automatically uses bones index
2. Emergency template returned with 0.5 confidence
3. All queries still get responses
4. Manual intervention for repair

#### Data Loss
1. Restore from latest version backup
2. Rebuild index if needed
3. Validate performance after restore
4. Update monitoring baselines

## 🔬 Testing

### Unit Tests
```bash
python -m pytest thinkerbell/tests/test_faiss_*.py
```

### Integration Tests  
```bash
python faiss_integration_demo.py
```

### Performance Tests
```bash
python -c "
from faiss_integration_demo import FAISSIntegrationDemo
demo = FAISSIntegrationDemo()
demo.run_performance_tests()
"
```

### Load Tests
```bash
# Custom load testing with concurrent queries
python scripts/load_test_faiss.py --queries 1000 --concurrent 10
```

## 📚 API Reference

### EnhancedTemplateManager

#### `find_best_template(extracted_info: Dict) -> Tuple[str, float]`
Find best template match for extracted contract information.

#### `load_dataset(dataset_path: str) -> bool`  
Load contract dataset into FAISS index.

#### `get_health_status() -> Dict`
Get comprehensive system health status.

### FAISSTemplateIndex

#### `search_templates(query_text: str, k: int = 5) -> List[QueryResult]`
Search for template matches with fallback handling.

#### `add_documents(documents: List[Dict]) -> bool`
Add documents to the main HNSW index.

#### `get_health_status() -> Dict`
Get index health and performance metrics.

## 🔮 Future Enhancements

### Planned Features
- Multi-modal embeddings (text + metadata)
- Dynamic index sharding for scale
- Real-time incremental updates
- Cross-language template support

### Performance Optimizations
- GPU acceleration for large scales
- Quantized embeddings for memory reduction
- Hierarchical clustering for better recall
- Custom distance functions for domain-specific matching

### Operational Improvements
- Automated A/B testing for parameters
- ML-driven optimization recommendations
- Predictive capacity planning
- Advanced anomaly detection

## 📞 Support

For issues or questions:
1. Check troubleshooting guide above
2. Review system logs in `faiss_indices/logs/`
3. Run health diagnostics: `manager.get_health_status()`
4. Check configuration: `faiss_config.json`

## 📄 License

Part of the Thinkerbell contract processing system. See main project license. 