"""
Exploratory Data Analysis (EDA) module for dataset visualization and analysis.

Provides comprehensive analysis including label frequencies, token distributions,
embeddings visualization, and co-occurrence analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np
from collections import Counter, defaultdict

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

# Dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    TSNE = None

# UMAP for better embeddings visualization
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Comprehensive dataset analysis and visualization.
    
    Generates charts, statistics, and reports for understanding
    dataset characteristics and quality.
    """
    
    def __init__(self, 
                 output_dir: Path,
                 model_name: str = "all-MiniLM-L6-v2",
                 figsize: Tuple[int, int] = (12, 8),
                 device: str = None):
        """
        Initialize dataset analyzer.
        
        Args:
            output_dir: Directory to save charts and reports
            model_name: Sentence transformer model for embeddings
            figsize: Default figure size for plots
            device: Device to use for computation ('cpu', 'cuda', etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.figsize = figsize
        self.device = device
        
        # Initialize model for embeddings
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from .device_utils import get_sentence_transformer_device
                model_device = get_sentence_transformer_device(device)
                self.model = SentenceTransformer(model_name, device=model_device)
                logger.info(f"Initialized sentence transformer: {model_name} on device: {model_device}")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence transformer: {e}")
        
        # Check visualization availability
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available. Visualization disabled.")
        
        # Set style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            if sns:
                sns.set_palette("husl")
    
    def analyze_dataset(self, 
                       samples: Dict[str, Dict[str, Any]], 
                       splits: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            samples: Dictionary of sample_id -> sample_data
            splits: Optional split assignments
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Starting EDA analysis of {len(samples)} samples")
        
        analysis_results = {
            'sample_count': len(samples),
            'splits': splits,
        }
        
        # Basic statistics
        analysis_results['basic_stats'] = self._analyze_basic_statistics(samples, splits)
        
        # Label analysis
        analysis_results['label_analysis'] = self._analyze_labels(samples, splits)
        
        # Text length analysis
        analysis_results['length_analysis'] = self._analyze_text_lengths(samples, splits)
        
        # Field co-occurrence analysis
        analysis_results['cooccurrence_analysis'] = self._analyze_field_cooccurrence(samples)
        
        # Generate visualizations
        if MATPLOTLIB_AVAILABLE:
            self._generate_all_visualizations(samples, splits, analysis_results)
        
        # Generate embeddings analysis if possible
        if self.model:
            analysis_results['embeddings_analysis'] = self._analyze_embeddings(samples, splits)
        
        # Save summary report
        self._save_dataset_summary(analysis_results)
        
        logger.info(f"EDA analysis complete. Results saved to {self.output_dir}")
        
        return analysis_results
    
    def _analyze_basic_statistics(self, 
                                 samples: Dict[str, Dict[str, Any]], 
                                 splits: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
        """Analyze basic dataset statistics."""
        stats = {
            'total_samples': len(samples),
            'ood_samples': sum(1 for s in samples.values() if s.get('is_ood', False)),
            'generator_versions': Counter(s.get('generator_version', 'unknown') for s in samples.values()),
            'styles': Counter(s.get('raw_input', {}).get('style', 'unknown') for s in samples.values()),
        }
        
        if splits:
            stats['split_sizes'] = {split: len(sample_ids) for split, sample_ids in splits.items()}
        
        # Calculate OOD ratio
        stats['ood_ratio'] = stats['ood_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
        
        return stats
    
    def _analyze_labels(self, 
                       samples: Dict[str, Dict[str, Any]], 
                       splits: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
        """Analyze label distributions and combinations."""
        
        # Single label frequencies
        document_types = Counter()
        complexities = Counter()
        industries = Counter()
        
        # Label combinations
        label_combinations = Counter()
        
        # Split-specific analysis
        split_analysis = {}
        if splits:
            for split_name, sample_ids in splits.items():
                split_analysis[split_name] = {
                    'document_types': Counter(),
                    'complexities': Counter(),
                    'industries': Counter()
                }
        
        # Analyze each sample
        for sample_id, sample_data in samples.items():
            classification = sample_data.get('classification', {})
            
            doc_type = classification.get('document_type', 'unknown')
            complexity = classification.get('complexity', 'unknown')
            industry = classification.get('industry', 'unknown')
            
            # Update global counters
            document_types[doc_type] += 1
            complexities[complexity] += 1
            industries[industry] += 1
            
            # Label combination
            combo = f"{doc_type}|{complexity}|{industry}"
            label_combinations[combo] += 1
            
            # Update split-specific counters
            if splits:
                for split_name, sample_ids in splits.items():
                    if sample_id in sample_ids:
                        split_analysis[split_name]['document_types'][doc_type] += 1
                        split_analysis[split_name]['complexities'][complexity] += 1
                        split_analysis[split_name]['industries'][industry] += 1
                        break
        
        return {
            'document_types': dict(document_types),
            'complexities': dict(complexities),
            'industries': dict(industries),
            'label_combinations': dict(label_combinations.most_common(20)),
            'split_analysis': split_analysis
        }
    
    def _analyze_text_lengths(self, 
                             samples: Dict[str, Dict[str, Any]], 
                             splits: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
        """Analyze text length distributions."""
        
        char_lengths = []
        token_lengths = []
        
        split_lengths = {}
        if splits:
            for split_name in splits.keys():
                split_lengths[split_name] = {
                    'char_lengths': [],
                    'token_lengths': []
                }
        
        # Collect length data
        for sample_id, sample_data in samples.items():
            char_len = sample_data.get('raw_input', {}).get('text', '')
            char_len = len(char_len) if isinstance(char_len, str) else 0
            
            token_len = sample_data.get('raw_input', {}).get('token_count', 0)
            
            char_lengths.append(char_len)
            token_lengths.append(token_len)
            
            # Split-specific collection
            if splits:
                for split_name, sample_ids in splits.items():
                    if sample_id in sample_ids:
                        split_lengths[split_name]['char_lengths'].append(char_len)
                        split_lengths[split_name]['token_lengths'].append(token_len)
                        break
        
        # Calculate statistics
        def calc_stats(values):
            if not values:
                return {}
            return {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
        
        results = {
            'char_length_stats': calc_stats(char_lengths),
            'token_length_stats': calc_stats(token_lengths),
            'char_lengths': char_lengths,
            'token_lengths': token_lengths,
        }
        
        # Split-specific statistics
        if splits:
            results['split_length_stats'] = {}
            for split_name, lengths in split_lengths.items():
                results['split_length_stats'][split_name] = {
                    'char_length_stats': calc_stats(lengths['char_lengths']),
                    'token_length_stats': calc_stats(lengths['token_lengths'])
                }
        
        return results
    
    def _analyze_field_cooccurrence(self, samples: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze co-occurrence of extracted fields."""
        
        # Track which fields appear together
        field_presence = defaultdict(list)
        field_pairs = defaultdict(int)
        
        for sample_data in samples.values():
            extracted_fields = sample_data.get('extracted_fields', {})
            
            # Get present fields (non-empty values)
            present_fields = []
            for field, value in extracted_fields.items():
                if value and value != 'N/A' and value != 'unknown':
                    field_presence[field].append(1)
                    present_fields.append(field)
                else:
                    field_presence[field].append(0)
            
            # Count field pair co-occurrences
            for i, field1 in enumerate(present_fields):
                for field2 in present_fields[i+1:]:
                    pair = tuple(sorted([field1, field2]))
                    field_pairs[pair] += 1
        
        # Calculate field frequencies
        field_frequencies = {}
        total_samples = len(samples)
        for field, presence_list in field_presence.items():
            field_frequencies[field] = sum(presence_list) / total_samples
        
        return {
            'field_frequencies': field_frequencies,
            'field_cooccurrences': dict(field_pairs),
            'most_common_pairs': dict(Counter(field_pairs).most_common(20))
        }
    
    def _analyze_embeddings(self, 
                           samples: Dict[str, Dict[str, Any]], 
                           splits: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
        """Analyze embeddings and generate dimensionality reduction plots."""
        if not self.model:
            return {}
        
        logger.info("Generating embeddings for visualization...")
        
        # Extract texts and metadata
        texts = []
        sample_ids = []
        complexities = []
        split_labels = []
        
        for sample_id, sample_data in samples.items():
            text = sample_data.get('raw_input', {}).get('text', '')
            if text:
                texts.append(text)
                sample_ids.append(sample_id)
                complexities.append(sample_data.get('classification', {}).get('complexity', 'unknown'))
                
                # Determine split
                split_label = 'unknown'
                if splits:
                    for split_name, split_sample_ids in splits.items():
                        if sample_id in split_sample_ids:
                            split_label = split_name
                            break
                split_labels.append(split_label)
        
        if len(texts) < 10:
            logger.warning("Too few samples for embeddings analysis")
            return {}
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Dimensionality reduction
        embeddings_2d = {}
        
        # PCA
        if SKLEARN_AVAILABLE and PCA:
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d['pca'] = pca.fit_transform(embeddings)
        
        # UMAP (preferred for embeddings)
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embeddings_2d['umap'] = reducer.fit_transform(embeddings)
        
        # t-SNE (fallback)
        elif SKLEARN_AVAILABLE and TSNE:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))
            embeddings_2d['tsne'] = tsne.fit_transform(embeddings)
        
        # Generate visualizations
        if MATPLOTLIB_AVAILABLE and embeddings_2d:
            self._plot_embeddings_scatter(embeddings_2d, complexities, split_labels)
        
        return {
            'embedding_shape': embeddings.shape,
            'reduction_methods': list(embeddings_2d.keys()),
            'sample_count': len(texts)
        }
    
    def _generate_all_visualizations(self, 
                                   samples: Dict[str, Dict[str, Any]], 
                                   splits: Optional[Dict[str, List[str]]],
                                   analysis_results: Dict[str, Any]):
        """Generate all visualization charts."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        logger.info("Generating visualization charts...")
        
        # Label frequency charts
        self._plot_label_frequencies(analysis_results['label_analysis'])
        
        # Length distribution charts
        self._plot_length_distributions(analysis_results['length_analysis'], splits)
        
        # Co-occurrence heatmap
        self._plot_cooccurrence_heatmap(analysis_results['cooccurrence_analysis'])
        
        # Split comparison charts
        if splits:
            self._plot_split_comparisons(analysis_results['label_analysis'], splits)
    
    def _plot_label_frequencies(self, label_analysis: Dict[str, Any]):
        """Plot label frequency charts."""
        
        # Document types
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Label Frequency Analysis', fontsize=16, fontweight='bold')
        
        # Document types
        doc_types = label_analysis['document_types']
        if doc_types:
            axes[0, 0].bar(doc_types.keys(), doc_types.values())
            axes[0, 0].set_title('Document Types')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Complexities
        complexities = label_analysis['complexities']
        if complexities:
            axes[0, 1].bar(complexities.keys(), complexities.values())
            axes[0, 1].set_title('Complexity Distribution')
        
        # Industries
        industries = label_analysis['industries']
        if industries:
            # Top 10 industries
            top_industries = dict(sorted(industries.items(), key=lambda x: x[1], reverse=True)[:10])
            axes[1, 0].bar(top_industries.keys(), top_industries.values())
            axes[1, 0].set_title('Top 10 Industries')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Label combinations
        combinations = label_analysis['label_combinations']
        if combinations:
            # Top 10 combinations
            combo_labels = []
            combo_counts = []
            for combo, count in list(combinations.items())[:10]:
                # Shorten labels for readability
                shortened = combo.replace('INFLUENCER_AGREEMENT', 'INFLU').replace('NOT_INFLUENCER', 'NOT_INF')
                combo_labels.append(shortened)
                combo_counts.append(count)
            
            axes[1, 1].barh(combo_labels, combo_counts)
            axes[1, 1].set_title('Top 10 Label Combinations')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'label_frequencies.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_length_distributions(self, length_analysis: Dict[str, Any], splits: Optional[Dict[str, List[str]]]):
        """Plot text length distributions."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Text Length Distributions', fontsize=16, fontweight='bold')
        
        # Character lengths histogram
        char_lengths = length_analysis['char_lengths']
        if char_lengths:
            axes[0, 0].hist(char_lengths, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Character Length Distribution')
            axes[0, 0].set_xlabel('Characters')
            axes[0, 0].set_ylabel('Frequency')
        
        # Token lengths histogram
        token_lengths = length_analysis['token_lengths']
        if token_lengths:
            axes[0, 1].hist(token_lengths, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Token Length Distribution')
            axes[0, 1].set_xlabel('Tokens')
            axes[0, 1].set_ylabel('Frequency')
        
        # Boxplots by split
        if splits and 'split_length_stats' in length_analysis:
            split_names = []
            split_char_lengths = []
            split_token_lengths = []
            
            for split_name in splits.keys():
                if split_name in length_analysis['split_length_stats']:
                    split_names.append(split_name)
                    # We need the raw data for boxplots, but we only have stats
                    # So we'll create a simple bar chart instead
            
            # Character length by split (means)
            char_means = []
            token_means = []
            for split_name in split_names:
                stats = length_analysis['split_length_stats'][split_name]
                char_means.append(stats['char_length_stats'].get('mean', 0))
                token_means.append(stats['token_length_stats'].get('mean', 0))
            
            if char_means:
                axes[1, 0].bar(split_names, char_means)
                axes[1, 0].set_title('Mean Character Length by Split')
                axes[1, 0].set_ylabel('Mean Characters')
            
            if token_means:
                axes[1, 1].bar(split_names, token_means)
                axes[1, 1].set_title('Mean Token Length by Split')
                axes[1, 1].set_ylabel('Mean Tokens')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'length_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cooccurrence_heatmap(self, cooccurrence_analysis: Dict[str, Any]):
        """Plot field co-occurrence heatmap."""
        if not sns:
            return
        
        field_frequencies = cooccurrence_analysis['field_frequencies']
        cooccurrences = cooccurrence_analysis['field_cooccurrences']
        
        if not field_frequencies:
            return
        
        # Create matrix
        fields = list(field_frequencies.keys())
        n_fields = len(fields)
        
        # Co-occurrence matrix (log scale)
        matrix = np.zeros((n_fields, n_fields))
        
        for i, field1 in enumerate(fields):
            for j, field2 in enumerate(fields):
                if i == j:
                    # Diagonal: field frequency
                    matrix[i, j] = field_frequencies[field1]
                else:
                    # Off-diagonal: co-occurrence
                    pair = tuple(sorted([field1, field2]))
                    count = cooccurrences.get(pair, 0)
                    # Log scale for better visualization
                    matrix[i, j] = np.log(count + 1)
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, 
                   xticklabels=fields, 
                   yticklabels=fields,
                   annot=True, 
                   fmt='.2f',
                   cmap='viridis',
                   cbar_kws={'label': 'Log(Co-occurrence + 1)'})
        plt.title('Field Co-occurrence Heatmap (Log Scale)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cooccurrence_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_split_comparisons(self, label_analysis: Dict[str, Any], splits: Dict[str, List[str]]):
        """Plot split comparison charts."""
        
        split_analysis = label_analysis.get('split_analysis', {})
        if not split_analysis:
            return
        
        # Create stacked bar charts for each label type
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Label Distribution Across Splits', fontsize=16, fontweight='bold')
        
        split_names = list(splits.keys())
        
        # Document types
        doc_types = set()
        for split_data in split_analysis.values():
            doc_types.update(split_data['document_types'].keys())
        
        doc_type_data = {}
        for doc_type in doc_types:
            doc_type_data[doc_type] = [split_analysis[split]['document_types'].get(doc_type, 0) 
                                      for split in split_names]
        
        bottom = np.zeros(len(split_names))
        for doc_type, counts in doc_type_data.items():
            axes[0].bar(split_names, counts, bottom=bottom, label=doc_type)
            bottom += counts
        
        axes[0].set_title('Document Types by Split')
        axes[0].legend()
        
        # Complexities
        complexities = ['simple', 'medium', 'complex']
        complexity_data = {}
        for complexity in complexities:
            complexity_data[complexity] = [split_analysis[split]['complexities'].get(complexity, 0) 
                                         for split in split_names]
        
        bottom = np.zeros(len(split_names))
        for complexity, counts in complexity_data.items():
            axes[1].bar(split_names, counts, bottom=bottom, label=complexity)
            bottom += counts
        
        axes[1].set_title('Complexity by Split')
        axes[1].legend()
        
        # Top industries
        all_industries = set()
        for split_data in split_analysis.values():
            all_industries.update(split_data['industries'].keys())
        
        # Get top 5 most common industries
        industry_totals = {}
        for industry in all_industries:
            total = sum(split_analysis[split]['industries'].get(industry, 0) for split in split_names)
            industry_totals[industry] = total
        
        top_industries = sorted(industry_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        
        industry_data = {}
        for industry, _ in top_industries:
            industry_data[industry] = [split_analysis[split]['industries'].get(industry, 0) 
                                     for split in split_names]
        
        bottom = np.zeros(len(split_names))
        for industry, counts in industry_data.items():
            axes[2].bar(split_names, counts, bottom=bottom, label=industry)
            bottom += counts
        
        axes[2].set_title('Top 5 Industries by Split')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'split_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_embeddings_scatter(self, 
                                embeddings_2d: Dict[str, np.ndarray], 
                                complexities: List[str], 
                                split_labels: List[str]):
        """Plot embeddings scatter plots."""
        
        # Color map for complexities
        complexity_colors = {'simple': 'blue', 'medium': 'orange', 'complex': 'red', 'unknown': 'gray'}
        
        # Marker map for splits
        split_markers = {'train': 'o', 'val': 's', 'test': '^', 'unknown': 'x'}
        
        for method, coords in embeddings_2d.items():
            plt.figure(figsize=(12, 8))
            
            # Plot each complexity-split combination
            for complexity in set(complexities):
                for split in set(split_labels):
                    mask = [(c == complexity and s == split) for c, s in zip(complexities, split_labels)]
                    if any(mask):
                        indices = [i for i, m in enumerate(mask) if m]
                        x_coords = coords[indices, 0]
                        y_coords = coords[indices, 1]
                        
                        plt.scatter(x_coords, y_coords, 
                                  c=complexity_colors.get(complexity, 'gray'),
                                  marker=split_markers.get(split, 'o'),
                                  alpha=0.6,
                                  s=50,
                                  label=f"{complexity}-{split}")
            
            plt.title(f'Sample Embeddings Visualization ({method.upper()})', fontsize=14, fontweight='bold')
            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')
            
            # Create custom legend
            complexity_patches = [mpatches.Patch(color=color, label=comp) 
                                for comp, color in complexity_colors.items() if comp in complexities]
            split_patches = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', 
                                      markersize=8, label=split, linestyle='None')
                           for split, marker in split_markers.items() if split in split_labels]
            
            legend1 = plt.legend(handles=complexity_patches, title='Complexity', loc='upper left')
            legend2 = plt.legend(handles=split_patches, title='Split', loc='upper right')
            plt.gca().add_artist(legend1)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'embeddings_{method}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_dataset_summary(self, analysis_results: Dict[str, Any]):
        """Save comprehensive dataset summary."""
        
        summary_file = self.output_dir / 'dataset_summary.json'
        
        # Create a clean summary (remove raw data arrays)
        summary = analysis_results.copy()
        
        # Remove large arrays to keep summary file manageable
        if 'length_analysis' in summary:
            if 'char_lengths' in summary['length_analysis']:
                del summary['length_analysis']['char_lengths']
            if 'token_lengths' in summary['length_analysis']:
                del summary['length_analysis']['token_lengths']
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved dataset summary to {summary_file}")
    
    def generate_red_flag_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate red flag summary for potential issues."""
        
        red_flags = []
        
        # Check class imbalance
        label_analysis = analysis_results.get('label_analysis', {})
        
        # Document type imbalance
        doc_types = label_analysis.get('document_types', {})
        if doc_types:
            total_docs = sum(doc_types.values())
            for doc_type, count in doc_types.items():
                ratio = count / total_docs
                if ratio < 0.1:  # Less than 10%
                    red_flags.append(f"Low representation of {doc_type}: {ratio:.1%} ({count} samples)")
                elif ratio > 0.9:  # More than 90%
                    red_flags.append(f"High dominance of {doc_type}: {ratio:.1%} ({count} samples)")
        
        # Complexity imbalance
        complexities = label_analysis.get('complexities', {})
        if complexities:
            total_complexity = sum(complexities.values())
            for complexity, count in complexities.items():
                ratio = count / total_complexity
                if ratio < 0.05:  # Less than 5%
                    red_flags.append(f"Very low {complexity} complexity samples: {ratio:.1%} ({count} samples)")
        
        # Check for long texts
        length_analysis = analysis_results.get('length_analysis', {})
        char_stats = length_analysis.get('char_length_stats', {})
        token_stats = length_analysis.get('token_length_stats', {})
        
        if char_stats.get('max', 0) > 5000:
            red_flags.append(f"Very long texts detected: max {char_stats['max']} characters")
        
        if token_stats.get('max', 0) > 1000:
            red_flags.append(f"Very long texts detected: max {token_stats['max']} tokens")
        
        # Check token distribution
        if token_stats.get('mean', 0) > 400:
            red_flags.append(f"High average token count: {token_stats['mean']:.0f} tokens")
        
        # Check duplicates kept (if dedup stats available)
        basic_stats = analysis_results.get('basic_stats', {})
        if 'ood_ratio' in basic_stats: # Changed from 'dedup_stats' to 'ood_ratio'
            ood_ratio = basic_stats['ood_ratio']
            if ood_ratio > 0.3:  # More than 30% OOD
                red_flags.append(f"High OOD contamination: {ood_ratio:.1%}")
            elif ood_ratio < 0.1:  # Less than 10% OOD
                red_flags.append(f"Low OOD contamination: {ood_ratio:.1%}")
        
        return red_flags 