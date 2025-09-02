#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Thinkerbell Chunk Dataset

Analyzes chunk distribution, labels, text lengths, and semantic clustering
to understand dataset characteristics before training.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Data processing
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")

# ML libraries
try:
    from sentence_transformers import SentenceTransformer
    import umap
    HAS_ML_DEPS = True
except ImportError as e:
    HAS_ML_DEPS = False
    print(f"⚠️ ML dependencies missing: {e}")
    print("Install with: pip install sentence-transformers umap-learn")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class ChunkEDA:
    """Exploratory Data Analysis for chunk datasets"""
    
    def __init__(self, 
                 splits_dir: str = "synthetic_dataset/splits",
                 output_dir: str = "eda_results",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.data = {}  # split -> list of chunks
        self.df = None  # Combined dataframe
        self.embeddings = {}  # split -> embeddings array
        self.stats = {}  # Summary statistics
        
        # Load sentence transformer model
        self.model = None
        self.has_ml_deps = HAS_ML_DEPS
        if self.has_ml_deps:
            try:
                logger.info(f"Loading sentence transformer: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.has_ml_deps = False
    
    def extract_document_id(self, chunk_id: str) -> Optional[str]:
        """Extract document ID from chunk ID"""
        # Pattern: sample_123_c1 -> sample_123
        match = re.match(r'(.*?)_c\d+$', chunk_id)
        return match.group(1) if match else None
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        if self.model and hasattr(self.model, 'tokenizer'):
            try:
                return len(self.model.tokenizer.encode(text))
            except:
                pass
        # Fallback: ~4 chars per token
        return len(text) // 4
    
    def load_chunks(self) -> None:
        """Load chunk data from all splits"""
        logger.info("📂 Loading chunk data from splits...")
        
        splits = ['train', 'val', 'test']
        total_chunks = 0
        
        for split in splits:
            split_dir = self.splits_dir / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            chunk_files = list(split_dir.glob("*.json"))
            logger.info(f"  {split}: Found {len(chunk_files)} chunks")
            
            chunks = []
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        
                        # Extract required fields
                        text = chunk_data.get('text', '')
                        labels = chunk_data.get('labels', [])
                        chunk_id = chunk_data.get('chunk_id', chunk_file.stem)
                        
                        # Process data
                        processed_chunk = {
                            'split': split,
                            'chunk_id': chunk_id,
                            'document_id': chunk_data.get('document_id') or self.extract_document_id(chunk_id),
                            'text': text,
                            'text_length_chars': len(text),
                            'text_length_tokens': self.count_tokens(text),
                            'labels': labels,
                            'label_combo': '+'.join(sorted(labels)) if labels else 'NO_LABELS',
                            'num_labels': len(labels)
                        }
                        chunks.append(processed_chunk)
                        
                except Exception as e:
                    logger.warning(f"Failed to load {chunk_file}: {e}")
            
            self.data[split] = chunks
            total_chunks += len(chunks)
        
        # Create combined dataframe
        all_chunks = []
        for split_chunks in self.data.values():
            all_chunks.extend(split_chunks)
        
        self.df = pd.DataFrame(all_chunks)
        logger.info(f"✅ Loaded {total_chunks} total chunks")
        logger.info(f"  Splits: {dict(self.df['split'].value_counts())}")
    
    def generate_embeddings(self) -> None:
        """Generate sentence embeddings for semantic analysis"""
        if not self.has_ml_deps or not self.model:
            logger.warning("⚠️ Skipping embeddings - model not available")
            return
        
        logger.info("🧠 Generating sentence embeddings...")
        
        for split in self.data.keys():
            texts = [chunk['text'] for chunk in self.data[split]]
            if texts:
                logger.info(f"  {split}: Encoding {len(texts)} texts...")
                embeddings = self.model.encode(texts, show_progress_bar=True)
                # Normalize embeddings
                embeddings = normalize(embeddings)
                self.embeddings[split] = embeddings
        
        logger.info("✅ Embeddings generated")
    
    def plot_label_frequency(self) -> None:
        """Bar chart: frequency of atomic labels across splits"""
        logger.info("📊 Creating label frequency plots...")
        
        # Atomic labels
        atomic_labels = []
        for _, row in self.df.iterrows():
            atomic_labels.extend([(label, row['split']) for label in row['labels']])
        
        atomic_df = pd.DataFrame(atomic_labels, columns=['label', 'split'])
        
        # Plot atomic labels
        plt.figure(figsize=(14, 8))
        
        # Count labels by split
        label_counts = atomic_df.groupby(['label', 'split']).size().unstack(fill_value=0)
        
        ax = label_counts.plot(kind='bar', figsize=(14, 8), width=0.8)
        plt.title('Frequency of Atomic Labels Across Splits', fontsize=16, pad=20)
        plt.xlabel('Labels', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Split', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'atomic_labels_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot label combinations
        plt.figure(figsize=(16, 10))
        combo_counts = self.df.groupby(['label_combo', 'split']).size().unstack(fill_value=0)
        
        # Only show top 20 combinations to avoid overcrowding
        top_combos = combo_counts.sum(axis=1).nlargest(20).index
        combo_counts_top = combo_counts.loc[top_combos]
        
        ax = combo_counts_top.plot(kind='bar', figsize=(16, 10), width=0.8)
        plt.title('Frequency of Top 20 Label Combinations Across Splits', fontsize=16, pad=20)
        plt.xlabel('Label Combinations', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Split', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'label_combos_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Label frequency plots saved")
    
    def plot_text_length_distribution(self) -> None:
        """Histograms: distribution of text lengths"""
        logger.info("📏 Creating text length distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        splits = ['train', 'val', 'test']
        colors = ['blue', 'orange', 'green']
        
        # Character length distribution
        ax1 = axes[0, 0]
        for split, color in zip(splits, colors):
            split_data = self.df[self.df['split'] == split]
            if not split_data.empty:
                ax1.hist(split_data['text_length_chars'], alpha=0.7, label=split, 
                        bins=30, color=color, density=True)
        ax1.set_title('Text Length Distribution (Characters)', fontsize=14)
        ax1.set_xlabel('Characters')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Token length distribution
        ax2 = axes[0, 1]
        for split, color in zip(splits, colors):
            split_data = self.df[self.df['split'] == split]
            if not split_data.empty:
                ax2.hist(split_data['text_length_tokens'], alpha=0.7, label=split, 
                        bins=30, color=color, density=True)
        ax2.set_title('Text Length Distribution (Tokens)', fontsize=14)
        ax2.set_xlabel('Tokens')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Character length by split (boxplot)
        ax3 = axes[1, 0]
        split_chars = [self.df[self.df['split'] == split]['text_length_chars'].values 
                      for split in splits]
        ax3.boxplot(split_chars, labels=splits)
        ax3.set_title('Character Length by Split', fontsize=14)
        ax3.set_ylabel('Characters')
        ax3.grid(True, alpha=0.3)
        
        # Token length by split (boxplot)
        ax4 = axes[1, 1]
        split_tokens = [self.df[self.df['split'] == split]['text_length_tokens'].values 
                       for split in splits]
        ax4.boxplot(split_tokens, labels=splits)
        ax4.set_title('Token Length by Split', fontsize=14)
        ax4.set_ylabel('Tokens')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'text_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Text length distribution plots saved")
    
    def plot_token_length_by_combo(self) -> None:
        """Boxplot: token length distribution by label combination"""
        logger.info("📦 Creating token length by label combo plot...")
        
        # Get top 15 most frequent label combinations
        top_combos = self.df['label_combo'].value_counts().head(15).index
        filtered_df = self.df[self.df['label_combo'].isin(top_combos)]
        
        plt.figure(figsize=(16, 10))
        
        # Create boxplot
        box_data = []
        labels = []
        for combo in top_combos:
            combo_data = filtered_df[filtered_df['label_combo'] == combo]['text_length_tokens']
            if not combo_data.empty:
                box_data.append(combo_data.values)
                labels.append(combo.replace('+', '+\n') if len(combo) > 20 else combo)
        
        plt.boxplot(box_data, labels=labels)
        plt.title('Token Length Distribution by Label Combination\n(Top 15 Most Frequent)', 
                 fontsize=16, pad=20)
        plt.xlabel('Label Combinations', fontsize=12)
        plt.ylabel('Token Length', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'token_length_by_combo.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Token length by combo plot saved")
    
    def plot_label_cooccurrence(self) -> None:
        """Heatmap: co-occurrence matrix of labels"""
        logger.info("🔥 Creating label co-occurrence heatmap...")
        
        # Get all unique labels
        all_labels = set()
        for labels in self.df['labels']:
            all_labels.update(labels)
        all_labels = sorted(list(all_labels))
        
        # Create co-occurrence matrix
        cooccurrence = np.zeros((len(all_labels), len(all_labels)))
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        
        for labels in self.df['labels']:
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    idx1, idx2 = label_to_idx[label1], label_to_idx[label2]
                    cooccurrence[idx1, idx2] += 1
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Use log scale for better visualization if values vary greatly
        cooccurrence_plot = np.log1p(cooccurrence)  # log(1 + x) to handle zeros
        
        sns.heatmap(cooccurrence_plot, 
                   xticklabels=all_labels, 
                   yticklabels=all_labels,
                   annot=True, 
                   fmt='.1f',
                   cmap='YlOrRd',
                   square=True)
        
        plt.title('Label Co-occurrence Matrix (log scale)', fontsize=16, pad=20)
        plt.xlabel('Labels', fontsize=12)
        plt.ylabel('Labels', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'label_cooccurrence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Label co-occurrence heatmap saved")
    
    def plot_embedding_scatter(self) -> None:
        """PCA + UMAP scatter plot of embeddings"""
        if not self.has_ml_deps or not self.embeddings:
            logger.warning("⚠️ Skipping embedding scatter - embeddings not available")
            return
        
        logger.info("🎨 Creating embedding scatter plot...")
        
        # Combine all embeddings
        all_embeddings = []
        all_labels = []
        all_splits = []
        
        for split, embeddings in self.embeddings.items():
            all_embeddings.append(embeddings)
            split_data = self.data[split]
            all_labels.extend([chunk['label_combo'] for chunk in split_data])
            all_splits.extend([split] * len(split_data))
        
        all_embeddings = np.vstack(all_embeddings)
        
        # PCA reduction to 50D
        logger.info("  Applying PCA (384D -> 50D)...")
        pca = PCA(n_components=50, random_state=42)
        embeddings_pca = pca.fit_transform(all_embeddings)
        
        # UMAP reduction to 2D
        logger.info("  Applying UMAP (50D -> 2D)...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings_pca)
        
        # Create scatter plot
        plt.figure(figsize=(16, 12))
        
        # Get unique label combinations and splits
        unique_combos = sorted(list(set(all_labels)))
        unique_splits = ['train', 'val', 'test']
        
        # Create color map for label combinations
        combo_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_combos)))
        combo_to_color = dict(zip(unique_combos, combo_colors))
        
        # Create marker map for splits
        split_markers = {'train': 'o', 'val': 's', 'test': '^'}
        
        # Plot points
        for split in unique_splits:
            split_mask = np.array(all_splits) == split
            if not split_mask.any():
                continue
                
            split_embeddings = embeddings_2d[split_mask]
            split_labels = np.array(all_labels)[split_mask]
            
            for combo in unique_combos:
                combo_mask = split_labels == combo
                if not combo_mask.any():
                    continue
                    
                combo_embeddings = split_embeddings[combo_mask]
                
                plt.scatter(combo_embeddings[:, 0], combo_embeddings[:, 1],
                          c=[combo_to_color[combo]], 
                          marker=split_markers[split],
                          s=50, alpha=0.7,
                          label=f'{combo} ({split})' if len(unique_combos) <= 10 else None)
        
        plt.title('Embedding Scatter Plot (PCA + UMAP)\nColor: Label Combo, Shape: Split', 
                 fontsize=16, pad=20)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        
        # Add legend if not too crowded
        if len(unique_combos) <= 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'embedding_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Embedding scatter plot saved")
    
    def compute_summary_stats(self) -> None:
        """Compute and save summary statistics"""
        logger.info("📊 Computing summary statistics...")
        
        stats = {}
        
        # Basic dataset stats
        stats['dataset_overview'] = {
            'total_chunks': len(self.df),
            'splits': dict(self.df['split'].value_counts()),
            'unique_documents': self.df['document_id'].nunique() if 'document_id' in self.df else None
        }
        
        # Label statistics
        all_labels = []
        for labels in self.df['labels']:
            all_labels.extend(labels)
        
        stats['label_stats'] = {
            'unique_atomic_labels': len(set(all_labels)),
            'atomic_label_frequency': dict(Counter(all_labels).most_common()),
            'unique_label_combos': self.df['label_combo'].nunique(),
            'label_combo_frequency': dict(self.df['label_combo'].value_counts().head(20))
        }
        
        # Text length statistics
        stats['text_length_stats'] = {
            'character_length': {
                'mean': float(self.df['text_length_chars'].mean()),
                'std': float(self.df['text_length_chars'].std()),
                'min': int(self.df['text_length_chars'].min()),
                'max': int(self.df['text_length_chars'].max()),
                'median': float(self.df['text_length_chars'].median())
            },
            'token_length': {
                'mean': float(self.df['text_length_tokens'].mean()),
                'std': float(self.df['text_length_tokens'].std()),
                'min': int(self.df['text_length_tokens'].min()),
                'max': int(self.df['text_length_tokens'].max()),
                'median': float(self.df['text_length_tokens'].median())
            }
        }
        
        # Split-specific statistics
        stats['split_stats'] = {}
        for split in ['train', 'val', 'test']:
            split_data = self.df[self.df['split'] == split]
            if not split_data.empty:
                split_combos = set(split_data['label_combo'])
                stats['split_stats'][split] = {
                    'chunks': len(split_data),
                    'unique_label_combos': len(split_combos),
                    'avg_text_length_chars': float(split_data['text_length_chars'].mean()),
                    'avg_text_length_tokens': float(split_data['text_length_tokens'].mean())
                }
        
        # Label combo overlap analysis
        if len(stats['split_stats']) >= 2:
            train_combos = set(self.df[self.df['split'] == 'train']['label_combo'])
            val_combos = set(self.df[self.df['split'] == 'val']['label_combo'])
            test_combos = set(self.df[self.df['split'] == 'test']['label_combo'])
            
            stats['label_combo_overlap'] = {
                'train_val_overlap': len(train_combos & val_combos) / len(train_combos | val_combos) if train_combos | val_combos else 0,
                'train_test_overlap': len(train_combos & test_combos) / len(train_combos | test_combos) if train_combos | test_combos else 0,
                'val_test_overlap': len(val_combos & test_combos) / len(val_combos | test_combos) if val_combos | test_combos else 0,
                'common_to_all_splits': len(train_combos & val_combos & test_combos)
            }
        
        # Embedding-based similarity analysis
        if self.embeddings:
            logger.info("  Computing embedding similarities...")
            sim_stats = {}
            
            # Compute cross-split similarities
            for split1 in ['train']:  # Compare train with val/test
                for split2 in ['val', 'test']:
                    if split1 in self.embeddings and split2 in self.embeddings:
                        emb1 = self.embeddings[split1]
                        emb2 = self.embeddings[split2]
                        
                        # Compute cosine similarity matrix
                        sim_matrix = cosine_similarity(emb1, emb2)
                        
                        # Find maximum similarities (potential data leaks)
                        max_sim = float(sim_matrix.max())
                        max_indices = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
                        
                        # Average similarity
                        avg_sim = float(sim_matrix.mean())
                        
                        sim_stats[f'{split1}_{split2}'] = {
                            'max_similarity': max_sim,
                            'avg_similarity': avg_sim,
                            'max_similarity_items': {
                                f'{split1}_index': int(max_indices[0]),
                                f'{split2}_index': int(max_indices[1])
                            }
                        }
            
            stats['similarity_analysis'] = sim_stats
        
        # Save statistics
        self.stats = stats
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        stats_serializable = convert_numpy_types(stats)
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        
        # Print key statistics
        self.print_summary_stats()
        
        logger.info("✅ Summary statistics computed and saved")
    
    def print_summary_stats(self) -> None:
        """Print key summary statistics"""
        print("\n" + "="*80)
        print("📊 CHUNK DATASET ANALYSIS SUMMARY")
        print("="*80)
        
        # Dataset overview
        overview = self.stats['dataset_overview']
        print(f"\n📋 Dataset Overview:")
        print(f"  Total chunks: {overview['total_chunks']:,}")
        for split, count in overview['splits'].items():
            percentage = (count / overview['total_chunks']) * 100
            print(f"  {split.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        # Label statistics
        label_stats = self.stats['label_stats']
        print(f"\n🏷️  Label Statistics:")
        print(f"  Unique atomic labels: {label_stats['unique_atomic_labels']}")
        print(f"  Unique label combinations: {label_stats['unique_label_combos']}")
        print(f"  Top label combinations:")
        for combo, count in list(label_stats['label_combo_frequency'].items())[:5]:
            print(f"    {combo}: {count}")
        
        # Text length statistics
        text_stats = self.stats['text_length_stats']
        print(f"\n📏 Text Length Statistics:")
        print(f"  Characters - Mean: {text_stats['character_length']['mean']:.1f}, "
              f"Range: {text_stats['character_length']['min']}-{text_stats['character_length']['max']}")
        print(f"  Tokens - Mean: {text_stats['token_length']['mean']:.1f}, "
              f"Range: {text_stats['token_length']['min']}-{text_stats['token_length']['max']}")
        
        # Label combo overlap
        if 'label_combo_overlap' in self.stats:
            overlap = self.stats['label_combo_overlap']
            print(f"\n🔄 Label Combination Overlap:")
            print(f"  Train-Val overlap: {overlap['train_val_overlap']:.1%}")
            print(f"  Train-Test overlap: {overlap['train_test_overlap']:.1%}")
            print(f"  Val-Test overlap: {overlap['val_test_overlap']:.1%}")
            print(f"  Common to all splits: {overlap['common_to_all_splits']}")
        
        # Similarity analysis
        if 'similarity_analysis' in self.stats:
            print(f"\n🧠 Semantic Similarity Analysis:")
            for comparison, sim_data in self.stats['similarity_analysis'].items():
                print(f"  {comparison.replace('_', ' vs ')}:")
                print(f"    Max similarity: {sim_data['max_similarity']:.4f}")
                print(f"    Avg similarity: {sim_data['avg_similarity']:.4f}")
                if sim_data['max_similarity'] > 0.95:
                    print(f"    ⚠️  HIGH SIMILARITY DETECTED - Potential data leak!")
        
        print(f"\n📁 Results saved to: {self.output_dir}/")
        print("="*80)
    
    def run_analysis(self) -> None:
        """Run complete EDA analysis"""
        logger.info("🎯 Starting Chunk Dataset EDA Analysis")
        logger.info("="*80)
        
        try:
            # Load data
            self.load_chunks()
            
            if self.df is None or self.df.empty:
                logger.error("❌ No data loaded. Check splits directory.")
                return
            
            # Generate embeddings
            self.generate_embeddings()
            
            # Create visualizations
            self.plot_label_frequency()
            self.plot_text_length_distribution()
            self.plot_token_length_by_combo()
            self.plot_label_cooccurrence()
            self.plot_embedding_scatter()
            
            # Compute statistics
            self.compute_summary_stats()
            
            logger.info("="*80)
            logger.info("🎉 EDA Analysis Complete!")
            logger.info(f"📁 All results saved to: {self.output_dir}/")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk Dataset EDA Analysis")
    parser.add_argument("--splits-dir", default="synthetic_dataset/splits",
                       help="Directory containing train/val/test splits")
    parser.add_argument("--output-dir", default="eda_results",
                       help="Output directory for results")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model for embeddings")
    
    args = parser.parse_args()
    
    # Run analysis
    eda = ChunkEDA(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        model_name=args.model
    )
    
    eda.run_analysis()


if __name__ == "__main__":
    main() 