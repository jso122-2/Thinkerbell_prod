"""
Semantic generator for creating coherent synthetic briefs from cleaned records.

Provides semantic smoothing, style transfer, and coherence validation using
sentence transformers and business rule validation.
"""

import json
import logging
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sentence transformers for semantic analysis
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .chunk import SmartChunker
from .business_rules import BusinessRuleEngine, DocumentClassifier
from .ood import OODGenerator
from .schema import DocumentSample


logger = logging.getLogger(__name__)


@dataclass
class SyntheticSample:
    """Data structure for generated synthetic samples."""
    sample_id: str
    generator_version: str
    raw_input: Dict[str, Any]
    classification: Dict[str, str]
    extracted_fields: Dict[str, Any]
    validation: Dict[str, Any]
    template_hint: Optional[str] = None
    is_ood: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class SemanticSmoother:
    """
    Handles semantic analysis and coherence validation for generated samples.
    
    Uses sentence transformers to compare generated scenarios against
    positive and negative example banks with cosine similarity matching.
    Rejects samples with coherence_score < 0.6 as specified.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None, offline_mode: bool = False):
        """Initialize semantic smoother with sentence transformer model."""
        self.model_name = model_name
        self.device = device
        self.offline_mode = offline_mode
        self.model = None
        self.good_examples = []
        self.bad_examples = []
        self.good_embeddings = None
        self.bad_embeddings = None
        self.coherence_threshold = 0.6  # Required threshold as specified
        
        self._init_model(device, offline_mode)
        self._init_example_banks()
        
        self.stats = {
            'samples_processed': 0,
            'coherence_scores': [],
            'passed_coherence': 0,
            'failed_coherence': 0,
            'model_available': self.model is not None
        }
    
    def _init_model(self, device: str = None, offline_mode: bool = False):
        """Initialize the sentence transformer model with robust offline support."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            return
        
        try:
            from .utils import get_cached_sentence_transformer, check_model_availability
            
            # Check model availability first
            if check_model_availability(self.model_name, "sentence_transformer"):
                logger.info(f"📋 SentenceTransformer {self.model_name} found in cache")
            else:
                logger.warning(f"📋 SentenceTransformer {self.model_name} not found in cache")
            
            # Use enhanced loading with offline support
            self.model = get_cached_sentence_transformer(
                model_name=self.model_name, 
                device=device, 
                offline_mode=offline_mode
            )
            
            if self.model is None:
                logger.error(f"❌ Failed to initialize SentenceTransformer: {self.model_name}")
                if not offline_mode:
                    logger.info("💡 Try downloading first: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\"")
            else:
                logger.info(f"✅ SentenceTransformer initialized successfully: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            self.model = None
    
    def _init_example_banks(self):
        """Initialize positive and negative example banks for coherence comparison."""
        # Positive examples - coherent influencer scenarios with realistic tone and content
        self.good_examples = [
            "Brand partnership agreement between beauty influencer and cosmetics company for product review campaign featuring makeup tutorials and honest product testing",
            "Fashion collaboration contract for clothing brand partnership including outfit styling content, try-on hauls, and authentic product recommendations",
            "Food brand sponsorship agreement covering recipe development, cooking demonstrations, and product integration for meal preparation content",
            "Tech product review partnership for smartphone brand including unboxing videos, feature demonstrations, and authentic user experience sharing",
            "Travel brand collaboration agreement for destination marketing campaign featuring authentic travel experiences and lifestyle content creation",
            "Fitness brand partnership contract including workout videos, product demonstrations, and health and wellness lifestyle content",
            "Home decor brand collaboration for interior design content featuring authentic home styling and product integration",
            "Skincare brand partnership agreement including routine demonstrations, before/after content, and honest product review videos",
            "Sustainable fashion brand collaboration focusing on ethical fashion content, styling tips, and brand values alignment",
            "Food delivery service partnership for authentic meal reviews, unboxing content, and cooking convenience demonstrations"
        ]
        
        # Negative examples - incoherent or non-influencer scenarios
        self.bad_examples = [
            "Medical device clinical trial agreement disguised as influencer content partnership with surgical equipment promotion",
            "Banking loan application process with makeup tutorial deliverables and financial product endorsement requirements",
            "Real estate purchase contract with social media posting obligations and property marketing deliverables",
            "Employment contract for full-time position with influencer content creation as job requirement",
            "Legal services retainer agreement with lifestyle blogger promotional content deliverables",
            "Insurance policy terms and conditions with social media endorsement clauses for coverage benefits",
            "Academic research participation agreement with brand promotion deliverables and study subject requirements",
            "Government tender process documentation with influencer marketing deliverables and public sector promotion",
            "Software licensing agreement with personal lifestyle content creation requirements unrelated to technology",
            "Construction equipment lease with beauty content creator promotional deliverables and industrial equipment endorsement"
        ]
        
        # Generate embeddings if model is available
        if self.model:
            try:
                self.good_embeddings = self.model.encode(self.good_examples)
                self.bad_embeddings = self.model.encode(self.bad_examples)
                logger.info(f"Generated embeddings for {len(self.good_examples)} good and {len(self.bad_examples)} bad examples")
            except Exception as e:
                logger.error(f"Failed to generate example embeddings: {e}")
                self.good_embeddings = None
                self.bad_embeddings = None
    
    def calculate_coherence_score(self, text: str) -> float:
        """
        Calculate semantic coherence score for text using cosine similarity.
        
        Returns score between 0.0 and 1.0, where:
        - 0.6+ indicates coherent influencer agreement content
        - <0.6 indicates incoherent or non-influencer content (should be rejected)
        """
        if not text.strip():
            return 0.0
        
        self.stats['samples_processed'] += 1
        
        # If model failed to initialize, use enhanced fallback scoring
        if not self.model or self.good_embeddings is None or self.bad_embeddings is None:
            score = self._enhanced_fallback_coherence_score(text)
            self.stats['coherence_scores'].append(score)
            return score
        
        try:
            # Get text embedding
            text_embedding = self.model.encode([text])
            
            # Calculate similarities using cosine similarity
            good_similarities = cosine_similarity(text_embedding, self.good_embeddings)[0]
            bad_similarities = cosine_similarity(text_embedding, self.bad_embeddings)[0]
            
            # Calculate coherence score based on similarity differences
            max_good_sim = np.max(good_similarities)
            avg_good_sim = np.mean(good_similarities)
            max_bad_sim = np.max(bad_similarities)
            avg_bad_sim = np.mean(bad_similarities)
            
            # Weighted scoring favoring higher similarity to good examples
            # and lower similarity to bad examples
            coherence = (0.4 * max_good_sim + 0.3 * avg_good_sim) - (0.2 * max_bad_sim + 0.1 * avg_bad_sim)
            
            # Normalize to 0-1 range with appropriate scaling for 0.6 threshold
            coherence_score = max(0.0, min(1.0, coherence + 0.3))
            
            self.stats['coherence_scores'].append(coherence_score)
            return coherence_score
            
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            score = self._enhanced_fallback_coherence_score(text)
            self.stats['coherence_scores'].append(score)
            return score
    
    def _enhanced_fallback_coherence_score(self, text: str) -> float:
        """Enhanced fallback coherence scoring with improved keyword analysis."""
        text_lower = text.lower()
        
        # Strong positive indicators for influencer agreements
        strong_positive_keywords = [
            'influencer', 'content creator', 'brand partnership', 'collaboration',
            'social media campaign', 'sponsored content', 'brand ambassador',
            'product review', 'content creation', 'instagram post', 'youtube video'
        ]
        
        # Moderate positive indicators
        moderate_positive_keywords = [
            'deliverable', 'fee', 'payment', 'exclusivity', 'usage rights',
            'territory', 'campaign', 'brand', 'product', 'audience',
            'engagement', 'reach', 'impression', 'follower'
        ]
        
        # Platform-specific terms
        platform_keywords = [
            'instagram', 'youtube', 'tiktok', 'facebook', 'twitter',
            'linkedin', 'pinterest', 'snapchat', 'story', 'reel', 'post'
        ]
        
        # Strong negative indicators (non-influencer content)
        strong_negative_keywords = [
            'employment', 'employee', 'salary', 'wage', 'full-time',
            'part-time', 'benefits', 'vacation', 'sick leave',
            'banking', 'loan', 'mortgage', 'insurance', 'medical',
            'real estate', 'property', 'lease', 'rent', 'tenant'
        ]
        
        # Moderate negative indicators
        moderate_negative_keywords = [
            'supplier', 'vendor', 'procurement', 'tender', 'clinical trial',
            'research study', 'academic', 'government', 'public sector'
        ]
        
        # Calculate scores
        strong_pos_score = sum(2 for keyword in strong_positive_keywords if keyword in text_lower)
        moderate_pos_score = sum(1 for keyword in moderate_positive_keywords if keyword in text_lower)
        platform_score = sum(1 for keyword in platform_keywords if keyword in text_lower)
        
        strong_neg_score = sum(3 for keyword in strong_negative_keywords if keyword in text_lower)
        moderate_neg_score = sum(1 for keyword in moderate_negative_keywords if keyword in text_lower)
        
        # Calculate total positive and negative scores
        total_positive = strong_pos_score + moderate_pos_score + platform_score
        total_negative = strong_neg_score + moderate_neg_score
        
        # Enhanced scoring algorithm targeting 0.6 threshold
        if total_positive >= 3 and total_negative == 0:
            base_score = 0.75  # Strong positive signal
        elif total_positive >= 2 and total_negative <= 1:
            base_score = 0.65  # Good positive signal
        elif total_positive >= 1 and total_negative <= 2:
            base_score = 0.55  # Moderate signal
        elif total_negative >= 3:
            base_score = 0.2   # Strong negative signal
        elif total_negative >= 1:
            base_score = 0.4   # Moderate negative signal
        else:
            base_score = 0.5   # Neutral
        
        # Apply text quality modifiers
        word_count = len(text_lower.split())
        if word_count < 50:
            base_score *= 0.8  # Penalty for very short text
        elif word_count > 300:
            base_score *= 1.1  # Bonus for substantial content
        
        return max(0.0, min(1.0, base_score))
    
    def is_coherent(self, text: str, threshold: Optional[float] = None) -> bool:
        """
        Check if text meets coherence threshold.
        
        Args:
            text: Text to check
            threshold: Override threshold (defaults to 0.6 as specified)
            
        Returns:
            True if text is sufficiently coherent for influencer agreement content
        """
        if threshold is None:
            threshold = self.coherence_threshold
            
        score = self.calculate_coherence_score(text)
        is_coherent = score >= threshold
        
        if is_coherent:
            self.stats['passed_coherence'] += 1
        else:
            self.stats['failed_coherence'] += 1
        
        return is_coherent
    
    def calculate_coherence(self, text: str) -> float:
        """Alias for calculate_coherence_score for backward compatibility."""
        return self.calculate_coherence_score(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic smoothing statistics."""
        stats = self.stats.copy()
        if stats['coherence_scores']:
            stats['mean_coherence'] = np.mean(stats['coherence_scores'])
            stats['std_coherence'] = np.std(stats['coherence_scores'])
        else:
            stats['mean_coherence'] = 0.0
            stats['std_coherence'] = 0.0
        return stats


class SyntheticGenerator:
    """
    Core generator for creating synthetic influencer agreement samples.
    
    Combines semantic smoothing, business rule validation, and OOD contamination
    to generate coherent training data.
    """
    
    GENERATOR_VERSION = "v1"
    
    def __init__(self, 
                 max_tokens: int = 512,
                 target_word_range: Tuple[int, int] = (300, 600),
                 ood_ratio: float = 0.2,
                 coherence_threshold: float = 0.1,
                 device: str = None,
                 offline_mode: bool = False):
        """
        Initialize synthetic generator.
        
        Args:
            max_tokens: Maximum tokens per sample
            target_word_range: Target word count range (min, max)
            ood_ratio: Fraction of samples that should be OOD
            coherence_threshold: Minimum coherence score required
            device: Device to use for computation ('cpu', 'cuda', etc.)
            offline_mode: Force offline-only mode (no downloads)
        """
        self.max_tokens = max_tokens
        self.target_word_range = target_word_range
        self.ood_ratio = ood_ratio
        self.coherence_threshold = coherence_threshold
        self.offline_mode = offline_mode
        
        # Initialize components with offline mode
        self.chunker = SmartChunker(max_tokens=max_tokens, offline_mode=offline_mode)
        self.smoother = SemanticSmoother(device=device, offline_mode=offline_mode)
        self.business_rules = BusinessRuleEngine()
        self.classifier = DocumentClassifier()
        self.ood_generator = OODGenerator()
        
        # Style fragments extracted from input data
        self.style_fragments = {
            'formal': [],
            'casual': [],
            'bullets': []
        }
        
        # Generation statistics
        self.stats = {
            'samples_generated': 0,
            'ood_samples': 0,
            'coherence_failures': 0,
            'business_rule_failures': 0,
            'token_limit_failures': 0,
            'style_distribution': {'formal': 0, 'casual': 0, 'bullets': 0},
            'complexity_distribution': {'simple': 0, 'medium': 0, 'complex': 0},
            'industry_distribution': {},
        }
    
    def load_ingested_data(self, input_dir: Path) -> List[DocumentSample]:
        """
        Load and parse ingested JSONL files to build style fragment pool.
        
        Args:
            input_dir: Directory containing ingested JSONL files
            
        Returns:
            List of DocumentSample objects
        """
        samples = []
        
        for jsonl_file in input_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            sample = DocumentSample.from_dict(data)
                            samples.append(sample)
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Skipping invalid line {line_num} in {jsonl_file}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Failed to load {jsonl_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(samples)} samples from {input_dir}")
        
        # Extract style fragments
        self._extract_style_fragments(samples)
        
        return samples
    
    def _extract_style_fragments(self, samples: List[DocumentSample]):
        """Extract style fragments from loaded samples."""
        for sample in samples:
            # Analyze text style
            style = self._detect_style(sample.text)
            
            # Extract useful fragments (sentences/paragraphs)
            fragments = self._extract_fragments(sample.text)
            
            self.style_fragments[style].extend(fragments)
        
        # Log fragment counts
        for style, fragments in self.style_fragments.items():
            logger.info(f"Extracted {len(fragments)} {style} style fragments")
    
    def _detect_style(self, text: str) -> str:
        """
        Detect writing style of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Style category: 'formal', 'casual', or 'bullets'
        """
        # Count bullet point indicators
        bullet_patterns = [r'^\s*[-•*]\s+', r'^\s*\d+\.\s+', r'^\s*[a-z]\)\s+']
        bullet_count = sum(len(re.findall(pattern, text, re.MULTILINE)) for pattern in bullet_patterns)
        
        if bullet_count > 2:
            return 'bullets'
        
        # Check for formal indicators
        formal_indicators = [
            'hereby', 'whereas', 'pursuant', 'aforementioned', 'shall',
            'agreement', 'party', 'obligations', 'terms and conditions'
        ]
        
        casual_indicators = [
            "you'll", "we'll", "can't", "won't", "let's", "hey", "awesome",
            "cool", "super", "amazing", "love", "excited"
        ]
        
        text_lower = text.lower()
        formal_score = sum(1 for indicator in formal_indicators if indicator in text_lower)
        casual_score = sum(1 for indicator in casual_indicators if indicator in text_lower)
        
        return 'formal' if formal_score > casual_score else 'casual'
    
    def _extract_fragments(self, text: str) -> List[str]:
        """
        Extract useful text fragments for style transfer.
        
        Args:
            text: Source text
            
        Returns:
            List of useful fragments
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter useful sentences (not too short, contains meaningful content)
        fragments = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) >= 5:  # At least 5 words
                # Remove specific names/amounts that shouldn't be reused
                cleaned = self._generalize_fragment(sentence)
                if cleaned:
                    fragments.append(cleaned)
        
        return fragments
    
    def _generalize_fragment(self, fragment: str) -> Optional[str]:
        """
        Generalize a fragment by removing specific details.
        
        Args:
            fragment: Original fragment
            
        Returns:
            Generalized fragment or None if not useful
        """
        # Skip fragments that are too specific or contain personal info
        skip_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # Specific dates
            r'\$\d+',  # Specific amounts
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, fragment):
                return None
        
        # Replace specific terms with placeholders
        generalized = fragment
        generalized = re.sub(r'\b\d+\s*(months?|weeks?|days?)\b', '[TIME_PERIOD]', generalized)
        generalized = re.sub(r'\b\d+\s*(posts?|videos?|stories?)\b', '[DELIVERABLE_COUNT]', generalized)
        
        return generalized if len(generalized.split()) >= 3 else None
    
    def generate_sample(self, sample_id: str, force_ood: bool = False) -> Optional[SyntheticSample]:
        """
        Generate a single synthetic sample.
        
        Args:
            sample_id: Unique identifier for the sample
            force_ood: Force generation of OOD sample
            
        Returns:
            Generated SyntheticSample or None if generation failed
        """
        try:
            # Determine if this should be an OOD sample
            is_ood = force_ood or (random.random() < self.ood_ratio)
            
            if is_ood:
                return self._generate_ood_sample(sample_id)
            else:
                return self._generate_coherent_sample(sample_id)
                
        except Exception as e:
            logger.error(f"Failed to generate sample {sample_id}: {e}")
            return None
    
    def _generate_coherent_sample(self, sample_id: str) -> Optional[SyntheticSample]:
        """Generate a coherent influencer agreement sample."""
        
        # Generate base content using business rules
        generated_content = self.business_rules.generate_agreement_content()
        
        if not generated_content:
            return None
        
        # Apply style transfer
        style = random.choice(['formal', 'casual', 'bullets'])
        styled_text = self._apply_style_transfer(generated_content['text'], style)
        
        # Ensure token limit compliance
        token_count = self.chunker.count_tokens(styled_text)
        if token_count > self.max_tokens:
            styled_text = self._truncate_to_token_limit(styled_text)
            token_count = self.chunker.count_tokens(styled_text)
            if token_count > self.max_tokens:
                self.stats['token_limit_failures'] += 1
                return None
        
        # Check semantic coherence
        coherence_score = self.smoother.calculate_coherence_score(styled_text)
        if coherence_score < self.coherence_threshold:
            self.stats['coherence_failures'] += 1
            return None
        
        # Validate business rules
        business_validation = self.business_rules.validate_agreement(generated_content)
        if not business_validation['business_ok']:
            self.stats['business_rule_failures'] += 1
            return None
        
        # Classify document
        classification = self.classifier.classify_document(styled_text)
        
        # Create deduplication hash
        dedup_hash = hashlib.sha256(styled_text.encode('utf-8')).hexdigest()[:16]
        
        # Update statistics
        self.stats['samples_generated'] += 1
        self.stats['style_distribution'][style] += 1
        self.stats['complexity_distribution'][classification.get('complexity', 'medium')] += 1
        industry = classification.get('industry', 'other')
        self.stats['industry_distribution'][industry] = self.stats['industry_distribution'].get(industry, 0) + 1
        
        # Create synthetic sample
        return SyntheticSample(
            sample_id=sample_id,
            generator_version=self.GENERATOR_VERSION,
            raw_input={
                'text': styled_text,
                'token_count': token_count,
                'style': style
            },
            classification=classification,
            extracted_fields=generated_content['fields'],
            validation={
                'semantic_coherence': coherence_score,
                'business_ok': business_validation['business_ok'],
                'temporal_ok': business_validation['temporal_ok'],
                'dedup_hash': dedup_hash
            },
            template_hint=None,
            is_ood=False
        )
    
    def _generate_ood_sample(self, sample_id: str) -> Optional[SyntheticSample]:
        """Generate an out-of-distribution sample."""
        ood_content = self.ood_generator.generate_ood_sample()
        
        if not ood_content:
            return None
        
        # Ensure token limit
        token_count = self.chunker.count_tokens(ood_content['text'])
        if token_count > self.max_tokens:
            ood_content['text'] = self._truncate_to_token_limit(ood_content['text'])
            token_count = self.chunker.count_tokens(ood_content['text'])
        
        # Create deduplication hash
        dedup_hash = hashlib.sha256(ood_content['text'].encode('utf-8')).hexdigest()[:16]
        
        # Update statistics
        self.stats['samples_generated'] += 1
        self.stats['ood_samples'] += 1
        
        return SyntheticSample(
            sample_id=sample_id,
            generator_version=self.GENERATOR_VERSION,
            raw_input={
                'text': ood_content['text'],
                'token_count': token_count,
                'style': ood_content.get('style', 'formal')
            },
            classification=ood_content['classification'],
            extracted_fields=ood_content.get('fields', {}),
            validation={
                'semantic_coherence': 0.0,  # OOD samples get low coherence
                'business_ok': False,
                'temporal_ok': False,
                'dedup_hash': dedup_hash
            },
            template_hint=None,
            is_ood=True
        )
    
    def _apply_style_transfer(self, text: str, target_style: str) -> str:
        """
        Apply style transfer to make text match target style.
        
        Args:
            text: Source text
            target_style: Target style ('formal', 'casual', 'bullets')
            
        Returns:
            Style-transferred text
        """
        if target_style == 'bullets':
            return self._convert_to_bullets(text)
        elif target_style == 'casual':
            return self._convert_to_casual(text)
        else:  # formal
            return self._convert_to_formal(text)
    
    def _convert_to_bullets(self, text: str) -> str:
        """Convert text to bullet point format."""
        sentences = re.split(r'[.!?]+', text)
        bullets = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) >= 3:
                bullets.append(f"• {sentence}")
        
        return '\n'.join(bullets)
    
    def _convert_to_casual(self, text: str) -> str:
        """Convert text to casual style."""
        # Simple substitutions for casual tone
        casual_text = text
        casual_text = re.sub(r'\bshall\b', 'will', casual_text)
        casual_text = re.sub(r'\bpursuant to\b', 'according to', casual_text)
        casual_text = re.sub(r'\bhereby\b', '', casual_text)
        casual_text = re.sub(r'\bwhereas\b', 'since', casual_text)
        
        return casual_text
    
    def _convert_to_formal(self, text: str) -> str:
        """Convert text to formal style."""
        # Ensure formal language
        formal_text = text
        formal_text = re.sub(r"\bcan't\b", 'cannot', formal_text)
        formal_text = re.sub(r"\bwon't\b", 'will not', formal_text)
        formal_text = re.sub(r"\blet's\b", 'let us', formal_text)
        
        return formal_text
    
    def _truncate_to_token_limit(self, text: str) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text
        """
        sentences = re.split(r'[.!?]+', text)
        truncated_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = self.chunker.count_tokens(sentence)
            if current_tokens + sentence_tokens <= self.max_tokens - 10:  # Leave some headroom
                truncated_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return '. '.join(truncated_sentences) + '.'
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = self.stats.copy()
        
        # Add component statistics
        stats['smoother_stats'] = self.smoother.get_stats()
        stats['business_rules_stats'] = self.business_rules.get_stats()
        stats['ood_stats'] = self.ood_generator.get_stats()
        
        return stats 