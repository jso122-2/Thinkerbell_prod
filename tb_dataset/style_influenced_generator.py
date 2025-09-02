"""
Style-influenced synthetic content generator.

Generates synthetic agreements that maintain the tone, structure, and 
phrasing patterns from real templates while creating original content.
"""

import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm function
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .template_ingestor import StyleProfile
from .generate import SyntheticGenerator, SyntheticSample
from .business_rules import BusinessRuleEngine
from .ood import OODGenerator
from .preamble_generator import PreambleGenerator

logger = logging.getLogger(__name__)


class StyleInfluencedGenerator:
    """
    Generates synthetic content influenced by real template styles.
    
    Combines style profiles from real templates with synthetic content
    generation to produce realistic agreements that maintain authentic
    tone and structure.
    """
    
    def __init__(self, style_profiles: Dict[str, StyleProfile], device: str = None, offline_mode: bool = False, cache_dir: Optional[Path] = None,
                 with_preambles: bool = True, min_coherence: float = 0.72, max_rewrites: int = 2):
        """
        Initialize style-influenced generator.
        
        Args:
            style_profiles: Dictionary mapping profile_id to StyleProfile
            device: Device to use for computation ('cpu', 'cuda', etc.)
            offline_mode: Force offline-only mode (no downloads)
            cache_dir: Directory for caching (used for mapping diagnostics)
            with_preambles: Whether to generate preambles for samples
            min_coherence: Minimum coherence threshold for preambles
            max_rewrites: Maximum preamble rewrite attempts
        """
        self.style_profiles = style_profiles
        self.device = device
        self.offline_mode = offline_mode
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./synthetic_dataset/cache")
        self.with_preambles = with_preambles
        self.min_coherence = min_coherence
        self.max_rewrites = max_rewrites
        
        # Initialize mapping diagnostics for generator-level failures
        from .mapping_diagnostics import MappingDiagnostics
        self.mapping_diagnostics = MappingDiagnostics(self.cache_dir)
        
        # Initialize base components
        self.base_generator = SyntheticGenerator(device=device, offline_mode=offline_mode)
        self.business_rules = BusinessRuleEngine()
        self.ood_generator = OODGenerator()
        
        # Initialize tokenizer for token counting with offline mode
        from .chunk import SmartChunker
        self.tokenizer = SmartChunker(offline_mode=offline_mode)
        
        # Initialize preamble generator if enabled
        self.preamble_generator = None
        if self.with_preambles:
            self.preamble_generator = PreambleGenerator(
                min_coherence=min_coherence,
                max_rewrites=max_rewrites
            )
        
        # Style application settings
        self.style_influence_strength = 0.7  # How much to apply style (0.0-1.0)
        
        logger.info(f"Initialized style-influenced generator with {len(style_profiles)} profiles, preambles={'enabled' if with_preambles else 'disabled'}")
    
    def generate_sample(self, 
                       sample_id: str,
                       target_industry: Optional[str] = None,
                       target_complexity: Optional[str] = None,
                       style_profile_id: Optional[str] = None,
                       is_ood: bool = False) -> Dict[str, Any]:
        """
        Generate a style-influenced synthetic sample.
        
        Args:
            sample_id: Unique identifier for the sample
            target_industry: Target industry (auto-detected if None)
            target_complexity: Target complexity (auto-selected if None)
            style_profile_id: Specific style profile to use (auto-selected if None)
            is_ood: Whether to generate an OOD sample
            
        Returns:
            Complete synthetic sample data
        """
        if is_ood:
            ood_sample = self.ood_generator.generate_ood_sample()
            if ood_sample:
                ood_sample['sample_id'] = sample_id
                return ood_sample
            else:
                raise ValueError("Failed to generate OOD sample")
        
        # Select style profile
        if style_profile_id and style_profile_id in self.style_profiles:
            style_profile = self.style_profiles[style_profile_id]
        else:
            style_profile = self._select_style_profile(target_industry, target_complexity)
        
        # Generate base content using business rules
        agreement_data = self.business_rules.generate_agreement_content()
        
        if not agreement_data:
            raise ValueError("Failed to generate agreement content")
        
        agreement_content = agreement_data.get('text', '')
        extracted_fields = agreement_data.get('fields', {})
        
        # Apply style influence
        styled_content = self._apply_style_influence(agreement_content, style_profile)
        
        # Generate preamble in style
        preamble = self._generate_styled_preamble(style_profile, target_industry, target_complexity)
        
        # Combine preamble and content
        full_content = f"{preamble}\n\n{styled_content}"
        
        # Count tokens and apply length constraints
        token_count = self.tokenizer.count_tokens(full_content)
        if token_count > 512:
            # Trim content while preserving structure
            full_content = self._trim_content_preserving_style(full_content, style_profile, 512)
            token_count = self.tokenizer.count_tokens(full_content)
        
        # Classify the document
        try:
            classification = self.business_rules.classify_document(full_content)
        except AttributeError:
            # Fallback classification
            classification = {
                'document_type': 'INFLUENCER_AGREEMENT',
                'complexity': target_complexity or 'medium',
                'industry': target_industry or 'fashion'
            }
        
        # Validate agreement
        validation = self.business_rules.validate_agreement(extracted_fields)
        
        # Add semantic coherence
        try:
            from .generate import SemanticSmoother
            smoother = SemanticSmoother()
            coherence = smoother.calculate_coherence(full_content)
            validation['semantic_coherence'] = coherence
        except Exception:
            validation['semantic_coherence'] = 0.8  # Default high score for style-influenced content
        
        # Determine style from profile
        style = self._determine_output_style(style_profile)
        
        # Create initial sample data
        sample_data = {
            "sample_id": sample_id,
            "generator_version": "v1_style_influenced",
            "raw_input": {
                "text": full_content,
                "token_count": token_count,
                "style": style
            },
            "classification": classification,
            "extracted_fields": extracted_fields,
            "validation": validation,
            "template_hint": None,
            "is_ood": False,
            "style_metadata": {
                "style_profile_id": style_profile.profile_id,
                "formality_score": style_profile.formality_score,
                "dominant_tone": style_profile.tone_markers.get('dominant_tone', 'formal'),
                "industry_indicators": list(style_profile.industry_indicators)[:5]
            }
        }
        
        # Generate preamble if enabled
        if self.preamble_generator:
            try:
                # Convert style profile to dict format for preamble generator
                style_profile_dict = {
                    "tone_markers": style_profile.tone_markers,
                    "style_keywords": list(style_profile.industry_indicators)[:15]
                }
                
                preamble_result = self.preamble_generator.generate_for_sample(
                    sample_data, 
                    style_profile_dict
                )
                
                if preamble_result:
                    # Update sample with preamble data
                    sample_data.update(preamble_result)
                    logger.debug(f"Generated preamble for sample {sample_id}")
                else:
                    # Preamble generation failed after rewrites - drop sample
                    logger.warning(f"Preamble generation failed for sample {sample_id}, dropping sample")
                    return None
                    
            except Exception as e:
                logger.error(f"Error generating preamble for sample {sample_id}: {e}")
                if self.max_rewrites > 0:  # If we're allowing rewrites, this is a hard failure
                    return None
                # Otherwise, continue without preamble
                logger.warning(f"Continuing without preamble for sample {sample_id}")
        
        return sample_data
    
    def get_preamble_stats(self) -> Optional[Dict[str, Any]]:
        """Get preamble generation statistics."""
        if self.preamble_generator:
            return self.preamble_generator.get_quality_report()
        return None
    
    def _select_style_profile(self, 
                             target_industry: Optional[str] = None,
                             target_complexity: Optional[str] = None) -> StyleProfile:
        """Select the most appropriate style profile."""
        if not self.style_profiles:
            raise ValueError("No style profiles available")
        
        # If only one profile, use it
        if len(self.style_profiles) == 1:
            selected_profile = list(self.style_profiles.values())[0]
            
            # Record the selection for diagnostics
            self.mapping_diagnostics.record_profile_selection_failure(
                target_industry=target_industry,
                target_complexity=target_complexity,
                available_profiles=[selected_profile.profile_id],
                profile_scores={selected_profile.profile_id: 1.0},
                selected_profile=selected_profile.profile_id,
                selection_reason="only_profile_available"
            )
            
            return selected_profile
        
        # Score profiles based on target criteria
        profile_scores = {}
        
        for profile_id, profile in self.style_profiles.items():
            score = 0.0
            
            # Industry matching
            if target_industry:
                industry_terms = [term for term in profile.industry_indicators 
                                if target_industry.lower() in term.lower()]
                score += len(industry_terms) * 2.0
            
            # Complexity matching
            if target_complexity and target_complexity in profile.complexity_indicators:
                complexity_data = profile.complexity_indicators[target_complexity]
                score += complexity_data.get('pattern_count', 0) * 1.0
            
            # Preference for more templates (more robust profile)
            score += profile.template_count * 0.1
            
            profile_scores[profile_id] = score
        
        # Select highest scoring profile, with randomness for ties
        selected_profile = None
        selection_reason = "unknown"
        
        if profile_scores:
            # Add small random factor to break ties
            for profile_id in profile_scores:
                profile_scores[profile_id] += random.random() * 0.1
            
            best_profile_id = max(profile_scores, key=profile_scores.get)
            selected_profile = self.style_profiles[best_profile_id]
            
            max_score = max(profile_scores.values())
            if max_score > 1.0:
                selection_reason = "high_score_match"
            elif max_score > 0.1:
                selection_reason = "low_score_match"
            else:
                selection_reason = "minimal_score_match"
        else:
            # Fallback: random selection
            selected_profile = random.choice(list(self.style_profiles.values()))
            profile_scores = {pid: 0.0 for pid in self.style_profiles.keys()}
            selection_reason = "random_fallback"
        
        # Record profile selection for diagnostics
        self.mapping_diagnostics.record_profile_selection_failure(
            target_industry=target_industry,
            target_complexity=target_complexity,
            available_profiles=list(self.style_profiles.keys()),
            profile_scores=profile_scores,
            selected_profile=selected_profile.profile_id,
            selection_reason=selection_reason
        )
        
        return selected_profile
    
    def _apply_style_influence(self, content: str, style_profile: StyleProfile) -> str:
        """Apply style profile influence to content."""
        styled_content = content
        
        # Apply tone adjustments
        styled_content = self._apply_tone_style(styled_content, style_profile)
        
        # Apply phrase vocabulary
        styled_content = self._apply_phrase_vocabulary(styled_content, style_profile)
        
        # Apply sentence structure patterns
        styled_content = self._apply_sentence_patterns(styled_content, style_profile)
        
        # Apply formality adjustments
        styled_content = self._apply_formality_level(styled_content, style_profile)
        
        return styled_content
    
    def _apply_tone_style(self, content: str, style_profile: StyleProfile) -> str:
        """Apply tone style from profile."""
        tone_markers = style_profile.tone_markers
        dominant_tone = tone_markers.get('dominant_tone', 'formal')
        
        if dominant_tone == 'formal':
            # Make content more formal
            replacements = {
                r'\bwe\'ll\b': 'we will',
                r'\bcan\'t\b': 'cannot',
                r'\bwon\'t\b': 'will not',
                r'\bdon\'t\b': 'do not',
                r'\bisn\'t\b': 'is not',
                r'\blet\'s\b': 'let us',
                r'\bthat\'s\b': 'that is',
                r'\bit\'s\b': 'it is'
            }
            
            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        elif dominant_tone == 'casual':
            # Keep content conversational
            replacements = {
                r'\bcannot\b': 'can\'t',
                r'\bwill not\b': 'won\'t',
                r'\bdo not\b': 'don\'t',
                r'\bis not\b': 'isn\'t',
                r'\bthat is\b': 'that\'s',
                r'\bit is\b': 'it\'s'
            }
            
            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _apply_phrase_vocabulary(self, content: str, style_profile: StyleProfile) -> str:
        """Apply characteristic phrases from the style profile."""
        phrases = style_profile.phrase_vocabulary
        
        if not phrases:
            return content
        
        # Select a few phrases to incorporate
        selected_phrases = random.sample(list(phrases), min(3, len(phrases)))
        
        # Simple substitution approach - replace generic terms with style-specific ones
        for phrase in selected_phrases:
            if 'agreement' in phrase.lower() and 'this agreement' in content.lower():
                content = content.replace('this agreement', phrase, 1)
            elif 'party' in phrase.lower() and 'the party' in content.lower():
                content = content.replace('the party', phrase, 1)
            elif 'terms' in phrase.lower() and 'the terms' in content.lower():
                content = content.replace('the terms', phrase, 1)
        
        return content
    
    def _apply_sentence_patterns(self, content: str, style_profile: StyleProfile) -> str:
        """Apply sentence structure patterns from the style profile."""
        structures = style_profile.sentence_structures
        
        if not structures:
            return content
        
        # Calculate average characteristics
        avg_length = sum(s.get('length', 15) for s in structures) / len(structures)
        has_subordinate_ratio = sum(1 for s in structures if s.get('has_subordinate_clause', False)) / len(structures)
        
        sentences = re.split(r'([.!?]+)', content)
        modified_sentences = []
        
        for i in range(0, len(sentences), 2):  # Process sentence pairs (text + punctuation)
            if i < len(sentences):
                sentence = sentences[i].strip()
                punctuation = sentences[i+1] if i+1 < len(sentences) else '.'
                
                if sentence:
                    # Adjust sentence based on profile patterns
                    modified_sentence = self._adjust_sentence_structure(sentence, avg_length, has_subordinate_ratio)
                    modified_sentences.append(modified_sentence + punctuation)
        
        return ' '.join(modified_sentences)
    
    def _adjust_sentence_structure(self, sentence: str, target_length: float, subordinate_ratio: float) -> str:
        """Adjust individual sentence structure."""
        words = sentence.split()
        current_length = len(words)
        
        # If sentence is much shorter than target, try to add detail
        if current_length < target_length * 0.7 and random.random() < 0.3:
            # Add clarifying phrases
            clarifiers = [
                'as specified herein',
                'in accordance with this agreement',
                'subject to the terms outlined',
                'pursuant to the provisions',
                'as mutually agreed upon'
            ]
            
            if random.random() < subordinate_ratio:
                clarifier = random.choice(clarifiers)
                # Insert clarifier at appropriate position
                if ',' in sentence:
                    parts = sentence.split(',', 1)
                    sentence = f"{parts[0]}, {clarifier},{parts[1]}"
                else:
                    sentence = f"{sentence}, {clarifier}"
        
        return sentence
    
    def _apply_formality_level(self, content: str, style_profile: StyleProfile) -> str:
        """Apply formality level adjustments."""
        formality_score = style_profile.formality_score
        
        if formality_score > 0.7:  # High formality
            # Add formal language patterns
            formal_replacements = {
                r'\bget\b': 'obtain',
                r'\buse\b': 'utilize',
                r'\bhelp\b': 'assist',
                r'\bshow\b': 'demonstrate',
                r'\bmake sure\b': 'ensure',
                r'\bfind out\b': 'determine'
            }
            
            for pattern, replacement in formal_replacements.items():
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        elif formality_score < 0.3:  # Low formality
            # Use more casual language
            casual_replacements = {
                r'\bobtain\b': 'get',
                r'\butilize\b': 'use',
                r'\bassist\b': 'help',
                r'\bdemonstrate\b': 'show',
                r'\bensure\b': 'make sure',
                r'\bdetermine\b': 'find out'
            }
            
            for pattern, replacement in casual_replacements.items():
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _generate_styled_preamble(self, 
                                 style_profile: StyleProfile,
                                 industry: Optional[str] = None,
                                 complexity: Optional[str] = None) -> str:
        """Generate a preamble that matches the style profile."""
        
        # Base preamble structure
        if style_profile.formality_score > 0.6:
            # Formal preamble
            preamble_template = """INFLUENCER MARKETING AGREEMENT

This Agreement ("Agreement") is entered into on {date} between {client} ("Client") and {influencer} ("Influencer") for the purpose of creating and distributing promotional content as specified herein.

WHEREAS, Client desires to engage Influencer's services for marketing and promotional activities; and
WHEREAS, Influencer possesses the requisite skills, platform, and audience to effectively promote Client's products/services;

NOW, THEREFORE, in consideration of the mutual covenants and agreements contained herein, the parties agree as follows:"""
        
        elif style_profile.formality_score < 0.4:
            # Casual preamble  
            preamble_template = """INFLUENCER COLLABORATION AGREEMENT

Hey there! This agreement is between {client} and {influencer} for an awesome collaboration that we're both excited about.

We're partnering up to create some amazing content that'll showcase {client}'s brand in an authentic way that resonates with {influencer}'s community.

Here's what we've agreed on:"""
        
        else:
            # Medium formality preamble
            preamble_template = """INFLUENCER PARTNERSHIP AGREEMENT

This agreement outlines the partnership between {client} and {influencer} for a collaborative marketing campaign.

Both parties agree to work together to create engaging content that promotes {client}'s brand while maintaining {influencer}'s authentic voice and style.

Campaign Details:"""
        
        # Fill in placeholders with style-appropriate content
        client_name = self._generate_styled_client_name(style_profile, industry)
        influencer_name = self._generate_styled_influencer_name(style_profile)
        date = "____________"  # To be filled in
        
        preamble = preamble_template.format(
            client=client_name,
            influencer=influencer_name,
            date=date
        )
        
        # Apply style-specific vocabulary
        if style_profile.industry_indicators:
            industry_terms = list(style_profile.industry_indicators)[:3]
            # Subtly incorporate industry terms
            for term in industry_terms:
                if term not in preamble.lower():
                    # Add term in appropriate context
                    if 'fashion' in term:
                        preamble = preamble.replace('products/services', f'{term} products')
                    elif 'food' in term:
                        preamble = preamble.replace('products/services', f'{term} offerings')
        
        return preamble
    
    def _generate_styled_client_name(self, style_profile: StyleProfile, industry: Optional[str] = None) -> str:
        """Generate a client name that fits the style profile."""
        if style_profile.formality_score > 0.6:
            # Formal business names
            if industry == 'fashion':
                return "Elegant Designs Ltd."
            elif industry == 'food':
                return "Gourmet Selections Corp."
            elif industry == 'tech':
                return "Innovation Systems Inc."
            else:
                return "Premium Brands Ltd."
        else:
            # Casual brand names
            if industry == 'fashion':
                return "Style Studio"
            elif industry == 'food':
                return "Tasty Treats Co."
            elif industry == 'tech':
                return "Tech Innovators"
            else:
                return "Cool Brand Co."
    
    def _generate_styled_influencer_name(self, style_profile: StyleProfile) -> str:
        """Generate an influencer name that fits the style profile."""
        if style_profile.formality_score > 0.6:
            return "[Influencer Name]"
        else:
            return "@InfluencerHandle"
    
    def _trim_content_preserving_style(self, content: str, style_profile: StyleProfile, max_tokens: int) -> str:
        """Trim content while preserving style characteristics."""
        current_tokens = self.tokenizer.count_tokens(content)
        
        if current_tokens <= max_tokens:
            return content
        
        # Split into sections
        sections = content.split('\n\n')
        
        # Preserve preamble (first section) and key sections
        essential_sections = [sections[0]]  # Always keep preamble
        
        # Add sections until we approach token limit
        remaining_tokens = max_tokens - self.tokenizer.count_tokens(essential_sections[0])
        
        for section in sections[1:]:
            section_tokens = self.tokenizer.count_tokens(section)
            if remaining_tokens >= section_tokens:
                essential_sections.append(section)
                remaining_tokens -= section_tokens
            else:
                # Trim this section if possible
                trimmed_section = self._trim_section(section, remaining_tokens, style_profile)
                if trimmed_section:
                    essential_sections.append(trimmed_section)
                break
        
        return '\n\n'.join(essential_sections)
    
    def _trim_section(self, section: str, max_tokens: int, style_profile: StyleProfile) -> str:
        """Trim a section while preserving style."""
        sentences = re.split(r'([.!?]+)', section)
        trimmed_sentences = []
        current_tokens = 0
        
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                punctuation = sentences[i+1] if i+1 < len(sentences) else '.'
                
                if sentence:
                    sentence_tokens = self.tokenizer.count_tokens(sentence + punctuation)
                    if current_tokens + sentence_tokens <= max_tokens:
                        trimmed_sentences.append(sentence + punctuation)
                        current_tokens += sentence_tokens
                    else:
                        break
        
        return ' '.join(trimmed_sentences)
    
    def _determine_output_style(self, style_profile: StyleProfile) -> str:
        """Determine output style classification based on profile."""
        if style_profile.formality_score > 0.6:
            return 'formal'
        elif style_profile.formality_score < 0.4:
            return 'casual'
        else:
            return 'professional'
    
    def generate_batch(self,
                      sample_count: int,
                      ood_ratio: float = 0.2,
                      industry_distribution: Optional[Dict[str, float]] = None,
                      complexity_distribution: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Generate a batch of style-influenced samples.
        
        Args:
            sample_count: Number of samples to generate
            ood_ratio: Ratio of OOD samples (0.0-1.0)
            industry_distribution: Distribution of industries
            complexity_distribution: Distribution of complexities
            
        Returns:
            List of generated samples
        """
        logger.info(f"Generating batch of {sample_count} style-influenced samples")
        
        samples = []
        ood_count = int(sample_count * ood_ratio)
        regular_count = sample_count - ood_count
        
        # Default distributions
        if not industry_distribution:
            industry_distribution = {'fashion': 0.4, 'food': 0.3, 'tech': 0.2, 'home': 0.1}
        
        if not complexity_distribution:
            complexity_distribution = {'simple': 0.5, 'medium': 0.3, 'complex': 0.2}
        
        # Generate regular samples with progress bar
        regular_range = range(regular_count)
        regular_progress = tqdm(regular_range, desc="Generating regular samples", unit="sample") if TQDM_AVAILABLE else regular_range
        
        for i in regular_progress:
            sample_id = f"sample_{i:06d}"
            
            # Select industry and complexity based on distribution
            industry = self._weighted_random_choice(industry_distribution)
            complexity = self._weighted_random_choice(complexity_distribution)
            
            try:
                sample = self.generate_sample(
                    sample_id=sample_id,
                    target_industry=industry,
                    target_complexity=complexity,
                    is_ood=False
                )
                
                # Ensure sample has required ID
                if sample and 'sample_id' in sample:
                    samples.append(sample)
                    if TQDM_AVAILABLE:
                        regular_progress.set_description(f"Generated {len(samples)} samples")
                else:
                    logger.warning(f"Generated sample {sample_id} missing sample_id field")
                
            except Exception as e:
                logger.warning(f"Failed to generate sample {sample_id}: {e}")
                continue
        
        # Generate OOD samples with progress bar
        ood_range = range(regular_count, sample_count)
        ood_progress = tqdm(ood_range, desc="Generating OOD samples", unit="sample") if TQDM_AVAILABLE else ood_range
        
        for i in ood_progress:
            sample_id = f"sample_{i:06d}"
            
            try:
                sample = self.generate_sample(
                    sample_id=sample_id,
                    is_ood=True
                )
                
                # Ensure sample has required ID
                if sample and 'sample_id' in sample:
                    samples.append(sample)
                    if TQDM_AVAILABLE:
                        ood_progress.set_description(f"Generated {len(samples)} total samples")
                else:
                    logger.warning(f"Generated OOD sample {sample_id} missing sample_id field")
                
            except Exception as e:
                logger.warning(f"Failed to generate OOD sample {sample_id}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(samples)} samples")
        
        # Save mapping diagnostics from this generation session
        self.mapping_diagnostics.save_failures_to_disk()
        
        # Log diagnostics summary
        diagnostics_summary = self.mapping_diagnostics.get_failure_summary()
        if diagnostics_summary['total_failures'] > 0:
            logger.warning(f"🔍 Profile selection diagnostics: {diagnostics_summary['total_failures']} issues recorded")
            logger.warning(f"📊 Profile selection success rate: {diagnostics_summary['success_rate']:.1f}%")
        
        return samples
    
    def _weighted_random_choice(self, weights: Dict[str, float]) -> str:
        """Make a weighted random choice from a distribution."""
        items = list(weights.keys())
        probabilities = list(weights.values())
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        return random.choices(items, weights=probabilities)[0] 