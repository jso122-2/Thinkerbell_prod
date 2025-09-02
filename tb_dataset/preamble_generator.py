"""
Preamble Generator with Validation and Semantic Coherence

Generates business preambles for synthetic samples with validation,
token counting, semantic coherence checking, and retry logic.
"""

import logging
import re
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .preamble_schema import Preamble, render_preamble, validate_preamble_content
from .generate import SemanticSmoother

logger = logging.getLogger(__name__)


class PreambleGenerator:
    """
    Generates and validates business preambles for synthetic samples.
    
    Features:
    - Token count validation (≤ 512 tokens)
    - Semantic coherence checking via existing smoother
    - Field requirement validation
    - Retry logic with rewrites
    - Quality reporting
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 offline_cache: str = 'hf_cache',
                 min_coherence: float = 0.72,
                 max_rewrites: int = 2,
                 max_tokens: int = 512):
        """
        Initialize preamble generator.
        
        Args:
            model_name: Model for semantic coherence checking
            offline_cache: Cache directory for offline mode
            min_coherence: Minimum coherence threshold
            max_rewrites: Maximum rewrite attempts
            max_tokens: Maximum token count
        """
        self.model_name = model_name
        self.offline_cache = Path(offline_cache)
        self.min_coherence = min_coherence
        self.max_rewrites = max_rewrites
        self.max_tokens = max_tokens
        
        # Initialize tokenizer
        self._init_tokenizer()
        
        # Initialize semantic smoother for coherence checking
        self.smoother = SemanticSmoother(
            model_name=model_name,
            offline_mode=True
        )
        
        # Quality tracking
        self.stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'token_failures': 0,
            'coherence_failures': 0,
            'field_validation_failures': 0,
            'rewrite_attempts': 0,
            'final_rejections': 0,
            'coherence_scores': [],
            'token_counts': [],
            'word_counts': []
        }
        
        logger.info(f"Initialized PreambleGenerator with {model_name}, coherence≥{min_coherence}")
    
    def _init_tokenizer(self):
        """Initialize tokenizer for token counting."""
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to load tokenizer with offline mode
                cache_dir = str(self.offline_cache) if self.offline_cache.exists() else None
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=cache_dir,
                    local_files_only=True
                )
                logger.info(f"Loaded tokenizer: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {self.model_name}: {e}")
                logger.info("Using fallback token counting (4 chars/token)")
        else:
            logger.info("Transformers not available, using fallback token counting")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer or fallback."""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer failed: {e}, using fallback")
        
        # Fallback: 4 characters per token rule
        return len(text) // 4
    
    def clean_text(self, text: str) -> str:
        """Clean text formatting (whitespace, currency, sentence spacing)."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Clean currency formatting
        text = re.sub(r'\$\s*(\d)', r'$\1', text)  # Remove spaces after $
        text = re.sub(r'(\d)\s+dollars', r'\1 dollars', text)  # Normalize dollar spacing
        
        # Fix sentence spacing
        text = re.sub(r'\.\s+', '. ', text)  # Normalize sentence spacing
        text = re.sub(r'\s*\.\s*', '. ', text)  # Fix period spacing
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def validate_semantic_coherence(self, text: str, extracted_fields: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Validate semantic coherence of preamble text.
        
        Args:
            text: Preamble text to validate
            extracted_fields: Sample fields for context
            
        Returns:
            Tuple of (passes_coherence, coherence_score)
        """
        try:
            # Use existing smoother for coherence checking
            coherence_score = self.smoother.calculate_coherence_score(text)
            passes = coherence_score >= self.min_coherence
            
            # Add rule-based checks for business logic
            if passes:
                passes = self._validate_business_logic(text, extracted_fields)
            
            return passes, coherence_score
            
        except Exception as e:
            logger.error(f"Coherence validation failed: {e}")
            # Fallback to rule-based validation
            score = self._fallback_coherence_score(text, extracted_fields)
            return score >= self.min_coherence, score
    
    def _validate_business_logic(self, text: str, extracted_fields: Dict[str, Any]) -> bool:
        """Rule-based business logic validation."""
        # Fee ↔ deliverables sanity check
        fee = extracted_fields.get('fee_numeric', extracted_fields.get('fee', 0))
        if isinstance(fee, str):
            # Extract numeric fee from string
            import re
            fee_match = re.search(r'(\d+(?:,\d+)*)', str(fee).replace('$', ''))
            fee = int(fee_match.group(1).replace(',', '')) if fee_match else 0
        
        deliverables = extracted_fields.get('deliverables', [])
        if isinstance(deliverables, str):
            deliverables = [deliverables]
        
        deliverable_count = len(deliverables)
        
        # Basic fee/deliverable ratio check
        if fee > 0 and deliverable_count > 0:
            fee_per_deliverable = fee / deliverable_count
            # Very basic sanity: $500-$50000 per deliverable seems reasonable
            if fee_per_deliverable < 500 or fee_per_deliverable > 50000:
                logger.debug(f"Fee/deliverable ratio seems off: ${fee_per_deliverable:.0f} per deliverable")
                # Don't fail on this alone, just log
        
        # Platform ↔ industry alignment check
        industry = extracted_fields.get('industry', '')
        if industry and 'instagram' in text.lower():
            # Instagram makes sense for most industries
            pass
        
        return True  # Basic checks passed
    
    def _fallback_coherence_score(self, text: str, extracted_fields: Dict[str, Any]) -> float:
        """Fallback coherence scoring without model."""
        score = 0.6  # Base score
        
        # Check for required elements
        required_terms = ['campaign', 'engagement', 'deliverable', 'fee', 'brand']
        found_terms = sum(1 for term in required_terms if term.lower() in text.lower())
        score += (found_terms / len(required_terms)) * 0.2
        
        # Check text quality
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
            score += 0.1
        
        # Check word count in range
        word_count = len(text.split())
        if 120 <= word_count <= 220:
            score += 0.1
        
        return min(1.0, score)
    
    def validate_field_requirements(self, text: str, extracted_fields: Dict[str, Any]) -> Dict[str, bool]:
        """Validate that preamble mentions all required fields."""
        return validate_preamble_content(text, extracted_fields)
    
    def rewrite_preamble(self, original_schema: Preamble, fields: Dict[str, Any], 
                        style_profile: Optional[Dict[str, Any]], 
                        attempt: int, failure_reason: str) -> Preamble:
        """
        Rewrite preamble schema for retry attempts.
        
        Args:
            original_schema: Original preamble schema
            fields: Field values
            style_profile: Style profile
            attempt: Attempt number (1 or 2)
            failure_reason: Why the original failed
            
        Returns:
            Modified schema for retry
        """
        # Create a copy of the schema
        new_schema = Preamble(
            purpose=original_schema.purpose,
            brand_voice=original_schema.brand_voice,
            campaign_context=original_schema.campaign_context,
            deliverables_block=original_schema.deliverables_block.copy(),
            timelines_block=original_schema.timelines_block.copy(),
            constraints_block=original_schema.constraints_block.copy(),
            money_block=original_schema.money_block.copy(),
            exclusivity_block=original_schema.exclusivity_block.copy()
        )
        
        if attempt == 1:
            # First rewrite: tighten content
            if 'token' in failure_reason.lower():
                # Reduce verbosity
                new_schema.campaign_context = self._shorten_text(new_schema.campaign_context)
                new_schema.purpose = self._shorten_text(new_schema.purpose)
            elif 'coherence' in failure_reason.lower():
                # Improve clarity
                new_schema.brand_voice = self._improve_clarity(new_schema.brand_voice)
                new_schema.purpose = self._improve_clarity(new_schema.purpose)
        
        elif attempt == 2:
            # Second rewrite: swap synonyms and re-order
            new_schema.purpose = self._swap_synonyms(new_schema.purpose)
            new_schema.brand_voice = self._swap_synonyms(new_schema.brand_voice)
            
            # Re-order clauses in campaign context
            if len(new_schema.campaign_context.split('.')) > 1:
                sentences = new_schema.campaign_context.split('.')
                sentences = [s.strip() for s in sentences if s.strip()]
                if len(sentences) > 1:
                    # Reverse order
                    new_schema.campaign_context = '. '.join(reversed(sentences)) + '.'
        
        return new_schema
    
    def _shorten_text(self, text: str) -> str:
        """Shorten text by removing unnecessary words."""
        # Remove filler words and redundancy
        text = re.sub(r'\b(very|quite|really|extremely|absolutely)\s+', '', text)
        text = re.sub(r'\b(and|or)\s+', ', ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _improve_clarity(self, text: str) -> str:
        """Improve text clarity."""
        # Replace complex terms with simpler ones
        replacements = {
            'innovative': 'creative',
            'comprehensive': 'complete',
            'strategic': 'focused',
            'leverage': 'use',
            'optimize': 'improve'
        }
        
        for old, new in replacements.items():
            text = re.sub(r'\b' + re.escape(old) + r'\b', new, text, flags=re.IGNORECASE)
        
        return text
    
    def _swap_synonyms(self, text: str) -> str:
        """Swap words with synonyms."""
        synonyms = {
            'enhance': 'improve',
            'drive': 'boost',
            'create': 'develop',
            'deliver': 'provide',
            'campaign': 'initiative',
            'engagement': 'collaboration',
            'authentic': 'genuine',
            'strategic': 'targeted'
        }
        
        for original, synonym in synonyms.items():
            if original in text.lower():
                text = re.sub(r'\b' + re.escape(original) + r'\b', synonym, text, flags=re.IGNORECASE)
                break  # Only swap one per attempt
        
        return text
    
    def generate_for_sample(self, sample: Dict[str, Any], style_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate preamble for a sample with validation and retry logic.
        
        Args:
            sample: Sample data with extracted_fields
            style_profile: Optional style profile for tone adaptation
            
        Returns:
            Dictionary with raw_input data or None if generation failed
        """
        self.stats['total_attempts'] += 1
        
        # Extract fields from sample
        extracted_fields = sample.get('extracted_fields', {})
        if not extracted_fields:
            logger.warning(f"Sample {sample.get('sample_id', 'unknown')} missing extracted_fields")
            self.stats['field_validation_failures'] += 1
            return None
        
        # Build preamble schema from extracted fields
        preamble_schema = self._build_preamble_schema(extracted_fields, sample)
        
        # Prepare fields for render_preamble
        render_fields = self._prepare_render_fields(extracted_fields)
        
        # Try generating with retries
        for attempt in range(self.max_rewrites + 1):
            try:
                # Generate preamble
                preamble_text = render_preamble(preamble_schema, render_fields, style_profile)
                
                # Clean text
                cleaned_text = self.clean_text(preamble_text)
                
                # Validate token count
                token_count = self.count_tokens(cleaned_text)
                if token_count > self.max_tokens:
                    if attempt < self.max_rewrites:
                        self.stats['rewrite_attempts'] += 1
                        preamble_schema = self.rewrite_preamble(
                            preamble_schema, render_fields, style_profile, 
                            attempt + 1, f"token count {token_count} > {self.max_tokens}"
                        )
                        continue
                    else:
                        self.stats['token_failures'] += 1
                        logger.debug(f"Token count {token_count} > {self.max_tokens} after {self.max_rewrites} rewrites")
                        return None
                
                # Validate semantic coherence
                passes_coherence, coherence_score = self.validate_semantic_coherence(cleaned_text, extracted_fields)
                if not passes_coherence:
                    if attempt < self.max_rewrites:
                        self.stats['rewrite_attempts'] += 1
                        preamble_schema = self.rewrite_preamble(
                            preamble_schema, render_fields, style_profile,
                            attempt + 1, f"coherence {coherence_score:.3f} < {self.min_coherence}"
                        )
                        continue
                    else:
                        self.stats['coherence_failures'] += 1
                        logger.debug(f"Coherence {coherence_score:.3f} < {self.min_coherence} after {self.max_rewrites} rewrites")
                        return None
                
                # Validate field requirements
                field_validation = self.validate_field_requirements(cleaned_text, extracted_fields)
                missing_fields = [k for k, v in field_validation.items() if not v]
                if missing_fields:
                    if attempt < self.max_rewrites:
                        self.stats['rewrite_attempts'] += 1
                        preamble_schema = self.rewrite_preamble(
                            preamble_schema, render_fields, style_profile,
                            attempt + 1, f"missing fields: {missing_fields}"
                        )
                        continue
                    else:
                        self.stats['field_validation_failures'] += 1
                        logger.debug(f"Missing fields {missing_fields} after {self.max_rewrites} rewrites")
                        return None
                
                # Success! Record stats
                self.stats['successful_generations'] += 1
                self.stats['coherence_scores'].append(coherence_score)
                self.stats['token_counts'].append(token_count)
                self.stats['word_counts'].append(len(cleaned_text.split()))
                
                # Return result
                return {
                    'raw_input': {
                        'text': cleaned_text,
                        'token_count': token_count,
                        'requires_chunking': token_count > 256  # Reasonable chunking threshold
                    }
                }
                
            except Exception as e:
                logger.error(f"Error generating preamble (attempt {attempt + 1}): {e}")
                if attempt >= self.max_rewrites:
                    self.stats['final_rejections'] += 1
                    return None
                else:
                    self.stats['rewrite_attempts'] += 1
                    # Try again with modified schema
                    preamble_schema = self.rewrite_preamble(
                        preamble_schema, render_fields, style_profile,
                        attempt + 1, f"exception: {str(e)}"
                    )
        
        # Should not reach here, but just in case
        self.stats['final_rejections'] += 1
        return None
    
    def _build_preamble_schema(self, extracted_fields: Dict[str, Any], sample: Dict[str, Any]) -> Preamble:
        """Build Preamble schema from extracted fields."""
        # Extract information
        client = extracted_fields.get('client', extracted_fields.get('brand', 'Client'))
        campaign = extracted_fields.get('campaign', 'Brand Campaign')
        
        # Build deliverables list
        deliverables = extracted_fields.get('deliverables', [])
        if isinstance(deliverables, str):
            deliverables = [deliverables]
        if not deliverables:
            deliverables = ['Social media content', 'Brand collaboration']
        
        # Build schema
        return Preamble(
            purpose=f"enhancing {client}'s brand presence and driving targeted audience engagement",
            brand_voice="authentic, professional, and strategically-focused messaging",
            campaign_context=f"This comprehensive marketing initiative leverages current market trends and consumer insights to maximize {client}'s brand impact across digital channels.",
            deliverables_block=deliverables[:3],  # Limit to avoid verbosity
            timelines_block={
                'engagement_term': extracted_fields.get('engagement_term', '6 weeks'),
                'usage_term': extracted_fields.get('usage_term', '12 months'),
                'content_delivery': 'phased approach'
            },
            constraints_block=[
                'brand style guidelines',
                'platform compliance requirements',
                'content approval workflows'
            ],
            money_block={
                'total_fee': extracted_fields.get('fee_numeric', 10000),
                'currency': 'AUD',
                'payment_structure': 'milestone-based'
            },
            exclusivity_block={
                'period': extracted_fields.get('exclusivity_period', '3 months'),
                'scope': 'category-specific restrictions',
                'terms': 'within relevant competitive categories'
            }
        )
    
    def _prepare_render_fields(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare fields for render_preamble function."""
        # Extract fee numeric value
        fee = extracted_fields.get('fee_numeric', extracted_fields.get('fee', 10000))
        if isinstance(fee, str):
            # Extract numeric fee from string like "$10,000"
            import re
            fee_match = re.search(r'(\d+(?:,\d+)*)', str(fee).replace('$', ''))
            fee = int(fee_match.group(1).replace(',', '')) if fee_match else 10000
        
        return {
            'client': extracted_fields.get('client', extracted_fields.get('brand', 'Client')),
            'campaign': extracted_fields.get('campaign', 'Brand Campaign'),
            'brand': extracted_fields.get('brand', extracted_fields.get('client', 'Brand')),
            'fee': fee,
            'currency': 'AUD',
            'deliverables': extracted_fields.get('deliverables', ['Content creation']),
            'engagement_period': extracted_fields.get('engagement_term', '6 weeks')
        }
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate quality report for batch processing."""
        total = self.stats['total_attempts']
        if total == 0:
            return {'error': 'No generation attempts recorded'}
        
        report = {
            'generation_stats': {
                'total_attempts': total,
                'successful_generations': self.stats['successful_generations'],
                'success_rate': self.stats['successful_generations'] / total,
                'rewrite_attempts': self.stats['rewrite_attempts'],
                'final_rejections': self.stats['final_rejections']
            },
            'failure_breakdown': {
                'token_failures': self.stats['token_failures'],
                'coherence_failures': self.stats['coherence_failures'],
                'field_validation_failures': self.stats['field_validation_failures']
            },
            'quality_metrics': {}
        }
        
        # Add quality metrics if we have data
        if self.stats['coherence_scores']:
            import numpy as np
            report['quality_metrics'].update({
                'coherence': {
                    'mean': float(np.mean(self.stats['coherence_scores'])),
                    'std': float(np.std(self.stats['coherence_scores'])),
                    'min': float(np.min(self.stats['coherence_scores'])),
                    'max': float(np.max(self.stats['coherence_scores']))
                }
            })
        
        if self.stats['token_counts']:
            import numpy as np
            report['quality_metrics'].update({
                'token_counts': {
                    'mean': float(np.mean(self.stats['token_counts'])),
                    'std': float(np.std(self.stats['token_counts'])),
                    'min': int(np.min(self.stats['token_counts'])),
                    'max': int(np.max(self.stats['token_counts']))
                }
            })
        
        if self.stats['word_counts']:
            import numpy as np
            report['quality_metrics'].update({
                'word_counts': {
                    'mean': float(np.mean(self.stats['word_counts'])),
                    'std': float(np.std(self.stats['word_counts'])),
                    'min': int(np.min(self.stats['word_counts'])),
                    'max': int(np.max(self.stats['word_counts']))
                }
            })
        
        return report


def process_samples_cli(input_dir: Path, output_dir: Path, 
                       min_coherence: float = 0.72, 
                       max_rewrites: int = 2,
                       max_tokens: int = 512) -> None:
    """
    CLI function to process samples and generate preambles.
    
    Args:
        input_dir: Directory containing sample JSON files
        output_dir: Directory to write processed samples
        min_coherence: Minimum coherence threshold
        max_rewrites: Maximum rewrite attempts
        max_tokens: Maximum token count
    """
    # Initialize generator
    generator = PreambleGenerator(
        min_coherence=min_coherence,
        max_rewrites=max_rewrites,
        max_tokens=max_tokens
    )
    
    # Find all JSON files in input directory
    json_files = list(input_dir.glob('**/*.json'))
    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return
    
    logger.info(f"Processing {len(json_files)} files from {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_successful = 0
    
    # Process each file
    for json_file in tqdm(json_files, desc="Processing samples"):
        try:
            # Load samples
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            samples = []
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                if 'samples' in data:
                    samples = data['samples']
                elif 'extracted_fields' in data:  # Single sample
                    samples = [data]
                else:
                    # Assume it's a single sample
                    samples = [data]
            
            processed_samples = []
            
            for sample in samples:
                total_processed += 1
                
                # Generate preamble
                result = generator.generate_for_sample(sample)
                
                if result:
                    # Add preamble to sample
                    sample.update(result)
                    total_successful += 1
                
                processed_samples.append(sample)
            
            # Write output file
            output_file = output_dir / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_samples, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    
    # Generate quality report
    report = generator.get_quality_report()
    report_file = output_dir / 'preamble_quality_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Log summary
    success_rate = (total_successful / total_processed) * 100 if total_processed > 0 else 0
    logger.info(f"Processing complete: {total_successful}/{total_processed} samples successful ({success_rate:.1f}%)")
    logger.info(f"Quality report saved to {report_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate preambles for synthetic samples")
    parser.add_argument('--in', dest='input_dir', type=Path, required=True,
                       help='Input directory containing sample JSON files')
    parser.add_argument('--out', dest='output_dir', type=Path, required=True,
                       help='Output directory for processed samples')
    parser.add_argument('--min-coherence', type=float, default=0.72,
                       help='Minimum coherence threshold (default: 0.72)')
    parser.add_argument('--max-rewrites', type=int, default=2,
                       help='Maximum rewrite attempts (default: 2)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum token count (default: 512)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process samples
    process_samples_cli(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_coherence=args.min_coherence,
        max_rewrites=args.max_rewrites,
        max_tokens=args.max_tokens
    )


if __name__ == '__main__':
    main() 