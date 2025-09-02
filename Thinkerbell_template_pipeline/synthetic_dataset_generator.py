#!/usr/bin/env python3
"""
Synthetic Influencer Agreement Dataset Generator
Generates realistic variations of influencer agreements for training AI document formatter
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re

# Import semantic smoother
try:
    # Try thinkerbell.utils path first since that's where the module exists
    from thinkerbell.utils.semantic_smoother import SemanticSmoother, BusinessLogicValidator
    HAS_SEMANTIC_SMOOTHER = True
except ImportError:
    try:
        # Fallback to direct import
        from semantic_smoother import SemanticSmoother, BusinessLogicValidator
        HAS_SEMANTIC_SMOOTHER = True
    except ImportError:
        HAS_SEMANTIC_SMOOTHER = False
        print("⚠️ Semantic smoother not available - using basic generation")

# Import text preprocessor
try:
    # Try thinkerbell.utils path first since that's where the module exists
    from thinkerbell.utils.text_preprocessor import TextPreprocessor
    HAS_TEXT_PREPROCESSOR = True
except ImportError:
    try:
        # Fallback to direct import
        from text_preprocessor import TextPreprocessor
        HAS_TEXT_PREPROCESSOR = True
    except ImportError:
        HAS_TEXT_PREPROCESSOR = False
        print("⚠️ Text preprocessor not available - using basic text processing")

# Import OOD contaminator
try:
    from ood_contamination import OODContaminator, OODSample
    HAS_OOD_CONTAMINATOR = True
except ImportError:
    try:
        from thinkerbell.utils.ood_contamination import OODContaminator, OODSample
        HAS_OOD_CONTAMINATOR = True
    except ImportError:
        HAS_OOD_CONTAMINATOR = False
        # Define a fallback OODSample class if import fails
        from dataclasses import dataclass
        from typing import Dict, List, Optional
        
        @dataclass
        class OODSample:
            """Fallback OODSample class when ood_contamination module is not available"""
            text: str
            label: str
            confidence_target: float
            sample_type: str
            ood_indicators: List[str]
            should_process: bool
            fallback_response: Optional[str] = None
            extracted_fields: Optional[Dict] = None
        
        print("⚠️ OOD contaminator not available - using basic generation")

class SyntheticDatasetGenerator:
    def __init__(self, use_semantic_smoothing: bool = True, use_text_preprocessing: bool = True, 
                 use_ood_contamination: bool = True, contamination_ratio: float = 0.2):
        # Initialize semantic smoother if available
        self.use_semantic_smoothing = use_semantic_smoothing and HAS_SEMANTIC_SMOOTHER
        if self.use_semantic_smoothing:
            self.semantic_smoother = SemanticSmoother()
            self.business_validator = BusinessLogicValidator()
            print("✅ Semantic smoother enabled")
        else:
            self.semantic_smoother = None
            self.business_validator = None
            print("⚠️ Using basic generation without semantic smoothing")
        
        # Initialize text preprocessor if available
        self.use_text_preprocessing = use_text_preprocessing and HAS_TEXT_PREPROCESSOR
        if self.use_text_preprocessing:
            self.text_preprocessor = TextPreprocessor()
            print("✅ Text preprocessor enabled")
        else:
            self.text_preprocessor = None
            print("⚠️ Using basic text processing")
        
        # Initialize OOD contaminator if available
        self.use_ood_contamination = use_ood_contamination and HAS_OOD_CONTAMINATOR
        self.contamination_ratio = contamination_ratio
        if self.use_ood_contamination:
            self.ood_contaminator = OODContaminator()
            print(f"✅ OOD contamination enabled (ratio: {contamination_ratio:.1%})")
        else:
            self.ood_contaminator = None
            print("⚠️ Using basic generation without OOD contamination")
        
        # Australian brands/clients pool - organized by industry
        self.australian_brands = {
            "fashion": [
                "Cotton On", "Country Road", "David Jones", "Myer", "Witchery", 
                "Portmans", "Sportsgirl", "Target", "Kmart", "Big W"
            ],
            "food": [
                "Woolworths", "Coles", "Queen Fine Foods", "Boost Juice", 
                "Guzman y Gomez", "Mad Mex", "Zambrero", "Roll'd", "Subway", 
                "Hungry Jack's", "KFC", "McDonald's", "Domino's", "Pizza Hut", 
                "Red Rooster", "Nando's", "Grill'd", "Betty's Burgers"
            ],
            "tech": [
                "JB Hi-Fi", "Harvey Norman", "Officeworks", "Telstra", 
                "Commonwealth Bank", "Qantas"
            ],
            "home": [
                "Bunnings", "IKEA", "Freedom", "Adairs", "Bed Bath N' Table", 
                "Pillow Talk", "Amart Furniture", "Fantastic Furniture"
            ],
            "beauty": [
                "Chemist Warehouse", "Priceline", "Sephora", "Mecca", "Lush"
            ],
            "automotive": [
                "Supercheap Auto", "Autobarn", "Repco"
            ]
        }
        
        # Flatten brands list for backward compatibility
        self.all_brands = []
        for industry_brands in self.australian_brands.values():
            self.all_brands.extend(industry_brands)
        
        # Influencer names pool
        self.influencer_names = [
            "Sarah Chen", "James Wilson", "Emma Thompson", "Michael Rodriguez", "Olivia Davis",
            "David Brown", "Sophie Anderson", "Christopher Lee", "Isabella Garcia", "Daniel Martinez",
            "Ava Johnson", "Matthew Taylor", "Mia White", "Andrew Clark", "Chloe Lewis",
            "Joshua Hall", "Lily Walker", "Ryan Allen", "Grace Young", "Nathan King",
            "Zoe Wright", "Tyler Scott", "Hannah Green", "Kevin Baker", "Natalie Adams",
            "Brandon Nelson", "Victoria Carter", "Justin Mitchell", "Aria Perez", "Sean Roberts",
            "Scarlett Turner", "Jordan Phillips", "Layla Campbell", "Cameron Parker", "Riley Evans",
            "Brooklyn Edwards", "Skylar Collins", "Peyton Stewart", "Morgan Morris", "Reagan Rogers",
            "Sage Reed", "Quinn Cook", "Blake Morgan", "Hayden Bell", "River Murphy",
            "Rowan Bailey", "Sage Cooper", "Quinn Richardson", "Blake Cox", "Hayden Howard"
        ]
        
        # Fee ranges by tier - updated with semantic constraints
        self.fee_tiers = {
            "micro": (1500, 5000),
            "mid": (5000, 15000),
            "premium": (15000, 35000)
        }
        
        # Deliverables options - organized by complexity
        self.deliverables = {
            "simple": [
                "Instagram post", "Instagram story", "Facebook post", "Twitter post",
                "LinkedIn post", "Pinterest pin", "15-second TikTok", "30-second Instagram reel"
            ],
            "medium": [
                "Instagram post", "Instagram story", "Instagram reel", "TikTok video",
                "Facebook post", "Facebook story", "YouTube short", "YouTube video",
                "LinkedIn post", "Twitter post", "Pinterest pin", "60-second YouTube video",
                "90-second product demo", "2-minute tutorial", "Product photography",
                "Lifestyle photography", "Media interview", "Radio interview"
            ],
            "premium": [
                "Instagram post", "Instagram story", "Instagram reel", "TikTok video",
                "Facebook post", "Facebook story", "YouTube short", "YouTube video",
                "LinkedIn post", "Twitter post", "Pinterest pin", "60-second YouTube video",
                "90-second product demo", "2-minute tutorial", "3-minute vlog",
                "5-minute review", "10-minute unboxing", "Product photography",
                "Lifestyle photography", "Event photography", "Behind-the-scenes content",
                "Professional headshots", "Brand photoshoot", "Product launch event",
                "Store opening", "Press conference", "Media interview", "Radio interview",
                "TV appearance", "Podcast guest", "Live stream", "Meet and greet",
                "Press release quote", "Blog post", "Article contribution",
                "Social media takeover", "Brand ambassador content"
            ]
        }
        
        # Campaign types by industry
        self.campaign_types = {
            "fashion": [
                "Spring Collection Launch", "Winter Range", "New Season", "Limited Edition",
                "Collaboration", "Accessories Launch", "Seasonal Promotion"
            ],
            "food": [
                "Product Launch", "Seasonal Campaign", "Recipe Series", "Health Initiative",
                "Food Safety Campaign", "Healthy Eating", "New Product Launch"
            ],
            "tech": [
                "Product Launch", "Tech Review", "Digital Services", "App Promotion",
                "New Technology", "Product Review", "Tech Tips"
            ],
            "home": [
                "Product Launch", "DIY Series", "Home Tips", "Collection Showcase",
                "Home Improvement", "Furniture Collection", "Decor Tips"
            ],
            "beauty": [
                "Product Launch", "Beauty Tips", "Skincare Routine", "Collection Review",
                "Skincare Launch", "Makeup Collection", "Beauty Tips", "Product Review"
            ],
            "automotive": [
                "Product Launch", "Travel Series", "Adventure Content", "Service Promotion",
                "Car Accessories", "Travel Services", "Product Review"
            ]
        }
        
        # Products/categories
        self.product_categories = [
            "Fashion & Apparel", "Beauty & Cosmetics", "Food & Beverage", "Technology & Electronics",
            "Home & Garden", "Travel & Tourism", "Finance & Insurance", "Health & Fitness",
            "Automotive", "Education", "Entertainment", "Sports & Recreation"
        ]
        
        # Template styles based on existing agreements
        self.template_styles = [
            "formal_contract", "casual_brief", "email_style", "bullet_points", "mixed_format"
        ]
        
        # Complexity levels
        self.complexity_levels = ["simple", "medium", "complex"]
        
        # Quality tracking
        self.quality_metrics = {
            "total_generated": 0,
            "passed_validation": 0,
            "token_length_stats": {"min": 0, "max": 0, "avg": 0},
            "quality_score_stats": {"min": 0, "max": 0, "avg": 0},
            "chunking_stats": {"total_chunked": 0, "avg_chunks": 0},
            "ood_stats": {
                "positive_samples": 0,
                "ood_negative_samples": 0,
                "edge_case_samples": 0,
                "contamination_ratio": 0.0
            }
        }
        
    def generate_synthetic_agreement(self, complexity: str = "medium") -> Dict:
        """Generate a single synthetic influencer agreement with semantic smoothing and text preprocessing"""
        
        if self.use_semantic_smoothing:
            agreement = self._generate_with_semantic_smoothing(complexity)
        else:
            agreement = self._generate_basic_agreement(complexity)
        
        # Apply text preprocessing if available
        if self.use_text_preprocessing and self.text_preprocessor:
            agreement = self._apply_text_preprocessing(agreement)
        
        # Update quality metrics
        self._update_quality_metrics(agreement)
        
        return agreement
    
    def _convert_ood_sample_to_agreement(self, ood_sample: OODSample) -> Dict:
        """Convert OOD sample to agreement format"""
        
        agreement = {
            "id": f"ood_{str(uuid.uuid4())[:8]}",
            "raw_input_text": ood_sample.text,
            "extracted_fields": ood_sample.extracted_fields or {},
            "template_match": "ood_sample",
            "complexity_level": "ood",
            "confidence_score": ood_sample.confidence_target,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sample_type": ood_sample.sample_type,
                "ood_indicators": ood_sample.ood_indicators,
                "should_process": ood_sample.should_process,
                "fallback_response": ood_sample.fallback_response
            },
            "classification": ood_sample.label,
            "confidence_target": ood_sample.confidence_target
        }
        
        # Apply text preprocessing if available
        if self.use_text_preprocessing and self.text_preprocessor:
            agreement = self._apply_text_preprocessing(agreement)
        
        return agreement
    
    def generate_dataset_with_ood_contamination(self, num_samples: int = 1000, 
                                              complexity_distribution: Dict[str, float] = None) -> List[Dict]:
        """Generate dataset with intentional OOD contamination"""
        
        if complexity_distribution is None:
            complexity_distribution = {
                "simple": 0.3,
                "medium": 0.5,
                "complex": 0.2
            }
        
        dataset = []
        
        # Calculate sample distribution
        num_positive = int(num_samples * (1 - self.contamination_ratio))
        num_ood_negative = int(num_samples * self.contamination_ratio * 0.5)
        num_edge_cases = int(num_samples * self.contamination_ratio * 0.5)
        
        print(f"📊 Generating dataset with OOD contamination:")
        print(f"  Positive samples: {num_positive}")
        print(f"  OOD negative samples: {num_ood_negative}")
        print(f"  Edge case samples: {num_edge_cases}")
        print(f"  Total contamination: {self.contamination_ratio:.1%}")
        
        # Generate positive samples
        print(f"\n🔄 Generating {num_positive} positive samples...")
        for i in range(num_positive):
            complexity = random.choices(
                list(complexity_distribution.keys()),
                weights=list(complexity_distribution.values())
            )[0]
            
            agreement = self.generate_synthetic_agreement(complexity)
            agreement["classification"] = "INFLUENCER_AGREEMENT"
            agreement["confidence_target"] = 0.85
            agreement["metadata"]["sample_type"] = "positive"
            
            dataset.append(agreement)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_positive} positive samples")
        
        # Generate OOD negative samples
        if self.use_ood_contamination and self.ood_contaminator:
            print(f"\n🔄 Generating {num_ood_negative} OOD negative samples...")
            for i in range(num_ood_negative):
                ood_sample = self.ood_contaminator.generate_ood_negative()
                agreement = self._convert_ood_sample_to_agreement(ood_sample)
                dataset.append(agreement)
                
                if (i + 1) % 50 == 0:
                    print(f"  Generated {i + 1}/{num_ood_negative} OOD negative samples")
        
        # Generate edge case samples
        if self.use_ood_contamination and self.ood_contaminator:
            print(f"\n🔄 Generating {num_edge_cases} edge case samples...")
            for i in range(num_edge_cases):
                edge_sample = self.ood_contaminator.generate_edge_case()
                agreement = self._convert_ood_sample_to_agreement(edge_sample)
                dataset.append(agreement)
                
                if (i + 1) % 50 == 0:
                    print(f"  Generated {i + 1}/{num_edge_cases} edge case samples")
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        # Update OOD statistics
        self._update_ood_statistics(dataset)
        
        # Print semantic smoother quality report if available
        if self.use_semantic_smoothing and self.semantic_smoother:
            self.semantic_smoother.print_quality_report()
        
        # Print text preprocessing quality report
        self._print_quality_report()
        
        return dataset
    
    def _update_ood_statistics(self, dataset: List[Dict]):
        """Update OOD statistics"""
        
        positive_count = sum(1 for item in dataset if item.get("metadata", {}).get("sample_type") == "positive")
        ood_negative_count = sum(1 for item in dataset if item.get("metadata", {}).get("sample_type") == "ood_negative")
        edge_case_count = sum(1 for item in dataset if item.get("metadata", {}).get("sample_type") == "edge_case")
        
        self.quality_metrics["ood_stats"]["positive_samples"] = positive_count
        self.quality_metrics["ood_stats"]["ood_negative_samples"] = ood_negative_count
        self.quality_metrics["ood_stats"]["edge_case_samples"] = edge_case_count
        self.quality_metrics["ood_stats"]["contamination_ratio"] = (ood_negative_count + edge_case_count) / len(dataset)
    
    def _apply_text_preprocessing(self, agreement: Dict) -> Dict:
        """Apply text preprocessing to the agreement"""
        
        # Process the raw input text
        processed = self.text_preprocessor.process_text(
            agreement['raw_input_text'],
            agreement.get('extracted_fields', {})
        )
        
        # Update agreement with processed text
        agreement['processed_text'] = {
            'cleaned_text': processed.cleaned_text,
            'token_count': processed.token_count,
            'requires_chunking': processed.requires_chunking,
            'quality_score': processed.quality_score
        }
        
        if processed.requires_chunking:
            agreement['processed_text']['chunks'] = processed.chunks
            agreement['processed_text']['chunk_labels'] = processed.chunk_labels
        
        # Validate the sample
        valid, checks = self.text_preprocessor.validate_training_sample(agreement)
        agreement['validation'] = {
            'valid_for_training': valid,
            'checks': checks
        }
        
        return agreement
    
    def _update_quality_metrics(self, agreement: Dict):
        """Update quality tracking metrics"""
        
        self.quality_metrics["total_generated"] += 1
        
        if agreement.get('validation', {}).get('valid_for_training', False):
            self.quality_metrics["passed_validation"] += 1
        
        # Update token length stats
        processed_text = agreement.get('processed_text')
        if processed_text:
            # Handle both ProcessedText dataclass and dict
            if hasattr(processed_text, 'token_count'):
                token_count = processed_text.token_count
            else:
                token_count = processed_text.get('token_count', 0)
            
            if token_count > 0:
                current_min = self.quality_metrics["token_length_stats"]["min"]
                current_max = self.quality_metrics["token_length_stats"]["max"]
                current_avg = self.quality_metrics["token_length_stats"]["avg"]
                
                if current_min == 0 or token_count < current_min:
                    self.quality_metrics["token_length_stats"]["min"] = token_count
                if token_count > current_max:
                    self.quality_metrics["token_length_stats"]["max"] = token_count
                
                # Update average
                total_samples = self.quality_metrics["total_generated"]
                new_avg = ((current_avg * (total_samples - 1)) + token_count) / total_samples
                self.quality_metrics["token_length_stats"]["avg"] = new_avg
            
            # Update quality score stats
            if hasattr(processed_text, 'quality_score'):
                quality_score = processed_text.quality_score
            else:
                quality_score = processed_text.get('quality_score', 0.0)
            
            if quality_score > 0:
                current_min = self.quality_metrics["quality_score_stats"]["min"]
                current_max = self.quality_metrics["quality_score_stats"]["max"]
                current_avg = self.quality_metrics["quality_score_stats"]["avg"]
                
                if current_min == 0 or quality_score < current_min:
                    self.quality_metrics["quality_score_stats"]["min"] = quality_score
                if quality_score > current_max:
                    self.quality_metrics["quality_score_stats"]["max"] = quality_score
                
                # Update average
                total_samples = self.quality_metrics["total_generated"]
                new_avg = ((current_avg * (total_samples - 1)) + quality_score) / total_samples
                self.quality_metrics["quality_score_stats"]["avg"] = new_avg
            
            # Update chunking stats
            if hasattr(processed_text, 'requires_chunking'):
                requires_chunking = processed_text.requires_chunking
            else:
                requires_chunking = processed_text.get('requires_chunking', False)
            
            if requires_chunking:
                self.quality_metrics["chunking_stats"]["total_chunked"] += 1
                
                if hasattr(processed_text, 'chunks'):
                    chunks = processed_text.chunks
                else:
                    chunks = processed_text.get('chunks', [])
                
                if chunks:
                    current_avg_chunks = self.quality_metrics["chunking_stats"]["avg_chunks"]
                    total_chunked = self.quality_metrics["chunking_stats"]["total_chunked"]
                    new_avg_chunks = ((current_avg_chunks * (total_chunked - 1)) + len(chunks)) / total_chunked
                    self.quality_metrics["chunking_stats"]["avg_chunks"] = new_avg_chunks
    
    def _generate_with_semantic_smoothing(self, complexity: str) -> Dict:
        """Generate agreement using semantic smoother"""
        
        # Get coherent parameters based on industry
        brand = self._select_coherent_brand(complexity)
        industry = self._get_industry_for_brand(brand)
        
        # Get coherent parameters for this brand and complexity
        if self.business_validator:
            coherent_params = self.business_validator.get_coherent_parameters(brand, complexity)
        else:
            coherent_params = {}
        
        # Generate with semantic constraints
        agreement = self._generate_constrained_agreement(brand, industry, complexity, coherent_params)
        
        # Apply semantic smoothing
        if self.semantic_smoother:
            smoothed_agreement = self.semantic_smoother.generate_coherent_sample(self, attempts=5)
            if smoothed_agreement:
                return smoothed_agreement
        
        return agreement
    
    def _generate_basic_agreement(self, complexity: str) -> Dict:
        """Generate agreement without semantic smoothing (original method)"""
        
        # Randomly select parameters
        brand = random.choice(self.all_brands)
        influencer = random.choice(self.influencer_names)
        fee_tier = random.choice(list(self.fee_tiers.keys()))
        fee_min, fee_max = self.fee_tiers[fee_tier]
        fee = random.randint(fee_min, fee_max)
        
        # Generate deliverables
        num_deliverables = random.randint(1, 5)
        deliverables_list = []
        for _ in range(num_deliverables):
            category = random.choice(list(self.deliverables.keys()))
            deliverable = random.choice(self.deliverables[category])
            quantity = random.randint(1, 3)
            deliverables_list.append(f"{quantity} x {deliverable}")
        
        # Generate dates
        start_date = datetime.now() + timedelta(days=random.randint(30, 180))
        exclusivity_period = random.randint(2, 12)
        engagement_term = random.randint(2, 6)
        usage_term = random.randint(6, 12)
        
        # Generate raw input text with variations
        raw_text = self._generate_raw_text(
            brand, influencer, fee, deliverables_list, 
            start_date, exclusivity_period, engagement_term, usage_term, complexity
        )
        
        # Extract fields (simulated extraction)
        extracted_fields = self._extract_fields(
            brand, influencer, fee, deliverables_list,
            start_date, exclusivity_period, engagement_term, usage_term
        )
        
        # Determine template match
        template_match = self._determine_template_match(complexity, deliverables_list)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(complexity, raw_text)
        
        return {
            "id": f"synth_{str(uuid.uuid4())[:8]}",
            "raw_input_text": raw_text,
            "extracted_fields": extracted_fields,
            "template_match": template_match,
            "complexity_level": complexity,
            "confidence_score": confidence_score,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "fee_tier": fee_tier,
                "num_deliverables": len(deliverables_list),
                "campaign_type": random.choice(self.campaign_types.get("fashion", ["Product Launch"])),
                "product_category": random.choice(self.product_categories)
            }
        }
    
    def _select_coherent_brand(self, complexity: str) -> str:
        """Select a brand that makes sense for the complexity level"""
        
        # For premium complexity, prefer established brands
        if complexity == "premium":
            premium_brands = ["David Jones", "Myer", "Woolworths", "Coles", "JB Hi-Fi", "Telstra"]
            return random.choice(premium_brands)
        elif complexity == "simple":
            simple_brands = ["Cotton On", "Target", "Kmart", "Boost Juice", "Subway"]
            return random.choice(simple_brands)
        else:
            # Medium complexity - mix of brands
            return random.choice(self.all_brands)
    
    def _get_industry_for_brand(self, brand: str) -> str:
        """Get industry for a given brand"""
        for industry, brands in self.australian_brands.items():
            if brand in brands:
                return industry
        return "fashion"  # Default fallback
    
    def _generate_constrained_agreement(self, brand: str, industry: str, complexity: str, coherent_params: Dict) -> Dict:
        """Generate agreement with semantic constraints"""
        
        # Select influencer based on industry
        influencer = random.choice(self.influencer_names)
        
        # Get fee range based on complexity
        fee_logic = {
            "simple": (1500, 8000),
            "medium": (8000, 18000),
            "premium": (18000, 35000)
        }
        fee_min, fee_max = fee_logic[complexity]
        fee = random.randint(fee_min, fee_max)
        
        # Generate deliverables based on complexity
        deliverable_options = self.deliverables[complexity]
        num_deliverables = random.randint(1, 3) if complexity == "simple" else random.randint(2, 5)
        deliverables_list = []
        for _ in range(num_deliverables):
            deliverable = random.choice(deliverable_options)
            quantity = random.randint(1, 3)
            deliverables_list.append(f"{quantity} x {deliverable}")
        
        # Generate campaign type based on industry
        campaign_type = random.choice(self.campaign_types.get(industry, ["Product Launch"]))
        
        # Generate dates with semantic constraints
        start_date = datetime.now() + timedelta(days=random.randint(30, 180))
        
        # Temporal constraints based on complexity
        if complexity == "simple":
            exclusivity_period = random.randint(2, 8)
            engagement_term = random.randint(1, 3)
            usage_term = random.randint(6, 12)
        elif complexity == "medium":
            exclusivity_period = random.randint(4, 12)
            engagement_term = random.randint(2, 6)
            usage_term = random.randint(8, 18)
        else:  # premium
            exclusivity_period = random.randint(8, 24)
            engagement_term = random.randint(3, 12)
            usage_term = random.randint(12, 36)
        
        # Generate raw input text
        raw_text = self._generate_raw_text(
            brand, influencer, fee, deliverables_list, 
            start_date, exclusivity_period, engagement_term, usage_term, complexity
        )
        
        # Extract fields
        extracted_fields = self._extract_fields(
            brand, influencer, fee, deliverables_list,
            start_date, exclusivity_period, engagement_term, usage_term
        )
        
        # Update campaign type in extracted fields
        extracted_fields["campaign"] = f"{brand} {campaign_type}"
        
        # Determine template match
        template_match = self._determine_template_match(complexity, deliverables_list)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(complexity, raw_text)
        
        return {
            "id": f"synth_{str(uuid.uuid4())[:8]}",
            "raw_input_text": raw_text,
            "extracted_fields": extracted_fields,
            "template_match": template_match,
            "complexity_level": complexity,
            "confidence_score": confidence_score,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "fee_tier": "mid" if fee < 15000 else "premium",
                "num_deliverables": len(deliverables_list),
                "campaign_type": campaign_type,
                "product_category": self._get_product_category_for_industry(industry),
                "industry": industry
            }
        }
    
    def _get_product_category_for_industry(self, industry: str) -> str:
        """Get product category for industry"""
        category_map = {
            "fashion": "Fashion & Apparel",
            "food": "Food & Beverage", 
            "tech": "Technology & Electronics",
            "home": "Home & Garden",
            "beauty": "Beauty & Cosmetics",
            "automotive": "Automotive"
        }
        return category_map.get(industry, "Fashion & Apparel")
    
    def _generate_raw_text(self, brand: str, influencer: str, fee: int, 
                          deliverables: List[str], start_date: datetime,
                          exclusivity_period: int, engagement_term: int, 
                          usage_term: int, complexity: str) -> str:
        """Generate realistic raw input text with variations"""
        
        # Base text variations
        text_variations = [
            # Formal brief
            f"""Need {influencer} for {brand} campaign. Budget around ${fee:,}, looking for {', '.join(deliverables)}. 
            {exclusivity_period} week exclusivity from other {self._get_competitor_category(brand)} brands. 
            Campaign runs {start_date.strftime('%B %Y')}. Usage rights for {usage_term} months. 
            Engagement period: {engagement_term} months.""",
            
            # Casual email style
            f"""Hey team, looking to work with {influencer} for {brand}. 
            Budget is approx ${fee:,} - need {', '.join(deliverables)}. 
            {exclusivity_period} weeks exclusivity, {engagement_term} month engagement. 
            Usage: {usage_term} months. Start date: {start_date.strftime('%B %Y')}.""",
            
            # Bullet point style
            f"""• Influencer: {influencer}
            • Brand: {brand}
            • Fee: ~${fee:,}
            • Deliverables: {', '.join(deliverables)}
            • Exclusivity: {exclusivity_period} weeks
            • Engagement: {engagement_term} months
            • Usage: {usage_term} months
            • Start: {start_date.strftime('%B %Y')}""",
            
            # Informal notes
            f"""{influencer} for {brand} - ${fee:,} budget
            Need: {', '.join(deliverables)}
            {exclusivity_period}w exclusivity, {engagement_term}m engagement
            Usage rights: {usage_term} months
            Start: {start_date.strftime('%B %Y')}"""
        ]
        
        # Add complexity variations
        if complexity == "simple":
            return random.choice(text_variations)
        elif complexity == "medium":
            # Add some noise and incomplete information
            base_text = random.choice(text_variations)
            if random.random() < 0.3:
                base_text += f"\n\nAdditional notes: TBC on exact dates, around ${fee:,} budget"
            if random.random() < 0.2:
                base_text += f"\n\nNeed to confirm: {random.choice(['usage rights', 'exclusivity period', 'deliverables'])}"
            return base_text
        else:  # complex
            # Add typos, abbreviations, and mixed formatting
            base_text = random.choice(text_variations)
            base_text = self._add_complexity_noise(base_text)
            return base_text
    
    def _add_complexity_noise(self, text: str) -> str:
        """Add realistic noise to make text more complex"""
        # Add typos
        typos = {
            "around": "arnd",
            "approximately": "approx",
            "deliverables": "delivs",
            "exclusivity": "exclus",
            "engagement": "engmt",
            "campaign": "camp",
            "budget": "budg"
        }
        
        for correct, typo in typos.items():
            if random.random() < 0.1:
                text = text.replace(correct, typo)
        
        # Add Australian colloquialisms
        aussie_terms = [
            " reckon", " mate", " no worries", " fair dinkum", " ripper"
        ]
        
        if random.random() < 0.2:
            text += random.choice(aussie_terms)
        
        # Add incomplete sentences
        if random.random() < 0.15:
            text += "\n\nNeed to sort out: TBC on final dates"
        
        return text
    
    def _get_competitor_category(self, brand: str) -> str:
        """Get competitor category based on brand"""
        grocery_brands = ["Woolworths", "Coles", "Aldi", "IGA"]
        tech_brands = ["JB Hi-Fi", "Harvey Norman", "Officeworks"]
        fashion_brands = ["David Jones", "Myer", "Cotton On", "Country Road"]
        
        if brand in grocery_brands:
            return "grocery"
        elif brand in tech_brands:
            return "electronics"
        elif brand in fashion_brands:
            return "fashion"
        else:
            return "similar"
    
    def _extract_fields(self, brand: str, influencer: str, fee: int,
                       deliverables: List[str], start_date: datetime,
                       exclusivity_period: int, engagement_term: int,
                       usage_term: int) -> Dict:
        """Simulate field extraction from raw text"""
        
        return {
            "influencer": influencer,
            "client": brand,
            "brand": brand,
            "campaign": f"{brand} Campaign",
            "fee": f"${fee:,}",
            "fee_numeric": fee,
            "deliverables": deliverables,
            "exclusivity_period": f"{exclusivity_period} weeks",
            "exclusivity_scope": self._get_competitor_category(brand),
            "engagement_term": f"{engagement_term} months",
            "usage_term": f"{usage_term} months",
            "territory": "Australia",
            "start_date": start_date.strftime("%B %Y")
        }
    
    def _determine_template_match(self, complexity: str, deliverables: List[str]) -> str:
        """Determine which template style this should match"""
        if complexity == "simple":
            return random.choice(["template_style_1", "template_style_2"])
        elif complexity == "medium":
            return random.choice(["template_style_2", "template_style_3", "template_style_4"])
        else:
            return random.choice(["template_style_4", "template_style_5"])
    
    def _calculate_confidence(self, complexity: str, raw_text: str) -> float:
        """Calculate confidence score for extraction"""
        base_confidence = {
            "simple": 0.9,
            "medium": 0.75,
            "complex": 0.6
        }
        
        # Adjust based on text characteristics
        confidence = base_confidence[complexity]
        
        # Reduce confidence for noisy text
        if len(re.findall(r'\b(TBC|approx|arnd|delivs)\b', raw_text)) > 0:
            confidence -= 0.1
        
        # Reduce confidence for incomplete information
        if "TBC" in raw_text or "around" in raw_text.lower():
            confidence -= 0.05
        
        return max(0.3, min(0.95, confidence))
    
    def generate_dataset(self, num_samples: int = 1000, 
                        complexity_distribution: Dict[str, float] = None) -> List[Dict]:
        """Generate a complete synthetic dataset with OOD contamination"""
        
        if self.use_ood_contamination:
            return self.generate_dataset_with_ood_contamination(num_samples, complexity_distribution)
        else:
            return self._generate_basic_dataset(num_samples, complexity_distribution)
    
    def _generate_basic_dataset(self, num_samples: int = 1000, 
                              complexity_distribution: Dict[str, float] = None) -> List[Dict]:
        """Generate basic dataset without OOD contamination (original method)"""
        
        if complexity_distribution is None:
            complexity_distribution = {
                "simple": 0.3,
                "medium": 0.5,
                "complex": 0.2
            }
        
        dataset = []
        
        for i in range(num_samples):
            # Select complexity based on distribution
            complexity = random.choices(
                list(complexity_distribution.keys()),
                weights=list(complexity_distribution.values())
            )[0]
            
            agreement = self.generate_synthetic_agreement(complexity)
            agreement["classification"] = "INFLUENCER_AGREEMENT"
            agreement["confidence_target"] = 0.85
            agreement["metadata"]["sample_type"] = "positive"
            
            dataset.append(agreement)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        return dataset
    
    def _print_quality_report(self):
        """Print quality metrics report"""
        
        metrics = self.quality_metrics
        
        print("\n📊 Text Preprocessing Quality Report:")
        print("=" * 50)
        print(f"Total generated: {metrics['total_generated']}")
        print(f"Passed validation: {metrics['passed_validation']}")
        print(f"Validation rate: {metrics['passed_validation']/metrics['total_generated']:.2%}")
        
        if metrics['token_length_stats']['min'] > 0:
            print(f"\nToken Length Statistics:")
            print(f"  Min: {metrics['token_length_stats']['min']}")
            print(f"  Max: {metrics['token_length_stats']['max']}")
            print(f"  Avg: {metrics['token_length_stats']['avg']:.1f}")
        
        if metrics['quality_score_stats']['min'] > 0:
            print(f"\nQuality Score Statistics:")
            print(f"  Min: {metrics['quality_score_stats']['min']:.3f}")
            print(f"  Max: {metrics['quality_score_stats']['max']:.3f}")
            print(f"  Avg: {metrics['quality_score_stats']['avg']:.3f}")
        
        if metrics['chunking_stats']['total_chunked'] > 0:
            print(f"\nChunking Statistics:")
            print(f"  Total chunked: {metrics['chunking_stats']['total_chunked']}")
            print(f"  Average chunks: {metrics['chunking_stats']['avg_chunks']:.1f}")
        
        # Print OOD statistics
        if self.use_ood_contamination:
            ood_stats = metrics['ood_stats']
            print(f"\nOOD Contamination Statistics:")
            print(f"  Positive samples: {ood_stats['positive_samples']}")
            print(f"  OOD negative samples: {ood_stats['ood_negative_samples']}")
            print(f"  Edge case samples: {ood_stats['edge_case_samples']}")
            print(f"  Contamination ratio: {ood_stats['contamination_ratio']:.1%}")
    
    def save_dataset(self, dataset: List[Dict], filename: str = "synthetic_influencer_agreements.json"):
        """Save dataset to JSON file"""
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_samples": len(dataset),
                "complexity_distribution": self._get_complexity_distribution(dataset),
                "fee_statistics": self._get_fee_statistics(dataset),
                "template_distribution": self._get_template_distribution(dataset),
                "semantic_smoothing_enabled": self.use_semantic_smoothing,
                "text_preprocessing_enabled": self.use_text_preprocessing,
                "ood_contamination_enabled": self.use_ood_contamination,
                "contamination_ratio": self.contamination_ratio,
                "quality_metrics": self.quality_metrics
            },
            "dataset": dataset
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset saved to {filename}")
        print(f"📊 Generated {len(dataset)} synthetic agreements")
    
    def _get_complexity_distribution(self, dataset: List[Dict]) -> Dict[str, int]:
        """Get distribution of complexity levels"""
        distribution = {}
        for item in dataset:
            complexity = item["complexity_level"]
            distribution[complexity] = distribution.get(complexity, 0) + 1
        return distribution
    
    def _get_fee_statistics(self, dataset: List[Dict]) -> Dict:
        """Get fee statistics"""
        fees = [item["extracted_fields"]["fee_numeric"] for item in dataset if "fee_numeric" in item["extracted_fields"]]
        if fees:
            return {
                "min": min(fees),
                "max": max(fees),
                "mean": sum(fees) / len(fees),
                "median": sorted(fees)[len(fees) // 2]
            }
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    
    def _get_template_distribution(self, dataset: List[Dict]) -> Dict[str, int]:
        """Get distribution of template matches"""
        distribution = {}
        for item in dataset:
            template = item["template_match"]
            distribution[template] = distribution.get(template, 0) + 1
        return distribution

def main():
    """Generate synthetic dataset"""
    print("🎯 Synthetic Influencer Agreement Dataset Generator")
    print("=" * 60)
    
    # Initialize generator with all features enabled
    generator = SyntheticDatasetGenerator(
        use_semantic_smoothing=True,
        use_text_preprocessing=True,
        use_ood_contamination=True,
        contamination_ratio=0.2
    )
    
    # Generate dataset with specified distribution
    complexity_distribution = {
        "simple": 0.25,    # 25% simple cases
        "medium": 0.55,    # 55% medium complexity
        "complex": 0.20    # 20% complex cases
    }
    
    print("🔄 Generating synthetic dataset with semantic smoothing, text preprocessing, and OOD contamination...")
    dataset = generator.generate_dataset(
        num_samples=1000,
        complexity_distribution=complexity_distribution
    )
    
    # Save the raw dataset
    output_file = "synthetic_influencer_agreements.json"
    generator.save_dataset(dataset, output_file)
    
    print(f"✅ Generated {len(dataset)} synthetic agreements with OOD contamination")
    
    # Print summary statistics
    print("\n📈 Dataset Summary:")
    print(f"Total samples: {len(dataset)}")
    
    complexity_dist = generator._get_complexity_distribution(dataset)
    print(f"Complexity distribution: {complexity_dist}")
    
    fee_stats = generator._get_fee_statistics(dataset)
    print(f"Fee range: ${fee_stats['min']:,} - ${fee_stats['max']:,}")
    print(f"Average fee: ${fee_stats['mean']:,.0f}")
    
    template_dist = generator._get_template_distribution(dataset)
    print(f"Template distribution: {template_dist}")
    
    # Print classification distribution
    classification_dist = {}
    for item in dataset:
        classification = item.get("classification", "UNKNOWN")
        classification_dist[classification] = classification_dist.get(classification, 0) + 1
    print(f"Classification distribution: {classification_dist}")
    
    print(f"\n✅ Dataset ready for training! Saved to: {output_file}")

if __name__ == "__main__":
    main() 