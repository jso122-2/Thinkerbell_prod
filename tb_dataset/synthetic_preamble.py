"""
Synthetic Business Preamble Generator for tb_dataset

Generates realistic business context preambles for training data.
Creates synthetic influencer agreement preambles that can be processed
through the tb_dataset pipeline (clean, chunk, emit JSONL).
"""

import random
import uuid
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .schema import DocumentSample
from .clean import TextCleaner
from .chunk import SmartChunker
from .utils import hash_id


class SyntheticPreambleGenerator:
    """
    Generates synthetic business preambles for influencer agreements.
    
    Creates realistic business context that can be processed through
    the tb_dataset pipeline alongside real documents.
    """
    
    def __init__(self):
        """Initialize parameter pools for preamble generation"""
        
        # Australian agency names
        self.agency_names = [
            "Thinkerbell", "Brandlink", "Stellar PR", "The Projects", "Amplify Agency",
            "Publicis Australia", "Ogilvy Sydney", "Leo Burnett Melbourne", "Saatchi & Saatchi",
            "DDB Sydney", "Clemenger BBDO", "M&C Saatchi Australia", "Havas Group", 
            "Wunderman Thompson", "GroupM Australia", "Mindshare", "Dentsu", "Initiative"
        ]
        
        # Australian talent names - diverse and realistic
        self.talent_names = [
            "Sarah Chen", "Michael Brown", "Leah Thompson", "Josh Nguyen", "Chloe Lewis",
            "Emma Wilson", "James Taylor", "Sophie Anderson", "Dylan Garcia", "Zoe Martin",
            "Oliver Davis", "Isabella Clark", "Mason Rodriguez", "Ava White", "Lucas Hall",
            "Mia Johnson", "Ethan Walker", "Grace Kim", "Noah Campbell", "Lily Zhang",
            "Jake O'Sullivan", "Ruby Patel", "Finn McCarthy", "Aisha Williams", "Liam Roberts"
        ]
        
        # Australian brands by industry
        self.brands_by_industry = {
            "fashion": [
                "Cotton On", "Country Road", "David Jones", "Myer", "Witchery", 
                "Portmans", "Sportsgirl", "Target Australia", "Big W", "Kmart Australia"
            ],
            "food": [
                "Woolworths", "Coles", "IGA", "Boost Juice", "Guzman y Gomez", 
                "Mad Mex", "Zambrero", "Roll'd", "Grill'd", "Betty's Burgers",
                "Red Rooster", "Nando's Australia", "Menulog", "Uber Eats"
            ],
            "tech": [
                "JB Hi-Fi", "Harvey Norman", "Officeworks", "Telstra", "Optus",
                "Commonwealth Bank", "ANZ", "Westpac", "NAB", "Canva"
            ],
            "home": [
                "Bunnings", "IKEA Australia", "Freedom", "Adairs", "Bed Bath N' Table",
                "Pillow Talk", "Amart Furniture", "Fantastic Furniture", "Koala"
            ],
            "beauty": [
                "Chemist Warehouse", "Priceline", "Sephora Australia", "Mecca", 
                "Lush Australia", "The Body Shop", "Aesop", "Frank Body"
            ],
            "travel": [
                "Qantas", "Jetstar", "Virgin Australia", "Flight Centre", "Webjet",
                "Tourism Australia", "Airbnb Australia", "Booking.com"
            ]
        }
        
        # Campaign context hooks
        self.context_hooks = {
            "seasonal": [
                "With summer approaching, Australians are embracing {theme}",
                "As autumn begins, consumers are focusing on {theme}",
                "With winter arriving, families are prioritizing {theme}",
                "As spring emerges, people are exploring {theme}",
                "During the holiday season, Australians are seeking {theme}"
            ],
            "trend_based": [
                "With the rise of {trend} across Australia, {theme}",
                "As {trend} gains momentum, consumers are {theme}",
                "Following the growing popularity of {trend}, people are {theme}",
                "With more Australians embracing {trend}, families are {theme}"
            ],
            "market_response": [
                "In response to changing consumer preferences, {theme}",
                "Following market research insights, Australians are {theme}",
                "As industry leaders recognize consumer needs, people are {theme}",
                "With evolving market demands, consumers are {theme}"
            ]
        }
        
        # Industry-specific themes and trends
        self.industry_context = {
            "fashion": {
                "themes": [
                    "seeking sustainable and ethically-made clothing options",
                    "prioritizing comfort and versatility in their wardrobes",
                    "looking for quality pieces that offer lasting value",
                    "embracing locally-designed and manufactured fashion"
                ],
                "trends": [
                    "sustainable fashion", "circular economy practices", "local manufacturing",
                    "slow fashion movement", "size-inclusive design", "gender-neutral clothing"
                ]
            },
            "food": {
                "themes": [
                    "choosing fresh, locally-sourced ingredients for family meals",
                    "prioritizing convenience without compromising on nutrition",
                    "exploring plant-based and health-conscious eating options",
                    "supporting local producers and sustainable food practices"
                ],
                "trends": [
                    "plant-based eating", "meal delivery services", "sustainable packaging",
                    "local food sourcing", "health-conscious choices", "zero-waste cooking"
                ]
            },
            "tech": {
                "themes": [
                    "adopting innovative solutions that simplify daily routines",
                    "prioritizing digital security and privacy protection",
                    "seeking reliable connectivity for work and entertainment",
                    "embracing smart home technology for enhanced living"
                ],
                "trends": [
                    "smart home integration", "cybersecurity awareness", "remote work solutions",
                    "sustainable technology", "digital wellness", "AI-powered services"
                ]
            },
            "home": {
                "themes": [
                    "creating functional and comfortable living environments",
                    "investing in energy-efficient home improvements",
                    "designing spaces that support work-from-home lifestyles",
                    "prioritizing sustainable and eco-friendly home solutions"
                ],
                "trends": [
                    "sustainable home design", "energy-efficient appliances", "multi-purpose spaces",
                    "indoor air quality", "smart home automation", "biophilic design"
                ]
            },
            "beauty": {
                "themes": [
                    "embracing natural and clean beauty routines",
                    "prioritizing skincare wellness over complex makeup",
                    "seeking inclusive products that celebrate diversity",
                    "choosing sustainable and ethically-sourced beauty items"
                ],
                "trends": [
                    "clean beauty movement", "minimalist skincare", "inclusive beauty standards",
                    "sustainable packaging", "personalized beauty", "wellness-focused routines"
                ]
            },
            "travel": {
                "themes": [
                    "planning sustainable and meaningful travel experiences",
                    "prioritizing domestic destinations and local tourism",
                    "seeking authentic cultural experiences and connections",
                    "balancing adventure with responsible travel practices"
                ],
                "trends": [
                    "sustainable tourism", "local travel experiences", "eco-friendly accommodations",
                    "digital nomad lifestyle", "experiential travel", "wellness tourism"
                ]
            }
        }
        
        # Campaign types and role definitions
        self.campaign_types = [
            "Brand Partnership", "Product Launch", "Seasonal Campaign", "Awareness Initiative",
            "Collection Showcase", "Community Engagement", "Sustainability Campaign", 
            "Digital Experience", "Cultural Celebration", "Innovation Series"
        ]
        
        self.role_types = {
            "simple": ["brand ambassador", "spokesperson"],
            "medium": ["campaign ambassador", "brand partner", "content creator", "lifestyle advocate"],
            "complex": ["brand ambassador", "campaign spokesperson", "strategic partner", 
                      "community advocate", "innovation champion", "cultural ambassador"]
        }
        
        # Value proposition templates
        self.value_propositions = [
            "reinforcing {brand}'s commitment to {value}",
            "highlighting {brand}'s leadership in {category}",
            "showcasing {brand}'s dedication to {cause}",
            "demonstrating {brand}'s understanding of {audience}",
            "amplifying {brand}'s mission to {purpose}",
            "supporting {brand}'s vision for {future}"
        ]
    
    def generate_preamble(self, 
                         brand: Optional[str] = None, 
                         industry: Optional[str] = None,
                         complexity: str = "medium") -> str:
        """
        Generate a single business preamble.
        
        Args:
            brand: Specific brand name (random if None)
            industry: Industry category (random if None) 
            complexity: Preamble complexity (simple, medium, complex)
            
        Returns:
            Generated preamble text
        """
        
        # Select parameters
        if industry is None:
            industry = random.choice(list(self.brands_by_industry.keys()))
        
        if brand is None:
            brand = random.choice(self.brands_by_industry[industry])
        
        agency = random.choice(self.agency_names)
        talent = random.choice(self.talent_names)
        campaign_type = random.choice(self.campaign_types)
        role = random.choice(self.role_types.get(complexity, self.role_types["medium"]))
        
        # Generate campaign name
        campaign_name = self._generate_campaign_name(brand, campaign_type)
        
        # Generate contextual hook
        context_sentence = self._generate_context_hook(industry)
        
        # Generate role description
        role_description = self._generate_role_description(talent, role, complexity)
        
        # Generate value proposition
        value_prop = self._generate_value_proposition(brand, industry)
        
        # Assemble preamble
        preamble = (
            f"{agency} has engaged {talent} to provide services for the "
            f"{campaign_name} on behalf of {brand}. {context_sentence} "
            f"{role_description} {value_prop}"
        )
        
        return preamble
    
    def _generate_campaign_name(self, brand: str, campaign_type: str) -> str:
        """Generate a realistic campaign name"""
        
        modifiers = ["2024", "2025", "Spring", "Summer", "Autumn", "Winter", 
                    "New", "Future", "Next", "Fresh", "Premier", "Elite"]
        
        templates = [
            f"{brand} {campaign_type}",
            f"{random.choice(modifiers)} {campaign_type}",
            f"{brand} {random.choice(modifiers)} {campaign_type}",
            f"The {campaign_type}"
        ]
        
        return random.choice(templates)
    
    def _generate_context_hook(self, industry: str) -> str:
        """Generate industry-appropriate context hook"""
        
        industry_data = self.industry_context.get(industry, self.industry_context["fashion"])
        
        # Choose hook type
        hook_type = random.choice(list(self.context_hooks.keys()))
        hook_template = random.choice(self.context_hooks[hook_type])
        
        # Fill in context
        if hook_type == "trend_based":
            trend = random.choice(industry_data["trends"])
            theme = random.choice(industry_data["themes"])
            return hook_template.format(trend=trend, theme=theme)
        else:
            theme = random.choice(industry_data["themes"])
            return hook_template.format(theme=theme)
    
    def _generate_role_description(self, talent: str, role: str, complexity: str) -> str:
        """Generate role description based on complexity"""
        
        activities = {
            "simple": [
                "deliver authentic messaging across key social media platforms"
            ],
            "medium": [
                "create engaging content across digital and social media channels",
                "deliver key brand messages through various content formats",
                "engage authentically with target audiences across multiple platforms"
            ],
            "complex": [
                "lead comprehensive brand storytelling across all digital touchpoints",
                "develop authentic content strategies spanning multiple media formats",
                "serve as the primary brand voice across integrated marketing channels",
                "create meaningful connections with diverse audience segments"
            ]
        }
        
        activity = random.choice(activities.get(complexity, activities["medium"]))
        return f"As the campaign's {role}, {talent} will {activity}."
    
    def _generate_value_proposition(self, brand: str, industry: str) -> str:
        """Generate brand value proposition"""
        
        template = random.choice(self.value_propositions)
        
        value_contexts = {
            "fashion": {
                "value": "sustainable and ethical fashion practices",
                "category": "conscious fashion retail", 
                "cause": "environmental responsibility",
                "audience": "style-conscious Australian consumers",
                "purpose": "make sustainable fashion accessible",
                "future": "ethical fashion in Australia"
            },
            "food": {
                "value": "fresh, quality ingredients and nutrition",
                "category": "healthy food retail",
                "cause": "community wellness",
                "audience": "health-conscious Australian families", 
                "purpose": "support healthy eating habits",
                "future": "sustainable food systems"
            },
            "tech": {
                "value": "innovation and digital accessibility",
                "category": "consumer technology",
                "cause": "digital inclusion", 
                "audience": "technology-minded Australians",
                "purpose": "simplify digital experiences",
                "future": "connected communities"
            },
            "home": {
                "value": "comfortable and sustainable living",
                "category": "home improvement and design",
                "cause": "environmental stewardship",
                "audience": "Australian homeowners and renters",
                "purpose": "create better living spaces", 
                "future": "sustainable home environments"
            },
            "beauty": {
                "value": "authentic self-expression and wellness",
                "category": "inclusive beauty",
                "cause": "self-acceptance and diversity",
                "audience": "beauty enthusiasts across Australia",
                "purpose": "celebrate individual beauty",
                "future": "inclusive beauty standards"
            },
            "travel": {
                "value": "meaningful and sustainable travel experiences",
                "category": "responsible tourism",
                "cause": "cultural preservation",
                "audience": "adventure-seeking Australians", 
                "purpose": "create lasting travel memories",
                "future": "sustainable tourism"
            }
        }
        
        context = value_contexts.get(industry, value_contexts["fashion"])
        
        # Fill template with appropriate context
        filled_template = template.format(brand=brand, **context)
        return f"This campaign is {filled_template}."
    
    def generate_synthetic_documents(self, 
                                   count: int = 100,
                                   complexity_distribution: Optional[Dict[str, float]] = None,
                                   industries: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate a batch of synthetic preamble documents.
        
        Args:
            count: Number of preambles to generate
            complexity_distribution: Distribution of complexity levels
            industries: Industries to include (all if None)
            
        Returns:
            List of synthetic document dictionaries
        """
        
        if complexity_distribution is None:
            complexity_distribution = {"simple": 0.3, "medium": 0.5, "complex": 0.2}
        
        if industries is None:
            industries = list(self.brands_by_industry.keys())
        
        documents = []
        
        for i in range(count):
            # Select parameters
            industry = random.choice(industries)
            brand = random.choice(self.brands_by_industry[industry])
            
            # Select complexity based on distribution
            complexity = random.choices(
                list(complexity_distribution.keys()),
                weights=list(complexity_distribution.values()),
                k=1
            )[0]
            
            # Generate preamble
            preamble_text = self.generate_preamble(brand, industry, complexity)
            
            # Create document metadata
            doc_id = f"synthetic_preamble_{i+1:04d}"
            
            documents.append({
                "doc_id": doc_id,
                "source_type": "synthetic",
                "text": preamble_text,
                "brand": brand,
                "industry": industry,
                "complexity": complexity,
                "char_count": len(preamble_text),
                "word_count": len(preamble_text.split()),
                "generated_at": "2024-01-01T00:00:00Z"  # Would use actual timestamp
            })
        
        return documents
    
    def create_document_samples(self, 
                              synthetic_docs: List[Dict],
                              output_dir: Path,
                              clean: bool = True,
                              chunk: bool = True) -> List[DocumentSample]:
        """
        Convert synthetic documents to DocumentSample objects and process them.
        
        Args:
            synthetic_docs: List of synthetic document dictionaries
            output_dir: Output directory for processing
            clean: Whether to apply text cleaning
            chunk: Whether to apply chunking
            
        Returns:
            List of processed DocumentSample objects
        """
        
        # Initialize processors
        cleaner = TextCleaner() if clean else None
        chunker = SmartChunker() if chunk else None
        
        samples = []
        
        for doc in synthetic_docs:
            # Get text
            text = doc["text"]
            
            # Apply cleaning if requested
            if cleaner:
                text = cleaner.clean_text(text)
            
            # Apply chunking if requested  
            chunks = []
            needs_chunking = False
            
            if chunker:
                needs_chunking = chunker.needs_chunking(text)
                if needs_chunking:
                    chunks = chunker.chunk_text(text)
                else:
                    chunks = [text]
            else:
                chunks = [text]
            
            # Create DocumentSample
            sample = DocumentSample(
                source_path=f"synthetic://{doc['doc_id']}.txt",
                doc_id=doc["doc_id"],
                section="full",
                text=text,
                char_len=len(text),
                tok_len=len(text.split()),  # Simple word-based token count
                chunks=chunks,
                needs_chunking=needs_chunking,
                template_hint=f"business_preamble_{doc['industry']}"
            )
            
            samples.append(sample)
        
        return samples
    
    def export_to_jsonl(self, 
                       samples: List[DocumentSample], 
                       output_file: Path) -> None:
        """
        Export DocumentSample objects to JSONL format.
        
        Args:
            samples: List of DocumentSample objects
            output_file: Output JSONL file path
        """
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample.to_json() + '\n')


def main():
    """Standalone test and demo of synthetic preamble generator"""
    
    print("🎯 Synthetic Preamble Generator for tb_dataset")
    print("=" * 60)
    
    # Initialize generator
    generator = SyntheticPreambleGenerator()
    
    # Test 1: Generate individual preambles
    print("\n📝 Test 1: Individual Preamble Generation")
    for i in range(3):
        industry = random.choice(["fashion", "food", "tech"])
        complexity = random.choice(["simple", "medium", "complex"])
        
        preamble = generator.generate_preamble(industry=industry, complexity=complexity)
        print(f"\n{i+1}. {complexity.upper()} - {industry}:")
        print(f"   {preamble}")
    
    # Test 2: Generate batch of synthetic documents
    print(f"\n\n📊 Test 2: Batch Synthetic Document Generation")
    
    synthetic_docs = generator.generate_synthetic_documents(
        count=5,
        complexity_distribution={"simple": 0.4, "medium": 0.4, "complex": 0.2}
    )
    
    print(f"Generated {len(synthetic_docs)} synthetic documents:")
    for doc in synthetic_docs:
        print(f"  - {doc['doc_id']}: {doc['brand']} ({doc['industry']}, {doc['complexity']})")
        print(f"    {doc['text'][:80]}...")
    
    # Test 3: Create DocumentSample objects
    print(f"\n\n📋 Test 3: DocumentSample Creation and Processing")
    
    output_dir = Path("test_output")
    samples = generator.create_document_samples(
        synthetic_docs=synthetic_docs,
        output_dir=output_dir,
        clean=True,
        chunk=True
    )
    
    print(f"Created {len(samples)} DocumentSample objects:")
    for sample in samples[:2]:  # Show first 2
        print(f"\n  Sample: {sample.doc_id}")
        print(f"    Text length: {sample.char_len} chars, {sample.tok_len} tokens")
        print(f"    Chunks: {len(sample.chunks)} ({'chunked' if sample.needs_chunking else 'single'})")
        print(f"    Template: {sample.template_hint}")
    
    # Test 4: Export to JSONL
    print(f"\n\n💾 Test 4: JSONL Export")
    
    output_file = output_dir / "synthetic_preambles.jsonl"
    generator.export_to_jsonl(samples, output_file)
    
    if output_file.exists():
        file_size = output_file.stat().st_size
        print(f"✅ Exported to {output_file} ({file_size} bytes)")
        
        # Show first few lines
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:2]
        
        print(f"First entry:")
        sample_json = json.loads(lines[0])
        print(f"  doc_id: {sample_json['doc_id']}")
        print(f"  chunks: {len(sample_json['chunks'])}")
        print(f"  template_hint: {sample_json['template_hint']}")
    
    print(f"\n🎉 All tests completed successfully!")
    print(f"💡 Use this generator to create training data for business document understanding.")


if __name__ == "__main__":
    main() 