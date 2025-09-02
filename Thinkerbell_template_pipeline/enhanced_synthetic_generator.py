#!/usr/bin/env python3
"""
Enhanced Synthetic Dataset Generator for Thinkerbell
Addresses memorization issues with diverse templates, paraphrasing, and similarity checking
"""

import json
import random
import uuid
import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import warnings

warnings.filterwarnings("ignore")

@dataclass
class TemplateVariation:
    """Represents a text template variation with specific style and structure"""
    style: str  # formal, casual, bullet, abbreviated, conversational, etc.
    template: str
    complexity_level: str
    structural_elements: List[str]

class EnhancedSyntheticGenerator:
    """Enhanced synthetic data generator with template diversity and similarity checking"""
    
    def __init__(self, use_paraphrasing: bool = True, similarity_threshold: float = 0.92):
        print("🚀 Initializing Enhanced Synthetic Data Generator")
        
        # Core parameters
        self.similarity_threshold = similarity_threshold
        self.use_paraphrasing = use_paraphrasing
        
        # Initialize similarity model
        print("📊 Loading similarity model...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize paraphrasing pipeline if available
        self.paraphraser = None
        if use_paraphrasing:
            try:
                print("🔄 Loading paraphrasing model...")
                # Using a lightweight paraphrasing model
                self.paraphraser = pipeline("text2text-generation", 
                                           model="tuner007/pegasus_paraphrase", 
                                           max_length=512)
                print("✅ Paraphrasing enabled")
            except Exception as e:
                print(f"⚠️ Paraphrasing model not available: {e}")
                self.use_paraphrasing = False
        
        # Generated samples cache for similarity checking
        self.generated_samples: List[str] = []
        self.sample_embeddings: List[np.ndarray] = []
        
        # Initialize base data
        self._init_base_data()
        self._init_template_variations()
        self._init_noise_patterns()
        
        print("✅ Enhanced generator initialized")
    
    def _init_base_data(self):
        """Initialize base data pools"""
        
        # Extended Australian brands with industry context
        self.australian_brands = {
            "fashion": [
                "Cotton On", "Country Road", "David Jones", "Myer", "Witchery", 
                "Portmans", "Sportsgirl", "Target Australia", "Kmart", "Big W",
                "Seed Heritage", "Forever New", "Bardot", "Glue Store", "Dangerfield"
            ],
            "food": [
                "Woolworths", "Coles", "IGA", "Queen Fine Foods", "Boost Juice", 
                "Guzman y Gomez", "Mad Mex", "Zambrero", "Roll'd", "Subway Australia", 
                "Hungry Jack's", "KFC Australia", "McDonald's Australia", "Domino's Australia", 
                "Pizza Hut Australia", "Red Rooster", "Nando's", "Grill'd", "Betty's Burgers",
                "Schnitz", "San Churro", "Chatime", "The Coffee Club"
            ],
            "tech": [
                "JB Hi-Fi", "Harvey Norman", "Officeworks", "Telstra", "Optus",
                "Commonwealth Bank", "Qantas", "Virgin Australia", "Foxtel", "TPG"
            ],
            "beauty": [
                "Chemist Warehouse", "Priceline", "Sephora Australia", "Mecca", "Lush Australia",
                "Napoleon Perdis", "Face Halo", "Frank Body", "Bondi Sands", "Designer Brands"
            ],
            "automotive": [
                "Supercheap Auto", "Autobarn", "Repco", "Pedders", "Bridgestone"
            ],
            "home": [
                "Bunnings", "IKEA Australia", "Freedom", "Adairs", "Bed Bath N' Table", 
                "Temple & Webster", "West Elm", "Mocka", "Koala"
            ]
        }
        
        # Expanded influencer names with variety
        self.influencer_names = [
            # Traditional Australian names
            "Sarah Chen", "James Wilson", "Emma Thompson", "Michael Rodriguez", "Olivia Davis",
            "David Brown", "Sophie Anderson", "Christopher Lee", "Isabella Garcia", "Daniel Martinez",
            
            # Modern names
            "Ava Johnson", "Matthew Taylor", "Mia White", "Andrew Clark", "Chloe Lewis",
            "Joshua Hall", "Lily Walker", "Ryan Allen", "Grace Young", "Nathan King",
            
            # Multicultural Australian names
            "Priya Sharma", "Hassan Ali", "Maya Nguyen", "Dimitri Kostas", "Aisha Mohamed",
            "Lucas Romano", "Sienna Chan", "Marcus Okafor", "Zara Patel", "Adrian Popescu",
            
            # Social media style names
            "Zoe.Wright", "Tyler_Scott", "Hannah.Green", "Kevin.Baker", "Natalie_Adams",
            "Brandon.Nelson", "Victoria.Carter", "Justin_Mitchell", "Aria.Perez", "Sean.Roberts"
        ]
        
        # Enhanced fee structure with more realistic ranges
        self.fee_tiers = {
            "nano": (500, 2000),     # Nano influencers
            "micro": (2000, 8000),   # Micro influencers  
            "mid": (8000, 25000),    # Mid-tier influencers
            "macro": (25000, 75000), # Macro influencers
            "mega": (75000, 200000)  # Celebrity tier
        }
        
        # Realistic deliverables with quantities
        self.deliverables = {
            "simple": [
                "Instagram post", "Instagram story", "Facebook post", "TikTok video",
                "LinkedIn post", "YouTube Short", "Twitter post", "Pinterest pin"
            ],
            "medium": [
                "Instagram carousel", "Instagram reel", "YouTube video", "TikTok series",
                "Blog post", "Product review", "Unboxing video", "Tutorial content",
                "Live stream", "IGTV episode", "Podcast mention", "Newsletter feature"
            ],
            "complex": [
                "Multi-platform campaign", "Brand ambassador program", "Event appearance",
                "Product collaboration", "Long-form YouTube series", "Webinar hosting",
                "Brand photoshoot", "TV commercial", "Radio interview", "Press conference",
                "Product line endorsement", "Store opening event", "Brand workshop"
            ]
        }
    
    def _init_template_variations(self):
        """Initialize 10+ diverse template variations for each complexity/style combination"""
        
        self.template_variations = {
            "simple": [
                # Formal business style
                TemplateVariation(
                    style="formal_brief",
                    template="Partnership opportunity with {influencer} for {brand}. Proposed budget: ${fee:,}. Required deliverables: {deliverables}. Campaign duration: {engagement_term} months. Exclusivity period: {exclusivity_period} weeks. Territory: Australia. Start date: {start_date}.",
                    complexity_level="simple",
                    structural_elements=["partnership", "budget", "deliverables", "duration", "exclusivity"]
                ),
                
                # Casual email style
                TemplateVariation(
                    style="casual_email",
                    template="Hi! We'd love to work with {influencer} on a {brand} campaign. Budget is around ${fee:,} and we need {deliverables}. Looking at {engagement_term} months engagement with {exclusivity_period} weeks exclusivity. Can start {start_date}. Let me know your thoughts!",
                    complexity_level="simple",
                    structural_elements=["greeting", "collaboration", "budget", "timeline", "call_to_action"]
                ),
                
                # Bullet point style
                TemplateVariation(
                    style="bullet_points",
                    template="• Influencer: {influencer}\n• Brand: {brand}\n• Budget: ${fee:,}\n• Content: {deliverables}\n• Duration: {engagement_term} months\n• Exclusivity: {exclusivity_period} weeks\n• Launch: {start_date}",
                    complexity_level="simple",
                    structural_elements=["bullet_format", "concise", "structured"]
                ),
                
                # SMS/brief style
                TemplateVariation(
                    style="brief_sms",
                    template="{brand} collab with {influencer} - ${fee:,} budget, need {deliverables}, {engagement_term}m term, {exclusivity_period}w excl, starts {start_date}",
                    complexity_level="simple",
                    structural_elements=["abbreviated", "concise", "mobile_friendly"]
                ),
                
                # Proposal style
                TemplateVariation(
                    style="proposal_format",
                    template="We propose engaging {influencer} as brand partner for {brand}. Investment: ${fee:,}. Scope includes {deliverables} over {engagement_term} months. Exclusivity arrangement for {exclusivity_period} weeks. Commencement: {start_date}.",
                    complexity_level="simple",
                    structural_elements=["proposal", "investment", "scope", "arrangement"]
                ),
                
                # Question format
                TemplateVariation(
                    style="inquiry_format",
                    template="Would {influencer} be interested in partnering with {brand}? We have ${fee:,} allocated for {deliverables}. The campaign runs for {engagement_term} months with {exclusivity_period} weeks exclusivity. Timeline starts {start_date}.",
                    complexity_level="simple",
                    structural_elements=["question", "interest", "allocation", "timeline"]
                ),
                
                # Action-oriented style
                TemplateVariation(
                    style="action_oriented",
                    template="Seeking {influencer} for {brand} partnership. Budget: ${fee:,}. Deliverables: {deliverables}. Commitment: {engagement_term} months. Exclusivity: {exclusivity_period} weeks. Start: {start_date}. Next steps?",
                    complexity_level="simple",
                    structural_elements=["seeking", "commitment", "next_steps"]
                ),
                
                # Contract style
                TemplateVariation(
                    style="contract_summary",
                    template="Agreement summary: {influencer} x {brand}. Fee: ${fee:,}. Content requirements: {deliverables}. Term: {engagement_term} months. Non-compete: {exclusivity_period} weeks. Effective: {start_date}.",
                    complexity_level="simple",
                    structural_elements=["agreement", "requirements", "non_compete", "effective"]
                ),
                
                # Opportunity style
                TemplateVariation(
                    style="opportunity_pitch",
                    template="Exciting opportunity for {influencer}! {brand} campaign with ${fee:,} compensation. Creating {deliverables} content over {engagement_term} months. {exclusivity_period} week exclusivity required. Ready to start {start_date}!",
                    complexity_level="simple",
                    structural_elements=["excitement", "opportunity", "compensation", "ready"]
                ),
                
                # Collaboration style
                TemplateVariation(
                    style="collaboration_invite",
                    template="{brand} invites {influencer} to collaborate. Compensation package: ${fee:,}. Content creation: {deliverables}. Partnership duration: {engagement_term} months. Exclusivity commitment: {exclusivity_period} weeks. Launch date: {start_date}.",
                    complexity_level="simple",
                    structural_elements=["invitation", "package", "creation", "commitment", "launch"]
                )
            ],
            
            "medium": [
                # Detailed business proposal
                TemplateVariation(
                    style="detailed_proposal",
                    template="We're excited to propose a partnership between {influencer} and {brand}. Budget allocation: ${fee:,}. Content requirements include {deliverables} with specific brand guidelines. Campaign runs {engagement_term} months with {exclusivity_period} weeks exclusivity from competitors. Start date: {start_date}. Additional terms: performance metrics tracking, content approval process, and usage rights for {usage_term} months.",
                    complexity_level="medium",
                    structural_elements=["excitement", "allocation", "guidelines", "metrics", "approval", "usage_rights"]
                ),
                
                # Marketing campaign brief
                TemplateVariation(
                    style="campaign_brief",
                    template="Campaign Brief: {influencer} x {brand}\nObjective: Brand awareness and engagement\nBudget: ${fee:,}\nDeliverables: {deliverables}\nDuration: {engagement_term} months\nExclusivity: {exclusivity_period} weeks\nLaunch: {start_date}\nTargets: Australian market, 18-45 demographics\nKPIs: Reach, engagement, conversion tracking",
                    complexity_level="medium",
                    structural_elements=["objective", "demographics", "kpis", "targets", "tracking"]
                ),
                
                # Conversational pitch
                TemplateVariation(
                    style="conversational_pitch",
                    template="Hey {influencer}! Hope you're doing well. We've been following your content and think you'd be perfect for our {brand} campaign. We're looking at ${fee:,} for the partnership. The scope includes {deliverables} over {engagement_term} months. We'd need {exclusivity_period} weeks exclusivity, especially around competing brands. Timing-wise, we're aiming to kick off {start_date}. The campaign aligns well with your audience, and we're excited about the potential collaboration. What are your thoughts?",
                    complexity_level="medium",
                    structural_elements=["greeting", "following", "perfect", "scope", "competing", "aligns", "thoughts"]
                ),
                
                # Professional inquiry
                TemplateVariation(
                    style="professional_inquiry",
                    template="Good morning {influencer},\n\nWe represent {brand} and are interested in discussing a potential collaboration. Our proposed investment is ${fee:,} for content creation including {deliverables}. The engagement period would span {engagement_term} months with a {exclusivity_period} week non-compete clause. We're targeting a {start_date} launch. This partnership offers significant exposure opportunities and aligns with current market trends. We'd appreciate the opportunity to discuss further.",
                    complexity_level="medium",
                    structural_elements=["formal_greeting", "represent", "investment", "non_compete", "exposure", "trends", "appreciate"]
                ),
                
                # Campaign strategy format
                TemplateVariation(
                    style="strategy_format",
                    template="Influencer Marketing Strategy\nPartner: {influencer}\nBrand: {brand}\nInvestment: ${fee:,}\nContent Strategy: {deliverables}\nCampaign Timeline: {engagement_term} months\nCompetitor Exclusivity: {exclusivity_period} weeks\nGo-Live Date: {start_date}\nSuccess Metrics: Engagement rate, reach, brand sentiment\nContent Themes: Authentic integration, lifestyle alignment",
                    complexity_level="medium",
                    structural_elements=["strategy", "investment", "timeline", "metrics", "themes", "integration", "alignment"]
                ),
                
                # Partnership proposal
                TemplateVariation(
                    style="partnership_proposal",
                    template="Partnership Proposal for {influencer}\n\n{brand} seeks to establish a meaningful collaboration with compensation of ${fee:,}. The partnership encompasses {deliverables} creation across {engagement_term} months. Exclusivity requirements include {exclusivity_period} weeks restriction from similar brand partnerships. Campaign launch scheduled for {start_date}. This collaboration represents mutual value creation with substantial audience reach and brand alignment opportunities.",
                    complexity_level="medium",
                    structural_elements=["meaningful", "encompasses", "restriction", "mutual", "substantial", "alignment"]
                ),
                
                # Creative brief style
                TemplateVariation(
                    style="creative_brief",
                    template="Creative Brief: {brand} x {influencer}\nBrief Overview: Authentic brand partnership\nBudget: ${fee:,}\nCreative Requirements: {deliverables}\nCampaign Duration: {engagement_term} months\nBrand Protection: {exclusivity_period} weeks exclusivity\nProduction Start: {start_date}\nCreative Direction: Natural integration, storytelling focus\nAudience: Brand-aligned demographics\nMeasurement: Engagement, reach, sentiment analysis",
                    complexity_level="medium",
                    structural_elements=["authentic", "protection", "storytelling", "measurement", "sentiment"]
                ),
                
                # Business development format
                TemplateVariation(
                    style="business_development",
                    template="Business Development Opportunity\n\nInfluencer: {influencer}\nBrand Partner: {brand}\nCompensation Package: ${fee:,}\nContent Deliverables: {deliverables}\nEngagement Period: {engagement_term} months\nExclusivity Terms: {exclusivity_period} weeks\nProject Initiation: {start_date}\n\nThis partnership offers strategic value through audience alignment, content quality, and market positioning. Long-term relationship potential with performance-based extensions.",
                    complexity_level="medium",
                    structural_elements=["compensation_package", "strategic", "positioning", "performance_based", "extensions"]
                ),
                
                # Collaboration framework
                TemplateVariation(
                    style="collaboration_framework",
                    template="Collaboration Framework\n\n👤 Creator: {influencer}\n🏢 Brand: {brand}\n💰 Investment: ${fee:,}\n📱 Content: {deliverables}\n⏱️ Duration: {engagement_term} months\n🚫 Exclusivity: {exclusivity_period} weeks\n🚀 Launch: {start_date}\n\nFramework includes content approval workflows, performance tracking, and brand guideline adherence. Mutual success metrics with transparent reporting.",
                    complexity_level="medium",
                    structural_elements=["framework", "workflows", "adherence", "transparent", "reporting"]
                ),
                
                # Project specification
                TemplateVariation(
                    style="project_specification",
                    template="Project Specification: {influencer} Partnership\n\nClient: {brand}\nProject Value: ${fee:,}\nScope of Work: {deliverables}\nProject Timeline: {engagement_term} months\nRestriction Period: {exclusivity_period} weeks\nCommencement Date: {start_date}\n\nProject includes content creation, brand alignment verification, performance monitoring, and deliverable quality assurance. Payment schedule tied to milestone completion.",
                    complexity_level="medium",
                    structural_elements=["specification", "verification", "monitoring", "assurance", "milestone"]
                )
            ],
            
            "complex": [
                # Comprehensive partnership agreement
                TemplateVariation(
                    style="comprehensive_agreement",
                    template="Comprehensive Influencer Partnership Agreement\n\nParties: {influencer} (Creator) and {brand} (Brand)\nTotal Compensation: ${fee:,} (structured payment schedule)\nContent Portfolio: {deliverables}\nEngagement Period: {engagement_term} months\nExclusivity Clause: {exclusivity_period} weeks restriction on competitor collaborations\nCampaign Launch: {start_date}\n\nAdditional Terms:\n- Content approval process with 48-hour turnaround\n- Performance bonuses based on engagement metrics\n- Usage rights extension for additional compensation\n- Brand safety and compliance requirements\n- Cross-platform promotion strategy\n- Measurement and analytics reporting\n- Relationship renewal options based on performance\n\nSuccess is measured through reach, engagement rate, brand sentiment, and conversion metrics. This partnership represents a strategic alliance with potential for long-term collaboration.",
                    complexity_level="complex",
                    structural_elements=["comprehensive", "structured", "portfolio", "bonuses", "compliance", "analytics", "renewal", "alliance"]
                ),
                
                # Marketing campaign specification
                TemplateVariation(
                    style="marketing_specification",
                    template="Integrated Marketing Campaign Specification\n\nLead Creator: {influencer}\nBrand Partner: {brand}\nCampaign Investment: ${fee:,}\nContent Creation Scope: {deliverables}\nCampaign Duration: {engagement_term} months\nCompetitor Exclusivity: {exclusivity_period} weeks\nActivation Date: {start_date}\n\nCampaign Objectives:\n• Brand awareness amplification (target: 2M+ reach)\n• Engagement rate optimization (target: 4%+ average)\n• Authentic brand integration across platforms\n• Community building and audience growth\n• Conversion tracking and attribution\n\nContent Strategy:\n• Platform-specific optimization\n• Authentic storytelling approach\n• Brand message consistency\n• Audience engagement tactics\n• Cross-promotion coordination\n\nPerformance Framework:\n• Real-time analytics monitoring\n• Weekly performance reviews\n• Content optimization recommendations\n• ROI measurement and reporting\n• Long-term relationship assessment",
                    complexity_level="complex",
                    structural_elements=["integrated", "amplification", "optimization", "attribution", "coordination", "recommendations", "assessment"]
                ),
                
                # Strategic partnership proposal
                TemplateVariation(
                    style="strategic_partnership",
                    template="Strategic Partnership Proposal: {brand} x {influencer}\n\nExecutive Summary:\n{brand} proposes a comprehensive influencer partnership with {influencer}, representing a total investment of ${fee:,}. This strategic alliance encompasses {deliverables} across a {engagement_term}-month engagement period, with {exclusivity_period} weeks of category exclusivity.\n\nPartnership Framework:\nLaunch Date: {start_date}\nContent Strategy: Multi-platform brand integration\nAudience Targeting: Primary demographics 18-45, secondary 25-55\nGeographic Focus: Australian market with potential expansion\n\nValue Propositions:\n• Enhanced brand credibility through authentic creator partnership\n• Expanded audience reach and engagement\n• Content creation across multiple touchpoints\n• Long-term brand ambassador potential\n• Performance-driven compensation structure\n• Creative collaboration opportunities\n\nPerformance Metrics:\n• Reach and impression tracking\n• Engagement rate monitoring (target: 3.5%+)\n• Brand mention sentiment analysis\n• Website traffic attribution\n• Conversion tracking and ROI measurement\n• Social listening and brand awareness studies\n\nRisk Management:\n• Content approval workflows\n• Brand safety protocols\n• Crisis communication procedures\n• Performance milestone checkpoints\n• Contract flexibility provisions\n\nThis partnership represents mutual value creation with scalable success metrics and long-term relationship potential.",
                    complexity_level="complex",
                    structural_elements=["executive", "alliance", "demographics", "credibility", "touchpoints", "attribution", "protocols", "scalable"]
                ),
                
                # Brand ambassador program
                TemplateVariation(
                    style="ambassador_program",
                    template="Brand Ambassador Program: {influencer} x {brand}\n\nProgram Overview:\nWe're excited to invite {influencer} to join the {brand} Ambassador Program with an initial investment of ${fee:,}. This comprehensive program includes {deliverables} over {engagement_term} months, with {exclusivity_period} weeks category exclusivity.\n\nProgram Commencement: {start_date}\n\nAmbassador Benefits:\n• Competitive compensation structure\n• Exclusive product access and early releases\n• Behind-the-scenes brand experiences\n• Co-creation opportunities\n• Performance bonuses and incentives\n• Professional development support\n• Long-term partnership potential\n\nContent Requirements:\n• Authentic brand storytelling\n• Platform-optimized content creation\n• Community engagement and interaction\n• Brand event participation\n• Product showcase and reviews\n• Educational content development\n\nPerformance Excellence:\n• Monthly performance reviews\n• Quarterly business reviews\n• Annual partnership assessments\n• Continuous optimization strategies\n• Market trend analysis and adaptation\n• Competitive landscape monitoring\n\nSuccess Metrics:\n• Brand affinity measurement\n• Audience growth and engagement\n• Content performance analytics\n• Sales attribution and conversion\n• Brand sentiment tracking\n• Market share impact assessment\n\nProgram Evolution:\n• 6-month performance review\n• Program enhancement opportunities\n• Expansion into additional markets\n• Long-term strategic planning\n• Innovation and trend integration",
                    complexity_level="complex",
                    structural_elements=["ambassador", "comprehensive", "incentives", "optimization", "affinity", "evolution"]
                ),
                
                # Global campaign framework
                TemplateVariation(
                    style="global_framework",
                    template="Global Campaign Framework: {influencer} Partnership\n\nCampaign Lead: {influencer}\nGlobal Brand: {brand}\nTotal Investment: ${fee:,} (multi-market allocation)\nContent Ecosystem: {deliverables}\nCampaign Timeline: {engagement_term} months\nCategory Exclusivity: {exclusivity_period} weeks globally\nGlobal Launch: {start_date}\n\nMarket Strategy:\n• Primary Market: Australia/New Zealand\n• Secondary Markets: UK, Canada, US\n• Tertiary Opportunities: European expansion\n• Localization requirements per market\n• Cultural adaptation strategies\n• Regional compliance considerations\n\nContent Architecture:\n• Hero content for global distribution\n• Market-specific adaptations\n• Platform optimization (Instagram, TikTok, YouTube)\n• Cross-cultural messaging strategies\n• Multi-language considerations\n• Time zone coordination\n\nTechnology Integration:\n• Advanced analytics platform\n• Real-time performance monitoring\n• AI-driven content optimization\n• Automated reporting systems\n• ROI attribution modeling\n• Predictive performance analytics\n\nStakeholder Management:\n• Global brand team coordination\n• Regional marketing alignment\n• Legal and compliance oversight\n• Creative agency collaboration\n• Media planning integration\n• Public relations coordination\n\nInnovation Components:\n• Emerging platform experimentation\n• AR/VR content exploration\n• Interactive experience development\n• Community building initiatives\n• User-generated content strategies\n• Influencer network expansion\n\nRisk Mitigation:\n• Cultural sensitivity protocols\n• Crisis communication plans\n• Performance guarantee mechanisms\n• Contract flexibility provisions\n• Market-specific contingencies\n• Reputation management strategies",
                    complexity_level="complex",
                    structural_elements=["ecosystem", "localization", "architecture", "coordination", "stakeholder", "mitigation"]
                ),
            ]
        }
    
    def _init_noise_patterns(self):
        """Initialize noise injection patterns for realistic text variation"""
        
        self.noise_patterns = {
            "typos": {
                "the": ["teh", "hte"],
                "and": ["nd", "an"],
                "with": ["wth", "w/"],
                "campaign": ["campain", "campagin"],
                "influencer": ["inflencer", "influencr"],
                "budget": ["budgt", "buget"],
                "content": ["contnt", "conten"],
                "exclusive": ["exclusiv", "exlusive"],
                "deliverables": ["delivs", "deliverabls"],
                "engagement": ["engagment", "engagemt"],
                "approximately": ["approx", "aprox"],
                "partnership": ["partnrship", "partership"]
            },
            
            "synonyms": {
                "budget": ["investment", "compensation", "fee", "payment", "allocation"],
                "campaign": ["project", "collaboration", "partnership", "initiative"],
                "content": ["material", "assets", "deliverables", "creative"],
                "influencer": ["creator", "partner", "talent", "ambassador"],
                "brand": ["company", "client", "business", "organization"],
                "exclusive": ["sole", "unique", "dedicated", "restricted"],
                "create": ["produce", "develop", "generate", "craft"],
                "launch": ["start", "begin", "initiate", "kick-off"],
                "target": ["aim", "focus", "goal", "objective"]
            },
            
            "australian_colloquialisms": [
                "mate", "no worries", "fair dinkum", "ripper", "bonkers",
                "chockers", "bloody good", "ace", "beaut", "grouse"
            ],
            
            "business_abbreviations": {
                "as soon as possible": "ASAP",
                "to be confirmed": "TBC", 
                "to be determined": "TBD",
                "return on investment": "ROI",
                "key performance indicator": "KPI",
                "social media": "SM",
                "content creation": "CC",
                "brand awareness": "BA"
            },
            
            "incomplete_phrases": [
                "...need to confirm exact dates",
                "...pending approval from legal",
                "...subject to final budget confirmation",
                "...awaiting creative brief details",
                "...TBC on specific requirements",
                "...open to discussion on terms"
            ],
            
            "extra_clauses": [
                "Please let me know your thoughts on this.",
                "Looking forward to your response.",
                "This is just an initial proposal.",
                "Happy to discuss further details.",
                "Let's chat about the specifics.",
                "Would love to hear your feedback."
            ]
        }
    
    def _apply_noise_injection(self, text: str, noise_level: float = 0.3) -> str:
        """Apply controlled noise injection to text"""
        
        if random.random() > noise_level:
            return text
        
        # Apply typos (10% chance per word)
        if random.random() < 0.1:
            for correct, typos in self.noise_patterns["typos"].items():
                if correct in text.lower() and random.random() < 0.1:
                    text = text.replace(correct, random.choice(typos))
        
        # Apply synonyms (15% chance)
        if random.random() < 0.15:
            for word, synonyms in self.noise_patterns["synonyms"].items():
                if word in text.lower() and random.random() < 0.2:
                    text = text.replace(word, random.choice(synonyms))
        
        # Add Australian colloquialisms (20% chance)
        if random.random() < 0.2:
            colloquialism = random.choice(self.noise_patterns["australian_colloquialisms"])
            text += f" {colloquialism}!"
        
        # Add business abbreviations (10% chance)
        if random.random() < 0.1:
            for phrase, abbrev in self.noise_patterns["business_abbreviations"].items():
                if phrase in text.lower():
                    text = text.replace(phrase, abbrev)
        
        # Add incomplete phrases (15% chance)
        if random.random() < 0.15:
            incomplete = random.choice(self.noise_patterns["incomplete_phrases"])
            text += f"\n\n{incomplete}"
        
        # Add extra clauses (25% chance)
        if random.random() < 0.25:
            extra = random.choice(self.noise_patterns["extra_clauses"])
            text += f" {extra}"
        
        return text
    
    def _paraphrase_text(self, text: str) -> str:
        """Apply paraphrasing to create semantic variation"""
        
        if not self.use_paraphrasing or not self.paraphraser:
            return text
        
        try:
            # Split long text into sentences for better paraphrasing
            sentences = text.split('. ')
            paraphrased_sentences = []
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    paraphrased_sentences.append(sentence)
                    continue
                
                # Apply paraphrasing with fallback
                try:
                    result = self.paraphraser(f"paraphrase: {sentence.strip()}", 
                                            max_length=200, 
                                            num_return_sequences=1, 
                                            temperature=0.7)
                    paraphrased = result[0]['generated_text'].strip()
                    paraphrased_sentences.append(paraphrased)
                except:
                    paraphrased_sentences.append(sentence)  # Fallback to original
            
            return '. '.join(paraphrased_sentences)
            
        except Exception as e:
            print(f"⚠️ Paraphrasing failed: {e}")
            return text
    
    def _check_similarity(self, new_text: str) -> bool:
        """Check if new text is too similar to existing samples"""
        
        if not self.generated_samples:
            return True  # First sample is always acceptable
        
        # Embed the new text
        new_embedding = self.similarity_model.encode([new_text])[0]
        
        # Check similarity against existing samples
        for existing_embedding in self.sample_embeddings:
            similarity = np.dot(new_embedding, existing_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
            )
            
            if similarity > self.similarity_threshold:
                return False  # Too similar
        
        return True  # Acceptable similarity
    
    def _add_to_cache(self, text: str):
        """Add text to similarity checking cache"""
        embedding = self.similarity_model.encode([text])[0]
        self.generated_samples.append(text)
        self.sample_embeddings.append(embedding)
    
    def _generate_realistic_parameters(self, complexity: str) -> Dict:
        """Generate realistic and varied parameters"""
        
        # Select brand and determine industry
        industry = random.choice(list(self.australian_brands.keys()))
        brand = random.choice(self.australian_brands[industry])
        
        # Select influencer with variation
        influencer = random.choice(self.influencer_names)
        
        # Generate realistic fee based on complexity and industry
        if complexity == "simple":
            tier = random.choice(["nano", "micro"])
        elif complexity == "medium":
            tier = random.choice(["micro", "mid"])
        else:  # complex
            tier = random.choice(["mid", "macro", "mega"])
        
        fee_min, fee_max = self.fee_tiers[tier]
        # Add industry multipliers
        industry_multipliers = {
            "fashion": 1.2, "beauty": 1.3, "tech": 1.1, 
            "food": 1.0, "automotive": 1.15, "home": 0.95
        }
        multiplier = industry_multipliers.get(industry, 1.0)
        fee = int(random.randint(fee_min, fee_max) * multiplier)
        
        # Generate varied deliverables
        deliverable_pool = self.deliverables[complexity]
        num_deliverables = random.randint(1, min(4, len(deliverable_pool)))
        deliverables = random.sample(deliverable_pool, num_deliverables)
        
        # Add realistic quantities
        deliverable_quantities = []
        for deliverable in deliverables:
            quantity = random.randint(1, 5)
            deliverable_quantities.append(f"{quantity} x {deliverable}")
        
        # Generate realistic date ranges
        start_date = datetime.now() + timedelta(days=random.randint(14, 365))
        
        # Realistic terms based on complexity
        if complexity == "simple":
            engagement_term = random.randint(1, 3)
            exclusivity_period = random.randint(2, 8)
            usage_term = random.randint(6, 12)
        elif complexity == "medium":
            engagement_term = random.randint(2, 6)
            exclusivity_period = random.randint(4, 16)
            usage_term = random.randint(8, 24)
        else:  # complex
            engagement_term = random.randint(6, 18)
            exclusivity_period = random.randint(8, 52)
            usage_term = random.randint(12, 60)
        
        return {
            "brand": brand,
            "industry": industry,
            "influencer": influencer,
            "fee": fee,
            "fee_tier": tier,
            "deliverables": deliverable_quantities,
            "start_date": start_date,
            "engagement_term": engagement_term,
            "exclusivity_period": exclusivity_period,
            "usage_term": usage_term
        }
    
    def generate_diverse_sample(self, complexity: str = "medium", max_attempts: int = 50) -> Optional[Dict]:
        """Generate a single diverse sample with similarity checking"""
        
        for attempt in range(max_attempts):
            # Get realistic parameters
            params = self._generate_realistic_parameters(complexity)
            
            # Select random template variation
            template_variations = self.template_variations[complexity]
            template_var = random.choice(template_variations)
            
            # Generate base text using template
            try:
                raw_text = template_var.template.format(
                    influencer=params["influencer"],
                    brand=params["brand"],
                    fee=params["fee"],
                    deliverables=", ".join(params["deliverables"]),
                    engagement_term=params["engagement_term"],
                    exclusivity_period=params["exclusivity_period"],
                    usage_term=params["usage_term"],
                    start_date=params["start_date"].strftime("%B %Y")
                )
            except KeyError as e:
                print(f"⚠️ Template formatting error: {e}")
                continue
            
            # Apply paraphrasing for semantic variation
            if random.random() < 0.4:  # 40% chance of paraphrasing
                raw_text = self._paraphrase_text(raw_text)
            
            # Apply noise injection
            noise_level = {"simple": 0.2, "medium": 0.3, "complex": 0.5}[complexity]
            raw_text = self._apply_noise_injection(raw_text, noise_level)
            
            # Check similarity
            if self._check_similarity(raw_text):
                # Add to cache
                self._add_to_cache(raw_text)
                
                # Generate extracted fields
                extracted_fields = {
                    "influencer": params["influencer"],
                    "client": params["brand"],
                    "brand": params["brand"],
                    "industry": params["industry"],
                    "campaign": f"{params['brand']} Campaign",
                    "fee": f"${params['fee']:,}",
                    "fee_numeric": params["fee"],
                    "fee_tier": params["fee_tier"],
                    "deliverables": params["deliverables"],
                    "exclusivity_period": f"{params['exclusivity_period']} weeks",
                    "engagement_term": f"{params['engagement_term']} months",
                    "usage_term": f"{params['usage_term']} months",
                    "territory": "Australia",
                    "start_date": params["start_date"].strftime("%B %Y"),
                    "template_style": template_var.style,
                    "structural_elements": template_var.structural_elements
                }
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence(complexity, raw_text, template_var.style)
                
                # Determine classification
                classification = self._determine_classification(complexity, params)
                
                return {
                    "id": f"enhanced_{str(uuid.uuid4())[:8]}",
                    "raw_input_text": raw_text,
                    "extracted_fields": extracted_fields,
                    "template_match": template_var.style,
                    "complexity_level": complexity,
                    "confidence_score": confidence_score,
                    "classification": classification,
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "template_style": template_var.style,
                        "structural_elements": template_var.structural_elements,
                        "fee_tier": params["fee_tier"],
                        "industry": params["industry"],
                        "num_deliverables": len(params["deliverables"]),
                        "noise_applied": True,
                        "paraphrased": True if random.random() < 0.4 else False,
                        "generation_attempt": attempt + 1
                    }
                }
        
        print(f"⚠️ Failed to generate unique sample after {max_attempts} attempts")
        return None
    
    def _calculate_confidence(self, complexity: str, raw_text: str, style: str) -> float:
        """Calculate extraction confidence based on complexity and text characteristics"""
        
        base_confidence = {
            "simple": 0.85,
            "medium": 0.72,
            "complex": 0.58
        }
        
        confidence = base_confidence[complexity]
        
        # Adjust based on style clarity
        style_adjustments = {
            "formal_brief": 0.1, "bullet_points": 0.05, "contract_summary": 0.08,
            "brief_sms": -0.15, "conversational_pitch": -0.08, "casual_email": -0.05
        }
        confidence += style_adjustments.get(style, 0)
        
        # Adjust based on text characteristics
        if len(re.findall(r'\b(TBC|approx|arnd|delivs|budgt)\b', raw_text.lower())) > 0:
            confidence -= 0.1
        
        if "..." in raw_text or "TBC" in raw_text:
            confidence -= 0.08
        
        if len(raw_text.split()) < 20:
            confidence -= 0.05
        
        # Add slight randomness for realism
        confidence += random.uniform(-0.03, 0.03)
        
        return max(0.3, min(0.95, confidence))
    
    def _determine_classification(self, complexity: str, params: Dict) -> str:
        """Determine classification based on parameters"""
        
        # Main classification logic
        if params["fee"] > 50000:
            return "HIGH_VALUE_PARTNERSHIP"
        elif params["fee"] > 15000:
            return "STANDARD_COLLABORATION"
        elif len(params["deliverables"]) > 3:
            return "MULTI_CONTENT_CAMPAIGN"
        else:
            return "BASIC_INFLUENCER_AGREEMENT"
    
    def generate_dataset(self, num_samples: int = 1000, 
                        complexity_distribution: Dict[str, float] = None) -> List[Dict]:
        """Generate complete diverse dataset"""
        
        if complexity_distribution is None:
            complexity_distribution = {
                "simple": 0.3,
                "medium": 0.5,
                "complex": 0.2
            }
        
        print(f"🎯 Generating {num_samples} diverse samples...")
        print(f"📊 Complexity distribution: {complexity_distribution}")
        print(f"🎯 Similarity threshold: {self.similarity_threshold}")
        
        dataset = []
        complexity_counts = {
            "simple": int(num_samples * complexity_distribution["simple"]),
            "medium": int(num_samples * complexity_distribution["medium"]),
            "complex": int(num_samples * complexity_distribution["complex"])
        }
        
        # Adjust for rounding
        total_assigned = sum(complexity_counts.values())
        if total_assigned < num_samples:
            complexity_counts["medium"] += num_samples - total_assigned
        
        for complexity, count in complexity_counts.items():
            print(f"🔄 Generating {count} {complexity} samples...")
            
            for i in range(count):
                sample = self.generate_diverse_sample(complexity)
                if sample:
                    dataset.append(sample)
                
                if (i + 1) % 50 == 0:
                    print(f"  ✅ Generated {i + 1}/{count} {complexity} samples")
        
        print(f"✅ Generated {len(dataset)} unique samples")
        print(f"📊 Similarity check cache size: {len(self.generated_samples)}")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = "enhanced_synthetic_dataset.json"):
        """Save dataset with metadata"""
        
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_samples": len(dataset),
            "similarity_threshold": self.similarity_threshold,
            "paraphrasing_enabled": self.use_paraphrasing,
            "unique_templates": sum(len(templates) for templates in self.template_variations.values()),
            "complexity_distribution": {
                complexity: len([s for s in dataset if s["complexity_level"] == complexity])
                for complexity in ["simple", "medium", "complex"]
            },
            "style_distribution": {},
            "generator_version": "enhanced_v1.0"
        }
        
        # Calculate style distribution
        for sample in dataset:
            style = sample.get("template_match", "unknown")
            metadata["style_distribution"][style] = metadata["style_distribution"].get(style, 0) + 1
        
        output_data = {
            "metadata": metadata,
            "samples": dataset
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Saved {len(dataset)} samples to {filename}")
        print(f"📈 Template style diversity: {len(metadata['style_distribution'])} unique styles")
        return filename

def main():
    """Test the enhanced generator"""
    print("🧪 Testing Enhanced Synthetic Generator")
    
    # Initialize generator
    generator = EnhancedSyntheticGenerator(
        use_paraphrasing=True,
        similarity_threshold=0.92
    )
    
    # Generate small test dataset
    test_dataset = generator.generate_dataset(
        num_samples=100,
        complexity_distribution={
            "simple": 0.4,
            "medium": 0.4,
            "complex": 0.2
        }
    )
    
    # Save test dataset
    generator.save_dataset(test_dataset, "test_enhanced_dataset.json")
    
    print("✅ Enhanced generator test completed!")

if __name__ == "__main__":
    main() 