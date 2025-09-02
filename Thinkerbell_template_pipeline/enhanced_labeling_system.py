#!/usr/bin/env python3
"""
Enhanced Labeling System for Thinkerbell
Addresses overfitting by expanding label space with:
1. Sub-labels and finer-grained distinctions
2. Multi-label samples (2-3 labels per chunk)
3. "None of the above" distractor class
4. Proportional distribution maintenance
"""

import json
import os
import random
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class EnhancedLabelingSystem:
    """Enhanced labeling system with expanded label space and complexity"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed
        
        # Original 9 labels
        self.base_labels = [
            "brand", "campaign", "client", "deliverables", 
            "engagement_term", "exclusivity_scope", "fee", "usage_term", "exclusivity_period"
        ]
        
        # Define sub-labels for each base label to increase complexity
        self.sub_labels = {
            "brand": ["brand_name", "brand_type", "brand_category", "brand_mention"],
            "campaign": ["campaign_type", "campaign_timeline", "campaign_scope", "campaign_platform"],
            "client": ["client_name", "client_role", "client_requirements"],
            "deliverables": ["content_type", "content_count", "content_format", "content_quality"],
            "engagement_term": ["duration_weeks", "duration_months", "duration_specific"],
            "exclusivity_scope": ["exclusivity_brand", "exclusivity_category", "exclusivity_platform"],
            "fee": ["budget_amount", "budget_range", "payment_terms"],
            "usage_term": ["usage_duration", "usage_rights", "usage_scope"],
            "exclusivity_period": ["exclusivity_duration", "exclusivity_start", "exclusivity_end"]
        }
        
        # Content-specific sub-labels for more granular classification
        self.content_specific_labels = {
            "social_media": ["instagram_post", "instagram_story", "instagram_reel", "facebook_post", 
                           "facebook_story", "twitter_post", "tiktok_video", "youtube_video"],
            "content_format": ["photo", "video", "story", "reel", "live_stream", "blog_post"],
            "content_quality": ["professional", "user_generated", "lifestyle", "product_focused"],
            "engagement_type": ["collaboration", "sponsorship", "ambassador", "partnership"],
            "timing": ["immediate", "scheduled", "seasonal", "event_based"],
            "audience": ["target_demo", "follower_count", "engagement_rate", "niche_specific"]
        }
        
        # Distractor categories - realistic but different domains
        self.distractor_categories = {
            "employment": ["job_description", "salary_negotiation", "work_schedule", "benefits"],
            "real_estate": ["property_details", "lease_terms", "rental_price", "location"],
            "financial": ["loan_terms", "investment_details", "insurance_policy", "tax_info"],
            "legal_general": ["court_proceeding", "legal_advice", "contract_dispute", "compliance"],
            "academic": ["course_description", "academic_requirements", "research_proposal", "thesis"],
            "medical": ["treatment_plan", "medical_history", "prescription", "appointment"],
            "technical": ["software_specs", "hardware_requirements", "system_architecture", "api_docs"]
        }
        
        # Define label combination patterns with varying complexity
        self.label_patterns = {
            "minimal": 1,      # 1-2 labels (20% of samples)
            "moderate": 2,     # 3-4 labels (40% of samples) 
            "complex": 3,      # 5-6 labels (30% of samples)
            "comprehensive": 4  # 7+ labels (10% of samples)
        }
        
        self.pattern_distribution = {
            "minimal": 0.20,
            "moderate": 0.40,
            "complex": 0.30,
            "comprehensive": 0.10
        }

    def generate_enhanced_labels(self, original_chunk: Dict, complexity_level: Optional[str] = None) -> Dict:
        """Generate enhanced labels for a chunk with increased complexity"""
        
        if complexity_level is None:
            complexity_level = self._select_complexity_level()
        
        chunk_text = original_chunk.get("text", "")
        original_labels = set(original_chunk.get("labels", []))
        
        # Start with base labels that are actually present
        enhanced_labels = []
        sub_labels = []
        
        # Process each original label to add sub-labels
        for base_label in original_labels:
            if base_label in self.sub_labels:
                # Add the base label
                enhanced_labels.append(base_label)
                
                # Add relevant sub-labels based on text content
                relevant_sub_labels = self._extract_relevant_sub_labels(
                    chunk_text, base_label, complexity_level
                )
                sub_labels.extend(relevant_sub_labels)
        
        # Add content-specific labels based on text analysis
        content_labels = self._extract_content_specific_labels(chunk_text, complexity_level)
        enhanced_labels.extend(content_labels)
        
        # Combine all labels
        all_labels = enhanced_labels + sub_labels
        
        # Apply complexity filtering
        final_labels = self._apply_complexity_filtering(all_labels, complexity_level)
        
        # Create enhanced chunk
        enhanced_chunk = original_chunk.copy()
        enhanced_chunk["enhanced_labels"] = final_labels
        enhanced_chunk["complexity_level"] = complexity_level
        enhanced_chunk["label_count"] = len(final_labels)
        
        return enhanced_chunk

    def _select_complexity_level(self) -> str:
        """Select complexity level based on distribution"""
        rand_val = random.random()
        cumulative = 0
        
        for pattern, prob in self.pattern_distribution.items():
            cumulative += prob
            if rand_val <= cumulative:
                return pattern
        
        return "moderate"  # fallback

    def _extract_relevant_sub_labels(self, text: str, base_label: str, complexity: str) -> List[str]:
        """Extract relevant sub-labels for a base label based on text content"""
        text_lower = text.lower()
        sub_labels = []
        
        if base_label not in self.sub_labels:
            return sub_labels
        
        available_sub_labels = self.sub_labels[base_label]
        
        # Define patterns for each sub-label
        patterns = {
            "brand_name": [r'\b\w+(?=\s+campaign|\s+brand)', r'for\s+(\w+)', r'(\w+)\s+-\s+\$'],
            "brand_type": [r'restaurant|fashion|tech|beauty|fitness|food|retail'],
            "content_type": [r'instagram|facebook|twitter|tiktok|youtube|photo|video|post|story|reel'],
            "content_count": [r'\d+\s*x\s*', r'\d+\s+(?:post|story|reel|photo|video)'],
            "budget_amount": [r'\$[\d,]+', r'budget.*?\$[\d,]+'],
            "duration_weeks": [r'\d+\s*w(?:eek)?', r'\d+\s*week'],
            "duration_months": [r'\d+\s*m(?:onth)?', r'\d+\s*month'],
            "usage_duration": [r'usage.*?\d+\s*month', r'rights.*?\d+\s*month'],
            "exclusivity_duration": [r'exclusivity.*?\d+\s*week', r'\d+\s*week\s*exclusivity']
        }
        
        # Extract sub-labels based on patterns
        for sub_label in available_sub_labels:
            if sub_label in patterns:
                for pattern in patterns[sub_label]:
                    if re.search(pattern, text_lower):
                        sub_labels.append(sub_label)
                        break
        
        # Add random sub-labels based on complexity
        remaining_sub_labels = [sl for sl in available_sub_labels if sl not in sub_labels]
        
        complexity_counts = {
            "minimal": 0,
            "moderate": min(1, len(remaining_sub_labels)),
            "complex": min(2, len(remaining_sub_labels)),
            "comprehensive": min(3, len(remaining_sub_labels))
        }
        
        additional_count = complexity_counts.get(complexity, 1)
        if additional_count > 0 and remaining_sub_labels:
            additional_labels = random.sample(remaining_sub_labels, additional_count)
            sub_labels.extend(additional_labels)
        
        return sub_labels

    def _extract_content_specific_labels(self, text: str, complexity: str) -> List[str]:
        """Extract content-specific labels based on text analysis"""
        text_lower = text.lower()
        content_labels = []
        
        # Social media platform detection
        social_patterns = {
            "instagram_post": [r'instagram\s+post', r'ig\s+post'],
            "instagram_story": [r'instagram\s+story', r'ig\s+story'],
            "instagram_reel": [r'instagram\s+reel', r'ig\s+reel'],
            "facebook_post": [r'facebook\s+post', r'fb\s+post'],
            "facebook_story": [r'facebook\s+story', r'fb\s+story'],
            "twitter_post": [r'twitter\s+post', r'tweet'],
            "tiktok_video": [r'tiktok', r'tt\s+video'],
            "youtube_video": [r'youtube', r'yt\s+video']
        }
        
        for label, patterns in social_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    content_labels.append(label)
                    break
        
        # Content format detection
        format_patterns = {
            "photo": [r'photo', r'image', r'picture'],
            "video": [r'video', r'clip', r'footage'],
            "lifestyle": [r'lifestyle', r'casual', r'everyday'],
            "product_focused": [r'product', r'showcase', r'feature']
        }
        
        for label, patterns in format_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    content_labels.append(label)
                    break
        
        # Add random content labels based on complexity
        all_content_labels = []
        for category in self.content_specific_labels.values():
            all_content_labels.extend(category)
        
        remaining_labels = [cl for cl in all_content_labels if cl not in content_labels]
        
        complexity_counts = {
            "minimal": 0,
            "moderate": min(1, len(remaining_labels)),
            "complex": min(2, len(remaining_labels)),
            "comprehensive": min(3, len(remaining_labels))
        }
        
        additional_count = complexity_counts.get(complexity, 1)
        if additional_count > 0 and remaining_labels:
            additional_labels = random.sample(remaining_labels, additional_count)
            content_labels.extend(additional_labels)
        
        return content_labels

    def _apply_complexity_filtering(self, labels: List[str], complexity: str) -> List[str]:
        """Filter labels based on complexity level to control final count"""
        labels = list(set(labels))  # Remove duplicates
        
        target_counts = {
            "minimal": (1, 3),
            "moderate": (3, 6),
            "complex": (5, 9),
            "comprehensive": (7, 12)
        }
        
        min_count, max_count = target_counts.get(complexity, (3, 6))
        
        if len(labels) < min_count:
            # Need to add more labels - add some random relevant ones
            all_possible = []
            for sub_label_list in self.sub_labels.values():
                all_possible.extend(sub_label_list)
            for content_list in self.content_specific_labels.values():
                all_possible.extend(content_list)
            
            available = [l for l in all_possible if l not in labels]
            needed = min_count - len(labels)
            if available:
                additional = random.sample(available, min(needed, len(available)))
                labels.extend(additional)
        
        elif len(labels) > max_count:
            # Too many labels - randomly sample to reduce
            labels = random.sample(labels, max_count)
        
        return sorted(labels)

    def generate_distractor_samples(self, count: int) -> List[Dict]:
        """Generate realistic distractor samples that don't belong to any influencer marketing category"""
        distractor_samples = []
        
        # Templates for different distractor categories
        templates = {
            "employment": [
                "Seeking {role} for {company}. Salary range ${min_salary}-${max_salary}. {benefits} provided. Start date: {start_date}. Requirements: {requirements}.",
                "{company} is hiring {role}. {work_type} position with {schedule}. Competitive salary and {benefits}. Apply by {deadline}.",
                "Job opening: {role} at {company}. Responsibilities include {duties}. Offering ${salary} plus {benefits}. {location} based."
            ],
            "real_estate": [
                "Property for rent: {property_type} in {location}. ${rent}/month, {bedrooms}BR/{bathrooms}BA. Available {date}. Features: {features}.",
                "{property_type} for sale in {location}. ${price}, {square_feet} sq ft. {bedrooms} bedrooms, {bathrooms} bathrooms. Contact {agent}.",
                "Lease available: {property_type} at {address}. Monthly rent ${rent}, security deposit ${deposit}. {lease_term} lease term."
            ],
            "financial": [
                "Loan offer: {loan_type} with {rate}% APR. Amount up to ${amount}. Term: {term} months. {requirements}. Apply today.",
                "Investment opportunity: {investment_type} with projected {return}% return. Minimum investment ${min_amount}. Risk level: {risk}.",
                "Insurance policy: {insurance_type} coverage for ${coverage_amount}. Monthly premium ${premium}. Deductible: ${deductible}."
            ],
            "legal_general": [
                "Legal consultation for {case_type}. Attorney {attorney_name} specializing in {specialty}. Initial consultation ${fee}. Call {phone}.",
                "Court hearing scheduled for {date} regarding {case_matter}. Case #{case_number}. Appear at {courthouse} at {time}.",
                "Contract review needed for {contract_type}. Legal fees ${hourly_rate}/hour. Estimated {hours} hours required."
            ],
            "academic": [
                "Course: {course_name} ({course_code}). Prerequisites: {prerequisites}. Credits: {credits}. Professor: {professor}. Schedule: {schedule}.",
                "Research proposal: {topic} for {degree_program}. Advisor: {advisor}. Funding: ${amount}. Duration: {duration} months.",
                "Thesis defense: {student_name} presenting '{title}' on {date} at {time}. Committee: {committee_members}."
            ],
            "medical": [
                "Appointment: {patient_name} with Dr. {doctor} on {date} at {time}. Reason: {reason}. Insurance: {insurance}.",
                "Prescription: {medication} {dosage} for {condition}. Take {frequency}. Refills: {refills}. Pharmacy: {pharmacy}.",
                "Treatment plan for {condition}: {treatment} for {duration}. Follow-up in {followup_weeks} weeks. Cost: ${cost}."
            ],
            "technical": [
                "System requirements: {software} v{version}. Hardware: {cpu}, {ram}GB RAM, {storage}GB storage. OS: {os}.",
                "API documentation for {service}. Endpoint: {endpoint}. Authentication: {auth_type}. Rate limit: {rate_limit} requests/hour.",
                "Software upgrade: {system} to v{new_version}. Downtime: {downtime} hours on {date}. Backup before {backup_time}."
            ]
        }
        
        # Fill templates with realistic data
        placeholders = {
            "employment": {
                "role": ["Software Engineer", "Marketing Manager", "Sales Associate", "Data Analyst", "HR Coordinator"],
                "company": ["TechCorp", "DataSystems", "Global Solutions", "Innovation Labs", "NextGen Industries"],
                "min_salary": ["50000", "60000", "75000", "80000", "90000"],
                "max_salary": ["70000", "85000", "95000", "110000", "120000"],
                "benefits": ["health insurance", "401k matching", "flexible hours", "remote work option"],
                "start_date": ["immediately", "January 2026", "within 2 weeks", "March 1st"],
                "requirements": ["2+ years experience", "bachelor's degree", "relevant certifications"],
                "work_type": ["full-time", "part-time", "contract", "remote"],
                "schedule": ["flexible hours", "9-5 schedule", "shift work", "weekends included"],
                "salary": ["65000", "75000", "85000", "95000"],
                "location": ["downtown", "remote", "hybrid", "onsite"],
                "duties": ["project management", "client communication", "data analysis", "team leadership"],
                "deadline": ["end of month", "next Friday", "January 15th"]
            },
            "real_estate": {
                "property_type": ["apartment", "house", "condo", "townhouse", "studio"],
                "location": ["downtown", "suburbs", "city center", "residential area"],
                "rent": ["1200", "1500", "1800", "2200", "2500"],
                "bedrooms": ["1", "2", "3", "4"],
                "bathrooms": ["1", "1.5", "2", "2.5"],
                "date": ["immediately", "next month", "January 1st", "February 15th"],
                "features": ["parking included", "pet-friendly", "gym access", "balcony"],
                "price": ["250000", "350000", "450000", "550000"],
                "square_feet": ["800", "1200", "1600", "2000"],
                "agent": ["Sarah Johnson", "Mike Davis", "Lisa Chen", "Robert Smith"],
                "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St"],
                "deposit": ["1200", "1500", "2000", "2500"],
                "lease_term": ["12 month", "6 month", "month-to-month"]
            }
            # Add more placeholders for other categories...
        }
        
        for i in range(count):
            # Randomly select category
            category = random.choice(list(templates.keys()))
            template = random.choice(templates[category])
            
            # Fill template with random values
            if category in placeholders:
                filled_text = template
                for placeholder, values in placeholders[category].items():
                    if f"{{{placeholder}}}" in filled_text:
                        filled_text = filled_text.replace(f"{{{placeholder}}}", random.choice(values))
            else:
                filled_text = template
            
            # Create distractor sample
            distractor_sample = {
                "sample_id": f"distractor_{i+1:04d}",
                "chunk_id": f"distractor_{i+1:04d}_c1",
                "text": filled_text,
                "token_count": len(filled_text.split()),
                "labels": ["none_of_above"],
                "enhanced_labels": [f"distractor_{category}", "none_of_above"],
                "complexity_level": "distractor",
                "label_count": 2,
                "is_distractor": True,
                "distractor_category": category
            }
            
            distractor_samples.append(distractor_sample)
        
        return distractor_samples

    def create_multi_label_samples(self, enhanced_chunks: List[Dict], multi_label_ratio: float = 0.3) -> List[Dict]:
        """Create multi-label samples by combining chunks with overlapping content"""
        multi_label_samples = []
        num_samples = int(len(enhanced_chunks) * multi_label_ratio)
        
        # Group chunks by similar content types
        content_groups = defaultdict(list)
        for chunk in enhanced_chunks:
            labels = set(chunk.get("enhanced_labels", []))
            
            # Group by primary content type
            if any("instagram" in label for label in labels):
                content_groups["instagram"].append(chunk)
            elif any("facebook" in label for label in labels):
                content_groups["facebook"].append(chunk)
            elif any("photo" in label for label in labels):
                content_groups["photo"].append(chunk)
            elif any("video" in label for label in labels):
                content_groups["video"].append(chunk)
            else:
                content_groups["general"].append(chunk)
        
        # Create combinations
        for i in range(num_samples):
            # Select 2-3 chunks to combine
            combination_size = random.choice([2, 3])
            
            # Try to select from same content group for realistic combinations
            if content_groups:
                group_name = random.choice(list(content_groups.keys()))
                if len(content_groups[group_name]) >= combination_size:
                    selected_chunks = random.sample(content_groups[group_name], combination_size)
                else:
                    # Fall back to random selection
                    selected_chunks = random.sample(enhanced_chunks, combination_size)
            else:
                selected_chunks = random.sample(enhanced_chunks, combination_size)
            
            # Combine chunks
            combined_text_parts = []
            combined_labels = set()
            
            for chunk in selected_chunks:
                # Extract key information from each chunk
                text = chunk.get("text", "")
                # Take first sentence or key phrases
                sentences = text.split(". ")
                key_phrase = sentences[0] if sentences else text[:100]
                combined_text_parts.append(key_phrase.strip())
                
                # Combine labels
                chunk_labels = chunk.get("enhanced_labels", [])
                combined_labels.update(chunk_labels)
            
            # Create combined text
            combined_text = ". ".join(combined_text_parts) + "."
            
            # Create multi-label sample
            multi_label_sample = {
                "sample_id": f"multi_{i+1:04d}",
                "chunk_id": f"multi_{i+1:04d}_c1", 
                "text": combined_text,
                "token_count": len(combined_text.split()),
                "labels": sorted(list(combined_labels)),
                "enhanced_labels": sorted(list(combined_labels)),
                "complexity_level": "multi_label",
                "label_count": len(combined_labels),
                "is_multi_label": True,
                "source_samples": [chunk["sample_id"] for chunk in selected_chunks]
            }
            
            multi_label_samples.append(multi_label_sample)
        
        return multi_label_samples

    def process_dataset(self, input_dir: str, output_dir: str, 
                       distractor_ratio: float = 0.15, 
                       multi_label_ratio: float = 0.25) -> Dict:
        """Process entire dataset with enhanced labeling"""
        
        logger.info("=== ENHANCED LABELING SYSTEM PROCESSING ===")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load all existing chunks
        logger.info(f"Loading chunks from {input_dir}")
        all_chunks = []
        
        for split in ["train", "val", "test"]:
            split_dir = input_path / split
            if split_dir.exists():
                for chunk_file in split_dir.glob("*.json"):
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            chunk = json.load(f)
                            chunk['original_split'] = split
                            all_chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to load {chunk_file}: {e}")
        
        logger.info(f"Loaded {len(all_chunks)} original chunks")
        
        # Enhance existing chunks
        logger.info("Enhancing existing chunks with expanded labels...")
        enhanced_chunks = []
        for chunk in all_chunks:
            enhanced_chunk = self.generate_enhanced_labels(chunk)
            enhanced_chunks.append(enhanced_chunk)
        
        # Create multi-label samples
        logger.info("Creating multi-label samples...")
        multi_label_samples = self.create_multi_label_samples(
            enhanced_chunks, multi_label_ratio
        )
        
        # Generate distractor samples
        total_samples = len(enhanced_chunks) + len(multi_label_samples)
        num_distractors = int(total_samples * distractor_ratio)
        
        logger.info(f"Generating {num_distractors} distractor samples...")
        distractor_samples = self.generate_distractor_samples(num_distractors)
        
        # Combine all samples
        all_samples = enhanced_chunks + multi_label_samples + distractor_samples
        
        logger.info(f"Total samples: {len(all_samples)}")
        logger.info(f"  - Enhanced original: {len(enhanced_chunks)}")
        logger.info(f"  - Multi-label: {len(multi_label_samples)}")
        logger.info(f"  - Distractors: {len(distractor_samples)}")
        
        # Calculate label statistics
        stats = self._calculate_enhanced_stats(all_samples)
        
        # Save enhanced dataset
        enhanced_output_dir = output_path / "enhanced_splits"
        enhanced_output_dir.mkdir(exist_ok=True)
        
        # Save all samples together for now (will split later)
        all_samples_file = enhanced_output_dir / "all_enhanced_samples.json"
        with open(all_samples_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_file = enhanced_output_dir / "enhanced_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced dataset saved to {enhanced_output_dir}")
        
        return stats

    def _calculate_enhanced_stats(self, samples: List[Dict]) -> Dict:
        """Calculate statistics for enhanced dataset"""
        
        stats = {
            "total_samples": len(samples),
            "label_distribution": defaultdict(int),
            "complexity_distribution": defaultdict(int),
            "label_count_distribution": defaultdict(int),
            "unique_label_combinations": set(),
            "sample_types": defaultdict(int)
        }
        
        for sample in samples:
            labels = sample.get("enhanced_labels", [])
            complexity = sample.get("complexity_level", "unknown")
            
            # Count label occurrences
            for label in labels:
                stats["label_distribution"][label] += 1
            
            # Count complexity distribution
            stats["complexity_distribution"][complexity] += 1
            
            # Count label count distribution
            label_count = len(labels)
            stats["label_count_distribution"][str(label_count)] += 1
            
            # Track unique combinations
            combo = "|".join(sorted(labels))
            stats["unique_label_combinations"].add(combo)
            
            # Sample type distribution
            if sample.get("is_distractor", False):
                stats["sample_types"]["distractor"] += 1
            elif sample.get("is_multi_label", False):
                stats["sample_types"]["multi_label"] += 1
            else:
                stats["sample_types"]["enhanced_original"] += 1
        
        # Convert sets to lists for JSON serialization
        stats["unique_label_combinations"] = list(stats["unique_label_combinations"])
        stats["num_unique_combinations"] = len(stats["unique_label_combinations"])
        
        # Convert defaultdicts to regular dicts
        stats["label_distribution"] = dict(stats["label_distribution"])
        stats["complexity_distribution"] = dict(stats["complexity_distribution"])
        stats["label_count_distribution"] = dict(stats["label_count_distribution"])
        stats["sample_types"] = dict(stats["sample_types"])
        
        return stats

def main():
    """Main function to demonstrate enhanced labeling system"""
    
    # Initialize system
    labeling_system = EnhancedLabelingSystem(seed=42)
    
    # Process dataset
    input_dir = "synthetic_dataset/splits"
    output_dir = "synthetic_dataset"
    
    stats = labeling_system.process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        distractor_ratio=0.15,    # 15% distractor samples
        multi_label_ratio=0.25    # 25% multi-label samples
    )
    
    # Print summary
    logger.info("\n=== ENHANCED LABELING SUMMARY ===")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Unique label combinations: {stats['num_unique_combinations']}")
    logger.info(f"Sample types: {stats['sample_types']}")
    logger.info(f"Complexity distribution: {stats['complexity_distribution']}")
    
    logger.info("\nTop 10 most common labels:")
    label_counts = [(label, count) for label, count in stats['label_distribution'].items()]
    label_counts.sort(key=lambda x: x[1], reverse=True)
    for label, count in label_counts[:10]:
        logger.info(f"  {label}: {count}")

if __name__ == "__main__":
    main() 