#!/usr/bin/env python3
"""
Dataset Validation and Quality Control for Thinkerbell Synthetic Dataset
Ensures production-ready quality with distribution balance, human validation, and benchmark testing
"""

import json
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

@dataclass
class ValidationResult:
    """Result of dataset validation"""
    passed: bool
    score: float
    issues: List[str]
    recommendations: List[str]

class DatasetValidator:
    """Comprehensive dataset validation and quality control"""
    
    def __init__(self):
        # Distribution targets for balanced dataset
        self.distribution_targets = {
            "fee_ranges": {
                "low": (1500, 8000, 0.4),      # (min, max, target_ratio)
                "mid": (8000, 18000, 0.4),
                "premium": (18000, 35000, 0.2)
            },
            "complexity": {
                "simple": 0.5,
                "medium": 0.3, 
                "complex": 0.2
            },
            "industries": {
                "fashion": 0.25,
                "food": 0.25,
                "tech": 0.2,
                "home": 0.15,
                "beauty": 0.1,
                "automotive": 0.05
            },
            "text_styles": {
                "formal": 0.3,
                "casual": 0.4,
                "bullet_points": 0.3
            },
            "classification": {
                "INFLUENCER_AGREEMENT": 0.8,
                "NOT_INFLUENCER_AGREEMENT": 0.2
            }
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "human_extraction_accuracy": 0.85,
            "semantic_coherence_min": 0.6,
            "token_length_max": 512,
            "distribution_balance_min": 0.7,
            "max_weird_samples_percent": 0.1
        }
        
        # Track generation metrics
        self.generation_metrics = {
            "total_generated": 0,
            "validation_passed": 0,
            "distribution_scores": {},
            "quality_scores": {},
            "issues_found": []
        }
    
    def validate_distribution_balance(self, dataset: List[Dict]) -> Dict[str, float]:
        """Check if dataset distribution matches targets"""
        
        scores = {}
        
        # Check fee range distribution
        fee_ranges = []
        for item in dataset:
            fee = item.get("extracted_fields", {}).get("fee_numeric", 0)
            if fee > 0:
                if fee <= 8000:
                    fee_ranges.append("low")
                elif fee <= 18000:
                    fee_ranges.append("mid")
                else:
                    fee_ranges.append("premium")
        
        if fee_ranges:
            fee_dist = Counter(fee_ranges)
            total = len(fee_ranges)
            fee_balance_score = 0
            for range_name, target_ratio in self.distribution_targets["fee_ranges"].items():
                actual_ratio = fee_dist.get(range_name, 0) / total
                balance = 1 - abs(actual_ratio - target_ratio[2])
                fee_balance_score += balance
            scores["fee_balance"] = fee_balance_score / len(self.distribution_targets["fee_ranges"])
        else:
            scores["fee_balance"] = 0.0
        
        # Check complexity distribution
        complexity_dist = Counter(item.get("complexity_level", "medium") for item in dataset)
        total = len(dataset)
        complexity_balance_score = 0
        for complexity, target_ratio in self.distribution_targets["complexity"].items():
            actual_ratio = complexity_dist.get(complexity, 0) / total
            balance = 1 - abs(actual_ratio - target_ratio)
            complexity_balance_score += balance
        scores["complexity_balance"] = complexity_balance_score / len(self.distribution_targets["complexity"])
        
        # Check classification distribution
        classification_dist = Counter(item.get("classification", "UNKNOWN") for item in dataset)
        total = len(dataset)
        classification_balance_score = 0
        for classification, target_ratio in self.distribution_targets["classification"].items():
            actual_ratio = classification_dist.get(classification, 0) / total
            balance = 1 - abs(actual_ratio - target_ratio)
            classification_balance_score += balance
        scores["classification_balance"] = classification_balance_score / len(self.distribution_targets["classification"])
        
        # Overall distribution score
        scores["overall_balance"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def validate_human_extractability(self, dataset: List[Dict], sample_size: int = 50) -> Tuple[float, List[str]]:
        """Test if humans can reliably extract fields from the text"""
        
        # Sample dataset for human validation
        sample = random.sample(dataset, min(sample_size, len(dataset)))
        
        extraction_issues = []
        extractable_count = 0
        
        for item in sample:
            text = item.get("raw_input_text", "")
            expected_fields = item.get("extracted_fields", {})
            
            # Check if key fields are mentioned in text
            missing_fields = []
            
            # Check influencer name
            influencer = expected_fields.get("influencer", "")
            if influencer and influencer not in text and influencer != "TBC":
                missing_fields.append("influencer")
            
            # Check fee
            fee = expected_fields.get("fee", "")
            if fee and "$" not in text and "budget" not in text.lower():
                missing_fields.append("fee")
            
            # Check deliverables
            deliverables = expected_fields.get("deliverables", [])
            if deliverables and not any(deliverable.lower() in text.lower() for deliverable in deliverables):
                missing_fields.append("deliverables")
            
            # Check brand/client
            brand = expected_fields.get("brand", "")
            if brand and brand not in text:
                missing_fields.append("brand")
            
            if missing_fields:
                extraction_issues.append(f"Missing fields: {missing_fields}")
            else:
                extractable_count += 1
        
        accuracy = extractable_count / len(sample)
        return accuracy, extraction_issues
    
    def validate_semantic_coherence(self, dataset: List[Dict], sample_size: int = 50) -> Tuple[float, List[str]]:
        """Check semantic coherence of generated samples"""
        
        sample = random.sample(dataset, min(sample_size, len(dataset)))
        
        coherence_issues = []
        coherent_count = 0
        
        for item in sample:
            text = item.get("raw_input_text", "")
            extracted_fields = item.get("extracted_fields", {})
            
            # Check for business logic violations
            issues = []
            
            # Check fee-deliverable coherence
            fee = extracted_fields.get("fee_numeric", 0)
            deliverables = extracted_fields.get("deliverables", [])
            
            if fee > 20000 and len(deliverables) < 3:
                issues.append("High fee for few deliverables")
            
            if fee < 3000 and len(deliverables) > 5:
                issues.append("Low fee for many deliverables")
            
            # Check for obvious contradictions
            if "employment" in text.lower() and "influencer" in text.lower():
                issues.append("Employment language in influencer agreement")
            
            if "salary" in text.lower() and "fee" not in text.lower():
                issues.append("Salary language instead of fee")
            
            # Check for realistic brand-deliverable combinations
            brand = extracted_fields.get("brand", "")
            if brand in ["Woolworths", "Coles"] and "fashion" in str(deliverables).lower():
                issues.append("Grocery brand with fashion deliverables")
            
            if issues:
                coherence_issues.extend(issues)
            else:
                coherent_count += 1
        
        coherence_score = coherent_count / len(sample)
        return coherence_score, coherence_issues
    
    def validate_token_lengths(self, dataset: List[Dict]) -> Tuple[float, List[str]]:
        """Check token length distribution"""
        
        token_lengths = []
        length_issues = []
        
        for item in dataset:
            text = item.get("raw_input_text", "")
            # Rough token estimation (4 chars per token)
            estimated_tokens = len(text) / 4
            
            token_lengths.append(estimated_tokens)
            
            if estimated_tokens > self.quality_thresholds["token_length_max"]:
                length_issues.append(f"Text too long: {estimated_tokens:.0f} tokens")
        
        avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        max_length = max(token_lengths) if token_lengths else 0
        
        # Score based on how many are within limits
        within_limits = sum(1 for length in token_lengths if length <= self.quality_thresholds["token_length_max"])
        length_score = within_limits / len(token_lengths) if token_lengths else 0
        
        return length_score, length_issues
    
    def generate_validation_sample(self, dataset: List[Dict], size: int = 100) -> List[Dict]:
        """Generate a sample for manual human validation"""
        
        # Ensure diverse sample
        validation_sample = []
        
        # Get samples from each classification
        classifications = Counter(item.get("classification", "UNKNOWN") for item in dataset)
        
        for classification, count in classifications.items():
            classification_items = [item for item in dataset if item.get("classification") == classification]
            sample_size = min(size // len(classifications), len(classification_items))
            validation_sample.extend(random.sample(classification_items, sample_size))
        
        # Add some edge cases if available
        edge_cases = [item for item in dataset if item.get("metadata", {}).get("sample_type") == "edge_case"]
        if edge_cases:
            edge_sample_size = min(10, len(edge_cases))
            validation_sample.extend(random.sample(edge_cases, edge_sample_size))
        
        # Add some OOD samples if available
        ood_samples = [item for item in dataset if item.get("metadata", {}).get("sample_type") == "ood_negative"]
        if ood_samples:
            ood_sample_size = min(10, len(ood_samples))
            validation_sample.extend(random.sample(ood_samples, ood_sample_size))
        
        return validation_sample
    
    def generate_benchmark_test_set(self, dataset: List[Dict], test_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        """Generate benchmark test set with extra messy/realistic samples"""
        
        # Split dataset
        test_size = int(len(dataset) * test_ratio)
        test_indices = random.sample(range(len(dataset)), test_size)
        
        test_set = [dataset[i] for i in test_indices]
        train_set = [dataset[i] for i in range(len(dataset)) if i not in test_indices]
        
        # Make test set extra messy
        for item in test_set:
            text = item.get("raw_input_text", "")
            
            # Add some typos
            if random.random() < 0.2:
                text = text.replace("influencer", "influencr")
                text = text.replace("agreement", "agreemnt")
            
            # Add incomplete information
            if random.random() < 0.3:
                text += "\n\nAdditional notes: TBC on exact dates"
            
            # Add mixed formatting
            if random.random() < 0.15:
                text = text.replace(".", ".\n")
            
            item["raw_input_text"] = text
            item["metadata"]["test_set_messy"] = True
        
        return train_set, test_set
    
    def create_dataset_metadata(self, dataset: List[Dict], generator_config: Dict) -> Dict:
        """Create comprehensive dataset metadata with version control"""
        
        # Calculate statistics
        total_samples = len(dataset)
        classifications = Counter(item.get("classification", "UNKNOWN") for item in dataset)
        complexity_levels = Counter(item.get("complexity_level", "medium") for item in dataset)
        
        # Fee statistics
        fees = [item.get("extracted_fields", {}).get("fee_numeric", 0) for item in dataset]
        fees = [f for f in fees if f > 0]
        
        metadata = {
            "generator_version": "v1.0",
            "generation_date": datetime.now().isoformat(),
            "total_samples": total_samples,
            "generator_config": generator_config,
            "distribution_targets": self.distribution_targets,
            "quality_thresholds": self.quality_thresholds,
            "statistics": {
                "classification_distribution": dict(classifications),
                "complexity_distribution": dict(complexity_levels),
                "fee_statistics": {
                    "min": min(fees) if fees else 0,
                    "max": max(fees) if fees else 0,
                    "mean": sum(fees) / len(fees) if fees else 0,
                    "median": sorted(fees)[len(fees) // 2] if fees else 0
                }
            },
            "validation_scores": self.generation_metrics["quality_scores"],
            "issues_found": self.generation_metrics["issues_found"]
        }
        
        return metadata
    
    def comprehensive_validation(self, dataset: List[Dict], generator_config: Dict) -> ValidationResult:
        """Run comprehensive validation on dataset"""
        
        issues = []
        recommendations = []
        
        print("🔍 Running comprehensive dataset validation...")
        
        # 1. Distribution balance validation
        print("  📊 Checking distribution balance...")
        distribution_scores = self.validate_distribution_balance(dataset)
        self.generation_metrics["distribution_scores"] = distribution_scores
        
        overall_balance = distribution_scores["overall_balance"]
        if overall_balance < self.quality_thresholds["distribution_balance_min"]:
            issues.append(f"Poor distribution balance: {overall_balance:.3f}")
            recommendations.append("Adjust generator parameters to improve distribution balance")
        
        # 2. Human extractability validation
        print("  👤 Checking human extractability...")
        extraction_accuracy, extraction_issues = self.validate_human_extractability(dataset)
        self.generation_metrics["quality_scores"]["human_extraction"] = extraction_accuracy
        
        if extraction_accuracy < self.quality_thresholds["human_extraction_accuracy"]:
            issues.append(f"Poor human extractability: {extraction_accuracy:.3f}")
            recommendations.append("Simplify text generation or improve field mention consistency")
        
        issues.extend(extraction_issues)
        
        # 3. Semantic coherence validation
        print("  🧠 Checking semantic coherence...")
        coherence_score, coherence_issues = self.validate_semantic_coherence(dataset)
        self.generation_metrics["quality_scores"]["semantic_coherence"] = coherence_score
        
        if coherence_score < self.quality_thresholds["semantic_coherence_min"]:
            issues.append(f"Poor semantic coherence: {coherence_score:.3f}")
            recommendations.append("Improve semantic smoother or business logic validation")
        
        issues.extend(coherence_issues)
        
        # 4. Token length validation
        print("  📏 Checking token lengths...")
        length_score, length_issues = self.validate_token_lengths(dataset)
        self.generation_metrics["quality_scores"]["token_length"] = length_score
        
        if length_score < 0.9:  # 90% should be within limits
            issues.append(f"Too many long texts: {length_score:.3f}")
            recommendations.append("Implement better text chunking or reduce text length")
        
        issues.extend(length_issues)
        
        # 5. Overall quality assessment
        quality_scores = self.generation_metrics["quality_scores"]
        overall_score = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
        
        # Check for weird samples
        weird_sample_count = len([issue for issue in issues if "weird" in issue.lower()])
        weird_percentage = weird_sample_count / len(dataset) if dataset else 0
        
        if weird_percentage > self.quality_thresholds["max_weird_samples_percent"]:
            issues.append(f"Too many weird samples: {weird_percentage:.1%}")
            recommendations.append("Review and fix generator logic")
        
        # Determine if validation passed
        passed = (
            overall_balance >= self.quality_thresholds["distribution_balance_min"] and
            extraction_accuracy >= self.quality_thresholds["human_extraction_accuracy"] and
            coherence_score >= self.quality_thresholds["semantic_coherence_min"] and
            length_score >= 0.9 and
            weird_percentage <= self.quality_thresholds["max_weird_samples_percent"]
        )
        
        self.generation_metrics["validation_passed"] = 1 if passed else 0
        self.generation_metrics["issues_found"] = issues
        
        print(f"  ✅ Validation complete - Overall score: {overall_score:.3f}")
        print(f"  📊 Distribution balance: {overall_balance:.3f}")
        print(f"  👤 Human extractability: {extraction_accuracy:.3f}")
        print(f"  🧠 Semantic coherence: {coherence_score:.3f}")
        print(f"  📏 Token length compliance: {length_score:.3f}")
        
        if not passed:
            print(f"  ⚠️  Issues found: {len(issues)}")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"    - {issue}")
        
        return ValidationResult(
            passed=passed,
            score=overall_score,
            issues=issues,
            recommendations=recommendations
        )

def test_dataset_validation():
    """Test the dataset validation functionality"""
    
    print("🧪 Testing Dataset Validation")
    print("=" * 50)
    
    validator = DatasetValidator()
    
    # Test with sample data
    sample_dataset = [
        {
            "raw_input_text": "Need Sarah Chen for Woolworths campaign. Budget $5000. Instagram posts and stories.",
            "extracted_fields": {
                "influencer": "Sarah Chen",
                "brand": "Woolworths", 
                "fee_numeric": 5000,
                "deliverables": ["Instagram posts", "Instagram stories"]
            },
            "classification": "INFLUENCER_AGREEMENT",
            "complexity_level": "medium"
        },
        {
            "raw_input_text": "Position: Senior Developer. Salary: $95000 annually.",
            "extracted_fields": {},
            "classification": "NOT_INFLUENCER_AGREEMENT",
            "complexity_level": "simple"
        }
    ]
    
    # Test validation
    result = validator.comprehensive_validation(sample_dataset, {"test": True})
    
    print(f"\nValidation Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Issues: {len(result.issues)}")
    print(f"  Recommendations: {len(result.recommendations)}")
    
    # Test distribution balance
    distribution_scores = validator.validate_distribution_balance(sample_dataset)
    print(f"\nDistribution Scores:")
    for metric, score in distribution_scores.items():
        print(f"  {metric}: {score:.3f}")
    
    # Test human extractability
    accuracy, issues = validator.validate_human_extractability(sample_dataset)
    print(f"\nHuman Extractability:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Issues: {len(issues)}")

if __name__ == "__main__":
    test_dataset_validation() 