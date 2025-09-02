"""
Preamble Logic Validator

Validates business logic consistency in influencer agreement parameters.
Uses simple threshold rules for fast validation without AI/ML.
"""

from typing import Dict, List, Tuple, Any


def validate_preamble_logic(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate business logic consistency in preamble parameters.
    
    Args:
        params: Dictionary containing:
            - fee: Campaign fee amount (int/float)
            - deliverables_count: Number of deliverables (int)
            - industry: Industry category (str)
            - platforms: List of social media platforms (List[str])
            - campaign_length_months: Campaign duration in months (int/float)
            - exclusivity_months: Exclusivity period in months (int/float)
    
    Returns:
        Tuple of (is_valid: bool, issues: List[str])
    """
    
    issues = []
    
    # Extract parameters with defaults
    fee = params.get('fee', 0)
    deliverables_count = params.get('deliverables_count', 0)
    industry = params.get('industry', '').lower()
    platforms = params.get('platforms', [])
    campaign_length_months = params.get('campaign_length_months', 0)
    exclusivity_months = params.get('exclusivity_months', 0)
    
    # 1. Fee scale ↔ deliverable complexity validation
    issues.extend(_validate_fee_deliverable_match(fee, deliverables_count))
    
    # 2. Industry ↔ platform match validation
    issues.extend(_validate_industry_platform_match(industry, platforms))
    
    # 3. Campaign length ↔ exclusivity period validation
    issues.extend(_validate_campaign_exclusivity_match(campaign_length_months, exclusivity_months))
    
    # Return validation result
    is_valid = len(issues) == 0
    return is_valid, issues


def _validate_fee_deliverable_match(fee: float, deliverables_count: int) -> List[str]:
    """Validate fee scale matches deliverable complexity."""
    issues = []
    
    # Fee tier thresholds
    LOW_FEE_THRESHOLD = 5000
    MID_FEE_THRESHOLD = 15000
    HIGH_FEE_THRESHOLD = 35000
    
    # Deliverable count thresholds  
    LOW_DELIVERABLE_THRESHOLD = 2
    MID_DELIVERABLE_THRESHOLD = 5
    HIGH_DELIVERABLE_THRESHOLD = 10
    
    # Low fee validation
    if fee < LOW_FEE_THRESHOLD:
        if deliverables_count > MID_DELIVERABLE_THRESHOLD:
            issues.append(f"Low fee (${fee:,}) doesn't justify high deliverable count ({deliverables_count})")
    
    # Mid fee validation
    elif fee < MID_FEE_THRESHOLD:
        if deliverables_count > HIGH_DELIVERABLE_THRESHOLD:
            issues.append(f"Mid fee (${fee:,}) doesn't justify very high deliverable count ({deliverables_count})")
        elif deliverables_count < LOW_DELIVERABLE_THRESHOLD:
            issues.append(f"Mid fee (${fee:,}) too high for low deliverable count ({deliverables_count})")
    
    # High fee validation
    elif fee >= HIGH_FEE_THRESHOLD:
        if deliverables_count < MID_DELIVERABLE_THRESHOLD:
            issues.append(f"High fee (${fee:,}) not justified by low deliverable count ({deliverables_count})")
    
    # Very low deliverable count check
    if deliverables_count <= 0:
        issues.append("Deliverable count must be greater than 0")
    
    # Very high fee check
    if fee > 100000 and deliverables_count < HIGH_DELIVERABLE_THRESHOLD:
        issues.append(f"Extremely high fee (${fee:,}) requires extensive deliverables (>{HIGH_DELIVERABLE_THRESHOLD})")
    
    return issues


def _validate_industry_platform_match(industry: str, platforms: List[str]) -> List[str]:
    """Validate industry matches appropriate social media platforms."""
    issues = []
    
    if not platforms:
        issues.append("At least one platform must be specified")
        return issues
    
    # Convert platforms to lowercase for comparison
    platforms_lower = [p.lower() for p in platforms]
    
    # Industry-platform compatibility rules
    industry_platform_rules = {
        'fashion': {
            'preferred': ['instagram', 'tiktok', 'pinterest'],
            'acceptable': ['facebook', 'youtube', 'twitter'],
            'poor_fit': ['linkedin']
        },
        'food': {
            'preferred': ['instagram', 'tiktok', 'youtube'],
            'acceptable': ['facebook', 'pinterest', 'twitter'],
            'poor_fit': ['linkedin']
        },
        'tech': {
            'preferred': ['youtube', 'twitter', 'linkedin'],
            'acceptable': ['instagram', 'tiktok', 'facebook'],
            'poor_fit': ['pinterest']
        },
        'beauty': {
            'preferred': ['instagram', 'tiktok', 'youtube'],
            'acceptable': ['pinterest', 'facebook', 'twitter'],
            'poor_fit': ['linkedin']
        },
        'home': {
            'preferred': ['instagram', 'pinterest', 'youtube'],
            'acceptable': ['facebook', 'tiktok', 'twitter'],
            'poor_fit': ['linkedin']
        },
        'travel': {
            'preferred': ['instagram', 'youtube', 'tiktok'],
            'acceptable': ['facebook', 'pinterest', 'twitter'],
            'poor_fit': ['linkedin']
        },
        'finance': {
            'preferred': ['linkedin', 'youtube', 'twitter'],
            'acceptable': ['facebook', 'instagram'],
            'poor_fit': ['tiktok', 'pinterest']
        }
    }
    
    if industry in industry_platform_rules:
        rules = industry_platform_rules[industry]
        
        # Check for poor platform fits
        poor_platforms = [p for p in platforms_lower if p in rules['poor_fit']]
        if poor_platforms:
            issues.append(f"Platform(s) {poor_platforms} are poor fit for {industry} industry")
        
        # Check if all platforms are suboptimal
        preferred_platforms = [p for p in platforms_lower if p in rules['preferred']]
        acceptable_platforms = [p for p in platforms_lower if p in rules['acceptable']]
        
        if not preferred_platforms and not acceptable_platforms:
            issues.append(f"No suitable platforms for {industry} industry. Consider: {rules['preferred']}")
    
    # Platform-specific validation
    if 'linkedin' in platforms_lower and industry not in ['tech', 'finance', 'business']:
        issues.append("LinkedIn typically not effective for consumer-focused industries")
    
    if 'tiktok' in platforms_lower and industry == 'finance':
        issues.append("TikTok generally not suitable for financial services content")
    
    return issues


def _validate_campaign_exclusivity_match(campaign_length_months: float, exclusivity_months: float) -> List[str]:
    """Validate campaign length matches exclusivity period expectations."""
    issues = []
    
    if campaign_length_months <= 0:
        issues.append("Campaign length must be greater than 0 months")
        return issues
    
    if exclusivity_months < 0:
        issues.append("Exclusivity period cannot be negative")
        return issues
    
    # Basic exclusivity validation rules
    
    # 1. Exclusivity should generally be longer than campaign
    if exclusivity_months < campaign_length_months:
        issues.append(f"Exclusivity period ({exclusivity_months} months) shorter than campaign ({campaign_length_months} months)")
    
    # 2. Very short campaigns shouldn't have very long exclusivity
    if campaign_length_months <= 1 and exclusivity_months > 6:
        issues.append(f"Short campaign ({campaign_length_months} months) with excessive exclusivity ({exclusivity_months} months)")
    
    # 3. Long campaigns should have reasonable exclusivity
    if campaign_length_months >= 6 and exclusivity_months > campaign_length_months * 3:
        issues.append(f"Exclusivity period ({exclusivity_months} months) too long for campaign length ({campaign_length_months} months)")
    
    # 4. Very long exclusivity periods are generally unreasonable
    if exclusivity_months > 24:
        issues.append(f"Exclusivity period ({exclusivity_months} months) exceeds reasonable maximum (24 months)")
    
    # 5. Industry-specific exclusivity expectations
    if campaign_length_months <= 2 and exclusivity_months > 12:
        issues.append("Short-term campaigns shouldn't require more than 12 months exclusivity")
    
    return issues


def main():
    """Test the preamble logic validator with various scenarios."""
    
    print("🎯 Preamble Logic Validator Test")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "Valid Fashion Campaign",
            "params": {
                "fee": 8000,
                "deliverables_count": 4,
                "industry": "fashion",
                "platforms": ["instagram", "tiktok"],
                "campaign_length_months": 2,
                "exclusivity_months": 6
            }
        },
        {
            "name": "Invalid: Low Fee, High Deliverables",
            "params": {
                "fee": 3000,
                "deliverables_count": 8,
                "industry": "fashion",
                "platforms": ["instagram"],
                "campaign_length_months": 1,
                "exclusivity_months": 3
            }
        },
        {
            "name": "Invalid: Tech on TikTok Only",
            "params": {
                "fee": 12000,
                "deliverables_count": 3,
                "industry": "tech",
                "platforms": ["tiktok"],
                "campaign_length_months": 3,
                "exclusivity_months": 9
            }
        },
        {
            "name": "Invalid: Short Campaign, Long Exclusivity",
            "params": {
                "fee": 15000,
                "deliverables_count": 5,
                "industry": "beauty",
                "platforms": ["instagram", "youtube"],
                "campaign_length_months": 1,
                "exclusivity_months": 18
            }
        },
        {
            "name": "Valid Tech Campaign",
            "params": {
                "fee": 20000,
                "deliverables_count": 6,
                "industry": "tech",
                "platforms": ["youtube", "linkedin"],
                "campaign_length_months": 4,
                "exclusivity_months": 8
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"   Parameters: {test_case['params']}")
        
        is_valid, issues = validate_preamble_logic(test_case['params'])
        
        if is_valid:
            print("   ✅ VALID - No issues found")
        else:
            print(f"   ❌ INVALID - {len(issues)} issue(s):")
            for issue in issues:
                print(f"      - {issue}")
    
    print(f"\n✅ Validation tests complete!")
    print(f"💡 Function returns (is_valid: bool, issues: List[str])")
    print(f"   - Fast rule-based validation")
    print(f"   - No AI/ML dependencies") 
    print(f"   - Covers fee/deliverable, industry/platform, campaign/exclusivity logic")


if __name__ == "__main__":
    main() 