"""
Unified Business Preamble Generator

Orchestrates the complete preamble workflow:
1. Generate structured anchor paragraphs
2. Validate business logic coherence
3. Apply brand-specific tone adaptation
4. Return final preamble with validation metadata
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime

try:
    # Package imports
    from .structured_preamble import generate_preamble
    from .preamble_validator import validate_preamble_logic
    from .preamble_tone import apply_preamble_tone, get_supported_brands
except ImportError:
    # Standalone imports
    from structured_preamble import generate_preamble
    from preamble_validator import validate_preamble_logic
    from preamble_tone import apply_preamble_tone, get_supported_brands

# Setup module logger
logger = logging.getLogger(__name__)


class PreambleGenerationResult:
    """Result container for preamble generation with metadata."""
    
    def __init__(self, 
                 preamble: str,
                 validation_passed: bool,
                 validation_issues: List[str],
                 tone_applied: bool,
                 generation_attempts: int,
                 brand_supported: bool):
        self.preamble = preamble
        self.validation_passed = validation_passed
        self.validation_issues = validation_issues
        self.tone_applied = tone_applied
        self.generation_attempts = generation_attempts
        self.brand_supported = brand_supported
        self.generated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "preamble": self.preamble,
            "validation_passed": self.validation_passed,
            "validation_issues": self.validation_issues,
            "tone_applied": self.tone_applied,
            "generation_attempts": self.generation_attempts,
            "brand_supported": self.brand_supported,
            "generated_at": self.generated_at
        }


def generate_business_preamble(
    # Generation parameters
    client: str,
    talent: str,
    agency: str,
    deliverables: List[str],
    campaign_type: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    
    # Validation parameters
    fee: float,
    industry: str,
    platforms: List[str],
    campaign_length_months: float,
    exclusivity_months: float,
    
    # Optional parameters
    brand_name: Optional[str] = None,
    max_validation_attempts: int = 3,
    auto_adjust_params: bool = True,
    
    # Modular function overrides (for testing/customization)
    generator_func: Optional[Callable] = None,
    validator_func: Optional[Callable] = None,
    tone_func: Optional[Callable] = None
    
) -> PreambleGenerationResult:
    """
    Generate a complete business preamble with validation and tone adaptation.
    
    Args:
        # Core generation parameters
        client: Client/brand name
        talent: Talent/influencer name
        agency: Agency name
        deliverables: List of deliverable items
        campaign_type: Type of campaign
        start_date: Campaign start date
        end_date: Campaign end date
        
        # Validation parameters
        fee: Campaign fee amount
        industry: Industry category
        platforms: List of social media platforms
        campaign_length_months: Campaign duration in months
        exclusivity_months: Exclusivity period in months
        
        # Optional parameters
        brand_name: Brand name for tone application (defaults to client)
        max_validation_attempts: Maximum attempts to fix validation issues
        auto_adjust_params: Whether to automatically adjust parameters on validation failure
        
        # Modular overrides (for testing/customization)
        generator_func: Custom generation function
        validator_func: Custom validation function  
        tone_func: Custom tone application function
        
    Returns:
        PreambleGenerationResult with final preamble and metadata
    """
    
    logger.info(f"Starting business preamble generation for {client}")
    
    # Use provided functions or defaults
    gen_func = generator_func or generate_preamble
    val_func = validator_func or validate_preamble_logic
    tone_func_final = tone_func or apply_preamble_tone
    
    # Determine brand for tone application
    brand_for_tone = brand_name or client
    brand_supported = brand_for_tone in get_supported_brands()
    
    # Prepare validation parameters
    validation_params = {
        "fee": fee,
        "deliverables_count": len(deliverables),
        "industry": industry,
        "platforms": platforms,
        "campaign_length_months": campaign_length_months,
        "exclusivity_months": exclusivity_months
    }
    
    generation_attempts = 0
    final_validation_issues = []
    
    # Generation and validation loop
    while generation_attempts < max_validation_attempts:
        generation_attempts += 1
        
        logger.debug(f"Generation attempt {generation_attempts}")
        
        # Step 1: Generate structured preamble
        try:
            preamble = gen_func(
                client=client,
                talent=talent,
                agency=agency,
                deliverables=deliverables,
                campaign_type=campaign_type,
                start_date=start_date,
                end_date=end_date
            )
            logger.debug("Preamble generation successful")
            
        except Exception as e:
            logger.error(f"Preamble generation failed: {str(e)}")
            return PreambleGenerationResult(
                preamble="",
                validation_passed=False,
                validation_issues=[f"Generation failed: {str(e)}"],
                tone_applied=False,
                generation_attempts=generation_attempts,
                brand_supported=brand_supported
            )
        
        # Step 2: Validate business logic
        try:
            is_valid, validation_issues = val_func(validation_params)
            final_validation_issues = validation_issues.copy()
            
            if is_valid:
                logger.debug("Validation passed")
                break
            else:
                logger.warning(f"Validation failed (attempt {generation_attempts}): {validation_issues}")
                
                # Try to auto-adjust parameters if enabled and not final attempt
                if auto_adjust_params and generation_attempts < max_validation_attempts:
                    adjusted_params = _auto_adjust_validation_params(validation_params, validation_issues)
                    if adjusted_params != validation_params:
                        logger.info("Auto-adjusting parameters for next attempt")
                        validation_params.update(adjusted_params)
                        # Update derived values
                        fee = adjusted_params.get("fee", fee)
                        exclusivity_months = adjusted_params.get("exclusivity_months", exclusivity_months)
                        # Note: We can't easily adjust deliverables/platforms without changing the preamble
                        # so we primarily adjust numeric parameters
                    else:
                        logger.debug("No automatic adjustments possible")
                        break
                else:
                    logger.debug("Auto-adjustment disabled or final attempt")
                    break
                    
        except Exception as e:
            logger.error(f"Validation failed with error: {str(e)}")
            final_validation_issues = [f"Validation error: {str(e)}"]
            is_valid = False
            break
    
    # Step 3: Apply tone adaptation (regardless of validation status for demo purposes)
    tone_applied = False
    final_preamble = preamble
    
    if brand_supported:
        try:
            tone_preamble = tone_func_final(preamble, brand_for_tone)
            if tone_preamble != preamble:
                final_preamble = tone_preamble
                tone_applied = True
                logger.debug(f"Tone applied for {brand_for_tone}")
            else:
                logger.debug(f"No tone changes applied for {brand_for_tone}")
                
        except Exception as e:
            logger.error(f"Tone application failed: {str(e)}")
            # Continue with original preamble
            final_validation_issues.append(f"Tone application failed: {str(e)}")
    else:
        logger.debug(f"Brand {brand_for_tone} not supported for tone adaptation")
    
    # Step 4: Return results
    result = PreambleGenerationResult(
        preamble=final_preamble,
        validation_passed=is_valid,
        validation_issues=final_validation_issues,
        tone_applied=tone_applied,
        generation_attempts=generation_attempts,
        brand_supported=brand_supported
    )
    
    logger.info(f"Preamble generation complete: validation_passed={is_valid}, tone_applied={tone_applied}")
    
    return result


def _auto_adjust_validation_params(params: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
    """
    Attempt to automatically adjust parameters to fix validation issues.
    
    Args:
        params: Current validation parameters
        issues: List of validation issues
        
    Returns:
        Dictionary of adjusted parameters
    """
    
    adjusted = {}
    
    for issue in issues:
        issue_lower = issue.lower()
        
        # Fee adjustment logic
        if "low fee" in issue_lower and "high deliverable count" in issue_lower:
            # Increase fee to match deliverable count
            current_fee = params.get("fee", 0)
            deliverable_count = params.get("deliverables_count", 0)
            
            if deliverable_count > 5:
                suggested_fee = max(15000, deliverable_count * 2500)
            else:
                suggested_fee = max(8000, deliverable_count * 2000)
                
            adjusted["fee"] = suggested_fee
            logger.debug(f"Adjusted fee from {current_fee} to {suggested_fee}")
            
        elif "high fee" in issue_lower and "low deliverable count" in issue_lower:
            # This is harder to auto-fix as it requires changing deliverables
            # We can reduce fee instead
            current_fee = params.get("fee", 0)
            deliverable_count = params.get("deliverables_count", 0)
            
            suggested_fee = min(current_fee * 0.7, deliverable_count * 4000)
            adjusted["fee"] = suggested_fee
            logger.debug(f"Adjusted fee from {current_fee} to {suggested_fee}")
        
        # Exclusivity adjustment logic
        if "excessive exclusivity" in issue_lower:
            current_exclusivity = params.get("exclusivity_months", 0)
            campaign_length = params.get("campaign_length_months", 1)
            
            # Set exclusivity to 3x campaign length, max 12 months
            suggested_exclusivity = min(12, campaign_length * 3)
            adjusted["exclusivity_months"] = suggested_exclusivity
            logger.debug(f"Adjusted exclusivity from {current_exclusivity} to {suggested_exclusivity}")
            
        elif "exclusivity period" in issue_lower and "shorter than campaign" in issue_lower:
            campaign_length = params.get("campaign_length_months", 1)
            suggested_exclusivity = campaign_length * 2
            adjusted["exclusivity_months"] = suggested_exclusivity
            logger.debug(f"Adjusted exclusivity to {suggested_exclusivity} months")
    
    return adjusted


def main():
    """Test the unified business preamble generator."""
    
    # Setup logging for demo
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🎯 Unified Business Preamble Generator Test")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Valid Australia Post Campaign",
            "params": {
                "client": "Australia Post",
                "talent": "Emma Wilson",
                "agency": "Thinkerbell",
                "deliverables": ["Instagram posts", "community content"],
                "campaign_type": "Community Service Initiative",
                "start_date": "April 2024",
                "end_date": "June 2024",
                "fee": 15000,
                "industry": "government",
                "platforms": ["instagram", "facebook"],
                "campaign_length_months": 2,
                "exclusivity_months": 6
            }
        },
        {
            "name": "Invalid Campaign (Auto-fix Enabled)",
            "params": {
                "client": "Rexona",
                "talent": "Jake Thompson",
                "agency": "Ogilvy Sydney",
                "deliverables": ["TikTok videos", "Instagram reels", "workout content", "blog posts", "photos", "reviews"],
                "campaign_type": "Athletic Performance Challenge",
                "start_date": "March 2024",
                "end_date": "May 2024",
                "fee": 3000,  # Too low for 6 deliverables
                "industry": "beauty",
                "platforms": ["tiktok", "instagram"],
                "campaign_length_months": 1,
                "exclusivity_months": 18,  # Too long
                "auto_adjust_params": True
            }
        },
        {
            "name": "Unsupported Brand (No Tone)",
            "params": {
                "client": "Generic Brand",
                "talent": "Sofia Martinez",
                "agency": "Creative Agency",
                "deliverables": ["Instagram posts", "stories"],
                "campaign_type": "Product Launch",
                "start_date": "February 2024",
                "end_date": "April 2024",
                "fee": 10000,
                "industry": "tech",
                "platforms": ["instagram"],
                "campaign_length_months": 2,
                "exclusivity_months": 6
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Test {i}: {scenario['name']}")
        print("-" * 40)
        
        # Generate business preamble
        result = generate_business_preamble(**scenario['params'])
        
        # Show results
        print(f"✅ Generation completed in {result.generation_attempts} attempt(s)")
        print(f"📊 Validation passed: {'✅' if result.validation_passed else '❌'}")
        print(f"🎨 Tone applied: {'✅' if result.tone_applied else '❌'}")
        print(f"🏢 Brand supported: {'✅' if result.brand_supported else '❌'}")
        
        if result.validation_issues:
            print(f"⚠️  Validation issues ({len(result.validation_issues)}):")
            for issue in result.validation_issues:
                print(f"   • {issue}")
        
        print(f"📄 Preamble length: {len(result.preamble)} characters")
        print(f"📅 Generated at: {result.generated_at}")
        
        # Show first section of preamble
        if result.preamble:
            first_section = result.preamble.split('\n\n')[0]
            print(f"📝 First section: {first_section[:100]}...")
    
    print(f"\n✅ All tests completed!")
    print(f"💡 Key features demonstrated:")
    print(f"   • Unified workflow orchestration")
    print(f"   • Automatic parameter adjustment")
    print(f"   • Comprehensive error handling")
    print(f"   • Modular component design")
    print(f"   • Detailed result metadata")


if __name__ == "__main__":
    main() 