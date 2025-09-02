#!/usr/bin/env python3
"""
Complete Preamble Workflow Demo

Demonstrates the full pipeline:
1. Generate structured preamble
2. Validate business logic
3. Apply brand-specific tone
"""

from structured_preamble import generate_preamble
from preamble_validator import validate_preamble_logic
from preamble_tone import apply_preamble_tone, get_supported_brands

def complete_workflow_demo():
    """Demonstrate the complete preamble generation, validation, and tone workflow."""
    
    print("🎯 Complete Preamble Workflow: Generate → Validate → Apply Tone")
    print("=" * 70)
    
    # Example campaigns with different brands
    campaigns = [
        {
            "name": "Australia Post Service Campaign",
            "brand": "Australia Post",
            "generation_params": {
                "client": "Australia Post",
                "talent": "Emma Wilson",
                "agency": "Thinkerbell",
                "deliverables": ["Instagram posts", "community content"],
                "campaign_type": "Community Service Initiative",
                "start_date": "April 2024",
                "end_date": "June 2024"
            },
            "validation_params": {
                "fee": 15000,
                "deliverables_count": 2,
                "industry": "government",
                "platforms": ["instagram", "facebook"],
                "campaign_length_months": 2,
                "exclusivity_months": 6
            }
        },
        {
            "name": "Rexona Performance Campaign",
            "brand": "Rexona",
            "generation_params": {
                "client": "Rexona",
                "talent": "Jake Thompson",
                "agency": "Ogilvy Sydney",
                "deliverables": ["TikTok videos", "Instagram reels", "workout content"],
                "campaign_type": "Athletic Performance Challenge",
                "start_date": "March 2024",
                "end_date": "May 2024"
            },
            "validation_params": {
                "fee": 18000,
                "deliverables_count": 3,
                "industry": "beauty",
                "platforms": ["tiktok", "instagram"],
                "campaign_length_months": 2,
                "exclusivity_months": 8
            }
        },
        {
            "name": "Queen Fine Foods Recipe Series",
            "brand": "Queen Fine Foods",
            "generation_params": {
                "client": "Queen Fine Foods",
                "talent": "Sofia Martinez",
                "agency": "M&C Saatchi",
                "deliverables": ["Instagram posts", "recipe videos", "blog content"],
                "campaign_type": "Seasonal Baking Collection",
                "start_date": "February 2024",
                "end_date": "April 2024"
            },
            "validation_params": {
                "fee": 12000,
                "deliverables_count": 3,
                "industry": "food",
                "platforms": ["instagram", "youtube"],
                "campaign_length_months": 2,
                "exclusivity_months": 6
            }
        }
    ]
    
    for i, campaign in enumerate(campaigns, 1):
        print(f"\n📋 Campaign {i}: {campaign['name']}")
        print("=" * 50)
        
        # Step 1: Generate base preamble
        print("🔧 Step 1: Generate Structured Preamble")
        base_preamble = generate_preamble(**campaign['generation_params'])
        
        print(f"   Generated preamble:")
        sections = base_preamble.split('\n\n')
        for j, section in enumerate(sections, 1):
            print(f"     {j}. {section[:80]}...")
        
        print(f"   Base length: {len(base_preamble)} characters")
        
        # Step 2: Validate business logic
        print(f"\n🔍 Step 2: Validate Business Logic")
        is_valid, issues = validate_preamble_logic(campaign['validation_params'])
        
        if is_valid:
            print("   ✅ VALID - Business logic checks passed")
        else:
            print(f"   ❌ INVALID - {len(issues)} issue(s):")
            for issue in issues:
                print(f"      • {issue}")
        
        # Step 3: Apply brand tone (if validation passed or force demo)
        print(f"\n🎨 Step 3: Apply Brand Tone ({campaign['brand']})")
        
        if campaign['brand'] in get_supported_brands():
            tone_preamble = apply_preamble_tone(base_preamble, campaign['brand'])
            
            print("   ✅ Brand tone applied")
            print(f"   Tone-adjusted length: {len(tone_preamble)} characters")
            
            # Show tone transformation example
            base_words = set(base_preamble.lower().split())
            tone_words = set(tone_preamble.lower().split())
            new_words = tone_words - base_words
            
            if new_words:
                sample_new_words = list(new_words)[:5]
                print(f"   New brand-specific terms: {', '.join(sample_new_words)}")
            
            # Show before/after comparison for first sentence
            base_first = base_preamble.split('.')[0] + '.'
            tone_first = tone_preamble.split('.')[0] + '.'
            
            if base_first != tone_first:
                print(f"\n   Before: {base_first}")
                print(f"   After:  {tone_first}")
        else:
            print(f"   ⚠️ No tone rules defined for {campaign['brand']}")
            tone_preamble = base_preamble
        
        # Step 4: Final assessment
        print(f"\n📊 Step 4: Final Assessment")
        
        if is_valid and campaign['brand'] in get_supported_brands():
            print("   🎉 COMPLETE: Preamble generated, validated, and tone-adjusted")
            print("   📝 Ready for production use")
        elif is_valid:
            print("   ✅ VALID: Preamble generated and validated (generic tone)")
            print("   💡 Consider adding tone rules for brand consistency")
        else:
            print("   ⚠️ REQUIRES REVIEW: Business logic issues need resolution")
            print("   🔧 Adjust parameters and regenerate")
        
        print()
    
    # Summary of capabilities
    print("=" * 70)
    print("✅ Complete Workflow Demo Finished!")
    print()
    print("🔧 Available Functions:")
    print("   • generate_preamble() - Structured 4-section preambles")
    print("   • validate_preamble_logic() - Business logic validation")
    print("   • apply_preamble_tone() - Brand-specific tone adaptation")
    print()
    print(f"🎨 Supported Brand Tones: {', '.join(get_supported_brands())}")
    print()
    print("💡 Expansion Points:")
    print("   • Add new brand tone rules in preamble_tone.py")
    print("   • Extend validation rules in preamble_validator.py")
    print("   • Add new template sections in structured_preamble.py")


def tone_comparison_demo():
    """Show side-by-side tone comparisons for the same preamble."""
    
    print("\n" + "=" * 50)
    print("🎨 Tone Comparison Demo")
    print("=" * 50)
    
    # Generate a base preamble
    base_preamble = generate_preamble(
        client="Sample Brand",
        talent="Alex Chen",
        agency="Creative Agency",
        deliverables=["Instagram posts", "stories"],
        campaign_type="Product Launch",
        start_date="March 2024",
        end_date="May 2024"
    )
    
    print("📝 Base Preamble (first 200 chars):")
    print(f"   {base_preamble[:200]}...")
    print()
    
    # Apply different tones
    supported_brands = get_supported_brands()
    
    for brand in supported_brands:
        print(f"🎨 {brand} Tone:")
        
        # Replace brand name in preamble
        brand_preamble = base_preamble.replace("Sample Brand", brand)
        tone_preamble = apply_preamble_tone(brand_preamble, brand)
        
        # Show first sentence transformation
        base_first = brand_preamble.split('.')[0] + '.'
        tone_first = tone_preamble.split('.')[0] + '.'
        
        print(f"   Original: {base_first}")
        print(f"   Adapted:  {tone_first}")
        
        if base_first != tone_first:
            print("   ✅ Tone transformation applied")
        else:
            print("   ➡️ No changes in first sentence")
        print()


if __name__ == "__main__":
    complete_workflow_demo()
    tone_comparison_demo() 