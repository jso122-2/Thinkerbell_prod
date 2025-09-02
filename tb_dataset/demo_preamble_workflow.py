#!/usr/bin/env python3
"""
Demo: Complete Preamble Generation and Validation Workflow

Shows how to use generate_preamble() and validate_preamble_logic() together
for creating and validating business-consistent influencer agreement preambles.
"""

from structured_preamble import generate_preamble
from preamble_validator import validate_preamble_logic

def demo_workflow():
    """Demonstrate complete preamble generation and validation workflow."""
    
    print("🎯 Complete Preamble Generation & Validation Workflow")
    print("=" * 65)
    
    # Example campaign parameters
    campaign_scenarios = [
        {
            "name": "Valid Fashion Campaign",
            "generation_params": {
                "client": "David Jones",
                "talent": "Sarah Chen",
                "agency": "Thinkerbell", 
                "deliverables": ["Instagram posts", "Instagram stories", "product photography"],
                "campaign_type": "Spring Collection Launch",
                "start_date": "March 2024",
                "end_date": "May 2024"
            },
            "validation_params": {
                "fee": 12000,
                "deliverables_count": 3,
                "industry": "fashion",
                "platforms": ["instagram"],
                "campaign_length_months": 2,
                "exclusivity_months": 6
            }
        },
        {
            "name": "Invalid Tech Campaign (Platform Mismatch)",
            "generation_params": {
                "client": "JB Hi-Fi",
                "talent": "Michael Rodriguez",
                "agency": "Ogilvy Sydney",
                "deliverables": ["TikTok videos", "Instagram reels"],
                "campaign_type": "Gaming Hardware Review",
                "start_date": "June 2024", 
                "end_date": "July 2024"
            },
            "validation_params": {
                "fee": 8000,
                "deliverables_count": 2,
                "industry": "tech",
                "platforms": ["tiktok"],  # Poor fit for tech
                "campaign_length_months": 1,
                "exclusivity_months": 4
            }
        },
        {
            "name": "Invalid Budget Campaign (Fee/Deliverable Mismatch)",
            "generation_params": {
                "client": "Woolworths",
                "talent": "Emma Wilson",
                "agency": "M&C Saatchi",
                "deliverables": ["Instagram posts", "TikTok videos", "YouTube videos", "blog posts", "product photography", "recipe videos"],
                "campaign_type": "Fresh Food Initiative",
                "start_date": "April 2024",
                "end_date": "June 2024"
            },
            "validation_params": {
                "fee": 4000,  # Too low for 6 deliverables
                "deliverables_count": 6,
                "industry": "food",
                "platforms": ["instagram", "tiktok", "youtube"],
                "campaign_length_months": 2,
                "exclusivity_months": 8
            }
        }
    ]
    
    for i, scenario in enumerate(campaign_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print("-" * 50)
        
        # Step 1: Generate preamble
        print("🔧 Step 1: Generate Preamble")
        preamble = generate_preamble(**scenario['generation_params'])
        
        # Show abbreviated preamble
        sections = preamble.split('\n\n')
        print(f"   Generated {len(sections)} sections:")
        for j, section in enumerate(sections, 1):
            print(f"     {j}. {section[:80]}...")
        
        # Step 2: Validate business logic
        print(f"\n🔍 Step 2: Validate Business Logic")
        is_valid, issues = validate_preamble_logic(scenario['validation_params'])
        
        print(f"   Validation Parameters:")
        for key, value in scenario['validation_params'].items():
            print(f"     {key}: {value}")
        
        print(f"\n   Validation Result:")
        if is_valid:
            print("   ✅ VALID - All business logic checks passed")
        else:
            print(f"   ❌ INVALID - {len(issues)} issue(s) found:")
            for issue in issues:
                print(f"      • {issue}")
        
        # Step 3: Show combined result
        print(f"\n📊 Step 3: Combined Assessment")
        if is_valid:
            print("   🎉 APPROVED: Preamble generated and validated successfully")
            print("   📝 Ready for use in agreement documentation")
        else:
            print("   ⚠️  REQUIRES REVIEW: Business logic issues detected")
            print("   🔧 Recommend adjusting parameters before finalizing")
        
        print()
    
    print("=" * 65)
    print("✅ Workflow Demo Complete!")
    print()
    print("💡 Key Benefits of Combined Approach:")
    print("   • Structured generation ensures consistent 4-section format")
    print("   • Validation catches business logic inconsistencies")
    print("   • Fast rule-based validation (no AI/ML required)")
    print("   • Clear feedback for parameter adjustment")
    print()
    print("🔧 Typical Usage Pattern:")
    print("   1. Generate preamble with generate_preamble()")
    print("   2. Validate parameters with validate_preamble_logic()")
    print("   3. Adjust parameters if validation fails")
    print("   4. Regenerate and revalidate until approved")

def quick_validation_demo():
    """Quick demo of validation-only functionality."""
    
    print("\n" + "=" * 40)
    print("🔍 Quick Validation Demo")
    print("=" * 40)
    
    # Test quick validation scenarios
    test_params = [
        {"fee": 25000, "deliverables_count": 8, "industry": "beauty", 
         "platforms": ["instagram", "youtube"], "campaign_length_months": 3, "exclusivity_months": 9},
        {"fee": 2000, "deliverables_count": 10, "industry": "tech", 
         "platforms": ["tiktok"], "campaign_length_months": 1, "exclusivity_months": 24},
        {"fee": 50000, "deliverables_count": 2, "industry": "finance", 
         "platforms": ["linkedin"], "campaign_length_months": 6, "exclusivity_months": 12}
    ]
    
    for i, params in enumerate(test_params, 1):
        print(f"\nTest {i}: {params}")
        is_valid, issues = validate_preamble_logic(params)
        print(f"Result: {'✅ Valid' if is_valid else '❌ Invalid'}")
        if issues:
            for issue in issues:
                print(f"  • {issue}")

if __name__ == "__main__":
    demo_workflow()
    quick_validation_demo() 