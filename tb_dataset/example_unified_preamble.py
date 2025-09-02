#!/usr/bin/env python3
"""
Simple Example: Unified Business Preamble Generator

Shows how to use generate_business_preamble() for complete
preamble generation with validation and tone adaptation.
"""

from tb_dataset import generate_business_preamble, PreambleGenerationResult

def main():
    """Simple demonstration of unified preamble generation."""
    
    print("🎯 Unified Business Preamble Generator - Simple Example")
    print("=" * 60)
    
    # Example: Valid Australia Post campaign
    print("\n📝 Example: Australia Post Community Campaign")
    print("-" * 45)
    
    result = generate_business_preamble(
        # Generation parameters
        client="Australia Post",
        talent="Emma Wilson", 
        agency="Thinkerbell",
        deliverables=["Instagram posts", "community stories"],
        campaign_type="Community Engagement Initiative",
        start_date="April 2024",
        end_date="June 2024",
        
        # Validation parameters  
        fee=12000,
        industry="government",
        platforms=["instagram", "facebook"],
        campaign_length_months=2,
        exclusivity_months=6
    )
    
    # Show results
    print(f"✅ Success: {result.validation_passed}")
    print(f"🎨 Tone applied: {result.tone_applied}")
    print(f"📊 Attempts: {result.generation_attempts}")
    print(f"📄 Length: {len(result.preamble)} characters")
    
    if result.validation_issues:
        print(f"⚠️ Issues: {result.validation_issues}")
    
    # Show preamble sections
    sections = result.preamble.split('\n\n')
    print(f"\n📋 Generated Preamble ({len(sections)} sections):")
    for i, section in enumerate(sections, 1):
        print(f"\n{i}. {section}")
    
    # Example: Show result metadata
    print(f"\n📊 Result Metadata:")
    metadata = result.to_dict()
    for key, value in metadata.items():
        if key != 'preamble':  # Skip the full text
            print(f"   {key}: {value}")
    
    print(f"\n✅ Example complete!")
    print(f"💡 The unified function handles:")
    print(f"   • Structured generation (4 anchor sections)")
    print(f"   • Business logic validation") 
    print(f"   • Brand tone adaptation")
    print(f"   • Error handling and auto-adjustment")
    print(f"   • Comprehensive result metadata")

if __name__ == "__main__":
    main() 