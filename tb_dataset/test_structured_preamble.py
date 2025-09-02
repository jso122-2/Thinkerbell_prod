#!/usr/bin/env python3
"""
Test script demonstrating the structured generate_preamble() function

Shows the exact four-section output format as specified.
"""

from structured_preamble import generate_preamble
from datetime import datetime

def demonstrate_structure():
    """Demonstrate the four-section preamble structure"""
    
    print("🎯 Structured Preamble Function Demonstration")
    print("=" * 60)
    print("Function: generate_preamble(client, talent, agency, deliverables, campaign_type, start_date, end_date)")
    print("Output: Four paragraphs separated by line breaks")
    print()
    
    # Generate a sample preamble
    preamble = generate_preamble(
        client="Woolworths",
        talent="Emma Wilson", 
        agency="Thinkerbell",
        deliverables=["Instagram posts", "TikTok videos", "product photography"],
        campaign_type="Fresh Food Campaign",
        start_date="February 2024",
        end_date="April 2024"
    )
    
    # Split into sections to show structure
    sections = preamble.split('\n\n')
    
    print("📋 FOUR-SECTION OUTPUT STRUCTURE:")
    print()
    
    section_names = [
        "1. PARTIES INVOLVED (client, talent, agency)",
        "2. PURPOSE OF AGREEMENT / PROJECT OVERVIEW", 
        "3. SCOPE AT A GLANCE (deliverables, campaign type)",
        "4. HIGH-LEVEL TIMING REFERENCES"
    ]
    
    for i, (name, section) in enumerate(zip(section_names, sections)):
        print(f"{name}:")
        print(f"   {section}")
        print()
    
    print("✅ STRUCTURE VALIDATION:")
    print(f"   Total sections: {len(sections)} (expected: 4)")
    print(f"   Each section is one paragraph: {'✅' if all('.' in s for s in sections) else '❌'}")
    print(f"   Sections separated by line breaks: {'✅' if len(sections) == 4 else '❌'}")
    print(f"   Output is single string: {'✅' if isinstance(preamble, str) else '❌'}")
    print(f"   Contains all required elements: {'✅' if all(param in preamble for param in ['Woolworths', 'Emma Wilson', 'Thinkerbell', 'Fresh Food Campaign']) else '❌'}")

if __name__ == "__main__":
    demonstrate_structure() 