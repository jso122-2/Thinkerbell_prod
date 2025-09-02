"""
Structured Preamble Generator

Generates business preambles with consistent four-section structure:
1. Parties involved
2. Purpose of agreement / project overview  
3. Scope at a glance
4. High-level timing references
"""

from datetime import datetime
from typing import List, Union


def generate_preamble(client: str,
                     talent: str, 
                     agency: str,
                     deliverables: List[str],
                     campaign_type: str,
                     start_date: Union[str, datetime],
                     end_date: Union[str, datetime]) -> str:
    """
    Generate a structured business preamble with four anchor sections.
    
    Args:
        client: Client/brand name
        talent: Talent/influencer name  
        agency: Agency name
        deliverables: List of deliverable items
        campaign_type: Type of campaign
        start_date: Campaign start date
        end_date: Campaign end date
        
    Returns:
        Four-paragraph preamble string with sections separated by line breaks
    """
    
    # Section 1: Parties involved (client, talent, agency)
    parties_section = (
        f"{agency} has engaged {talent} to provide influencer services for {client}. "
        f"This agreement establishes the working relationship between {agency} as the "
        f"coordinating agency, {talent} as the content creator and brand ambassador, "
        f"and {client} as the sponsoring brand for the upcoming campaign initiative."
    )
    
    # Section 2: Purpose of agreement / project overview
    purpose_section = (
        f"The purpose of this engagement is to execute a comprehensive {campaign_type} "
        f"that effectively communicates {client}'s brand values and product messaging "
        f"to target audiences. This collaboration aims to leverage {talent}'s authentic "
        f"voice and established audience connection to drive brand awareness, engagement, "
        f"and consumer consideration for {client}'s offerings."
    )
    
    # Section 3: Scope at a glance (deliverables, campaign type)
    deliverables_text = ", ".join(deliverables[:-1]) + f", and {deliverables[-1]}" if len(deliverables) > 1 else deliverables[0]
    scope_section = (
        f"The scope of work encompasses {deliverables_text} as part of the {campaign_type} "
        f"initiative. {talent} will create and deliver content that aligns with {client}'s "
        f"brand guidelines and campaign objectives, ensuring authentic integration of brand "
        f"messaging within {talent}'s established content style and audience expectations."
    )
    
    # Section 4: High-level timing references
    # Format dates if they're datetime objects
    if isinstance(start_date, datetime):
        start_str = start_date.strftime("%B %Y")
    else:
        start_str = str(start_date)
        
    if isinstance(end_date, datetime):
        end_str = end_date.strftime("%B %Y")
    else:
        end_str = str(end_date)
    
    timing_section = (
        f"The campaign timeline spans from {start_str} through {end_str}, with content "
        f"creation and publication scheduled according to the agreed campaign calendar. "
        f"All deliverables will be completed within the specified timeframe, allowing "
        f"for appropriate review periods and any necessary revisions to ensure optimal "
        f"campaign performance and brand alignment."
    )
    
    # Combine all sections with line breaks
    preamble = f"{parties_section}\n\n{purpose_section}\n\n{scope_section}\n\n{timing_section}"
    
    return preamble


def main():
    """Test the structured preamble generator"""
    
    print("🎯 Structured Preamble Generator Test")
    print("=" * 50)
    
    # Test case 1: Basic campaign
    print("\n📝 Test 1: Basic Fashion Campaign")
    
    preamble1 = generate_preamble(
        client="David Jones",
        talent="Sarah Chen",
        agency="Thinkerbell",
        deliverables=["Instagram posts", "Instagram stories", "product photography"],
        campaign_type="Spring Collection Launch",
        start_date="March 2024",
        end_date="May 2024"
    )
    
    print(preamble1)
    
    # Test case 2: Complex campaign with datetime objects
    print(f"\n\n📝 Test 2: Complex Tech Campaign")
    
    preamble2 = generate_preamble(
        client="JB Hi-Fi",
        talent="Michael Rodriguez", 
        agency="Ogilvy Sydney",
        deliverables=["YouTube review videos", "TikTok unboxing content", "Instagram reels", "blog posts"],
        campaign_type="Holiday Electronics Promotion",
        start_date=datetime(2024, 11, 1),
        end_date=datetime(2024, 12, 31)
    )
    
    print(preamble2)
    
    # Test case 3: Simple campaign
    print(f"\n\n📝 Test 3: Simple Food Campaign")
    
    preamble3 = generate_preamble(
        client="Guzman y Gomez",
        talent="Emma Thompson",
        agency="M&C Saatchi",
        deliverables=["Instagram posts"],
        campaign_type="New Menu Launch",
        start_date="June 2024", 
        end_date="July 2024"
    )
    
    print(preamble3)
    
    print(f"\n✅ All tests completed!")
    print(f"💡 Each preamble contains exactly 4 sections as specified:")
    print(f"   1. Parties involved")
    print(f"   2. Purpose of agreement / project overview") 
    print(f"   3. Scope at a glance")
    print(f"   4. High-level timing references")


if __name__ == "__main__":
    main() 