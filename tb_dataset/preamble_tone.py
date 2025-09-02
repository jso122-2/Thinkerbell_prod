"""
Preamble Tone Transformer

Applies brand-specific tone transformations to preamble text using
rule-based substitutions and phrasing adjustments.
"""

import re
from typing import Dict, List, Tuple


def apply_preamble_tone(preamble_text: str, brand_name: str) -> str:
    """
    Apply brand-specific tone transformation to preamble text.
    
    Args:
        preamble_text: Original preamble text to transform
        brand_name: Brand name to determine tone application
        
    Returns:
        Tone-adapted preamble text with core information preserved
    """
    
    # Normalize brand name for matching
    brand_key = brand_name.lower().strip()
    
    # Get tone rules for the brand
    tone_rules = _get_brand_tone_rules(brand_key)
    
    if not tone_rules:
        # No specific tone rules found, return original text
        return preamble_text
    
    # Apply tone transformation
    transformed_text = _apply_tone_rules(preamble_text, tone_rules)
    
    return transformed_text


def _get_brand_tone_rules(brand_key: str) -> Dict:
    """
    Get tone transformation rules for specific brands.
    
    Args:
        brand_key: Normalized brand name (lowercase)
        
    Returns:
        Dictionary containing tone transformation rules
    """
    
    # EXPANSION POINT: Add new brand tone rules here
    # Each brand should have: word_substitutions, phrase_patterns, tone_modifiers
    
    brand_tone_rules = {
        
        # Australia Post - Formal, government-adjacent tone
        "australia post": {
            "word_substitutions": {
                "campaign": "service initiative",
                "content": "communications",
                "deliverables": "service offerings", 
                "engagement": "partnership arrangement",
                "brand ambassador": "official representative",
                "influencer": "community advocate",
                "authentic": "professional",
                "leverage": "utilize",
                "drive": "facilitate",
                "execute": "implement"
            },
            "phrase_patterns": [
                (r"aims to leverage", "is designed to utilize"),
                (r"brand awareness", "public awareness"),
                (r"target audiences", "the Australian community"),
                (r"established audience connection", "proven community engagement"),
                (r"content creation", "communication development"),
                (r"campaign objectives", "service delivery objectives")
            ],
            "tone_modifiers": {
                "formality_level": "high",
                "add_governance_language": True,
                "emphasize_service": True
            }
        },
        
        # Rexona - Energetic, sport/active tone
        "rexona": {
            "word_substitutions": {
                "engagement": "active partnership",
                "collaboration": "dynamic collaboration", 
                "execute": "power through",
                "implement": "launch into action",
                "deliver": "crush",
                "create": "energize",
                "authentic": "high-energy",
                "established": "proven performance",
                "effective": "powerhouse",
                "comprehensive": "all-out",
                "initiative": "challenge"
            },
            "phrase_patterns": [
                (r"aims to leverage", "is ready to unleash"),
                (r"drive brand awareness", "boost brand energy"),
                (r"target audiences", "active communities"),
                (r"brand values", "performance values"),
                (r"consumer consideration", "active engagement"),
                (r"content creation", "performance content"),
                (r"campaign timeline", "action timeline"),
                (r"appropriate review", "performance review")
            ],
            "tone_modifiers": {
                "energy_level": "high",
                "sports_language": True,
                "action_oriented": True
            }
        },
        
        # Queen Fine Foods - Playful, culinary flair
        "queen fine foods": {
            "word_substitutions": {
                "campaign": "culinary journey",
                "engagement": "delicious partnership",
                "collaboration": "recipe for success",
                "execute": "whip up",
                "deliver": "serve up",
                "create": "craft",
                "authentic": "naturally delicious",
                "established": "well-seasoned",
                "effective": "perfectly blended",
                "comprehensive": "full-flavored",
                "initiative": "cooking adventure"
            },
            "phrase_patterns": [
                (r"aims to leverage", "is ready to blend"),
                (r"drive brand awareness", "stir up brand excitement"),
                (r"target audiences", "food-loving communities"),
                (r"brand values", "culinary values"),
                (r"consumer consideration", "taste-bud temptation"),
                (r"content creation", "recipe development"),
                (r"campaign timeline", "cooking schedule"),
                (r"brand messaging", "flavor story"),
                (r"appropriate review", "taste testing")
            ],
            "tone_modifiers": {
                "playfulness": "high",
                "culinary_language": True,
                "warm_tone": True
            }
        }
        
        # EXPANSION INSTRUCTIONS:
        # To add a new brand tone, copy this template:
        #
        # "new_brand_name": {
        #     "word_substitutions": {
        #         "generic_word": "brand_specific_word",
        #         # Add 8-12 key substitutions that reflect brand personality
        #     },
        #     "phrase_patterns": [
        #         (r"original_phrase_pattern", "brand_specific_replacement"),
        #         # Add 6-10 phrase transformations using regex patterns
        #     ],
        #     "tone_modifiers": {
        #         "tone_characteristic": True/False,
        #         # Add 2-4 tone characteristics for future expansion
        #     }
        # }
        #
        # Brand tone categories to consider:
        # - Luxury brands: elevated, sophisticated language
        # - Tech brands: innovative, forward-thinking terms
        # - Health brands: wellness, vitality language
        # - Finance brands: secure, trustworthy terminology
        # - Fashion brands: trendy, style-conscious phrasing
        # - Food brands: sensory, appetite-appealing language
        # - Automotive: performance, reliability focus
        # - Travel: adventure, discovery language
    }
    
    return brand_tone_rules.get(brand_key, {})


def _apply_tone_rules(text: str, tone_rules: Dict) -> str:
    """
    Apply tone transformation rules to text.
    
    Args:
        text: Original text to transform
        tone_rules: Dictionary containing transformation rules
        
    Returns:
        Transformed text with tone applied
    """
    
    transformed_text = text
    
    # 1. Apply word substitutions
    word_substitutions = tone_rules.get("word_substitutions", {})
    for original_word, replacement_word in word_substitutions.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(original_word) + r'\b'
        transformed_text = re.sub(pattern, replacement_word, transformed_text, flags=re.IGNORECASE)
    
    # 2. Apply phrase pattern transformations
    phrase_patterns = tone_rules.get("phrase_patterns", [])
    for pattern, replacement in phrase_patterns:
        transformed_text = re.sub(pattern, replacement, transformed_text, flags=re.IGNORECASE)
    
    # 3. Apply tone modifiers (for future expansion)
    tone_modifiers = tone_rules.get("tone_modifiers", {})
    
    # EXPANSION POINT: Add tone modifier logic here
    # Example implementations:
    
    if tone_modifiers.get("formality_level") == "high":
        # Make text more formal (future implementation)
        pass
    
    if tone_modifiers.get("energy_level") == "high":
        # Add energetic punctuation or emphasis (future implementation)
        pass
    
    if tone_modifiers.get("playfulness") == "high":
        # Add playful elements (future implementation) 
        pass
    
    return transformed_text


def get_supported_brands() -> List[str]:
    """
    Get list of brands with defined tone rules.
    
    Returns:
        List of supported brand names
    """
    
    # This function helps users know which brands have tone rules
    supported_brands = [
        "Australia Post",
        "Rexona", 
        "Queen Fine Foods"
    ]
    
    return supported_brands


def main():
    """Test the preamble tone transformer with different brands."""
    
    print("🎯 Preamble Tone Transformer Test")
    print("=" * 50)
    
    # Sample preamble for testing
    original_preamble = """Thinkerbell has engaged Sarah Chen to provide influencer services for Australia Post. This agreement establishes the working relationship between Thinkerbell as the coordinating agency, Sarah Chen as the content creator and brand ambassador, and Australia Post as the sponsoring brand for the upcoming campaign initiative.

The purpose of this engagement is to execute a comprehensive delivery awareness campaign that effectively communicates Australia Post's brand values and product messaging to target audiences. This collaboration aims to leverage Sarah Chen's authentic voice and established audience connection to drive brand awareness, engagement, and consumer consideration for Australia Post's offerings.

The scope of work encompasses Instagram posts, Instagram stories, and product photography as part of the delivery awareness campaign initiative. Sarah Chen will create and deliver content that aligns with Australia Post's brand guidelines and campaign objectives, ensuring authentic integration of brand messaging within Sarah Chen's established content style and audience expectations.

The campaign timeline spans from March 2024 through May 2024, with content creation and publication scheduled according to the agreed campaign calendar. All deliverables will be completed within the specified timeframe, allowing for appropriate review periods and any necessary revisions to ensure optimal campaign performance and brand alignment."""
    
    print(f"\n📝 Original Preamble (Australia Post):")
    print(f"Length: {len(original_preamble)} characters")
    print(f"First 150 chars: {original_preamble[:150]}...")
    
    # Test each brand tone
    test_brands = ["Australia Post", "Rexona", "Queen Fine Foods", "Unknown Brand"]
    
    for i, brand in enumerate(test_brands, 1):
        print(f"\n🎨 Test {i}: {brand} Tone")
        print("-" * 30)
        
        # Apply tone transformation
        transformed = apply_preamble_tone(original_preamble, brand)
        
        if transformed == original_preamble:
            print("   No tone rules applied (brand not supported)")
        else:
            print("   ✅ Tone transformation applied")
            
            # Show key changes
            changes = _find_key_changes(original_preamble, transformed)
            print(f"   Key changes detected: {len(changes)}")
            
            # Show first few changes
            for j, (original_phrase, new_phrase) in enumerate(changes[:3]):
                print(f"      {j+1}. '{original_phrase}' → '{new_phrase}'")
            
            if len(changes) > 3:
                print(f"      ... and {len(changes) - 3} more changes")
        
        print(f"   Length: {len(transformed)} characters (original: {len(original_preamble)})")
    
    # Test specific transformations
    print(f"\n🔍 Detailed Transformation Example: Rexona")
    print("-" * 45)
    
    rexona_preamble = original_preamble.replace("Australia Post", "Rexona")
    rexona_transformed = apply_preamble_tone(rexona_preamble, "Rexona")
    
    print("Original excerpt:")
    print(f"   {rexona_preamble.split('.')[1][:100]}...")
    print("\nTransformed excerpt:")
    print(f"   {rexona_transformed.split('.')[1][:100]}...")
    
    print(f"\n✅ Test complete!")
    print(f"💡 Supported brands: {', '.join(get_supported_brands())}")


def _find_key_changes(original: str, transformed: str) -> List[Tuple[str, str]]:
    """Find key differences between original and transformed text."""
    
    # Simple word-level comparison to identify changes
    original_words = original.lower().split()
    transformed_words = transformed.lower().split()
    
    changes = []
    
    # Find word substitutions (simplified approach)
    for i, (orig_word, trans_word) in enumerate(zip(original_words, transformed_words)):
        if orig_word != trans_word:
            changes.append((orig_word, trans_word))
    
    return changes[:10]  # Return first 10 changes


if __name__ == "__main__":
    main() 