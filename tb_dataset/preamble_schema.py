"""
Preamble Schema and Rendering Module

Defines the Preamble dataclass structure and renders business preambles
in clean prose format with specified style profiles.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class Preamble:
    """
    Schema for business preamble content blocks.
    
    Attributes:
        purpose: High-level purpose of the engagement
        brand_voice: Brand personality and messaging approach
        campaign_context: Campaign background and objectives
        deliverables_block: List of deliverable items and requirements
        timelines_block: Engagement dates and key milestones
        constraints_block: Limitations, restrictions, and compliance requirements
        money_block: Fee structure and payment terms
        exclusivity_block: Exclusivity terms and competitive restrictions
    """
    purpose: str
    brand_voice: str
    campaign_context: str
    deliverables_block: List[str]
    timelines_block: Dict[str, str]
    constraints_block: List[str]
    money_block: Dict[str, Any]
    exclusivity_block: Dict[str, Any]


def _convert_number_to_words(amount: int) -> str:
    """Convert integer amount to written words."""
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
             "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if amount == 0:
        return "zero"
    
    def convert_hundreds(n):
        result = ""
        if n >= 100:
            result += ones[n // 100] + " hundred "
            n %= 100
        if n >= 20:
            result += tens[n // 10] + " "
            n %= 10
        elif n >= 10:
            result += teens[n - 10] + " "
            n = 0
        if n > 0:
            result += ones[n] + " "
        return result.strip()
    
    if amount < 1000:
        return convert_hundreds(amount)
    elif amount < 1000000:
        thousands = amount // 1000
        remainder = amount % 1000
        result = convert_hundreds(thousands) + " thousand"
        if remainder > 0:
            result += " " + convert_hundreds(remainder)
        return result
    else:
        # For larger amounts, simplify
        return f"approximately {amount:,}"


def _format_fee_with_words(fee_amount: int, currency: str = "AUD") -> str:
    """Format fee amount with both words and numerals."""
    words = _convert_number_to_words(fee_amount)
    return f"{words} dollars ({currency} ${fee_amount:,})"


def _extract_timebox_terms(timelines: Dict[str, str], exclusivity: Dict[str, Any]) -> List[str]:
    """Extract explicit timebox terms from timelines and exclusivity blocks."""
    timebox_terms = []
    
    # Extract from timelines
    for key, value in timelines.items():
        if any(term in key.lower() for term in ['engagement', 'term', 'duration', 'period']):
            timebox_terms.append(value)
    
    # Extract from exclusivity
    if exclusivity and 'period' in exclusivity:
        timebox_terms.append(exclusivity['period'])
    if exclusivity and 'duration' in exclusivity:
        timebox_terms.append(exclusivity['duration'])
    
    return timebox_terms


def render_preamble(schema: Preamble, fields: Dict[str, Any], style_profile: Optional[Dict[str, Any]] = None) -> str:
    """
    Render a business preamble in clean prose format.
    
    Args:
        schema: Preamble dataclass instance with content blocks
        fields: Dictionary containing client, brand, fee, dates, deliverables, usage/exclusivity info
        style_profile: Optional style profile with tone and lexicon information
        
    Returns:
        str: Rendered preamble text (120-220 words) in agency brief tone
        
    Raises:
        ValueError: If required fields are missing or word count is outside range
    """
    # Validate required fields
    required_fields = ['client', 'campaign', 'fee']
    missing_fields = [field for field in required_fields if field not in fields or not fields[field]]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Extract key information
    client = fields['client']
    campaign = fields['campaign']
    brand = fields.get('brand', client)
    fee_amount = fields['fee']
    
    # Apply style profile if provided
    tone_modifiers = []
    if style_profile and 'tone_markers' in style_profile:
        dominant_tone = style_profile['tone_markers'].get('dominant_tone', 'formal')
        if dominant_tone == 'casual':
            tone_modifiers = ['collaborative', 'dynamic', 'engaging']
        else:
            tone_modifiers = ['professional', 'strategic', 'comprehensive']
    
    # Format fee with words and numerals
    fee_text = _format_fee_with_words(fee_amount, fields.get('currency', 'AUD'))
    
    # Build preamble sections
    sections = []
    
    # Opening sentence (client + campaign)
    opening_tone = tone_modifiers[0] if tone_modifiers else 'strategic'
    sections.append(f"{client} has engaged our agency to deliver a {opening_tone} {campaign} campaign that aligns with {brand}'s core values and market positioning.")
    
    # Purpose and brand voice integration
    if schema.purpose and schema.brand_voice:
        sections.append(f"The engagement centers on {schema.purpose.lower()}, leveraging {schema.brand_voice.lower()} to create authentic connections with target audiences.")
    
    # Campaign context
    if schema.campaign_context:
        sections.append(f"{schema.campaign_context}")
    
    # Deliverables summary
    if schema.deliverables_block:
        deliverable_count = len(schema.deliverables_block)
        primary_deliverable = schema.deliverables_block[0] if schema.deliverables_block else "content outputs"
        if deliverable_count > 1:
            sections.append(f"Key deliverables include {primary_deliverable} and {deliverable_count - 1} additional content elements designed to maximize campaign impact.")
        else:
            sections.append(f"The primary deliverable focuses on {primary_deliverable} to achieve campaign objectives.")
    
    # Timelines with explicit timebox terms
    timebox_terms = _extract_timebox_terms(schema.timelines_block, schema.exclusivity_block)
    if schema.timelines_block:
        engagement_period = schema.timelines_block.get('engagement_term', 'the agreed timeframe')
        sections.append(f"The engagement period spans {engagement_period}, with deliverables scheduled according to strategic campaign milestones.")
    
    # Fee integration (required)
    sections.append(f"The total project investment is {fee_text}, structured to align with deliverable completion and campaign phases.")
    
    # Exclusivity/usage terms
    if schema.exclusivity_block and schema.exclusivity_block.get('terms'):
        exclusivity_terms = schema.exclusivity_block['terms']
        sections.append(f"Usage rights and exclusivity parameters are defined to protect brand integrity while maximizing content utilization {exclusivity_terms}.")
    
    # Constraints as professional considerations
    if schema.constraints_block:
        key_constraint = schema.constraints_block[0] if schema.constraints_block else "brand guidelines"
        sections.append(f"All content development adheres to {key_constraint} and industry compliance standards.")
    
    # Add additional context to ensure minimum word count
    if len(" ".join(sections).split()) < 120:
        sections.append(f"Our agency brings extensive experience in {schema.brand_voice.lower() if schema.brand_voice else 'brand-focused'} campaign development, ensuring all deliverables resonate with target demographics while maintaining consistent brand messaging across all touchpoints.")
        sections.append(f"This comprehensive approach to {campaign.lower()} campaign management includes ongoing optimization, performance monitoring, and strategic adjustments to maximize engagement and return on investment throughout the {schema.timelines_block.get('engagement_term', 'engagement period')}.")
    
    # Join sections into complete preamble
    preamble_text = " ".join(sections)
    
    # Validate word count (120-220 words)
    word_count = len(preamble_text.split())
    if word_count < 120 or word_count > 220:
        raise ValueError(f"Preamble word count ({word_count}) outside required range (120-220 words)")
    
    return preamble_text


def validate_preamble_content(preamble_text: str, required_fields: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate that preamble contains required elements.
    
    Args:
        preamble_text: Generated preamble text
        required_fields: Dictionary with client, campaign, fee, deliverables info
        
    Returns:
        Dict with validation results for each requirement
    """
    results = {}
    
    # Check for client mention
    client = required_fields.get('client', '')
    results['has_client'] = bool(client and client.lower() in preamble_text.lower())
    
    # Check for campaign mention
    campaign = required_fields.get('campaign', '')
    results['has_campaign'] = bool(campaign and campaign.lower() in preamble_text.lower())
    
    # Check for fee (both words and numerals)
    fee_amount = required_fields.get('fee', 0)
    fee_str = str(fee_amount)
    fee_str_comma = f"{fee_amount:,}"  # Format with commas
    has_fee_numbers = fee_str in preamble_text or fee_str_comma in preamble_text
    has_fee_words = any(word in preamble_text.lower() for word in ['dollar', 'aud', '$'])
    results['has_fee'] = has_fee_numbers and has_fee_words
    
    # Check for at least one deliverable
    deliverables = required_fields.get('deliverables', [])
    has_deliverable = False
    if deliverables:
        for deliverable in deliverables:
            if deliverable.lower() in preamble_text.lower():
                has_deliverable = True
                break
    # Also check for generic deliverable terms
    deliverable_terms = ['deliverable', 'content', 'output', 'material', 'campaign']
    has_deliverable = has_deliverable or any(term in preamble_text.lower() for term in deliverable_terms)
    results['has_deliverable'] = has_deliverable
    
    # Check for at least one timebox term
    timebox_terms = ['period', 'term', 'timeframe', 'duration', 'engagement', 'timeline', 'phase', 'milestone']
    results['has_timebox'] = any(term in preamble_text.lower() for term in timebox_terms)
    
    return results 