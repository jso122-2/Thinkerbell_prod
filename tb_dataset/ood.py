"""
Out-of-distribution (OOD) sample generator for robustness testing.

Generates negative samples (non-influencer agreements) and edge cases
(minimal info, international, B2B) to improve model robustness.
"""

import random
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class OODGenerator:
    """
    Generates out-of-distribution samples for model robustness.
    
    Creates two types of OOD samples as specified:
    1. Pure negatives (10%): Non-influencer documents (NDAs, employment, business contracts)
    2. Edge cases (10%): Minimal info influencer agreements, wrong industry, international
    """
    
    def __init__(self):
        """Initialize OOD generator with enhanced templates and data."""
        self.setup_negative_templates()
        self.setup_edge_case_templates()
        self.setup_international_data()
        
        # OOD distribution ratios as specified
        self.pure_negative_ratio = 0.5  # 50% of OOD samples are pure negatives
        self.edge_case_ratio = 0.5      # 50% of OOD samples are edge cases
        
        self.stats = {
            'ood_samples_generated': 0,
            'negative_samples': 0,
            'edge_case_samples': 0,
            'ood_types': {},
            'distribution_target': {
                'pure_negative': self.pure_negative_ratio,
                'edge_case': self.edge_case_ratio
            }
        }
    
    def setup_negative_templates(self):
        """Set up templates for pure negative samples."""
        self.negative_templates = {
            'employment': {
                'template': """EMPLOYMENT CONTRACT

This Employment Agreement is entered into between {company} ("Company") and {employee} ("Employee").

POSITION: {position}
SALARY: ${salary:,} per annum
START DATE: {start_date}

DUTIES AND RESPONSIBILITIES:
{responsibilities}

EMPLOYMENT TERMS:
- Full-time position, 38 hours per week
- 4 weeks annual leave
- Sick leave as per company policy
- Probationary period: 6 months

CONFIDENTIALITY:
Employee agrees to maintain confidentiality of company information.

This agreement is governed by Australian employment law.""",
                'data': {
                    'companies': ['TechCorp Pty Ltd', 'Marketing Solutions', 'Digital Agency', 'Creative Studio'],
                    'positions': ['Marketing Coordinator', 'Content Writer', 'Social Media Manager', 'Graphic Designer'],
                    'responsibilities': [
                        'Manage social media accounts and create content',
                        'Develop marketing campaigns and promotional materials',
                        'Write blog posts and website content',
                        'Design graphics and visual content'
                    ]
                }
            },
            'supplier': {
                'template': """SUPPLIER AGREEMENT

Agreement between {client_company} ("Client") and {supplier_company} ("Supplier").

SERVICES: {services}
PAYMENT TERMS: {payment_terms}
DURATION: {duration}

SCOPE OF SERVICES:
{service_details}

PRICING:
- Service fee: ${fee:,}
- Payment schedule: {payment_schedule}
- Additional costs as agreed

TERMS AND CONDITIONS:
- Supplier warrants professional service delivery
- Client responsible for timely payment
- Either party may terminate with 30 days notice

Signed: {date}""",
                'data': {
                    'client_companies': ['Retail Corp', 'Fashion House', 'Beauty Brand', 'Tech Startup'],
                    'supplier_companies': ['Creative Agency', 'Production House', 'Marketing Firm', 'Design Studio'],
                    'services': [
                        'Video production services',
                        'Photography and editing',
                        'Marketing campaign development',
                        'Brand strategy consulting'
                    ],
                    'service_details': [
                        'Professional video production including scripting, filming, and post-production',
                        'Complete photography package with editing and retouching services',
                        'Comprehensive marketing campaign from concept to execution',
                        'Strategic brand positioning and marketing consultation'
                    ]
                }
            },
            'legal': {
                'template': """NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("NDA") is entered into between {party_a} and {party_b}.

PURPOSE: {purpose}
EFFECTIVE DATE: {date}

CONFIDENTIAL INFORMATION:
Both parties acknowledge that confidential information may be disclosed including:
- Business strategies and plans
- Financial information
- Technical specifications
- Customer data and lists

OBLIGATIONS:
1. Maintain strict confidentiality
2. Use information solely for intended purpose
3. Return all materials upon request
4. Notify of any unauthorized disclosure

TERM: This agreement remains in effect for {term} years.

GOVERNING LAW: This agreement is governed by the laws of Australia.""",
                'data': {
                    'purposes': [
                        'Potential business partnership discussions',
                        'Joint venture exploration',
                        'Technology licensing negotiations',
                        'Merger and acquisition discussions'
                    ]
                }
            },
            'real_estate': {
                'template': """LEASE AGREEMENT

PROPERTY: {property_address}
LANDLORD: {landlord}
TENANT: {tenant}

LEASE TERMS:
- Monthly rent: ${rent:,}
- Lease period: {lease_period}
- Bond: ${bond:,}
- Utilities: {utilities}

PROPERTY DETAILS:
{property_details}

CONDITIONS:
- No pets without written consent
- No smoking inside the property
- Tenant responsible for garden maintenance
- 24 hours notice required for inspections

Both parties agree to the terms outlined above.

Date: {date}""",
                'data': {
                    'property_types': [
                        '2 bedroom apartment in CBD',
                        '3 bedroom house with garden',
                        'Studio apartment near university',
                        'Commercial office space'
                    ],
                    'utilities': [
                        'Tenant responsible for all utilities',
                        'Water included, electricity/gas separate',
                        'All utilities included in rent',
                        'Shared utilities between tenants'
                    ]
                }
            }
        }
    
    def setup_edge_case_templates(self):
        """Set up templates for edge case influencer samples."""
        self.edge_case_templates = {
            'minimal_info': {
                'template': """Collab with {brand}

Content: {deliverables}
Pay: {fee}
Time: {duration}

{minimal_details}

Let me know if you're interested!""",
                'data': {
                    'minimal_details': [
                        'Quick turnaround needed.',
                        'Flexible on timing.',
                        'Standard T&Cs apply.',
                        'Details to be confirmed.',
                        'More info available on request.'
                    ]
                }
            },
            'international': {
                'template': """INTERNATIONAL COLLABORATION AGREEMENT

Brand: {brand} ({country})
Influencer: {influencer}
Campaign: {campaign}

DELIVERABLES:
{deliverables}

COMPENSATION:
- Fee: ${fee:,} {currency}
- Currency conversion at time of payment
- Tax obligations as per {country} law

INTERNATIONAL TERMS:
- Content must comply with {country} advertising standards
- Time zone coordination required
- Cultural sensitivity guidelines apply
- Translation services available if needed

USAGE RIGHTS:
- Global usage permitted
- Content suitable for {region} markets
- Local adaptation may be required

Duration: {duration}
Territory: {territory}""",
                'data': {
                    'countries': ['United States', 'United Kingdom', 'Germany', 'Japan', 'Singapore'],
                    'currencies': ['USD', 'GBP', 'EUR', 'JPY', 'SGD'],
                    'regions': ['North America', 'Europe', 'Asia-Pacific', 'Global', 'English-speaking markets'],
                    'territories': ['Worldwide', 'English-speaking countries', 'APAC region', 'European Union']
                }
            },
            'b2b_influencer': {
                'template': """B2B INFLUENCER PARTNERSHIP

Business Influencer: {influencer}
Corporate Client: {client}
Industry Focus: {industry}

PROFESSIONAL CONTENT:
{deliverables}

B2B SPECIFIC TERMS:
- Content targets business professionals
- LinkedIn primary platform
- Professional tone required
- Industry expertise demonstration
- Thought leadership positioning

COMPENSATION: {fee}
EXCLUSIVITY: {exclusivity}
PROFESSIONAL STANDARDS: Content must maintain industry credibility

This agreement follows standard B2B influencer practices.""",
                'data': {
                    'industries': [
                        'Technology and SaaS',
                        'Financial Services',
                        'Professional Consulting',
                        'Healthcare Technology',
                        'Manufacturing and Industrial'
                    ],
                    'deliverables': [
                        ['LinkedIn thought leadership posts', 'Industry webinar participation'],
                        ['Professional case study creation', 'B2B networking content'],
                        ['Executive interview series', 'Industry trend analysis'],
                        ['Product demonstration videos', 'Professional testimonials']
                    ]
                }
            },
            'incomplete': {
                'template': """Re: Partnership Opportunity

Hi {influencer},

We're interested in working with you for {brand}.

Some details:
- {partial_info}
- Budget: TBD
- Timeline: ASAP

Can we schedule a call to discuss?

Let me know your availability.

Thanks!""",
                'data': {
                    'partial_info': [
                        'Instagram content needed',
                        'Product launch campaign',
                        'Social media posts',
                        'Brand awareness content',
                        'Lifestyle content creation'
                    ]
                }
            }
        }
    
    def setup_international_data(self):
        """Set up international brands and data."""
        self.international_brands = {
            'United States': ['Nike', 'Apple', 'Coca-Cola', 'McDonald\'s', 'Starbucks'],
            'United Kingdom': ['ASOS', 'Burberry', 'Marks & Spencer', 'Tesco', 'John Lewis'],
            'Germany': ['Adidas', 'BMW', 'Mercedes-Benz', 'SAP', 'Siemens'],
            'Japan': ['Sony', 'Nintendo', 'Toyota', 'Uniqlo', 'Muji'],
            'Singapore': ['Grab', 'Lazada', 'Shopee', 'DBS Bank', 'Singapore Airlines']
        }
    
    def generate_ood_sample(self, ood_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate an OOD sample.
        
        Args:
            ood_type: 'negative', 'edge_case', or None for random selection
            
        Returns:
            Complete OOD sample dictionary or None if generation fails
        """
        try:
            # Determine OOD type
            if ood_type is None:
                ood_type = 'negative' if random.random() < self.pure_negative_ratio else 'edge_case'
            
            if ood_type == 'negative':
                return self._generate_pure_negative_sample()
            elif ood_type == 'edge_case':
                return self._generate_edge_case_sample()
            else:
                # Random selection
                return self._generate_pure_negative_sample() if random.random() < 0.5 else self._generate_edge_case_sample()
                
        except Exception as e:
            logger.error(f"Failed to generate OOD sample: {e}")
            return None
    
    def _generate_pure_negative_sample(self) -> Dict[str, Any]:
        """
        Generate pure negative sample (NOT_INFLUENCER_AGREEMENT).
        
        Includes NDAs, employment contracts, supplier agreements, real estate, etc.
        """
        negative_types = ['nda', 'employment', 'supplier', 'real_estate', 'insurance', 'legal_services']
        sample_type = random.choice(negative_types)
        
        if sample_type == 'nda':
            content = self._generate_nda_sample()
        elif sample_type == 'employment':
            content = self._generate_employment_sample()
        elif sample_type == 'supplier':
            content = self._generate_supplier_sample()
        elif sample_type == 'real_estate':
            content = self._generate_real_estate_sample()
        elif sample_type == 'insurance':
            content = self._generate_insurance_sample()
        else:  # legal_services
            content = self._generate_legal_services_sample()
        
        # Create negative sample structure
        sample = {
            'raw_input': content['text'],
            'extracted_fields': content['fields'],
            'classification': {
                'document_type': 'NOT_INFLUENCER_AGREEMENT',
                'complexity': content.get('complexity', 'medium'),
                'industry': content.get('industry', 'other'),
                'confidence_target': 0.95  # High confidence for clear negatives
            },
            'validation': {
                'semantic_coherence': 0.2,  # Low coherence for non-influencer content
                'business_logic': False,
                'temporal_logic': content.get('temporal_valid', True),
                'overall_valid': False
            },
            'is_ood': True,
            'ood_type': 'pure_negative',
            'ood_subtype': sample_type,
            'style_metadata': {
                'style_profile_id': 'formal_legal',
                'formality_score': 0.9,
                'tone': 'formal'
            }
        }
        
        self.stats['ood_samples_generated'] += 1
        self.stats['negative_samples'] += 1
        self.stats['ood_types'][sample_type] = self.stats['ood_types'].get(sample_type, 0) + 1
        
        return sample
    
    def _generate_edge_case_sample(self) -> Dict[str, Any]:
        """
        Generate edge case influencer agreement.
        
        Includes minimal info agreements, wrong industry matches, international formats.
        """
        edge_types = ['minimal_info', 'wrong_industry', 'international', 'incomplete_terms', 'unusual_platform']
        edge_type = random.choice(edge_types)
        
        if edge_type == 'minimal_info':
            content = self._generate_minimal_info_agreement()
        elif edge_type == 'wrong_industry':
            content = self._generate_wrong_industry_agreement()
        elif edge_type == 'international':
            content = self._generate_international_agreement()
        elif edge_type == 'incomplete_terms':
            content = self._generate_incomplete_agreement()
        else:  # unusual_platform
            content = self._generate_unusual_platform_agreement()
        
        # Create edge case sample structure
        sample = {
            'raw_input': content['text'],
            'extracted_fields': content['fields'],
            'classification': {
                'document_type': 'INFLUENCER_AGREEMENT',
                'complexity': content.get('complexity', 'simple'),
                'industry': content.get('industry', 'other'),
                'confidence_target': 0.6  # Lower confidence for edge cases
            },
            'validation': {
                'semantic_coherence': content.get('coherence', 0.4),  # Lower coherence
                'business_logic': content.get('business_valid', False),
                'temporal_logic': content.get('temporal_valid', False),
                'overall_valid': False
            },
            'is_ood': True,
            'ood_type': 'edge_case',
            'ood_subtype': edge_type,
            'style_metadata': content.get('style_metadata', {
                'style_profile_id': 'edge_case',
                'formality_score': 0.3,
                'tone': 'casual'
            })
        }
        
        self.stats['ood_samples_generated'] += 1
        self.stats['edge_case_samples'] += 1
        self.stats['ood_types'][edge_type] = self.stats['ood_types'].get(edge_type, 0) + 1
        
        return sample
    
    def _generate_nda_sample(self) -> Dict[str, Any]:
        """Generate NDA (Non-Disclosure Agreement) sample."""
        companies = ['TechCorp Pty Ltd', 'Innovation Labs', 'Digital Solutions Inc', 'Creative Media Group']
        purposes = [
            'Potential business partnership discussions',
            'Joint venture exploration', 
            'Technology licensing negotiations',
            'Merger and acquisition discussions',
            'Product development collaboration'
        ]
        
        company_a = random.choice(companies)
        company_b = random.choice(companies)
        while company_b == company_a:
            company_b = random.choice(companies)
            
        purpose = random.choice(purposes)
        term_years = random.choice([2, 3, 5])
        
        text = f"""NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("NDA") is entered into between {company_a} and {company_b}.

PURPOSE: {purpose}
EFFECTIVE DATE: {datetime.now().strftime('%B %d, %Y')}

CONFIDENTIAL INFORMATION:
Both parties acknowledge that confidential information may be disclosed including:
- Business strategies and plans
- Financial information and projections
- Technical specifications and intellectual property
- Customer data and contact lists
- Marketing strategies and trade secrets

OBLIGATIONS:
1. Maintain strict confidentiality of all disclosed information
2. Use information solely for the stated purpose
3. Return all materials upon completion or request
4. Notify immediately of any unauthorized disclosure
5. Limit access to authorized personnel only

TERM: This agreement remains in effect for {term_years} years from the effective date.

GOVERNING LAW: This agreement is governed by the laws of Australia and any disputes shall be resolved through binding arbitration.

Both parties acknowledge understanding and agreement to these terms."""
        
        return {
            'text': text,
            'fields': {
                'party_a': company_a,
                'party_b': company_b,
                'purpose': purpose,
                'term_years': term_years,
                'document_type': 'NDA'
            },
            'complexity': 'medium',
            'industry': 'legal',
            'temporal_valid': True
        }
    
    def _generate_employment_sample(self) -> Dict[str, Any]:
        """Generate employment contract sample."""
        companies = ['Marketing Solutions Pty Ltd', 'Digital Agency Corp', 'Creative Studio', 'Media House']
        positions = ['Marketing Coordinator', 'Content Writer', 'Social Media Manager', 'Graphic Designer']
        salaries = [55000, 65000, 70000, 80000, 90000]
        
        company = random.choice(companies)
        position = random.choice(positions)
        salary = random.choice(salaries)
        start_date = datetime.now() + timedelta(days=random.randint(7, 30))
        
        responsibilities = {
            'Marketing Coordinator': 'Coordinate marketing campaigns, manage project timelines, and liaise with external vendors',
            'Content Writer': 'Create engaging content for websites, blogs, and marketing materials',
            'Social Media Manager': 'Manage company social media accounts and develop content strategies',
            'Graphic Designer': 'Design visual content for marketing campaigns and brand materials'
        }
        
        text = f"""EMPLOYMENT CONTRACT

This Employment Agreement is entered into between {company} ("Company") and the Employee.

POSITION: {position}
SALARY: ${salary:,} per annum
START DATE: {start_date.strftime('%B %d, %Y')}

DUTIES AND RESPONSIBILITIES:
{responsibilities[position]}

EMPLOYMENT TERMS:
- Full-time position, 38 hours per week
- 4 weeks annual leave per year
- Sick leave as per company policy
- Probationary period: 6 months
- Standard superannuation contributions

CONFIDENTIALITY:
Employee agrees to maintain confidentiality of all company information and not disclose trade secrets or proprietary information.

TERMINATION:
Either party may terminate this agreement with 4 weeks written notice.

This agreement is governed by Australian employment law and Fair Work regulations."""
        
        return {
            'text': text,
            'fields': {
                'company': company,
                'position': position,
                'salary': salary,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'document_type': 'EMPLOYMENT'
            },
            'complexity': 'medium',
            'industry': 'employment',
            'temporal_valid': True
        }
    
    def _generate_minimal_info_agreement(self) -> Dict[str, Any]:
        """Generate minimal information influencer agreement (edge case)."""
        brands = ['Brand X', 'Company ABC', 'Product Co']
        platforms = ['social media', 'online', 'digital platforms']
        
        brand = random.choice(brands)
        platform = random.choice(platforms)
        fee = random.randint(200, 800)
        
        text = f"""Agreement with {brand}

Content creation for {platform}.

Fee: ${fee}

Post about product. 

End."""
        
        return {
            'text': text,
            'fields': {
                'brand': brand,
                'platform': platform,
                'fee_numeric': fee,
                'deliverables': ['post about product'],
                'document_type': 'INFLUENCER_AGREEMENT'
            },
            'complexity': 'simple',
            'industry': 'other',
            'coherence': 0.3,  # Low coherence due to minimal info
            'business_valid': False,  # Fails business validation
            'temporal_valid': False,  # No temporal terms
            'style_metadata': {
                'style_profile_id': 'minimal',
                'formality_score': 0.1,
                'tone': 'informal'
            }
        }
    
    def _generate_wrong_industry_agreement(self) -> Dict[str, Any]:
        """Generate influencer agreement with mismatched industry-platform alignment."""
        # Intentionally mismatched combinations
        mismatches = [
            {'industry': 'tech', 'platforms': ['Pinterest', 'fashion blog'], 'product': 'enterprise software'},
            {'industry': 'construction', 'platforms': ['beauty Instagram', 'makeup tutorial'], 'product': 'heavy machinery'},
            {'industry': 'legal services', 'platforms': ['food TikTok', 'cooking channel'], 'product': 'legal consultation'},
            {'industry': 'medical', 'platforms': ['gaming stream', 'esports content'], 'product': 'pharmaceutical products'}
        ]
        
        mismatch = random.choice(mismatches)
        fee = random.randint(1000, 5000)
        
        text = f"""INFLUENCER PARTNERSHIP AGREEMENT

Brand: {mismatch['product'].title()} Solutions
Influencer: Content Creator

CAMPAIGN DETAILS:
Industry: {mismatch['industry'].title()}
Product: {mismatch['product']}
Platforms: {', '.join(mismatch['platforms'])}

DELIVERABLES:
- 2 posts featuring {mismatch['product']}
- 1 story highlighting key benefits
- Authentic review content

FEE: ${fee:,}
CAMPAIGN DURATION: 4 weeks
USAGE RIGHTS: 12 months

This creates an obvious mismatch between industry and platform targeting."""
        
        return {
            'text': text,
            'fields': {
                'industry': mismatch['industry'],
                'platforms': mismatch['platforms'],
                'product': mismatch['product'],
                'fee_numeric': fee,
                'deliverables': [f"2 posts featuring {mismatch['product']}", "1 story highlighting benefits"],
                'document_type': 'INFLUENCER_AGREEMENT'
            },
            'complexity': 'medium',
            'industry': mismatch['industry'],
            'coherence': 0.2,  # Very low coherence due to mismatch
            'business_valid': False,  # Fails industry-platform alignment
            'temporal_valid': True,
            'style_metadata': {
                'style_profile_id': 'mismatched',
                'formality_score': 0.6,
                'tone': 'formal'
            }
        }
    
    def _generate_international_agreement(self) -> Dict[str, Any]:
        """Generate international influencer agreement with different format/terms."""
        countries = [
            {'name': 'United Kingdom', 'currency': 'GBP', 'amount': random.randint(800, 3000)},
            {'name': 'Canada', 'currency': 'CAD', 'amount': random.randint(1000, 4000)},
            {'name': 'Singapore', 'currency': 'SGD', 'amount': random.randint(1200, 5000)},
            {'name': 'Germany', 'currency': 'EUR', 'amount': random.randint(900, 3500)}
        ]
        
        country = random.choice(countries)
        brands = ['GlobalFashion Ltd', 'EuroTech Solutions', 'International Beauty Co', 'WorldWide Home Goods']
        
        brand = random.choice(brands)
        
        text = f"""INFLUENCER COLLABORATION AGREEMENT
{country['name']} Format

PARTIES:
Brand: {brand} ({country['name']})
Content Creator: Influencer Partner

COLLABORATION TERMS:
Compensation: {country['amount']} {country['currency']}
Content Requirements: As per local regulations
Platform Usage: Subject to {country['name']} digital marketing laws

DELIVERABLES:
- Content creation per agreed schedule
- Compliance with local advertising standards
- Proper disclosure per {country['name']} regulations

DURATION: Campaign period as specified
GOVERNING LAW: Laws of {country['name']}

This agreement follows {country['name']} influencer marketing guidelines and may not align with Australian standards."""
        
        return {
            'text': text,
            'fields': {
                'brand': brand,
                'country': country['name'],
                'currency': country['currency'],
                'fee_numeric': country['amount'],
                'deliverables': ['content creation', 'compliance requirements'],
                'document_type': 'INFLUENCER_AGREEMENT'
            },
            'complexity': 'medium',
            'industry': 'international',
            'coherence': 0.5,  # Moderate coherence but format differences
            'business_valid': False,  # May fail local business validation
            'temporal_valid': True,
            'style_metadata': {
                'style_profile_id': 'international',
                'formality_score': 0.7,
                'tone': 'formal'
            }
        }
    
    def _generate_b2b_data(self, template_data: Dict) -> Dict[str, Any]:
        """Generate B2B influencer data."""
        return {
            'influencer': self._generate_person_name(),
            'client': self._generate_company_name(),
            'industry': random.choice(template_data['industries']),
            'deliverables': '\n'.join(random.choice(template_data['deliverables'])),
            'fee': f"${random.randint(3000, 20000):,}",
            'exclusivity': f"{random.randint(2, 6)} months in {random.choice(template_data['industries'])}"
        }
    
    def _generate_incomplete_data(self, template_data: Dict) -> Dict[str, Any]:
        """Generate incomplete agreement data."""
        brands = ['StartupCo', 'NewBrand', 'LocalBusiness', 'TechFirm']
        
        return {
            'influencer': self._generate_person_name(),
            'brand': random.choice(brands),
            'partial_info': random.choice(template_data['partial_info'])
        }
    
    def _generate_person_name(self) -> str:
        """Generate a realistic person name."""
        first_names = ['Alex', 'Jordan', 'Taylor', 'Casey', 'Morgan', 'Riley', 'Avery', 'Quinn',
                      'Sarah', 'Mike', 'Emma', 'James', 'Lisa', 'David', 'Rachel', 'Tom']
        last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Miller', 'Moore', 'Taylor',
                     'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia']
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_company_name(self) -> str:
        """Generate a realistic company name."""
        prefixes = ['Tech', 'Digital', 'Creative', 'Smart', 'Modern', 'Global', 'Elite', 'Prime']
        suffixes = ['Solutions', 'Media', 'Group', 'Corp', 'Agency', 'Studio', 'Labs', 'Works']
        
        return f"{random.choice(prefixes)} {random.choice(suffixes)} Pty Ltd"
    
    def _generate_address(self) -> str:
        """Generate a realistic address."""
        street_numbers = range(1, 999)
        street_names = ['Collins', 'Bourke', 'Flinders', 'Spencer', 'Elizabeth', 'Queen', 'King', 'George']
        street_types = ['Street', 'Avenue', 'Road', 'Lane', 'Place']
        suburbs = ['Melbourne', 'Sydney', 'Brisbane', 'Perth', 'Adelaide', 'Darwin', 'Hobart']
        
        return f"{random.choice(street_numbers)} {random.choice(street_names)} {random.choice(street_types)}, {random.choice(suburbs)}"
    
    def _generate_recent_date(self) -> str:
        """Generate a recent date string."""
        base_date = datetime.now()
        random_days = random.randint(-30, 30)
        date = base_date + timedelta(days=random_days)
        return date.strftime("%B %d, %Y")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get OOD generation statistics."""
        stats = self.stats.copy()
        if stats['ood_samples_generated'] > 0:
            stats['negative_ratio'] = stats['negative_samples'] / stats['ood_samples_generated']
            stats['edge_case_ratio'] = stats['edge_case_samples'] / stats['ood_samples_generated']
        else:
            stats['negative_ratio'] = 0.0
            stats['edge_case_ratio'] = 0.0
        
        return stats 

    def _generate_supplier_sample(self) -> Dict[str, Any]:
        """Generate supplier agreement sample."""
        client_companies = ['Retail Corp Pty Ltd', 'Fashion House Australia', 'Beauty Brand Co', 'Tech Startup Inc']
        supplier_companies = ['Creative Agency Solutions', 'Production House Media', 'Marketing Firm Plus', 'Design Studio Pro']
        services = [
            'Video production services',
            'Photography and editing services',
            'Marketing campaign development',
            'Brand strategy consulting',
            'Content creation services'
        ]
        
        client = random.choice(client_companies)
        supplier = random.choice(supplier_companies)
        service = random.choice(services)
        fee = random.randint(5000, 25000)
        duration = random.choice(['3 months', '6 months', '12 months'])
        
        text = f"""SUPPLIER AGREEMENT

Agreement between {client} ("Client") and {supplier} ("Supplier").

SERVICES: {service}
PAYMENT TERMS: Net 30 days
DURATION: {duration}

SCOPE OF SERVICES:
Professional {service.lower()} including planning, execution, and delivery of agreed outcomes as per industry standards.

PRICING:
- Service fee: ${fee:,}
- Payment schedule: Monthly invoicing
- Additional costs as agreed in writing

TERMS AND CONDITIONS:
- Supplier warrants professional service delivery
- Client responsible for timely payment and provision of required materials
- Either party may terminate with 30 days written notice
- All intellectual property remains with Client upon payment

Signed: {datetime.now().strftime('%B %d, %Y')}"""
        
        return {
            'text': text,
            'fields': {
                'client_company': client,
                'supplier_company': supplier,
                'services': service,
                'fee_numeric': fee,
                'duration': duration,
                'document_type': 'SUPPLIER_AGREEMENT'
            },
            'complexity': 'medium',
            'industry': 'business_services',
            'temporal_valid': True
        }
    
    def _generate_real_estate_sample(self) -> Dict[str, Any]:
        """Generate real estate lease agreement sample."""
        addresses = [
            '123 Collins Street, Melbourne VIC 3000',
            '456 George Street, Sydney NSW 2000', 
            '789 Queen Street, Brisbane QLD 4000',
            '321 King William Street, Adelaide SA 5000'
        ]
        
        property_types = ['2-bedroom apartment', '3-bedroom house', '1-bedroom unit', 'studio apartment']
        
        address = random.choice(addresses)
        property_type = random.choice(property_types)
        rent = random.randint(350, 1200)
        bond = rent * 4
        lease_months = random.choice([6, 12, 18, 24])
        
        text = f"""RESIDENTIAL LEASE AGREEMENT

PROPERTY: {address}
DESCRIPTION: {property_type.title()}

PARTIES:
Landlord: Property Management Solutions Pty Ltd
Tenant: Residential Tenant

LEASE TERMS:
Weekly Rent: ${rent}
Bond: ${bond} (4 weeks rent)
Lease Period: {lease_months} months
Commencement: {(datetime.now() + timedelta(days=14)).strftime('%B %d, %Y')}

PROPERTY CONDITIONS:
- Unfurnished residential property
- Includes standard fixtures and fittings
- Garden maintenance responsibility of tenant
- No pets without written consent

UTILITIES:
Tenant responsible for electricity, gas, water usage, and internet connection.

TERMINATION:
Standard notice periods apply as per residential tenancy legislation.

This agreement is governed by state residential tenancy laws."""
        
        return {
            'text': text,
            'fields': {
                'property_address': address,
                'property_type': property_type,
                'weekly_rent': rent,
                'bond_amount': bond,
                'lease_months': lease_months,
                'document_type': 'LEASE_AGREEMENT'
            },
            'complexity': 'medium',
            'industry': 'real_estate',
            'temporal_valid': True
        }
    
    def _generate_insurance_sample(self) -> Dict[str, Any]:
        """Generate insurance policy sample."""
        insurance_types = [
            {'type': 'Professional Indemnity', 'coverage': 2000000, 'premium': 1200},
            {'type': 'Public Liability', 'coverage': 5000000, 'premium': 800},
            {'type': 'Business Insurance', 'coverage': 1000000, 'premium': 1500},
            {'type': 'Product Liability', 'coverage': 3000000, 'premium': 950}
        ]
        
        companies = ['Australian Insurance Group', 'Professional Cover Pty Ltd', 'Business Protection Insurance']
        
        insurance = random.choice(insurance_types)
        company = random.choice(companies)
        policy_number = f"POL{random.randint(100000, 999999)}"
        
        text = f"""INSURANCE POLICY

INSURER: {company}
POLICY NUMBER: {policy_number}
POLICY TYPE: {insurance['type']} Insurance

INSURED PARTY: Business Entity/Professional

COVERAGE DETAILS:
Coverage Limit: ${insurance['coverage']:,}
Annual Premium: ${insurance['premium']:,}
Policy Period: 12 months from issue date

COVERED RISKS:
This policy provides protection against professional risks, claims, and liabilities arising from business operations within the defined scope.

CONDITIONS:
- Policy subject to terms and conditions document
- Claims must be reported within specified timeframes
- Deductible applies as per policy schedule
- Regular premium payments required to maintain coverage

EXCLUSIONS:
Standard exclusions apply including intentional acts, criminal activity, and circumstances outside policy scope.

This policy is issued subject to Australian insurance regulations."""
        
        return {
            'text': text,
            'fields': {
                'insurer': company,
                'policy_number': policy_number,
                'insurance_type': insurance['type'],
                'coverage_amount': insurance['coverage'],
                'annual_premium': insurance['premium'],
                'document_type': 'INSURANCE_POLICY'
            },
            'complexity': 'medium',
            'industry': 'insurance',
            'temporal_valid': True
        }
    
    def _generate_legal_services_sample(self) -> Dict[str, Any]:
        """Generate legal services retainer agreement."""
        law_firms = ['Legal Partners Pty Ltd', 'Corporate Law Solutions', 'Business Legal Services', 'Professional Law Group']
        service_types = [
            'Corporate legal services',
            'Contract review and drafting',
            'Intellectual property advice',
            'Employment law consultation',
            'Commercial dispute resolution'
        ]
        
        firm = random.choice(law_firms)
        service = random.choice(service_types)
        hourly_rate = random.choice([350, 420, 500, 650])
        retainer = random.randint(2000, 10000)
        
        text = f"""LEGAL SERVICES RETAINER AGREEMENT

LAW FIRM: {firm}
CLIENT: Business Client Entity

SERVICES: {service}

ENGAGEMENT TERMS:
Hourly Rate: ${hourly_rate}
Retainer Amount: ${retainer:,}
Billing Cycle: Monthly

SCOPE OF SERVICES:
{service} including advice, document preparation, correspondence, and representation as required within the defined legal matter scope.

RETAINER CONDITIONS:
- Retainer held in trust account
- Applied against fees and costs as incurred
- Monthly statements provided
- Top-up required when retainer depleted

TERMS:
- Professional standards and ethics apply
- Client responsible for timely instruction provision
- Fee estimates provided for significant matters
- Either party may terminate engagement with notice

This agreement is subject to legal profession regulations and client care rules."""
        
        return {
            'text': text,
            'fields': {
                'law_firm': firm,
                'service_type': service,
                'hourly_rate': hourly_rate,
                'retainer_amount': retainer,
                'document_type': 'LEGAL_RETAINER'
            },
            'complexity': 'complex',
            'industry': 'legal_services',
            'temporal_valid': True
        }
    
    def _generate_incomplete_agreement(self) -> Dict[str, Any]:
        """Generate incomplete influencer agreement (edge case)."""
        brands = ['Trendy Fashion', 'Beauty Essentials', 'Lifestyle Co']
        
        brand = random.choice(brands)
        
        text = f"""DRAFT - Collaboration Agreement

Brand: {brand}

Hey! Looking forward to working together.

Details:
- Post something about our products
- We'll send you stuff
- Let us know your rates

Timeline: Sometime next month?

Usage: 

Exclusivity: 

Payment: To be discussed

Contact: marketing@{brand.lower().replace(' ', '')}.com

Note: This is just a draft - final terms TBD"""
        
        return {
            'text': text,
            'fields': {
                'brand': brand,
                'contact_email': f"marketing@{brand.lower().replace(' ', '')}.com",
                'deliverables': ['post about products'],
                'status': 'DRAFT',
                'document_type': 'INFLUENCER_AGREEMENT'
            },
            'complexity': 'simple',
            'industry': 'other',
            'coherence': 0.3,  # Low coherence due to incomplete terms
            'business_valid': False,  # Fails business validation
            'temporal_valid': False,  # No temporal terms
            'style_metadata': {
                'style_profile_id': 'incomplete',
                'formality_score': 0.2,
                'tone': 'casual'
            }
        }
    
    def _generate_unusual_platform_agreement(self) -> Dict[str, Any]:
        """Generate agreement for unusual/niche platforms (edge case)."""
        unusual_platforms = [
            {'platform': 'Clubhouse', 'content': 'audio room hosting'},
            {'platform': 'Discord', 'content': 'community server management'},
            {'platform': 'Twitch', 'content': 'live streaming sessions'},
            {'platform': 'OnlyFans', 'content': 'exclusive content creation'},
            {'platform': 'Patreon', 'content': 'subscription content'},
            {'platform': 'Medium', 'content': 'thought leadership articles'},
            {'platform': 'Substack', 'content': 'newsletter writing'}
        ]
        
        brands = ['Innovative Tech Brand', 'Digital Media Company', 'Creative Platform Solutions']
        
        platform_info = random.choice(unusual_platforms)
        brand = random.choice(brands)
        fee = random.randint(500, 3000)
        
        text = f"""DIGITAL PLATFORM COLLABORATION

Brand: {brand}
Platform: {platform_info['platform']}
Content Creator: Digital Influencer

CAMPAIGN OVERVIEW:
Platform-specific {platform_info['content']} for brand awareness and engagement within the {platform_info['platform']} community.

DELIVERABLES:
- {platform_info['content']} featuring brand messaging
- Platform-native content format
- Community engagement and interaction

COMPENSATION: ${fee:,}

PLATFORM REQUIREMENTS:
Content must comply with {platform_info['platform']} community guidelines and terms of service.

DURATION: 4-week campaign period

NOTE: This agreement covers non-traditional social media platforms and may require specialized knowledge of platform dynamics."""
        
        return {
            'text': text,
            'fields': {
                'brand': brand,
                'platform': platform_info['platform'],
                'content_type': platform_info['content'],
                'fee_numeric': fee,
                'deliverables': [f"{platform_info['content']} featuring brand"],
                'document_type': 'INFLUENCER_AGREEMENT'
            },
            'complexity': 'medium',
            'industry': 'digital_media',
            'coherence': 0.6,  # Moderate coherence but unusual platform
            'business_valid': False,  # May fail platform alignment validation
            'temporal_valid': True,
            'style_metadata': {
                'style_profile_id': 'digital_native',
                'formality_score': 0.5,
                'tone': 'professional_casual'
            }
        } 