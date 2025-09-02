"""
Mapping diagnostics module for tracking style profile failures.

This module captures detailed information about why style profile mappings fail,
including template names, content snippets, and failure reasons.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MappingFailure:
    """Represents a single mapping failure instance."""
    
    def __init__(self, 
                 failure_type: str,
                 template_name: Optional[str] = None,
                 content_snippet: Optional[str] = None,
                 profile_id: Optional[str] = None,
                 target_industry: Optional[str] = None,
                 target_complexity: Optional[str] = None,
                 reason: Optional[str] = None,
                 additional_data: Optional[Dict[str, Any]] = None):
        """
        Initialize a mapping failure record.
        
        Args:
            failure_type: Type of failure ('industry_unknown', 'complexity_unknown', 'profile_selection_failed', etc.)
            template_name: Name of the source template
            content_snippet: Excerpt of the content that failed mapping
            profile_id: ID of the style profile (if any)
            target_industry: Target industry that was requested
            target_complexity: Target complexity that was requested
            reason: Human-readable reason for the failure
            additional_data: Any additional diagnostic data
        """
        self.timestamp = datetime.now().isoformat()
        self.failure_type = failure_type
        self.template_name = template_name
        self.content_snippet = content_snippet[:200] if content_snippet else None  # Limit snippet length
        self.profile_id = profile_id
        self.target_industry = target_industry
        self.target_complexity = target_complexity
        self.reason = reason
        self.additional_data = additional_data or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert failure to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'failure_type': self.failure_type,
            'template_name': self.template_name,
            'content_snippet': self.content_snippet,
            'profile_id': self.profile_id,
            'target_industry': self.target_industry,
            'target_complexity': self.target_complexity,
            'reason': self.reason,
            'additional_data': self.additional_data
        }


class MappingDiagnostics:
    """
    Comprehensive mapping diagnostics system.
    
    Tracks all style profile mapping failures and provides detailed
    analysis for debugging purposes.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize mapping diagnostics.
        
        Args:
            cache_dir: Directory to store mapping failure logs
        """
        self.cache_dir = Path(cache_dir)
        self.failures = []
        self.session_stats = {
            'total_mappings_attempted': 0,
            'successful_mappings': 0,
            'failed_mappings': 0,
            'failure_types': {},
            'unknown_industries': 0,
            'unknown_complexities': 0,
            'profile_selection_failures': 0
        }
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized mapping diagnostics with cache dir: {cache_dir}")
    
    def record_template_mapping_failure(self, 
                                      template_name: str,
                                      template_text: str,
                                      profile_id: str,
                                      failed_attribute: str,
                                      attempted_value: Optional[str] = None,
                                      fallback_value: Optional[str] = None,
                                      scoring_data: Optional[Dict[str, Any]] = None):
        """
        Record a template-level mapping failure.
        
        Args:
            template_name: Name of the template file
            template_text: Full text content of the template
            profile_id: ID of the created profile
            failed_attribute: Which attribute failed ('industry' or 'complexity')
            attempted_value: Value that was attempted to be assigned
            fallback_value: Value that was assigned as fallback
            scoring_data: Data about scoring attempts (industry scores, complexity metrics, etc.)
        """
        # Create detailed failure reason
        if failed_attribute == 'industry':
            if not scoring_data or not scoring_data.get('industry_scores'):
                reason = f"No industry terms found in template text. Used fallback: '{fallback_value}'"
            else:
                industry_scores = scoring_data.get('industry_scores', {})
                best_score = max(industry_scores.values()) if industry_scores else 0
                reason = f"Low industry match confidence. Best score: {best_score}. Used fallback: '{fallback_value}'"
        elif failed_attribute == 'complexity':
            complexity_metrics = scoring_data.get('complexity_metrics', {}) if scoring_data else {}
            avg_sentence_length = complexity_metrics.get('avg_sentence_length', 'unknown')
            clause_density = complexity_metrics.get('clause_density', 'unknown')
            reason = f"Complexity determination failed. Metrics - Avg sentence length: {avg_sentence_length}, Clause density: {clause_density}. Used fallback: '{fallback_value}'"
        else:
            reason = f"Unknown attribute '{failed_attribute}' failed mapping. Used fallback: '{fallback_value}'"
        
        failure = MappingFailure(
            failure_type=f"template_{failed_attribute}_mapping",
            template_name=template_name,
            content_snippet=template_text,
            profile_id=profile_id,
            reason=reason,
            additional_data={
                'failed_attribute': failed_attribute,
                'attempted_value': attempted_value,
                'fallback_value': fallback_value,
                'scoring_data': scoring_data or {}
            }
        )
        
        self.failures.append(failure)
        self.session_stats['failed_mappings'] += 1
        
        if failed_attribute == 'industry':
            self.session_stats['unknown_industries'] += 1
        elif failed_attribute == 'complexity':
            self.session_stats['unknown_complexities'] += 1
        
        failure_type = f"{failed_attribute}_mapping_failure"
        self.session_stats['failure_types'][failure_type] = self.session_stats['failure_types'].get(failure_type, 0) + 1
        
        logger.warning(f"🔍 Template mapping failure recorded: {template_name} - {failed_attribute}")
        logger.debug(f"Failure reason: {reason}")
    
    def record_profile_selection_failure(self,
                                       target_industry: Optional[str],
                                       target_complexity: Optional[str],
                                       available_profiles: List[str],
                                       profile_scores: Dict[str, float],
                                       selected_profile: str,
                                       selection_reason: str):
        """
        Record a style profile selection issue.
        
        Args:
            target_industry: Requested industry
            target_complexity: Requested complexity  
            available_profiles: List of available profile IDs
            profile_scores: Scores assigned to each profile
            selected_profile: Profile that was actually selected
            selection_reason: Why this profile was selected (e.g., 'random_fallback', 'low_score_match')
        """
        # Determine if this is actually a failure
        is_failure = False
        reason = ""
        
        if not profile_scores or all(score == 0 for score in profile_scores.values()):
            is_failure = True
            reason = f"No profiles matched target criteria. Industry: {target_industry}, Complexity: {target_complexity}. Used random selection."
        elif max(profile_scores.values()) < 0.5:  # Low confidence threshold
            is_failure = True
            best_score = max(profile_scores.values())
            reason = f"Low confidence profile match. Best score: {best_score:.3f}. Target - Industry: {target_industry}, Complexity: {target_complexity}"
        
        if is_failure:
            failure = MappingFailure(
                failure_type="profile_selection_low_confidence",
                target_industry=target_industry,
                target_complexity=target_complexity,
                profile_id=selected_profile,
                reason=reason,
                additional_data={
                    'available_profiles': available_profiles,
                    'profile_scores': profile_scores,
                    'selection_reason': selection_reason,
                    'max_score': max(profile_scores.values()) if profile_scores else 0
                }
            )
            
            self.failures.append(failure)
            self.session_stats['profile_selection_failures'] += 1
            self.session_stats['failed_mappings'] += 1
            self.session_stats['failure_types']['profile_selection_failure'] = self.session_stats['failure_types'].get('profile_selection_failure', 0) + 1
            
            logger.warning(f"🔍 Profile selection failure: {reason}")
        
        self.session_stats['total_mappings_attempted'] += 1
    
    def record_successful_mapping(self, 
                                template_name: Optional[str] = None,
                                profile_id: Optional[str] = None,
                                industry: Optional[str] = None,
                                complexity: Optional[str] = None):
        """Record a successful mapping for statistics."""
        self.session_stats['successful_mappings'] += 1
        self.session_stats['total_mappings_attempted'] += 1
        
        logger.debug(f"✅ Successful mapping - Template: {template_name}, Profile: {profile_id}, Industry: {industry}, Complexity: {complexity}")
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get a summary of all mapping failures."""
        return {
            'total_failures': len(self.failures),
            'session_stats': self.session_stats,
            'failure_breakdown': {
                'by_type': self.session_stats['failure_types'],
                'unknown_industries': self.session_stats['unknown_industries'],
                'unknown_complexities': self.session_stats['unknown_complexities'],
                'profile_selection_failures': self.session_stats['profile_selection_failures']
            },
            'success_rate': (self.session_stats['successful_mappings'] / max(self.session_stats['total_mappings_attempted'], 1)) * 100
        }
    
    def save_failures_to_disk(self):
        """Save all mapping failures to JSON file."""
        failures_file = self.cache_dir / 'mapping_failures.json'
        
        # Prepare data for saving
        save_data = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'total_failures': len(self.failures),
                'session_stats': self.session_stats
            },
            'failures': [failure.to_dict() for failure in self.failures],
            'summary': self.get_failure_summary()
        }
        
        try:
            with open(failures_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved {len(self.failures)} mapping failures to {failures_file}")
            
            # Also save a summary file for quick reference
            summary_file = self.cache_dir / 'mapping_failures_summary.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_failure_summary(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved mapping failure summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save mapping failures: {e}")
    
    def load_previous_failures(self):
        """Load failures from previous sessions for analysis."""
        failures_file = self.cache_dir / 'mapping_failures.json'
        
        if failures_file.exists():
            try:
                with open(failures_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                previous_failures = data.get('failures', [])
                logger.info(f"📂 Loaded {len(previous_failures)} previous mapping failures for reference")
                
                return previous_failures
                
            except Exception as e:
                logger.warning(f"Failed to load previous mapping failures: {e}")
        
        return []
    
    def analyze_common_failure_patterns(self) -> Dict[str, Any]:
        """Analyze common patterns in mapping failures."""
        if not self.failures:
            return {'analysis': 'No failures to analyze'}
        
        # Group failures by type
        by_type = {}
        for failure in self.failures:
            failure_type = failure.failure_type
            if failure_type not in by_type:
                by_type[failure_type] = []
            by_type[failure_type].append(failure)
        
        # Analyze each type
        analysis = {
            'total_failures': len(self.failures),
            'by_type': {}
        }
        
        for failure_type, type_failures in by_type.items():
            analysis['by_type'][failure_type] = {
                'count': len(type_failures),
                'common_templates': [],
                'common_reasons': []
            }
            
            # Find most common templates for this failure type
            template_counts = {}
            reason_counts = {}
            
            for failure in type_failures:
                if failure.template_name:
                    template_counts[failure.template_name] = template_counts.get(failure.template_name, 0) + 1
                if failure.reason:
                    reason_counts[failure.reason] = reason_counts.get(failure.reason, 0) + 1
            
            # Get top 3 most common templates and reasons
            analysis['by_type'][failure_type]['common_templates'] = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            analysis['by_type'][failure_type]['common_reasons'] = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return analysis
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of mapping diagnostics.
        
        This method provides the interface expected by the pipeline.
        """
        summary = self.get_failure_summary()
        
        # Add additional summary information
        summary.update({
            "timestamp": datetime.now().isoformat(),
            "processing_time": getattr(self, 'processing_time', 0.0),
            "total_mappings_attempted": self.session_stats.get('total_mappings_attempted', 0),
            "successful_mappings": self.session_stats.get('successful_mappings', 0),
            "failed_mappings": self.session_stats.get('failed_mappings', 0),
            "common_failure_reasons": [
                reason for reason, _ in sorted(
                    self.session_stats.get('failure_types', {}).items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            ] if self.session_stats.get('failure_types') else []
        })
        
        return summary 