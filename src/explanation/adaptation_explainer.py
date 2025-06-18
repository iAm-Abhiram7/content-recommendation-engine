"""
Adaptation Explainer for Content Recommendation Engine

This module provides explainable AI capabilities for recommendation adaptations,
generating human-readable explanations for why and how recommendations change.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict

from ..utils.logging import setup_logger


class ExplanationType(Enum):
    """Types of explanations"""
    FEATURE_IMPORTANCE = "feature_importance"
    PREFERENCE_CHANGE = "preference_change"
    TEMPORAL_PATTERN = "temporal_pattern"
    COLLABORATIVE_SIGNAL = "collaborative_signal"
    CONTENT_SIMILARITY = "content_similarity"
    EXPLORATION_EXPLOITATION = "exploration_exploitation"
    DRIFT_ADAPTATION = "drift_adaptation"
    SEASONAL_ADJUSTMENT = "seasonal_adjustment"


class ExplanationLevel(Enum):
    """Levels of explanation detail"""
    BRIEF = "brief"           # One sentence
    SUMMARY = "summary"       # Paragraph
    DETAILED = "detailed"     # Multiple paragraphs
    TECHNICAL = "technical"   # Technical details


@dataclass
class ExplanationComponent:
    """Individual explanation component"""
    type: ExplanationType
    importance: float  # 0-1
    description: str
    evidence: Dict[str, Any]
    confidence: float  # 0-1


@dataclass
class AdaptationExplanation:
    """Complete explanation for an adaptation"""
    adaptation_id: str
    user_id: str
    timestamp: datetime
    adaptation_type: str
    components: List[ExplanationComponent]
    overall_explanation: str
    key_factors: List[str]
    confidence_score: float
    recommendation_changes: Dict[str, Any]


class AdaptationExplainer:
    """
    Generates explanations for recommendation adaptations using
    multiple explanation strategies and evidence sources
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        self.min_importance_threshold = self.config.get('min_importance_threshold', 0.1)
        self.max_components = self.config.get('max_components', 5)
        self.explanation_templates = self._load_explanation_templates()
        
        # State
        self.explanation_history: List[AdaptationExplanation] = []
        self.user_explanation_preferences: Dict[str, Dict[str, Any]] = {}
        
        self.logger = setup_logger(__name__)
    
    def explain_adaptation(self, 
                          adaptation_data: Dict[str, Any],
                          user_context: Dict[str, Any] = None,
                          level: ExplanationLevel = ExplanationLevel.SUMMARY) -> AdaptationExplanation:
        """Generate explanation for a recommendation adaptation"""
        try:
            user_id = adaptation_data.get('user_id')
            adaptation_id = adaptation_data.get('adaptation_id', f"adapt_{int(datetime.now().timestamp())}")
            
            # Extract explanation components
            components = self._extract_explanation_components(adaptation_data, user_context)
            
            # Filter and rank components
            significant_components = self._filter_and_rank_components(components)
            
            # Generate overall explanation
            overall_explanation = self._generate_overall_explanation(
                significant_components, level, user_context
            )
            
            # Extract key factors
            key_factors = self._extract_key_factors(significant_components)
            
            # Calculate confidence
            confidence_score = self._calculate_explanation_confidence(significant_components)
            
            # Get recommendation changes
            rec_changes = self._summarize_recommendation_changes(adaptation_data)
            
            explanation = AdaptationExplanation(
                adaptation_id=adaptation_id,
                user_id=user_id,
                timestamp=datetime.now(),
                adaptation_type=adaptation_data.get('type', 'unknown'),
                components=significant_components,
                overall_explanation=overall_explanation,
                key_factors=key_factors,
                confidence_score=confidence_score,
                recommendation_changes=rec_changes
            )
            
            # Store explanation
            self.explanation_history.append(explanation)
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return self._default_explanation(adaptation_data)
    
    def explain_recommendation_set(self,
                                 recommendations: List[Dict[str, Any]],
                                 user_context: Dict[str, Any] = None,
                                 level: ExplanationLevel = ExplanationLevel.BRIEF) -> List[Dict[str, Any]]:
        """Generate explanations for a set of recommendations"""
        explanations = []
        
        for i, rec in enumerate(recommendations):
            try:
                # Generate individual explanation
                explanation = self._explain_single_recommendation(rec, user_context, level)
                explanations.append({
                    'recommendation_id': rec.get('id', f'rec_{i}'),
                    'explanation': explanation,
                    'confidence': rec.get('confidence', 0.5)
                })
            except Exception as e:
                self.logger.error(f"Error explaining recommendation {i}: {e}")
                explanations.append({
                    'recommendation_id': rec.get('id', f'rec_{i}'),
                    'explanation': 'Unable to generate explanation',
                    'confidence': 0.0
                })
        
        return explanations
    
    def explain_preference_change(self,
                                old_preferences: Dict[str, float],
                                new_preferences: Dict[str, float],
                                change_evidence: Dict[str, Any],
                                level: ExplanationLevel = ExplanationLevel.SUMMARY) -> str:
        """Explain changes in user preferences"""
        try:
            # Calculate preference changes
            changes = {}
            all_features = set(old_preferences.keys()) | set(new_preferences.keys())
            
            for feature in all_features:
                old_val = old_preferences.get(feature, 0.0)
                new_val = new_preferences.get(feature, 0.0)
                change = new_val - old_val
                
                if abs(change) > 0.1:  # Significant change threshold
                    changes[feature] = {
                        'old_value': old_val,
                        'new_value': new_val,
                        'change': change,
                        'percentage_change': (change / max(old_val, 0.01)) * 100
                    }
            
            # Sort by magnitude of change
            sorted_changes = sorted(
                changes.items(),
                key=lambda x: abs(x[1]['change']),
                reverse=True
            )
            
            # Generate explanation based on level
            if level == ExplanationLevel.BRIEF:
                return self._generate_brief_preference_explanation(sorted_changes[:2])
            elif level == ExplanationLevel.SUMMARY:
                return self._generate_summary_preference_explanation(sorted_changes[:3], change_evidence)
            else:
                return self._generate_detailed_preference_explanation(sorted_changes, change_evidence)
                
        except Exception as e:
            self.logger.error(f"Error explaining preference change: {e}")
            return "Your preferences have been updated based on your recent activity."
    
    def _extract_explanation_components(self, adaptation_data: Dict[str, Any],
                                      user_context: Dict[str, Any] = None) -> List[ExplanationComponent]:
        """Extract individual explanation components"""
        components = []
        
        # Feature importance component
        if 'feature_importance' in adaptation_data:
            importance_data = adaptation_data['feature_importance']
            top_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_features:
                components.append(ExplanationComponent(
                    type=ExplanationType.FEATURE_IMPORTANCE,
                    importance=0.8,
                    description=f"The most influential factors were {', '.join([f[0] for f in top_features])}",
                    evidence={'top_features': top_features},
                    confidence=0.9
                ))
        
        # Preference change component
        if 'preference_changes' in adaptation_data:
            pref_changes = adaptation_data['preference_changes']
            significant_changes = [f for f, change in pref_changes.items() if abs(change) > 0.2]
            
            if significant_changes:
                components.append(ExplanationComponent(
                    type=ExplanationType.PREFERENCE_CHANGE,
                    importance=0.9,
                    description=f"Your preferences for {', '.join(significant_changes[:2])} have changed",
                    evidence={'changed_preferences': significant_changes},
                    confidence=0.85
                ))
        
        # Temporal pattern component
        if 'temporal_patterns' in adaptation_data:
            temporal_info = adaptation_data['temporal_patterns']
            
            components.append(ExplanationComponent(
                type=ExplanationType.TEMPORAL_PATTERN,
                importance=0.6,
                description=f"Time-based patterns show {temporal_info.get('pattern_type', 'changes')}",
                evidence=temporal_info,
                confidence=0.7
            ))
        
        # Drift adaptation component
        if 'drift_detected' in adaptation_data and adaptation_data['drift_detected']:
            drift_info = adaptation_data.get('drift_info', {})
            
            components.append(ExplanationComponent(
                type=ExplanationType.DRIFT_ADAPTATION,
                importance=0.7,
                description=f"Adapting to changes in your behavior patterns",
                evidence=drift_info,
                confidence=0.8
            ))
        
        # Collaborative signal component
        if 'collaborative_signals' in adaptation_data:
            collab_info = adaptation_data['collaborative_signals']
            
            components.append(ExplanationComponent(
                type=ExplanationType.COLLABORATIVE_SIGNAL,
                importance=0.5,
                description="Based on users with similar preferences",
                evidence=collab_info,
                confidence=0.6
            ))
        
        # Exploration component
        if 'exploration_factor' in adaptation_data:
            exploration = adaptation_data['exploration_factor']
            
            if exploration > 0.3:
                components.append(ExplanationComponent(
                    type=ExplanationType.EXPLORATION_EXPLOITATION,
                    importance=exploration,
                    description="Including diverse recommendations to help you discover new interests",
                    evidence={'exploration_factor': exploration},
                    confidence=0.7
                ))
        
        return components
    
    def _filter_and_rank_components(self, components: List[ExplanationComponent]) -> List[ExplanationComponent]:
        """Filter and rank explanation components by importance"""
        # Filter by minimum importance
        filtered = [c for c in components if c.importance >= self.min_importance_threshold]
        
        # Sort by importance
        ranked = sorted(filtered, key=lambda x: x.importance, reverse=True)
        
        # Limit to max components
        return ranked[:self.max_components]
    
    def _generate_overall_explanation(self, components: List[ExplanationComponent],
                                    level: ExplanationLevel,
                                    user_context: Dict[str, Any] = None) -> str:
        """Generate overall explanation text"""
        if not components:
            return "Recommendations updated based on your recent activity."
        
        # Get user's preferred explanation style
        user_id = user_context.get('user_id') if user_context else None
        explanation_prefs = self.user_explanation_preferences.get(user_id, {})
        
        if level == ExplanationLevel.BRIEF:
            return self._generate_brief_explanation(components)
        elif level == ExplanationLevel.SUMMARY:
            return self._generate_summary_explanation(components, explanation_prefs)
        elif level == ExplanationLevel.DETAILED:
            return self._generate_detailed_explanation(components, explanation_prefs)
        else:  # TECHNICAL
            return self._generate_technical_explanation(components)
    
    def _generate_brief_explanation(self, components: List[ExplanationComponent]) -> str:
        """Generate brief one-sentence explanation"""
        if not components:
            return "Recommendations updated."
        
        primary_component = components[0]
        
        templates = {
            ExplanationType.PREFERENCE_CHANGE: "Updated recommendations based on your changing preferences.",
            ExplanationType.FEATURE_IMPORTANCE: "Recommendations adjusted based on key factors in your preferences.",
            ExplanationType.DRIFT_ADAPTATION: "Recommendations adapted to your evolving interests.",
            ExplanationType.TEMPORAL_PATTERN: "Recommendations updated based on your usage patterns.",
            ExplanationType.COLLABORATIVE_SIGNAL: "Recommendations improved using similar users' preferences.",
            ExplanationType.EXPLORATION_EXPLOITATION: "Added diverse recommendations to help you explore new content."
        }
        
        return templates.get(primary_component.type, "Recommendations have been updated.")
    
    def _generate_summary_explanation(self, components: List[ExplanationComponent],
                                    user_prefs: Dict[str, Any] = None) -> str:
        """Generate summary paragraph explanation"""
        if not components:
            return "Your recommendations have been updated based on your recent activity and preferences."
        
        explanation_parts = []
        
        # Start with primary reason
        primary = components[0]
        
        if primary.type == ExplanationType.PREFERENCE_CHANGE:
            explanation_parts.append("We've noticed changes in your preferences")
        elif primary.type == ExplanationType.DRIFT_ADAPTATION:
            explanation_parts.append("Your interests appear to be evolving")
        elif primary.type == ExplanationType.TEMPORAL_PATTERN:
            explanation_parts.append("Based on your usage patterns")
        else:
            explanation_parts.append("Based on your recent activity")
        
        # Add supporting reasons
        supporting_types = [c.type for c in components[1:3]]
        
        if ExplanationType.COLLABORATIVE_SIGNAL in supporting_types:
            explanation_parts.append("and insights from users with similar tastes")
        
        if ExplanationType.EXPLORATION_EXPLOITATION in supporting_types:
            explanation_parts.append("We've also included some diverse options to help you discover new content")
        
        # Conclude
        explanation_parts.append("so we've updated your recommendations accordingly.")
        
        return " ".join(explanation_parts) + "."
    
    def _generate_detailed_explanation(self, components: List[ExplanationComponent],
                                     user_prefs: Dict[str, Any] = None) -> str:
        """Generate detailed multi-paragraph explanation"""
        paragraphs = []
        
        # Introduction
        paragraphs.append("Here's why your recommendations have changed:")
        
        # Explain each significant component
        for component in components:
            if component.importance > 0.5:
                paragraph = self._component_to_detailed_text(component)
                if paragraph:
                    paragraphs.append(paragraph)
        
        # Conclusion
        paragraphs.append(
            "These updates help ensure your recommendations stay relevant and interesting. "
            "You can adjust your preferences anytime in your settings."
        )
        
        return "\n\n".join(paragraphs)
    
    def _generate_technical_explanation(self, components: List[ExplanationComponent]) -> str:
        """Generate technical explanation with details"""
        sections = ["Technical Explanation of Recommendation Adaptation:"]
        
        for i, component in enumerate(components, 1):
            sections.append(f"{i}. {component.type.value.replace('_', ' ').title()}:")
            sections.append(f"   - Importance: {component.importance:.3f}")
            sections.append(f"   - Confidence: {component.confidence:.3f}")
            sections.append(f"   - Description: {component.description}")
            
            if component.evidence:
                sections.append(f"   - Evidence: {json.dumps(component.evidence, indent=6)}")
        
        return "\n".join(sections)
    
    def _component_to_detailed_text(self, component: ExplanationComponent) -> str:
        """Convert component to detailed text explanation"""
        templates = {
            ExplanationType.PREFERENCE_CHANGE: (
                "We detected significant changes in your preferences. "
                f"{component.description}. This suggests your interests are evolving, "
                "so we've adjusted your recommendations to better match your current tastes."
            ),
            ExplanationType.DRIFT_ADAPTATION: (
                "Your behavior patterns have shifted over time. "
                "Our system has detected this drift and adapted your recommendation model "
                "to better reflect your current interests rather than outdated preferences."
            ),
            ExplanationType.TEMPORAL_PATTERN: (
                f"Your usage patterns show specific trends: {component.description}. "
                "We've incorporated these temporal patterns to provide more timely "
                "and contextually relevant recommendations."
            ),
            ExplanationType.COLLABORATIVE_SIGNAL: (
                "We've found users with similar preferences to yours and incorporated "
                "insights from their behavior. This helps us recommend content that "
                "people with similar tastes have enjoyed."
            ),
            ExplanationType.EXPLORATION_EXPLOITATION: (
                "To help you discover new content, we've balanced recommendations between "
                "items we're confident you'll like and some diverse options that might "
                "introduce you to new interests."
            )
        }
        
        return templates.get(component.type, component.description)
    
    def _explain_single_recommendation(self, recommendation: Dict[str, Any],
                                     user_context: Dict[str, Any] = None,
                                     level: ExplanationLevel = ExplanationLevel.BRIEF) -> str:
        """Explain why a specific item was recommended"""
        try:
            # Extract explanation factors
            factors = []
            
            # Content similarity
            if 'content_similarity' in recommendation:
                similarity = recommendation['content_similarity']
                if similarity > 0.7:
                    factors.append(f"similar to content you've enjoyed")
            
            # User preference match
            if 'preference_match' in recommendation:
                match_score = recommendation['preference_match']
                if match_score > 0.8:
                    factors.append(f"matches your preferences well")
            
            # Popularity/trending
            if 'popularity_score' in recommendation:
                popularity = recommendation['popularity_score']
                if popularity > 0.7:
                    factors.append(f"popular among users like you")
            
            # Recent behavior
            if 'recent_behavior_match' in recommendation:
                recency = recommendation['recent_behavior_match']
                if recency > 0.6:
                    factors.append(f"aligns with your recent activity")
            
            # Exploration factor
            if 'exploration_factor' in recommendation:
                exploration = recommendation['exploration_factor']
                if exploration > 0.5:
                    factors.append(f"selected to help you explore new content")
            
            # Generate explanation based on factors
            if not factors:
                return "Recommended based on your profile"
            
            if level == ExplanationLevel.BRIEF:
                return f"Recommended because it's {factors[0]}"
            else:
                factor_text = ", ".join(factors[:-1])
                if len(factors) > 1:
                    factor_text += f", and {factors[-1]}"
                else:
                    factor_text = factors[0]
                
                return f"This item was recommended because it's {factor_text}."
                
        except Exception as e:
            self.logger.error(f"Error explaining single recommendation: {e}")
            return "Recommended for you"
    
    def _generate_brief_preference_explanation(self, changes: List[Tuple[str, Dict]]) -> str:
        """Generate brief preference change explanation"""
        if not changes:
            return "Your preferences have been updated."
        
        feature, change_data = changes[0]
        change_direction = "increased" if change_data['change'] > 0 else "decreased"
        
        return f"Your preference for {feature} has {change_direction}."
    
    def _generate_summary_preference_explanation(self, changes: List[Tuple[str, Dict]],
                                               evidence: Dict[str, Any] = None) -> str:
        """Generate summary preference change explanation"""
        if not changes:
            return "Your preferences have been updated based on your recent activity."
        
        explanation_parts = []
        
        # Main changes
        main_changes = []
        for feature, change_data in changes:
            direction = "increased" if change_data['change'] > 0 else "decreased"
            main_changes.append(f"your preference for {feature} has {direction}")
        
        if len(main_changes) == 1:
            explanation_parts.append(f"We noticed that {main_changes[0]}")
        else:
            explanation_parts.append(f"We noticed that {', '.join(main_changes[:-1])}, and {main_changes[-1]}")
        
        # Add evidence if available
        if evidence:
            if 'recent_interactions' in evidence:
                explanation_parts.append(f"based on your recent interactions")
            if 'feedback_signals' in evidence:
                explanation_parts.append(f"and your feedback")
        
        explanation_parts.append("Your recommendations have been updated accordingly.")
        
        return " ".join(explanation_parts) + "."
    
    def _generate_detailed_preference_explanation(self, changes: List[Tuple[str, Dict]],
                                                evidence: Dict[str, Any] = None) -> str:
        """Generate detailed preference change explanation"""
        paragraphs = []
        
        # Introduction
        paragraphs.append("Here's how your preferences have changed:")
        
        # Detail each significant change
        for feature, change_data in changes[:5]:  # Top 5 changes
            change_pct = abs(change_data['percentage_change'])
            direction = "increased" if change_data['change'] > 0 else "decreased"
            
            paragraph = (f"Your preference for {feature} has {direction} by "
                        f"{change_pct:.1f}%. This change ")
            
            if change_pct > 50:
                paragraph += "represents a significant shift in your interests."
            elif change_pct > 20:
                paragraph += "indicates a notable evolution in your preferences."
            else:
                paragraph += "shows a gradual adjustment in your tastes."
            
            paragraphs.append(paragraph)
        
        # Add evidence explanation
        if evidence:
            evidence_paragraph = "These changes were detected based on "
            evidence_sources = []
            
            if 'recent_interactions' in evidence:
                evidence_sources.append("your recent interactions")
            if 'feedback_signals' in evidence:
                evidence_sources.append("your explicit feedback")
            if 'behavioral_patterns' in evidence:
                evidence_sources.append("changes in your behavior patterns")
            
            if evidence_sources:
                evidence_paragraph += ", ".join(evidence_sources) + "."
                paragraphs.append(evidence_paragraph)
        
        # Conclusion
        paragraphs.append(
            "We've updated your recommendation model to reflect these preference changes, "
            "ensuring you receive more relevant and personalized suggestions."
        )
        
        return "\n\n".join(paragraphs)
    
    def _extract_key_factors(self, components: List[ExplanationComponent]) -> List[str]:
        """Extract key factors from explanation components"""
        factors = []
        
        for component in components[:3]:  # Top 3 components
            if component.type == ExplanationType.PREFERENCE_CHANGE:
                factors.append("Preference evolution")
            elif component.type == ExplanationType.DRIFT_ADAPTATION:
                factors.append("Behavior drift adaptation")
            elif component.type == ExplanationType.TEMPORAL_PATTERN:
                factors.append("Temporal patterns")
            elif component.type == ExplanationType.COLLABORATIVE_SIGNAL:
                factors.append("Similar user insights")
            elif component.type == ExplanationType.EXPLORATION_EXPLOITATION:
                factors.append("Exploration for discovery")
            else:
                factors.append(component.type.value.replace('_', ' ').title())
        
        return factors
    
    def _calculate_explanation_confidence(self, components: List[ExplanationComponent]) -> float:
        """Calculate overall confidence in the explanation"""
        if not components:
            return 0.5
        
        # Weight by importance and average confidence
        total_weight = sum(c.importance for c in components)
        if total_weight == 0:
            return 0.5
        
        weighted_confidence = sum(c.confidence * c.importance for c in components) / total_weight
        
        return weighted_confidence
    
    def _summarize_recommendation_changes(self, adaptation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize how recommendations changed"""
        changes = {}
        
        if 'before_recommendations' in adaptation_data and 'after_recommendations' in adaptation_data:
            before = adaptation_data['before_recommendations']
            after = adaptation_data['after_recommendations']
            
            # Calculate changes in recommendation categories
            changes['new_items'] = len([item for item in after if item not in before])
            changes['removed_items'] = len([item for item in before if item not in after])
            changes['reordered_items'] = len(set(before) & set(after)) - changes['new_items']
        
        if 'category_shifts' in adaptation_data:
            changes['category_changes'] = adaptation_data['category_shifts']
        
        if 'diversity_change' in adaptation_data:
            changes['diversity_delta'] = adaptation_data['diversity_change']
        
        return changes
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates"""
        return {
            'default': "Your recommendations have been updated based on your recent activity.",
            'preference_change': "We've updated your recommendations because your preferences for {features} have changed.",
            'drift_adaptation': "Your recommendations have been adapted to reflect changes in your behavior patterns.",
            'temporal_pattern': "Based on your usage patterns, we've adjusted your recommendations for better timing.",
            'exploration': "We've included diverse recommendations to help you discover new content you might enjoy."
        }
    
    def _default_explanation(self, adaptation_data: Dict[str, Any]) -> AdaptationExplanation:
        """Return default explanation when generation fails"""
        return AdaptationExplanation(
            adaptation_id=adaptation_data.get('adaptation_id', 'unknown'),
            user_id=adaptation_data.get('user_id', 'unknown'),
            timestamp=datetime.now(),
            adaptation_type=adaptation_data.get('type', 'unknown'),
            components=[],
            overall_explanation="Your recommendations have been updated based on your recent activity.",
            key_factors=['Recent activity'],
            confidence_score=0.5,
            recommendation_changes={}
        )
    
    def get_explanation_history(self, user_id: str, limit: int = 10) -> List[AdaptationExplanation]:
        """Get explanation history for a user"""
        user_explanations = [exp for exp in self.explanation_history if exp.user_id == user_id]
        return sorted(user_explanations, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def update_user_explanation_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user's explanation preferences"""
        self.user_explanation_preferences[user_id] = preferences
