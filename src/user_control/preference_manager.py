"""
Preference Manager for Content Recommendation Engine

This module provides comprehensive user preference management capabilities,
allowing users to view, edit, and control their preference profiles.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict

from ..utils.logging import setup_logger


class PreferenceCategory(Enum):
    """Categories of preferences"""
    CONTENT_TYPE = "content_type"      # Movies, books, music, etc.
    GENRE = "genre"                    # Action, comedy, drama, etc.
    STYLE = "style"                    # Visual style, writing style, etc.
    THEME = "theme"                    # Themes and topics
    TEMPORAL = "temporal"              # Time-based preferences
    CONTEXTUAL = "contextual"          # Context-dependent preferences
    BEHAVIORAL = "behavioral"          # Interaction patterns
    QUALITY = "quality"                # Quality indicators
    DEMOGRAPHIC = "demographic"        # Age rating, target audience, etc.
    TECHNICAL = "technical"            # Technical aspects


class PreferenceWeight(Enum):
    """Weight levels for preferences"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


class PreferenceSource(Enum):
    """Sources of preference data"""
    EXPLICIT = "explicit"              # User directly specified
    IMPLICIT = "implicit"              # Inferred from behavior
    FEEDBACK = "feedback"              # From ratings/feedback
    SOCIAL = "social"                  # From social connections
    CONTEXTUAL = "contextual"          # From context
    SYSTEM = "system"                  # System defaults


@dataclass
class PreferenceItem:
    """Individual preference item"""
    feature: str
    value: float  # -1.0 to 1.0
    weight: float  # 0.0 to 1.0
    category: PreferenceCategory
    source: PreferenceSource
    confidence: float  # 0.0 to 1.0
    last_updated: datetime
    update_count: int = 0
    locked: bool = False  # User-locked preference
    notes: str = ""


@dataclass
class PreferenceProfile:
    """Complete user preference profile"""
    user_id: str
    preferences: Dict[str, PreferenceItem] = field(default_factory=dict)
    meta_preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    def __post_init__(self):
        if not self.preferences:
            self.preferences = {}
        if not self.meta_preferences:
            self.meta_preferences = {
                'diversity_preference': 0.5,
                'novelty_preference': 0.5,
                'popularity_bias': 0.5,
                'exploration_tendency': 0.5,
                'explanation_preference': 'medium'
            }


class PreferenceManager:
    """
    Manages user preference profiles with full CRUD operations,
    preference learning, and user control capabilities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # User preference profiles
        self.user_profiles: Dict[str, PreferenceProfile] = {}
        
        # Preference templates and defaults
        self.preference_templates = self._load_preference_templates()
        self.default_preferences = self._load_default_preferences()
        
        # Preference change history
        self.preference_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Configuration
        self.auto_learning_enabled = self.config.get('auto_learning_enabled', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_preferences_per_category = self.config.get('max_preferences_per_category', 50)
        
        self.logger = setup_logger(__name__)
    
    def get_user_profile(self, user_id: str) -> PreferenceProfile:
        """Get user's preference profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self._create_default_profile(user_id)
        return self.user_profiles[user_id]
    
    def update_preference(self, user_id: str, feature: str, value: float,
                         category: PreferenceCategory = None,
                         source: PreferenceSource = PreferenceSource.EXPLICIT,
                         confidence: float = 1.0,
                         weight: float = None) -> bool:
        """Update a specific preference"""
        try:
            profile = self.get_user_profile(user_id)
            
            # Validate value
            value = max(-1.0, min(1.0, value))
            confidence = max(0.0, min(1.0, confidence))
            
            # Get existing preference or create new
            if feature in profile.preferences:
                pref = profile.preferences[feature]
                
                # Check if preference is locked
                if pref.locked and source != PreferenceSource.EXPLICIT:
                    return False
                
                # Update existing preference
                old_value = pref.value
                pref.value = value
                pref.confidence = confidence
                pref.last_updated = datetime.now()
                pref.update_count += 1
                
                if weight is not None:
                    pref.weight = max(0.0, min(1.0, weight))
                
                # Record change in history
                self._record_preference_change(user_id, feature, old_value, value, source)
            else:
                # Create new preference
                pref = PreferenceItem(
                    feature=feature,
                    value=value,
                    weight=weight or 0.5,
                    category=category or self._infer_category(feature),
                    source=source,
                    confidence=confidence,
                    last_updated=datetime.now(),
                    update_count=1
                )
                
                profile.preferences[feature] = pref
                
                # Record creation in history
                self._record_preference_change(user_id, feature, None, value, source)
            
            # Update profile metadata
            profile.last_updated = datetime.now()
            profile.version += 1
            
            self.logger.debug(f"Updated preference {feature} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating preference: {e}")
            return False
    
    def get_preference(self, user_id: str, feature: str) -> Optional[PreferenceItem]:
        """Get a specific preference"""
        profile = self.get_user_profile(user_id)
        return profile.preferences.get(feature)
    
    def get_preferences_by_category(self, user_id: str, 
                                   category: PreferenceCategory) -> Dict[str, PreferenceItem]:
        """Get all preferences in a category"""
        profile = self.get_user_profile(user_id)
        return {
            feature: pref for feature, pref in profile.preferences.items()
            if pref.category == category
        }
    
    def get_top_preferences(self, user_id: str, limit: int = 10,
                           category: PreferenceCategory = None) -> List[Tuple[str, PreferenceItem]]:
        """Get top preferences by value and confidence"""
        profile = self.get_user_profile(user_id)
        
        # Filter by category if specified
        preferences = profile.preferences
        if category:
            preferences = {
                f: p for f, p in preferences.items() if p.category == category
            }
        
        # Sort by weighted score (value * confidence * weight)
        sorted_prefs = sorted(
            preferences.items(),
            key=lambda x: abs(x[1].value) * x[1].confidence * x[1].weight,
            reverse=True
        )
        
        return sorted_prefs[:limit]
    
    def remove_preference(self, user_id: str, feature: str) -> bool:
        """Remove a preference"""
        try:
            profile = self.get_user_profile(user_id)
            
            if feature in profile.preferences:
                pref = profile.preferences[feature]
                
                # Check if preference is locked
                if pref.locked:
                    return False
                
                old_value = pref.value
                del profile.preferences[feature]
                
                # Record removal in history
                self._record_preference_change(user_id, feature, old_value, None, 
                                             PreferenceSource.EXPLICIT)
                
                # Update profile metadata
                profile.last_updated = datetime.now()
                profile.version += 1
                
                self.logger.debug(f"Removed preference {feature} for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing preference: {e}")
            return False
    
    def lock_preference(self, user_id: str, feature: str) -> bool:
        """Lock a preference to prevent automatic updates"""
        try:
            profile = self.get_user_profile(user_id)
            
            if feature in profile.preferences:
                profile.preferences[feature].locked = True
                profile.last_updated = datetime.now()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error locking preference: {e}")
            return False
    
    def unlock_preference(self, user_id: str, feature: str) -> bool:
        """Unlock a preference to allow automatic updates"""
        try:
            profile = self.get_user_profile(user_id)
            
            if feature in profile.preferences:
                profile.preferences[feature].locked = False
                profile.last_updated = datetime.now()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error unlocking preference: {e}")
            return False
    
    def bulk_update_preferences(self, user_id: str, 
                               preferences: Dict[str, Dict[str, Any]]) -> bool:
        """Update multiple preferences at once"""
        try:
            success_count = 0
            
            for feature, pref_data in preferences.items():
                success = self.update_preference(
                    user_id=user_id,
                    feature=feature,
                    value=pref_data.get('value', 0.0),
                    category=PreferenceCategory(pref_data.get('category', 'content_type')),
                    source=PreferenceSource(pref_data.get('source', 'explicit')),
                    confidence=pref_data.get('confidence', 1.0),
                    weight=pref_data.get('weight', 0.5)
                )
                
                if success:
                    success_count += 1
            
            self.logger.info(f"Bulk updated {success_count}/{len(preferences)} preferences for user {user_id}")
            return success_count == len(preferences)
            
        except Exception as e:
            self.logger.error(f"Error in bulk update: {e}")
            return False
    
    def get_preference_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's preferences"""
        profile = self.get_user_profile(user_id)
        
        # Count by category
        category_counts = defaultdict(int)
        category_avg_values = defaultdict(list)
        source_counts = defaultdict(int)
        
        for pref in profile.preferences.values():
            category_counts[pref.category.value] += 1
            category_avg_values[pref.category.value].append(pref.value)
            source_counts[pref.source.value] += 1
        
        # Calculate averages
        category_averages = {
            cat: np.mean(values) for cat, values in category_avg_values.items()
        }
        
        # Get strongest preferences
        strong_positive = [(f, p) for f, p in profile.preferences.items() 
                          if p.value > 0.7 and p.confidence > 0.7]
        strong_negative = [(f, p) for f, p in profile.preferences.items() 
                          if p.value < -0.7 and p.confidence > 0.7]
        
        return {
            'total_preferences': len(profile.preferences),
            'preferences_by_category': dict(category_counts),
            'preferences_by_source': dict(source_counts),
            'category_averages': category_averages,
            'strong_positive_count': len(strong_positive),
            'strong_negative_count': len(strong_negative),
            'locked_preferences_count': sum(1 for p in profile.preferences.values() if p.locked),
            'meta_preferences': profile.meta_preferences,
            'profile_age_days': (datetime.now() - profile.created_at).days,
            'last_updated': profile.last_updated.isoformat()
        }
    
    def get_preference_trends(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get preference change trends over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_changes = [
            change for change in self.preference_history.get(user_id, [])
            if change['timestamp'] > cutoff_date
        ]
        
        if not recent_changes:
            return {}
        
        # Analyze trends
        trending_up = []
        trending_down = []
        new_preferences = []
        removed_preferences = []
        
        for change in recent_changes:
            if change['old_value'] is None:
                new_preferences.append(change['feature'])
            elif change['new_value'] is None:
                removed_preferences.append(change['feature'])
            else:
                change_magnitude = change['new_value'] - change['old_value']
                if change_magnitude > 0.2:
                    trending_up.append((change['feature'], change_magnitude))
                elif change_magnitude < -0.2:
                    trending_down.append((change['feature'], abs(change_magnitude)))
        
        return {
            'period_days': days,
            'total_changes': len(recent_changes),
            'new_preferences': new_preferences,
            'removed_preferences': removed_preferences,
            'trending_up': sorted(trending_up, key=lambda x: x[1], reverse=True)[:5],
            'trending_down': sorted(trending_down, key=lambda x: x[1], reverse=True)[:5],
            'change_frequency': len(recent_changes) / days
        }
    
    def suggest_preference_adjustments(self, user_id: str) -> List[Dict[str, Any]]:
        """Suggest preference adjustments based on patterns"""
        profile = self.get_user_profile(user_id)
        suggestions = []
        
        # Find preferences with low confidence that could be refined
        low_confidence_prefs = [
            (feature, pref) for feature, pref in profile.preferences.items()
            if pref.confidence < 0.5 and not pref.locked
        ]
        
        for feature, pref in low_confidence_prefs:
            suggestions.append({
                'type': 'refine_confidence',
                'feature': feature,
                'current_value': pref.value,
                'current_confidence': pref.confidence,
                'suggestion': 'Consider providing more feedback on this preference'
            })
        
        # Find contradictory preferences
        contradictions = self._find_contradictory_preferences(user_id)
        for contradiction in contradictions:
            suggestions.append({
                'type': 'resolve_contradiction',
                'features': contradiction['features'],
                'suggestion': contradiction['suggestion']
            })
        
        # Suggest missing preferences in active categories
        active_categories = self._get_active_categories(user_id)
        for category in active_categories:
            if len(self.get_preferences_by_category(user_id, category)) < 3:
                suggestions.append({
                    'type': 'expand_category',
                    'category': category.value,
                    'suggestion': f'Add more preferences in {category.value} to improve recommendations'
                })
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    def reset_preferences(self, user_id: str, category: PreferenceCategory = None) -> bool:
        """Reset user preferences (all or by category)"""
        try:
            profile = self.get_user_profile(user_id)
            
            if category:
                # Reset specific category
                to_remove = [
                    feature for feature, pref in profile.preferences.items()
                    if pref.category == category and not pref.locked
                ]
                
                for feature in to_remove:
                    del profile.preferences[feature]
                    self._record_preference_change(user_id, feature, 
                                                 profile.preferences.get(feature, {}).get('value'),
                                                 None, PreferenceSource.EXPLICIT)
            else:
                # Reset all non-locked preferences
                locked_prefs = {
                    feature: pref for feature, pref in profile.preferences.items()
                    if pref.locked
                }
                
                # Record removals
                for feature, pref in profile.preferences.items():
                    if not pref.locked:
                        self._record_preference_change(user_id, feature, pref.value, 
                                                     None, PreferenceSource.EXPLICIT)
                
                profile.preferences = locked_prefs
            
            # Update profile metadata
            profile.last_updated = datetime.now()
            profile.version += 1
            
            self.logger.info(f"Reset preferences for user {user_id}" + 
                           (f" in category {category.value}" if category else ""))
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting preferences: {e}")
            return False
    
    def export_preferences(self, user_id: str) -> Dict[str, Any]:
        """Export user preferences to a portable format"""
        profile = self.get_user_profile(user_id)
        
        # Convert preferences to serializable format
        serializable_prefs = {}
        for feature, pref in profile.preferences.items():
            serializable_prefs[feature] = {
                'value': pref.value,
                'weight': pref.weight,
                'category': pref.category.value,
                'source': pref.source.value,
                'confidence': pref.confidence,
                'last_updated': pref.last_updated.isoformat(),
                'update_count': pref.update_count,
                'locked': pref.locked,
                'notes': pref.notes
            }
        
        return {
            'user_id': user_id,
            'preferences': serializable_prefs,
            'meta_preferences': profile.meta_preferences,
            'created_at': profile.created_at.isoformat(),
            'last_updated': profile.last_updated.isoformat(),
            'version': profile.version,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_preferences(self, user_data: Dict[str, Any], 
                          merge: bool = True) -> bool:
        """Import user preferences from exported data"""
        try:
            user_id = user_data['user_id']
            
            if not merge:
                # Replace all preferences
                self.user_profiles[user_id] = PreferenceProfile(user_id=user_id)
            
            profile = self.get_user_profile(user_id)
            
            # Import preferences
            for feature, pref_data in user_data['preferences'].items():
                pref = PreferenceItem(
                    feature=feature,
                    value=pref_data['value'],
                    weight=pref_data['weight'],
                    category=PreferenceCategory(pref_data['category']),
                    source=PreferenceSource(pref_data['source']),
                    confidence=pref_data['confidence'],
                    last_updated=datetime.fromisoformat(pref_data['last_updated']),
                    update_count=pref_data['update_count'],
                    locked=pref_data['locked'],
                    notes=pref_data['notes']
                )
                
                profile.preferences[feature] = pref
            
            # Import meta preferences
            profile.meta_preferences.update(user_data.get('meta_preferences', {}))
            
            # Update metadata
            profile.last_updated = datetime.now()
            profile.version += 1
            
            self.logger.info(f"Imported preferences for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing preferences: {e}")
            return False
    
    def _create_default_profile(self, user_id: str) -> PreferenceProfile:
        """Create default preference profile for new user"""
        profile = PreferenceProfile(user_id=user_id)
        
        # Add default preferences from template
        for feature, default_value in self.default_preferences.items():
            pref = PreferenceItem(
                feature=feature,
                value=default_value,
                weight=0.3,  # Lower weight for defaults
                category=self._infer_category(feature),
                source=PreferenceSource.SYSTEM,
                confidence=0.5,
                last_updated=datetime.now()
            )
            profile.preferences[feature] = pref
        
        return profile
    
    def _infer_category(self, feature: str) -> PreferenceCategory:
        """Infer category from feature name"""
        feature_lower = feature.lower()
        
        # Simple rule-based inference
        if any(word in feature_lower for word in ['genre', 'style', 'type']):
            return PreferenceCategory.GENRE
        elif any(word in feature_lower for word in ['time', 'hour', 'day', 'season']):
            return PreferenceCategory.TEMPORAL
        elif any(word in feature_lower for word in ['quality', 'rating', 'score']):
            return PreferenceCategory.QUALITY
        elif any(word in feature_lower for word in ['theme', 'topic', 'subject']):
            return PreferenceCategory.THEME
        else:
            return PreferenceCategory.CONTENT_TYPE
    
    def _record_preference_change(self, user_id: str, feature: str, 
                                old_value: Optional[float], new_value: Optional[float],
                                source: PreferenceSource):
        """Record preference change in history"""
        change = {
            'feature': feature,
            'old_value': old_value,
            'new_value': new_value,
            'source': source.value,
            'timestamp': datetime.now()
        }
        
        self.preference_history[user_id].append(change)
        
        # Keep only recent history (last 1000 changes)
        if len(self.preference_history[user_id]) > 1000:
            self.preference_history[user_id] = self.preference_history[user_id][-1000:]
    
    def _find_contradictory_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        """Find contradictory preferences that might need resolution"""
        profile = self.get_user_profile(user_id)
        contradictions = []
        
        # This is a simplified implementation
        # In practice, this would use domain knowledge and semantic similarity
        
        # Example: Look for opposite preferences in similar features
        features = list(profile.preferences.keys())
        for i, feature1 in enumerate(features):
            for feature2 in features[i+1:]:
                pref1 = profile.preferences[feature1]
                pref2 = profile.preferences[feature2]
                
                # Check if features are semantically similar but have opposite values
                if (self._features_similar(feature1, feature2) and
                    pref1.value * pref2.value < -0.5 and  # Opposite signs, strong values
                    pref1.confidence > 0.7 and pref2.confidence > 0.7):
                    
                    contradictions.append({
                        'features': [feature1, feature2],
                        'values': [pref1.value, pref2.value],
                        'suggestion': f'You have conflicting preferences for {feature1} and {feature2}'
                    })
        
        return contradictions
    
    def _features_similar(self, feature1: str, feature2: str) -> bool:
        """Check if two features are semantically similar"""
        # Simplified similarity check
        words1 = set(feature1.lower().split())
        words2 = set(feature2.lower().split())
        
        # If they share significant words, consider them similar
        intersection = words1 & words2
        union = words1 | words2
        
        if len(union) == 0:
            return False
        
        similarity = len(intersection) / len(union)
        return similarity > 0.5
    
    def _get_active_categories(self, user_id: str) -> List[PreferenceCategory]:
        """Get categories that user actively uses"""
        profile = self.get_user_profile(user_id)
        
        category_activity = defaultdict(int)
        for pref in profile.preferences.values():
            if pref.confidence > 0.5:
                category_activity[pref.category] += 1
        
        # Return categories with at least 2 preferences
        return [cat for cat, count in category_activity.items() if count >= 2]
    
    def _load_preference_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load preference templates for different domains"""
        return {
            'movies': {
                'action': 0.0, 'comedy': 0.0, 'drama': 0.0, 'horror': 0.0,
                'sci_fi': 0.0, 'romance': 0.0, 'thriller': 0.0, 'documentary': 0.0
            },
            'music': {
                'rock': 0.0, 'pop': 0.0, 'jazz': 0.0, 'classical': 0.0,
                'electronic': 0.0, 'hip_hop': 0.0, 'country': 0.0, 'folk': 0.0
            },
            'books': {
                'fiction': 0.0, 'non_fiction': 0.0, 'mystery': 0.0, 'fantasy': 0.0,
                'biography': 0.0, 'science': 0.0, 'history': 0.0, 'self_help': 0.0
            }
        }
    
    def _load_default_preferences(self) -> Dict[str, float]:
        """Load default preferences for new users"""
        return {
            'diversity': 0.5,
            'novelty': 0.3,
            'popularity': 0.4,
            'quality': 0.7
        }
    
    async def update_preferences(self, user_id: str, preferences: Dict[str, Any], merge_strategy: str = "update") -> Dict[str, Any]:
        """Update user preferences with given strategy"""
        try:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}
            
            current_prefs = self.user_preferences[user_id]
            
            if merge_strategy == "replace":
                self.user_preferences[user_id] = preferences.copy()
            elif merge_strategy == "merge":
                # Merge preferences with weighted average
                for key, value in preferences.items():
                    if key in current_prefs:
                        # Average current and new values
                        current_prefs[key] = (current_prefs[key] + value) / 2
                    else:
                        current_prefs[key] = value
            else:  # "update" strategy
                current_prefs.update(preferences)
            
            # Record the change
            for key, value in preferences.items():
                old_value = current_prefs.get(key) if merge_strategy != "replace" else None
                self._record_preference_change(user_id, key, old_value, value, PreferenceSource.EXPLICIT)
            
            self.preference_last_updated[user_id] = datetime.now()
            
            return {
                'success': True,
                'updated_preferences': self.user_preferences[user_id],
                'merge_strategy': merge_strategy,
                'changes_recorded': len(preferences)
            }
            
        except Exception as e:
            logger.error(f"Error updating preferences for user {user_id}: {e}")
            return {'success': False, 'error': str(e)}
