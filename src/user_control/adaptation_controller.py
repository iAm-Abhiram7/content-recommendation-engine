"""
Adaptation Controller for Content Recommendation Engine

This module provides user control over recommendation adaptations, allowing users
to customize how the system adapts to their changing preferences and behaviors.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json

from ..utils.logging import setup_logger


class ControlLevel(Enum):
    """Levels of user control over adaptations"""
    MINIMAL = "minimal"        # System controls everything
    BASIC = "basic"           # Basic on/off controls
    MODERATE = "moderate"     # Category-level controls
    ADVANCED = "advanced"     # Fine-grained controls
    EXPERT = "expert"         # Full algorithmic controls
    FULL = "full"             # Complete user control


class AdaptationPolicy(Enum):
    """Adaptation policies"""
    AGGRESSIVE = "aggressive"     # Fast adaptation to changes
    BALANCED = "balanced"        # Moderate adaptation speed
    CONSERVATIVE = "conservative" # Slow, careful adaptation
    MANUAL = "manual"            # User-triggered only
    DISABLED = "disabled"        # No adaptation


class AdaptationType(Enum):
    """Types of adaptations"""
    PREFERENCE_DRIFT = "preference_drift"
    SEASONAL_ADJUSTMENT = "seasonal_adjustment"
    BEHAVIORAL_CHANGE = "behavioral_change"
    FEEDBACK_INTEGRATION = "feedback_integration"
    EXPLORATION_BALANCE = "exploration_balance"
    DIVERSITY_ADJUSTMENT = "diversity_adjustment"


@dataclass
class AdaptationSettings:
    """User's adaptation settings"""
    policy: AdaptationPolicy
    control_level: ControlLevel
    enabled_adaptations: List[AdaptationType]
    adaptation_speed: float  # 0.0 - 1.0
    confidence_threshold: float  # 0.0 - 1.0
    require_confirmation: bool
    auto_rollback: bool
    notification_preferences: Dict[str, bool]


@dataclass
class AdaptationRequest:
    """Request for adaptation control"""
    user_id: str
    adaptation_type: AdaptationType
    action: str  # 'enable', 'disable', 'configure'
    parameters: Dict[str, Any]
    timestamp: datetime


@dataclass
class AdaptationHistory:
    """History of adaptation actions"""
    user_id: str
    timestamp: datetime
    adaptation_type: AdaptationType
    action: str
    previous_settings: Dict[str, Any]
    new_settings: Dict[str, Any]
    user_initiated: bool


class AdaptationController:
    """
    Controls how recommendation adaptations are applied based on user preferences,
    providing transparency and control over system behavior
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # User settings storage
        self.user_settings: Dict[str, AdaptationSettings] = {}
        self.adaptation_history: List[AdaptationHistory] = []
        
        # Default settings
        self.default_settings = AdaptationSettings(
            policy=AdaptationPolicy.BALANCED,
            control_level=ControlLevel.BASIC,
            enabled_adaptations=[
                AdaptationType.PREFERENCE_DRIFT,
                AdaptationType.FEEDBACK_INTEGRATION
            ],
            adaptation_speed=0.5,
            confidence_threshold=0.7,
            require_confirmation=False,
            auto_rollback=True,
            notification_preferences={
                'major_changes': True,
                'new_features': True,
                'performance_updates': False,
                'weekly_summary': True
            }
        )
        
        # Callbacks for different events
        self.adaptation_callbacks: Dict[str, List[Callable]] = {
            'before_adaptation': [],
            'after_adaptation': [],
            'user_action': [],
            'rollback': []
        }
        
        # Pending adaptations waiting for user approval
        self.pending_adaptations: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger = setup_logger(__name__)
    
    def get_user_settings(self, user_id: str) -> AdaptationSettings:
        """Get adaptation settings for a user"""
        return self.user_settings.get(user_id, self.default_settings)
    
    def update_user_settings(self, user_id: str, settings: AdaptationSettings) -> bool:
        """Update user's adaptation settings"""
        try:
            # Store previous settings for history
            previous_settings = self.user_settings.get(user_id, self.default_settings)
            
            # Update settings
            self.user_settings[user_id] = settings
            
            # Record in history
            self.adaptation_history.append(AdaptationHistory(
                user_id=user_id,
                timestamp=datetime.now(),
                adaptation_type=AdaptationType.PREFERENCE_DRIFT,  # Generic
                action='settings_update',
                previous_settings=asdict(previous_settings),
                new_settings=asdict(settings),
                user_initiated=True
            ))
            
            # Trigger callbacks
            self._trigger_callbacks('user_action', {
                'user_id': user_id,
                'action': 'settings_update',
                'settings': settings
            })
            
            self.logger.info(f"Updated adaptation settings for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating user settings: {e}")
            return False
    
    def can_adapt(self, user_id: str, adaptation_type: AdaptationType, 
                  confidence: float = None) -> bool:
        """Check if adaptation is allowed for user"""
        settings = self.get_user_settings(user_id)
        
        # Check if adaptation type is enabled
        if adaptation_type not in settings.enabled_adaptations:
            return False
        
        # Check policy
        if settings.policy == AdaptationPolicy.DISABLED:
            return False
        
        if settings.policy == AdaptationPolicy.MANUAL:
            return False  # Manual adaptations need explicit approval
        
        # Check confidence threshold
        if confidence is not None and confidence < settings.confidence_threshold:
            return False
        
        return True
    
    async def request_adaptation(self, user_id: str, adaptation_type: AdaptationType,
                               adaptation_data: Dict[str, Any]) -> str:
        """Request permission for adaptation"""
        settings = self.get_user_settings(user_id)
        
        # Generate request ID
        request_id = f"{user_id}_{adaptation_type.value}_{int(datetime.now().timestamp())}"
        
        # Check if confirmation is required
        if settings.require_confirmation or settings.policy == AdaptationPolicy.MANUAL:
            # Add to pending adaptations
            if user_id not in self.pending_adaptations:
                self.pending_adaptations[user_id] = []
            
            self.pending_adaptations[user_id].append({
                'request_id': request_id,
                'adaptation_type': adaptation_type,
                'adaptation_data': adaptation_data,
                'timestamp': datetime.now(),
                'status': 'pending'
            })
            
            # Notify user if notifications are enabled
            if settings.notification_preferences.get('major_changes', True):
                await self._notify_user_adaptation_request(user_id, adaptation_type, adaptation_data)
            
            return request_id
        else:
            # Auto-approve
            return await self.approve_adaptation(user_id, request_id, adaptation_data)
    
    async def approve_adaptation(self, user_id: str, request_id: str,
                               adaptation_data: Dict[str, Any] = None) -> str:
        """Approve a pending adaptation"""
        try:
            # Find pending adaptation
            pending = None
            if user_id in self.pending_adaptations:
                for adaptation in self.pending_adaptations[user_id]:
                    if adaptation['request_id'] == request_id:
                        pending = adaptation
                        break
            
            if not pending:
                # If not found in pending, create new approval
                if adaptation_data:
                    pending = {
                        'request_id': request_id,
                        'adaptation_type': AdaptationType.PREFERENCE_DRIFT,
                        'adaptation_data': adaptation_data,
                        'timestamp': datetime.now(),
                        'status': 'approved'
                    }
            
            if pending:
                # Mark as approved
                pending['status'] = 'approved'
                pending['approved_at'] = datetime.now()
                
                # Record in history
                self.adaptation_history.append(AdaptationHistory(
                    user_id=user_id,
                    timestamp=datetime.now(),
                    adaptation_type=pending['adaptation_type'],
                    action='approved',
                    previous_settings={},
                    new_settings=pending['adaptation_data'],
                    user_initiated=True
                ))
                
                # Trigger callbacks
                self._trigger_callbacks('after_adaptation', {
                    'user_id': user_id,
                    'request_id': request_id,
                    'adaptation_type': pending['adaptation_type'],
                    'adaptation_data': pending['adaptation_data']
                })
                
                self.logger.info(f"Approved adaptation {request_id} for user {user_id}")
                return 'approved'
            
            return 'not_found'
            
        except Exception as e:
            self.logger.error(f"Error approving adaptation: {e}")
            return 'error'
    
    async def reject_adaptation(self, user_id: str, request_id: str, 
                              reason: str = None) -> bool:
        """Reject a pending adaptation"""
        try:
            # Find and remove pending adaptation
            if user_id in self.pending_adaptations:
                for i, adaptation in enumerate(self.pending_adaptations[user_id]):
                    if adaptation['request_id'] == request_id:
                        adaptation['status'] = 'rejected'
                        adaptation['rejected_at'] = datetime.now()
                        adaptation['rejection_reason'] = reason
                        
                        # Record in history
                        self.adaptation_history.append(AdaptationHistory(
                            user_id=user_id,
                            timestamp=datetime.now(),
                            adaptation_type=adaptation['adaptation_type'],
                            action='rejected',
                            previous_settings={},
                            new_settings={'rejection_reason': reason},
                            user_initiated=True
                        ))
                        
                        self.logger.info(f"Rejected adaptation {request_id} for user {user_id}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error rejecting adaptation: {e}")
            return False
    
    def get_pending_adaptations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending adaptations for a user"""
        return self.pending_adaptations.get(user_id, [])
    
    async def rollback_adaptation(self, user_id: str, adaptation_id: str) -> bool:
        """Rollback a previous adaptation"""
        try:
            # Find adaptation in history
            target_adaptation = None
            for adaptation in reversed(self.adaptation_history):
                if (adaptation.user_id == user_id and 
                    adaptation.timestamp.isoformat() == adaptation_id):
                    target_adaptation = adaptation
                    break
            
            if not target_adaptation:
                return False
            
            # Restore previous settings
            if target_adaptation.previous_settings:
                # This would typically restore the recommendation model state
                # For now, we'll record the rollback action
                
                self.adaptation_history.append(AdaptationHistory(
                    user_id=user_id,
                    timestamp=datetime.now(),
                    adaptation_type=target_adaptation.adaptation_type,
                    action='rollback',
                    previous_settings=target_adaptation.new_settings,
                    new_settings=target_adaptation.previous_settings,
                    user_initiated=True
                ))
                
                # Trigger callbacks
                self._trigger_callbacks('rollback', {
                    'user_id': user_id,
                    'original_adaptation': target_adaptation,
                    'rollback_timestamp': datetime.now()
                })
                
                self.logger.info(f"Rolled back adaptation for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error rolling back adaptation: {e}")
            return False
    
    def set_adaptation_speed(self, user_id: str, speed: float) -> bool:
        """Set adaptation speed for a user (0.0 = slow, 1.0 = fast)"""
        try:
            settings = self.get_user_settings(user_id)
            settings.adaptation_speed = max(0.0, min(1.0, speed))
            return self.update_user_settings(user_id, settings)
        except Exception as e:
            self.logger.error(f"Error setting adaptation speed: {e}")
            return False
    
    def set_confidence_threshold(self, user_id: str, threshold: float) -> bool:
        """Set confidence threshold for adaptations"""
        try:
            settings = self.get_user_settings(user_id)
            settings.confidence_threshold = max(0.0, min(1.0, threshold))
            return self.update_user_settings(user_id, settings)
        except Exception as e:
            self.logger.error(f"Error setting confidence threshold: {e}")
            return False
    
    def enable_adaptation_type(self, user_id: str, adaptation_type: AdaptationType) -> bool:
        """Enable a specific adaptation type for user"""
        try:
            settings = self.get_user_settings(user_id)
            if adaptation_type not in settings.enabled_adaptations:
                settings.enabled_adaptations.append(adaptation_type)
                return self.update_user_settings(user_id, settings)
            return True
        except Exception as e:
            self.logger.error(f"Error enabling adaptation type: {e}")
            return False
    
    def disable_adaptation_type(self, user_id: str, adaptation_type: AdaptationType) -> bool:
        """Disable a specific adaptation type for user"""
        try:
            settings = self.get_user_settings(user_id)
            if adaptation_type in settings.enabled_adaptations:
                settings.enabled_adaptations.remove(adaptation_type)
                return self.update_user_settings(user_id, settings)
            return True
        except Exception as e:
            self.logger.error(f"Error disabling adaptation type: {e}")
            return False
    
    def get_adaptation_history(self, user_id: str, limit: int = 50) -> List[AdaptationHistory]:
        """Get adaptation history for a user"""
        user_history = [h for h in self.adaptation_history if h.user_id == user_id]
        return sorted(user_history, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_adaptation_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get adaptation statistics for a user"""
        history = self.get_adaptation_history(user_id)
        
        if not history:
            return {}
        
        # Count adaptations by type
        type_counts = {}
        action_counts = {}
        
        for adaptation in history:
            adaptation_type = adaptation.adaptation_type.value
            action = adaptation.action
            
            type_counts[adaptation_type] = type_counts.get(adaptation_type, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate time since last adaptation
        last_adaptation = history[0] if history else None
        time_since_last = None
        if last_adaptation:
            time_since_last = (datetime.now() - last_adaptation.timestamp).total_seconds() / 3600
        
        # Calculate user engagement with controls
        user_initiated_count = sum(1 for h in history if h.user_initiated)
        engagement_rate = user_initiated_count / len(history) if history else 0
        
        return {
            'total_adaptations': len(history),
            'adaptations_by_type': type_counts,
            'actions_by_type': action_counts,
            'user_initiated_percentage': engagement_rate * 100,
            'hours_since_last_adaptation': time_since_last,
            'current_settings': asdict(self.get_user_settings(user_id))
        }
    
    def add_callback(self, event: str, callback: Callable):
        """Add callback for adaptation events"""
        if event in self.adaptation_callbacks:
            self.adaptation_callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]):
        """Trigger callbacks for an event"""
        for callback in self.adaptation_callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in callback for {event}: {e}")
    
    async def _notify_user_adaptation_request(self, user_id: str, 
                                            adaptation_type: AdaptationType,
                                            adaptation_data: Dict[str, Any]):
        """Notify user about adaptation request"""
        # This would typically send a notification through the user interface
        # For now, we'll log the notification
        self.logger.info(f"Notification: User {user_id} has a pending {adaptation_type.value} adaptation request")
    
    def export_user_settings(self, user_id: str) -> Dict[str, Any]:
        """Export user's adaptation settings"""
        settings = self.get_user_settings(user_id)
        history = self.get_adaptation_history(user_id, limit=100)
        
        return {
            'user_id': user_id,
            'settings': asdict(settings),
            'history': [asdict(h) for h in history],
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_user_settings(self, user_data: Dict[str, Any]) -> bool:
        """Import user's adaptation settings"""
        try:
            user_id = user_data['user_id']
            settings_data = user_data['settings']
            
            # Reconstruct settings
            settings = AdaptationSettings(
                policy=AdaptationPolicy(settings_data['policy']),
                control_level=ControlLevel(settings_data['control_level']),
                enabled_adaptations=[AdaptationType(t) for t in settings_data['enabled_adaptations']],
                adaptation_speed=settings_data['adaptation_speed'],
                confidence_threshold=settings_data['confidence_threshold'],
                require_confirmation=settings_data['require_confirmation'],
                auto_rollback=settings_data['auto_rollback'],
                notification_preferences=settings_data['notification_preferences']
            )
            
            return self.update_user_settings(user_id, settings)
            
        except Exception as e:
            self.logger.error(f"Error importing user settings: {e}")
            return False
