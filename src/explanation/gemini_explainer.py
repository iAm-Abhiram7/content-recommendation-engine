"""
Gemini Explainer for Content Recommendation Engine

This module uses Google's Gemini API to generate natural language explanations
for recommendation adaptations, providing human-friendly, contextual explanations.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..content_understanding.gemini_client import GeminiClient
from ..utils.logging import setup_logger


class ExplanationStyle(Enum):
    """Style of explanation"""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    DETAILED = "detailed"


@dataclass
class ExplanationRequest:
    """Request for explanation generation"""
    adaptation_data: Dict[str, Any]
    user_context: Dict[str, Any]
    style: ExplanationStyle
    max_length: int = 200
    include_reasoning: bool = True
    personalization_level: str = "medium"


@dataclass
class GeminiExplanationResponse:
    """Response from Gemini explanation generation"""
    explanation: str
    reasoning: Optional[str]
    confidence_score: float
    style_used: ExplanationStyle
    tokens_used: int
    generation_time: float


class GeminiExplainer:
    """
    Uses Google Gemini to generate natural language explanations
    for recommendation adaptations and changes
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize Gemini client
        self.gemini_client = GeminiClient(self.config.get('gemini_config', {}))
        
        # Configuration
        self.default_style = ExplanationStyle(self.config.get('default_style', 'friendly'))
        self.max_retries = self.config.get('max_retries', 3)
        self.temperature = self.config.get('temperature', 0.7)
        self.safety_settings = self._get_safety_settings()
        
        # Prompts and templates
        self.system_prompts = self._load_system_prompts()
        self.style_modifiers = self._load_style_modifiers()
        
        # State tracking
        self.usage_stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_tokens': 0,
            'average_response_time': 0
        }
        
        self.logger = setup_logger(__name__)
    
    async def generate_explanation(self, request: ExplanationRequest) -> GeminiExplanationResponse:
        """Generate explanation using Gemini"""
        start_time = datetime.now()
        self.usage_stats['total_requests'] += 1
        
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(request)
            
            # Generate explanation
            response = await self._generate_with_retry(prompt, request)
            
            # Process response
            explanation_response = self._process_response(response, request, start_time)
            
            self.usage_stats['successful_generations'] += 1
            self._update_usage_stats(explanation_response)
            
            return explanation_response
            
        except Exception as e:
            self.logger.error(f"Error generating Gemini explanation: {e}")
            self.usage_stats['failed_generations'] += 1
            return self._fallback_explanation(request, start_time)
    
    async def explain_adaptation_batch(self, 
                                     requests: List[ExplanationRequest]) -> List[GeminiExplanationResponse]:
        """Generate explanations for multiple adaptations"""
        tasks = [self.generate_explanation(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Error in batch explanation {i}: {response}")
                processed_responses.append(self._fallback_explanation(requests[i], datetime.now()))
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    async def explain_preference_evolution(self,
                                         user_id: str,
                                         preference_history: Dict[str, List],
                                         style: ExplanationStyle = None) -> str:
        """Explain how user preferences have evolved over time"""
        try:
            style = style or self.default_style
            
            # Prepare evolution analysis prompt
            prompt = self._prepare_evolution_prompt(user_id, preference_history, style)
            
            # Generate explanation
            response = await self.gemini_client.generate_response(
                prompt,
                temperature=self.temperature,
                max_tokens=300
            )
            
            return response.get('text', 'Unable to analyze preference evolution')
            
        except Exception as e:
            self.logger.error(f"Error explaining preference evolution: {e}")
            return "Your preferences have evolved based on your interactions and feedback."
    
    async def explain_recommendation_reasons(self,
                                           item_data: Dict[str, Any],
                                           user_profile: Dict[str, Any],
                                           recommendation_factors: Dict[str, Any],
                                           style: ExplanationStyle = None) -> str:
        """Explain why a specific item was recommended"""
        try:
            style = style or self.default_style
            
            # Prepare recommendation reasoning prompt
            prompt = self._prepare_recommendation_prompt(
                item_data, user_profile, recommendation_factors, style
            )
            
            # Generate explanation
            response = await self.gemini_client.generate_response(
                prompt,
                temperature=self.temperature,
                max_tokens=150
            )
            
            return response.get('text', 'Recommended based on your preferences')
            
        except Exception as e:
            self.logger.error(f"Error explaining recommendation reasons: {e}")
            return "This item matches your interests and preferences."
    
    async def generate_comparative_explanation(self,
                                             old_recommendations: List[Dict],
                                             new_recommendations: List[Dict],
                                             adaptation_reason: str,
                                             style: ExplanationStyle = None) -> str:
        """Generate explanation comparing old and new recommendations"""
        try:
            style = style or self.default_style
            
            # Analyze differences
            differences = self._analyze_recommendation_differences(
                old_recommendations, new_recommendations
            )
            
            # Prepare comparative prompt
            prompt = self._prepare_comparative_prompt(differences, adaptation_reason, style)
            
            # Generate explanation
            response = await self.gemini_client.generate_response(
                prompt,
                temperature=self.temperature,
                max_tokens=250
            )
            
            return response.get('text', 'Your recommendations have been updated.')
            
        except Exception as e:
            self.logger.error(f"Error generating comparative explanation: {e}")
            return "We've updated your recommendations to better match your current interests."
    
    def _prepare_prompt(self, request: ExplanationRequest) -> str:
        """Prepare prompt for Gemini based on request"""
        # Get base system prompt
        system_prompt = self.system_prompts['adaptation_explanation']
        
        # Get style modifier
        style_modifier = self.style_modifiers.get(request.style, "")
        
        # Prepare adaptation data summary
        adaptation_summary = self._summarize_adaptation_data(request.adaptation_data)
        
        # Prepare user context
        user_context_summary = self._summarize_user_context(request.user_context)
        
        # Construct full prompt
        prompt = f"""
{system_prompt}

{style_modifier}

ADAPTATION DATA:
{adaptation_summary}

USER CONTEXT:
{user_context_summary}

REQUIREMENTS:
- Maximum length: {request.max_length} words
- Include reasoning: {request.include_reasoning}
- Personalization level: {request.personalization_level}

Generate a clear, helpful explanation for why the user's recommendations changed.
"""
        
        return prompt
    
    def _prepare_evolution_prompt(self, user_id: str, 
                                preference_history: Dict[str, List],
                                style: ExplanationStyle) -> str:
        """Prepare prompt for preference evolution explanation"""
        style_modifier = self.style_modifiers.get(style, "")
        
        # Summarize preference changes
        changes_summary = self._summarize_preference_changes(preference_history)
        
        prompt = f"""
You are an AI assistant explaining how a user's preferences have evolved over time.

{style_modifier}

PREFERENCE EVOLUTION DATA:
{changes_summary}

Generate a clear explanation of how this user's preferences have changed, highlighting:
1. The most significant changes
2. Potential reasons for these changes
3. What this means for their future recommendations

Keep the explanation under 200 words and make it personal and insightful.
"""
        
        return prompt
    
    def _prepare_recommendation_prompt(self, item_data: Dict[str, Any],
                                     user_profile: Dict[str, Any],
                                     factors: Dict[str, Any],
                                     style: ExplanationStyle) -> str:
        """Prepare prompt for recommendation reasoning"""
        style_modifier = self.style_modifiers.get(style, "")
        
        # Summarize key factors
        factors_summary = self._summarize_recommendation_factors(factors)
        
        prompt = f"""
You are an AI assistant explaining why a specific item was recommended to a user.

{style_modifier}

ITEM: {item_data.get('title', 'Unknown')}
GENRE/CATEGORY: {item_data.get('genre', 'Unknown')}

RECOMMENDATION FACTORS:
{factors_summary}

USER INTERESTS: {', '.join(user_profile.get('interests', []))}

Generate a brief, personalized explanation for why this item was recommended.
Keep it under 100 words and focus on the most compelling reasons.
"""
        
        return prompt
    
    def _prepare_comparative_prompt(self, differences: Dict[str, Any],
                                  adaptation_reason: str,
                                  style: ExplanationStyle) -> str:
        """Prepare prompt for comparative explanation"""
        style_modifier = self.style_modifiers.get(style, "")
        
        prompt = f"""
You are an AI assistant explaining how a user's recommendations have changed.

{style_modifier}

CHANGES IN RECOMMENDATIONS:
- New items added: {differences.get('new_items', 0)}
- Items removed: {differences.get('removed_items', 0)}
- Category shifts: {differences.get('category_changes', 'None')}
- Diversity change: {differences.get('diversity_change', 'No change')}

REASON FOR ADAPTATION: {adaptation_reason}

Generate a clear explanation of how and why the recommendations changed.
Keep it under 150 words and focus on helping the user understand the improvements.
"""
        
        return prompt
    
    async def _generate_with_retry(self, prompt: str, 
                                 request: ExplanationRequest) -> Dict[str, Any]:
        """Generate response with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.gemini_client.generate_response(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=request.max_length + 50,  # Buffer for processing
                    safety_settings=self.safety_settings
                )
                
                if response and 'text' in response:
                    return response
                else:
                    raise ValueError("Empty or invalid response from Gemini")
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise last_exception
    
    def _process_response(self, response: Dict[str, Any], 
                         request: ExplanationRequest,
                         start_time: datetime) -> GeminiExplanationResponse:
        """Process Gemini response into structured format"""
        explanation_text = response.get('text', '')
        
        # Extract reasoning if included
        reasoning = None
        if request.include_reasoning and '|REASONING|' in explanation_text:
            parts = explanation_text.split('|REASONING|')
            explanation_text = parts[0].strip()
            reasoning = parts[1].strip() if len(parts) > 1 else None
        
        # Calculate metrics
        generation_time = (datetime.now() - start_time).total_seconds()
        tokens_used = response.get('token_count', len(explanation_text.split()))
        
        # Estimate confidence based on response quality
        confidence_score = self._estimate_confidence(explanation_text, request)
        
        return GeminiExplanationResponse(
            explanation=explanation_text,
            reasoning=reasoning,
            confidence_score=confidence_score,
            style_used=request.style,
            tokens_used=tokens_used,
            generation_time=generation_time
        )
    
    def _estimate_confidence(self, explanation: str, request: ExplanationRequest) -> float:
        """Estimate confidence in the generated explanation"""
        confidence = 0.8  # Base confidence
        
        # Length check
        if len(explanation.split()) < 10:
            confidence -= 0.3
        elif len(explanation.split()) > request.max_length * 1.5:
            confidence -= 0.2
        
        # Quality indicators
        if 'because' in explanation.lower() or 'due to' in explanation.lower():
            confidence += 0.1
        
        if 'recommendations' in explanation.lower():
            confidence += 0.1
        
        # Generic responses penalty
        generic_phrases = ['based on your preferences', 'matches your interests']
        if any(phrase in explanation.lower() for phrase in generic_phrases):
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _summarize_adaptation_data(self, adaptation_data: Dict[str, Any]) -> str:
        """Summarize adaptation data for prompt"""
        summary_parts = []
        
        if 'type' in adaptation_data:
            summary_parts.append(f"Adaptation type: {adaptation_data['type']}")
        
        if 'trigger' in adaptation_data:
            summary_parts.append(f"Triggered by: {adaptation_data['trigger']}")
        
        if 'confidence' in adaptation_data:
            summary_parts.append(f"Confidence: {adaptation_data['confidence']:.2f}")
        
        if 'affected_features' in adaptation_data:
            features = ', '.join(adaptation_data['affected_features'][:3])
            summary_parts.append(f"Main affected features: {features}")
        
        if 'improvement_metrics' in adaptation_data:
            metrics = adaptation_data['improvement_metrics']
            summary_parts.append(f"Expected improvement: {metrics}")
        
        return '\n'.join(summary_parts)
    
    def _summarize_user_context(self, user_context: Dict[str, Any]) -> str:
        """Summarize user context for prompt"""
        summary_parts = []
        
        if 'user_id' in user_context:
            summary_parts.append(f"User: {user_context['user_id']}")
        
        if 'recent_activity' in user_context:
            activity = user_context['recent_activity']
            summary_parts.append(f"Recent activity: {activity}")
        
        if 'preference_strength' in user_context:
            strength = user_context['preference_strength']
            summary_parts.append(f"Preference strength: {strength}")
        
        if 'engagement_level' in user_context:
            engagement = user_context['engagement_level']
            summary_parts.append(f"Engagement level: {engagement}")
        
        if 'time_of_day' in user_context:
            time_info = user_context['time_of_day']
            summary_parts.append(f"Usage time: {time_info}")
        
        return '\n'.join(summary_parts)
    
    def _summarize_preference_changes(self, preference_history: Dict[str, List]) -> str:
        """Summarize preference changes for evolution explanation"""
        changes = []
        
        for feature, history in preference_history.items():
            if len(history) >= 2:
                start_val = history[0]
                end_val = history[-1]
                change = end_val - start_val
                
                if abs(change) > 0.1:
                    direction = "increased" if change > 0 else "decreased"
                    changes.append(f"{feature}: {direction} by {abs(change):.2f}")
        
        return '\n'.join(changes[:5])  # Top 5 changes
    
    def _summarize_recommendation_factors(self, factors: Dict[str, Any]) -> str:
        """Summarize recommendation factors"""
        factor_parts = []
        
        for factor, value in factors.items():
            if isinstance(value, (int, float)):
                factor_parts.append(f"{factor}: {value:.2f}")
            else:
                factor_parts.append(f"{factor}: {value}")
        
        return '\n'.join(factor_parts)
    
    def _analyze_recommendation_differences(self, old_recs: List[Dict], 
                                          new_recs: List[Dict]) -> Dict[str, Any]:
        """Analyze differences between old and new recommendations"""
        old_ids = {rec.get('id') for rec in old_recs}
        new_ids = {rec.get('id') for rec in new_recs}
        
        differences = {
            'new_items': len(new_ids - old_ids),
            'removed_items': len(old_ids - new_ids),
            'common_items': len(old_ids & new_ids)
        }
        
        # Analyze category changes
        old_categories = [rec.get('category') for rec in old_recs]
        new_categories = [rec.get('category') for rec in new_recs]
        
        old_category_counts = {}
        new_category_counts = {}
        
        for cat in old_categories:
            old_category_counts[cat] = old_category_counts.get(cat, 0) + 1
        
        for cat in new_categories:
            new_category_counts[cat] = new_category_counts.get(cat, 0) + 1
        
        category_changes = {}
        all_categories = set(old_category_counts.keys()) | set(new_category_counts.keys())
        
        for cat in all_categories:
            old_count = old_category_counts.get(cat, 0)
            new_count = new_category_counts.get(cat, 0)
            if old_count != new_count:
                category_changes[cat] = new_count - old_count
        
        differences['category_changes'] = category_changes
        
        return differences
    
    def _load_system_prompts(self) -> Dict[str, str]:
        """Load system prompts for different explanation types"""
        return {
            'adaptation_explanation': """
You are an AI assistant that explains recommendation system adaptations to users.
Your goal is to help users understand why their recommendations changed in a way that builds trust and understanding.

Key principles:
1. Be clear and concise
2. Focus on user benefits
3. Use accessible language
4. Be honest about limitations
5. Personalize the explanation
""",
            'preference_evolution': """
You are an expert in analyzing user preference evolution.
Explain how and why user preferences have changed over time in an insightful way.
""",
            'recommendation_reasoning': """
You are an AI that explains why specific items were recommended.
Focus on the connection between the item and the user's interests.
"""
        }
    
    def _load_style_modifiers(self) -> Dict[ExplanationStyle, str]:
        """Load style modifiers for different explanation styles"""
        return {
            ExplanationStyle.CASUAL: "Use a casual, conversational tone. Be friendly and approachable.",
            ExplanationStyle.PROFESSIONAL: "Use professional, polite language. Be informative and respectful.",
            ExplanationStyle.TECHNICAL: "Use technical terms appropriately. Include specific details and metrics.",
            ExplanationStyle.FRIENDLY: "Be warm and friendly. Use 'you' and 'your' frequently. Show enthusiasm.",
            ExplanationStyle.CONCISE: "Be as brief as possible while still being helpful. Use short sentences.",
            ExplanationStyle.DETAILED: "Provide comprehensive explanations with examples and context."
        }
    
    def _get_safety_settings(self) -> Dict[HarmCategory, HarmBlockThreshold]:
        """Get safety settings for Gemini"""
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
    
    def _fallback_explanation(self, request: ExplanationRequest, 
                            start_time: datetime) -> GeminiExplanationResponse:
        """Generate fallback explanation when Gemini fails"""
        fallback_explanations = {
            ExplanationStyle.CASUAL: "Hey! We've updated your recommendations based on what you've been enjoying lately.",
            ExplanationStyle.PROFESSIONAL: "Your recommendations have been updated based on your recent activity and preferences.",
            ExplanationStyle.TECHNICAL: "Recommendation model updated using recent interaction data and preference signals.",
            ExplanationStyle.FRIENDLY: "Good news! We've personalized your recommendations to better match your interests.",
            ExplanationStyle.CONCISE: "Recommendations updated based on your preferences.",
            ExplanationStyle.DETAILED: "We've analyzed your recent activity and updated your recommendations to better reflect your current interests and preferences."
        }
        
        explanation = fallback_explanations.get(request.style, fallback_explanations[ExplanationStyle.FRIENDLY])
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return GeminiExplanationResponse(
            explanation=explanation,
            reasoning=None,
            confidence_score=0.6,
            style_used=request.style,
            tokens_used=len(explanation.split()),
            generation_time=generation_time
        )
    
    def _update_usage_stats(self, response: GeminiExplanationResponse):
        """Update usage statistics"""
        # Update running averages
        total_requests = self.usage_stats['total_requests']
        
        current_avg_tokens = self.usage_stats['average_tokens']
        self.usage_stats['average_tokens'] = (
            (current_avg_tokens * (total_requests - 1) + response.tokens_used) / total_requests
        )
        
        current_avg_time = self.usage_stats['average_response_time']
        self.usage_stats['average_response_time'] = (
            (current_avg_time * (total_requests - 1) + response.generation_time) / total_requests
        )
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.usage_stats.copy()
    
    def reset_usage_statistics(self):
        """Reset usage statistics"""
        self.usage_stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_tokens': 0,
            'average_response_time': 0
        }
