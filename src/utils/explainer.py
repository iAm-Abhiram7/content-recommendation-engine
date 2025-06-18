"""
Recommendation Explainer Module

This module provides explanations for recommendations using various techniques
including feature importance, similarity explanations, and natural language generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class RecommendationExplainer:
    """Generates explanations for recommendations."""
    
    def __init__(self, use_gemini: bool = True):
        """
        Initialize the explainer.
        
        Args:
            use_gemini: Whether to use Gemini for natural language explanations
        """
        self.use_gemini = use_gemini
        self.explanation_templates = self._load_explanation_templates()
        self.gemini_client = None
        
        if use_gemini:
            try:
                from ..content_understanding.gemini_client import GeminiClient
                self.gemini_client = GeminiClient()
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
                self.use_gemini = False
    
    def _load_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Load explanation templates for different recommendation types."""
        return {
            'collaborative': {
                'similar_users': "Users with similar preferences also liked {items}",
                'user_neighborhood': "Based on users who rated {seed_items} highly",
                'matrix_factorization': "Recommended based on your rating patterns",
                'popularity': "Popular among users with similar tastes"
            },
            'content_based': {
                'genre_similarity': "Because you liked {similar_items} in {genres}",
                'feature_match': "Matches your preferences for {features}",
                'director_actor': "From {director} or starring {actors} you've enjoyed",
                'embedding_similarity': "Similar content to items you've rated highly"
            },
            'knowledge_based': {
                'trending': "Currently trending and highly rated",
                'new_releases': "New release matching your interests",
                'acclaimed': "Critically acclaimed with awards/recognition",
                'contextual': "Perfect for {context} based on your preferences"
            },
            'hybrid': {
                'multi_factor': "Combines your preferences, similar users, and content features",
                'weighted_ensemble': "Recommended by multiple algorithms with confidence {confidence}",
                'diverse_selection': "Selected for diversity while matching your tastes"
            },
            'personalization': {
                'short_term': "Based on your recent activity and current mood",
                'long_term': "Aligns with your established long-term preferences",
                'sequential': "Following your viewing patterns and progression",
                'context_aware': "Tailored for {time_of_day} viewing"
            }
        }
    
    def generate_explanation(self, 
                           recommendation: Dict[str, Any],
                           user_profile: Dict[str, Any],
                           item_features: Dict[str, Any],
                           recommendation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a recommendation.
        
        Args:
            recommendation: The recommendation with item_id, score, method
            user_profile: User's profile and preferences
            item_features: Features of the recommended item
            recommendation_context: Context information about how it was generated
            
        Returns:
            Dictionary containing multiple types of explanations
        """
        try:
            explanations = {}
            
            # Template-based explanation
            template_explanation = self._generate_template_explanation(
                recommendation, user_profile, item_features, recommendation_context
            )
            explanations['template'] = template_explanation
            
            # Feature importance explanation
            feature_explanation = self._generate_feature_explanation(
                recommendation, user_profile, item_features, recommendation_context
            )
            explanations['features'] = feature_explanation
            
            # Similarity explanation
            similarity_explanation = self._generate_similarity_explanation(
                recommendation, user_profile, item_features, recommendation_context
            )
            explanations['similarity'] = similarity_explanation
            
            # Counterfactual explanation
            counterfactual_explanation = self._generate_counterfactual_explanation(
                recommendation, user_profile, item_features
            )
            explanations['counterfactual'] = counterfactual_explanation
            
            # Natural language explanation (using Gemini if available)
            if self.use_gemini and self.gemini_client:
                nl_explanation = self._generate_natural_language_explanation(
                    recommendation, user_profile, item_features, explanations
                )
                explanations['natural_language'] = nl_explanation
            
            # Confidence and reasoning
            explanations['confidence'] = recommendation.get('confidence', recommendation.get('score', 0.5))
            explanations['reasoning_chain'] = self._build_reasoning_chain(recommendation_context)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                'template': "This item is recommended for you",
                'error': str(e)
            }
    
    def _generate_template_explanation(self, 
                                     recommendation: Dict[str, Any],
                                     user_profile: Dict[str, Any],
                                     item_features: Dict[str, Any],
                                     context: Dict[str, Any]) -> str:
        """Generate explanation using templates."""
        try:
            method = context.get('method', 'hybrid')
            strategy = context.get('strategy', 'unknown')
            
            # Get base template
            if method in self.explanation_templates:
                templates = self.explanation_templates[method]
                if strategy in templates:
                    template = templates[strategy]
                else:
                    template = list(templates.values())[0]  # Use first available
            else:
                template = "Recommended based on your preferences"
            
            # Fill in template variables
            explanation = self._fill_template_variables(
                template, recommendation, user_profile, item_features, context
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error in template explanation: {e}")
            return "This item is recommended for you"
    
    def _fill_template_variables(self, 
                               template: str,
                               recommendation: Dict[str, Any],
                               user_profile: Dict[str, Any],
                               item_features: Dict[str, Any],
                               context: Dict[str, Any]) -> str:
        """Fill variables in explanation templates."""
        try:
            # Extract relevant information
            item_genres = item_features.get('genres', [])
            if isinstance(item_genres, str):
                item_genres = item_genres.split('|')
            
            similar_items = context.get('similar_items', [])
            seed_items = context.get('seed_items', [])
            confidence = recommendation.get('confidence', 0.5)
            
            # Prepare substitutions
            substitutions = {
                'items': ', '.join(similar_items[:3]) if similar_items else 'similar content',
                'genres': ', '.join(item_genres[:2]) if item_genres else 'your preferred genres',
                'similar_items': ', '.join(similar_items[:2]) if similar_items else 'items you liked',
                'seed_items': ', '.join(seed_items[:2]) if seed_items else 'your rated items',
                'confidence': f"{confidence:.1%}",
                'director': item_features.get('director', 'directors you like'),
                'actors': ', '.join((item_features.get('actors', []))[:2]) if item_features.get('actors') else 'actors you enjoy',
                'features': ', '.join(item_genres[:2]) if item_genres else 'content features',
                'context': context.get('context_type', 'current situation'),
                'time_of_day': context.get('time_of_day', 'current time')
            }
            
            # Perform substitutions
            explanation = template
            for key, value in substitutions.items():
                explanation = explanation.replace(f'{{{key}}}', str(value))
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error filling template variables: {e}")
            return template
    
    def _generate_feature_explanation(self, 
                                    recommendation: Dict[str, Any],
                                    user_profile: Dict[str, Any],
                                    item_features: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feature-based explanation."""
        try:
            feature_importance = {}
            explanations = {}
            
            # Genre matching
            user_genres = set(user_profile.get('preferred_genres', []))
            item_genres = set(item_features.get('genres', []))
            if isinstance(item_genres, str):
                item_genres = set(item_features.get('genres', '').split('|'))
            
            genre_overlap = user_genres & item_genres
            if genre_overlap:
                feature_importance['genres'] = len(genre_overlap) / len(user_genres | item_genres)
                explanations['genres'] = f"Matches your preference for {', '.join(list(genre_overlap)[:2])}"
            
            # Rating alignment
            user_avg_rating = user_profile.get('average_rating', 3.5)
            item_avg_rating = item_features.get('average_rating', 3.5)
            if item_avg_rating >= user_avg_rating - 0.5:
                feature_importance['rating'] = min(item_avg_rating / 5.0, 1.0)
                explanations['rating'] = f"High rating ({item_avg_rating:.1f}/5.0) matches your standards"
            
            # Year preference
            user_year_pref = user_profile.get('preferred_years', [])
            item_year = item_features.get('year')
            if item_year and user_year_pref:
                year_score = 1.0 if int(item_year) in user_year_pref else 0.5
                feature_importance['year'] = year_score
                explanations['year'] = f"From {item_year}, a year you've enjoyed"
            
            # Director/Actor preference
            user_directors = set(user_profile.get('preferred_directors', []))
            user_actors = set(user_profile.get('preferred_actors', []))
            item_director = item_features.get('director', '')
            item_actors = set(item_features.get('actors', []))
            
            if item_director in user_directors:
                feature_importance['director'] = 1.0
                explanations['director'] = f"Directed by {item_director}, whom you've enjoyed"
            
            actor_overlap = user_actors & item_actors
            if actor_overlap:
                feature_importance['actors'] = len(actor_overlap) / len(user_actors | item_actors)
                explanations['actors'] = f"Stars {', '.join(list(actor_overlap)[:2])}"
            
            return {
                'importance_scores': feature_importance,
                'feature_explanations': explanations,
                'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            }
            
        except Exception as e:
            logger.error(f"Error in feature explanation: {e}")
            return {'importance_scores': {}, 'feature_explanations': {}}
    
    def _generate_similarity_explanation(self, 
                                       recommendation: Dict[str, Any],
                                       user_profile: Dict[str, Any],
                                       item_features: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate similarity-based explanation."""
        try:
            similar_items = context.get('similar_items', [])
            similarity_scores = context.get('similarity_scores', [])
            user_items = user_profile.get('rated_items', [])
            
            explanations = {}
            
            if similar_items:
                # Find user's items that are similar to recommendation
                user_similar = []
                for item in user_items:
                    if item in similar_items:
                        user_similar.append(item)
                
                if user_similar:
                    explanations['user_similarity'] = f"Similar to {', '.join(user_similar[:2])} which you rated highly"
                else:
                    explanations['general_similarity'] = f"Similar to popular items: {', '.join(similar_items[:2])}"
            
            # Collaborative filtering explanation
            similar_users = context.get('similar_users', [])
            if similar_users:
                explanations['user_based'] = f"Users with similar tastes (similarity: {np.mean(similarity_scores):.2f}) also liked this"
            
            # Content similarity
            content_similarity = context.get('content_similarity', 0.0)
            if content_similarity > 0.7:
                explanations['content_similarity'] = f"High content similarity ({content_similarity:.2f}) to your preferences"
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error in similarity explanation: {e}")
            return {}
    
    def _generate_counterfactual_explanation(self, 
                                           recommendation: Dict[str, Any],
                                           user_profile: Dict[str, Any],
                                           item_features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate counterfactual explanations (what would change the recommendation)."""
        try:
            counterfactuals = {}
            
            # Genre counterfactuals
            item_genres = item_features.get('genres', [])
            if isinstance(item_genres, str):
                item_genres = item_genres.split('|')
            
            if item_genres:
                counterfactuals['genre'] = f"If this weren't {item_genres[0]}, it might not be recommended"
            
            # Rating counterfactuals
            item_rating = item_features.get('average_rating', 0)
            if item_rating > 4.0:
                counterfactuals['rating'] = "If this had lower ratings, it wouldn't be recommended"
            
            # Recency counterfactuals
            item_year = item_features.get('year')
            if item_year and int(item_year) >= 2020:
                counterfactuals['recency'] = "If this weren't a recent release, it might rank lower"
            
            return counterfactuals
            
        except Exception as e:
            logger.error(f"Error in counterfactual explanation: {e}")
            return {}
    
    def _generate_natural_language_explanation(self, 
                                             recommendation: Dict[str, Any],
                                             user_profile: Dict[str, Any],
                                             item_features: Dict[str, Any],
                                             explanations: Dict[str, Any]) -> str:
        """Generate natural language explanation using Gemini."""
        try:
            if not self.gemini_client:
                return explanations.get('template', 'This item is recommended for you')
            
            # Prepare context for Gemini
            context = {
                'item_title': item_features.get('title', 'this item'),
                'item_genres': item_features.get('genres', []),
                'item_rating': item_features.get('average_rating', 0),
                'user_preferences': {
                    'genres': user_profile.get('preferred_genres', []),
                    'avg_rating': user_profile.get('average_rating', 3.5),
                    'recent_items': user_profile.get('recent_items', [])[:3]
                },
                'recommendation_score': recommendation.get('score', 0.5),
                'method': recommendation.get('method', 'hybrid'),
                'feature_explanations': explanations.get('features', {}).get('feature_explanations', {}),
                'similarity_explanations': explanations.get('similarity', {})
            }
            
            prompt = self._build_explanation_prompt(context)
            
            # Get explanation from Gemini
            gemini_response = self.gemini_client.generate_explanation(prompt)
            
            if gemini_response and 'explanation' in gemini_response:
                return gemini_response['explanation']
            else:
                return explanations.get('template', 'This item is recommended for you')
                
        except Exception as e:
            logger.error(f"Error generating natural language explanation: {e}")
            return explanations.get('template', 'This item is recommended for you')
    
    def _build_explanation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for natural language explanation generation."""
        try:
            prompt = f"""
Generate a personalized, friendly explanation for why "{context['item_title']}" is recommended to this user.

Item Details:
- Title: {context['item_title']}
- Genres: {context['item_genres']}
- Rating: {context['item_rating']}/5.0

User Preferences:
- Preferred genres: {context['user_preferences']['genres']}
- Average rating preference: {context['user_preferences']['avg_rating']}/5.0
- Recent items: {context['user_preferences']['recent_items']}

Recommendation Details:
- Score: {context['recommendation_score']:.2f}
- Method: {context['method']}
- Feature matches: {context['feature_explanations']}
- Similarity reasons: {context['similarity_explanations']}

Please provide a conversational, personalized explanation (2-3 sentences) that:
1. Explains why this specific item matches their preferences
2. Mentions specific genres/features that align
3. Sounds natural and engaging
4. Avoids technical jargon

Example: "Since you enjoyed action movies like... and prefer highly-rated content, this film should be perfect for you. It combines the thrilling elements you love with..."
"""
            return prompt
            
        except Exception as e:
            logger.error(f"Error building explanation prompt: {e}")
            return "Generate a recommendation explanation for this user."
    
    def _build_reasoning_chain(self, context: Dict[str, Any]) -> List[str]:
        """Build a chain of reasoning steps for the recommendation."""
        try:
            chain = []
            
            method = context.get('method', 'unknown')
            chain.append(f"Applied {method} recommendation method")
            
            if context.get('user_similarity'):
                chain.append("Found users with similar preferences")
            
            if context.get('content_features'):
                chain.append("Analyzed content features and user preferences")
            
            if context.get('diversity_boost'):
                chain.append("Applied diversity boosting for varied recommendations")
            
            if context.get('context_awareness'):
                chain.append("Considered current context and time")
            
            score = context.get('final_score', 0)
            chain.append(f"Generated final recommendation score: {score:.3f}")
            
            return chain
            
        except Exception as e:
            logger.error(f"Error building reasoning chain: {e}")
            return ["Generated recommendation using hybrid approach"]
    
    def generate_group_explanation(self, 
                                  group_recommendation: Dict[str, Any],
                                  group_members: List[Dict[str, Any]],
                                  aggregation_method: str) -> Dict[str, Any]:
        """Generate explanation for group recommendations."""
        try:
            explanations = {}
            
            # Method-specific explanations
            if aggregation_method == 'average':
                explanations['method'] = "Recommended based on the average preferences of all group members"
            elif aggregation_method == 'least_misery':
                explanations['method'] = "Selected to ensure no group member strongly dislikes it"
            elif aggregation_method == 'most_pleasure':
                explanations['method'] = "Chosen because at least one member will love it"
            elif aggregation_method == 'fairness':
                explanations['method'] = "Balanced to fairly represent everyone's preferences"
            
            # Group consensus analysis
            member_scores = group_recommendation.get('member_scores', {})
            if member_scores:
                avg_score = np.mean(list(member_scores.values()))
                min_score = min(member_scores.values())
                max_score = max(member_scores.values())
                
                explanations['consensus'] = {
                    'average_appeal': avg_score,
                    'minimum_satisfaction': min_score,
                    'maximum_enthusiasm': max_score,
                    'agreement_level': 1 - (max_score - min_score)  # Lower variance = higher agreement
                }
            
            # Member-specific explanations
            member_explanations = {}
            for member in group_members:
                member_id = member.get('user_id')
                if member_id in member_scores:
                    score = member_scores[member_id]
                    if score > 0.7:
                        member_explanations[member_id] = "Will likely love this choice"
                    elif score > 0.5:
                        member_explanations[member_id] = "Should enjoy this recommendation"
                    else:
                        member_explanations[member_id] = "May find this acceptable"
            
            explanations['member_explanations'] = member_explanations
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating group explanation: {e}")
            return {'method': 'Group recommendation generated using collaborative approach'}
    
    def explain_recommendation_changes(self, 
                                     old_recommendations: List[Dict[str, Any]],
                                     new_recommendations: List[Dict[str, Any]],
                                     change_reason: str) -> Dict[str, Any]:
        """Explain why recommendations changed."""
        try:
            old_items = set(rec['item_id'] for rec in old_recommendations)
            new_items = set(rec['item_id'] for rec in new_recommendations)
            
            added_items = new_items - old_items
            removed_items = old_items - new_items
            unchanged_items = old_items & new_items
            
            explanation = {
                'change_summary': {
                    'added_count': len(added_items),
                    'removed_count': len(removed_items),
                    'unchanged_count': len(unchanged_items),
                    'total_changes': len(added_items) + len(removed_items)
                },
                'change_reason': change_reason,
                'added_items': list(added_items)[:5],  # Limit to top 5
                'removed_items': list(removed_items)[:5]
            }
            
            # Generate natural language explanation
            if change_reason == 'preference_drift':
                explanation['message'] = f"Updated {len(added_items)} recommendations based on your evolving preferences"
            elif change_reason == 'new_feedback':
                explanation['message'] = f"Refined recommendations based on your recent ratings and feedback"
            elif change_reason == 'context_change':
                explanation['message'] = f"Adapted recommendations for your current context and situation"
            else:
                explanation['message'] = f"Updated recommendations: added {len(added_items)}, removed {len(removed_items)}"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining recommendation changes: {e}")
            return {'change_reason': change_reason, 'message': 'Recommendations have been updated'}
    
    def get_explanation_quality_score(self, explanation: Dict[str, Any]) -> float:
        """Evaluate the quality of an explanation."""
        try:
            score = 0.0
            max_score = 0.0
            
            # Check for different explanation types
            if explanation.get('template'):
                score += 0.2
            max_score += 0.2
            
            if explanation.get('features', {}).get('feature_explanations'):
                score += 0.3
            max_score += 0.3
            
            if explanation.get('similarity'):
                score += 0.2
            max_score += 0.2
            
            if explanation.get('natural_language'):
                score += 0.3
            max_score += 0.3
            
            # Quality indicators
            reasoning_chain = explanation.get('reasoning_chain', [])
            if len(reasoning_chain) >= 3:
                score += 0.1
            
            confidence = explanation.get('confidence', 0)
            if confidence > 0.7:
                score += 0.1
            
            max_score += 0.2
            
            return score / max_score if max_score > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error computing explanation quality: {e}")
            return 0.0
