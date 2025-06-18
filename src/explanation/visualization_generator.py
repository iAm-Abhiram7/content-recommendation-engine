"""
Visualization Generator for Content Recommendation Engine

This module generates visualizations to help users understand recommendation
adaptations, preference changes, and system behavior through charts and graphs.
"""

import logging
import io
import base64
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from ..utils.logging import setup_logger


class VisualizationType(Enum):
    """Types of visualizations"""
    PREFERENCE_EVOLUTION = "preference_evolution"
    RECOMMENDATION_CHANGES = "recommendation_changes"
    CONFIDENCE_TIMELINE = "confidence_timeline"
    FEATURE_IMPORTANCE = "feature_importance"
    DRIFT_DETECTION = "drift_detection"
    SEASONAL_PATTERNS = "seasonal_patterns"
    USER_ENGAGEMENT = "user_engagement"
    CATEGORY_DISTRIBUTION = "category_distribution"
    ADAPTATION_IMPACT = "adaptation_impact"


class OutputFormat(Enum):
    """Output formats for visualizations"""
    PNG = "png"
    SVG = "svg"
    HTML = "html"
    JSON = "json"
    BASE64 = "base64"


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    width: int = 800
    height: int = 600
    theme: str = "default"
    color_palette: List[str] = None
    interactive: bool = True
    show_grid: bool = True
    show_legend: bool = True
    title_font_size: int = 16
    axis_font_size: int = 12


@dataclass
class VisualizationResult:
    """Result of visualization generation"""
    visualization_type: VisualizationType
    output_format: OutputFormat
    data: Union[str, bytes, Dict]
    metadata: Dict[str, Any]
    generation_time: float
    file_size: Optional[int] = None


class VisualizationGenerator:
    """
    Generates interactive and static visualizations to explain
    recommendation adaptations and user preference changes
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Color palettes
        self.color_palettes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'preference': ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'],
            'confidence': ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c'],
            'drift': ['#3498db', '#e74c3c', '#f39c12'],
            'engagement': ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        }
        
        # Set up plotting themes
        self._setup_themes()
        
        self.logger = setup_logger(__name__)
    
    def generate_preference_evolution_chart(self,
                                          preference_history: Dict[str, List[Tuple[datetime, float]]],
                                          output_format: OutputFormat = OutputFormat.HTML,
                                          features: List[str] = None) -> VisualizationResult:
        """Generate preference evolution timeline chart"""
        start_time = datetime.now()
        
        try:
            if not preference_history:
                return self._empty_visualization_result(VisualizationType.PREFERENCE_EVOLUTION, output_format)
            
            # Prepare data
            df_data = []
            features_to_plot = features or list(preference_history.keys())[:5]  # Top 5 features
            
            for feature in features_to_plot:
                if feature in preference_history:
                    for timestamp, value in preference_history[feature]:
                        df_data.append({
                            'timestamp': timestamp,
                            'feature': feature,
                            'value': value
                        })
            
            df = pd.DataFrame(df_data)
            
            if df.empty:
                return self._empty_visualization_result(VisualizationType.PREFERENCE_EVOLUTION, output_format)
            
            # Create visualization
            if self.config.interactive:
                result = self._create_interactive_preference_evolution(df, output_format)
            else:
                result = self._create_static_preference_evolution(df, output_format)
            
            result.generation_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating preference evolution chart: {e}")
            return self._error_visualization_result(VisualizationType.PREFERENCE_EVOLUTION, output_format)
    
    def generate_recommendation_changes_chart(self,
                                            before_recommendations: List[Dict],
                                            after_recommendations: List[Dict],
                                            output_format: OutputFormat = OutputFormat.HTML) -> VisualizationResult:
        """Generate chart showing recommendation changes"""
        start_time = datetime.now()
        
        try:
            # Analyze changes
            changes = self._analyze_recommendation_changes(before_recommendations, after_recommendations)
            
            if self.config.interactive:
                result = self._create_interactive_recommendation_changes(changes, output_format)
            else:
                result = self._create_static_recommendation_changes(changes, output_format)
            
            result.generation_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation changes chart: {e}")
            return self._error_visualization_result(VisualizationType.RECOMMENDATION_CHANGES, output_format)
    
    def generate_confidence_timeline(self,
                                   confidence_history: List[Tuple[datetime, float]],
                                   output_format: OutputFormat = OutputFormat.HTML) -> VisualizationResult:
        """Generate confidence score timeline"""
        start_time = datetime.now()
        
        try:
            if not confidence_history:
                return self._empty_visualization_result(VisualizationType.CONFIDENCE_TIMELINE, output_format)
            
            # Prepare data
            df = pd.DataFrame(confidence_history, columns=['timestamp', 'confidence'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if self.config.interactive:
                result = self._create_interactive_confidence_timeline(df, output_format)
            else:
                result = self._create_static_confidence_timeline(df, output_format)
            
            result.generation_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating confidence timeline: {e}")
            return self._error_visualization_result(VisualizationType.CONFIDENCE_TIMELINE, output_format)
    
    def generate_feature_importance_chart(self,
                                        feature_importance: Dict[str, float],
                                        output_format: OutputFormat = OutputFormat.HTML) -> VisualizationResult:
        """Generate feature importance bar chart"""
        start_time = datetime.now()
        
        try:
            if not feature_importance:
                return self._empty_visualization_result(VisualizationType.FEATURE_IMPORTANCE, output_format)
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_features[:10])  # Top 10 features
            
            if self.config.interactive:
                result = self._create_interactive_feature_importance(features, importance, output_format)
            else:
                result = self._create_static_feature_importance(features, importance, output_format)
            
            result.generation_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating feature importance chart: {e}")
            return self._error_visualization_result(VisualizationType.FEATURE_IMPORTANCE, output_format)
    
    def generate_drift_detection_chart(self,
                                     drift_history: List[Dict[str, Any]],
                                     output_format: OutputFormat = OutputFormat.HTML) -> VisualizationResult:
        """Generate drift detection visualization"""
        start_time = datetime.now()
        
        try:
            if not drift_history:
                return self._empty_visualization_result(VisualizationType.DRIFT_DETECTION, output_format)
            
            # Prepare data
            df = pd.DataFrame(drift_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if self.config.interactive:
                result = self._create_interactive_drift_detection(df, output_format)
            else:
                result = self._create_static_drift_detection(df, output_format)
            
            result.generation_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating drift detection chart: {e}")
            return self._error_visualization_result(VisualizationType.DRIFT_DETECTION, output_format)
    
    def generate_seasonal_patterns_chart(self,
                                       seasonal_data: Dict[str, Any],
                                       output_format: OutputFormat = OutputFormat.HTML) -> VisualizationResult:
        """Generate seasonal patterns visualization"""
        start_time = datetime.now()
        
        try:
            if not seasonal_data:
                return self._empty_visualization_result(VisualizationType.SEASONAL_PATTERNS, output_format)
            
            if self.config.interactive:
                result = self._create_interactive_seasonal_patterns(seasonal_data, output_format)
            else:
                result = self._create_static_seasonal_patterns(seasonal_data, output_format)
            
            result.generation_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating seasonal patterns chart: {e}")
            return self._error_visualization_result(VisualizationType.SEASONAL_PATTERNS, output_format)
    
    def generate_dashboard(self,
                          dashboard_data: Dict[str, Any],
                          output_format: OutputFormat = OutputFormat.HTML) -> VisualizationResult:
        """Generate comprehensive dashboard with multiple visualizations"""
        start_time = datetime.now()
        
        try:
            if self.config.interactive:
                result = self._create_interactive_dashboard(dashboard_data, output_format)
            else:
                result = self._create_static_dashboard(dashboard_data, output_format)
            
            result.generation_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard: {e}")
            return self._error_visualization_result(VisualizationType.ADAPTATION_IMPACT, output_format)
    
    def _create_interactive_preference_evolution(self, df: pd.DataFrame, 
                                               output_format: OutputFormat) -> VisualizationResult:
        """Create interactive preference evolution chart using Plotly"""
        fig = go.Figure()
        
        colors = self.color_palettes['preference']
        
        for i, feature in enumerate(df['feature'].unique()):
            feature_data = df[df['feature'] == feature]
            
            fig.add_trace(go.Scatter(
                x=feature_data['timestamp'],
                y=feature_data['value'],
                mode='lines+markers',
                name=feature,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title='Preference Evolution Over Time',
            xaxis_title='Time',
            yaxis_title='Preference Strength',
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.show_legend,
            hovermode='x unified'
        )
        
        return self._format_plotly_output(fig, VisualizationType.PREFERENCE_EVOLUTION, output_format)
    
    def _create_static_preference_evolution(self, df: pd.DataFrame,
                                          output_format: OutputFormat) -> VisualizationResult:
        """Create static preference evolution chart using Matplotlib"""
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        colors = self.color_palettes['preference']
        
        for i, feature in enumerate(df['feature'].unique()):
            feature_data = df[df['feature'] == feature]
            ax.plot(feature_data['timestamp'], feature_data['value'], 
                   label=feature, color=colors[i % len(colors)], linewidth=2, marker='o')
        
        ax.set_title('Preference Evolution Over Time', fontsize=self.config.title_font_size)
        ax.set_xlabel('Time', fontsize=self.config.axis_font_size)
        ax.set_ylabel('Preference Strength', fontsize=self.config.axis_font_size)
        
        if self.config.show_legend:
            ax.legend()
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._format_matplotlib_output(fig, VisualizationType.PREFERENCE_EVOLUTION, output_format)
    
    def _create_interactive_recommendation_changes(self, changes: Dict[str, Any],
                                                 output_format: OutputFormat) -> VisualizationResult:
        """Create interactive recommendation changes visualization"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('New vs Removed Items', 'Category Changes', 
                          'Score Distribution', 'Rank Changes'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # New vs Removed items
        fig.add_trace(
            go.Bar(x=['New Items', 'Removed Items'], 
                  y=[changes.get('new_items', 0), changes.get('removed_items', 0)],
                  name='Item Changes',
                  marker_color=['#2ecc71', '#e74c3c']),
            row=1, col=1
        )
        
        # Category changes
        if 'category_changes' in changes:
            categories = list(changes['category_changes'].keys())
            category_deltas = list(changes['category_changes'].values())
            
            fig.add_trace(
                go.Bar(x=categories, y=category_deltas, name='Category Changes',
                      marker_color=['#3498db' if x >= 0 else '#e74c3c' for x in category_deltas]),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Recommendation Changes Analysis',
            width=self.config.width,
            height=self.config.height,
            showlegend=False
        )
        
        return self._format_plotly_output(fig, VisualizationType.RECOMMENDATION_CHANGES, output_format)
    
    def _create_static_recommendation_changes(self, changes: Dict[str, Any],
                                            output_format: OutputFormat) -> VisualizationResult:
        """Create static recommendation changes visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.config.width/100, self.config.height/100))
        
        # New vs Removed items
        items = ['New Items', 'Removed Items']
        values = [changes.get('new_items', 0), changes.get('removed_items', 0)]
        colors = ['#2ecc71', '#e74c3c']
        
        ax1.bar(items, values, color=colors)
        ax1.set_title('New vs Removed Items')
        ax1.set_ylabel('Count')
        
        # Category changes
        if 'category_changes' in changes:
            categories = list(changes['category_changes'].keys())
            category_deltas = list(changes['category_changes'].values())
            colors = ['#3498db' if x >= 0 else '#e74c3c' for x in category_deltas]
            
            ax2.bar(categories, category_deltas, color=colors)
            ax2.set_title('Category Changes')
            ax2.set_ylabel('Change in Count')
            ax2.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        ax3.axis('off')
        ax4.axis('off')
        
        plt.tight_layout()
        
        return self._format_matplotlib_output(fig, VisualizationType.RECOMMENDATION_CHANGES, output_format)
    
    def _create_interactive_confidence_timeline(self, df: pd.DataFrame,
                                              output_format: OutputFormat) -> VisualizationResult:
        """Create interactive confidence timeline"""
        fig = go.Figure()
        
        # Add confidence line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence Score',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6),
            fill='tonexty'
        ))
        
        # Add confidence bands
        high_confidence = [0.8] * len(df)
        medium_confidence = [0.6] * len(df)
        low_confidence = [0.4] * len(df)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=high_confidence,
            fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=medium_confidence,
            fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
            fillcolor='rgba(46, 204, 113, 0.2)', name='High Confidence', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=low_confidence,
            fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
            fillcolor='rgba(241, 196, 15, 0.2)', name='Medium Confidence', showlegend=False
        ))
        
        fig.update_layout(
            title='Confidence Score Timeline',
            xaxis_title='Time',
            yaxis_title='Confidence Score',
            yaxis=dict(range=[0, 1]),
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.show_legend
        )
        
        return self._format_plotly_output(fig, VisualizationType.CONFIDENCE_TIMELINE, output_format)
    
    def _create_static_confidence_timeline(self, df: pd.DataFrame,
                                         output_format: OutputFormat) -> VisualizationResult:
        """Create static confidence timeline"""
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        # Plot confidence line
        ax.plot(df['timestamp'], df['confidence'], color='#3498db', linewidth=2, marker='o')
        
        # Add confidence bands
        ax.axhspan(0.8, 1.0, alpha=0.2, color='#2ecc71', label='High Confidence')
        ax.axhspan(0.6, 0.8, alpha=0.2, color='#f1c40f', label='Medium Confidence')
        ax.axhspan(0.4, 0.6, alpha=0.2, color='#e67e22', label='Low Confidence')
        ax.axhspan(0.0, 0.4, alpha=0.2, color='#e74c3c', label='Very Low Confidence')
        
        ax.set_title('Confidence Score Timeline', fontsize=self.config.title_font_size)
        ax.set_xlabel('Time', fontsize=self.config.axis_font_size)
        ax.set_ylabel('Confidence Score', fontsize=self.config.axis_font_size)
        ax.set_ylim(0, 1)
        
        if self.config.show_legend:
            ax.legend()
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._format_matplotlib_output(fig, VisualizationType.CONFIDENCE_TIMELINE, output_format)
    
    def _create_interactive_feature_importance(self, features: Tuple, importance: Tuple,
                                             output_format: OutputFormat) -> VisualizationResult:
        """Create interactive feature importance chart"""
        fig = go.Figure()
        
        colors = ['#e74c3c' if imp > 0.7 else '#f39c12' if imp > 0.4 else '#3498db' 
                 for imp in importance]
        
        fig.add_trace(go.Bar(
            x=list(importance),
            y=list(features),
            orientation='h',
            marker_color=colors,
            text=[f'{imp:.3f}' for imp in importance],
            textposition='inside'
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            width=self.config.width,
            height=self.config.height,
            showlegend=False
        )
        
        return self._format_plotly_output(fig, VisualizationType.FEATURE_IMPORTANCE, output_format)
    
    def _create_static_feature_importance(self, features: Tuple, importance: Tuple,
                                        output_format: OutputFormat) -> VisualizationResult:
        """Create static feature importance chart"""
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        colors = ['#e74c3c' if imp > 0.7 else '#f39c12' if imp > 0.4 else '#3498db' 
                 for imp in importance]
        
        bars = ax.barh(features, importance, color=colors)
        
        # Add value labels on bars
        for bar, imp in zip(bars, importance):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.3f}', va='center', fontsize=10)
        
        ax.set_title('Feature Importance', fontsize=self.config.title_font_size)
        ax.set_xlabel('Importance Score', fontsize=self.config.axis_font_size)
        ax.set_ylabel('Features', fontsize=self.config.axis_font_size)
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return self._format_matplotlib_output(fig, VisualizationType.FEATURE_IMPORTANCE, output_format)
    
    def _create_interactive_drift_detection(self, df: pd.DataFrame,
                                          output_format: OutputFormat) -> VisualizationResult:
        """Create interactive drift detection visualization"""
        fig = go.Figure()
        
        # Add drift score line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df.get('drift_score', [0] * len(df)),
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=6)
        ))
        
        # Add threshold line
        threshold = 0.5
        fig.add_hline(y=threshold, line_dash="dash", line_color="orange", 
                     annotation_text="Drift Threshold")
        
        # Mark drift detection points
        drift_points = df[df.get('drift_detected', False)]
        if not drift_points.empty:
            fig.add_trace(go.Scatter(
                x=drift_points['timestamp'],
                y=drift_points.get('drift_score', [threshold] * len(drift_points)),
                mode='markers',
                marker=dict(size=12, color='red', symbol='triangle-up'),
                name='Drift Detected'
            ))
        
        fig.update_layout(
            title='Concept Drift Detection',
            xaxis_title='Time',
            yaxis_title='Drift Score',
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.show_legend
        )
        
        return self._format_plotly_output(fig, VisualizationType.DRIFT_DETECTION, output_format)
    
    def _create_static_drift_detection(self, df: pd.DataFrame,
                                     output_format: OutputFormat) -> VisualizationResult:
        """Create static drift detection visualization"""
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        # Plot drift score
        ax.plot(df['timestamp'], df.get('drift_score', [0] * len(df)), 
               color='#e74c3c', linewidth=2, marker='o', label='Drift Score')
        
        # Add threshold line
        threshold = 0.5
        ax.axhline(y=threshold, color='orange', linestyle='--', label='Drift Threshold')
        
        # Mark drift detection points
        drift_points = df[df.get('drift_detected', False)]
        if not drift_points.empty:
            ax.scatter(drift_points['timestamp'], 
                      drift_points.get('drift_score', [threshold] * len(drift_points)),
                      color='red', s=100, marker='^', label='Drift Detected', zorder=5)
        
        ax.set_title('Concept Drift Detection', fontsize=self.config.title_font_size)
        ax.set_xlabel('Time', fontsize=self.config.axis_font_size)
        ax.set_ylabel('Drift Score', fontsize=self.config.axis_font_size)
        
        if self.config.show_legend:
            ax.legend()
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._format_matplotlib_output(fig, VisualizationType.DRIFT_DETECTION, output_format)
    
    def _create_interactive_seasonal_patterns(self, seasonal_data: Dict[str, Any],
                                            output_format: OutputFormat) -> VisualizationResult:
        """Create interactive seasonal patterns visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Patterns', 'Weekly Patterns', 
                          'Monthly Patterns', 'Seasonal Summary')
        )
        
        # Daily pattern
        if 'daily' in seasonal_data:
            daily = seasonal_data['daily']
            hours = list(range(24))
            fig.add_trace(
                go.Scatter(x=hours, y=daily.get('values', [0]*24), 
                          mode='lines+markers', name='Daily Pattern'),
                row=1, col=1
            )
        
        # Weekly pattern
        if 'weekly' in seasonal_data:
            weekly = seasonal_data['weekly']
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig.add_trace(
                go.Scatter(x=days, y=weekly.get('values', [0]*7), 
                          mode='lines+markers', name='Weekly Pattern'),
                row=1, col=2
            )
        
        # Monthly pattern
        if 'monthly' in seasonal_data:
            monthly = seasonal_data['monthly']
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig.add_trace(
                go.Scatter(x=months, y=monthly.get('values', [0]*12), 
                          mode='lines+markers', name='Monthly Pattern'),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Seasonal Usage Patterns',
            width=self.config.width,
            height=self.config.height,
            showlegend=False
        )
        
        return self._format_plotly_output(fig, VisualizationType.SEASONAL_PATTERNS, output_format)
    
    def _create_static_seasonal_patterns(self, seasonal_data: Dict[str, Any],
                                       output_format: OutputFormat) -> VisualizationResult:
        """Create static seasonal patterns visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.config.width/100, self.config.height/100))
        
        # Daily pattern
        if 'daily' in seasonal_data:
            daily = seasonal_data['daily']
            hours = list(range(24))
            ax1.plot(hours, daily.get('values', [0]*24), marker='o')
            ax1.set_title('Daily Pattern')
            ax1.set_xlabel('Hour')
            ax1.set_ylabel('Activity')
        
        # Weekly pattern
        if 'weekly' in seasonal_data:
            weekly = seasonal_data['weekly']
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax2.plot(days, weekly.get('values', [0]*7), marker='o')
            ax2.set_title('Weekly Pattern')
            ax2.set_xlabel('Day')
            ax2.set_ylabel('Activity')
            ax2.tick_params(axis='x', rotation=45)
        
        # Monthly pattern
        if 'monthly' in seasonal_data:
            monthly = seasonal_data['monthly']
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax3.plot(months, monthly.get('values', [0]*12), marker='o')
            ax3.set_title('Monthly Pattern')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Activity')
            ax3.tick_params(axis='x', rotation=45)
        
        # Hide empty subplot
        ax4.axis('off')
        
        plt.tight_layout()
        
        return self._format_matplotlib_output(fig, VisualizationType.SEASONAL_PATTERNS, output_format)
    
    def _create_interactive_dashboard(self, dashboard_data: Dict[str, Any],
                                    output_format: OutputFormat) -> VisualizationResult:
        """Create comprehensive interactive dashboard"""
        # This would create a multi-panel dashboard combining multiple visualizations
        # For brevity, returning a placeholder implementation
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Preference Evolution', 'Confidence Timeline',
                          'Feature Importance', 'Drift Detection',
                          'Recommendation Changes', 'Seasonal Patterns'),
            vertical_spacing=0.08
        )
        
        # Add placeholder data for each subplot
        # In a real implementation, this would use the actual dashboard_data
        
        fig.update_layout(
            title='Recommendation System Dashboard',
            width=self.config.width * 1.5,
            height=self.config.height * 2,
            showlegend=False
        )
        
        return self._format_plotly_output(fig, VisualizationType.ADAPTATION_IMPACT, output_format)
    
    def _create_static_dashboard(self, dashboard_data: Dict[str, Any],
                               output_format: OutputFormat) -> VisualizationResult:
        """Create comprehensive static dashboard"""
        fig, axes = plt.subplots(3, 2, figsize=(self.config.width/50, self.config.height/50))
        
        # Placeholder implementation
        for i, ax in enumerate(axes.flat):
            ax.text(0.5, 0.5, f'Dashboard Panel {i+1}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Panel {i+1}')
        
        plt.tight_layout()
        
        return self._format_matplotlib_output(fig, VisualizationType.ADAPTATION_IMPACT, output_format)
    
    def _analyze_recommendation_changes(self, before: List[Dict], 
                                      after: List[Dict]) -> Dict[str, Any]:
        """Analyze changes between recommendation sets"""
        before_ids = {rec.get('id') for rec in before}
        after_ids = {rec.get('id') for rec in after}
        
        changes = {
            'new_items': len(after_ids - before_ids),
            'removed_items': len(before_ids - after_ids),
            'common_items': len(before_ids & after_ids)
        }
        
        # Analyze category changes
        before_categories = {}
        after_categories = {}
        
        for rec in before:
            cat = rec.get('category', 'unknown')
            before_categories[cat] = before_categories.get(cat, 0) + 1
        
        for rec in after:
            cat = rec.get('category', 'unknown')
            after_categories[cat] = after_categories.get(cat, 0) + 1
        
        category_changes = {}
        all_categories = set(before_categories.keys()) | set(after_categories.keys())
        
        for cat in all_categories:
            before_count = before_categories.get(cat, 0)
            after_count = after_categories.get(cat, 0)
            if before_count != after_count:
                category_changes[cat] = after_count - before_count
        
        changes['category_changes'] = category_changes
        
        return changes
    
    def _format_plotly_output(self, fig: go.Figure, viz_type: VisualizationType,
                            output_format: OutputFormat) -> VisualizationResult:
        """Format Plotly figure according to output format"""
        metadata = {
            'library': 'plotly',
            'interactive': True,
            'width': self.config.width,
            'height': self.config.height
        }
        
        if output_format == OutputFormat.HTML:
            html_str = pio.to_html(fig, include_plotlyjs=True)
            return VisualizationResult(
                visualization_type=viz_type,
                output_format=output_format,
                data=html_str,
                metadata=metadata,
                generation_time=0.0,
                file_size=len(html_str.encode())
            )
        elif output_format == OutputFormat.JSON:
            json_str = pio.to_json(fig)
            return VisualizationResult(
                visualization_type=viz_type,
                output_format=output_format,
                data=json_str,
                metadata=metadata,
                generation_time=0.0,
                file_size=len(json_str.encode())
            )
        else:
            # Convert to image format
            img_bytes = pio.to_image(fig, format=output_format.value, 
                                   width=self.config.width, height=self.config.height)
            
            if output_format == OutputFormat.BASE64:
                img_b64 = base64.b64encode(img_bytes).decode()
                return VisualizationResult(
                    visualization_type=viz_type,
                    output_format=output_format,
                    data=img_b64,
                    metadata=metadata,
                    generation_time=0.0,
                    file_size=len(img_bytes)
                )
            else:
                return VisualizationResult(
                    visualization_type=viz_type,
                    output_format=output_format,
                    data=img_bytes,
                    metadata=metadata,
                    generation_time=0.0,
                    file_size=len(img_bytes)
                )
    
    def _format_matplotlib_output(self, fig: Figure, viz_type: VisualizationType,
                                output_format: OutputFormat) -> VisualizationResult:
        """Format Matplotlib figure according to output format"""
        metadata = {
            'library': 'matplotlib',
            'interactive': False,
            'width': self.config.width,
            'height': self.config.height
        }
        
        if output_format == OutputFormat.BASE64:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return VisualizationResult(
                visualization_type=viz_type,
                output_format=output_format,
                data=img_b64,
                metadata=metadata,
                generation_time=0.0,
                file_size=len(buffer.getvalue())
            )
        else:
            buffer = io.BytesIO()
            fig.savefig(buffer, format=output_format.value, dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_bytes = buffer.getvalue()
            plt.close(fig)
            
            return VisualizationResult(
                visualization_type=viz_type,
                output_format=output_format,
                data=img_bytes,
                metadata=metadata,
                generation_time=0.0,
                file_size=len(img_bytes)
            )
    
    def _setup_themes(self):
        """Setup plotting themes"""
        # Matplotlib theme
        plt.style.use('default')
        sns.set_palette(self.color_palettes['default'])
        
        # Plotly theme
        pio.templates.default = "plotly_white"
    
    def _empty_visualization_result(self, viz_type: VisualizationType, 
                                  output_format: OutputFormat) -> VisualizationResult:
        """Return empty visualization result"""
        return VisualizationResult(
            visualization_type=viz_type,
            output_format=output_format,
            data="No data available for visualization",
            metadata={'error': 'empty_data'},
            generation_time=0.0
        )
    
    def _error_visualization_result(self, viz_type: VisualizationType,
                                  output_format: OutputFormat) -> VisualizationResult:
        """Return error visualization result"""
        return VisualizationResult(
            visualization_type=viz_type,
            output_format=output_format,
            data="Error generating visualization",
            metadata={'error': 'generation_error'},
            generation_time=0.0
        )
