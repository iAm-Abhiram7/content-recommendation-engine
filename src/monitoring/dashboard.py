"""
Monitoring dashboard for the adaptive learning system.
Provides real-time metrics, visualizations, and system health monitoring.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import numpy as np

from ..pipeline_integration import AdaptiveLearningPipeline
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class AdaptiveLearningMonitor:
    """
    Comprehensive monitoring system for the adaptive learning pipeline.
    Tracks metrics, generates insights, and provides dashboard data.
    """
    
    def __init__(self, pipeline: AdaptiveLearningPipeline):
        """Initialize the monitoring system."""
        self.pipeline = pipeline
        self.metrics_history = []
        self.alerts = []
        self._monitoring = False
        
    async def start_monitoring(self, interval: int = 60):
        """Start continuous monitoring."""
        self._monitoring = True
        logger.info("Starting adaptive learning monitoring")
        
        while self._monitoring:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self._monitoring = False
        logger.info("Stopped adaptive learning monitoring")
    
    async def _collect_metrics(self):
        """Collect current system metrics."""
        try:
            status = self.pipeline.get_pipeline_status()
            timestamp = datetime.utcnow()
            
            metrics = {
                'timestamp': timestamp,
                'pipeline_running': status.get('running', False),
                'total_feedback_processed': status.get('metrics', {}).get('total_feedback_processed', 0),
                'total_adaptations': status.get('metrics', {}).get('total_adaptations', 0),
                'drift_detections': status.get('metrics', {}).get('drift_detections', 0),
                'active_users': status.get('metrics', {}).get('active_users', 0),
                'component_health': self._calculate_component_health(status.get('components', {}))
            }
            
            # Add performance metrics
            performance_metrics = await self._get_performance_metrics()
            metrics.update(performance_metrics)
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Keep only last 24 hours of metrics
            cutoff_time = timestamp - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _calculate_component_health(self, components: Dict[str, Any]) -> float:
        """Calculate overall component health score."""
        if not components:
            return 0.0
        
        healthy_components = 0
        total_components = len(components)
        
        for component, status in components.items():
            if isinstance(status, dict) and status.get('status') == 'healthy':
                healthy_components += 1
            elif status == 'healthy':
                healthy_components += 1
        
        return healthy_components / total_components if total_components > 0 else 0.0
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance-related metrics."""
        try:
            # Get feedback processing performance
            feedback_stats = await self.pipeline.feedback_processor.get_performance_stats()
            
            # Get drift detection performance
            drift_stats = await self.pipeline.drift_detector.get_performance_stats()
            
            # Get adaptation performance
            adaptation_stats = await self.pipeline.adaptation_engine.get_performance_stats()
            
            return {
                'avg_feedback_processing_time': feedback_stats.get('avg_processing_time', 0),
                'feedback_queue_size': feedback_stats.get('queue_size', 0),
                'drift_detection_accuracy': drift_stats.get('accuracy', 0),
                'false_positive_rate': drift_stats.get('false_positive_rate', 0),
                'adaptation_success_rate': adaptation_stats.get('success_rate', 0),
                'avg_adaptation_time': adaptation_stats.get('avg_adaptation_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        try:
            if not self.metrics_history:
                return
            
            latest_metrics = self.metrics_history[-1]
            alerts = []
            
            # Check component health
            component_health = latest_metrics.get('component_health', 1.0)
            if component_health < 0.8:
                alerts.append({
                    'level': 'warning',
                    'type': 'component_health',
                    'message': f"Component health degraded: {component_health:.2%}",
                    'timestamp': latest_metrics['timestamp']
                })
            
            # Check feedback processing queue
            queue_size = latest_metrics.get('feedback_queue_size', 0)
            if queue_size > 10000:
                alerts.append({
                    'level': 'error',
                    'type': 'queue_overflow',
                    'message': f"Feedback queue size critical: {queue_size}",
                    'timestamp': latest_metrics['timestamp']
                })
            elif queue_size > 5000:
                alerts.append({
                    'level': 'warning',
                    'type': 'queue_high',
                    'message': f"Feedback queue size high: {queue_size}",
                    'timestamp': latest_metrics['timestamp']
                })
            
            # Check drift detection accuracy
            drift_accuracy = latest_metrics.get('drift_detection_accuracy', 1.0)
            if drift_accuracy < 0.7:
                alerts.append({
                    'level': 'warning',
                    'type': 'drift_accuracy',
                    'message': f"Drift detection accuracy low: {drift_accuracy:.2%}",
                    'timestamp': latest_metrics['timestamp']
                })
            
            # Check false positive rate
            false_positive_rate = latest_metrics.get('false_positive_rate', 0.0)
            if false_positive_rate > 0.2:
                alerts.append({
                    'level': 'warning',
                    'type': 'false_positives',
                    'message': f"High false positive rate: {false_positive_rate:.2%}",
                    'timestamp': latest_metrics['timestamp']
                })
            
            # Add new alerts
            self.alerts.extend(alerts)
            
            # Keep only recent alerts (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.alerts = [
                alert for alert in self.alerts 
                if alert['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            if not self.metrics_history:
                return {'error': 'No metrics available'}
            
            latest_metrics = self.metrics_history[-1]
            
            # Calculate trends
            trends = self._calculate_trends()
            
            # Generate visualizations
            visualizations = self._generate_visualizations()
            
            # Get system summary
            summary = self._get_system_summary(latest_metrics)
            
            return {
                'summary': summary,
                'trends': trends,
                'visualizations': visualizations,
                'alerts': self.alerts[-10:],  # Last 10 alerts
                'last_updated': latest_metrics['timestamp'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {'error': str(e)}
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate metric trends."""
        if len(self.metrics_history) < 2:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        trends = {}
        
        # Calculate trends for key metrics
        numeric_columns = [
            'total_feedback_processed', 'total_adaptations', 
            'drift_detections', 'active_users', 'component_health'
        ]
        
        for column in numeric_columns:
            if column in df.columns:
                # Calculate hourly trend
                recent_values = df[column].tail(60).values  # Last hour
                if len(recent_values) > 1:
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    trends[f'{column}_trend'] = trend
                    trends[f'{column}_current'] = recent_values[-1]
                    
                    # Calculate percentage change
                    if len(recent_values) >= 2 and recent_values[-2] != 0:
                        pct_change = ((recent_values[-1] - recent_values[-2]) / recent_values[-2]) * 100
                        trends[f'{column}_pct_change'] = pct_change
        
        return trends
    
    def _generate_visualizations(self) -> Dict[str, Any]:
        """Generate dashboard visualizations."""
        if len(self.metrics_history) < 2:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        visualizations = {}
        
        try:
            # Time series plot for key metrics
            fig_metrics = go.Figure()
            
            # Add traces for each metric
            metrics_to_plot = [
                ('total_feedback_processed', 'Feedback Processed'),
                ('total_adaptations', 'Adaptations'),
                ('drift_detections', 'Drift Detections'),
                ('active_users', 'Active Users')
            ]
            
            for metric, name in metrics_to_plot:
                if metric in df.columns:
                    fig_metrics.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        mode='lines+markers',
                        name=name,
                        line=dict(width=2)
                    ))
            
            fig_metrics.update_layout(
                title='System Metrics Over Time',
                xaxis_title='Time',
                yaxis_title='Count',
                hovermode='x unified',
                template='plotly_white'
            )
            
            visualizations['metrics_timeline'] = json.loads(
                json.dumps(fig_metrics, cls=PlotlyJSONEncoder)
            )
            
            # Component health gauge
            latest_health = df['component_health'].iloc[-1] if 'component_health' in df.columns else 0
            
            fig_health = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=latest_health * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Component Health (%)"},
                delta={'reference': 95},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            visualizations['component_health'] = json.loads(
                json.dumps(fig_health, cls=PlotlyJSONEncoder)
            )
            
            # Performance metrics heatmap
            performance_metrics = [
                'avg_feedback_processing_time',
                'drift_detection_accuracy',
                'adaptation_success_rate'
            ]
            
            performance_data = []
            for metric in performance_metrics:
                if metric in df.columns:
                    recent_values = df[metric].tail(24).values  # Last 24 data points
                    performance_data.append(recent_values)
            
            if performance_data:
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=performance_data,
                    y=performance_metrics,
                    colorscale='RdYlGn',
                    text=performance_data,
                    texttemplate='%{text:.3f}',
                    showscale=True
                ))
                
                fig_heatmap.update_layout(
                    title='Performance Metrics Heatmap',
                    xaxis_title='Time (Recent 24 Points)',
                    yaxis_title='Metrics'
                )
                
                visualizations['performance_heatmap'] = json.loads(
                    json.dumps(fig_heatmap, cls=PlotlyJSONEncoder)
                )
            
            # Alert distribution pie chart
            if self.alerts:
                alert_counts = {}
                for alert in self.alerts:
                    alert_type = alert['type']
                    alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
                
                fig_alerts = go.Figure(data=[go.Pie(
                    labels=list(alert_counts.keys()),
                    values=list(alert_counts.values()),
                    hole=.3
                )])
                
                fig_alerts.update_layout(
                    title='Alert Distribution (Last 24h)',
                    annotations=[dict(text='Alerts', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                
                visualizations['alert_distribution'] = json.loads(
                    json.dumps(fig_alerts, cls=PlotlyJSONEncoder)
                )
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _get_system_summary(self, latest_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get system summary statistics."""
        return {
            'status': 'running' if latest_metrics.get('pipeline_running', False) else 'stopped',
            'component_health': latest_metrics.get('component_health', 0.0),
            'total_feedback_processed': latest_metrics.get('total_feedback_processed', 0),
            'total_adaptations': latest_metrics.get('total_adaptations', 0),
            'active_users': latest_metrics.get('active_users', 0),
            'recent_alerts': len([
                alert for alert in self.alerts 
                if alert['timestamp'] > datetime.utcnow() - timedelta(hours=1)
            ]),
            'avg_processing_time': latest_metrics.get('avg_feedback_processing_time', 0),
            'queue_status': self._get_queue_status(latest_metrics),
            'uptime': self._calculate_uptime()
        }
    
    def _get_queue_status(self, metrics: Dict[str, Any]) -> str:
        """Get queue status description."""
        queue_size = metrics.get('feedback_queue_size', 0)
        
        if queue_size == 0:
            return 'empty'
        elif queue_size < 1000:
            return 'normal'
        elif queue_size < 5000:
            return 'high'
        else:
            return 'critical'
    
    def _calculate_uptime(self) -> float:
        """Calculate system uptime percentage."""
        if len(self.metrics_history) < 2:
            return 100.0
        
        # Calculate uptime based on pipeline_running status
        running_count = sum(1 for m in self.metrics_history if m.get('pipeline_running', False))
        total_count = len(self.metrics_history)
        
        return (running_count / total_count) * 100 if total_count > 0 else 0.0
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        try:
            # Get user preferences and confidence
            preferences = await self.pipeline.preference_tracker.get_user_preferences(user_id)
            confidence = await self.pipeline.confidence_scorer.calculate_confidence(user_id, preferences)
            
            # Get drift analysis
            drift_analysis = await self.pipeline.drift_detector.analyze_user_drift(user_id)
            
            # Get adaptation history
            adaptation_history = await self.pipeline.preference_manager.get_adaptation_history(user_id)
            
            # Get feedback stats
            feedback_stats = await self.pipeline.feedback_collector.get_user_feedback_stats(user_id)
            
            return {
                'user_id': user_id,
                'preferences': preferences,
                'confidence': confidence,
                'drift_analysis': drift_analysis,
                'adaptation_history': adaptation_history,
                'feedback_stats': feedback_stats,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user analytics for {user_id}: {e}")
            return {'error': str(e)}
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary."""
        if not self.alerts:
            return {'total': 0, 'by_level': {}, 'by_type': {}}
        
        # Count by level
        by_level = {}
        by_type = {}
        
        for alert in self.alerts:
            level = alert['level']
            alert_type = alert['type']
            
            by_level[level] = by_level.get(level, 0) + 1
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        return {
            'total': len(self.alerts),
            'by_level': by_level,
            'by_type': by_type,
            'recent_critical': [
                alert for alert in self.alerts 
                if alert['level'] == 'error' 
                and alert['timestamp'] > datetime.utcnow() - timedelta(hours=1)
            ]
        }
    
    async def export_metrics(self, format: str = 'json', time_range: str = '24h') -> str:
        """Export metrics data."""
        try:
            # Parse time range
            if time_range == '1h':
                cutoff = datetime.utcnow() - timedelta(hours=1)
            elif time_range == '24h':
                cutoff = datetime.utcnow() - timedelta(hours=24)
            elif time_range == '7d':
                cutoff = datetime.utcnow() - timedelta(days=7)
            else:
                cutoff = datetime.utcnow() - timedelta(hours=24)
            
            # Filter metrics
            filtered_metrics = [
                m for m in self.metrics_history 
                if m['timestamp'] > cutoff
            ]
            
            if format == 'json':
                return json.dumps(filtered_metrics, default=str, indent=2)
            elif format == 'csv':
                df = pd.DataFrame(filtered_metrics)
                return df.to_csv(index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return f"Error: {e}"
