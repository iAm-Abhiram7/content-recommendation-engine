"""
Intelligent Alerting System
ML-based alerting with anomaly detection and alert management
"""

import asyncio
import logging
import json
import smtplib
import aiohttp
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

logger = logging.getLogger(__name__)

@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    metric: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'anomaly'
    threshold: float
    severity: str  # 'info', 'warning', 'critical'
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default
    tags: List[str] = field(default_factory=list)

@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_id: str
    metric: str
    value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    id: str
    type: str  # 'email', 'slack', 'webhook', 'pagerduty'
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[str] = field(default_factory=lambda: ['warning', 'critical'])

class AnomalyDetector:
    """ML-based anomaly detection for metrics"""
    
    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.models = {}
        self.scalers = {}
        self.metric_history = {}
    
    def add_metric_point(self, metric_name: str, value: float, timestamp: float):
        """Add a metric point for anomaly detection"""
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append((timestamp, value))
        
        # Keep only recent history
        cutoff_time = timestamp - 86400  # 24 hours
        self.metric_history[metric_name] = [
            (ts, val) for ts, val in self.metric_history[metric_name]
            if ts >= cutoff_time
        ]
        
        # Train model if we have enough data
        if len(self.metric_history[metric_name]) >= self.window_size:
            self._train_model(metric_name)
    
    def _train_model(self, metric_name: str):
        """Train anomaly detection model for a metric"""
        try:
            data = np.array([val for _, val in self.metric_history[metric_name]])
            
            # Prepare features (value, time-based features)
            timestamps = np.array([ts for ts, _ in self.metric_history[metric_name]])
            
            # Create time-based features
            hours = (timestamps % 86400) / 3600  # Hour of day
            days = ((timestamps // 86400) % 7)   # Day of week
            
            features = np.column_stack([
                data,
                hours[-len(data):],
                days[-len(data):],
                np.gradient(data),  # Rate of change
                np.convolve(data, np.ones(min(5, len(data)))/(min(5, len(data))), mode='same')  # Moving average
            ])
            
            # Handle infinite or NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
            
            features_scaled = self.scalers[metric_name].fit_transform(features)
            
            # Train isolation forest
            self.models[metric_name] = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.models[metric_name].fit(features_scaled)
            
        except Exception as e:
            logger.error(f"Error training anomaly model for {metric_name}: {e}")
    
    def is_anomaly(self, metric_name: str, value: float, timestamp: float) -> Dict[str, Any]:
        """Check if a metric value is anomalous"""
        if metric_name not in self.models or metric_name not in self.metric_history:
            return {'is_anomaly': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
        
        try:
            # Prepare features for current point
            recent_data = np.array([val for _, val in self.metric_history[metric_name][-20:]])
            recent_timestamps = np.array([ts for ts, _ in self.metric_history[metric_name][-20:]])
            
            hour = (timestamp % 86400) / 3600
            day = ((timestamp // 86400) % 7)
            
            # Calculate gradient (rate of change)
            if len(recent_data) > 1:
                gradient = value - recent_data[-1]
            else:
                gradient = 0.0
            
            # Calculate moving average
            if len(recent_data) >= 5:
                moving_avg = np.mean(recent_data[-5:])
            else:
                moving_avg = np.mean(recent_data) if len(recent_data) > 0 else value
            
            features = np.array([[value, hour, day, gradient, moving_avg]])
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            features_scaled = self.scalers[metric_name].transform(features)
            
            # Predict anomaly
            prediction = self.models[metric_name].predict(features_scaled)[0]
            score = self.models[metric_name].decision_function(features_scaled)[0]
            
            is_anomaly = prediction == -1
            confidence = abs(score)  # Distance from decision boundary
            
            # Determine reason for anomaly
            reason = 'normal'
            if is_anomaly:
                if len(recent_data) > 0:
                    if abs(value - np.mean(recent_data)) > 3 * np.std(recent_data):
                        reason = 'statistical_outlier'
                    elif gradient > np.percentile([val for _, val in self.metric_history[metric_name]], 95):
                        reason = 'rapid_increase'
                    elif gradient < np.percentile([val for _, val in self.metric_history[metric_name]], 5):
                        reason = 'rapid_decrease'
                    else:
                        reason = 'pattern_anomaly'
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'reason': reason,
                'score': score
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomaly for {metric_name}: {e}")
            return {'is_anomaly': False, 'confidence': 0.0, 'reason': 'error'}

class NotificationService:
    """Multi-channel notification service"""
    
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
    
    def add_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.channels[channel.id] = channel
    
    async def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        for channel in self.channels.values():
            if not channel.enabled:
                continue
            
            if alert.severity not in channel.severity_filter:
                continue
            
            try:
                if channel.type == 'email':
                    await self._send_email(alert, channel)
                elif channel.type == 'slack':
                    await self._send_slack(alert, channel)
                elif channel.type == 'webhook':
                    await self._send_webhook(alert, channel)
                elif channel.type == 'pagerduty':
                    await self._send_pagerduty(alert, channel)
                    
            except Exception as e:
                logger.error(f"Error sending alert through {channel.id}: {e}")
    
    async def _send_email(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        config = channel.config
        
        msg = MimeMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(config['to_emails'])
        msg['Subject'] = f"[{alert.severity.upper()}] {alert.message}"
        
        body = f"""
        Alert: {alert.message}
        
        Metric: {alert.metric}
        Current Value: {alert.value}
        Threshold: {alert.threshold}
        Severity: {alert.severity}
        Time: {alert.timestamp}
        
        Tags: {', '.join(alert.tags)}
        
        Alert ID: {alert.id}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        if config.get('use_tls'):
            server.starttls()
        if config.get('username'):
            server.login(config['username'], config['password'])
        
        server.send_message(msg)
        server.quit()
    
    async def _send_slack(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        config = channel.config
        
        # Emoji based on severity
        emoji = {
            'info': ':information_source:',
            'warning': ':warning:',
            'critical': ':rotating_light:'
        }.get(alert.severity, ':exclamation:')
        
        # Color based on severity
        color = {
            'info': '#36a64f',
            'warning': '#ff9500',
            'critical': '#ff0000'
        }.get(alert.severity, '#cccccc')
        
        payload = {
            "text": f"{emoji} Alert: {alert.message}",
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {"title": "Metric", "value": alert.metric, "short": True},
                        {"title": "Value", "value": str(alert.value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Severity", "value": alert.severity, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                        {"title": "Alert ID", "value": alert.id, "short": True}
                    ]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['webhook_url'], json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack webhook returned {response.status}")
    
    async def _send_webhook(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        config = channel.config
        
        payload = {
            "alert_id": alert.id,
            "rule_id": alert.rule_id,
            "metric": alert.metric,
            "value": alert.value,
            "threshold": alert.threshold,
            "severity": alert.severity,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "tags": alert.tags,
            "metadata": alert.metadata
        }
        
        headers = config.get('headers', {})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['url'], json=payload, headers=headers) as response:
                if response.status not in [200, 201, 202]:
                    raise Exception(f"Webhook returned {response.status}")
    
    async def _send_pagerduty(self, alert: Alert, channel: NotificationChannel):
        """Send PagerDuty notification"""
        config = channel.config
        
        payload = {
            "routing_key": config['integration_key'],
            "event_action": "trigger",
            "dedup_key": f"{alert.rule_id}_{alert.metric}",
            "payload": {
                "summary": alert.message,
                "severity": alert.severity,
                "source": "recommendation-engine",
                "component": alert.metric,
                "group": "recommendations",
                "class": "metric_alert",
                "custom_details": {
                    "metric": alert.metric,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "alert_id": alert.id,
                    "tags": alert.tags
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            ) as response:
                if response.status != 202:
                    raise Exception(f"PagerDuty API returned {response.status}")

class AlertManager:
    """Intelligent alert management system"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.anomaly_detector = AnomalyDetector()
        self.notification_service = NotificationService()
        
        # Alert suppression and grouping
        self.suppressed_alerts: Set[str] = set()
        self.alert_groups: Dict[str, List[str]] = {}
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.notification_service.add_channel(channel)
    
    async def check_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Check metric against all relevant rules"""
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        # Add to anomaly detector
        self.anomaly_detector.add_metric_point(metric_name, value, timestamp)
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if rule.metric != metric_name:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule.id):
                continue
            
            # Evaluate rule
            should_alert = await self._evaluate_rule(rule, value, timestamp)
            
            if should_alert:
                await self._trigger_alert(rule, value, timestamp)
    
    async def _evaluate_rule(self, rule: AlertRule, value: float, timestamp: float) -> bool:
        """Evaluate if rule should trigger alert"""
        if rule.condition == 'greater_than':
            return value > rule.threshold
        elif rule.condition == 'less_than':
            return value < rule.threshold
        elif rule.condition == 'equals':
            return abs(value - rule.threshold) < 0.001
        elif rule.condition == 'anomaly':
            anomaly_result = self.anomaly_detector.is_anomaly(rule.metric, value, timestamp)
            return anomaly_result['is_anomaly'] and anomaly_result['confidence'] > rule.threshold
        
        return False
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_id not in self.last_alert_times:
            return False
        
        rule = self.rules[rule_id]
        last_alert = self.last_alert_times[rule_id]
        cooldown_end = last_alert + timedelta(seconds=rule.cooldown_seconds)
        
        return datetime.now() < cooldown_end
    
    async def _trigger_alert(self, rule: AlertRule, value: float, timestamp: float):
        """Trigger an alert"""
        alert_id = f"alert_{rule.id}_{int(timestamp)}"
        
        # Check for existing active alert
        existing_key = f"{rule.id}_{rule.metric}"
        if existing_key in self.active_alerts:
            # Update existing alert
            self.active_alerts[existing_key].value = value
            self.active_alerts[existing_key].timestamp = datetime.fromtimestamp(timestamp)
            return
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            metric=rule.metric,
            value=value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=self._generate_alert_message(rule, value),
            timestamp=datetime.fromtimestamp(timestamp),
            tags=rule.tags.copy()
        )
        
        # Add to active alerts
        self.active_alerts[existing_key] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.id] = alert.timestamp
        
        # Log alert
        logger.warning(f"ALERT TRIGGERED: {alert.message}")
        
        # Send notifications
        await self.notification_service.send_alert(alert)
        
        # Apply intelligent grouping and suppression
        await self._apply_alert_intelligence(alert)
    
    def _generate_alert_message(self, rule: AlertRule, value: float) -> str:
        """Generate human-readable alert message"""
        if rule.condition == 'greater_than':
            return f"{rule.name}: {rule.metric} is {value:.3f}, exceeding threshold of {rule.threshold:.3f}"
        elif rule.condition == 'less_than':
            return f"{rule.name}: {rule.metric} is {value:.3f}, below threshold of {rule.threshold:.3f}"
        elif rule.condition == 'anomaly':
            return f"{rule.name}: Anomaly detected in {rule.metric} (value: {value:.3f})"
        else:
            return f"{rule.name}: {rule.metric} triggered with value {value:.3f}"
    
    async def _apply_alert_intelligence(self, alert: Alert):
        """Apply intelligent alert grouping and suppression"""
        # Group related alerts
        group_key = f"{alert.severity}_{alert.metric.split('_')[0]}"  # Group by severity and metric prefix
        
        if group_key not in self.alert_groups:
            self.alert_groups[group_key] = []
        
        self.alert_groups[group_key].append(alert.id)
        
        # Auto-suppress if too many similar alerts
        if len(self.alert_groups[group_key]) > 5:
            logger.info(f"Suppressing alert group {group_key} due to high volume")
            self.suppressed_alerts.add(group_key)
    
    async def resolve_alert(self, rule_id: str, metric: str):
        """Resolve an active alert"""
        alert_key = f"{rule_id}_{metric}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_timestamp = datetime.now()
            
            del self.active_alerts[alert_key]
            
            logger.info(f"ALERT RESOLVED: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        recent_alerts = self.get_alert_history(24)
        
        severity_counts = {'info': 0, 'warning': 0, 'critical': 0}
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1
        
        metric_counts = {}
        for alert in recent_alerts:
            metric_counts[alert.metric] = metric_counts.get(alert.metric, 0) + 1
        
        return {
            'active_alerts': len(self.active_alerts),
            'total_alerts_24h': len(recent_alerts),
            'severity_distribution': severity_counts,
            'top_metrics': sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'suppressed_groups': len(self.suppressed_alerts),
            'mean_time_to_resolve': self._calculate_mttr()
        }
    
    def _calculate_mttr(self) -> float:
        """Calculate mean time to resolution"""
        resolved_alerts = [
            alert for alert in self.alert_history
            if alert.resolved and alert.resolved_timestamp
        ]
        
        if not resolved_alerts:
            return 0.0
        
        resolution_times = [
            (alert.resolved_timestamp - alert.timestamp).total_seconds()
            for alert in resolved_alerts
        ]
        
        return sum(resolution_times) / len(resolution_times)

# Pre-configured alert rules for recommendation system
def create_default_alert_rules() -> List[AlertRule]:
    """Create default alert rules for recommendation system"""
    return [
        AlertRule(
            id="ndcg_low",
            name="NDCG@10 Below Target",
            metric="ndcg_at_10",
            condition="less_than",
            threshold=0.35,
            severity="warning",
            cooldown_seconds=300,
            tags=["recommendation_quality", "performance"]
        ),
        AlertRule(
            id="ndcg_critical",
            name="NDCG@10 Critically Low",
            metric="ndcg_at_10",
            condition="less_than",
            threshold=0.30,
            severity="critical",
            cooldown_seconds=180,
            tags=["recommendation_quality", "critical"]
        ),
        AlertRule(
            id="diversity_low",
            name="Diversity Below Target",
            metric="diversity_score",
            condition="less_than",
            threshold=0.7,
            severity="warning",
            cooldown_seconds=300,
            tags=["recommendation_quality", "diversity"]
        ),
        AlertRule(
            id="response_time_high",
            name="High Response Time",
            metric="response_time_ms",
            condition="greater_than",
            threshold=100.0,
            severity="warning",
            cooldown_seconds=120,
            tags=["performance", "latency"]
        ),
        AlertRule(
            id="response_time_critical",
            name="Critical Response Time",
            metric="response_time_ms",
            condition="greater_than",
            threshold=500.0,
            severity="critical",
            cooldown_seconds=60,
            tags=["performance", "critical"]
        ),
        AlertRule(
            id="error_rate_high",
            name="High Error Rate",
            metric="error_rate",
            condition="greater_than",
            threshold=0.01,
            severity="warning",
            cooldown_seconds=300,
            tags=["performance", "errors"]
        ),
        AlertRule(
            id="cpu_usage_high",
            name="High CPU Usage",
            metric="cpu_percent",
            condition="greater_than",
            threshold=80.0,
            severity="warning",
            cooldown_seconds=600,
            tags=["system", "cpu"]
        ),
        AlertRule(
            id="memory_usage_high",
            name="High Memory Usage",
            metric="memory_percent",
            condition="greater_than",
            threshold=85.0,
            severity="warning",
            cooldown_seconds=600,
            tags=["system", "memory"]
        ),
        AlertRule(
            id="ndcg_anomaly",
            name="NDCG Anomaly Detection",
            metric="ndcg_at_10",
            condition="anomaly",
            threshold=0.7,  # Confidence threshold
            severity="info",
            cooldown_seconds=900,
            tags=["recommendation_quality", "anomaly"]
        ),
        AlertRule(
            id="throughput_anomaly",
            name="Throughput Anomaly Detection",
            metric="throughput_rps",
            condition="anomaly",
            threshold=0.8,
            severity="info",
            cooldown_seconds=600,
            tags=["performance", "anomaly"]
        )
    ]
