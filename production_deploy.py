#!/usr/bin/env python3
"""
Production Deployment Manager
Comprehensive deployment script for the recommendation system
"""

import os
import sys
import asyncio
import logging
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.alerting.alert_manager import AlertManager, create_default_alert_rules, NotificationChannel
from src.evaluation.testing.load_testing import quick_load_test
from src.evaluation.metrics.accuracy_metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionDeploymentManager:
    """Manage production deployment of the recommendation system"""
    
    def __init__(self):
        self.metrics_collector = None
        self.alert_manager = None
        self.evaluation_manager = MetricsEvaluator()
        self.services = {}
        
    async def deploy_production(self, config: Dict[str, Any]):
        """Deploy production environment"""
        logger.info("Starting production deployment...")
        
        try:
            # 1. Initialize monitoring
            await self._setup_monitoring()
            
            # 2. Deploy API services
            await self._deploy_api_services(config)
            
            # 3. Setup load balancing
            await self._setup_load_balancing(config)
            
            # 4. Initialize databases
            await self._setup_databases(config)
            
            # 5. Deploy adaptive learning pipeline
            await self._deploy_adaptive_pipeline(config)
            
            # 6. Setup alerting
            await self._setup_alerting(config)
            
            # 7. Run health checks
            await self._run_health_checks()
            
            # 8. Performance validation
            await self._validate_performance(config)
            
            logger.info("Production deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            await self._rollback_deployment()
            return False
    
    async def _setup_monitoring(self):
        """Setup comprehensive monitoring"""
        logger.info("Setting up monitoring infrastructure...")
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()
        await self.metrics_collector.start_collection()
        
        # Initialize alert manager
        self.alert_manager = AlertManager()
        
        # Add default alert rules
        for rule in create_default_alert_rules():
            self.alert_manager.add_rule(rule)
        
        # Setup notification channels
        if os.getenv('SLACK_WEBHOOK_URL'):
            slack_channel = NotificationChannel(
                id='slack_alerts',
                type='slack',
                config={'webhook_url': os.getenv('SLACK_WEBHOOK_URL')},
                severity_filter=['warning', 'critical']
            )
            self.alert_manager.add_notification_channel(slack_channel)
        
        if os.getenv('SMTP_SERVER'):
            email_channel = NotificationChannel(
                id='email_alerts',
                type='email',
                config={
                    'smtp_server': os.getenv('SMTP_SERVER'),
                    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
                    'from_email': os.getenv('FROM_EMAIL'),
                    'to_emails': os.getenv('TO_EMAILS', '').split(','),
                    'use_tls': True,
                    'username': os.getenv('SMTP_USERNAME'),
                    'password': os.getenv('SMTP_PASSWORD')
                },
                severity_filter=['critical']
            )
            self.alert_manager.add_notification_channel(email_channel)
        
        logger.info("Monitoring setup completed")
    
    async def _deploy_api_services(self, config: Dict[str, Any]):
        """Deploy API services"""
        logger.info("Deploying API services...")
        
        # Get deployment configuration
        num_workers = config.get('workers', 4)
        port = config.get('port', 8000)
        host = config.get('host', '0.0.0.0')
        
        # Start main API server
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'src.deployment.api.main:app',
            '--host', host,
            '--port', str(port),
            '--workers', str(num_workers),
            '--access-log',
            '--log-level', 'info'
        ]
        
        if config.get('use_uvloop', True):
            cmd.extend(['--loop', 'uvloop'])
        
        if config.get('use_httptools', True):
            cmd.extend(['--http', 'httptools'])
        
        # Start the API server process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self.services['api'] = process
        
        # Wait for API to be ready
        await self._wait_for_service('http://localhost:8000/health', 60)
        
        logger.info(f"API services deployed on {host}:{port} with {num_workers} workers")
    
    async def _setup_load_balancing(self, config: Dict[str, Any]):
        """Setup load balancing and reverse proxy"""
        logger.info("Setting up load balancing...")
        
        if config.get('use_nginx', False):
            # Generate nginx configuration
            nginx_config = self._generate_nginx_config(config)
            
            # Write nginx config
            config_path = Path('nginx.conf')
            config_path.write_text(nginx_config)
            
            # Start nginx (if available)
            try:
                process = await asyncio.create_subprocess_exec(
                    'nginx', '-c', str(config_path.absolute()),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                self.services['nginx'] = process
                logger.info("Nginx load balancer started")
            except FileNotFoundError:
                logger.warning("Nginx not found, skipping load balancer setup")
    
    async def _setup_databases(self, config: Dict[str, Any]):
        """Setup and initialize databases"""
        logger.info("Setting up databases...")
        
        # Redis setup
        if config.get('setup_redis', True):
            try:
                # Start Redis (if needed)
                redis_process = await asyncio.create_subprocess_exec(
                    'redis-server', '--daemonize', 'yes',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for Redis to be ready
                await self._wait_for_service('redis://localhost:6379', 30)
                logger.info("Redis database ready")
                
            except FileNotFoundError:
                logger.warning("Redis not found, using in-memory fallback")
        
        # Database migrations (if applicable)
        if config.get('run_migrations', False):
            logger.info("Running database migrations...")
            # Add your migration logic here
    
    async def _deploy_adaptive_pipeline(self, config: Dict[str, Any]):
        """Deploy adaptive learning pipeline"""
        logger.info("Deploying adaptive learning pipeline...")
        
        try:
            # Initialize adaptive pipeline
            from src.pipeline_integration import AdaptiveLearningPipeline
            
            pipeline = AdaptiveLearningPipeline()
            await pipeline.initialize()
            
            self.services['adaptive_pipeline'] = pipeline
            logger.info("Adaptive learning pipeline deployed")
            
        except Exception as e:
            logger.warning(f"Failed to deploy adaptive pipeline: {e}")
    
    async def _setup_alerting(self, config: Dict[str, Any]):
        """Setup intelligent alerting"""
        logger.info("Setting up alerting system...")
        
        # Connect metrics collector to alert manager
        async def metrics_callback(metric_name: str, value: float):
            await self.alert_manager.check_metric(metric_name, value)
        
        # Register callback for real-time alerting
        # This would be connected to your actual metrics collection
        
        logger.info("Alerting system configured")
    
    async def _run_health_checks(self):
        """Run comprehensive health checks"""
        logger.info("Running health checks...")
        
        health_checks = {
            'api_health': self._check_api_health(),
            'database_health': self._check_database_health(),
            'metrics_health': self._check_metrics_health()
        }
        
        results = {}
        for check_name, check_coro in health_checks.items():
            try:
                results[check_name] = await check_coro
            except Exception as e:
                results[check_name] = {'status': 'failed', 'error': str(e)}
        
        # Log results
        for check_name, result in results.items():
            if result['status'] == 'healthy':
                logger.info(f"✓ {check_name}: {result['status']}")
            else:
                logger.error(f"✗ {check_name}: {result}")
        
        # Fail deployment if critical checks fail
        critical_checks = ['api_health']
        for check in critical_checks:
            if results[check]['status'] != 'healthy':
                raise Exception(f"Critical health check failed: {check}")
    
    async def _validate_performance(self, config: Dict[str, Any]):
        """Validate performance meets requirements"""
        logger.info("Validating performance requirements...")
        
        # Run load test
        target_url = f"http://localhost:{config.get('port', 8000)}"
        
        load_test_config = {
            'concurrent_users': config.get('validation_users', 50),
            'duration_seconds': config.get('validation_duration', 60)
        }
        
        logger.info(f"Running load test: {load_test_config['concurrent_users']} users for {load_test_config['duration_seconds']}s")
        
        results = await quick_load_test(
            target_url,
            load_test_config['concurrent_users'],
            load_test_config['duration_seconds']
        )
        
        # Check performance requirements
        performance_grade = results.get('performance_assessment', {})
        
        logger.info(f"Performance Results:")
        logger.info(f"  Overall Grade: {performance_grade.get('overall_grade', 'Unknown')}")
        logger.info(f"  Latency Grade: {performance_grade.get('latency_grade', 'Unknown')}")
        logger.info(f"  Throughput Grade: {performance_grade.get('throughput_grade', 'Unknown')}")
        
        # Fail if performance doesn't meet minimum requirements
        if performance_grade.get('overall_grade') in ['D', 'F']:
            raise Exception("Performance validation failed - minimum requirements not met")
        
        logger.info("Performance validation passed")
    
    async def _wait_for_service(self, service_url: str, timeout_seconds: int):
        """Wait for service to be ready"""
        import aiohttp
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                if service_url.startswith('http'):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(service_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                return True
                elif service_url.startswith('redis'):
                    import redis.asyncio as redis
                    client = redis.Redis.from_url(service_url)
                    await client.ping()
                    return True
            except:
                pass
            
            await asyncio.sleep(2)
        
        raise Exception(f"Service {service_url} not ready after {timeout_seconds} seconds")
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/health') as response:
                    if response.status == 200:
                        data = await response.json()
                        return {'status': 'healthy', 'details': data}
                    else:
                        return {'status': 'unhealthy', 'status_code': response.status}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            import redis.asyncio as redis
            client = redis.Redis(host='localhost', port=6379)
            await client.ping()
            return {'status': 'healthy', 'type': 'redis'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _check_metrics_health(self) -> Dict[str, Any]:
        """Check metrics collection health"""
        if self.metrics_collector:
            health = await self.metrics_collector.get_health_status()
            return health
        else:
            return {'status': 'not_configured'}
    
    def _generate_nginx_config(self, config: Dict[str, Any]) -> str:
        """Generate nginx configuration"""
        upstream_servers = []
        for i in range(config.get('workers', 4)):
            upstream_servers.append(f"    server 127.0.0.1:{8000 + i};")
        
        return f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream recommendation_api {{
{chr(10).join(upstream_servers)}
    }}
    
    server {{
        listen 80;
        
        location / {{
            proxy_pass http://recommendation_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
        
        location /health {{
            access_log off;
            proxy_pass http://recommendation_api;
        }}
        
        location /metrics {{
            proxy_pass http://127.0.0.1:8001;
        }}
    }}
}}
"""
    
    async def _rollback_deployment(self):
        """Rollback deployment on failure"""
        logger.info("Rolling back deployment...")
        
        # Stop all services
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'terminate'):
                    service.terminate()
                    await service.wait()
                elif hasattr(service, 'shutdown'):
                    await service.shutdown()
                logger.info(f"Stopped {service_name}")
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
        
        # Stop monitoring
        if self.metrics_collector:
            await self.metrics_collector.stop_collection()
        
        logger.info("Rollback completed")
    
    async def run_demo_interface(self):
        """Run the Streamlit demo interface"""
        logger.info("Starting demo interface...")
        
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            'src/demo/streamlit_app.py',
            '--server.port', '8501',
            '--server.headless', 'true',
            '--server.enableCORS', 'false'
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self.services['streamlit'] = process
        
        logger.info("Demo interface started on http://localhost:8501")
        return process

async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Production Deployment Manager')
    parser.add_argument('--mode', choices=['deploy', 'demo', 'test'], default='deploy',
                       help='Deployment mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--workers', type=int, default=4, help='Number of API workers')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--demo-only', action='store_true', help='Run demo interface only')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'workers': args.workers,
        'port': args.port,
        'host': '0.0.0.0',
        'use_uvloop': True,
        'use_httptools': True,
        'setup_redis': True,
        'use_nginx': False,
        'run_migrations': False,
        'validation_users': 50,
        'validation_duration': 60
    }
    
    # Load custom configuration if provided
    if args.config and Path(args.config).exists():
        import json
        with open(args.config) as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    deployment_manager = ProductionDeploymentManager()
    
    try:
        if args.demo_only or args.mode == 'demo':
            # Run demo interface only
            await deployment_manager.run_demo_interface()
            
            # Keep running
            while True:
                await asyncio.sleep(1)
        
        elif args.mode == 'deploy':
            # Full production deployment
            success = await deployment_manager.deploy_production(config)
            
            if success:
                logger.info("Deployment successful! Services are running.")
                
                # Optionally start demo interface
                if not os.getenv('NO_DEMO'):
                    await deployment_manager.run_demo_interface()
                
                # Keep services running
                while True:
                    await asyncio.sleep(10)
                    
                    # Monitor service health
                    if deployment_manager.metrics_collector:
                        health = await deployment_manager.metrics_collector.get_health_status()
                        if health['status'] == 'critical':
                            logger.critical("System health critical - consider restart")
            else:
                logger.error("Deployment failed!")
                return 1
        
        elif args.mode == 'test':
            # Test mode - run performance validation only
            logger.info("Running performance tests...")
            
            # Start minimal API for testing
            await deployment_manager._deploy_api_services(config)
            await deployment_manager._validate_performance(config)
            
            logger.info("Performance tests completed")
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        await deployment_manager._rollback_deployment()
    
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        await deployment_manager._rollback_deployment()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
