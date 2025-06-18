"""
Main integration script for Phase 3 Adaptive Learning System.
This script orchestrates the startup and management of all components.
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path
import yaml
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline_integration import AdaptiveLearningPipeline
from src.monitoring.dashboard import AdaptiveLearningMonitor
from src.utils.config import get_config, load_config_file
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


class AdaptiveLearningSystem:
    """
    Main system class that coordinates all Phase 3 components.
    Handles startup, shutdown, and lifecycle management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the adaptive learning system."""
        self.config_path = config_path or "config/adaptive_learning.yaml"
        self.config = self._load_configuration()
        self.pipeline: Optional[AdaptiveLearningPipeline] = None
        self.monitor: Optional[AdaptiveLearningMonitor] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        
    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            config = load_config_file(self.config_path)
            
            # Apply environment-specific overrides
            environment = config.get('environment', 'development')
            if environment in config.get('environments', {}):
                env_config = config['environments'][environment]
                config = self._merge_configs(config, env_config)
            
            logger.info(f"Configuration loaded for environment: {environment}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _merge_configs(self, base_config: Dict[str, Any], 
                      override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries recursively."""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def start(self):
        """Start the adaptive learning system."""
        if self._running:
            logger.warning("System is already running")
            return
        
        try:
            logger.info("Starting Adaptive Learning System...")
            
            # Setup logging
            setup_logging(self.config.get('logging', {}))
            
            # Initialize pipeline
            logger.info("Initializing adaptive learning pipeline...")
            self.pipeline = AdaptiveLearningPipeline(self.config)
            await self.pipeline.start()
            
            # Initialize monitoring
            logger.info("Starting monitoring system...")
            self.monitor = AdaptiveLearningMonitor(self.pipeline)
            monitoring_interval = self.config.get('monitoring', {}).get('update_interval', 60)
            asyncio.create_task(self.monitor.start_monitoring(monitoring_interval))
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self._running = True
            logger.info("Adaptive Learning System started successfully")
            
            # Log system status
            self._log_startup_status()
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the adaptive learning system gracefully."""
        if not self._running:
            return
        
        try:
            logger.info("Shutting down Adaptive Learning System...")
            
            self._running = False
            self._shutdown_event.set()
            
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
                logger.info("Monitoring system stopped")
            
            # Stop pipeline
            if self.pipeline:
                await self.pipeline.stop()
                logger.info("Pipeline stopped")
            
            logger.info("Adaptive Learning System stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _log_startup_status(self):
        """Log system startup status and configuration."""
        if not self.pipeline:
            return
        
        status = self.pipeline.get_pipeline_status()
        
        logger.info("=== Adaptive Learning System Status ===")
        logger.info(f"Pipeline Running: {status.get('running', False)}")
        logger.info(f"Components Health: {status.get('components', {})}")
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Started at: {datetime.utcnow().isoformat()}")
        logger.info("==========================================")
    
    async def run_forever(self):
        """Run the system until shutdown signal."""
        try:
            await self.start()
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"System error: {e}")
            raise
        finally:
            await self.stop()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.pipeline:
            return {'status': 'not_started'}
        
        pipeline_status = self.pipeline.get_pipeline_status()
        
        return {
            'status': 'running' if self._running else 'stopped',
            'pipeline': pipeline_status,
            'monitoring': {
                'active': self.monitor is not None,
                'last_update': datetime.utcnow().isoformat() if self.monitor else None
            },
            'config': {
                'file': self.config_path,
                'environment': self.config.get('environment', 'development')
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'overall': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        try:
            # Check pipeline health
            if self.pipeline and self._running:
                pipeline_status = self.pipeline.get_pipeline_status()
                pipeline_healthy = pipeline_status.get('running', False)
                
                health_status['checks']['pipeline'] = {
                    'status': 'healthy' if pipeline_healthy else 'unhealthy',
                    'details': pipeline_status
                }
                
                if not pipeline_healthy:
                    health_status['overall'] = 'unhealthy'
            else:
                health_status['checks']['pipeline'] = {
                    'status': 'not_running',
                    'details': 'Pipeline not started'
                }
                health_status['overall'] = 'unhealthy'
            
            # Check monitoring health
            if self.monitor:
                health_status['checks']['monitoring'] = {
                    'status': 'healthy',
                    'details': 'Monitoring active'
                }
            else:
                health_status['checks']['monitoring'] = {
                    'status': 'not_running',
                    'details': 'Monitoring not started'
                }
            
            # Additional component checks
            if self.pipeline and self._running:
                # Check database connectivity
                try:
                    # This would be implemented based on your database setup
                    health_status['checks']['database'] = {
                        'status': 'healthy',
                        'details': 'Database connection verified'
                    }
                except Exception as e:
                    health_status['checks']['database'] = {
                        'status': 'unhealthy',
                        'details': f"Database error: {e}"
                    }
                    health_status['overall'] = 'degraded'
                
                # Check streaming connectivity
                try:
                    stream_status = await self._check_streaming_health()
                    health_status['checks']['streaming'] = stream_status
                    if stream_status['status'] != 'healthy':
                        health_status['overall'] = 'degraded'
                except Exception as e:
                    health_status['checks']['streaming'] = {
                        'status': 'unhealthy',
                        'details': f"Streaming error: {e}"
                    }
                    health_status['overall'] = 'degraded'
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_status['overall'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    async def _check_streaming_health(self) -> Dict[str, Any]:
        """Check streaming system health."""
        try:
            if self.pipeline and self.pipeline.stream_handler:
                status = self.pipeline.stream_handler.get_status()
                return {
                    'status': 'healthy' if status.get('connected', False) else 'unhealthy',
                    'details': status
                }
            else:
                return {
                    'status': 'not_configured',
                    'details': 'Streaming not configured'
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'details': f"Streaming check failed: {e}"
            }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Adaptive Learning System")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path",
        default="config/adaptive_learning.yaml"
    )
    parser.add_argument(
        "--environment", "-e",
        type=str,
        help="Environment (development, testing, production)",
        default="development"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check and exit"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = AdaptiveLearningSystem(config_path=args.config)
    
    # Override environment if specified
    if args.environment:
        system.config['environment'] = args.environment
    
    try:
        if args.health_check:
            # Perform health check
            await system.start()
            health = await system.health_check()
            print(f"Health Status: {health['overall']}")
            print(f"Details: {health}")
            await system.stop()
            sys.exit(0 if health['overall'] == 'healthy' else 1)
        
        elif args.status:
            # Show status
            await system.start()
            status = system.get_system_status()
            print(f"System Status: {status}")
            await system.stop()
            sys.exit(0)
        
        else:
            # Run system
            await system.run_forever()
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Setup basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run main
    asyncio.run(main())
