"""
Stream Handler Module for Content Recommendation Engine

This module provides high-level stream handling capabilities for managing
multiple event streams, routing, filtering, and coordination between
different streaming backends (Kafka, Redis, WebSocket).
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Callable, Any, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import redis
from kafka import KafkaProducer, KafkaConsumer
import websockets
from enum import Enum

from .event_processor import EventProcessor, EventType, ProcessingResult
from ..utils.logging import setup_logger
from ..utils.redis_client import get_redis_client, RedisConfig


class StreamType(Enum):
    """Stream type enumeration"""
    KAFKA = "kafka"
    REDIS = "redis"
    WEBSOCKET = "websocket"
    HTTP_SSE = "http_sse"


@dataclass
class StreamConfig:
    """Stream configuration"""
    stream_type: StreamType
    connection_params: Dict[str, Any]
    topics: List[str]
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0
    enable_compression: bool = True
    enable_ssl: bool = False


@dataclass
class StreamStats:
    """Stream statistics"""
    messages_processed: int = 0
    messages_failed: int = 0
    avg_processing_time: float = 0.0
    last_message_time: Optional[datetime] = None
    stream_health: str = "healthy"
    error_rate: float = 0.0


class StreamRouter:
    """Routes events to appropriate handlers based on rules"""
    
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self.default_handler: Optional[Callable] = None
        self.logger = setup_logger(__name__)
    
    def add_rule(self, condition: Callable[[Dict], bool], 
                 handler: Callable[[Dict], Any], priority: int = 0):
        """Add routing rule"""
        self.rules.append({
            'condition': condition,
            'handler': handler,
            'priority': priority
        })
        # Sort by priority (higher first)
        self.rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def set_default_handler(self, handler: Callable[[Dict], Any]):
        """Set default handler for unmatched events"""
        self.default_handler = handler
    
    async def route(self, event: Dict[str, Any]) -> Any:
        """Route event to appropriate handler"""
        try:
            for rule in self.rules:
                if rule['condition'](event):
                    return await self._call_handler(rule['handler'], event)
            
            # Use default handler if no rules match
            if self.default_handler:
                return await self._call_handler(self.default_handler, event)
            
            self.logger.warning(f"No handler found for event: {event.get('type', 'unknown')}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error routing event: {e}")
            return None
    
    async def _call_handler(self, handler: Callable, event: Dict) -> Any:
        """Call handler (sync or async)"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(event)
        else:
            return handler(event)


class StreamFilter:
    """Filters events based on criteria"""
    
    def __init__(self):
        self.filters: List[Callable[[Dict], bool]] = []
        self.logger = setup_logger(__name__)
    
    def add_filter(self, filter_func: Callable[[Dict], bool]):
        """Add filter function"""
        self.filters.append(filter_func)
    
    def should_process(self, event: Dict[str, Any]) -> bool:
        """Check if event should be processed"""
        try:
            for filter_func in self.filters:
                if not filter_func(event):
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error in filter: {e}")
            return False


class StreamHandler:
    """
    High-level stream handler that manages multiple streams and provides
    unified interface for event processing
    """
    
    def __init__(self, config: Dict[str, StreamConfig]):
        self.config = config
        self.streams: Dict[str, Any] = {}
        self.processors: Dict[str, EventProcessor] = {}
        self.stats: Dict[str, StreamStats] = {}
        self.router = StreamRouter()
        self.filter = StreamFilter()
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.logger = setup_logger(__name__)
        
        # Initialize processors for each stream
        for stream_name, stream_config in config.items():
            self.processors[stream_name] = EventProcessor()
            self.stats[stream_name] = StreamStats()
    
    def add_routing_rule(self, condition: Callable[[Dict], bool], 
                        handler: Callable[[Dict], Any], priority: int = 0):
        """Add event routing rule"""
        self.router.add_rule(condition, handler, priority)
    
    def add_filter(self, filter_func: Callable[[Dict], bool]):
        """Add event filter"""
        self.filter.add_filter(filter_func)
    
    async def start(self):
        """Start all stream handlers"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting stream handler")
        
        # Start stream consumers
        for stream_name, stream_config in self.config.items():
            task = asyncio.create_task(
                self._start_stream_consumer(stream_name, stream_config)
            )
            self.tasks.append(task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_streams())
        self.tasks.append(monitor_task)
    
    async def stop(self):
        """Stop all stream handlers"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping stream handler")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close connections
        for stream_name, stream in self.streams.items():
            await self._close_stream(stream_name, stream)
        
        self.executor.shutdown(wait=True)
    
    async def _start_stream_consumer(self, stream_name: str, config: StreamConfig):
        """Start consumer for a specific stream"""
        try:
            if config.stream_type == StreamType.KAFKA:
                await self._start_kafka_consumer(stream_name, config)
            elif config.stream_type == StreamType.REDIS:
                await self._start_redis_consumer(stream_name, config)
            elif config.stream_type == StreamType.WEBSOCKET:
                await self._start_websocket_consumer(stream_name, config)
            else:
                self.logger.error(f"Unsupported stream type: {config.stream_type}")
                
        except Exception as e:
            self.logger.error(f"Error starting stream consumer {stream_name}: {e}")
            self.stats[stream_name].stream_health = "error"
    
    async def _start_kafka_consumer(self, stream_name: str, config: StreamConfig):
        """Start Kafka consumer"""
        consumer = KafkaConsumer(
            *config.topics,
            **config.connection_params,
            auto_offset_reset='latest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=1000
        )
        
        self.streams[stream_name] = consumer
        
        while self.running:
            try:
                message_pack = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        await self._process_message(stream_name, message.value)
                        
            except Exception as e:
                self.logger.error(f"Kafka consumer error: {e}")
                await asyncio.sleep(1)
    
    async def _start_redis_consumer(self, stream_name: str, config: StreamConfig):
        """Start Redis consumer"""
        try:
            redis_conn = get_redis_client()
            if not redis_conn or not redis_conn.is_healthy():
                self.logger.error(f"Redis not available for stream {stream_name}")
                return
            
            redis_client = redis_conn.client
            pubsub = redis_client.pubsub()
            
            for topic in config.topics:
                pubsub.subscribe(topic)
            
            self.streams[stream_name] = pubsub
            
            while self.running:
                try:
                    message = pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        data = json.loads(message['data'].decode('utf-8'))
                        await self._process_message(stream_name, data)
                        
                except Exception as e:
                    self.logger.error(f"Redis consumer error: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Failed to start Redis consumer for {stream_name}: {e}")
            self.stats[stream_name].stream_health = "error"
    
    async def _start_websocket_consumer(self, stream_name: str, config: StreamConfig):
        """Start WebSocket consumer"""
        uri = config.connection_params.get('uri')
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.streams[stream_name] = websocket
                    
                    async for message in websocket:
                        data = json.loads(message)
                        await self._process_message(stream_name, data)
                        
            except Exception as e:
                self.logger.error(f"WebSocket consumer error: {e}")
                await asyncio.sleep(5)  # Retry connection
    
    async def _process_message(self, stream_name: str, message: Dict[str, Any]):
        """Process individual message"""
        start_time = datetime.now()
        
        try:
            # Apply filters
            if not self.filter.should_process(message):
                return
            
            # Process through event processor
            processor = self.processors[stream_name]
            result = await processor.process_event(message)
            
            if result.success:
                # Route to appropriate handler
                await self.router.route(message)
                
                # Update stats
                stats = self.stats[stream_name]
                stats.messages_processed += 1
                stats.last_message_time = datetime.now()
                
                # Update average processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                if stats.avg_processing_time == 0:
                    stats.avg_processing_time = processing_time
                else:
                    stats.avg_processing_time = (
                        stats.avg_processing_time * 0.9 + processing_time * 0.1
                    )
            else:
                stats.messages_failed += 1
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.stats[stream_name].messages_failed += 1
    
    async def _monitor_streams(self):
        """Monitor stream health and performance"""
        while self.running:
            try:
                for stream_name, stats in self.stats.items():
                    # Calculate error rate
                    total_messages = stats.messages_processed + stats.messages_failed
                    if total_messages > 0:
                        stats.error_rate = stats.messages_failed / total_messages
                    
                    # Check stream health
                    if stats.last_message_time:
                        time_since_last = datetime.now() - stats.last_message_time
                        if time_since_last > timedelta(minutes=5):
                            stats.stream_health = "stale"
                        elif stats.error_rate > 0.1:
                            stats.stream_health = "degraded"
                        else:
                            stats.stream_health = "healthy"
                    
                    self.logger.debug(f"Stream {stream_name} stats: {asdict(stats)}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in stream monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _close_stream(self, stream_name: str, stream: Any):
        """Close stream connection"""
        try:
            if hasattr(stream, 'close'):
                if asyncio.iscoroutinefunction(stream.close):
                    await stream.close()
                else:
                    stream.close()
        except Exception as e:
            self.logger.error(f"Error closing stream {stream_name}: {e}")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stream statistics"""
        return {
            stream_name: asdict(stats) 
            for stream_name, stats in self.stats.items()
        }
    
    async def publish_event(self, stream_name: str, event: Dict[str, Any]):
        """Publish event to specific stream"""
        try:
            config = self.config.get(stream_name)
            if not config:
                raise ValueError(f"Stream {stream_name} not configured")
            
            if config.stream_type == StreamType.KAFKA:
                await self._publish_kafka(config, event)
            elif config.stream_type == StreamType.REDIS:
                await self._publish_redis(config, event)
            else:
                self.logger.warning(f"Publishing not supported for {config.stream_type}")
                
        except Exception as e:
            self.logger.error(f"Error publishing event: {e}")
    
    async def _publish_kafka(self, config: StreamConfig, event: Dict[str, Any]):
        """Publish to Kafka"""
        producer = KafkaProducer(
            **config.connection_params,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        # Publish to all configured topics
        for topic in config.topics:
            producer.send(topic, event)
        
        producer.flush()
        producer.close()
    
    async def _publish_redis(self, config: StreamConfig, event: Dict[str, Any]):
        """Publish to Redis"""
        redis_client = redis.Redis(**config.connection_params)
        
        # Publish to all configured channels
        for channel in config.topics:
            redis_client.publish(channel, json.dumps(event))


# Utility functions for creating common stream configurations
def create_kafka_config(bootstrap_servers: List[str], topics: List[str], 
                       group_id: str = None) -> StreamConfig:
    """Create Kafka stream configuration"""
    connection_params = {
        'bootstrap_servers': bootstrap_servers,
        'group_id': group_id or 'default_group'
    }
    
    return StreamConfig(
        stream_type=StreamType.KAFKA,
        connection_params=connection_params,
        topics=topics
    )


def create_redis_config(host: str, port: int, topics: List[str], 
                       password: str = None) -> StreamConfig:
    """Create Redis stream configuration"""
    connection_params = {
        'host': host,
        'port': port,
        'decode_responses': True
    }
    
    if password:
        connection_params['password'] = password
    
    return StreamConfig(
        stream_type=StreamType.REDIS,
        connection_params=connection_params,
        topics=topics
    )


def create_websocket_config(uri: str) -> StreamConfig:
    """Create WebSocket stream configuration"""
    return StreamConfig(
        stream_type=StreamType.WEBSOCKET,
        connection_params={'uri': uri},
        topics=[]
    )
