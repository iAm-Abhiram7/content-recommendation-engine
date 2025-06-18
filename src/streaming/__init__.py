"""
Streaming Infrastructure Module

This module provides streaming infrastructure for real-time adaptive learning:
- Event stream processing with Kafka/Redis
- High-throughput interaction processing
- Event deduplication and ordering
- Lambda architecture for batch-stream hybrid processing
"""

from .event_processor import EventProcessor, EventType, ProcessingResult, EventProcessingConfig
from .stream_handler import StreamHandler, StreamType, StreamConfig, StreamRouter, StreamFilter
from .batch_stream_sync import BatchStreamSynchronizer, SyncStrategy, DataFreshness, ConflictResolver, SyncConfig

__all__ = [
    'EventProcessor',
    'EventType',
    'ProcessingResult',
    'EventProcessingConfig',
    'StreamHandler',
    'StreamType',
    'StreamConfig',
    'StreamRouter', 
    'StreamFilter',
    'BatchStreamSynchronizer',
    'SyncStrategy',
    'DataFreshness',
    'ConflictResolver',
    'SyncConfig'
]
