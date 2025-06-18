"""
Batch-Stream Synchronization Module for Content Recommendation Engine

This module handles synchronization between batch processing and real-time
streaming systems, ensuring consistency and managing data freshness across
different processing paradigms.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import sqlite3
import redis
from kafka import KafkaProducer, KafkaConsumer
import pandas as pd
import numpy as np
from collections import defaultdict, deque

from ..utils.logging import setup_logger


class SyncStrategy(Enum):
    """Synchronization strategy"""
    EVENTUAL_CONSISTENCY = "eventual"
    STRONG_CONSISTENCY = "strong"
    HYBRID = "hybrid"
    PERIODIC_RECONCILIATION = "periodic"


class DataFreshness(Enum):
    """Data freshness levels"""
    REAL_TIME = "realtime"      # < 1 second
    NEAR_REAL_TIME = "near_realtime"  # < 10 seconds
    FRESH = "fresh"             # < 1 minute
    ACCEPTABLE = "acceptable"   # < 10 minutes
    STALE = "stale"            # > 10 minutes


@dataclass
class SyncConfig:
    """Synchronization configuration"""
    strategy: SyncStrategy
    batch_interval: timedelta
    stream_buffer_size: int = 1000
    reconciliation_interval: timedelta = timedelta(hours=1)
    max_lag_tolerance: timedelta = timedelta(minutes=5)
    enable_conflict_resolution: bool = True
    consistency_level: str = "eventual"


@dataclass
class SyncMetrics:
    """Synchronization metrics"""
    batch_updates: int = 0
    stream_updates: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    avg_lag: float = 0.0
    data_freshness: DataFreshness = DataFreshness.FRESH
    last_sync_time: Optional[datetime] = None
    sync_success_rate: float = 1.0


class ConflictResolver:
    """Resolves conflicts between batch and stream data"""
    
    def __init__(self):
        self.resolution_strategies = {
            'timestamp_wins': self._timestamp_based_resolution,
            'stream_wins': self._stream_wins_resolution,
            'batch_wins': self._batch_wins_resolution,
            'merge': self._merge_resolution,
            'custom': self._custom_resolution
        }
        self.custom_resolvers: Dict[str, Callable] = {}
        self.logger = setup_logger(__name__)
    
    def add_custom_resolver(self, data_type: str, resolver: Callable):
        """Add custom conflict resolver for specific data type"""
        self.custom_resolvers[data_type] = resolver
    
    def resolve_conflict(self, batch_data: Dict[str, Any], 
                        stream_data: Dict[str, Any],
                        strategy: str = 'timestamp_wins') -> Dict[str, Any]:
        """Resolve conflict between batch and stream data"""
        try:
            resolver = self.resolution_strategies.get(strategy, 
                                                    self._timestamp_based_resolution)
            return resolver(batch_data, stream_data)
        except Exception as e:
            self.logger.error(f"Error resolving conflict: {e}")
            return stream_data  # Default to stream data
    
    def _timestamp_based_resolution(self, batch_data: Dict, 
                                   stream_data: Dict) -> Dict:
        """Resolve based on timestamp - most recent wins"""
        batch_ts = batch_data.get('timestamp', 0)
        stream_ts = stream_data.get('timestamp', 0)
        
        return stream_data if stream_ts > batch_ts else batch_data
    
    def _stream_wins_resolution(self, batch_data: Dict, 
                               stream_data: Dict) -> Dict:
        """Stream data always wins"""
        return stream_data
    
    def _batch_wins_resolution(self, batch_data: Dict, 
                              stream_data: Dict) -> Dict:
        """Batch data always wins"""
        return batch_data
    
    def _merge_resolution(self, batch_data: Dict, 
                         stream_data: Dict) -> Dict:
        """Merge both data sources"""
        merged = batch_data.copy()
        merged.update(stream_data)
        merged['_sources'] = ['batch', 'stream']
        return merged
    
    def _custom_resolution(self, batch_data: Dict, 
                          stream_data: Dict) -> Dict:
        """Use custom resolver if available"""
        data_type = batch_data.get('type') or stream_data.get('type')
        resolver = self.custom_resolvers.get(data_type)
        
        if resolver:
            return resolver(batch_data, stream_data)
        else:
            return self._timestamp_based_resolution(batch_data, stream_data)


class DataVersionManager:
    """Manages data versions and consistency"""
    
    def __init__(self, storage_path: str = ":memory:"):
        self.storage_path = storage_path
        self.conn = sqlite3.connect(storage_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.logger = setup_logger(__name__)
        self._setup_tables()
    
    def _setup_tables(self):
        """Setup version tracking tables"""
        with self.lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS data_versions (
                    entity_id TEXT,
                    entity_type TEXT,
                    version INTEGER,
                    data_source TEXT,
                    timestamp DATETIME,
                    data_hash TEXT,
                    data_size INTEGER,
                    PRIMARY KEY (entity_id, entity_type, data_source)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    batch_version INTEGER,
                    stream_version INTEGER,
                    timestamp DATETIME,
                    status TEXT
                )
            """)
            
            self.conn.commit()
    
    def record_version(self, entity_id: str, entity_type: str, 
                      data_source: str, data_hash: str, data_size: int):
        """Record new data version"""
        with self.lock:
            cursor = self.conn.execute("""
                SELECT version FROM data_versions 
                WHERE entity_id = ? AND entity_type = ? AND data_source = ?
            """, (entity_id, entity_type, data_source))
            
            result = cursor.fetchone()
            new_version = (result[0] + 1) if result else 1
            
            self.conn.execute("""
                INSERT OR REPLACE INTO data_versions
                (entity_id, entity_type, version, data_source, timestamp, data_hash, data_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (entity_id, entity_type, new_version, data_source, 
                  datetime.now(), data_hash, data_size))
            
            self.conn.commit()
            return new_version
    
    def get_version_info(self, entity_id: str, entity_type: str) -> Dict[str, Any]:
        """Get version information for entity"""
        with self.lock:
            cursor = self.conn.execute("""
                SELECT data_source, version, timestamp, data_hash, data_size
                FROM data_versions
                WHERE entity_id = ? AND entity_type = ?
            """, (entity_id, entity_type))
            
            versions = {}
            for row in cursor.fetchall():
                source, version, timestamp, data_hash, size = row
                versions[source] = {
                    'version': version,
                    'timestamp': datetime.fromisoformat(timestamp),
                    'hash': data_hash,
                    'size': size
                }
            
            return versions
    
    def create_checkpoint(self, checkpoint_id: str, batch_version: int, 
                         stream_version: int) -> bool:
        """Create synchronization checkpoint"""
        try:
            with self.lock:
                self.conn.execute("""
                    INSERT INTO sync_checkpoints
                    (checkpoint_id, batch_version, stream_version, timestamp, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (checkpoint_id, batch_version, stream_version, 
                      datetime.now(), 'created'))
                
                self.conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error creating checkpoint: {e}")
            return False


class BatchStreamSynchronizer:
    """
    Synchronizes batch and stream processing systems to ensure data consistency
    and manage freshness across different processing paradigms
    """
    
    def __init__(self, config: SyncConfig, redis_config: Dict[str, Any] = None):
        self.config = config
        self.metrics = SyncMetrics()
        self.conflict_resolver = ConflictResolver()
        self.version_manager = DataVersionManager()
        
        # Redis for coordination
        self.redis_client = None
        if redis_config:
            self.redis_client = redis.Redis(**redis_config)
        
        # Data buffers
        self.stream_buffer = deque(maxlen=config.stream_buffer_size)
        self.batch_buffer = deque(maxlen=1000)
        
        # State tracking
        self.running = False
        self.last_batch_time = datetime.now()
        self.last_reconciliation = datetime.now()
        self.data_freshness_cache: Dict[str, Tuple[datetime, Any]] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.sync_lock = threading.Lock()
        
        self.logger = setup_logger(__name__)
    
    async def start(self):
        """Start synchronization services"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting batch-stream synchronizer")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._batch_sync_loop()),
            asyncio.create_task(self._reconciliation_loop()),
            asyncio.create_task(self._freshness_monitor()),
            asyncio.create_task(self._metrics_updater())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop synchronization services"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Stopped batch-stream synchronizer")
    
    def add_stream_update(self, entity_id: str, entity_type: str, 
                         data: Dict[str, Any], timestamp: datetime = None):
        """Add stream update to buffer"""
        update = {
            'entity_id': entity_id,
            'entity_type': entity_type,
            'data': data,
            'timestamp': timestamp or datetime.now(),
            'source': 'stream'
        }
        
        self.stream_buffer.append(update)
        self.metrics.stream_updates += 1
        
        # Update freshness cache
        cache_key = f"{entity_type}:{entity_id}"
        self.data_freshness_cache[cache_key] = (update['timestamp'], data)
    
    def add_batch_update(self, entity_id: str, entity_type: str, 
                        data: Dict[str, Any], timestamp: datetime = None):
        """Add batch update to buffer"""
        update = {
            'entity_id': entity_id,
            'entity_type': entity_type,
            'data': data,
            'timestamp': timestamp or datetime.now(),
            'source': 'batch'
        }
        
        self.batch_buffer.append(update)
        self.metrics.batch_updates += 1
    
    async def get_latest_data(self, entity_id: str, entity_type: str, 
                             max_age: timedelta = None) -> Optional[Dict[str, Any]]:
        """Get latest data for entity with freshness consideration"""
        cache_key = f"{entity_type}:{entity_id}"
        
        # Check cache first
        if cache_key in self.data_freshness_cache:
            timestamp, data = self.data_freshness_cache[cache_key]
            
            if max_age is None or (datetime.now() - timestamp) <= max_age:
                return {
                    'data': data,
                    'timestamp': timestamp,
                    'freshness': self._calculate_freshness(timestamp),
                    'source': 'cache'
                }
        
        # Look in buffers
        latest_stream = self._find_latest_in_buffer(
            self.stream_buffer, entity_id, entity_type
        )
        latest_batch = self._find_latest_in_buffer(
            self.batch_buffer, entity_id, entity_type
        )
        
        # Resolve conflicts if both exist
        if latest_stream and latest_batch:
            resolved = self.conflict_resolver.resolve_conflict(
                latest_batch['data'], latest_stream['data']
            )
            timestamp = max(latest_stream['timestamp'], latest_batch['timestamp'])
            
            return {
                'data': resolved,
                'timestamp': timestamp,
                'freshness': self._calculate_freshness(timestamp),
                'source': 'resolved'
            }
        
        # Return whichever exists
        latest = latest_stream or latest_batch
        if latest:
            return {
                'data': latest['data'],
                'timestamp': latest['timestamp'],
                'freshness': self._calculate_freshness(latest['timestamp']),
                'source': latest['source']
            }
        
        return None
    
    def _find_latest_in_buffer(self, buffer: deque, entity_id: str, 
                              entity_type: str) -> Optional[Dict]:
        """Find latest update for entity in buffer"""
        latest = None
        
        for update in reversed(buffer):
            if (update['entity_id'] == entity_id and 
                update['entity_type'] == entity_type):
                if latest is None or update['timestamp'] > latest['timestamp']:
                    latest = update
        
        return latest
    
    def _calculate_freshness(self, timestamp: datetime) -> DataFreshness:
        """Calculate data freshness level"""
        age = datetime.now() - timestamp
        
        if age < timedelta(seconds=1):
            return DataFreshness.REAL_TIME
        elif age < timedelta(seconds=10):
            return DataFreshness.NEAR_REAL_TIME
        elif age < timedelta(minutes=1):
            return DataFreshness.FRESH
        elif age < timedelta(minutes=10):
            return DataFreshness.ACCEPTABLE
        else:
            return DataFreshness.STALE
    
    async def _batch_sync_loop(self):
        """Main batch synchronization loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.batch_interval.total_seconds())
                
                if self.config.strategy in [SyncStrategy.STRONG_CONSISTENCY, 
                                          SyncStrategy.HYBRID]:
                    await self._perform_batch_sync()
                
                self.last_batch_time = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error in batch sync loop: {e}")
                await asyncio.sleep(60)
    
    async def _reconciliation_loop(self):
        """Periodic reconciliation loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.reconciliation_interval.total_seconds())
                
                if self.config.strategy in [SyncStrategy.PERIODIC_RECONCILIATION, 
                                          SyncStrategy.HYBRID]:
                    await self._perform_reconciliation()
                
                self.last_reconciliation = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(300)
    
    async def _freshness_monitor(self):
        """Monitor data freshness"""
        while self.running:
            try:
                # Clean up stale cache entries
                current_time = datetime.now()
                stale_keys = []
                
                for key, (timestamp, _) in self.data_freshness_cache.items():
                    age = current_time - timestamp
                    if age > timedelta(hours=1):  # Remove entries older than 1 hour
                        stale_keys.append(key)
                
                for key in stale_keys:
                    del self.data_freshness_cache[key]
                
                # Update overall freshness metric
                if self.data_freshness_cache:
                    timestamps = [ts for ts, _ in self.data_freshness_cache.values()]
                    avg_age = sum((current_time - ts).total_seconds() 
                                for ts in timestamps) / len(timestamps)
                    
                    if avg_age < 1:
                        self.metrics.data_freshness = DataFreshness.REAL_TIME
                    elif avg_age < 10:
                        self.metrics.data_freshness = DataFreshness.NEAR_REAL_TIME
                    elif avg_age < 60:
                        self.metrics.data_freshness = DataFreshness.FRESH
                    elif avg_age < 600:
                        self.metrics.data_freshness = DataFreshness.ACCEPTABLE
                    else:
                        self.metrics.data_freshness = DataFreshness.STALE
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in freshness monitor: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_updater(self):
        """Update synchronization metrics"""
        while self.running:
            try:
                # Calculate lag
                if self.last_batch_time:
                    self.metrics.avg_lag = (
                        datetime.now() - self.last_batch_time
                    ).total_seconds()
                
                # Calculate success rate
                total_operations = (self.metrics.batch_updates + 
                                  self.metrics.stream_updates)
                if total_operations > 0:
                    failures = self.metrics.conflicts_detected - self.metrics.conflicts_resolved
                    self.metrics.sync_success_rate = max(0, 1 - failures / total_operations)
                
                self.metrics.last_sync_time = datetime.now()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    async def _perform_batch_sync(self):
        """Perform batch synchronization"""
        with self.sync_lock:
            # Process conflicts between batch and stream buffers
            conflicts_found = 0
            conflicts_resolved = 0
            
            # Group updates by entity
            entity_updates = defaultdict(list)
            
            # Collect all updates
            for update in list(self.batch_buffer) + list(self.stream_buffer):
                key = f"{update['entity_type']}:{update['entity_id']}"
                entity_updates[key].append(update)
            
            # Resolve conflicts for each entity
            for entity_key, updates in entity_updates.items():
                if len(updates) > 1:
                    # Sort by timestamp
                    updates.sort(key=lambda x: x['timestamp'])
                    
                    # Check for conflicts (different sources with close timestamps)
                    for i in range(len(updates) - 1):
                        curr = updates[i]
                        next_update = updates[i + 1]
                        
                        if (curr['source'] != next_update['source'] and
                            abs((next_update['timestamp'] - curr['timestamp']).total_seconds()) < 60):
                            
                            conflicts_found += 1
                            
                            # Resolve conflict
                            resolved_data = self.conflict_resolver.resolve_conflict(
                                curr['data'], next_update['data']
                            )
                            
                            # Update cache with resolved data
                            self.data_freshness_cache[entity_key.replace(':', ':', 1)] = (
                                next_update['timestamp'], resolved_data
                            )
                            
                            conflicts_resolved += 1
            
            self.metrics.conflicts_detected += conflicts_found
            self.metrics.conflicts_resolved += conflicts_resolved
            
            self.logger.info(f"Batch sync completed: {conflicts_found} conflicts found, "
                           f"{conflicts_resolved} resolved")
    
    async def _perform_reconciliation(self):
        """Perform full data reconciliation"""
        self.logger.info("Starting full reconciliation")
        
        try:
            # Create checkpoint
            checkpoint_id = f"reconciliation_{int(time.time())}"
            
            # This would typically involve:
            # 1. Comparing batch and stream data stores
            # 2. Identifying discrepancies
            # 3. Resolving conflicts
            # 4. Updating authoritative data store
            
            # For now, we'll just clear old buffers and cache
            current_time = datetime.now()
            
            # Clear old buffer entries
            cutoff_time = current_time - self.config.max_lag_tolerance
            
            self.stream_buffer = deque(
                [update for update in self.stream_buffer 
                 if update['timestamp'] > cutoff_time],
                maxlen=self.config.stream_buffer_size
            )
            
            self.batch_buffer = deque(
                [update for update in self.batch_buffer 
                 if update['timestamp'] > cutoff_time],
                maxlen=1000
            )
            
            self.logger.info("Reconciliation completed")
            
        except Exception as e:
            self.logger.error(f"Error during reconciliation: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get synchronization metrics"""
        return asdict(self.metrics)
    
    def get_freshness_report(self) -> Dict[str, Any]:
        """Get data freshness report"""
        current_time = datetime.now()
        freshness_distribution = defaultdict(int)
        
        for timestamp, _ in self.data_freshness_cache.values():
            freshness = self._calculate_freshness(timestamp)
            freshness_distribution[freshness.value] += 1
        
        return {
            'overall_freshness': self.metrics.data_freshness.value,
            'cached_entities': len(self.data_freshness_cache),
            'freshness_distribution': dict(freshness_distribution),
            'last_update': current_time.isoformat()
        }
