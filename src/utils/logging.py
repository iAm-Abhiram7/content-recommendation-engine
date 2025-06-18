"""
Comprehensive logging configuration for the Content Recommendation Engine
"""
import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger
import json
from datetime import datetime


class LoggingConfig:
    """Centralized logging configuration"""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Setup console logging
        self._setup_console_logging()
        
        # Setup file logging
        self._setup_file_logging()
        
        # Setup structured logging
        self._setup_structured_logging()
    
    def _setup_console_logging(self):
        """Setup console logging with colors"""
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
    
    def _setup_file_logging(self):
        """Setup file logging with rotation"""
        # General application logs
        logger.add(
            self.log_dir / "app.log",
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="100 MB",
            retention="7 days",
            compression="zip"
        )
        
        # Error logs
        logger.add(
            self.log_dir / "errors.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="50 MB",
            retention="30 days",
            compression="zip"
        )
        
        # Data processing logs
        logger.add(
            self.log_dir / "data_processing.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="200 MB",
            retention="14 days",
            compression="zip",
            filter=lambda record: "data_processing" in record["name"]
        )
        
        # API logs
        logger.add(
            self.log_dir / "api.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="100 MB",
            retention="7 days",
            compression="zip",
            filter=lambda record: "api" in record["name"]
        )
    
    def _setup_structured_logging(self):
        """Setup structured JSON logging for monitoring"""
        # Disable structured logging for now to avoid format issues
        pass
    
    def _json_formatter(self, record):
        """Format log records as JSON"""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "module": record["module"],
            "thread": record["thread"].name,
            "process": record["process"].name
        }
        
        # Add extra fields if present
        if record.get("extra"):
            log_entry["extra"] = record["extra"]
        
        return json.dumps(log_entry) + "\n"


class DataProcessingLogger:
    """Specialized logger for data processing operations"""
    
    def __init__(self, component_name: str):
        self.logger = logger.bind(component=component_name)
        self.component_name = component_name
        self.metrics = {}
    
    def log_processing_start(self, operation: str, data_size: int):
        """Log the start of a data processing operation"""
        self.logger.info(
            f"Starting {operation}",
            extra={
                "operation": operation,
                "data_size": data_size,
                "component": self.component_name,
                "stage": "start"
            }
        )
    
    def log_processing_progress(self, operation: str, processed: int, total: int):
        """Log processing progress"""
        progress = (processed / total) * 100 if total > 0 else 0
        self.logger.info(
            f"{operation} progress: {processed}/{total} ({progress:.1f}%)",
            extra={
                "operation": operation,
                "processed": processed,
                "total": total,
                "progress_percent": progress,
                "component": self.component_name,
                "stage": "progress"
            }
        )
    
    def log_processing_complete(self, operation: str, duration: float, success_count: int, error_count: int):
        """Log completion of a data processing operation"""
        self.logger.info(
            f"Completed {operation} in {duration:.2f}s - Success: {success_count}, Errors: {error_count}",
            extra={
                "operation": operation,
                "duration_seconds": duration,
                "success_count": success_count,
                "error_count": error_count,
                "component": self.component_name,
                "stage": "complete"
            }
        )
    
    def log_data_quality_metrics(self, dataset: str, metrics: dict):
        """Log data quality metrics"""
        self.logger.info(
            f"Data quality metrics for {dataset}",
            extra={
                "dataset": dataset,
                "metrics": metrics,
                "component": self.component_name,
                "stage": "quality_check"
            }
        )
    
    def log_error_with_context(self, operation: str, error: Exception, context: dict):
        """Log errors with detailed context"""
        self.logger.error(
            f"Error in {operation}: {str(error)}",
            extra={
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "component": self.component_name,
                "stage": "error"
            }
        )


class APILogger:
    """Specialized logger for API operations"""
    
    def __init__(self):
        self.logger = logger.bind(component="api")
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def log_request(self, method: str, endpoint: str, user_id: Optional[str] = None):
        """Log API request"""
        self.logger.info(
            f"{method} {endpoint}",
            extra={
                "method": method,
                "endpoint": endpoint,
                "user_id": user_id,
                "component": "api",
                "stage": "request"
            }
        )
    
    def log_response(self, method: str, endpoint: str, status_code: int, duration: float):
        """Log API response"""
        self.logger.info(
            f"{method} {endpoint} - {status_code} ({duration:.3f}s)",
            extra={
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_seconds": duration,
                "component": "api",
                "stage": "response"
            }
        )
    
    def log_gemini_api_call(self, operation: str, token_count: int, duration: float, success: bool):
        """Log Gemini API calls"""
        self.logger.info(
            f"Gemini API {operation} - Tokens: {token_count}, Duration: {duration:.2f}s, Success: {success}",
            extra={
                "operation": operation,
                "token_count": token_count,
                "duration_seconds": duration,
                "success": success,
                "component": "gemini_client",
                "stage": "api_call"
            }
        )


class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self):
        self.logger = logger.bind(component="performance")
    
    def log_execution_time(self, function_name: str, duration: float, parameters: dict = None):
        """Log function execution time"""
        self.logger.info(
            f"{function_name} executed in {duration:.3f}s",
            extra={
                "function": function_name,
                "duration_seconds": duration,
                "parameters": parameters or {},
                "component": "performance",
                "stage": "timing"
            }
        )
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage"""
        self.logger.info(
            f"{operation} memory usage: {memory_mb:.2f}MB",
            extra={
                "operation": operation,
                "memory_mb": memory_mb,
                "component": "performance",
                "stage": "memory"
            }
        )
    
    def log_throughput(self, operation: str, items_processed: int, duration: float):
        """Log processing throughput"""
        throughput = items_processed / duration if duration > 0 else 0
        self.logger.info(
            f"{operation} throughput: {throughput:.2f} items/second",
            extra={
                "operation": operation,
                "items_processed": items_processed,
                "duration_seconds": duration,
                "throughput": throughput,
                "component": "performance",
                "stage": "throughput"
            }
        )


# Initialize logging
logging_config = LoggingConfig()

# Export specialized loggers
def get_data_processing_logger(component_name: str) -> DataProcessingLogger:
    """Get a data processing logger for a specific component"""
    return DataProcessingLogger(component_name)

def get_api_logger() -> APILogger:
    """Get an API logger"""
    return APILogger()

def get_performance_logger() -> PerformanceLogger:
    """Get a performance logger"""
    return PerformanceLogger()

def setup_logger(name: str, level: str = "INFO"):
    """Setup a logger with the given name and level"""
    return logger.bind(component=name)

# Export the main logger
__all__ = ["logger", "get_data_processing_logger", "get_api_logger", "get_performance_logger", "setup_logger"]
