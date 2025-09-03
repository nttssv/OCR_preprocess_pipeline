#!/usr/bin/env python3
"""
Logging Configuration
Centralized logging setup for the API
"""

import os
import logging
import logging.handlers
from datetime import datetime

from .config import settings


def setup_logging():
    """Set up logging configuration"""
    
    # Create logs directory
    log_dir = os.path.dirname(settings.LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                settings.LOG_FILE,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Logging initialized - Level: {settings.LOG_LEVEL}")
    logger.info(f"📝 Log file: {settings.LOG_FILE}")


def get_logger(name: str):
    """Get a logger with the specified name"""
    return logging.getLogger(name)