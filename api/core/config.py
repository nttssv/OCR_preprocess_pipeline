#!/usr/bin/env python3
"""
API Configuration
Central configuration management for the Document Processing Pipeline API
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Document Processing Pipeline API"
    VERSION: str = "1.0.0"
    
    # Security
    SECRET_KEY: str = "document-processing-pipeline-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ]
    
    # File Processing
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]
    
    # Directory Configuration
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "api_uploads")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "processed_documents")  # Dedicated folder for processed files
    TEMP_DIR: str = os.path.join(BASE_DIR, "api_temp")
    CACHE_DIR: str = os.path.join(BASE_DIR, "api_cache")
    
    # Database
    DATABASE_URL: str = "sqlite:///./api_documents.db"
    
    # Processing Configuration
    MAX_WORKERS: int = 4
    TASK_TIMEOUT: int = 300  # 5 minutes
    DEFAULT_DPI: int = 250
    
    # Default transformation pipeline
    DEFAULT_TRANSFORMATIONS: List[str] = [
        "task_1_orientation_correction",
        "task_2_skew_detection", 
        "task_3_cropping"
    ]
    
    # Performance thresholds
    LARGE_FILE_THRESHOLD: int = 50 * 1024 * 1024  # 50MB
    PROCESSING_TIME_THRESHOLD: int = 120  # 2 minutes
    
    # Caching and Performance
    ENABLE_CACHING: bool = True
    CACHE_EXPIRY_HOURS: int = 24
    MAX_CACHE_SIZE: int = 1024 * 1024 * 1024  # 1GB
    
    # Redis-like cache settings
    MEMORY_CACHE_SIZE: int = 1000  # Number of entries in memory cache
    MEMORY_CACHE_TTL: int = 3600   # Default TTL in seconds
    ENABLE_FILE_CACHE: bool = True
    
    # Performance indexes and optimization
    ENABLE_SEARCH_INDEXES: bool = True
    SEARCH_RESULTS_CACHE_TTL: int = 600  # 10 minutes
    PAGE_ACCESS_CACHE_TTL: int = 1800   # 30 minutes
    
    # File management optimization
    THUMBNAIL_SIZE: tuple = (200, 200)
    ENABLE_THUMBNAILS: bool = True
    PREGENERATE_PAGE_IMAGES: bool = True
    
    # Analytics and monitoring
    ENABLE_ACCESS_TRACKING: bool = True
    ACCESS_LOG_RETENTION_DAYS: int = 30
    PERFORMANCE_METRICS_ENABLED: bool = True
    
    # Cleanup
    CLEANUP_TEMP_FILES: bool = True
    CLEANUP_AFTER_HOURS: int = 48
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = os.path.join(BASE_DIR, "logs", "api.log")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


# Transformation pipeline configurations
TRANSFORMATION_CONFIGS = {
    "deskewing": {
        "name": "Deskewing Only",
        "description": "Fast deskewing with orientation correction and basic cropping",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping"],
        "expected_time": "5-15 seconds",
        "quality": "balanced"
    },
    
    "basic": {
        "name": "Basic Processing", 
        "description": "Orientation correction and cropping without skew detection",
        "tasks": ["task_1_orientation_correction", "task_3_cropping"],
        "expected_time": "2-8 seconds",
        "quality": "fast"
    },
    
    "enhanced": {
        "name": "Enhanced Processing",
        "description": "Full preprocessing with noise reduction and contrast enhancement",
        "tasks": [
            "task_1_orientation_correction",
            "task_2_skew_detection", 
            "task_3_cropping",
            "task_4_size_dpi_standardization",
            "task_5_noise_reduction",
            "task_6_contrast_enhancement"
        ],
        "expected_time": "30-90 seconds",
        "quality": "high"
    },
    
    "comprehensive": {
        "name": "Comprehensive Processing",
        "description": "Complete pipeline with all features including color handling and segmentation",
        "tasks": [
            "task_9_document_deduplication",
            "task_1_orientation_correction",
            "task_2_skew_detection",
            "task_3_cropping", 
            "task_4_size_dpi_standardization",
            "task_5_noise_reduction",
            "task_6_contrast_enhancement",
            "task_8_color_handling",
            "task_7_multipage_segmentation",
            "task_10_language_detection",
            "task_11_metadata_extraction",
            "task_12_output_specifications"
        ],
        "expected_time": "2-5 minutes",
        "quality": "maximum"
    }
}


def get_transformation_config(transformation_type: str = "deskewing"):
    """Get configuration for a specific transformation type"""
    return TRANSFORMATION_CONFIGS.get(transformation_type, TRANSFORMATION_CONFIGS["deskewing"])


def get_all_transformation_types():
    """Get all available transformation types"""
    return list(TRANSFORMATION_CONFIGS.keys())


# File size and processing time thresholds for fallback strategies
FALLBACK_STRATEGIES = {
    "small_file": {
        "max_size": 10 * 1024 * 1024,  # 10MB
        "strategy": "enhanced",
        "timeout": 60
    },
    "medium_file": {
        "max_size": 50 * 1024 * 1024,  # 50MB
        "strategy": "deskewing", 
        "timeout": 120
    },
    "large_file": {
        "max_size": 100 * 1024 * 1024,  # 100MB
        "strategy": "basic",
        "timeout": 180
    }
}


def get_fallback_strategy(file_size: int):
    """Get appropriate fallback strategy based on file size"""
    for strategy_name, config in FALLBACK_STRATEGIES.items():
        if file_size <= config["max_size"]:
            return config
    
    # Default for very large files
    return {
        "strategy": "basic",
        "timeout": 300
    }