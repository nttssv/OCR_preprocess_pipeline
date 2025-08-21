#!/usr/bin/env python3
"""
Pipeline Configuration File
Configure settings for the end-to-end document processing pipeline
"""

import os

# Pipeline Configuration
PIPELINE_CONFIG = {
    # General Settings
    "pipeline_name": "Document Processing Pipeline v1.0",
    "version": "1.0.0",
    "description": "End-to-End Document Processing: Skew → Crop → Orient",
    
    # Folder Configuration
    "folders": {
        "input": "input",
        "output": "output", 
        "temp": "temp",
        "logs": "logs"
    },
    
    # Task Configuration
    "tasks": {
        "task_1_orientation_correction": {
            "name": "Orientation Correction",
            "enabled": True,
            "description": "Detect and correct upside-down or sideways pages",
            "order": 1,
            "dependencies": [],
            "output_format": "png",
            "settings": {
                "detection_methods": ["text_structure", "character_patterns", "reading_direction"],
                "rotation_angles": [0, 90, 180, 270],
                "confidence_threshold": 0.6
            }
        },
        
        "task_2_skew_detection": {
            "name": "Skew Detection & Correction",
            "enabled": True,
            "description": "Detect and correct document skew angles",
            "order": 2,
            "dependencies": ["task_1_orientation_correction"],
            "output_format": "png",
            "settings": {
                "max_angle": 15,  # Maximum skew angle to detect
                "angle_precision": 0.1,  # Angle precision in degrees
                "method": "hough_lines"  # Detection method
            }
        },
        
        "task_3_cropping": {
            "name": "Document Cropping",
            "enabled": True,
            "description": "Remove blank borders, punch holes, and scanner edges",
            "order": 3,
            "dependencies": ["task_2_skew_detection"],
            "output_format": "png",
            "settings": {
                "margin": 10,  # Margin to keep around text
                "remove_holes": True,  # Remove punch holes
                "remove_edges": True,  # Remove scanner edges
                "method": "extreme_tight"  # Cropping method
            }
        },
        
        "task_4_size_dpi_standardization": {
            "name": "Size & DPI Standardization",
            "enabled": True,
            "description": "Standardize image dimensions and improve DPI to 250 for optimal OCR processing",
            "order": 4,
            "dependencies": ["task_3_cropping"],
            "output_format": "png",
            "settings": {
                "target_dpi": 250,
                "standard_width": 2079,  # A4 width at 250 DPI
                "standard_height": 2923,  # A4 height at 250 DPI
                "enhancement_methods": ["clahe", "denoising", "sharpening", "thresholding"],
                "maintain_aspect_ratio": True
            }
        },
        
        "task_5_noise_reduction": {
            "name": "Noise Reduction & Denoising",
            "enabled": True,
            "description": "Remove noise, artifacts, and bleed-through from document images",
            "order": 5,
            "dependencies": ["task_4_size_dpi_standardization"],
            "output_format": "png",
            "settings": {
                "median_filter_size": 3,
                "nlm_h": 10,
                "nlm_template_window": 7,
                "nlm_search_window": 21,
                "background_threshold": 0.8,
                "bleedthrough_alpha": 0.3,
                "enable_median_filter": True,
                "enable_nlm_denoising": True,
                "enable_background_removal": True,
                "enable_bleedthrough_removal": True,
                "enable_bilateral_filter": True
            }
        },
        
        "task_6_contrast_enhancement": {
            "name": "Contrast & Brightness Enhancement",
            "enabled": True,
            "description": "Enhance contrast, brightness, and text clarity using CLAHE, gamma correction, and adaptive techniques",
            "order": 6,
            "dependencies": ["task_5_noise_reduction"],
            "output_format": "png",
            "settings": {
                "enable_clahe": True,
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": [8, 8],
                "enable_histogram_equalization": True,
                "enable_gamma_correction": True,
                "gamma_value": 1.2,
                "enable_contrast_stretching": True,
                "contrast_percentile_low": 2,
                "contrast_percentile_high": 98,
                "enable_adaptive_enhancement": True,
                "enable_sharpening": True,
                "sharpening_strength": 1.5,
                "enable_brightness_adjustment": True,
                "target_brightness": 180,
                "brightness_tolerance": 30,
                "enhancement_mode": "adaptive"
            }
        }
    },
    
    # File Processing Configuration
    "file_processing": {
        "supported_formats": {
            "images": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
            "documents": [".pdf"],
            "archives": [".zip", ".rar", ".7z"]
        },
        "max_file_size": 100 * 1024 * 1024,  # 100MB
        "batch_size": 10,  # Process 10 files at a time
        "parallel_processing": False  # Enable for faster processing
    },
    
    # Quality Settings
    "quality": {
        "output_dpi": 300,
        "compression_quality": 95,
        "preserve_metadata": True,
        "backup_originals": True
    },
    
    # Performance Settings
    "performance": {
        "memory_limit": 2 * 1024 * 1024 * 1024,  # 2GB
        "timeout_per_task": 300,  # 5 minutes per task
        "max_retries": 3,
        "cleanup_temp_files": True
    },
    
    # Logging Configuration
    "logging": {
        "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_rotation": True,
        "max_log_files": 10,
        "log_file_size": 10 * 1024 * 1024  # 10MB
    },
    
    # Error Handling
    "error_handling": {
        "continue_on_error": True,  # Continue processing other files if one fails
        "save_error_logs": True,
        "notify_on_failure": False,
        "retry_failed_tasks": True
    }
}

# Task Extension Template (for future tasks)
TASK_TEMPLATE = {
    "task_X_new_task": {
        "name": "New Task Name",
        "enabled": True,
        "description": "Description of what this task does",
        "order": 4,  # Order in pipeline
        "dependencies": ["task_3_orientation_correction"],  # What it depends on
        "output_format": "png",  # Output format
        "settings": {
            "param1": "value1",
            "param2": "value2"
        }
    }
}

# Pipeline Execution Modes
EXECUTION_MODES = {
    "full_pipeline": {
        "name": "Full Pipeline",
        "description": "Run all enabled tasks in sequence",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction", "task_6_contrast_enhancement"]
    },
    
    "orient_only": {
        "name": "Orientation Only",
        "description": "Only run orientation correction",
        "tasks": ["task_1_orientation_correction"]
    },
    
    "skew_only": {
        "name": "Skew Detection Only",
        "description": "Only run skew detection and correction",
        "tasks": ["task_2_skew_detection"]
    },
    
    "crop_only": {
        "name": "Cropping Only", 
        "description": "Only run document cropping",
        "tasks": ["task_3_cropping"]
    },
    
    "orient_and_skew": {
        "name": "Orient + Skew",
        "description": "Run orientation correction and skew detection",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection"]
    },
    
    "orient_skew_crop": {
        "name": "Orient + Skew + Crop",
        "description": "Run orientation, skew and cropping",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping"]
    },
    
    "with_dpi_standardization": {
        "name": "Full Pipeline + DPI Standardization",
        "description": "Run all tasks including size and DPI standardization",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization"]
    },
    
    "noise_only": {
        "name": "Noise Reduction Only",
        "description": "Only run noise reduction and denoising",
        "tasks": ["task_5_noise_reduction"]
    },
    
    "with_denoising": {
        "name": "With Denoising",
        "description": "Run all tasks including noise reduction",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction"]
    },
    
    "contrast_only": {
        "name": "Contrast Enhancement Only",
        "description": "Only run contrast and brightness enhancement",
        "tasks": ["task_6_contrast_enhancement"]
    },
    
    "with_enhancement": {
        "name": "With Enhancement",
        "description": "Run all tasks including contrast enhancement",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction", "task_6_contrast_enhancement"]
    }
}

# Output Organization
OUTPUT_ORGANIZATION = {
    "structure": "by_file",  # "by_file" or "by_task"
    "naming_convention": "task_name_filename",
    "include_original": True,
    "include_intermediate": False,  # Include intermediate results
    "create_summary": True,
    "summary_format": "html"  # "html", "json", "csv"
}

# Validation Rules
VALIDATION_RULES = {
    "input_validation": {
        "check_file_exists": True,
        "check_file_size": True,
        "check_file_format": True,
        "validate_image_dimensions": True
    },
    
    "output_validation": {
        "check_output_quality": True,
        "compare_file_sizes": True,
        "validate_output_format": True
    }
}

def get_pipeline_config():
    """Get the main pipeline configuration"""
    return PIPELINE_CONFIG

def get_execution_mode(mode_name):
    """Get configuration for a specific execution mode"""
    return EXECUTION_MODES.get(mode_name, EXECUTION_MODES["full_pipeline"])

def get_task_config(task_name):
    """Get configuration for a specific task"""
    return PIPELINE_CONFIG["tasks"].get(task_name)

def is_task_enabled(task_name):
    """Check if a specific task is enabled"""
    task_config = get_task_config(task_name)
    return task_config and task_config.get("enabled", False)

def get_task_dependencies(task_name):
    """Get dependencies for a specific task"""
    task_config = get_task_config(task_name)
    return task_config.get("dependencies", []) if task_config else []

def get_supported_formats():
    """Get all supported file formats"""
    return PIPELINE_CONFIG["file_processing"]["supported_formats"]

def get_output_folder():
    """Get the output folder path"""
    return PIPELINE_CONFIG["folders"]["output"]

def get_temp_folder():
    """Get the temporary folder path"""
    return PIPELINE_CONFIG["folders"]["temp"]

def get_logging_config():
    """Get logging configuration"""
    return PIPELINE_CONFIG["logging"]

def get_quality_settings():
    """Get quality settings"""
    return PIPELINE_CONFIG["quality"]

def get_performance_settings():
    """Get performance settings"""
    return PIPELINE_CONFIG["performance"]

def get_error_handling_config():
    """Get error handling configuration"""
    return PIPELINE_CONFIG["error_handling"]

def get_output_organization():
    """Get output organization settings"""
    return PIPELINE_CONFIG.get("output_organization", OUTPUT_ORGANIZATION)

def get_validation_rules():
    """Get validation rules"""
    return PIPELINE_CONFIG.get("validation_rules", VALIDATION_RULES)

# Example usage
if __name__ == "__main__":
    print("🔧 Pipeline Configuration")
    print("=" * 40)
    
    config = get_pipeline_config()
    print(f"Pipeline: {config['pipeline_name']}")
    print(f"Version: {config['version']}")
    print(f"Description: {config['description']}")
    
    print("\n📋 Available Tasks:")
    for task_id, task_config in config['tasks'].items():
        status = "✅ Enabled" if task_config['enabled'] else "❌ Disabled"
        print(f"   {task_config['order']}. {task_config['name']} - {status}")
    
    print("\n🚀 Execution Modes:")
    for mode_id, mode_config in EXECUTION_MODES.items():
        print(f"   {mode_id}: {mode_config['name']}")
    
    print("\n📁 Output Organization:")
    output_config = get_output_organization()
    print(f"   Structure: {output_config['structure']}")
    print(f"   Include Original: {output_config['include_original']}")
    print(f"   Create Summary: {output_config['create_summary']}")
