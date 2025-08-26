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
        
        "task_3_cropping_standalone": {
            "name": "Document Cropping (Standalone)",
            "enabled": True,
            "description": "Fast cropping without dependencies for production mode",
            "order": 3,
            "dependencies": [],  # No dependencies for standalone mode
            "output_format": "png",
            "settings": {
                "margin": 5,  # Smaller margin for speed
                "remove_holes": False,  # Skip for speed
                "remove_edges": False,  # Skip for speed
                "method": "basic"  # Fastest method
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
        },
        
        "task_7_multipage_segmentation": {
            "name": "Multi-page & Region Segmentation",
            "enabled": True,  # Enabled for full pipeline
            "description": "Multi-page detection, page splitting, and ROI isolation",
            "order": 7,
            "dependencies": ["task_8_color_handling"],  # FIXED: Run after color handling
            "output_format": "png",
            "settings": {
                # Multi-page detection
                "min_gap_width_ratio": 0.05,
                "min_gap_height_ratio": 0.3,
                "projection_threshold": 0.95,
                
                # Page splitting
                "enable_page_splitting": True,
                "split_margin": 20,
                
                # Region of interest
                "enable_roi_isolation": True,
                "color_preservation_threshold": 30,
                "stamp_signature_detection": True,
                "text_to_grayscale": True,
                
                # Manual verification
                "enable_manual_verification": False,
                "confidence_threshold": 0.8
            }
        },
        
        "task_8_color_handling": {
            "name": "Color Handling",
            "enabled": True,  # Enabled for full pipeline with color management
            "description": "Intelligent color management with stamp/signature preservation and dual output",
            "order": 8,
            "dependencies": ["task_6_contrast_enhancement"],  # FIXED: Run after contrast enhancement (before segmentation)
            "output_format": "png",
            "settings": {
                # Color Detection  
                "enable_color_detection": True,
                "stamp_color_threshold": 15,          # Reduced for stamps with white text inside
                "signature_color_threshold": 20,      # Increased for more reliable signature detection
                "color_saturation_threshold": 50,     # Better color detection
                "color_value_threshold": 50,
                
                # Color Preservation
                "preserve_stamp_colors": True,
                "preserve_signature_colors": True,
                "color_region_expansion": 10,
                "minimum_region_size": 500,
                
                # Grayscale Conversion
                "grayscale_method": "adaptive",
                "preserve_text_contrast": True,
                "contrast_enhancement_factor": 1.2,
                "gamma_correction": 1.1,
                
                # Dual Output
                "enable_dual_output": True,
                "color_suffix": "_color_archive",
                "grayscale_suffix": "_grayscale_ocr",
                "color_quality": 95,
                "grayscale_compression": True,
                
                # Output Options
                "create_comparison": True,
                "highlight_preserved_regions": True,
                "save_region_masks": False
            }
        },
        
        "task_9_document_deduplication": {
            "name": "Document Deduplication",
            "enabled": True,
            "description": "Perceptual hashing for detecting repeated attachments and avoiding redundant OCR",
            "order": 9,
            "dependencies": [],  # Can run independently at the start
            "output_format": "json",  # Primarily generates reports
            "settings": {
                # Perceptual Hashing Settings
                "phash_size": 16,                    # Hash dimension (16x16 = 256-bit hash)
                "phash_threshold": 10,               # Hamming distance threshold for duplicates
                "enable_phash": True,                # Enable perceptual hashing
                "enable_content_hash": True,         # Enable content-based hashing
                "enable_size_check": True,           # Enable size-based pre-filtering
                
                # Duplicate Detection Settings
                "similarity_threshold": 85,          # Percentage similarity for duplicates
                "size_tolerance": 0.1,               # 10% size difference tolerance
                "enable_rotation_invariant": True,   # Detect rotated duplicates
                "enable_scale_invariant": True,      # Detect scaled duplicates
                
                # Processing Options
                "skip_duplicates": True,             # Skip processing duplicates
                "index_duplicates": True,            # Index duplicates in database
                "generate_fingerprints": True,       # Generate audit fingerprints
                "create_comparison": True,           # Create comparison images
                
                # Database Settings
                "db_path": "deduplication/dedup_index.db",
                "enable_audit_trail": True,         # Maintain audit trail
                "max_similar_files": 10,            # Max similar files to track per hash
                
                # Output Settings
                "save_duplicate_report": True,      # Save duplicate detection report
                "highlight_duplicates": True,       # Highlight duplicates in comparisons
                "dedup_suffix": "_dedup_report"     # Suffix for deduplication reports
            }
        },
        
        "task_10_language_detection": {
            "name": "Language & Script Detection",
            "enabled": True,
            "description": "Detect dominant script and language for multilingual OCR path optimization",
            "order": 10,
            "dependencies": ["task_9_document_deduplication"],
            "output_format": "json",
            "settings": {
                # Script Detection Settings
                "enable_script_detection": True,
                "supported_scripts": ["latin", "vietnamese", "numeric", "mrz"],
                "confidence_threshold": 0.7,
                
                # Language Detection Settings
                "enable_language_detection": True,
                "supported_languages": ["eng", "vie", "fra", "spa", "deu"],
                "tesseract_config": "--oem 3 --psm 6",
                
                # Vietnamese Specific
                "detect_vietnamese_diacritics": True,
                "mrz_detection": True,
                "handwriting_detection": True,
                
                # Output Settings
                "save_language_report": True,
                "create_script_overlay": True,
                "print_detected_language": True
            }
        },
        
        "task_11_metadata_extraction": {
            "name": "Metadata Extraction (Pre-OCR)",
            "enabled": True,
            "description": "Extract comprehensive metadata for orchestration layer",
            "order": 11,
            "dependencies": ["task_10_language_detection"],
            "output_format": "json",
            "settings": {
                # Analysis Settings
                "analyze_resolution": True,
                "analyze_color_depth": True,
                "analyze_text_density": True,
                "analyze_image_density": True,
                "analyze_graphics_presence": True,
                "analyze_table_presence": True,
                
                # Quality Analysis
                "analyze_image_quality": True,
                "detect_blur": True,
                "detect_noise": True,
                "detect_low_contrast": True,
                
                # Document Features
                "detect_document_type": True,
                "detect_form_fields": True,
                "detect_signatures": True,
                "detect_stamps": True,
                
                # Output Settings
                "save_sidecar_json": True,
                "include_thumbnail": True,
                "thumbnail_size": [200, 200]
            }
        },
        
        "task_12_output_specifications": {
            "name": "Output Specifications",
            "enabled": True,
            "description": "Generate standardized outputs with comprehensive metadata and audit logs",
            "order": 12,
            "dependencies": ["task_11_metadata_extraction"],
            "output_format": "multiple",
            "settings": {
                # Output Format Settings
                "standardized_formats": ["tiff", "png"],
                "primary_format": "tiff",
                "generate_pdf": True,
                "pdf_dpi": 300,
                
                # Quality Settings
                "tiff_compression": "lzw",
                "png_compression": 6,
                "jpeg_quality": 95,
                
                # Metadata Standards
                "generate_comprehensive_metadata": True,
                "include_processing_chain": True,
                "include_qc_flags": True,
                "include_audit_trail": True,
                
                # Quality Control
                "detect_multi_page": True,
                "flag_low_contrast": True,
                "flag_blur": True,
                "flag_orientation_issues": True,
                
                # Output Organization
                "create_document_folder": True,
                "include_thumbnails": True,
                "generate_summary_report": True,
                "create_reproducibility_manifest": True
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
        "parallel_processing": True,  # Enable for faster processing
        "max_workers": None  # Auto-detect based on CPU cores (None = auto)
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
        "tasks": ["task_9_document_deduplication", "task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction", "task_6_contrast_enhancement", "task_8_color_handling", "task_7_multipage_segmentation", "task_10_language_detection", "task_11_metadata_extraction", "task_12_output_specifications"]
    },
    
    "orient_only": {
        "name": "Orientation Only",
        "description": "Only run orientation correction",
        "tasks": ["task_1_orientation_correction"]
    },
    
    "skew_only": {
        "name": "Skew Detection Only",
        "description": "Only run skew detection and correction",
        "tasks": ["task_2_skew_detection"],
        "standalone_mode": True  # Allow task to run without dependencies
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
        "tasks": ["task_6_contrast_enhancement"],
        "standalone_mode": True  # Allow task to run without dependencies
    },
    
    "with_enhancement": {
        "name": "With Enhancement",
        "description": "Run all tasks including contrast enhancement",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction", "task_6_contrast_enhancement"]
    },
    
    "segmentation_only": {
        "name": "Multi-page Segmentation Only",
        "description": "Only run multi-page detection and region segmentation",
        "tasks": ["task_7_multipage_segmentation"],
        "standalone_mode": True  # Allow task to run without dependencies
    },
    
    "with_segmentation": {
        "name": "Full Pipeline + Segmentation",
        "description": "Run all tasks including multi-page segmentation",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction", "task_6_contrast_enhancement", "task_7_multipage_segmentation"]
    },
    
    "color_handling_only": {
        "name": "Color Handling Only",
        "description": "Only run color management with stamp/signature preservation",
        "tasks": ["task_8_color_handling"],
        "standalone_mode": True
    },
    
    "with_color_handling": {
        "name": "With Color Handling",
        "description": "Run all tasks including intelligent color management",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction", "task_6_contrast_enhancement", "task_8_color_handling", "task_7_multipage_segmentation"]
    },
    
    "deduplication_only": {
        "name": "Document Deduplication Only",
        "description": "Only run document deduplication with perceptual hashing",
        "tasks": ["task_9_document_deduplication"],
        "standalone_mode": True
    },
    
    "with_deduplication": {
        "name": "With Deduplication",
        "description": "Run all tasks with document deduplication at the start",
        "tasks": ["task_9_document_deduplication", "task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction", "task_6_contrast_enhancement", "task_7_multipage_segmentation", "task_8_color_handling"]
    },
    
    "language_detection_only": {
        "name": "Language & Script Detection Only",
        "description": "Only run language and script detection analysis",
        "tasks": ["task_10_language_detection"],
        "standalone_mode": True
    },
    
    "metadata_extraction_only": {
        "name": "Metadata Extraction Only",
        "description": "Only run comprehensive metadata extraction",
        "tasks": ["task_11_metadata_extraction"],
        "standalone_mode": True
    },
    
    "output_standardization_only": {
        "name": "Output Standardization Only",
        "description": "Only run output standardization and audit generation",
        "tasks": ["task_12_output_specifications"],
        "standalone_mode": True
    },
    
    "comprehensive_pipeline": {
        "name": "Comprehensive Document Processing",
        "description": "Complete pipeline with all analysis and standardization features",
        "tasks": ["task_9_document_deduplication", "task_10_language_detection", "task_11_metadata_extraction", "task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping", "task_4_size_dpi_standardization", "task_5_noise_reduction", "task_6_contrast_enhancement", "task_8_color_handling", "task_7_multipage_segmentation", "task_12_output_specifications"]
    },
    
    "production_mode": {
        "name": "Production Mode",
        "description": "Ultra-fast pipeline with orientation correction, skew detection, and cropping in under 5 seconds",
        "tasks": ["task_1_orientation_correction", "task_2_skew_detection", "task_3_cropping"],
        "production_settings": {
            "single_output_only": True,
            "disable_comparisons": True,
            "disable_intermediate_files": True,
            "disable_color_archive": True,
            "disable_metadata_extraction": True,
            "final_format": "png",
            "cleanup_temp_files": True,
            "fast_processing": True,
            "skip_validation": True
        }
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
