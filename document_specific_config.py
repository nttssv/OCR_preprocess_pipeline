#!/usr/bin/env python3
"""
Document-Specific Configuration
==============================
Optimized settings for different document types to fix quality issues
"""

# Configuration for documents with blurriness and gray background issues
CONSERVATIVE_CONFIG = {
    # Task 4: DPI Standardization - Reduce artifacts
    "task_4_size_dpi_standardization": {
        "target_dpi": 300,  # Higher DPI for better quality
        "enhancement_methods": ["clahe", "sharpening"],  # Remove denoising and thresholding
        "maintain_aspect_ratio": True,
        "sharpening_strength": 0.8,  # Reduced sharpening to prevent artifacts
        "use_lanczos_only": True  # Use only Lanczos for cleaner upscaling
    },
    
    # Task 5: Noise Reduction - Much more conservative
    "task_5_noise_reduction": {
        "median_filter_size": 1,           # Minimal median filtering
        "nlm_h": 3,                        # Much weaker NLM denoising  
        "nlm_template_window": 5,          # Smaller template window
        "nlm_search_window": 11,           # Smaller search window
        "background_threshold": 0.95,      # More aggressive background cleaning
        "bleedthrough_alpha": 0.1,         # Minimal bleed-through removal
        "enable_median_filter": False,     # Disable median filter (causes blur)
        "enable_nlm_denoising": False,     # Disable NLM denoising (causes blur)  
        "enable_background_removal": True, # Keep background cleaning
        "enable_bleedthrough_removal": False, # Disable bleed-through (causes gray)
        "enable_bilateral_filter": False,  # Disable bilateral filter (causes blur)
    },
    
    # Task 6: Contrast Enhancement - More conservative
    "task_6_contrast_enhancement": {
        "enable_clahe": True,                    
        "clahe_clip_limit": 1.5,                # Reduced clip limit  
        "clahe_tile_grid_size": (4, 4),         # Smaller tiles for finer control
        "enable_histogram_equalization": False, # Disable global histogram eq
        "enable_gamma_correction": True,         
        "gamma_value": 1.0,                     # No gamma correction
        "enable_contrast_stretching": True,     
        "contrast_percentile_low": 1,           # More conservative stretching
        "contrast_percentile_high": 99,         
        "enable_adaptive_enhancement": False,   # Disable adaptive enhancement
        "enable_sharpening": True,              
        "sharpening_strength": 0.8,             # Reduced sharpening
        "enable_brightness_adjustment": True,   
        "target_brightness": 240,               # Brighter target (whiter background)
        "brightness_tolerance": 20,             
        "enhancement_mode": 'conservative',     # Conservative mode
    }
}

# Configuration for clean scanned documents (like your others)
CLEAN_DOCUMENT_CONFIG = {
    # Keep normal processing for clean documents
    "task_4_size_dpi_standardization": {
        "target_dpi": 250,
        "enhancement_methods": ["clahe", "denoising", "sharpening"],
        "maintain_aspect_ratio": True
    },
    
    "task_5_noise_reduction": {
        "median_filter_size": 3,
        "nlm_h": 10, 
        "enable_median_filter": True,
        "enable_nlm_denoising": True,
        "enable_background_removal": True,
        "enable_bleedthrough_removal": True,
        "enable_bilateral_filter": True,
    },
    
    "task_6_contrast_enhancement": {
        "enable_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
        "enable_histogram_equalization": True,
        "enable_gamma_correction": True,
        "gamma_value": 1.2,
        "enable_contrast_stretching": True,
        "enable_adaptive_enhancement": True,
        "enable_sharpening": True,
        "sharpening_strength": 1.5,
        "enhancement_mode": 'adaptive',
    }
}

def get_document_config(filename):
    """Get appropriate configuration based on filename or document characteristics"""
    
    # Documents that need conservative processing
    problematic_files = [
        "deskew-pdf-before.jpeg",
        "deskew-pdf-before.jpg", 
        "deskew-pdf-before.png"
    ]
    
    if any(problem_file in filename.lower() for problem_file in problematic_files):
        return CONSERVATIVE_CONFIG
    else:
        return CLEAN_DOCUMENT_CONFIG

def apply_document_config(pipeline, filename):
    """Apply document-specific configuration to pipeline"""
    
    config = get_document_config(filename)
    
    # Apply configuration to pipeline tasks
    for task_id, task_config in config.items():
        if hasattr(pipeline, 'pipeline_config') and task_id in pipeline.pipeline_config:
            # Update task settings
            if hasattr(pipeline.task_manager, 'tasks') and task_id in pipeline.task_manager.tasks:
                task_instance = pipeline.task_manager.tasks[task_id]
                if hasattr(task_instance, 'config'):
                    task_instance.config.update(task_config)
                    print(f"✅ Applied conservative config to {task_id}")
