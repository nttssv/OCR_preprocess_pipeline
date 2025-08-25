#!/usr/bin/env python3
"""
Fast Production Mode Configuration
==================================
Ultra-optimized settings for speed (under 5 seconds processing)
"""

# Ultra-fast configuration for production mode
ULTRA_FAST_CONFIG = {
    # Task 1: Orientation Correction - Fastest detection
    "task_1_orientation_correction": {
        "confidence_threshold": 0.5,  # Even lower threshold for speed
        "fast_mode": True,
        "skip_validation": True,
        "method": "fast",
        "max_resolution": 1000,  # Limit resolution for speed
        "skip_quality_check": True
    },
    
    # Task 2: Skew Detection - Fastest correction
    "task_2_skew_detection": {
        "angle_threshold": 1.0,  # Higher threshold to skip minor corrections
        "fast_mode": True,
        "skip_validation": True,
        "method": "hough_fast",
        "max_resolution": 1000,  # Limit resolution for speed
        "skip_quality_check": True,
        "quick_detection": True
    },
    
    # Task 3: Cropping - Ultra-minimal processing
    "task_3_cropping": {
        "margin": 5,  # Slightly larger margin to avoid edge cases
        "remove_holes": False,  # Skip hole removal for speed
        "remove_edges": False,  # Skip edge removal for speed
        "method": "basic",  # Fastest cropping method
        "skip_validation": True,
        "skip_quality_check": True,
        "fast_mode": True,
        "minimal_processing": True,
        "max_resolution": 2000,  # Limit processing resolution
        "skip_artifact_removal": True,
        "quick_crop": True
    }
}

def get_fast_document_config(filename):
    """Get ultra-fast configuration for production mode"""
    return ULTRA_FAST_CONFIG

def apply_fast_document_config(pipeline, filename):
    """Apply ultra-fast configuration to pipeline"""
    
    config = get_fast_document_config(filename)
    
    print(f"⚡ Applying ULTRA_FAST_CONFIG to {filename} (speed optimized)")
    
    # Apply configuration to pipeline tasks
    applied_count = 0
    for task_id, task_config in config.items():
        if (hasattr(pipeline, 'task_manager') and 
            pipeline.task_manager and 
            hasattr(pipeline.task_manager, 'tasks') and 
            task_id in pipeline.task_manager.tasks):
            
            task_instance = pipeline.task_manager.tasks[task_id]
            if hasattr(task_instance, 'config'):
                task_instance.config.update(task_config)
                applied_count += 1
                print(f"   ⚡ Fast config applied to {task_id}")
    
    print(f"   📊 Applied fast configuration to {applied_count}/{len(config)} tasks")
