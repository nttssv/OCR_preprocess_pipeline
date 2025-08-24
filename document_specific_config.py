#!/usr/bin/env python3
"""
Document-Specific Configuration
==============================
Optimized settings for different document types to fix quality issues
"""

# MODERATE sharpening configuration for file 4 (reduces artifacts)
MODERATE_SHARPENING_CONFIG = {
    # Task 4: DPI Standardization - Moderate sharpening to prevent artifacts
    "task_4_size_dpi_standardization": {
        "target_dpi": 300,  # Higher DPI for better quality
        "enhancement_methods": ["moderate_sharpening"],  # Use moderate approach
        "maintain_aspect_ratio": True,
        "sharpening_strength": 1.5,  # Moderate sharpening
        "use_lanczos_only": True,  # Use only Lanczos for cleaner scaling
        "preserve_white_background": False,  # Allow normal enhancement
        "background_brightness_target": 240,  # Normal background
        "apply_unsharp_mask": True,  # Apply moderate unsharp mask
        "unsharp_sigma": 1.2,  # Larger sigma for smoother edges
        "unsharp_strength": 0.8,  # Reduced strength to prevent artifacts
        "post_sharpen_kernel": "light",  # Light kernel sharpening
        "apply_clahe": True,  # Gentle contrast enhancement
        "clahe_clip_limit": 1.5,  # Moderate enhancement
        "enhancement_mode": "moderate_sharpening"  # Special mode
    },
    
    # Task 5: Noise Reduction - Minimal processing to preserve sharpness
    "task_5_noise_reduction": {
        "enable_median_filter": False,     # Disable median filter (causes blur)
        "enable_nlm_denoising": False,     # Disable NLM denoising (causes blur)
        "enable_bilateral_filter": False,  # Disable bilateral filter (causes blur)
        "enable_background_removal": True, # Light background cleaning
        "enable_bleedthrough_removal": False, # Disable (causes blur)
        "background_threshold": 0.97,      # Conservative
        "preserve_text_sharpness": True,    # Priority: maintain sharpness
        "force_background_white": False,    # Don't force pure white
        "white_intensity_boost": 1.0        # No boost
    },
    
    # Task 6: Contrast Enhancement - Moderate enhancement with sharpening control
    "task_6_contrast_enhancement": {
        "enable_clahe": True,
        "clahe_clip_limit": 1.5,                # Conservative CLAHE
        "clahe_tile_grid_size": (8, 8),         # Standard tiles
        "enable_histogram_equalization": False, # Disable (can reduce sharpness)
        "enable_gamma_correction": True,        # Enable gamma for brightness
        "gamma_value": 0.95,                    # Light gamma adjustment
        "enable_contrast_stretching": True,
        "contrast_percentile_low": 2.0,         # Conservative stretching
        "contrast_percentile_high": 98.0,
        "enable_adaptive_enhancement": False,   # Disable adaptive (can blur)
        "enable_sharpening": True,              # Enable moderate sharpening
        "sharpening_strength": 1.0,             # Moderate sharpening
        "enable_brightness_adjustment": True,
        "target_brightness": 240,               # Normal background
        "brightness_tolerance": 10,             # Reasonable tolerance
        "enhancement_mode": 'moderate_sharpening', # Moderate sharpening mode
        "force_white_background": False,        # Don't force pure white
        "white_threshold": 200,                 # Standard threshold
        "background_multiplier": 1.0            # No background boost
    },
    
    # Task 8: Color Handling - Enhanced color preservation with brightness preservation
    "task_8_color_handling": {
        "enable_color_detection": True,
        "stamp_color_threshold": 15,            # Lower threshold = detect more stamps
        "signature_color_threshold": 10,        # Lower threshold = detect more signatures
        "color_saturation_threshold": 25,       # Lower threshold = detect lighter colors
        "color_value_threshold": 30,            # Lower threshold = detect darker colors
        "preserve_stamp_colors": True,
        "preserve_signature_colors": True,
        "color_region_expansion": 15,           # Expand region more to capture edges
        "minimum_region_size": 200,             # Smaller minimum = detect more regions
        "red_detection_enhanced": True,         # Enhanced red detection for stamps
        "blue_detection_enhanced": True,        # Enhanced blue detection for signatures
        "preserve_text_contrast": True,
        "contrast_enhancement_factor": 1.0,     # No darkening - preserve brightness
        "gamma_correction": 1.0                 # No gamma correction to prevent darkening
    }
}

# Configuration for documents needing extra sharpening (like file 4) - AGGRESSIVE VERSION
SHARP_ENHANCEMENT_CONFIG = {
    # Task 4: DPI Standardization - AGGRESSIVE sharpening
    "task_4_size_dpi_standardization": {
        "target_dpi": 300,  # Higher DPI for better quality
        "enhancement_methods": ["aggressive_sharpening"],  # Focus on sharpening
        "maintain_aspect_ratio": True,
        "sharpening_strength": 3.0,  # Very aggressive sharpening
        "use_lanczos_only": True,  # Use only Lanczos for cleaner scaling
        "preserve_white_background": True,  # Still preserve white background
        "background_brightness_target": 255,  # Pure white background
        "apply_unsharp_mask": True,  # NEW: Apply unsharp mask
        "unsharp_sigma": 1.0,  # NEW: Unsharp mask parameters
        "unsharp_strength": 1.5,  # NEW: Medium strength
        "post_sharpen_kernel": "standard"  # NEW: Apply kernel sharpening after scaling
    },
    
    # Task 5: Noise Reduction - MINIMAL processing to preserve sharpness
    "task_5_noise_reduction": {
        "enable_median_filter": False,     # Disable median filter (causes blur)
        "enable_nlm_denoising": False,     # Disable NLM denoising (causes blur)
        "enable_bilateral_filter": False,  # Disable bilateral filter (causes blur)
        "enable_background_removal": False, # Skip all noise reduction to preserve detail
        "enable_bleedthrough_removal": False, # Disable (causes blur)
        "background_threshold": 0.99,      # VERY conservative
        "preserve_text_sharpness": True,    # Priority: maintain sharpness
        "force_background_white": True,     # Force backgrounds to pure white
        "white_intensity_boost": 1.2        # Boost white pixel intensity
    },
    
    # Task 6: Contrast Enhancement - BALANCED enhancement with sharpening
    "task_6_contrast_enhancement": {
        "enable_clahe": True,
        "clahe_clip_limit": 1.2,                # Conservative CLAHE to avoid artifacts
        "clahe_tile_grid_size": (4, 4),         # Smaller tiles for detail preservation
        "enable_histogram_equalization": False, # Disable (can reduce sharpness)
        "enable_gamma_correction": True,        # Enable gamma to brighten backgrounds
        "gamma_value": 0.85,                    # Slightly brighten
        "enable_contrast_stretching": True,
        "contrast_percentile_low": 1.0,         # Conservative stretching
        "contrast_percentile_high": 99.0,
        "enable_adaptive_enhancement": False,   # Disable adaptive (can blur)
        "enable_sharpening": True,              # ENABLE aggressive sharpening
        "sharpening_strength": 2.0,             # Strong sharpening
        "enable_brightness_adjustment": True,
        "target_brightness": 250,               # Bright white background
        "brightness_tolerance": 5,              # Tighter tolerance
        "enhancement_mode": 'sharp_preservation', # NEW: Focus on sharpness
        "force_white_background": True,         # Force pure white backgrounds
        "white_threshold": 215,                 # Slightly higher threshold
        "background_multiplier": 1.1,           # Modest background boost
        "apply_final_unsharp": True,            # NEW: Final unsharp mask
        "final_unsharp_sigma": 0.8,             # NEW: Fine detail enhancement
        "final_unsharp_strength": 1.2           # NEW: Light final sharpening
    },
    
    # Task 8: Color Handling - Enhanced color preservation for stamps/signatures
    "task_8_color_handling": {
        "enable_color_detection": True,
        "stamp_color_threshold": 15,            # Lower threshold = detect more stamps
        "signature_color_threshold": 10,        # Lower threshold = detect more signatures
        "color_saturation_threshold": 25,       # Lower threshold = detect lighter colors
        "color_value_threshold": 30,            # Lower threshold = detect darker colors
        "preserve_stamp_colors": True,
        "preserve_signature_colors": True,
        "color_region_expansion": 15,           # Expand region more to capture edges
        "minimum_region_size": 200,             # Smaller minimum = detect more regions
        "red_detection_enhanced": True,         # Enhanced red detection for stamps
        "blue_detection_enhanced": True,        # Enhanced blue detection for signatures
        "preserve_text_contrast": True,
        "contrast_enhancement_factor": 1.0,     # No darkening - preserve brightness
        "gamma_correction": 1.0                 # No gamma correction to prevent darkening
    }
}

# Configuration for high-quality white background preservation (DEFAULT)
WHITE_BACKGROUND_CONFIG = {
    # Task 4: DPI Standardization - Preserve crisp quality
    "task_4_size_dpi_standardization": {
        "target_dpi": 300,  # Higher DPI for better quality
        "enhancement_methods": ["minimal_clahe"],  # Only minimal enhancement
        "maintain_aspect_ratio": True,
        "sharpening_strength": 0.3,  # Very light sharpening
        "use_lanczos_only": True,  # Use only Lanczos for cleaner scaling
        "preserve_white_background": True,  # Ensure white background preservation
        "background_brightness_target": 255  # Pure white background
    },
    
    # Task 5: Noise Reduction - Minimal processing to preserve quality
    "task_5_noise_reduction": {
        "enable_median_filter": False,     # Disable median filter (causes blur)
        "enable_nlm_denoising": False,     # Disable NLM denoising (causes blur)
        "enable_bilateral_filter": False,  # Disable bilateral filter (causes blur)
        "enable_background_removal": True, # Only clean obvious artifacts
        "enable_bleedthrough_removal": False, # Disable (causes gray backgrounds)
        "background_threshold": 0.99,      # VERY aggressive background cleaning
        "preserve_text_sharpness": True,    # Prioritize text sharpness
        "force_background_white": True,     # Force backgrounds to pure white
        "white_intensity_boost": 1.2        # Boost white pixel intensity
    },
    
    # Task 6: Contrast Enhancement - AGGRESSIVE white background enhancement
    "task_6_contrast_enhancement": {
        "enable_clahe": True,
        "clahe_clip_limit": 1.5,                # Slightly more aggressive CLAHE
        "clahe_tile_grid_size": (2, 2),         # Larger tiles for smoother result
        "enable_histogram_equalization": False, # Disable (can cause gray backgrounds)
        "enable_gamma_correction": True,        # Enable gamma to brighten backgrounds
        "gamma_value": 0.8,                     # Lower gamma = brighter backgrounds
        "enable_contrast_stretching": True,
        "contrast_percentile_low": 0.1,         # More aggressive stretching
        "contrast_percentile_high": 99.9,
        "enable_adaptive_enhancement": False,   # Disable adaptive (can cause artifacts)
        "enable_sharpening": False,             # Disable sharpening (can cause halos)
        "sharpening_strength": 0.0,
        "enable_brightness_adjustment": True,
        "target_brightness": 252,               # Very bright white background
        "brightness_tolerance": 5,              # Tighter tolerance
        "enhancement_mode": 'aggressive_white_preservation', # Super aggressive white
        "force_white_background": True,         # Force pure white backgrounds
        "white_threshold": 220,                 # Lower threshold = more pixels considered white
        "background_multiplier": 1.15           # Boost background brightness by 15%
    },
    
    # Task 8: Color Handling - Enhanced color preservation for stamps/signatures
    "task_8_color_handling": {
        "enable_color_detection": True,
        "stamp_color_threshold": 15,            # Lower threshold = detect more stamps
        "signature_color_threshold": 10,        # Lower threshold = detect more signatures
        "color_saturation_threshold": 25,       # Lower threshold = detect lighter colors
        "color_value_threshold": 30,            # Lower threshold = detect darker colors
        "preserve_stamp_colors": True,
        "preserve_signature_colors": True,
        "color_region_expansion": 15,           # Expand region more to capture edges
        "minimum_region_size": 200,             # Smaller minimum = detect more regions
        "red_detection_enhanced": True,         # Enhanced red detection for stamps
        "blue_detection_enhanced": True,        # Enhanced blue detection for signatures
        "preserve_text_contrast": True,
        "contrast_enhancement_factor": 1.0,     # No darkening - preserve brightness
        "gamma_correction": 1.0                 # No gamma correction to prevent darkening
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
    
    # Special handling for file 4 - apply MODERATE sharpening (prevents artifacts)
    if "4.png" in filename or "4_" in filename:
        print(f"   🔧 File 4 detected: Applying MODERATE_SHARPENING_CONFIG to prevent artifacts")
        return MODERATE_SHARPENING_CONFIG
    
    # Apply white background preservation to ALL other documents by default
    # This prevents blurriness and gray artifacts
    
    # Only use aggressive processing for documents that specifically need it
    aggressive_processing_files = [
        "very_dirty_scan",
        "low_quality_photo", 
        "extremely_noisy"
    ]
    
    if any(aggressive_file in filename.lower() for aggressive_file in aggressive_processing_files):
        return CLEAN_DOCUMENT_CONFIG  # Use aggressive processing only when needed
    else:
        return WHITE_BACKGROUND_CONFIG  # Use white-preserving config by default

def apply_document_config(pipeline, filename):
    """Apply document-specific configuration to pipeline"""
    
    config = get_document_config(filename)
    
    # Determine config name for logging
    if config == WHITE_BACKGROUND_CONFIG:
        config_name = "WHITE_BACKGROUND_CONFIG"
    elif config == MODERATE_SHARPENING_CONFIG:
        config_name = "MODERATE_SHARPENING_CONFIG"
    elif config == SHARP_ENHANCEMENT_CONFIG:
        config_name = "SHARP_ENHANCEMENT_CONFIG"
    else:
        config_name = "CLEAN_DOCUMENT_CONFIG"
    
    print(f"🎨 Applying {config_name} to {filename} (optimized processing)")
    
    # Debug: Check what's available
    print(f"   🔍 Debug: Pipeline has task_manager: {hasattr(pipeline, 'task_manager')}")
    if hasattr(pipeline, 'task_manager') and pipeline.task_manager:
        print(f"   🔍 Debug: Task manager has tasks: {hasattr(pipeline.task_manager, 'tasks')}")
        if hasattr(pipeline.task_manager, 'tasks'):
            print(f"   🔍 Debug: Available tasks: {list(pipeline.task_manager.tasks.keys())}")
    
    # Apply configuration to pipeline tasks
    applied_count = 0
    for task_id, task_config in config.items():
        print(f"   🔧 Attempting to apply config for {task_id}...")
        
        # Check if pipeline has task manager and the specific task
        if (hasattr(pipeline, 'task_manager') and 
            pipeline.task_manager and 
            hasattr(pipeline.task_manager, 'tasks') and 
            task_id in pipeline.task_manager.tasks):
            
            task_instance = pipeline.task_manager.tasks[task_id]
            if hasattr(task_instance, 'config'):
                # Store original config for comparison
                original_keys = set(task_instance.config.keys())
                
                # Update configuration
                task_instance.config.update(task_config)
                
                # Show what was applied
                new_keys = set(task_config.keys())
                updated_keys = new_keys.intersection(original_keys)
                added_keys = new_keys - original_keys
                
                print(f"   ✅ Applied config to {task_id}:")
                if updated_keys:
                    print(f"      📝 Updated: {', '.join(sorted(updated_keys))}")
                if added_keys:
                    print(f"      ➕ Added: {', '.join(sorted(added_keys))}")
                
                applied_count += 1
            else:
                print(f"   ❌ Task {task_id} has no 'config' attribute")
        else:
            print(f"   ⚠️  Task {task_id} not found in task manager")
    
    print(f"   📊 Successfully applied configuration to {applied_count}/{len(config)} tasks")
