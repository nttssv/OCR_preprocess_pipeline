#!/usr/bin/env python3
"""
Task 6: Contrast & Brightness Enhancement
==========================================

This task handles contrast and brightness optimization for document images:
- Contrast & brightness normalization (histogram equalization, CLAHE)
- Adaptive histogram equalization for local contrast improvement
- Light sharpening for enhanced text clarity
- Gamma correction for brightness adjustment
- Adaptive enhancement based on image characteristics

Part of the End-to-End Document Processing Pipeline.
"""

import cv2
import numpy as np
import os
import logging
from PIL import Image, ImageEnhance

class ContrastEnhancementTask:
    """
    Comprehensive contrast and brightness enhancement for document images
    """
    
    def __init__(self, config=None):
        """Initialize contrast enhancement task with configuration"""
        
        self.task_id = "task_6_contrast_enhancement"
        self.task_name = "Contrast & Brightness Enhancement"
        
        # Default configuration
        self.config = {
            'enable_clahe': True,                    # Enable CLAHE (Contrast Limited Adaptive Histogram Equalization)
            'clahe_clip_limit': 2.0,                # CLAHE clip limit
            'clahe_tile_grid_size': (8, 8),         # CLAHE tile grid size
            'enable_histogram_equalization': True,   # Enable global histogram equalization
            'enable_gamma_correction': True,         # Enable gamma correction
            'gamma_value': 1.2,                     # Gamma value for correction (>1 brightens, <1 darkens)
            'enable_contrast_stretching': True,     # Enable contrast stretching
            'contrast_percentile_low': 2,           # Lower percentile for contrast stretching
            'contrast_percentile_high': 98,         # Upper percentile for contrast stretching
            'enable_adaptive_enhancement': True,    # Enable adaptive enhancement based on image analysis
            'enable_sharpening': True,              # Enable light sharpening
            'sharpening_strength': 1.5,             # Sharpening kernel strength
            'enable_brightness_adjustment': True,   # Enable automatic brightness adjustment
            'target_brightness': 180,               # Target mean brightness (0-255)
            'brightness_tolerance': 30,             # Tolerance for brightness adjustment
            'enhancement_mode': 'adaptive',         # 'adaptive', 'conservative', 'aggressive'
            
            # White Background Enhancement Settings (for document quality fixes)
            'force_white_background': False,        # Force pure white backgrounds
            'white_threshold': 240,                 # Threshold for pixels considered white
            'background_multiplier': 1.0,           # Brightness multiplier for background pixels
            'aggressive_white_preservation': False   # Enable aggressive white background preservation
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def run(self, input_path, file_type, output_folder):
        """
        Main execution method for the task manager
        
        Args:
            input_path: Path to input image
            file_type: Type of file being processed
            output_folder: Folder to save outputs
            
        Returns:
            Dictionary with task results
        """
        
        try:
            # Create task-specific output folder
            task_output = os.path.join(output_folder, "task6_contrast_enhancement")
            os.makedirs(task_output, exist_ok=True)
            
            # Process the image
            processed_image_path, comparison_path = self.process_image(input_path, task_output)
            
            if processed_image_path and comparison_path:
                return {
                    'status': 'completed',
                    'output': processed_image_path,
                    'comparison': comparison_path,
                    'task_id': self.task_id,
                    'file_type': file_type,
                    'note': 'Contrast and brightness enhancement completed successfully'
                }
            else:
                return {
                    'status': 'failed',
                    'output': None,
                    'comparison': None,
                    'task_id': self.task_id,
                    'file_type': file_type,
                    'note': 'Contrast enhancement failed'
                }
                
        except Exception as e:
            self.logger.error(f"Error in contrast enhancement task: {str(e)}")
            return {
                'status': 'failed',
                'output_path': None,
                'comparison_path': None,
                'task_id': self.task_id,
                'file_type': file_type,
                'note': f'Error: {str(e)}'
            }
    
    def process_image(self, image_path, output_folder):
        """Process image for contrast and brightness enhancement"""
        
        try:
            self.logger.info(f"   📏 Step 1: Loading image for contrast enhancement...")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"   ❌ Could not load image: {image_path}")
                return None, None
            
            original_image = image.copy()
            height, width = image.shape[:2]
            
            self.logger.info(f"   📏 Original size: {width}x{height}")
            
            # Step 2: Analyze image characteristics for adaptive enhancement
            self.logger.info(f"   🔍 Step 2: Analyzing image characteristics...")
            image_stats = self._analyze_image_characteristics(image)
            self.logger.info(f"   📊 Brightness: {image_stats['mean_brightness']:.1f}, Contrast: {image_stats['contrast_ratio']:.2f}, Dynamic range: {image_stats['dynamic_range']}")
            
            # Step 3: Apply enhancement techniques
            self.logger.info(f"   ✨ Step 3: Applying contrast and brightness enhancement...")
            
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            enhanced = gray.copy()
            enhancement_steps = []
            
            # 3.1: Brightness adjustment (if needed)
            if self.config['enable_brightness_adjustment']:
                enhanced, brightness_adjusted = self._adjust_brightness(enhanced, image_stats)
                if brightness_adjusted:
                    enhancement_steps.append("Brightness Adjustment")
                    self.logger.info(f"   🔧 Applied brightness adjustment")
            
            # 3.2: Gamma correction for better visibility
            if self.config['enable_gamma_correction']:
                enhanced = self._apply_gamma_correction(enhanced, image_stats)
                enhancement_steps.append("Gamma Correction")
                self.logger.info(f"   🔧 Applied gamma correction (γ={self._calculate_optimal_gamma(image_stats):.2f})")
            
            # 3.3: Contrast stretching for better dynamic range
            if self.config['enable_contrast_stretching']:
                enhanced = self._apply_contrast_stretching(enhanced, image_stats)
                enhancement_steps.append("Contrast Stretching")
                self.logger.info(f"   🔧 Applied contrast stretching")
            
            # 3.4: Histogram equalization (conditional)
            if self.config['enable_histogram_equalization'] and self._should_apply_histogram_eq(image_stats):
                enhanced = cv2.equalizeHist(enhanced)
                enhancement_steps.append("Histogram Equalization")
                self.logger.info(f"   🔧 Applied global histogram equalization")
            
            # 3.5: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if self.config['enable_clahe']:
                enhanced = self._apply_clahe(enhanced, image_stats)
                enhancement_steps.append("CLAHE")
                self.logger.info(f"   🔧 Applied CLAHE enhancement")
            
            # 3.6: Light sharpening for text clarity
            if self.config['enable_sharpening']:
                enhanced = self._apply_light_sharpening(enhanced, image_stats)
                enhancement_steps.append("Light Sharpening")
                self.logger.info(f"   🔧 Applied light sharpening")
            
            # Step 4: White Background Enhancement (if enabled)
            if (self.config['force_white_background'] or 
                self.config['enhancement_mode'] == 'aggressive_white_preservation'):
                enhanced = self._apply_white_background_enhancement(enhanced, image_stats)
                enhancement_steps.append("White Background Enhancement")
                self.logger.info(f"   🔧 Applied white background enhancement")
            
            # Step 5: Final optimization
            self.logger.info(f"   🎯 Step 5: Final optimization...")
            enhanced = self._final_optimization(enhanced, image_stats)
            
            # Convert back to BGR for output
            if len(original_image.shape) == 3:
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_bgr = enhanced
            
            # Step 6: Save outputs
            self.logger.info(f"   💾 Step 6: Saving enhanced image...")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_enhanced.png"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save enhanced image
            cv2.imwrite(output_path, enhanced_bgr)
            self.logger.info(f"   💾 Enhanced image saved: {output_filename}")
            
            # Step 6: Create comparison image
            self.logger.info(f"   🔍 Step 7: Creating comparison visualization...")
            comparison_path = self._create_comparison_image(original_image, enhanced_bgr, 
                                                          output_folder, base_name, 
                                                          enhancement_steps, image_stats)
            
            return output_path, comparison_path
            
        except Exception as e:
            self.logger.error(f"   ❌ Error in contrast enhancement processing: {str(e)}")
            return None, None
    
    def _analyze_image_characteristics(self, image):
        """Analyze image characteristics for adaptive enhancement"""
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        stats = {}
        
        # 1. Basic brightness and contrast metrics
        stats['mean_brightness'] = np.mean(gray)
        stats['std_brightness'] = np.std(gray)
        stats['min_intensity'] = np.min(gray)
        stats['max_intensity'] = np.max(gray)
        stats['dynamic_range'] = stats['max_intensity'] - stats['min_intensity']
        
        # 2. Contrast ratio (Weber contrast approximation)
        if stats['mean_brightness'] > 0:
            stats['contrast_ratio'] = stats['std_brightness'] / stats['mean_brightness']
        else:
            stats['contrast_ratio'] = 0
        
        # 3. Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        stats['histogram'] = hist.flatten()
        
        # 4. Dynamic range utilization
        stats['range_utilization'] = stats['dynamic_range'] / 255.0
        
        # 5. Classify image type for adaptive processing
        stats['is_low_contrast'] = stats['contrast_ratio'] < 0.3
        stats['is_dark'] = stats['mean_brightness'] < 100
        stats['is_bright'] = stats['mean_brightness'] > 180
        stats['needs_enhancement'] = stats['is_low_contrast'] or stats['range_utilization'] < 0.7
        
        # 6. Document-specific characteristics
        stats['is_scanned_document'] = (stats['range_utilization'] < 0.8 and 
                                       100 < stats['mean_brightness'] < 200)
        
        return stats
    
    def _adjust_brightness(self, image, image_stats):
        """Adjust overall brightness if needed"""
        
        current_brightness = image_stats['mean_brightness']
        target = self.config['target_brightness']
        tolerance = self.config['brightness_tolerance']
        
        # Check if adjustment is needed
        if abs(current_brightness - target) <= tolerance:
            return image, False
        
        # For aggressive white preservation, never darken - only brighten
        if self.config.get('enhancement_mode') == 'aggressive_white_preservation':
            if current_brightness < target:
                # Always brighten towards target, more aggressive adjustment
                adjustment = min(100, target - current_brightness)  # Increased max adjustment
                adjusted = cv2.add(image, np.ones(image.shape, dtype=np.uint8) * int(adjustment))
                return adjusted, True
            else:
                # Don't darken in white preservation mode
                return image, False
        
        # Original logic for other modes
        # Calculate adjustment
        if current_brightness < target - tolerance:
            # Image is too dark - brighten
            adjustment = min(50, target - current_brightness)
            adjusted = cv2.add(image, np.ones(image.shape, dtype=np.uint8) * int(adjustment))
        elif current_brightness > target + tolerance:
            # Image is too bright - darken
            adjustment = min(50, current_brightness - target)
            adjusted = cv2.subtract(image, np.ones(image.shape, dtype=np.uint8) * int(adjustment))
        else:
            return image, False
        
        return adjusted, True
    
    def _calculate_optimal_gamma(self, image_stats):
        """Calculate optimal gamma value based on image characteristics"""
        
        base_gamma = self.config['gamma_value']
        
        # For aggressive white preservation mode, always use base gamma or lower (brighter)
        if self.config.get('enhancement_mode') == 'aggressive_white_preservation':
            # Never increase gamma (darken) in white preservation mode
            return base_gamma
        
        # Original adaptive logic for other modes
        # Adjust gamma based on image brightness
        if image_stats['is_dark']:
            # Dark images need more brightening
            return min(base_gamma + 0.3, 2.0)
        elif image_stats['is_bright']:
            # Bright images might need slight darkening
            return max(base_gamma - 0.2, 0.8)
        else:
            return base_gamma
    
    def _apply_gamma_correction(self, image, image_stats):
        """Apply gamma correction for brightness enhancement"""
        
        gamma = self._calculate_optimal_gamma(image_stats)
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def _apply_contrast_stretching(self, image, image_stats):
        """Apply contrast stretching to improve dynamic range"""
        
        # Skip contrast stretching if we're preserving white backgrounds
        # Contrast stretching can darken white backgrounds!
        if (self.config.get('enhancement_mode') == 'aggressive_white_preservation' or 
            self.config.get('force_white_background', False)):
            self.logger.info(f"     🔧 Skipping contrast stretching to preserve white background")
            return image
        
        # Calculate percentiles for stretching
        low_percentile = np.percentile(image, self.config['contrast_percentile_low'])
        high_percentile = np.percentile(image, self.config['contrast_percentile_high'])
        
        # Avoid division by zero
        if high_percentile <= low_percentile:
            return image
        
        # Apply contrast stretching
        stretched = np.clip((image - low_percentile) * 255.0 / (high_percentile - low_percentile), 0, 255)
        
        return stretched.astype(np.uint8)
    
    def _should_apply_histogram_eq(self, image_stats):
        """Determine if global histogram equalization should be applied"""
        
        # NEVER apply histogram equalization in white background preservation mode!
        # Histogram equalization redistributes pixel values and can darken white backgrounds
        if (self.config.get('enhancement_mode') == 'aggressive_white_preservation' or 
            self.config.get('force_white_background', False)):
            return False
        
        # Apply histogram equalization only for very low contrast images
        # and avoid for already well-distributed images
        if self.config['enhancement_mode'] == 'conservative':
            return False
        
        return (image_stats['is_low_contrast'] and 
                image_stats['range_utilization'] < 0.5 and
                not image_stats['is_bright'])
    
    def _apply_clahe(self, image, image_stats):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        
        # Adjust CLAHE parameters based on image characteristics
        if image_stats['is_scanned_document']:
            # More conservative for scanned documents
            clip_limit = min(self.config['clahe_clip_limit'], 1.5)
            tile_size = (12, 12)  # Larger tiles for documents
        else:
            clip_limit = self.config['clahe_clip_limit']
            tile_size = self.config['clahe_tile_grid_size']
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        # Apply CLAHE
        return clahe.apply(image)
    
    def _apply_light_sharpening(self, image, image_stats):
        """Apply light sharpening for enhanced text clarity"""
        
        # Adjust sharpening strength based on image characteristics
        if image_stats['is_low_contrast']:
            # More aggressive sharpening for low contrast images
            strength = self.config['sharpening_strength'] * 1.2
        elif image_stats['is_scanned_document']:
            # Conservative sharpening for scanned documents
            strength = self.config['sharpening_strength'] * 0.8
        else:
            strength = self.config['sharpening_strength']
        
        # Create sharpening kernel
        center_value = 1 + (4 * strength)
        edge_value = -strength
        
        kernel = np.array([[0, edge_value, 0],
                          [edge_value, center_value, edge_value],
                          [0, edge_value, 0]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Ensure values stay in valid range
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _final_optimization(self, image, image_stats):
        """Apply final optimization touches"""
        
        # Check for special sharpening enhancement mode (for blur correction)
        if self.config.get('enhancement_mode') == 'sharp_preservation':
            self.logger.info(f"     🔪 Sharp preservation mode - applying enhanced sharpening")
            return self._apply_enhanced_final_sharpening(image, image_stats)
        
        # Check for moderate sharpening mode (prevents artifacts)
        elif self.config.get('enhancement_mode') == 'moderate_sharpening':
            self.logger.info(f"     🔧 Moderate sharpening mode - applying balanced enhancement")
            return self._apply_moderate_final_sharpening(image, image_stats)
        
        # Skip ALL aggressive processing if we're in white background preservation mode
        if (self.config.get('enhancement_mode') == 'aggressive_white_preservation' or 
            self.config.get('force_white_background', False)):
            # For white background preservation, do minimal processing to avoid undoing our work
            self.logger.info(f"     🔧 Preserving white background - minimal final optimization")
            return image
        
        # Light noise reduction after enhancement (very conservative)
        if self.config['enhancement_mode'] != 'conservative':
            # Very light bilateral filtering to smooth noise without losing detail
            optimized = cv2.bilateralFilter(image, 3, 10, 10)
        else:
            optimized = image
        
        # Ensure full dynamic range is utilized (if not already)
        # But NEVER do this for white background preservation mode
        if (image_stats['range_utilization'] < 0.9 and 
            not self.config.get('force_white_background', False) and
            self.config.get('enhancement_mode') != 'aggressive_white_preservation'):
            min_val, max_val = np.min(optimized), np.max(optimized)
            if max_val > min_val:
                optimized = ((optimized - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
        
        return optimized
    
    def _apply_enhanced_final_sharpening(self, image, image_stats):
        """Apply enhanced sharpening for blur correction in final stage"""
        
        enhanced = image.copy()
        
        # Apply final unsharp mask if configured
        if self.config.get('apply_final_unsharp', False):
            sigma = self.config.get('final_unsharp_sigma', 0.8)
            strength = self.config.get('final_unsharp_strength', 1.2)
            
            self.logger.info(f"     🔪 Applying final unsharp mask (sigma={sigma}, strength={strength})")
            enhanced = self._apply_unsharp_mask(enhanced, sigma, strength)
        
        # Apply aggressive sharpening if sharpening strength is high
        if self.config.get('sharpening_strength', 1.0) >= 2.0:
            self.logger.info(f"     🔪 Applying aggressive final sharpening")
            enhanced = self._apply_aggressive_sharpening(enhanced)
        
        return enhanced
    
    def _apply_moderate_final_sharpening(self, image, image_stats):
        """Apply moderate final sharpening that prevents artifacts"""
        
        enhanced = image.copy()
        
        # Apply gentle unsharp mask
        sigma = 1.0  # Larger sigma for smoother sharpening
        strength = 0.6  # Lower strength to prevent artifacts
        
        self.logger.info(f"     🔧 Applying moderate unsharp mask (sigma={sigma}, strength={strength})")
        enhanced = self._apply_unsharp_mask(enhanced, sigma, strength)
        
        # Apply very light kernel sharpening
        self.logger.info(f"     🔧 Applying light kernel sharpening")
        light_kernel = np.array([[ 0, -0.25,  0],
                                [-0.25, 2.0, -0.25],
                                [ 0, -0.25,  0]])
        enhanced = cv2.filter2D(enhanced, -1, light_kernel)
        
        # Gentle contrast boost without over-enhancement
        if image_stats.get('contrast_ratio', 1.0) < 3.0:  # Only if low contrast
            self.logger.info(f"     🔧 Applying gentle contrast boost")
            # enhanced is already grayscale, so just apply CLAHE directly
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _apply_unsharp_mask(self, image, sigma=0.8, strength=1.2, threshold=0):
        """Apply unsharp mask for sharpening"""
        # Convert to float
        image_float = image.astype(np.float64)
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image_float, (0, 0), sigma)
        
        # Create unsharp mask
        mask = image_float - blurred
        
        # Apply threshold
        mask = np.where(np.abs(mask) < threshold, 0, mask)
        
        # Apply strength and add back to original
        sharpened = image_float + strength * mask
        
        # Clip values and convert back
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def _apply_aggressive_sharpening(self, image):
        """Apply aggressive sharpening kernel for severe blur correction"""
        
        # Use a strong sharpening kernel
        kernel = np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Ensure values stay in valid range
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _apply_white_background_enhancement(self, image, image_stats):
        """
        Apply aggressive white background enhancement to fix dark/gray backgrounds
        """
        
        white_threshold = self.config.get('white_threshold', 240)
        background_multiplier = self.config.get('background_multiplier', 1.0)
        
        # Create mask for likely background pixels (high brightness, low variation)
        # Use a more aggressive threshold to catch gray backgrounds
        background_mask = image >= white_threshold
        
        # Also include pixels that are reasonably bright (potential gray backgrounds)
        # Much more aggressive - include darker pixels that might be backgrounds
        gray_background_mask = (image >= 150) & (image < white_threshold)
        
        # Apply background enhancement
        enhanced = image.copy().astype(np.float32)
        
        # Force true white backgrounds
        if self.config.get('force_white_background', False):
            # Force pixels above threshold to pure white (255)
            enhanced[background_mask] = 255
            self.logger.info(f"     🔧 Forced {np.sum(background_mask)} pixels to pure white")
        
        # Apply background multiplier to brighten gray areas
        if background_multiplier > 1.0:
            # Apply multiplier to gray background areas
            enhanced[gray_background_mask] = np.clip(
                enhanced[gray_background_mask] * background_multiplier, 0, 255
            )
            self.logger.info(f"     🔧 Enhanced {np.sum(gray_background_mask)} gray background pixels (×{background_multiplier:.2f})")
            
            # Force very bright gray areas to pure white
            very_bright_gray = (image >= 180) & (image < white_threshold)
            enhanced[very_bright_gray] = 255
            self.logger.info(f"     🔧 Forced {np.sum(very_bright_gray)} bright gray pixels to pure white")
            
            # Force moderately bright areas to pure white (more aggressive)
            moderately_bright = (image >= 160) & (image < 180)
            enhanced[moderately_bright] = 255
            self.logger.info(f"     🔧 Forced {np.sum(moderately_bright)} moderately bright pixels to pure white")
        
        # Additional aggressive white preservation for document backgrounds
        if self.config.get('enhancement_mode') == 'aggressive_white_preservation':
            # Apply a more aggressive background cleaning approach
            
            # Create a mask for non-text regions (likely backgrounds)
            # Text regions have higher local variance
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
            variance = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
            
            # Low variance + high brightness = likely background
            low_variance_mask = variance < 150  # Higher variance threshold
            bright_mask = image > 150  # Much lower brightness threshold
            aggressive_bg_mask = low_variance_mask & bright_mask
            
            # Force these areas to pure white
            enhanced[aggressive_bg_mask] = 255
            self.logger.info(f"     🔧 Aggressive white preservation: {np.sum(aggressive_bg_mask)} pixels")
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _create_comparison_image(self, original, enhanced, output_folder, base_name, 
                               enhancement_steps, image_stats):
        """Create comparison image showing original vs enhanced"""
        
        try:
            # Resize for comparison if needed
            max_width = 1200
            if original.shape[1] > max_width:
                scale = max_width / original.shape[1]
                new_width = max_width
                new_height = int(original.shape[0] * scale)
                original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_AREA)
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create side-by-side comparison
            height, width = original.shape[:2]
            comparison = np.zeros((height + 150, width * 2 + 60, 3), dtype=np.uint8)
            comparison.fill(255)  # White background
            
            # Place images
            comparison[50:50+height, 30:30+width] = original
            comparison[50:50+height, width+60:width+60+width] = enhanced
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (30, 30), font, 1, (0, 0, 0), 2)
            cv2.putText(comparison, "Enhanced", (width + 60, 30), font, 1, (0, 0, 0), 2)
            
            # Add enhancement info
            info_y = height + 80
            enhancement_text = f"Applied: {', '.join(enhancement_steps)}"
            cv2.putText(comparison, enhancement_text, (30, info_y), font, 0.6, (0, 100, 0), 1)
            
            # Add statistics
            original_brightness = image_stats['mean_brightness']
            # enhanced is already grayscale, so just use it directly
            enhanced_brightness = np.mean(enhanced)
            
            stats_text = f"Brightness: {original_brightness:.0f} -> {enhanced_brightness:.0f} | Contrast Ratio: {image_stats['contrast_ratio']:.2f}"
            cv2.putText(comparison, stats_text, (30, info_y + 25), font, 0.6, (100, 0, 0), 1)
            
            # Save comparison
            comparison_filename = f"{base_name}_enhancement_comparison.png"
            comparison_path = os.path.join(output_folder, comparison_filename)
            cv2.imwrite(comparison_path, comparison)
            
            self.logger.info(f"   💾 Comparison saved: {comparison_filename}")
            return comparison_path
            
        except Exception as e:
            self.logger.error(f"   ❌ Error creating comparison image: {str(e)}")
            return None
