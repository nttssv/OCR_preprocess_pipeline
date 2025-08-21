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
                    'success': True,
                    'output': processed_image_path,
                    'comparison': comparison_path,
                    'task_id': self.task_id,
                    'file_type': file_type,
                    'note': 'Contrast and brightness enhancement completed successfully'
                }
            else:
                return {
                    'success': False,
                    'output': None,
                    'comparison': None,
                    'task_id': self.task_id,
                    'file_type': file_type,
                    'note': 'Contrast enhancement failed'
                }
                
        except Exception as e:
            self.logger.error(f"Error in contrast enhancement task: {str(e)}")
            return {
                'success': False,
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
            
            # Step 4: Final optimization
            self.logger.info(f"   🎯 Step 4: Final optimization...")
            enhanced = self._final_optimization(enhanced, image_stats)
            
            # Convert back to BGR for output
            if len(original_image.shape) == 3:
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_bgr = enhanced
            
            # Step 5: Save outputs
            self.logger.info(f"   💾 Step 5: Saving enhanced image...")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_enhanced.png"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save enhanced image
            cv2.imwrite(output_path, enhanced_bgr)
            self.logger.info(f"   💾 Enhanced image saved: {output_filename}")
            
            # Step 6: Create comparison image
            self.logger.info(f"   🔍 Step 6: Creating comparison visualization...")
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
        
        # Light noise reduction after enhancement (very conservative)
        if self.config['enhancement_mode'] != 'conservative':
            # Very light bilateral filtering to smooth noise without losing detail
            optimized = cv2.bilateralFilter(image, 3, 10, 10)
        else:
            optimized = image
        
        # Ensure full dynamic range is utilized (if not already)
        if image_stats['range_utilization'] < 0.9:
            min_val, max_val = np.min(optimized), np.max(optimized)
            if max_val > min_val:
                optimized = ((optimized - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
        
        return optimized
    
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
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            enhanced_brightness = np.mean(enhanced_gray)
            
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
