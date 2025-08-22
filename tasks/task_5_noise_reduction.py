#!/usr/bin/env python3
"""
Task 5: Noise Reduction & Denoising
====================================

This task handles various noise reduction and cleaning operations:
- Denoising (median filtering, non-local means)
- Background removal
- Bleed-through removal (ink showing from reverse side)
- Salt and pepper noise removal
- Scanner artifact removal

Part of the End-to-End Document Processing Pipeline.
"""

import cv2
import numpy as np
import os
import logging
from PIL import Image

class NoiseReductionTask:
    """
    Comprehensive noise reduction and denoising for document images
    """
    
    def __init__(self, config=None):
        """Initialize noise reduction task with configuration"""
        
        self.task_id = "task_5_noise_reduction"
        self.task_name = "Noise Reduction & Denoising"
        
        # Default configuration
        self.config = {
            'median_filter_size': 3,           # Size for median filtering (noise removal)
            'nlm_h': 10,                       # Non-local means filter strength
            'nlm_template_window': 7,          # Non-local means template window
            'nlm_search_window': 21,           # Non-local means search window
            'background_threshold': 0.8,       # Threshold for background detection
            'bleedthrough_alpha': 0.3,         # Bleed-through removal strength
            'morphology_kernel_size': 2,       # Kernel size for morphological operations
            'bilateral_d': 9,                  # Bilateral filter diameter
            'bilateral_sigma_color': 75,       # Bilateral filter sigma color
            'bilateral_sigma_space': 75,       # Bilateral filter sigma space
            'enable_median_filter': True,      # Enable median filtering
            'enable_nlm_denoising': True,      # Enable non-local means denoising
            'enable_background_removal': True, # Enable background cleaning
            'enable_bleedthrough_removal': True, # Enable bleed-through removal
            'enable_bilateral_filter': True,  # Enable bilateral filtering
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
            task_output = os.path.join(output_folder, "task5_noise_reduction")
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
                    'note': 'Noise reduction and denoising completed successfully'
                }
            else:
                return {
                    'status': 'failed',
                    'output': None,
                    'comparison': None,
                    'task_id': self.task_id,
                    'file_type': file_type,
                    'note': 'Noise reduction failed'
                }
                
        except Exception as e:
            self.logger.error(f"Error in noise reduction task: {str(e)}")
            return {
                'status': 'failed',
                'output_path': None,
                'comparison_path': None,
                'task_id': self.task_id,
                'file_type': file_type,
                'note': f'Error: {str(e)}'
            }
    
    def process_image(self, image_path, output_folder):
        """Process image for noise reduction and denoising"""
        
        try:
            self.logger.info(f"   📏 Step 1: Loading image for noise reduction...")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"   ❌ Could not load image: {image_path}")
                return None, None
            
            original_image = image.copy()
            height, width = image.shape[:2]
            
            self.logger.info(f"   📏 Original size: {width}x{height}")
            
            # Step 2: Analyze image characteristics
            self.logger.info(f"   🔍 Step 2: Analyzing image for optimal denoising...")
            image_stats = self._analyze_image_noise(image)
            self.logger.info(f"   📊 Noise level: {image_stats['noise_level']:.3f}, Background variance: {image_stats['background_variance']:.1f}")
            
            # Step 3: Apply noise reduction techniques
            self.logger.info(f"   🧹 Step 3: Applying noise reduction techniques...")
            
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            processed = gray.copy()
            processing_steps = []
            
            # 3.1: Median filtering for salt and pepper noise
            if self.config['enable_median_filter'] and image_stats['salt_pepper_noise'] > 0.01:
                self.logger.info(f"   🔧 Applying median filter (kernel: {self.config['median_filter_size']})")
                processed = cv2.medianBlur(processed, self.config['median_filter_size'])
                processing_steps.append("Median Filter")
            
            # 3.2: Bilateral filtering for edge-preserving smoothing
            if self.config['enable_bilateral_filter']:
                self.logger.info(f"   🔧 Applying bilateral filter for edge-preserving denoising")
                processed = cv2.bilateralFilter(processed, 
                                              self.config['bilateral_d'],
                                              self.config['bilateral_sigma_color'],
                                              self.config['bilateral_sigma_space'])
                processing_steps.append("Bilateral Filter")
            
            # 3.3: Non-local means denoising for texture preservation
            if self.config['enable_nlm_denoising'] and image_stats['noise_level'] > 0.02:
                self.logger.info(f"   🔧 Applying non-local means denoising")
                processed = cv2.fastNlMeansDenoising(processed,
                                                   None,
                                                   self.config['nlm_h'],
                                                   self.config['nlm_template_window'],
                                                   self.config['nlm_search_window'])
                processing_steps.append("NL-Means Denoising")
            
            # 3.4: Background cleaning
            if self.config['enable_background_removal']:
                self.logger.info(f"   🔧 Cleaning background artifacts")
                processed = self._clean_background(processed, image_stats)
                processing_steps.append("Background Cleaning")
            
            # 3.5: Bleed-through removal
            if self.config['enable_bleedthrough_removal'] and image_stats['has_bleedthrough']:
                self.logger.info(f"   🔧 Removing bleed-through artifacts")
                processed = self._remove_bleedthrough(processed, image_stats)
                processing_steps.append("Bleed-through Removal")
            
            # Step 4: Final cleanup
            self.logger.info(f"   ✨ Step 4: Final cleanup and optimization...")
            processed = self._final_cleanup(processed)
            
            # Convert back to BGR for output
            if len(original_image.shape) == 3:
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                processed_bgr = processed
            
            # Step 5: Save outputs
            self.logger.info(f"   💾 Step 5: Saving processed image...")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_denoised.png"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save processed image
            cv2.imwrite(output_path, processed_bgr)
            self.logger.info(f"   💾 Denoised image saved: {output_filename}")
            
            # Step 6: Create comparison image
            self.logger.info(f"   🔍 Step 6: Creating comparison visualization...")
            comparison_path = self._create_comparison_image(original_image, processed_bgr, 
                                                          output_folder, base_name, 
                                                          processing_steps, image_stats)
            
            return output_path, comparison_path
            
        except Exception as e:
            self.logger.error(f"   ❌ Error in noise reduction processing: {str(e)}")
            return None, None
    
    def _analyze_image_noise(self, image):
        """Analyze image to determine noise characteristics"""
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        stats = {}
        
        # 1. Overall noise level using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        stats['noise_level'] = laplacian.var() / 10000.0  # Normalized
        
        # 2. Background variance
        # Estimate background as areas with low gradient
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        background_mask = gradient < np.percentile(gradient, 20)
        if np.any(background_mask):
            stats['background_variance'] = np.var(gray[background_mask])
        else:
            stats['background_variance'] = 0
        
        # 3. Salt and pepper noise detection
        # Count isolated white and black pixels
        kernel = np.ones((3,3), np.uint8)
        white_noise = cv2.morphologyEx((gray > 240).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        black_noise = cv2.morphologyEx((gray < 15).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        stats['salt_pepper_noise'] = (np.sum(white_noise) + np.sum(black_noise)) / gray.size
        
        # 4. Bleed-through detection
        # Look for faint, large-scale patterns that might be from reverse side
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray.astype(np.float32), blurred.astype(np.float32))
        # Bleed-through typically creates subtle, large patterns
        stats['has_bleedthrough'] = np.mean(diff) > 5 and np.std(diff) > 8
        
        # 5. Text density for processing decisions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        stats['text_density'] = np.sum(binary > 0) / binary.size
        
        return stats
    
    def _clean_background(self, image, image_stats):
        """Clean background artifacts and scanner noise"""
        
        # Create a mask for likely background areas
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        
        # Background areas should be relatively uniform
        gradient = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
        background_mask = (gradient < np.percentile(gradient, 30)) & (image > 200)
        
        if np.any(background_mask):
            # Smooth background areas more aggressively
            smoothed = cv2.GaussianBlur(image, (5, 5), 0)
            image[background_mask] = smoothed[background_mask]
        
        # Remove small artifacts using morphological operations
        kernel = np.ones((2, 2), np.uint8)
        
        # Remove small white spots (likely scanner artifacts)
        white_spots = (image > 250).astype(np.uint8)
        cleaned_spots = cv2.morphologyEx(white_spots, cv2.MORPH_OPEN, kernel)
        spots_to_clean = white_spots - cleaned_spots
        
        if np.any(spots_to_clean):
            # Fill with neighboring values
            image[spots_to_clean > 0] = cv2.GaussianBlur(image, (3, 3), 0)[spots_to_clean > 0]
        
        return image
    
    def _remove_bleedthrough(self, image, image_stats):
        """Remove bleed-through artifacts from double-sided documents"""
        
        # Bleed-through typically appears as faint, large-scale patterns
        # We can estimate it using a large gaussian blur and subtract
        
        # Create a very blurred version to estimate bleed-through pattern
        large_blur = cv2.GaussianBlur(image, (31, 31), 0)
        
        # The difference between original and blur indicates local variations
        # Bleed-through contributes to the blur but not to local text features
        diff = image.astype(np.float32) - large_blur.astype(np.float32)
        
        # Enhance the difference (text features) while reducing bleed-through
        alpha = self.config['bleedthrough_alpha']
        enhanced = large_blur.astype(np.float32) + (1 + alpha) * diff
        
        # Ensure values stay in valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _final_cleanup(self, image):
        """Final cleanup and optimization"""
        
        # Light sharpening to restore any detail lost during denoising
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Ensure values are in valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def _create_comparison_image(self, original, processed, output_folder, base_name, 
                               processing_steps, image_stats):
        """Create comparison image showing original vs processed"""
        
        try:
            # Resize for comparison if needed
            max_width = 1200
            if original.shape[1] > max_width:
                scale = max_width / original.shape[1]
                new_width = max_width
                new_height = int(original.shape[0] * scale)
                original = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_AREA)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create side-by-side comparison
            height, width = original.shape[:2]
            comparison = np.zeros((height + 150, width * 2 + 60, 3), dtype=np.uint8)
            comparison.fill(255)  # White background
            
            # Place images
            comparison[50:50+height, 30:30+width] = original
            comparison[50:50+height, width+60:width+60+width] = processed
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (30, 30), font, 1, (0, 0, 0), 2)
            cv2.putText(comparison, "Denoised", (width + 60, 30), font, 1, (0, 0, 0), 2)
            
            # Add processing info
            info_y = height + 80
            processing_text = f"Applied: {', '.join(processing_steps)}"
            cv2.putText(comparison, processing_text, (30, info_y), font, 0.6, (0, 100, 0), 1)
            
            stats_text = f"Noise Level: {image_stats['noise_level']:.3f} | Background Var: {image_stats['background_variance']:.1f}"
            cv2.putText(comparison, stats_text, (30, info_y + 25), font, 0.6, (100, 0, 0), 1)
            
            # Save comparison
            comparison_filename = f"{base_name}_denoising_comparison.png"
            comparison_path = os.path.join(output_folder, comparison_filename)
            cv2.imwrite(comparison_path, comparison)
            
            self.logger.info(f"   💾 Comparison saved: {comparison_filename}")
            return comparison_path
            
        except Exception as e:
            self.logger.error(f"   ❌ Error creating comparison image: {str(e)}")
            return None
