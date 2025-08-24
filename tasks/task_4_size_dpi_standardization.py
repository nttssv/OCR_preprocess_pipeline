#!/usr/bin/env python3
"""
Task 4: Size Standardization & DPI Improvement
Standardize image dimensions and improve DPI to 300 for optimal OCR processing
"""

import cv2
import numpy as np
import os
from PIL import Image
import logging

class SizeDPIStandardizationTask:
    """Task 4: Standardize image size and improve DPI to 300"""
    
    def __init__(self, logger):
        self.logger = logger
        self.task_name = "Size & DPI Standardization"
        
        # Default configuration
        self.config = {
            'target_dpi': 250,
            'standard_width': 2079,  # A4 width at 250 DPI (8.27 inches)
            'standard_height': 2923,  # A4 height at 250 DPI (11.69 inches)
            'enhancement_methods': ["clahe", "denoising", "sharpening"],
            'maintain_aspect_ratio': True,
            'sharpening_strength': 1.0,
            'use_lanczos_only': False,
            'preserve_white_background': False,
            'background_brightness_target': 255
        }
        
        # Legacy properties for backward compatibility
        self.target_dpi = self.config['target_dpi']
        self.standard_width = self.config['standard_width']
        self.standard_height = self.config['standard_height']
        
    def run(self, input_file, file_type, output_folder):
        """
        Run size and DPI standardization on the input file
        
        Args:
            input_file (str): Path to input file
            file_type (str): Type of file ('image' or 'pdf')
            output_folder (str): Folder to save results
            
        Returns:
            dict: Task result with status and output path
        """
        
        try:
            self.logger.info(f"🔄 Running {self.task_name} on {os.path.basename(input_file)}")
            
            # Create task-specific output folder
            task_output = os.path.join(output_folder, "task4_size_dpi_standardization")
            os.makedirs(task_output, exist_ok=True)
            
            if file_type == 'pdf':
                self.logger.warning(f"⚠️  Task 3: PDF files not supported for DPI standardization, skipping")
                return {
                    'status': 'skipped',
                    'input': input_file,
                    'output': input_file,  # Pass through unchanged
                    'file_type': file_type,
                    'note': 'PDF files not supported for DPI standardization'
                }
            
            # Process the image
            output_image, comparison_path = self.process_image(input_file, task_output)
            
            if output_image is not None:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_filename = f"{base_name}_standardized_250dpi.png"
                output_path = os.path.join(task_output, output_filename)
                
                return {
                    'status': 'completed',
                    'input': input_file,
                    'output': output_path,
                    'comparison': comparison_path,
                    'file_type': 'image',
                    'note': f'Successfully standardized to 250 DPI'
                }
            else:
                return {
                    'status': 'failed',
                    'input': input_file,
                    'output': None,
                    'file_type': file_type,
                    'note': 'Size/DPI standardization failed'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error in Task 3 (Size & DPI Standardization): {str(e)}")
            return {
                'status': 'failed',
                'input': input_file,
                'output': None,
                'file_type': file_type,
                'note': f'Error: {str(e)}'
            }
        
    def process_image(self, image_path, output_folder):
        """Process image for size standardization and DPI improvement"""
        
        try:
            self.logger.info(f"   📏 Step 1: Loading image for size/DPI standardization...")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"   ❌ Could not load image: {image_path}")
                return None, None
            
            height, width = image.shape[:2]
            self.logger.info(f"   📏 Original size: {width}x{height}")
            
            # Step 2: Calculate target dimensions based on aspect ratio
            self.logger.info(f"   📐 Step 2: Calculating optimal dimensions for 250 DPI...")
            
            # Calculate aspect ratio
            aspect_ratio = width / height
            
            # Determine target dimensions while maintaining aspect ratio
            if aspect_ratio > 1:  # Landscape
                target_width = min(self.standard_width, int(self.standard_height * aspect_ratio))
                target_height = int(target_width / aspect_ratio)
            else:  # Portrait
                target_height = min(self.standard_height, int(self.standard_width / aspect_ratio))
                target_width = int(target_height * aspect_ratio)
            
            self.logger.info(f"   📐 Target dimensions: {target_width}x{target_height} (maintaining aspect ratio)")
            
            # Step 3: Resize image with optimal interpolation
            self.logger.info(f"   🔄 Step 3: Resizing with optimal interpolation...")
            
            # Choose interpolation method based on scaling factor
            scale_factor = max(target_width / width, target_height / height)
            
            if scale_factor > 1.0:
                # Upscaling: Use different strategies based on scale factor
                if scale_factor > 4.0:
                    # Very high upscaling - use multi-step approach for better quality
                    self.logger.info(f"   📏 High upscaling by {scale_factor:.2f}x - using multi-step approach")
                    resized_image = self._multi_step_upscale(image, target_width, target_height)
                else:
                    # Moderate upscaling - use Lanczos
                    self.logger.info(f"   📏 Upscaling by {scale_factor:.2f}x - using Lanczos interpolation")
                    resized_image = cv2.resize(image, (target_width, target_height), 
                                             interpolation=cv2.INTER_LANCZOS4)
                
                # Apply advanced sharpening after upscaling to reduce blur
                # BUT skip aggressive sharpening if preserving white background
                if self.config.get('preserve_white_background', False):
                    self.logger.info(f"   🔧 Preserving white background - using minimal sharpening only")
                    # Very light sharpening only
                    kernel = np.array([[ 0, -0.05,  0],
                                      [-0.05, 1.2, -0.05],
                                      [ 0, -0.05,  0]])
                    resized_image = cv2.filter2D(resized_image, -1, kernel)
                else:
                    self.logger.info(f"   🔪 Applying advanced sharpening after upscaling...")
                    resized_image = self._advanced_sharpen_image(resized_image, scale_factor)
            else:
                # Downscaling: Use INTER_AREA for best quality
                self.logger.info(f"   📏 Downscaling by {scale_factor:.2f}x - using INTER_AREA interpolation")
                resized_image = cv2.resize(image, (target_width, target_height), 
                                         interpolation=cv2.INTER_AREA)
            
            # Step 4: Image enhancement for OCR optimization
            self.logger.info(f"   ✨ Step 4: Enhancing image for OCR optimization...")
            
            enhanced_image = self._enhance_for_ocr(resized_image)
            
            # Step 5: Save with 300 DPI metadata
            self.logger.info(f"   💾 Step 5: Saving with 250 DPI metadata...")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_standardized_{self.target_dpi}dpi.png"
            output_path = os.path.join(output_folder, output_filename)
            
            # Save with PIL to set DPI metadata
            enhanced_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(enhanced_rgb)
            
            # Set DPI metadata
            pil_image.info['dpi'] = (self.target_dpi, self.target_dpi)
            
            # Save with high quality
            pil_image.save(output_path, 'PNG', dpi=(self.target_dpi, self.target_dpi), 
                         optimize=True, quality=95)
            
            self.logger.info(f"   💾 Standardized image saved: {output_filename}")
            
            # Step 6: Create comparison image
            self.logger.info(f"   🔍 Step 6: Creating comparison visualization...")
            
            comparison_filename = f"{base_name}_standardization_comparison.png"
            comparison_path = os.path.join(output_folder, comparison_filename)
            
            self._create_comparison_image(image, enhanced_image, comparison_path, 
                                       width, height, target_width, target_height)
            
            self.logger.info(f"   💾 Comparison saved: {comparison_filename}")
            
            # Return both processed image and comparison
            return enhanced_image, comparison_path
            
        except Exception as e:
            self.logger.error(f"   ❌ Error in size/DPI standardization: {str(e)}")
            return None, None
    
    def _enhance_for_ocr(self, image):
        """Enhanced OCR optimization with multiple enhancement modes"""
        
        # If preserving white background, check for special sharpening config
        if self.config.get('preserve_white_background', False):
            # BUT if we have special sharpening config, apply that
            if (self.config.get('apply_unsharp_mask', False) or 
                self.config.get('post_sharpen_kernel')):
                self.logger.info(f"   🔪 Applying ENHANCED SHARPENING for blur correction")
                return self._apply_enhanced_sharpening(image)
            else:
                self.logger.info(f"   🔧 Preserving white background - skipping OCR enhancement")
                return image  # Return image unchanged
        
        # Check for moderate sharpening mode (prevents artifacts)
        elif self.config.get('enhancement_mode') == 'moderate_sharpening':
            self.logger.info(f"   🔧 Applying MODERATE SHARPENING (artifact prevention)")
            return self._apply_moderate_sharpening(image)
        
        # Standard enhancement for normal documents
        else:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check if the image has bright background (most documents should)
            mean_brightness = np.mean(gray)
            is_bright_background = mean_brightness > 128
            
            if not is_bright_background:
                # For dark images, apply minimal processing to avoid darkening further
                self.logger.info(f"   📝 Dark background detected (brightness: {mean_brightness:.1f}) - minimal processing")
                
                # Just light sharpening, no contrast adjustment
                kernel = np.array([[ 0, -0.05,  0],
                                  [-0.05, 1.2, -0.05],
                                  [ 0, -0.05,  0]])
                sharpened = cv2.filter2D(gray, -1, kernel)
                enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
                
            else:
                # For bright backgrounds, apply very gentle enhancement
                self.logger.info(f"   📝 Bright background detected (brightness: {mean_brightness:.1f}) - gentle enhancement")
                
                # Very minimal CLAHE to avoid darkening
                clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(32, 32))  # Even more conservative
                enhanced = clahe.apply(gray)
                
                # Ensure we don't darken the image
                enhanced = np.maximum(enhanced, gray)  # Take the brighter of original or enhanced
                
                # Very light sharpening only
                kernel = np.array([[-0.05, -0.05, -0.05],
                                  [-0.05,  1.25, -0.05],
                                  [-0.05, -0.05, -0.05]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                
                enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
    
    def _apply_enhanced_sharpening(self, image):
        """Apply enhanced sharpening for blur correction"""
        
        enhanced = image.copy()
        
        # Step 1: Apply unsharp mask if configured
        if self.config.get('apply_unsharp_mask', False):
            sigma = self.config.get('unsharp_sigma', 1.0)
            strength = self.config.get('unsharp_strength', 1.5)
            
            self.logger.info(f"   🔪 Applying unsharp mask (sigma={sigma}, strength={strength})")
            enhanced = self._apply_unsharp_mask(enhanced, sigma, strength)
        
        # Step 2: Apply kernel sharpening if configured
        kernel_type = self.config.get('post_sharpen_kernel')
        if kernel_type:
            self.logger.info(f"   🔪 Applying {kernel_type} kernel sharpening")
            enhanced = self._apply_kernel_sharpening(enhanced, kernel_type)
        
        return enhanced
    
    def _apply_moderate_sharpening(self, image):
        """Apply moderate sharpening that prevents artifacts while improving quality"""
        
        enhanced = image.copy()
        
        # Step 1: Apply moderate unsharp mask if configured
        if self.config.get('apply_unsharp_mask', False):
            sigma = self.config.get('unsharp_sigma', 1.2)  # Larger sigma = smoother
            strength = self.config.get('unsharp_strength', 0.8)  # Lower strength = less artifacts
            
            self.logger.info(f"   🔧 Applying moderate unsharp mask (sigma={sigma}, strength={strength})")
            enhanced = self._apply_unsharp_mask(enhanced, sigma, strength)
        
        # Step 2: Apply light kernel sharpening
        kernel_type = self.config.get('post_sharpen_kernel', 'light')
        if kernel_type:
            self.logger.info(f"   🔧 Applying {kernel_type} kernel sharpening")
            enhanced = self._apply_kernel_sharpening(enhanced, kernel_type)
        
        # Step 3: Apply gentle CLAHE for better contrast without over-enhancement
        if self.config.get('apply_clahe', False):
            clip_limit = self.config.get('clahe_clip_limit', 1.5)
            self.logger.info(f"   🔧 Applying gentle CLAHE (clip_limit={clip_limit})")
            
            # Convert to LAB color space for better CLAHE results
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE only to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge back
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _apply_unsharp_mask(self, image, sigma=1.0, strength=1.5, threshold=0):
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
    
    def _apply_kernel_sharpening(self, image, kernel_type="standard"):
        """Apply different kernel-based sharpening"""
        
        kernels = {
            "light": np.array([[ 0, -0.1,  0],
                              [-0.1, 1.4, -0.1],
                              [ 0, -0.1,  0]]),
            
            "standard": np.array([[ 0, -1,  0],
                                 [-1,  5, -1],
                                 [ 0, -1,  0]]),
            
            "strong": np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]]),
            
            "conservative": np.array([[ 0, -0.25,  0],
                                     [-0.25, 2.0, -0.25],
                                     [ 0, -0.25,  0]])
        }
        
        kernel = kernels.get(kernel_type, kernels["standard"])
        return cv2.filter2D(image, -1, kernel)
    
    def _sharpen_image(self, image):
        """Apply advanced sharpening to reduce blur from upscaling"""
        
        # Convert to float for better precision
        image_float = image.astype(np.float32) / 255.0
        
        # Create sharpening kernel (unsharp mask)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9.0
        
        # Apply sharpening
        sharpened = cv2.filter2D(image_float, -1, kernel)
        
        # Clamp values to valid range
        sharpened = np.clip(sharpened, 0, 1)
        
        # Convert back to uint8
        sharpened_uint8 = (sharpened * 255).astype(np.uint8)
        
        return sharpened_uint8
    
    def _multi_step_upscale(self, image, target_width, target_height):
        """Multi-step upscaling for very high scale factors to reduce artifacts"""
        
        current_image = image.copy()
        current_h, current_w = current_image.shape[:2]
        
        # Calculate how many steps we need (max 2x per step)
        scale_w = target_width / current_w
        scale_h = target_height / current_h
        total_scale = max(scale_w, scale_h)
        
        steps = int(np.ceil(np.log2(total_scale)))
        
        for step in range(steps):
            # Calculate intermediate target size
            if step == steps - 1:
                # Final step - reach exact target
                next_w, next_h = target_width, target_height
            else:
                # Intermediate step - 2x scale
                next_w = min(current_w * 2, target_width)
                next_h = min(current_h * 2, target_height)
            
            # Upscale with Lanczos
            current_image = cv2.resize(current_image, (next_w, next_h), 
                                     interpolation=cv2.INTER_LANCZOS4)
            
            # Light sharpening after each step (except the last)
            if step < steps - 1:
                current_image = self._light_sharpen_image(current_image)
            
            current_w, current_h = next_w, next_h
        
        return current_image
    
    def _light_sharpen_image(self, image):
        """Apply light sharpening during multi-step upscaling"""
        kernel = np.array([[ 0, -0.5,  0],
                          [-0.5, 3.0, -0.5],
                          [ 0, -0.5,  0]])
        return cv2.filter2D(image, -1, kernel)
    
    def _advanced_sharpen_image(self, image, scale_factor):
        """Apply advanced sharpening based on scale factor"""
        
        if scale_factor > 6.0:
            # Very high upscaling - aggressive sharpening needed
            return self._aggressive_sharpen_image(image)
        elif scale_factor > 3.0:
            # High upscaling - strong sharpening
            return self._strong_sharpen_image(image)
        else:
            # Moderate upscaling - standard sharpening
            return self._sharpen_image(image)
    
    def _aggressive_sharpen_image(self, image):
        """Aggressive sharpening for very high upscaling factors - CONSERVATIVE VERSION"""
        
        # Convert to float for better precision
        image_float = image.astype(np.float32) / 255.0
        
        # Very light unsharp mask to avoid massive darkening
        blurred = cv2.GaussianBlur(image_float, (3, 3), 1.0)
        unsharp = image_float + 0.3 * (image_float - blurred)  # Reduced from 0.8 to 0.3
        
        # Much lighter edge enhancement
        kernel = np.array([[-0.2, -0.3, -0.2],
                          [-0.3,  2.8,  -0.3],  # Reduced from 7 to 2.8
                          [-0.2, -0.3, -0.2]])
        enhanced = cv2.filter2D(unsharp, -1, kernel)
        
        # Prevent significant darkening
        result = np.maximum(enhanced, image_float * 0.8)  # Ensure we don't go too dark
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
    
    def _strong_sharpen_image(self, image):
        """Strong sharpening for high upscaling factors - CONSERVATIVE VERSION"""
        
        # Much more conservative sharpening to prevent darkening
        image_float = image.astype(np.float32) / 255.0
        
        # Very light unsharp mask to avoid darkening
        blurred = cv2.GaussianBlur(image_float, (3, 3), 0.8)
        unsharp = image_float + 0.2 * (image_float - blurred)  # Reduced from 0.6 to 0.2
        
        # Much lighter edge sharpening kernel
        kernel = np.array([[-0.1, -0.2, -0.1],
                          [-0.2,  2.2,  -0.2],  # Reduced from 4.0 to 2.2
                          [-0.1, -0.2, -0.1]])
        sharpened = cv2.filter2D(unsharp, -1, kernel)
        
        # Ensure we don't darken the image significantly
        result = np.maximum(sharpened, image_float * 0.85)  # Prevent massive darkening
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
    
    def _create_comparison_image(self, original, processed, output_path, 
                                orig_w, orig_h, new_w, new_h):
        """Create comparison image showing original vs processed"""
        
        try:
            # Resize original to match processed dimensions for fair comparison
            original_resized = cv2.resize(original, (new_w, new_h), 
                                        interpolation=cv2.INTER_LANCZOS4)
            
            # Create side-by-side comparison
            comparison = np.hstack([original_resized, processed])
            
            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (0, 0, 255)  # Red text
            
            # Label original image
            cv2.putText(comparison, f"Original: {orig_w}x{orig_h}", 
                       (10, 30), font, font_scale, color, thickness)
            
            # Label processed image
            cv2.putText(comparison, f"Standardized: {new_w}x{new_h} @ 300 DPI", 
                       (new_w + 10, 30), font, font_scale, color, thickness)
            
            # Add DPI information
            cv2.putText(comparison, f"DPI: 300 (OCR Optimized)", 
                       (new_w + 10, 70), font, font_scale, color, thickness)
            
            # Save comparison
            cv2.imwrite(output_path, comparison)
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not create comparison image: {str(e)}")
    
    def get_task_info(self):
        """Get task information"""
        return {
            'name': self.task_name,
            'description': f'Standardize image dimensions and improve DPI to {self.target_dpi} for optimal OCR processing',
            'target_dpi': self.target_dpi,
            'standard_width': self.standard_width,
            'standard_height': self.standard_height
        }
