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
        self.target_dpi = 250
        self.standard_width = 2079  # A4 width at 250 DPI (8.27 inches)
        self.standard_height = 2923  # A4 height at 250 DPI (11.69 inches)
        
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
            task_output = os.path.join(output_folder, "task3_size_dpi_standardization")
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
                    'status': 'success',
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
                # Upscaling: Use Lanczos for best quality, then sharpen
                self.logger.info(f"   📏 Upscaling by {scale_factor:.2f}x - using Lanczos interpolation")
                resized_image = cv2.resize(image, (target_width, target_height), 
                                         interpolation=cv2.INTER_LANCZOS4)
                
                # Apply sharpening after upscaling to reduce blur
                self.logger.info(f"   🔪 Applying sharpening filter after upscaling...")
                sharpened = self._sharpen_image(resized_image)
                resized_image = sharpened
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
        """Enhance image for optimal OCR processing"""
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Noise reduction using bilateral filter
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 3. Sharpening using unsharp mask
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        
        # 4. Adaptive thresholding for better text separation
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 5. Morphological operations to clean up text
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for output
        enhanced_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr
    
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
