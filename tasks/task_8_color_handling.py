#!/usr/bin/env python3
"""
Task 8: Color Handling
======================

This task handles intelligent color management for document processing:
- Maintain color fidelity for legal stamps and signatures (red/blue)
- Convert other text regions to grayscale for storage optimization and OCR speed
- Optional dual-output: color version for archiving, grayscale version for OCR processing

Features:
- Advanced color detection for stamps (red/orange) and signatures (blue/black)
- Intelligent region segmentation based on color analysis
- High-quality grayscale conversion preserving text contrast
- Dual-output system for different use cases
- Storage optimization without losing important visual information

Part of the End-to-End Document Processing Pipeline.
"""

import cv2
import numpy as np
import os
import logging
from PIL import Image
from typing import List, Tuple, Dict, Optional

class ColorHandlingTask:
    """
    Intelligent color handling for document processing with stamp/signature preservation
    """
    
    def __init__(self, config=None, logger=None):
        """Initialize color handling task with configuration"""
        
        self.task_id = "task_8_color_handling"
        self.task_name = "Color Handling"
        self.logger = logger or logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            # Color Detection Settings
            'enable_color_detection': True,           # Enable automatic color region detection
            'stamp_color_threshold': 25,              # Minimum percentage of red/orange pixels for stamp detection
            'signature_color_threshold': 15,          # Minimum percentage of blue/black pixels for signature detection
            'color_saturation_threshold': 40,         # Minimum saturation for color detection
            'color_value_threshold': 50,              # Minimum value (brightness) for color detection
            
            # Color Preservation Settings
            'preserve_stamp_colors': True,            # Preserve red/orange stamp colors
            'preserve_signature_colors': True,        # Preserve blue signature colors
            'color_region_expansion': 10,             # Pixels to expand around detected color regions
            'minimum_region_size': 500,               # Minimum area (pixels) for color region preservation
            
            # Grayscale Conversion Settings
            'grayscale_method': 'adaptive',           # 'adaptive', 'weighted', 'luminance'
            'preserve_text_contrast': True,           # Enhance text contrast during grayscale conversion
            'contrast_enhancement_factor': 1.0,      # Factor for contrast enhancement (no change)
            'gamma_correction': 1.0,                  # Gamma correction for grayscale conversion (no darkening)
            
            # Dual Output Settings
            'enable_dual_output': True,               # Create both color and grayscale versions
            'color_suffix': '_color_archive',         # Suffix for color archive version
            'grayscale_suffix': '_grayscale_ocr',     # Suffix for grayscale OCR version
            'color_quality': 95,                      # JPEG quality for color archive (high quality)
            'grayscale_compression': True,            # Enable compression for grayscale version
            
            # Output Settings
            'create_comparison': True,                # Create before/after comparison
            'highlight_preserved_regions': True,     # Highlight preserved color regions in comparison
            'save_region_masks': False,               # Save intermediate color detection masks (debug)
        }
        
        # Override with provided config
        if config:
            self.config.update(config)
        
        self.detected_regions = []  # Store detected color regions for comparison
    
    def run(self, input_file, file_type, output_folder):
        """
        Run color handling on the input file
        
        Args:
            input_file (str): Path to input file
            file_type (str): Type of file ('image' or 'pdf')
            output_folder (str): Folder to save results
            
        Returns:
            dict: Task result with status and output paths
        """
        
        try:
            self.logger.info(f"🔄 Running {self.task_name} on {os.path.basename(input_file)}")
            
            # Create task-specific output folder
            task_output = os.path.join(output_folder, "task8_color_handling")
            os.makedirs(task_output, exist_ok=True)
            
            if file_type == 'pdf':
                self.logger.warning(f"⚠️  Task 8: PDF files should be pre-converted to images")
                return {
                    'status': 'skipped',
                    'input': input_file,
                    'output': input_file,
                    'file_type': file_type,
                    'note': 'PDF files should be pre-converted to images'
                }
            
            # Process image
            result = self._process_image_color_handling(input_file, task_output)
            
            if result:
                self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
                return result
            else:
                raise Exception("Color handling processing failed")
                
        except Exception as e:
            self.logger.error(f"❌ {self.task_name} failed: {str(e)}")
            return {
                'status': 'failed',
                'input': input_file,
                'output': input_file,
                'file_type': file_type,
                'error': str(e)
            }
    
    def _process_image_color_handling(self, image_path, output_folder):
        """Process single image for color handling"""
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Could not load image: {image_path}")
            
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            self.logger.info(f"   🖼️  Processing image: {filename} ({image.shape[1]}x{image.shape[0]})")
            
            # Step 1: Detect color regions (stamps and signatures)
            color_regions = self._detect_color_regions(image)
            stamp_regions = [r for r in color_regions if r['type'] == 'stamp']
            signature_regions = [r for r in color_regions if r['type'] == 'signature']
            
            self.logger.info(f"   🎨 Detected {len(stamp_regions)} stamp regions, {len(signature_regions)} signature regions")
            
            # Step 2: Create color-preserved version
            if self.config['enable_dual_output']:
                color_preserved = self._create_color_preserved_version(image, color_regions)
                
                # Save color archive version
                color_output_path = os.path.join(output_folder, f"{base_name}{self.config['color_suffix']}.png")
                cv2.imwrite(color_output_path, color_preserved)
                self.logger.info(f"   📁 Color archive saved: {os.path.basename(color_output_path)}")
            else:
                color_output_path = None
                color_preserved = image.copy()
            
            # Step 3: Create grayscale OCR version
            grayscale_ocr = self._create_grayscale_ocr_version(image, color_regions)
            
            # Save grayscale OCR version
            grayscale_output_path = os.path.join(output_folder, f"{base_name}{self.config['grayscale_suffix']}.png")
            cv2.imwrite(grayscale_output_path, grayscale_ocr)
            self.logger.info(f"   📄 Grayscale OCR saved: {os.path.basename(grayscale_output_path)}")
            
            # Step 4: Create comparison image
            comparison_path = None
            if self.config['create_comparison']:
                comparison = self._create_comparison_image(image, color_preserved, grayscale_ocr, color_regions, filename)
                comparison_path = os.path.join(output_folder, f"{base_name}_color_handling_comparison.png")
                cv2.imwrite(comparison_path, comparison)
                self.logger.info(f"   📊 Comparison saved: {os.path.basename(comparison_path)}")
            
            # Step 5: Save region masks (debug)
            mask_paths = []
            if self.config['save_region_masks']:
                mask_paths = self._save_region_masks(color_regions, output_folder, base_name)
            
            # Step 6: Copy important outputs to main output folder
            main_output_folder = None
            try:
                # Get the main output folder (similar logic to multipage segmentation)
                # The output_folder parameter is typically the worker temp folder
                # We need to go up and find the actual output folder
                temp_path_parts = output_folder.split(os.sep)
                for i, part in enumerate(temp_path_parts):
                    if part == 'temp':
                        # Reconstruct path to main output
                        main_output_folder = os.sep.join(temp_path_parts[:i] + ['output'])
                        break
                
                if main_output_folder and os.path.exists(main_output_folder):
                    # Copy color archive to main output
                    if color_output_path:
                        main_color_path = os.path.join(main_output_folder, f"{base_name}_color_archive.png")
                        import shutil
                        shutil.copy2(color_output_path, main_color_path)
                        self.logger.info(f"   📁 Color archive copied to main output: {os.path.basename(main_color_path)}")
                    
                    # Copy grayscale OCR to main output  
                    main_grayscale_path = os.path.join(main_output_folder, f"{base_name}_grayscale_ocr.png")
                    shutil.copy2(grayscale_output_path, main_grayscale_path)
                    self.logger.info(f"   📁 Grayscale OCR copied to main output: {os.path.basename(main_grayscale_path)}")
                    
                    # Copy comparison to main output
                    if comparison_path:
                        main_comparison_path = os.path.join(main_output_folder, f"{base_name}_color_handling_comparison.png")
                        shutil.copy2(comparison_path, main_comparison_path)
                        self.logger.info(f"   📁 Comparison copied to main output: {os.path.basename(main_comparison_path)}")
                        
            except Exception as e:
                self.logger.warning(f"   ⚠️  Could not copy files to main output: {str(e)}")
            
            # Determine primary output (grayscale for OCR pipeline)
            primary_output = grayscale_output_path
            
            return {
                'status': 'completed',
                'input': image_path,
                'output': primary_output,
                'color_archive': color_output_path,
                'grayscale_ocr': grayscale_output_path,
                'comparison': comparison_path,
                'region_masks': mask_paths,
                'file_type': 'image',
                'task': self.task_id,
                'regions_detected': len(color_regions),
                'stamps_detected': len(stamp_regions),
                'signatures_detected': len(signature_regions)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def _detect_color_regions(self, image):
        """
        Detect color regions (stamps and signatures) in the image with improved shape and color analysis
        
        Returns:
            List of dictionaries with region info: {'bbox': (x,y,w,h), 'type': 'stamp'/'signature', 'confidence': float}
        """
        
        self.logger.info(f"   🔍 Step 1: Enhanced stamp/signature detection with shape analysis")
        
        # Convert to multiple color spaces for better detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        height, width = image.shape[:2]
        
        # Enhanced color ranges with better thresholds for real-world documents
        # RED/ORANGE range for stamps (VERY inclusive for stamps with various lighting)
        red_lower1 = np.array([0, 30, 30])     # Very broad lower red range (includes pink/light red)
        red_upper1 = np.array([25, 255, 255])  # Extended to include orange/amber
        red_lower2 = np.array([155, 30, 30])   # Very broad upper red range  
        red_upper2 = np.array([180, 255, 255])
        
        # BLUE/PURPLE range for signatures (broader for ink variations)
        blue_lower = np.array([85, 30, 30])    # Broader blue range (includes navy)
        blue_upper = np.array([145, 255, 255]) # Includes purple/violet ink
        
        # GREEN range for additional stamps/seals
        green_lower = np.array([40, 30, 30])
        green_upper = np.array([85, 255, 255])
        
        # PURPLE/MAGENTA range for special inks
        purple_lower = np.array([145, 30, 30])
        purple_upper = np.array([170, 255, 255])
        
        # Create comprehensive color masks for all types
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        
        # Combine all colored regions for comprehensive detection
        all_color_mask = cv2.bitwise_or(red_mask, blue_mask)
        all_color_mask = cv2.bitwise_or(all_color_mask, green_mask)
        all_color_mask = cv2.bitwise_or(all_color_mask, purple_mask)
        
        # Advanced morphological operations for better shape preservation
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        
        # Clean red mask (for stamps - preserve circular shapes)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_medium)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Clean blue mask (for signatures - preserve linear shapes)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_small)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_small)
        
        regions = []
        
        # Enhanced stamp detection (looking for shapes in ALL colors, not just red)
        if self.config['preserve_stamp_colors']:
            # Check red stamps
            stamp_regions = self._detect_circular_stamps(red_mask, image, 'red')
            for region in stamp_regions:
                regions.append(region)
                x, y, w, h = region['bbox']
                self.logger.info(f"     🔴 RED STAMP detected: ({x},{y},{w},{h}) confidence={region['confidence']:.1f}% circularity={region.get('circularity', 0):.2f}")
            
            # Check green stamps  
            green_stamp_regions = self._detect_circular_stamps(green_mask, image, 'green')
            for region in green_stamp_regions:
                regions.append(region)
                x, y, w, h = region['bbox']
                self.logger.info(f"     🟢 GREEN STAMP detected: ({x},{y},{w},{h}) confidence={region['confidence']:.1f}% circularity={region.get('circularity', 0):.2f}")
            
            # Check purple stamps
            purple_stamp_regions = self._detect_circular_stamps(purple_mask, image, 'purple')
            for region in purple_stamp_regions:
                regions.append(region)
                x, y, w, h = region['bbox']
                self.logger.info(f"     🟣 PURPLE STAMP detected: ({x},{y},{w},{h}) confidence={region['confidence']:.1f}% circularity={region.get('circularity', 0):.2f}")
        
        # Enhanced signature detection (looking for text/linear shapes in blue/purple)
        if self.config['preserve_signature_colors']:
            # Check blue signatures
            signature_regions = self._detect_linear_signatures(blue_mask, image, 'blue')
            for region in signature_regions:
                regions.append(region)
                x, y, w, h = region['bbox']
                self.logger.info(f"     🔵 BLUE SIGNATURE detected: ({x},{y},{w},{h}) confidence={region['confidence']:.1f}% aspect={region.get('aspect_ratio', 0):.2f}")
            
            # Check purple signatures
            purple_sig_regions = self._detect_linear_signatures(purple_mask, image, 'purple')
            for region in purple_sig_regions:
                regions.append(region)
                x, y, w, h = region['bbox']
                self.logger.info(f"     🟣 PURPLE SIGNATURE detected: ({x},{y},{w},{h}) confidence={region['confidence']:.1f}% aspect={region.get('aspect_ratio', 0):.2f}")
        
        self.detected_regions = regions  # Store for comparison
        return regions
    
    def _detect_circular_stamps(self, color_mask, original_image, color_name='unknown'):
        """
        Detect circular/round stamps in red color mask
        """
        regions = []
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (stamps are usually medium-sized) - REDUCED minimum for small stamps
            if area < 100 or area > 50000:  # Lower minimum to catch smaller stamps
                if area > 0:  # Debug log for filtered areas
                    self.logger.debug(f"     🔍 Filtered {color_name} region: area={area} (size filter)")
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate shape properties for stamp detection
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Circularity: 4π*area/perimeter² (1.0 = perfect circle)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Aspect ratio (stamps are often square/circular)
            aspect_ratio = w / h if h > 0 else 0
            
            # Solidity (area/convex_hull_area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Debug log all potential stamps
            self.logger.info(f"     🔍 {color_name.upper()} region analysis: area={area}, circ={circularity:.2f}, aspect={aspect_ratio:.2f}, solid={solidity:.2f}")
            
            # RELAXED stamp criteria for real-world detection:
            # 1. Basic circularity (circularity > 0.1) - very permissive
            # 2. Not extremely elongated (aspect ratio between 0.3 and 3.0) - very broad
            # 3. Some solidity (solidity > 0.4) - allows for worn stamps
            if (circularity > 0.1 and 
                0.3 <= aspect_ratio <= 3.0 and 
                solidity > 0.4):
                
                # Calculate color confidence
                region_mask = color_mask[y:y+h, x:x+w]
                color_pixels = np.sum(region_mask > 0)
                total_pixels = region_mask.shape[0] * region_mask.shape[1]
                confidence = (color_pixels / total_pixels) * 100
                
                self.logger.info(f"     🎯 {color_name.upper()} stamp candidate passed shape tests: confidence={confidence:.1f}%")
                
                # VERY LOW threshold for stamps (many stamps have light areas or text)
                if confidence >= 5:  # Very permissive threshold
                    # Expand region slightly to ensure full stamp is captured
                    expansion = 15  # Larger expansion for stamps
                    height, width = original_image.shape[:2]
                    x_exp = max(0, x - expansion)
                    y_exp = max(0, y - expansion)
                    w_exp = min(width - x_exp, w + 2 * expansion)
                    h_exp = min(height - y_exp, h + 2 * expansion)
                    
                    regions.append({
                        'bbox': (x_exp, y_exp, w_exp, h_exp),
                        'type': 'stamp',
                        'confidence': confidence,
                        'area': area,
                        'circularity': circularity,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity
                    })
        
        return regions
    
    def _detect_linear_signatures(self, color_mask, original_image, color_name='unknown'):
        """
        Detect linear/text-like signatures in color mask
        """
        regions = []
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (signatures are usually smaller than stamps) - REDUCED minimum
            if area < 50 or area > 20000:  # Much lower minimum to catch small signatures
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate shape properties for signature detection
            aspect_ratio = w / h if h > 0 else 0
            
            # VERY RELAXED signature criteria (signatures come in many forms):
            # 1. Any reasonable aspect ratio (not a perfect square)
            # 2. Reasonable size (already filtered above)
            if aspect_ratio > 0.5 or aspect_ratio < 0.5:  # Almost any shape
                
                # Calculate color confidence
                region_mask = color_mask[y:y+h, x:x+w]
                color_pixels = np.sum(region_mask > 0)
                total_pixels = region_mask.shape[0] * region_mask.shape[1]
                confidence = (color_pixels / total_pixels) * 100
                
                self.logger.info(f"     🔍 {color_name.upper()} signature analysis: area={area}, aspect={aspect_ratio:.2f}, confidence={confidence:.1f}%")
                
                # VERY LOW threshold for signatures (ink can be light or faded)
                if confidence >= 3:  # Very permissive threshold for signatures
                    # Expand region slightly
                    expansion = 10
                    height, width = original_image.shape[:2]
                    x_exp = max(0, x - expansion)
                    y_exp = max(0, y - expansion)
                    w_exp = min(width - x_exp, w + 2 * expansion)
                    h_exp = min(height - y_exp, h + 2 * expansion)
                    
                    regions.append({
                        'bbox': (x_exp, y_exp, w_exp, h_exp),
                        'type': 'signature',
                        'confidence': confidence,
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
        
        return regions
    
    def _create_color_preserved_version(self, image, color_regions):
        """
        Create version with color fidelity preserved for important regions
        """
        
        self.logger.info(f"   🎨 Step 2: Creating color-preserved archive version")
        
        # Start with original image for archive
        color_preserved = image.copy()
        
        # Apply any color enhancement if needed
        if self.config.get('enhance_colors', False):
            # Enhance saturation slightly for better color fidelity
            hsv = cv2.cvtColor(color_preserved, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.1)  # Increase saturation by 10%
            color_preserved = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        self.logger.info(f"     ✅ Color regions preserved: {len(color_regions)} regions")
        
        return color_preserved
    
    def _create_grayscale_ocr_version(self, image, color_regions):
        """
        Create optimized grayscale version for OCR while preserving important color regions
        """
        
        self.logger.info(f"   📄 Step 3: Creating grayscale OCR version with color region preservation")
        
        # Start with original image to preserve colors
        result_image = image.copy()
        
        # Create grayscale mask for non-color regions
        if self.config['grayscale_method'] == 'weighted':
            # Weighted average (similar to RGB to grayscale formula)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.config['grayscale_method'] == 'luminance':
            # Use luminance channel from LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            gray = lab[:, :, 0]
        else:  # adaptive
            # Adaptive method based on image characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast for better OCR while preserving brightness
            if self.config['preserve_text_contrast']:
                # Apply very conservative CLAHE to avoid darkening
                clip_limit = min(1.2, self.config.get('contrast_enhancement_factor', 1.0) * 1.2)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))  # Larger tiles for gentler effect
                gray = clahe.apply(gray)
                self.logger.info(f"     📊 Applied conservative CLAHE with clip limit: {clip_limit}")
                
                # Brightness preservation: ensure we don't darken the image
                original_brightness = np.mean(gray)
                if original_brightness > 0:
                    # Normalize to preserve overall brightness level
                    brightness_factor = 1.02  # Slight brightness boost to counteract any darkening
                    gray = np.clip(gray.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
                    new_brightness = np.mean(gray)
                    self.logger.info(f"     💡 Brightness preservation: {original_brightness:.1f} → {new_brightness:.1f}")
        
        # Convert grayscale back to 3-channel
        grayscale_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Create a mask for color regions that should be preserved
        preserve_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Mark color regions in the preservation mask
        for region in color_regions:
            x, y, w, h = region['bbox']
            preserve_mask[y:y+h, x:x+w] = 255
            self.logger.info(f"     🎨 Marked {region['type']} region for color preservation: ({x},{y},{w},{h})")
        
        # Create inverse mask for grayscale areas
        grayscale_mask = cv2.bitwise_not(preserve_mask)
        
        # Apply grayscale only to non-color regions
        # Where preserve_mask is 255 (white), keep original color
        # Where preserve_mask is 0 (black), use grayscale
        
        # Store original brightness for preservation
        original_brightness = np.mean(result_image)
        
        for c in range(3):  # For each color channel
            result_image[:, :, c] = np.where(preserve_mask == 255, 
                                           result_image[:, :, c],  # Keep original color
                                           grayscale_3ch[:, :, c])  # Use grayscale
        
        # Brightness preservation: ensure final result maintains brightness
        final_brightness = np.mean(result_image)
        if original_brightness > 0 and final_brightness > 0:
            brightness_ratio = original_brightness / final_brightness
            if brightness_ratio > 1.02:  # Only adjust if there's significant darkening
                # Apply gentle brightness correction
                brightness_correction = min(1.1, brightness_ratio)  # Cap the correction
                result_image = np.clip(result_image.astype(np.float32) * brightness_correction, 0, 255).astype(np.uint8)
                corrected_brightness = np.mean(result_image)
                self.logger.info(f"     💡 Brightness correction: {final_brightness:.1f} → {corrected_brightness:.1f} (factor: {brightness_correction:.3f})")
        
        # Log preservation details
        preserved_regions = len(color_regions)
        total_preserved_area = sum(r['bbox'][2] * r['bbox'][3] for r in color_regions)
        total_image_area = image.shape[0] * image.shape[1]
        preservation_percentage = (total_preserved_area / total_image_area) * 100
        
        # Count preserved pixels more accurately
        preserved_pixels = np.sum(preserve_mask == 255)
        preservation_percentage_actual = (preserved_pixels / total_image_area) * 100
        
        for region in color_regions:
            x, y, w, h = region['bbox']
            if region['type'] == 'stamp':
                self.logger.info(f"     🔴 STAMP preserved in full color at ({x},{y},{w},{h})")
            elif region['type'] == 'signature':
                self.logger.info(f"     🔵 SIGNATURE preserved in full color at ({x},{y},{w},{h})")
        
        self.logger.info(f"     ✅ Color preservation complete: {preserved_regions} regions preserved ({preservation_percentage_actual:.1f}% of image)")
        
        # Apply final enhancements to non-color regions only
        if self.config.get('gamma_correction', 1.0) != 1.0 and self.config.get('gamma_correction', 1.0) < 1.0:
            # Only apply gamma correction if it's brightening (< 1.0), not darkening (> 1.0)
            gamma = self.config['gamma_correction']
            self.logger.info(f"     🔆 Applying brightness gamma correction: {gamma}")
            # Apply gamma correction only to grayscale areas
            enhanced = np.power(result_image / 255.0, gamma) * 255.0
            enhanced = enhanced.astype(np.uint8)
            
            # Blend enhanced version only in non-color areas
            for c in range(3):
                result_image[:, :, c] = np.where(preserve_mask == 255,
                                               result_image[:, :, c],  # Keep original color regions
                                               enhanced[:, :, c])      # Use enhanced grayscale
        elif self.config.get('gamma_correction', 1.0) > 1.0:
            self.logger.info(f"     ⚠️  Skipping darkening gamma correction: {self.config.get('gamma_correction', 1.0)}")
        
        return result_image
    
    def _create_comparison_image(self, original, color_preserved, grayscale_ocr, color_regions, filename):
        """Create a comparison image showing original, color archive, and grayscale OCR versions"""
        
        # Resize all images to same height for comparison
        target_height = 800
        
        def resize_for_comparison(img):
            h, w = img.shape[:2]
            if h > target_height:
                scale = target_height / h
                new_w = int(w * scale)
                return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
            return img
        
        orig_resized = resize_for_comparison(original)
        color_resized = resize_for_comparison(color_preserved)
        gray_resized = resize_for_comparison(grayscale_ocr)
        
        # Create three-panel comparison
        if self.config['enable_dual_output']:
            comparison = np.hstack([orig_resized, color_resized, gray_resized])
            labels = ["ORIGINAL", "COLOR ARCHIVE", "GRAYSCALE OCR"]
            x_positions = [50, orig_resized.shape[1] + 50, orig_resized.shape[1] + color_resized.shape[1] + 50]
        else:
            comparison = np.hstack([orig_resized, gray_resized])
            labels = ["ORIGINAL", "GRAYSCALE OCR"]
            x_positions = [50, orig_resized.shape[1] + 50]
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        for i, (label, x_pos) in enumerate(zip(labels, x_positions)):
            # Add white background for text
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(comparison, (x_pos - 5, 20), (x_pos + text_size[0] + 5, 55), (255, 255, 255), -1)
            cv2.putText(comparison, label, (x_pos, 45), font, font_scale, (0, 0, 0), thickness)
        
        # Highlight preserved regions if enabled
        if self.config['highlight_preserved_regions'] and color_regions:
            scale_factor = target_height / original.shape[0] if original.shape[0] > target_height else 1.0
            
            for region in color_regions:
                x, y, w, h = region['bbox']
                # Scale coordinates
                x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
                
                # Draw bounding boxes on appropriate panels
                color = (0, 255, 0) if region['type'] == 'stamp' else (255, 0, 0)  # Green for stamps, Red for signatures
                
                # Draw on grayscale OCR panel
                if self.config['enable_dual_output']:
                    offset_x = orig_resized.shape[1] + color_resized.shape[1]
                else:
                    offset_x = orig_resized.shape[1]
                
                cv2.rectangle(comparison, (offset_x + x, y), (offset_x + x + w, y + h), color, 2)
                
                # Add label
                label = region['type'].upper()
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(comparison, (offset_x + x, y - 20), (offset_x + x + text_size[0] + 4, y), color, -1)
                cv2.putText(comparison, label, (offset_x + x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary info at bottom
        info_text = f"File: {filename} | Regions: {len(color_regions)} | Stamps: {len([r for r in color_regions if r['type'] == 'stamp'])} | Signatures: {len([r for r in color_regions if r['type'] == 'signature'])}"
        cv2.putText(comparison, info_text, (20, comparison.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return comparison
    
    def _save_region_masks(self, color_regions, output_folder, base_name):
        """Save region detection masks for debugging"""
        
        mask_paths = []
        
        for i, region in enumerate(color_regions):
            region_type = region['type']
            mask_filename = f"{base_name}_{region_type}_mask_{i}.png"
            mask_path = os.path.join(output_folder, mask_filename)
            
            # Create mask image
            x, y, w, h = region['bbox']
            mask = np.zeros((h, w), dtype=np.uint8)
            mask.fill(255)  # White mask
            
            cv2.imwrite(mask_path, mask)
            mask_paths.append(mask_path)
        
        return mask_paths
