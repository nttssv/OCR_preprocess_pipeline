#!/usr/bin/env python3
"""
Blank Page Detection Utility
Detects if a page is blank or has minimal content to avoid unnecessary processing
"""

import cv2
import numpy as np
import logging
from pathlib import Path

class BlankPageDetector:
    """Detects blank pages and pages with minimal content"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Thresholds for blank page detection - MORE CONSERVATIVE for accuracy
        self.blank_threshold = 0.98  # 98% of pixels must be "blank" to consider page blank
        self.min_text_density = 0.005  # Minimum 0.5% text density to consider page non-blank
        self.blank_pixel_threshold = 245  # Pixels above this value are considered "blank"
        
    def is_blank_page(self, image_path):
        """
        Detect if a page is blank or has minimal content
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: {
                'is_blank': bool,
                'blank_percentage': float,
                'text_density': float,
                'reason': str
            }
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    'is_blank': True,
                    'blank_percentage': 1.0,
                    'text_density': 0.0,
                    'reason': 'Failed to load image'
                }
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate blank percentage (pixels above threshold)
            total_pixels = gray.size
            blank_pixels = np.sum(gray > self.blank_pixel_threshold)
            blank_percentage = blank_pixels / total_pixels
            
            # Calculate text density using edge detection
            text_density = self._calculate_text_density(gray)
            
            # Calculate average brightness to detect dark backgrounds
            avg_brightness = np.mean(gray)
            dark_background = avg_brightness < 100  # Consider background dark if average brightness < 100
            
            # Determine if page is blank
            is_blank = False
            reason = ""
            
            # Special case: If page has significant text content (>3%), don't mark as blank regardless of blank percentage
            if text_density > 0.03:
                is_blank = False
                reason = f"Page has significant content: {text_density:.3%} text density (above 3% threshold)"
            elif blank_percentage > self.blank_threshold:
                is_blank = True
                reason = f"Page is {blank_percentage:.1%} blank (threshold: {self.blank_threshold:.1%})"
            elif text_density < self.min_text_density:
                is_blank = True
                reason = f"Text density too low: {text_density:.3%} (threshold: {self.min_text_density:.1%})"
            elif dark_background and text_density < 0.005:  # Special case: dark background with very low text density
                is_blank = True
                reason = f"Dark background detected (avg brightness: {avg_brightness:.1f}) with low text density: {text_density:.3%}"
            else:
                reason = f"Page has content: {text_density:.3%} text density, {blank_percentage:.1%} blank, brightness: {avg_brightness:.1f}"
            
            result = {
                'is_blank': is_blank,
                'blank_percentage': blank_percentage,
                'text_density': text_density,
                'reason': reason
            }
            
            self.logger.info(f"Blank page detection for {Path(image_path).name}: {result['reason']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in blank page detection: {e}")
            return {
                'is_blank': False,
                'blank_percentage': 0.0,
                'text_density': 0.0,
                'reason': f'Error: {str(e)}'
            }
    
    def _calculate_text_density(self, gray_image):
        """
        Calculate text density using edge detection and morphological operations
        
        Args:
            gray_image (np.ndarray): Grayscale image
            
        Returns:
            float: Text density as percentage of image
        """
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            
            # Edge detection using Canny
            edges = cv2.Canny(blurred, 50, 150)
            
            # Morphological operations to connect text components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Count non-zero pixels (text areas)
            text_pixels = np.count_nonzero(dilated)
            total_pixels = dilated.size
            
            # Calculate text density
            text_density = text_pixels / total_pixels
            
            return text_density
            
        except Exception as e:
            self.logger.error(f"Error calculating text density: {e}")
            return 0.0
    
    def create_blank_page_result(self, image_path, output_dir):
        """
        Create a result indicating the page is blank
        
        Args:
            image_path (str): Path to the original image
            output_dir (str): Output directory for results
            
        Returns:
            dict: Result information for blank page
        """
        try:
            image_name = Path(image_path).stem
            output_path = Path(output_dir) / f"{image_name}_blank_page_result"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create a simple text file indicating the page is blank
            result_file = output_path / f"{image_name}_blank_page_info.txt"
            with open(result_file, 'w') as f:
                f.write(f"BLANK PAGE DETECTED\n")
                f.write(f"==================\n")
                f.write(f"Original file: {image_path}\n")
                f.write(f"Detection: Page is blank or has minimal content\n")
                f.write(f"Action: Skipped processing to save resources\n")
                f.write(f"Timestamp: {Path(image_path).stat().st_mtime}\n")
            
            # Create a simple comparison image (original + blank page indicator)
            self._create_blank_page_comparison(image_path, output_path)
            
            return {
                'status': 'blank_page',
                'output': str(output_path),
                'message': 'Page detected as blank - processing skipped',
                'files_created': [str(result_file)]
            }
            
        except Exception as e:
            self.logger.error(f"Error creating blank page result: {e}")
            return {
                'status': 'error',
                'output': '',
                'message': f'Error creating blank page result: {str(e)}',
                'files_created': []
            }
    
    def _create_blank_page_comparison(self, image_path, output_path):
        """
        Create a comparison image showing the original page with blank page indicator
        
        Args:
            image_path (str): Path to the original image
            output_path (Path): Output directory path
        """
        try:
            # Load original image
            image = cv2.imread(str(image_path))
            if image is None:
                return
            
            # Create a blank page indicator overlay
            height, width = image.shape[:2]
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            text = "BLANK PAGE"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Center the text
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2
            
            # Add semi-transparent background
            cv2.rectangle(overlay, (text_x - 20, text_y - text_height - 20), 
                         (text_x + text_width + 20, text_y + 20), (255, 255, 255), -1)
            
            # Add text
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
            
            # Blend with original image
            alpha = 0.7
            result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
            
            # Save comparison
            comparison_path = output_path / f"{Path(image_path).stem}_blank_page_comparison.png"
            cv2.imwrite(str(comparison_path), result)
            
        except Exception as e:
            self.logger.error(f"Error creating blank page comparison: {e}")
