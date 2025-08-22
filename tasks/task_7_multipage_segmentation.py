#!/usr/bin/env python3
"""
Task 7: Multi-page & Region Segmentation
=========================================
- Multi-page per scan detection using projection profile analysis
- Page splitting for double-page scans
- Region of interest isolation (preserve color stamps/signatures, convert text to grayscale)
"""

import cv2
import numpy as np
import os
from PIL import Image
import logging
from typing import List, Tuple, Dict, Optional

class MultiPageSegmentationTask:
    """Task 7: Multi-page & Region Segmentation"""
    
    def __init__(self, logger):
        self.logger = logger
        self.task_name = "Multi-page & Region Segmentation"
        
        # Configuration parameters
        self.config = {
            # Multi-page detection
            "min_gap_width_ratio": 0.05,  # Minimum gap width as ratio of image width
            "min_gap_height_ratio": 0.3,  # Minimum gap height as ratio of image height
            "projection_threshold": 0.95,  # White pixel ratio threshold for gap detection
            
            # Page splitting
            "enable_page_splitting": True,
            "split_margin": 20,  # Pixels to add around each split page
            
            # Region of interest
            "enable_roi_isolation": True,
            "color_preservation_threshold": 30,  # Color saturation threshold
            "stamp_signature_detection": True,
            "text_to_grayscale": True,
            
            # Manual verification
            "enable_manual_verification": False,  # For ambiguous cases
            "confidence_threshold": 0.8,  # Below this requires manual verification
        }
        
    def run(self, input_file, file_type, output_folder):
        """
        Run multi-page segmentation on the input file
        
        Args:
            input_file (str): Path to input file
            file_type (str): Type of file ('image' or 'pdf')
            output_folder (str): Folder to save results
            
        Returns:
            dict: Task result with status and output path(s)
        """
        
        try:
            self.logger.info(f"🔄 Running {self.task_name} on {os.path.basename(input_file)}")
            
            # Create task-specific output folder
            task_output = os.path.join(output_folder, "task7_multipage_segmentation")
            os.makedirs(task_output, exist_ok=True)
            
            if file_type == 'pdf':
                self.logger.warning(f"⚠️  Task 7: PDF files should be pre-converted to images")
                return {
                    'status': 'skipped',
                    'input': input_file,
                    'output': input_file,
                    'file_type': file_type,
                    'note': 'PDF files should be pre-converted to images'
                }
            
            # Load image
            image = cv2.imread(input_file)
            if image is None:
                raise ValueError(f"Could not load image: {input_file}")
            
            self.logger.info(f"   📏 Image size: {image.shape[1]}x{image.shape[0]}")
            
            # Step 1: Multi-page detection using projection profile analysis
            pages_detected = self._detect_multiple_pages(image)
            
            if len(pages_detected) <= 1:
                self.logger.info(f"   📄 Single page detected - applying intelligent processing")
                
                # Still apply ROI isolation if enabled
                if self.config['enable_roi_isolation']:
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    processed_image = self._apply_intelligent_roi_isolation(image, base_name)
                    output_path = os.path.join(task_output, f"{base_name}_segmented.png")
                    cv2.imwrite(output_path, processed_image)
                    
                    # Create comparison
                    self._create_comparison(image, processed_image, task_output, os.path.basename(input_file))
                    
                    return {
                        'status': 'completed',
                        'input': input_file,
                        'output': output_path,
                        'file_type': file_type,
                        'pages_detected': 1,
                        'pages_split': [output_path]
                    }
                else:
                    return {
                        'status': 'completed',
                        'input': input_file,
                        'output': input_file,
                        'file_type': file_type,
                        'pages_detected': 1,
                        'pages_split': [input_file]
                    }
            
            # Step 2: Page splitting for multi-page documents
            split_pages = []
            if self.config['enable_page_splitting']:
                split_pages = self._split_pages(image, pages_detected, task_output, os.path.basename(input_file))
                self.logger.info(f"   ✂️  Split into {len(split_pages)} pages")
            
            # Step 3: Apply ROI isolation to each split page
            final_pages = []
            if self.config['enable_roi_isolation'] and split_pages:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                for i, page_path in enumerate(split_pages):
                    page_image = cv2.imread(page_path)
                    
                    # Extract text content from each page (remove blank areas)
                    processed_page = self._extract_text_content(page_image)
                    
                    # Save processed page
                    processed_path = os.path.join(task_output, f"{base_name}_page_{i+1}_segmented.png")
                    cv2.imwrite(processed_path, processed_page)
                    final_pages.append(processed_path)
                    
                    # Create comparison for each page
                    self._create_page_comparison(page_image, processed_page, task_output, f"{base_name}_page_{i+1}")
            else:
                final_pages = split_pages
            
            # Create overall comparison
            if len(pages_detected) > 1:
                self._create_multipage_comparison(image, final_pages, task_output, os.path.basename(input_file))
            
            # For multi-page documents, copy all split pages to main output folder
            all_outputs = []
            if len(final_pages) > 1:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                
                # Determine the actual main output folder
                # If output_folder is a temp worker folder, get the parent's parent (output)
                if "temp/worker_" in output_folder:
                    # Navigate up from temp/worker_XX_filename/ to get to main output
                    main_output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(output_folder))), "output")
                else:
                    main_output_folder = output_folder
                
                # Ensure the main output folder exists
                os.makedirs(main_output_folder, exist_ok=True)
                
                for i, page_path in enumerate(final_pages):
                    page_name = f"{base_name}_page_{i+1}.png"
                    main_page_path = os.path.join(main_output_folder, page_name)
                    import shutil
                    shutil.copy2(page_path, main_page_path)
                    all_outputs.append(main_page_path)
                    self.logger.info(f"   📄 Copied split page to main output: {page_name}")
                    self.logger.info(f"   📁 Main output folder: {main_output_folder}")
            
            # Determine primary output (first page or original if no splitting)
            primary_output = all_outputs[0] if all_outputs else (final_pages[0] if final_pages else input_file)
            
            return {
                'status': 'completed',
                'input': input_file,
                'output': primary_output,
                'file_type': file_type,
                'pages_detected': len(pages_detected),
                'pages_split': final_pages,
                'all_outputs': all_outputs if all_outputs else final_pages
            }
            
        except Exception as e:
            self.logger.error(f"❌ Task 7 failed: {str(e)}")
            return {
                'status': 'failed',
                'input': input_file,
                'output': input_file,
                'file_type': file_type,
                'error': str(e)
            }
    
    def _detect_multiple_pages(self, image):
        """
        Detect multiple pages using projection profile analysis
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of tuples: [(x_start, x_end), ...] for each detected page
        """
        
        self.logger.info(f"   🔍 Step 1: Multi-page detection using projection profile analysis")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Create vertical projection profile
        # Sum pixel intensities along each column (vertical projection)
        projection = np.sum(gray, axis=0)
        
        # Normalize to 0-1 range (1 = all white, 0 = all black)
        projection_normalized = projection / (height * 255)
        
        # Find large blank vertical gaps
        min_gap_width = int(width * self.config['min_gap_width_ratio'])
        gap_threshold = self.config['projection_threshold']
        
        # Detect gaps (consecutive columns with high white pixel ratio)
        gaps = []
        in_gap = False
        gap_start = 0
        
        for x in range(width):
            is_gap = projection_normalized[x] > gap_threshold
            
            if is_gap and not in_gap:
                # Start of a gap
                gap_start = x
                in_gap = True
            elif not is_gap and in_gap:
                # End of a gap
                gap_width = x - gap_start
                if gap_width >= min_gap_width:
                    gaps.append((gap_start, x))
                in_gap = False
        
        # Handle gap at the end
        if in_gap:
            gap_width = width - gap_start
            if gap_width >= min_gap_width:
                gaps.append((gap_start, width))
        
        # Convert gaps to page boundaries
        pages = []
        last_end = 0
        
        for gap_start, gap_end in gaps:
            if gap_start > last_end:
                pages.append((last_end, gap_start))
            last_end = gap_end
        
        # Add final page if there's content after the last gap
        if last_end < width:
            pages.append((last_end, width))
        
        # Filter out pages that are too narrow
        min_page_width = int(width * 0.2)  # Minimum 20% of image width
        pages = [(start, end) for start, end in pages if end - start >= min_page_width]
        
        self.logger.info(f"   📊 Detected {len(pages)} pages with {len(gaps)} vertical gaps")
        
        return pages
    
    def _split_pages(self, image, page_boundaries, output_folder, base_filename):
        """
        Split image into separate pages based on detected boundaries
        
        Args:
            image: Input image (BGR)
            page_boundaries: List of (x_start, x_end) tuples
            output_folder: Output directory
            base_filename: Base filename for output
            
        Returns:
            List of output file paths
        """
        
        self.logger.info(f"   ✂️  Step 2: Splitting into {len(page_boundaries)} pages")
        
        split_pages = []
        base_name = os.path.splitext(base_filename)[0]
        
        for i, (x_start, x_end) in enumerate(page_boundaries):
            # Add margin around the split
            margin = self.config['split_margin']
            x_start_with_margin = max(0, x_start - margin)
            x_end_with_margin = min(image.shape[1], x_end + margin)
            
            # Extract page region
            page_image = image[:, x_start_with_margin:x_end_with_margin]
            
            # Save split page
            page_filename = f"{base_name}_page_{i+1}_split.png"
            page_path = os.path.join(output_folder, page_filename)
            cv2.imwrite(page_path, page_image)
            split_pages.append(page_path)
            
            self.logger.info(f"     📄 Page {i+1}: {page_image.shape[1]}x{page_image.shape[0]} saved")
        
        return split_pages
    
    def _apply_roi_isolation(self, image):
        """
        Apply region of interest isolation
        - Preserve areas with stamps/signatures in color
        - Convert body text to grayscale/binary
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Processed image (BGR)
        """
        
        self.logger.info(f"   🎨 Step 3: Applying ROI isolation")
        
        if not self.config['enable_roi_isolation']:
            return image
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect colored regions (stamps, signatures, etc.)
        # High saturation indicates colored content
        saturation = hsv[:, :, 1]
        color_mask = saturation > self.config['color_preservation_threshold']
        
        # Create output image
        result = image.copy()
        
        if self.config['text_to_grayscale']:
            # Convert non-colored areas to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Apply grayscale to areas that are not colored
            result[~color_mask] = gray_bgr[~color_mask]
            
            self.logger.info(f"     🖤 Converted {np.sum(~color_mask)} pixels to grayscale")
            self.logger.info(f"     🌈 Preserved {np.sum(color_mask)} colored pixels")
        
        return result
    
    def _has_significant_blank_space(self, image):
        """
        Analyze if image has significant blank/white space that should be removed
        Returns True if the image has large blank areas
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to identify white/blank areas (bright pixels)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Calculate the percentage of white/blank pixels
        total_pixels = white_mask.shape[0] * white_mask.shape[1]
        white_pixels = np.sum(white_mask == 255)
        white_percentage = white_pixels / total_pixels
        
        # If more than 60% of the image is white/blank space, consider it for text extraction
        has_blank_space = white_percentage > 0.6
        
        self.logger.info(f"     📊 White space analysis: {white_percentage:.1%} blank area")
        if has_blank_space:
            self.logger.info(f"     🎯 Significant blank space detected - will extract text content")
        
        return has_blank_space
    
    def _has_colored_elements(self, image):
        """
        Analyze if image has significant colored elements (stamps, signatures)
        Returns True if colored elements are detected
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for stamps (red/orange) and signatures (blue/black)
        # Red/Orange range (stamps)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([15, 255, 255])
        red_lower2 = np.array([165, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        # Blue range (signatures)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Create masks for colored regions
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Combine all color masks
        color_mask = red_mask1 + red_mask2 + blue_mask
        
        # Calculate percentage of colored pixels
        total_pixels = color_mask.shape[0] * color_mask.shape[1]
        colored_pixels = np.sum(color_mask > 0)
        colored_percentage = colored_pixels / total_pixels
        
        # If more than 1% of pixels are colored, consider it significant
        has_colors = colored_percentage > 0.01
        
        self.logger.info(f"     🌈 Color analysis: {colored_percentage:.1%} colored pixels")
        if has_colors:
            self.logger.info(f"     🎨 Colored elements detected (stamps/signatures)")
        
        return has_colors
    
    def _apply_intelligent_roi_isolation(self, image, base_name):
        """
        Apply intelligent ROI isolation based on image content analysis
        - Analyze image features to determine optimal processing
        - High blank space: Extract text content (remove blank areas)
        - Colored elements detected: Preserve colors, convert text to grayscale
        - General case: Standard ROI isolation
        """
        
        # Analyze image content features
        has_blank_space = self._has_significant_blank_space(image)
        has_colored_elements = self._has_colored_elements(image)
        
        if has_blank_space:
            self.logger.info(f"     🎯 High blank space detected - extracting text content")
            return self._extract_text_content(image)
        elif has_colored_elements:
            self.logger.info(f"     🎨 Colored elements detected - preserving colors")
            return self._preserve_colored_elements(image)
        else:
            self.logger.info(f"     📝 Standard ROI isolation")
            return self._apply_roi_isolation(image)
    
    def _extract_text_content(self, image):
        """Extract only text content areas from the image (remove blank areas)"""
        
        self.logger.info(f"     📝 Extracting text content (removing blank areas)")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find text areas (darker areas)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find text regions using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours of text regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.logger.info(f"     ⚠️  No text content found")
            return image
        
        # Find bounding box that encompasses all text
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add some padding around text
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        # Extract the text region
        text_region = image[y:y+h, x:x+w]
        
        text_pixels = w * h
        total_pixels = image.shape[0] * image.shape[1]
        
        self.logger.info(f"     📝 Extracted text region: {w}x{h} ({text_pixels/total_pixels*100:.1f}% of original)")
        
        return text_region
    
    def _preserve_colored_elements(self, image):
        """
        Enhanced color preservation for stamps/signatures
        - Better detection of colored elements
        - Convert text areas to high-contrast binary
        - Store stamp/signature locations for highlighting
        """
        
        self.logger.info(f"     🎨 Enhanced color preservation for stamps/signatures")
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Detect colored regions using multiple criteria
        saturation = hsv[:, :, 1]
        a_channel = lab[:, :, 1]  # Green-Red axis
        b_channel = lab[:, :, 2]  # Blue-Yellow axis
        
        # More sophisticated color detection - be more selective
        high_saturation = saturation > 40  # Higher threshold to be more selective
        significant_color_variation = (np.abs(a_channel - 128) > 15) | (np.abs(b_channel - 128) > 15)
        color_mask = high_saturation & significant_color_variation  # AND instead of OR for stricter detection
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = color_mask.astype(bool)
        
        # Find and store stamp/signature bounding boxes for highlighting
        self.stamp_signature_boxes = self._detect_stamp_signature_boxes(color_mask, image)
        
        # Create high-contrast binary for text areas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding for better text contrast
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Start with binary version
        result = binary_bgr.copy()
        
        # Preserve original colors in colored regions
        result[color_mask] = image[color_mask]
        
        colored_pixels = np.sum(color_mask)
        binary_pixels = np.sum(~color_mask)
        
        self.logger.info(f"     🌈 Preserved {colored_pixels} colored pixels")
        self.logger.info(f"     ⚫ Converted {binary_pixels} pixels to high-contrast binary")
        
        # Count stamps vs signatures
        stamps = sum(1 for box in self.stamp_signature_boxes if box[4] == 'stamp')
        signatures = sum(1 for box in self.stamp_signature_boxes if box[4] == 'signature')
        self.logger.info(f"     📦 Detected {len(self.stamp_signature_boxes)} regions: {stamps} stamps, {signatures} signatures")
        
        return result
    
    def _detect_stamp_signature_boxes(self, color_mask, image):
        """
        Detect bounding boxes around stamp/signature regions with color-based classification
        
        Args:
            color_mask: Boolean mask of colored regions
            image: Original image for color analysis
            
        Returns:
            List of (x, y, w, h, type) tuples where type is 'stamp' or 'signature'
        """
        
        # Find contours of colored regions
        contours, _ = cv2.findContours(color_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        min_area = 500  # Minimum area for stamp/signature detection
        max_area = image.shape[0] * image.shape[1] * 0.3  # Maximum 30% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:  # Filter out too-large regions
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip regions that are too large (likely background)
                region_ratio = area / (image.shape[0] * image.shape[1])
                if region_ratio > 0.2:  # Skip if region is >20% of image
                    continue
                
                # Analyze the dominant color in this region to classify stamp vs signature
                region = image[y:y+h, x:x+w]
                region_type = self._classify_stamp_or_signature(region)
                
                # Add some padding around the detected region
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1], w + 2*padding)
                h = min(image.shape[0], h + 2*padding)
                boxes.append((x, y, w, h, region_type))
        
        return boxes
    
    def _classify_stamp_or_signature(self, region):
        """
        Classify a colored region as stamp or signature based on color analysis
        
        Args:
            region: Image region (BGR format)
            
        Returns:
            'stamp' if predominantly red, 'signature' if blue/black
        """
        
        # Convert to HSV for better color analysis
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        # Red hue ranges (stamps are usually red) - more permissive ranges
        red_lower1 = np.array([0, 30, 30])      # Lower red range (more permissive)
        red_upper1 = np.array([15, 255, 255])   # Wider range
        red_lower2 = np.array([165, 30, 30])    # Upper red range (more permissive)
        red_upper2 = np.array([180, 255, 255])
        
        # Orange-red range (some stamps might be orange-ish)
        orange_lower = np.array([15, 30, 30])
        orange_upper = np.array([25, 255, 255])
        
        # Blue hue range (signatures can be blue)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Black/dark range (signatures can be black) - more restrictive for value
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 80])  # Slightly higher value threshold
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv_region, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv_region, red_lower2, red_upper2)
        orange_mask = cv2.inRange(hsv_region, orange_lower, orange_upper)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = cv2.bitwise_or(red_mask, orange_mask)  # Include orange-red
        
        blue_mask = cv2.inRange(hsv_region, blue_lower, blue_upper)
        black_mask = cv2.inRange(hsv_region, black_lower, black_upper)
        
        # Count pixels for each color
        red_pixels = np.sum(red_mask > 0)
        blue_pixels = np.sum(blue_mask > 0) 
        black_pixels = np.sum(black_mask > 0)
        total_pixels = region.shape[0] * region.shape[1]
        
        # Calculate percentages for better classification
        red_percentage = red_pixels / total_pixels
        blue_percentage = blue_pixels / total_pixels
        black_percentage = black_pixels / total_pixels
        
        # Classify based on dominant color with thresholds
        if red_percentage > 0.1 and red_pixels > blue_pixels and red_pixels > black_pixels:
            return 'stamp'  # Red/orange dominant = stamp
        elif blue_percentage > 0.1 or black_percentage > 0.3:
            return 'signature'  # Blue or significant black = signature
        else:
            return 'signature'  # Default to signature for unclear cases
    
    def _create_comparison(self, original, processed, output_folder, filename):
        """Create side-by-side comparison image with stamp/signature highlighting"""
        
        # Resize images to same height for comparison
        height = max(original.shape[0], processed.shape[0])
        
        # Resize original
        scale_orig = height / original.shape[0]
        new_width_orig = int(original.shape[1] * scale_orig)
        resized_orig = cv2.resize(original, (new_width_orig, height))
        
        # Resize processed
        scale_proc = height / processed.shape[0]
        new_width_proc = int(processed.shape[1] * scale_proc)
        resized_proc = cv2.resize(processed, (new_width_proc, height))
        
        # Draw bounding boxes on original image if stamps/signatures were detected
        resized_orig_with_boxes = resized_orig.copy()
        
        if hasattr(self, 'stamp_signature_boxes') and self.stamp_signature_boxes:
            for i, box_info in enumerate(self.stamp_signature_boxes):
                x, y, w, h, region_type = box_info
                
                # Scale the bounding box coordinates
                scaled_x = int(x * scale_orig)
                scaled_y = int(y * scale_orig)
                scaled_w = int(w * scale_orig)
                scaled_h = int(h * scale_orig)
                
                # Draw bounding box with different colors based on actual color analysis
                if region_type == 'stamp':
                    color = (0, 255, 0)  # Green for stamps (red colored)
                    label = "STAMP"
                else:  # region_type == 'signature'
                    color = (255, 0, 0)  # Blue for signatures (blue/black colored)
                    label = "SIGNATURE"
                
                # Draw thicker bounding box for visibility
                cv2.rectangle(resized_orig_with_boxes, (scaled_x, scaled_y), 
                             (scaled_x + scaled_w, scaled_y + scaled_h), color, 3)
                
                # Add label above the box with background
                label_y = max(scaled_y - 10, 20)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(resized_orig_with_boxes, (scaled_x, label_y - text_size[1] - 5), 
                             (scaled_x + text_size[0] + 5, label_y + 5), color, -1)
                cv2.putText(resized_orig_with_boxes, label, (scaled_x + 2, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create side-by-side comparison with spacing
        gap = 10
        comparison_width = new_width_orig + new_width_proc + gap
        comparison = np.ones((height, comparison_width, 3), dtype=np.uint8) * 255
        
        # Place images
        comparison[:height, :new_width_orig] = resized_orig_with_boxes
        comparison[:height, new_width_orig+gap:] = resized_proc
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "ORIGINAL", (10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(comparison, "PROCESSED", (new_width_orig + gap + 10, 30), font, 0.8, (0, 0, 0), 2)
        
        # Add legend for boxes if any were detected
        if hasattr(self, 'stamp_signature_boxes') and self.stamp_signature_boxes:
            legend_y = height - 60
            cv2.putText(comparison, "Legend:", (10, legend_y), font, 0.6, (0, 0, 0), 1)
            cv2.rectangle(comparison, (10, legend_y + 10), (30, legend_y + 25), (0, 255, 0), 2)
            cv2.putText(comparison, "STAMP", (35, legend_y + 23), font, 0.5, (0, 0, 0), 1)
            cv2.rectangle(comparison, (120, legend_y + 10), (140, legend_y + 25), (255, 0, 0), 2)
            cv2.putText(comparison, "SIGNATURE", (145, legend_y + 23), font, 0.5, (0, 0, 0), 1)
        
        # Save comparison
        base_name = os.path.splitext(filename)[0]
        comparison_path = os.path.join(output_folder, f"{base_name}_segmentation_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        
        self.logger.info(f"   💾 Comparison saved: {os.path.basename(comparison_path)}")
    
    def _create_page_comparison(self, original, processed, output_folder, page_name):
        """Create comparison for individual page"""
        
        # Resize to same height
        height = max(original.shape[0], processed.shape[0])
        
        scale_orig = height / original.shape[0]
        new_width_orig = int(original.shape[1] * scale_orig)
        resized_orig = cv2.resize(original, (new_width_orig, height))
        
        scale_proc = height / processed.shape[0]
        new_width_proc = int(processed.shape[1] * scale_proc)
        resized_proc = cv2.resize(processed, (new_width_proc, height))
        
        # Create comparison
        comparison = np.hstack([resized_orig, resized_proc])
        
        # Save comparison
        comparison_path = os.path.join(output_folder, f"{page_name}_roi_comparison.png")
        cv2.imwrite(comparison_path, comparison)
    
    def _create_multipage_comparison(self, original, split_pages, output_folder, filename):
        """Create comparison showing original vs split pages"""
        
        # Load split page images
        split_images = []
        for page_path in split_pages:
            page_img = cv2.imread(page_path)
            if page_img is not None:
                split_images.append(page_img)
        
        if not split_images:
            return
        
        # Create a grid layout for split pages
        if len(split_images) == 2:
            # Side by side for 2 pages
            combined_splits = np.hstack(split_images)
        else:
            # Stack vertically for more pages
            combined_splits = np.vstack(split_images)
        
        # Resize original and combined splits to same height
        target_height = 800  # Fixed height for comparison
        
        orig_scale = target_height / original.shape[0]
        orig_width = int(original.shape[1] * orig_scale)
        resized_orig = cv2.resize(original, (orig_width, target_height))
        
        split_scale = target_height / combined_splits.shape[0]
        split_width = int(combined_splits.shape[1] * split_scale)
        resized_splits = cv2.resize(combined_splits, (split_width, target_height))
        
        # Create final comparison
        comparison = np.hstack([resized_orig, resized_splits])
        
        # Save comparison
        base_name = os.path.splitext(filename)[0]
        comparison_path = os.path.join(output_folder, f"{base_name}_multipage_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        
        self.logger.info(f"   💾 Multi-page comparison saved: {os.path.basename(comparison_path)}")
