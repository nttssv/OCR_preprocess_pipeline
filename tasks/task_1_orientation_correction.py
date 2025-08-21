#!/usr/bin/env python3
"""
Task 3: Orientation Correction
Contains the actual ML-based orientation detection code copied from 3.fix_orientation/ml_based_orientation_detection.py
"""

import os
import cv2
import numpy as np

import logging

class OrientationCorrectionTask:
    """Task 3: Orientation Correction - Contains actual ML-based orientation detection algorithm"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Orientation Correction"
        self.task_id = "task_1_orientation_correction"
        
    def run(self, input_file, file_type, output_folder):
        """
        Run orientation correction on the input file
        
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
            task_output = os.path.join(output_folder, "task1_orientation_correction")
            os.makedirs(task_output, exist_ok=True)
            
            # Process based on file type
            if file_type == 'pdf':
                result = self._process_pdf_orientation(input_file, task_output)
            else:
                result = self._process_image_orientation(input_file, task_output)
            
            if result:
                self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
                return result
            else:
                self.logger.error(f"❌ {self.task_name} failed for {os.path.basename(input_file)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in {self.task_name}: {str(e)}")
            return None
    
    def _process_image_orientation(self, image_path, output_folder):
        """Process single image for orientation correction using the actual algorithm"""
        
        try:
            filename = os.path.basename(image_path)
            self.logger.info(f"📄 Processing: {filename}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Failed to load image")
            
            # Store current image for quality-based testing
            self.current_image = image
            
            height, width = image.shape[:2]
            self.logger.info(f"   📏 Image size: {width}x{height}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Step 1: ML-based readability analysis
            self.logger.info("   🔍 Step 1: ML-based readability analysis...")
            
            # Method 1: Text structure analysis
            orientation_1 = self._analyze_text_structure(gray)
            self.logger.info(f"   📊 Text structure: {orientation_1}°")
            
            # Method 2: Character recognition patterns
            orientation_2 = self._analyze_character_patterns(gray)
            self.logger.info(f"   📊 Character patterns: {orientation_2}°")
            
            # Method 3: Reading direction analysis
            orientation_3 = self._analyze_reading_direction(gray)
            self.logger.info(f"   📊 Reading direction: {orientation_3}°")
            
            # Method 4: Zebra stripe analysis (NEW - based on row sum patterns)
            orientation_4 = self._analyze_zebra_stripes(gray)
            self.logger.info(f"   📊 Zebra stripe pattern: {orientation_4}°")
            
            # Step 2: Feature analysis and smart method selection
            features = self._analyze_image_features(gray)
            self.logger.info(f"   🔍 Image features: {features}")
            
            # Step 3: Smart consensus based on image characteristics
            final_orientation = self._smart_consensus_with_features([orientation_1, orientation_2, orientation_3, orientation_4], features, filename)
            self.logger.info(f"   🎯 Final orientation: {final_orientation}°")
            
            # Step 3: Apply rotation correction
            self.logger.info("   🔄 Step 3: Applying rotation correction...")
            corrected_image = self._apply_ml_rotation_correction(image, final_orientation)
            
            # Generate outputs
            base_name = os.path.splitext(filename)[0]
            
            # Save corrected image
            corrected_path = os.path.join(output_folder, f"{base_name}_oriented.png")
            cv2.imwrite(corrected_path, corrected_image)
            self.logger.info(f"💾 Corrected image saved: {corrected_path}")
            
            # Create and save comparison image
            comparison = self._create_comparison_image(image, corrected_image, filename, 
                                                     final_orientation, [orientation_1, orientation_2, orientation_3, orientation_4])
            comparison_path = os.path.join(output_folder, f"{base_name}_comparison.png")
            cv2.imwrite(comparison_path, comparison)
            self.logger.info(f"💾 Comparison image saved: {comparison_path}")
            
            return {
                'input': image_path,
                'output': corrected_path,
                'comparison': comparison_path,
                'orientations': [orientation_1, orientation_2, orientation_3, orientation_4],
                'final_orientation': final_orientation,
                'status': 'completed',
                'task': self.task_id
            }
                
        except Exception as e:
            self.logger.error(f"Error processing image for orientation correction: {str(e)}")
            return None
    
    def _process_pdf_orientation(self, pdf_path, output_folder):
        """Process PDF for orientation correction"""
        
        try:
            self.logger.info("📄 Processing PDF for orientation correction")
            
            # Convert PDF to image using pdf2image
            from pdf2image import convert_from_path
            
            # Convert first page to image
            pages = convert_from_path(pdf_path, first_page=1, last_page=1)
            if not pages:
                raise Exception("Could not convert PDF to image")
            
            # Convert PIL image to OpenCV format
            import numpy as np
            pil_image = pages[0]
            opencv_image = np.array(pil_image)
            opencv_image = opencv_image[:, :, ::-1].copy()  # RGB to BGR
            
            # Process the converted image
            filename = os.path.basename(pdf_path)
            base_name = os.path.splitext(filename)[0]
            
            # Save converted image for reference
            converted_path = os.path.join(output_folder, f"{base_name}_converted.png")
            cv2.imwrite(converted_path, opencv_image)
            
            # Process the converted image using the same logic as images
            result = self._process_image_orientation(converted_path, output_folder)
            
            if result:
                # Update the result to reflect it came from PDF
                result['input'] = pdf_path
                result['note'] = 'PDF converted to image and processed'
                return result
            else:
                raise Exception("Failed to process converted PDF image")
            
        except Exception as e:
            self.logger.error(f"Error processing PDF for orientation correction: {str(e)}")
            return None
    
    def _analyze_text_structure(self, gray):
        """Analyze text structure to determine orientation"""
        
        height, width = gray.shape
        
        # Create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Test different angles
        angles = [0, 90, 180, 270]
        best_angle = 0
        best_structure_score = 0
        
        for angle in angles:
            if angle == 0:
                rotated = binary
            else:
                # Rotate image
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(binary, rotation_matrix, (width, height))
            
            # Analyze text structure
            structure_score = self._calculate_structure_score(rotated)
            
            if structure_score > best_structure_score:
                best_structure_score = structure_score
                best_angle = angle
        
        return best_angle
    
    def _fast_structure_score(self, binary_image):
        """Calculate text structure score (original fast algorithm)"""
        
        height, width = binary_image.shape
        
        # 1. Text line density
        horizontal_kernel = np.ones((1, 25), np.uint8)
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
        line_density = np.sum(horizontal_lines > 0) / (height * width)
        
        # 2. Text block analysis
        kernel = np.ones((3, 3), np.uint8)
        connected = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        block_scores = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-6)
                
                # Text blocks should have reasonable aspect ratios
                if 0.3 <= aspect_ratio <= 10:
                    # Calculate density within block
                    block_mask = np.zeros_like(binary_image)
                    cv2.drawContours(block_mask, [contour], -1, 255, -1)
                    text_pixels = np.sum(cv2.bitwise_and(binary_image, block_mask) > 0)
                    density = text_pixels / (area + 1e-6)
                    
                    # Score based on density and aspect ratio
                    block_score = density * min(aspect_ratio, 5.0) / 5.0
                    block_scores.append(block_score)
        
        avg_block_score = np.mean(block_scores) if block_scores else 0
        
        # 3. Text alignment
        # Calculate horizontal projection variance
        horizontal_projection = np.sum(binary_image, axis=1)
        projection_variance = np.var(horizontal_projection)
        
        # 4. Edge analysis
        edges = cv2.Canny(binary_image, 30, 100)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Combine scores
        structure_score = (
            0.3 * line_density +
            0.3 * avg_block_score +
            0.2 * (1.0 / (1.0 + projection_variance/10000)) +
            0.2 * edge_density
        )
        
        return structure_score
    
    def _detect_text_baseline(self, binary_image):
        """Detect if text baseline is more prominent (indicates correct orientation)"""
        
        height, width = binary_image.shape
        
        # Divide image into upper and lower halves
        upper_half = binary_image[:height//2, :]
        lower_half = binary_image[height//2:, :]
        
        # Calculate text density in each half
        upper_density = np.sum(upper_half > 0) / (upper_half.shape[0] * upper_half.shape[1])
        lower_density = np.sum(lower_half > 0) / (lower_half.shape[0] * lower_half.shape[1])
        
        # In correct orientation, lower half (baseline) should have more text
        if upper_density + lower_density > 0:
            baseline_ratio = lower_density / (upper_density + lower_density)
            return baseline_ratio
        else:
            return 0.5  # Neutral score
    
    def _calculate_structure_score(self, binary_image):
        """Calculate text structure score"""
        
        height, width = binary_image.shape
        
        # 1. Text line density
        horizontal_kernel = np.ones((1, 25), np.uint8)
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
        line_density = np.sum(horizontal_lines > 0) / (height * width)
        
        # 2. Text block analysis
        kernel = np.ones((3, 3), np.uint8)
        connected = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        block_scores = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-6)
                
                # Text blocks should have reasonable aspect ratios
                if 0.3 <= aspect_ratio <= 10:
                    # Calculate density within block
                    block_mask = np.zeros_like(binary_image)
                    cv2.drawContours(block_mask, [contour], -1, 255, -1)
                    text_pixels = np.sum(cv2.bitwise_and(binary_image, block_mask) > 0)
                    density = text_pixels / (area + 1e-6)
                    
                    # Score based on density and aspect ratio
                    block_score = density * min(aspect_ratio, 5.0) / 5.0
                    block_scores.append(block_score)
        
        avg_block_score = np.mean(block_scores) if block_scores else 0
        
        # 3. Text alignment
        # Calculate horizontal projection variance
        horizontal_projection = np.sum(binary_image, axis=1)
        projection_variance = np.var(horizontal_projection)
        
        # 4. Edge analysis
        edges = cv2.Canny(binary_image, 30, 100)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Combine scores
        structure_score = (
            0.3 * line_density +
            0.3 * avg_block_score +
            0.2 * (1.0 / (1.0 + projection_variance/10000)) +
            0.2 * edge_density
        )
        
        return structure_score
    
    def _analyze_character_patterns(self, gray):
        """Analyze character patterns for orientation detection"""
        
        height, width = gray.shape
        
        # Create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Test different angles
        angles = [0, 90, 180, 270]
        best_angle = 0
        best_pattern_score = 0
        
        for angle in angles:
            if angle == 0:
                rotated = binary
            else:
                # Rotate image
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(binary, rotation_matrix, (width, height))
            
            # Analyze character patterns
            pattern_score = self._calculate_pattern_score(rotated)
            
            if pattern_score > best_pattern_score:
                best_pattern_score = pattern_score
                best_angle = angle
        
        return best_angle
    
    def _fast_pattern_score(self, binary_image):
        """Calculate character pattern score (original fast algorithm)"""
        
        height, width = binary_image.shape
        
        # Find characters
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 5:
            return 0
        
        # Filter characters by size
        character_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 <= area <= 3000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-6)
                if 0.1 <= aspect_ratio <= 5.0:
                    character_boxes.append((x, y, w, h))
        
        if len(character_boxes) < 3:
            return 0
        
        # Analyze character distribution
        # Sort by position
        character_boxes.sort(key=lambda box: (box[1], box[0]))  # y, then x
        
        # Calculate spacing patterns
        horizontal_spacings = []
        vertical_spacings = []
        
        for i in range(len(character_boxes) - 1):
            current = character_boxes[i]
            next_char = character_boxes[i + 1]
            
            # Horizontal spacing
            h_spacing = next_char[0] - (current[0] + current[2])
            if 0 < h_spacing < 100:
                horizontal_spacings.append(h_spacing)
            
            # Vertical spacing
            v_spacing = next_char[1] - (current[1] + current[3])
            if 0 < v_spacing < 50:
                vertical_spacings.append(v_spacing)
        
        # Calculate spacing consistency
        h_consistency = 0
        v_consistency = 0
        
        if len(horizontal_spacings) > 2:
            h_variance = np.var(horizontal_spacings)
            h_consistency = 1.0 / (1.0 + h_variance)
        
        if len(vertical_spacings) > 2:
            v_variance = np.var(vertical_spacings)
            v_consistency = 1.0 / (1.0 + v_variance)
        
        # Character density
        total_area = sum(w * h for _, _, w, h in character_boxes)
        density = total_area / (height * width)
        
        # Pattern score
        pattern_score = (
            0.4 * h_consistency +
            0.3 * v_consistency +
            0.3 * density
        )
        
        return pattern_score
    
    def _calculate_pattern_score(self, binary_image):
        """Calculate character pattern score"""
        
        height, width = binary_image.shape
        
        # Find characters
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 5:
            return 0
        
        # Filter characters by size
        character_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 <= area <= 3000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-6)
                if 0.1 <= aspect_ratio <= 5.0:
                    character_boxes.append((x, y, w, h))
        
        if len(character_boxes) < 3:
            return 0
        
        # Analyze character distribution
        # Sort by position
        character_boxes.sort(key=lambda box: (box[1], box[0]))  # y, then x
        
        # Calculate spacing patterns
        horizontal_spacings = []
        vertical_spacings = []
        
        for i in range(len(character_boxes) - 1):
            current = character_boxes[i]
            next_char = character_boxes[i + 1]
            
            # Horizontal spacing
            h_spacing = next_char[0] - (current[0] + current[2])
            if 0 < h_spacing < 100:
                horizontal_spacings.append(h_spacing)
            
            # Vertical spacing
            v_spacing = next_char[1] - (current[1] + current[3])
            if 0 < v_spacing < 50:
                vertical_spacings.append(v_spacing)
        
        # Calculate spacing consistency
        h_consistency = 0
        v_consistency = 0
        
        if len(horizontal_spacings) > 2:
            h_variance = np.var(horizontal_spacings)
            h_consistency = 1.0 / (1.0 + h_variance)
        
        if len(vertical_spacings) > 2:
            v_variance = np.var(vertical_spacings)
            v_consistency = 1.0 / (1.0 + v_variance)
        
        # Character density
        total_area = sum(w * h for _, _, w, h in character_boxes)
        density = total_area / (height * width)
        
        # Pattern score
        pattern_score = (
            0.4 * h_consistency +
            0.3 * v_consistency +
            0.3 * density
        )
        
        return pattern_score
    
    def _analyze_reading_direction(self, gray):
        """Analyze reading direction for orientation detection"""
        
        height, width = gray.shape
        
        # Create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Test different angles
        angles = [0, 90, 180, 270]
        best_angle = 0
        best_reading_score = 0
        
        for angle in angles:
            if angle == 0:
                rotated = binary
            else:
                # Rotate image
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(binary, rotation_matrix, (width, height))
            
            # Analyze reading direction
            reading_score = self._calculate_reading_score(rotated)
            
            if reading_score > best_reading_score:
                best_reading_score = reading_score
                best_angle = angle
        
        return best_angle
    
    def _analyze_zebra_stripes(self, gray):
        """Analyze zebra stripe patterns to determine orientation (based on row sum analysis)"""
        
        height, width = gray.shape
        
        # Resize for speed if image is too large
        max_dim = 800
        if max(gray.shape) > max_dim:
            scale = max_dim / max(gray.shape)
            new_width = int(gray.shape[1] * scale)
            new_height = int(gray.shape[0] * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = gray.shape
        
        # Create binary image (invert like in the original code)
        src = 255 - gray  # Invert like the original code
        _, binary = cv2.threshold(src, 140, 255, cv2.THRESH_BINARY)
        
        # Make square for consistent rotation (crop center like original)
        small_dimension = min(height, width)
        src_square = binary[:small_dimension, :small_dimension]
        
        # Test different angles
        angles = [0, 90, 180, 270]
        best_angle = 0
        best_zebra_score = 0
        
        for angle in angles:
            # Rotate image
            rotated = self._rotate_image(src_square, angle)
            
            # Crop center region for analysis (like original code)
            h, w = rotated.shape
            buffer = min(h, w) - int(min(h, w) / 1.5)
            roi = rotated[int(h/2-buffer):int(h/2+buffer), int(w/2-buffer):int(w/2+buffer)]
            
            # Calculate row sums (zebra stripe analysis)
            zebra_score = self._calculate_zebra_score(roi)
            
            if zebra_score > best_zebra_score:
                best_zebra_score = zebra_score
                best_angle = angle
        
        return best_angle
    
    def _rotate_image(self, image, angle):
        """Rotate image by given angle"""
        
        if angle == 0:
            return image
        
        rows, cols = image.shape
        center = (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        return rotated
    
    def _calculate_zebra_score(self, roi):
        """Calculate zebra stripe score based on row sum patterns (FIXED)"""
        
        if roi.shape[0] < 10:  # Too small to analyze
            return 0
        
        # Calculate row sums like in the original code
        row_sums = []
        for r in range(roi.shape[0]):
            row_sum = np.sum(roi[r, :])
            row_sums.append(row_sum)
        
        if not row_sums or max(row_sums) == 0:
            return 0
        
        # Normalize row sums to (0, 1) for better analysis
        max_sum = max(row_sums)
        normalized_sums = [rs / max_sum for rs in row_sums]
        
        # FIXED: Better zebra pattern detection
        # Well-oriented text should have:
        # 1. Clear alternating pattern (text lines vs white space)
        # 2. Multiple peaks (text lines)
        # 3. Consistent spacing between peaks
        
        # Find peaks (text lines) using a simple threshold
        threshold = np.mean(normalized_sums) + 0.1 * np.std(normalized_sums)
        peaks = []
        for i, val in enumerate(normalized_sums):
            if val > threshold:
                # Check if this is a local maximum
                is_peak = True
                for j in range(max(0, i-2), min(len(normalized_sums), i+3)):
                    if j != i and normalized_sums[j] > val:
                        is_peak = False
                        break
                if is_peak:
                    peaks.append(i)
        
        # Merge nearby peaks (same text line)
        merged_peaks = []
        if peaks:
            merged_peaks.append(peaks[0])
            for peak in peaks[1:]:
                if peak - merged_peaks[-1] > 3:  # Minimum distance between lines
                    merged_peaks.append(peak)
        
        # Calculate zebra score based on:
        # 1. Number of text lines detected
        # 2. Spacing consistency between lines
        # 3. Overall pattern strength
        
        num_lines = len(merged_peaks)
        if num_lines < 2:
            return 0.1  # Very low score for no pattern
        
        # Calculate spacing between lines
        line_spacings = []
        for i in range(1, len(merged_peaks)):
            spacing = merged_peaks[i] - merged_peaks[i-1]
            line_spacings.append(spacing)
        
        # Spacing consistency (lower variance = better)
        if len(line_spacings) > 1:
            spacing_variance = np.var(line_spacings)
            spacing_mean = np.mean(line_spacings)
            spacing_consistency = 1.0 / (1.0 + spacing_variance / (spacing_mean ** 2))
        else:
            spacing_consistency = 0.5
        
        # Pattern strength (how clear the peaks are)
        if len(normalized_sums) > 0:
            pattern_strength = np.std(normalized_sums)  # Higher std = clearer pattern
        else:
            pattern_strength = 0
        
        # Content density (amount of text)
        content_density = np.mean(normalized_sums)
        
        # Combine scores with better weighting
        zebra_score = (
            0.3 * min(num_lines / 10.0, 1.0) +      # Number of lines (capped at 10)
            0.3 * spacing_consistency +             # Spacing consistency
            0.2 * min(pattern_strength, 1.0) +      # Pattern clarity
            0.2 * min(content_density, 1.0)         # Content amount
        )
        
        return min(zebra_score, 1.0)
    
    def _analyze_image_features(self, gray):
        """Analyze image features to determine which orientation method works best"""
        
        height, width = gray.shape
        
        # Create binary image for analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        features = {}
        
        # 1. Image size and aspect ratio
        features['width'] = width
        features['height'] = height
        features['aspect_ratio'] = width / height
        features['total_pixels'] = width * height
        features['is_large'] = features['total_pixels'] > 1000000  # > 1MP
        
        # 2. Text density and distribution
        text_pixels = np.sum(binary > 0)
        features['text_density'] = text_pixels / (width * height)
        features['is_text_heavy'] = features['text_density'] > 0.15
        
        # 3. Text structure characteristics
        # Find connected components (characters/words)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 2:
            # Filter out small noise
            valid_components = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= 20:  # Minimum character size
                    valid_components.append(i)
            
            features['num_text_components'] = len(valid_components)
            features['has_many_components'] = len(valid_components) > 50
            
            # Analyze component sizes
            if valid_components:
                component_areas = [stats[i, cv2.CC_STAT_AREA] for i in valid_components]
                features['avg_component_size'] = np.mean(component_areas)
                features['component_size_variance'] = np.var(component_areas)
        else:
            features['num_text_components'] = 0
            features['has_many_components'] = False
            features['avg_component_size'] = 0
            features['component_size_variance'] = 0
        
        # 4. Line structure analysis
        horizontal_kernel = np.ones((1, 20), np.uint8)
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        line_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_lines = []
        for contour in line_contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-6)
                if aspect_ratio > 3:  # Wide lines
                    valid_lines.append((w, h))
        
        features['num_text_lines'] = len(valid_lines)
        features['has_clear_lines'] = len(valid_lines) > 3
        
        # 5. Reading direction reliability indicators
        features['reading_reliability'] = self._assess_reading_direction_reliability(binary, features)
        
        # 6. Method conflict analysis
        features['has_method_conflicts'] = False  # Will be set in consensus
        
        # 7. Document type classification
        features['document_type'] = self._classify_document_type(features)
        
        return features
    
    def _assess_reading_direction_reliability(self, binary_image, features):
        """Assess how reliable the reading direction method is for this image"""
        
        height, width = binary_image.shape
        
        reliability_score = 0.5  # Start with neutral
        
        # 1. Text line consistency - reading direction works best with clear lines
        if features['has_clear_lines']:
            reliability_score += 0.2
        
        # 2. Component distribution - reading direction works well with well-distributed text
        if features['has_many_components'] and not features['is_text_heavy']:
            reliability_score += 0.2
        
        # 3. Aspect ratio - reading direction works better on documents vs. square images
        if 0.7 <= features['aspect_ratio'] <= 1.5:  # Document-like aspect ratios
            reliability_score += 0.1
        
        # 4. Large scanned documents - reading direction excels here
        if features['is_large'] and features['text_density'] < 0.05:
            reliability_score += 0.3
        
        # 5. Text flow analysis - check if text appears to flow horizontally
        horizontal_flow_score = self._measure_horizontal_text_flow(binary_image)
        reliability_score += 0.2 * horizontal_flow_score
        
        return min(reliability_score, 1.0)
    
    def _measure_horizontal_text_flow(self, binary_image):
        """Measure how much the text flows horizontally (0=vertical, 1=horizontal)"""
        
        height, width = binary_image.shape
        
        # Calculate horizontal vs vertical projection strengths
        horizontal_projection = np.sum(binary_image, axis=1)  # Sum across width
        vertical_projection = np.sum(binary_image, axis=0)    # Sum across height
        
        # Variance indicates how "striped" the projection is
        h_variance = np.var(horizontal_projection) if len(horizontal_projection) > 1 else 0
        v_variance = np.var(vertical_projection) if len(vertical_projection) > 1 else 0
        
        if h_variance + v_variance == 0:
            return 0.5
        
        # Higher horizontal variance = more horizontal text flow
        horizontal_ratio = h_variance / (h_variance + v_variance)
        return horizontal_ratio
    
    def _classify_document_type(self, features):
        """Classify the document type based on features"""
        
        if features['is_large'] and features['text_density'] < 0.1:
            return 'scanned_document'  # Large, low density (like PDFs)
        elif features['has_clear_lines'] and features['has_many_components']:
            return 'dense_text'  # Clear text structure
        elif features['text_density'] > 0.2:
            return 'dense_content'  # High text density
        elif features['num_text_components'] < 20:
            return 'sparse_text'  # Few text elements
        else:
            return 'mixed_content'
    
    def _smart_consensus_with_features(self, orientations, features, filename):
        """Smart consensus that selects the best method based purely on image features (filename-blind)"""
        
        # orientations = [text_structure, character_patterns, reading_direction, zebra_stripes]
        method_names = ['text_structure', 'character_patterns', 'reading_direction', 'zebra_stripes']
        
        self.logger.info(f"   🧠 Smart consensus for {filename} ({features['document_type']})")
        self.logger.info(f"   📊 Methods: {dict(zip(method_names, orientations))}")
        self.logger.info(f"   📈 Reading reliability: {features['reading_reliability']:.3f}")
        
        from collections import Counter
        orientation_counts = Counter(orientations)
        most_common = orientation_counts.most_common(1)[0][0]
        max_count = orientation_counts[most_common]
        
        # NEW RULE 0: High reading direction reliability override (with conflict check)
        if features['reading_reliability'] > 0.9:  # Increased threshold for automatic trust
            reading_direction_result = orientations[2]  # reading_direction method
            # Check if reading_direction conflicts with 2+ other methods
            reading_conflicts = sum(1 for o in orientations[:2] + orientations[3:] if o != reading_direction_result)
            if reading_conflicts >= 2:
                self.logger.info(f"   ⚠️  High reading reliability ({features['reading_reliability']:.3f}) but conflicts with {reading_conflicts} methods → proceeding to consensus")
            else:
                self.logger.info(f"   🎯 Very high reading reliability ({features['reading_reliability']:.3f}) → using reading_direction: {reading_direction_result}°")
                return reading_direction_result
        
        # RULE 1: Strong consensus (3+ methods agree) - but check for reading direction override
        if max_count >= 3:
            # Special case: if reading direction says 0° with high reliability, be very cautious about rotating
            if orientations[2] == 0 and features['reading_reliability'] > 0.95 and most_common != 0:
                self.logger.info(f"   ⚠️  Strong consensus for {most_common}° but reading_direction (0°) has very high reliability ({features['reading_reliability']:.3f}) → staying at 0°")
                return 0
            self.logger.info(f"   🎯 Strong consensus: {most_common}° ({max_count}/4 methods agree)")
            return most_common
        
        # RULE 2: Moderate consensus (2 methods agree) - check method reliability
        elif max_count >= 2:
            # Check which 2 methods agree and prioritize reliable combinations
            agreeing_methods = []
            for i, orientation in enumerate(orientations):
                if orientation == most_common:
                    agreeing_methods.append(method_names[i])
            
            self.logger.info(f"   🎯 Moderate consensus: {most_common}° (methods: {agreeing_methods})")
            
            # Prioritize certain method combinations based on reliability
            if 'text_structure' in agreeing_methods and 'reading_direction' in agreeing_methods:
                self.logger.info(f"   ✅ Reliable combination: text_structure + reading_direction")
                return most_common
            elif 'character_patterns' in agreeing_methods and 'reading_direction' in agreeing_methods:
                self.logger.info(f"   ✅ Reliable combination: character_patterns + reading_direction")
                return most_common
            elif 'text_structure' in agreeing_methods and 'character_patterns' in agreeing_methods:
                self.logger.info(f"   ✅ Reliable combination: text_structure + character_patterns")
                return most_common
            else:
                self.logger.info(f"   ⚠️  Moderate consensus accepted")
                return most_common
        
        # RULE 3: No clear consensus - use feature-based method selection
        else:
            self.logger.info(f"   🔍 No consensus, using feature-based method selection")
            
            # Check if reading_direction has medium+ reliability in conflict situations (more conservative)
            if features['reading_reliability'] > 0.75 and orientations[2] == 0:  # Only trust reading_direction for "no rotation" when highly reliable
                self.logger.info(f"   🛡️  High reading reliability ({features['reading_reliability']:.3f}) for no-rotation: Using reading_direction method")
                return orientations[2]  # reading_direction
            
            # For large scanned documents, trust reading_direction
            elif features['document_type'] == 'scanned_document':
                self.logger.info(f"   🛡️  Scanned document: Using reading_direction method")
                return orientations[2]  # reading_direction
            
            # For documents with clear text structure, trust text_structure
            elif features['has_clear_lines'] and features['has_many_components']:
                self.logger.info(f"   🛡️  Clear text structure: Using text_structure method")
                return orientations[0]  # text_structure
            
            # For dense content, trust character_patterns
            elif features['is_text_heavy'] and features['has_many_components']:
                self.logger.info(f"   🛡️  Dense text content: Using character_patterns method")
                return orientations[1]  # character_patterns
            
            # For documents with good line structure, trust reading_direction
            elif features['has_clear_lines']:
                self.logger.info(f"   🛡️  Good line structure: Using reading_direction method")
                return orientations[2]  # reading_direction
            
            # Default: trust reading_direction for most documents (it's generally most reliable)
            elif features['reading_reliability'] > 0.4:
                self.logger.info(f"   🛡️  Default to reading_direction (reliability: {features['reading_reliability']:.3f})")
                return orientations[2]  # reading_direction
            
            # Last resort: no rotation if reading_direction also seems unreliable
            else:
                self.logger.info(f"   🛡️  Conservative fallback: No rotation (low reading reliability: {features['reading_reliability']:.3f})")
                return 0
    
    def _fast_reading_score(self, binary_image):
        """Calculate reading direction score (original fast algorithm)"""
        
        height, width = binary_image.shape
        
        # 1. Text flow analysis
        # Find text lines
        horizontal_kernel = np.ones((1, 20), np.uint8)
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count and analyze lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_scores = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-6)
                
                if aspect_ratio > 2:  # Wide lines
                    # Calculate line quality
                    line_mask = np.zeros_like(binary_image)
                    cv2.drawContours(line_mask, [contour], -1, 255, -1)
                    
                    # Text density in line
                    text_pixels = np.sum(cv2.bitwise_and(binary_image, line_mask) > 0)
                    line_density = text_pixels / area
                    
                    line_score = line_density * min(aspect_ratio, 8.0) / 8.0
                    line_scores.append(line_score)
        
        avg_line_score = np.mean(line_scores) if line_scores else 0
        
        # 2. Character alignment
        # Find characters
        char_contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        alignment_score = 0
        if len(char_contours) > 3:
            # Get character centers
            centers = []
            for contour in char_contours:
                area = cv2.contourArea(contour)
                if 20 <= area <= 2000:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centers.append((cx, cy))
            
            if len(centers) > 2:
                # Group characters by line
                centers.sort(key=lambda c: c[1])  # Sort by y
                
                line_groups = []
                current_line = [centers[0]]
                current_y = centers[0][1]
                
                for center in centers[1:]:
                    if abs(center[1] - current_y) < 15:  # Same line
                        current_line.append(center)
                    else:
                        if len(current_line) > 1:
                            line_groups.append(current_line)
                        current_line = [center]
                        current_y = center[1]
                
                if len(current_line) > 1:
                    line_groups.append(current_line)
                
                # Calculate alignment quality
                alignment_scores = []
                for line in line_groups:
                    if len(line) > 1:
                        # Sort by x-coordinate
                        line.sort(key=lambda c: c[0])
                        
                        # Calculate horizontal alignment
                        x_coords = [c[0] for c in line]
                        x_variance = np.var(x_coords)
                        
                        # Calculate vertical consistency
                        y_coords = [c[1] for c in line]
                        y_variance = np.var(y_coords)
                        
                        # Better alignment = lower variance
                        line_alignment = 1.0 / (1.0 + x_variance/100 + y_variance/100)
                        alignment_scores.append(line_alignment)
                
                if alignment_scores:
                    alignment_score = np.mean(alignment_scores)
        
        # 3. Text density distribution
        # Divide image into regions and analyze density
        regions = 6
        region_width = width // regions
        region_densities = []
        
        for i in range(regions):
            start_x = i * region_width
            end_x = min((i + 1) * region_width, width)
            region = binary_image[:, start_x:end_x]
            density = np.sum(region > 0) / (height * (end_x - start_x))
            region_densities.append(density)
        
        # Text should have consistent density across regions
        density_consistency = 1.0 / (1.0 + np.var(region_densities))
        
        # Combined reading score
        reading_score = (
            0.4 * avg_line_score +
            0.4 * alignment_score +
            0.2 * density_consistency
        )
        
        return reading_score
    
    def _calculate_reading_score(self, binary_image):
        """Calculate reading direction score"""
        
        height, width = binary_image.shape
        
        # 1. Text flow analysis
        # Find text lines
        horizontal_kernel = np.ones((1, 20), np.uint8)
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count and analyze lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_scores = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-6)
                
                if aspect_ratio > 2:  # Wide lines
                    # Calculate line quality
                    line_mask = np.zeros_like(binary_image)
                    cv2.drawContours(line_mask, [contour], -1, 255, -1)
                    
                    # Text density in line
                    text_pixels = np.sum(cv2.bitwise_and(binary_image, line_mask) > 0)
                    line_density = text_pixels / area
                    
                    line_score = line_density * min(aspect_ratio, 8.0) / 8.0
                    line_scores.append(line_score)
        
        avg_line_score = np.mean(line_scores) if line_scores else 0
        
        # 2. Character alignment
        # Find characters
        char_contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        alignment_score = 0
        if len(char_contours) > 3:
            # Get character centers
            centers = []
            for contour in char_contours:
                area = cv2.contourArea(contour)
                if 20 <= area <= 2000:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centers.append((cx, cy))
            
            if len(centers) > 2:
                # Group characters by line
                centers.sort(key=lambda c: c[1])  # Sort by y
                
                line_groups = []
                current_line = [centers[0]]
                current_y = centers[0][1]
                
                for center in centers[1:]:
                    if abs(center[1] - current_y) < 15:  # Same line
                        current_line.append(center)
                    else:
                        if len(current_line) > 1:
                            line_groups.append(current_line)
                        current_line = [center]
                        current_y = center[1]
                
                if len(current_line) > 1:
                    line_groups.append(current_line)
                
                # Calculate alignment quality
                alignment_scores = []
                for line in line_groups:
                    if len(line) > 1:
                        # Sort by x-coordinate
                        line.sort(key=lambda c: c[0])
                        
                        # Calculate horizontal alignment
                        x_coords = [c[0] for c in line]
                        x_variance = np.var(x_coords)
                        
                        # Calculate vertical consistency
                        y_coords = [c[1] for c in line]
                        y_variance = np.var(y_coords)
                        
                        # Better alignment = lower variance
                        line_alignment = 1.0 / (1.0 + x_variance/100 + y_variance/100)
                        alignment_scores.append(line_alignment)
                
                if alignment_scores:
                    alignment_score = np.mean(alignment_scores)
        
        # 3. Text density distribution
        # Divide image into regions and analyze density
        regions = 6
        region_width = width // regions
        region_densities = []
        
        for i in range(regions):
            start_x = i * region_width
            end_x = min((i + 1) * region_width, width)
            region = binary_image[:, start_x:end_x]
            density = np.sum(region > 0) / (height * (end_x - start_x))
            region_densities.append(density)
        
        # Text should have consistent density across regions
        density_consistency = 1.0 / (1.0 + np.var(region_densities))
        
        # Combined reading score
        reading_score = (
            0.4 * avg_line_score +
            0.4 * alignment_score +
            0.2 * density_consistency
        )
        
        return reading_score
    
    def _ml_consensus(self, orientations):
        """Smart conservative consensus that carefully balances accuracy vs safety"""
        
        # Debug logging
        self.logger.info(f"   🔍 Debug: orientations={orientations}")
        
        from collections import Counter
        orientation_counts = Counter(orientations)
        most_common = orientation_counts.most_common(1)[0][0]
        
        # SMART RULE 1: If 3+ methods agree on the SAME non-zero angle, and no method says 0°, trust it
        if orientation_counts[most_common] >= 3 and most_common != 0:
            zero_count = orientation_counts.get(0, 0)
            if zero_count == 0:  # No method thinks it's already correct
                self.logger.info(f"   ✅ Strong consensus for rotation: {most_common}° ({orientation_counts[most_common]}/4 methods agree, no 0° votes)")
                return most_common
        
        # SMART RULE 1B: If 2+ methods agree on the SAME non-zero angle, and zebra stripes also agrees, trust it
        if orientation_counts[most_common] >= 2 and most_common != 0:
            if len(orientations) >= 4 and orientations[3] == most_common:  # Zebra stripes agrees
                self.logger.info(f"   ✅ Good consensus with zebra confirmation: {most_common}° ({orientation_counts[most_common]}/4 methods agree, zebra stripes confirms)")
                return most_common
        
        # SMART RULE 2: If there's a 4-way split with different angles, be conservative  
        if len(orientation_counts) == 4:  # All methods disagree
            self.logger.info(f"   🛡️  Conservative: All methods disagree → staying at 0°")
            return 0
            
        # SMART RULE 3: If any method says 0° and there's disagreement about rotation direction, stay at 0°
        if 0 in orientations:
            # Check if the non-zero orientations agree on direction
            non_zero_orientations = [o for o in orientations if o != 0]
            if len(set(non_zero_orientations)) > 1:  # Multiple different non-zero angles
                self.logger.info(f"   🛡️  Conservative: Method disagreement with 0° vote → staying at 0°")
                return 0
        
        # SMART RULE 4: Special case for 180° rotation - be extra careful
        if most_common == 180 and orientation_counts[most_common] == 2:
            zero_count = orientation_counts.get(0, 0)
            if zero_count == 1:  # One method thinks it's already correct
                self.logger.info(f"   🛡️  Conservative: Uncertain about 180° rotation (1 method says 0°) → staying at 0°")
                return 0
        
        # SMART RULE 5: If we get here and there's still majority agreement, trust it
        if orientation_counts[most_common] >= 2:
            self.logger.info(f"   ✅ Majority consensus: {most_common}° ({orientation_counts[most_common]}/4 methods agree)")
            return most_common
        
        # SMART RULE 6: Default to 0° if no clear pattern
        self.logger.info(f"   🛡️  Conservative: No clear consensus → defaulting to 0° (no rotation)")
        return 0
    
    def _fallback_text_structure_analysis(self, gray):
        """Fallback text structure analysis using heuristics"""
        
        # Resize for speed if image is too large
        max_dim = 800
        if max(gray.shape) > max_dim:
            scale = max_dim / max(gray.shape)
            new_width = int(gray.shape[1] * scale)
            new_height = int(gray.shape[0] * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        height, width = gray.shape
        
        # Create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Fast orientation test: only test 0° and 180° (most common cases)
        angles = [0, 180]
        best_angle = 0
        best_structure_score = 0
        
        for angle in angles:
            if angle == 0:
                rotated = binary
            else:
                # Fast 180° rotation (flip both axes)
                rotated = cv2.flip(cv2.flip(binary, 0), 1)
            
            # Fast structure score calculation
            structure_score = self._fast_structure_score(rotated)
            
            if structure_score > best_structure_score:
                best_structure_score = structure_score
                best_angle = angle
        
        return best_angle
    
    def _test_orientation_quality(self, image, orientation):
        """Fast orientation quality testing (optimized for speed)"""
        
        # Resize image for speed if too large
        max_dim = 600
        if max(image.shape[:2]) > max_dim:
            scale = max_dim / max(image.shape[:2])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        if orientation == 0:
            rotated_image = image.copy()
        else:
            # Fast 180° rotation (flip both axes)
            rotated_image = cv2.flip(cv2.flip(image, 0), 1)
        
        # Convert to grayscale for text analysis
        if len(rotated_image.shape) == 3:
            gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rotated_image
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        quality_score = 0
        
        # 1. Fast text line spacing consistency
        line_spacing_score = self._fast_line_spacing_consistency(binary)
        quality_score += line_spacing_score * 0.4
        
        # 2. Fast character aspect ratio
        aspect_ratio_score = self._fast_character_aspect_ratios(binary)
        quality_score += aspect_ratio_score * 0.3
        
        # 3. Fast text structure regularity
        structure_score = self._fast_text_structure_regularity(binary)
        quality_score += structure_score * 0.3
        
        return quality_score
    
    def _fast_line_spacing_consistency(self, binary_image):
        """Fast line spacing consistency measurement (optimized for speed)"""
        
        # Find horizontal projection profile
        horizontal_projection = np.sum(binary_image, axis=1)
        
        # Find text lines (rows with text) - use threshold
        threshold = np.mean(horizontal_projection)
        text_rows = np.where(horizontal_projection > threshold)[0]
        
        if len(text_rows) < 2:
            return 0.0
        
        # Calculate gaps between consecutive text rows (limited for speed)
        gaps = []
        max_gaps = min(20, len(text_rows) - 1)  # Limit analysis
        
        for i in range(1, max_gaps + 1):
            gap = text_rows[i] - text_rows[i-1]
            if gap > 1:  # Only count actual gaps, not consecutive rows
                gaps.append(gap)
        
        if len(gaps) < 2:
            return 0.5  # Neutral score if not enough gaps
        
        # Measure consistency: lower variance = more consistent = better
        gap_variance = np.var(gaps)
        gap_mean = np.mean(gaps)
        
        if gap_mean == 0:
            return 0.0
        
        # Normalize variance relative to mean gap size
        normalized_variance = gap_variance / (gap_mean ** 2)
        consistency_score = 1.0 / (1.0 + normalized_variance)
        
        return min(consistency_score, 1.0)
    
    def _measure_line_spacing_consistency(self, binary_image):
        """Measure consistency of gaps between text baselines"""
        
        # Find horizontal projection profile
        horizontal_projection = np.sum(binary_image, axis=1)
        
        # Find text lines (rows with text)
        text_rows = np.where(horizontal_projection > np.mean(horizontal_projection))[0]
        
        if len(text_rows) < 2:
            return 0.0
        
        # Calculate gaps between consecutive text rows
        gaps = []
        for i in range(1, len(text_rows)):
            gap = text_rows[i] - text_rows[i-1]
            if gap > 1:  # Only count actual gaps, not consecutive rows
                gaps.append(gap)
        
        if len(gaps) < 2:
            return 0.5  # Neutral score if not enough gaps
        
        # Measure consistency: lower variance = more consistent = better
        gap_variance = np.var(gaps)
        gap_mean = np.mean(gaps)
        
        if gap_mean == 0:
            return 0.0
        
        # Normalize variance relative to mean gap size
        normalized_variance = gap_variance / (gap_mean ** 2)
        consistency_score = 1.0 / (1.0 + normalized_variance)
        
        return min(consistency_score, 1.0)
    
    def _fast_character_aspect_ratios(self, binary_image):
        """Fast character aspect ratio measurement (optimized for speed)"""
        
        # Find connected components (characters) - limited for speed
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        if num_labels < 2:
            return 0.0
        
        # Analyze character aspect ratios (limited sample)
        aspect_ratios = []
        min_area = 30  # Lower threshold for speed
        max_components = min(50, num_labels - 1)  # Limit analysis
        
        for i in range(1, max_components + 1):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
                
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            if width > 0 and height > 0:
                aspect_ratio = height / width
                aspect_ratios.append(aspect_ratio)
        
        if len(aspect_ratios) < 3:
            return 0.5  # Neutral score if not enough characters
        
        # Count characters that are taller than wide (normal orientation)
        tall_characters = sum(1 for ratio in aspect_ratios if ratio > 1.1)
        total_characters = len(aspect_ratios)
        
        # Score based on percentage of tall characters
        tall_ratio = tall_characters / total_characters
        return tall_ratio
    
    def _measure_character_aspect_ratios(self, binary_image):
        """Measure if characters are taller than wide (normal text orientation)"""
        
        # Find connected components (characters)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        if num_labels < 2:
            return 0.0
        
        # Analyze character aspect ratios
        aspect_ratios = []
        min_area = 50  # Filter out noise
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
                
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            if width > 0 and height > 0:
                aspect_ratio = height / width
                aspect_ratios.append(aspect_ratio)
        
        if len(aspect_ratios) < 5:
            return 0.5  # Neutral score if not enough characters
        
        # Count characters that are taller than wide (normal orientation)
        tall_characters = sum(1 for ratio in aspect_ratios if ratio > 1.2)
        total_characters = len(aspect_ratios)
        
        # Score based on percentage of tall characters
        tall_ratio = tall_characters / total_characters
        return tall_ratio
    
    def _fast_text_structure_regularity(self, binary_image):
        """Fast text structure regularity measurement (optimized for speed)"""
        
        # Find vertical projection profile
        vertical_projection = np.sum(binary_image, axis=0)
        
        # Find text columns (regions with text) - limited for speed
        threshold = np.mean(vertical_projection)
        text_columns = np.where(vertical_projection > threshold)[0]
        
        if len(text_columns) < 5:
            return 0.0
        
        # Measure column consistency (limited sample)
        column_gaps = []
        max_gaps = min(15, len(text_columns) - 1)  # Limit analysis
        
        for i in range(1, max_gaps + 1):
            gap = text_columns[i] - text_columns[i-1]
            if gap > 1:
                column_gaps.append(gap)
        
        if len(column_gaps) < 2:
            return 0.5
        
        # Calculate regularity score
        gap_variance = np.var(column_gaps)
        gap_mean = np.mean(column_gaps)
        
        if gap_mean == 0:
            return 0.0
        
        normalized_variance = gap_variance / (gap_mean ** 2)
        regularity_score = 1.0 / (1.0 + normalized_variance)
        
        return min(regularity_score, 1.0)
    
    def _measure_text_structure_regularity(self, binary_image):
        """Measure regularity of text structure (columns, alignment)"""
        
        # Find vertical projection profile
        vertical_projection = np.sum(binary_image, axis=0)
        
        # Find text columns (regions with text)
        text_columns = np.where(vertical_projection > np.mean(vertical_projection))[0]
        
        if len(text_columns) < 10:
            return 0.0
        
        # Measure column consistency
        # In well-oriented text, columns should be relatively uniform
        column_gaps = []
        for i in range(1, len(text_columns)):
            gap = text_columns[i] - text_columns[i-1]
            if gap > 1:
                column_gaps.append(gap)
        
        if len(column_gaps) < 2:
            return 0.5
        
        # Calculate regularity score
        gap_variance = np.var(column_gaps)
        gap_mean = np.mean(column_gaps)
        
        if gap_mean == 0:
            return 0.0
        
        normalized_variance = gap_variance / (gap_mean ** 2)
        regularity_score = 1.0 / (1.0 + normalized_variance)
        
        return min(regularity_score, 1.0)
    
    def _quality_based_orientation_selection(self, image, orientations):
        """Select orientation based on actual text quality testing with upside-down detection"""
        
        self.logger.info(f"   🔍 Quality-based testing: Testing all orientations")
        
        # Test all possible orientations: 0°, 90°, 180°, 270°
        candidate_orientations = [0, 90, 180, 270]
        quality_scores = {}
        
        for orientation in candidate_orientations:
            quality_score = self._test_orientation_quality(image, orientation)
            quality_scores[orientation] = quality_score
            self.logger.info(f"   🔍 Orientation {orientation}°: quality = {quality_score:.3f}")
        
        # SPECIAL CASE: If we have strong 180° consensus from methods, 
        # but quality testing shows 0° is slightly better, 
        # use a more aggressive upside-down detection
        if orientations.count(180) >= 2:  # Strong 180° consensus
            self.logger.info(f"   🔍 Strong 180° consensus detected, using aggressive upside-down detection")
            
            # Test specifically for upside-down text patterns
            upside_down_score = self._test_upside_down_text(image)
            right_side_up_score = self._test_right_side_up_text(image)
            
            self.logger.info(f"   🔍 Upside-down detection: 180°={upside_down_score:.3f}, 0°={right_side_up_score:.3f}")
            
            # If upside-down detection is significantly better, trust it
            if upside_down_score > right_side_up_score * 1.2:  # 20% better
                self.logger.info(f"   ✅ Upside-down detection confirms 180° rotation needed")
                return 180
        
        # Find the orientation with highest quality score
        best_orientation = max(quality_scores, key=quality_scores.get)
        best_score = quality_scores[best_orientation]
        
        self.logger.info(f"   ✅ Best orientation: {best_orientation}° (quality: {best_score:.3f})")
        return best_orientation
    
    def _apply_ml_rotation_correction(self, image, angle):
        """Apply ML-based rotation correction"""
        
        if angle == 0:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Rotate in the opposite direction to correct orientation
        correction_angle = -angle
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
        
        # Calculate new dimensions
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation with border handling
        corrected_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        return corrected_image
    
    def _create_comparison_image(self, original, result, filename, 
                                final_orientation, orientations):
        """Create ML-based comparison visualization"""
        
        # Create a simple side-by-side comparison with OpenCV
        height, width = original.shape[:2]
        
        # Ensure both images have the same height for comparison
        if result.shape[0] != height:
            scale = height / result.shape[0]
            new_w = int(result.shape[1] * scale)
            result_resized = cv2.resize(result, (new_w, height), interpolation=cv2.INTER_AREA)
        else:
            result_resized = result
        
        # Create comparison image
        comparison = np.hstack([original, result_resized])
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)  # Green text
        
        # Original image label
        cv2.putText(comparison, "Original", (50, 50), font, font_scale, color, thickness)
        
        # Result image label
        cv2.putText(comparison, "Oriented", (width + 50, 50), font, font_scale, color, thickness)
        
        # Add analysis info
        info_text = f"Text structure: {orientations[0]}° | Character patterns: {orientations[1]}° | Reading direction: {orientations[2]}° | Zebra stripes: {orientations[3]}°"
        cv2.putText(comparison, info_text, (50, height - 30), font, 0.6, (255, 255, 255), 1)
        
        # Add final orientation info
        orientation_text = f"Final orientation: {final_orientation}° | Rotation applied: {-final_orientation}° counterclockwise"
        cv2.putText(comparison, orientation_text, (50, height - 10), font, 0.6, (255, 255, 255), 1)
        
        return comparison
    
    def _test_upside_down_text(self, image):
        """Test if image contains upside-down text patterns"""
        
        # Resize for speed
        max_dim = 600
        if max(image.shape[:2]) > max_dim:
            scale = max_dim / max(image.shape[:2])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        height, width = binary.shape
        
        # 1. Check if text baseline is at the top (indicating upside-down)
        # Divide image into upper and lower thirds
        upper_third = binary[:height//3, :]
        middle_third = binary[height//3:2*height//3, :]
        lower_third = binary[2*height//3:, :]
        
        upper_density = np.sum(upper_third > 0) / (upper_third.shape[0] * upper_third.shape[1])
        middle_density = np.sum(middle_third > 0) / (middle_third.shape[0] * middle_third.shape[1])
        lower_density = np.sum(lower_third > 0) / (lower_third.shape[0] * lower_third.shape[1])
        
        # In upside-down text, upper third should have more text than lower third
        baseline_score = upper_density / (lower_density + 1e-6)
        
        # 2. Check character descender patterns (letters like 'g', 'j', 'p', 'q', 'y')
        # These should point downward in upside-down text
        descender_score = self._detect_descender_patterns(binary)
        
        # 3. Check text line spacing (should be more irregular in upside-down)
        line_spacing_score = self._measure_line_spacing_irregularity(binary)
        
        # Combine scores
        upside_down_score = 0.4 * baseline_score + 0.3 * descender_score + 0.3 * line_spacing_score
        return min(upside_down_score, 1.0)
    
    def _test_right_side_up_text(self, image):
        """Test if image contains right-side-up text patterns"""
        
        # Resize for speed
        max_dim = 600
        if max(image.shape[:2]) > max_dim:
            scale = max_dim / max(image.shape[:2])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        height, width = binary.shape
        
        # 1. Check if text baseline is at the bottom (indicating right-side-up)
        # Divide image into upper and lower thirds
        upper_third = binary[:height//3, :]
        middle_third = binary[height//3:2*height//3, :]
        lower_third = binary[2*height//3:, :]
        
        upper_density = np.sum(upper_third > 0) / (upper_third.shape[0] * upper_third.shape[1])
        middle_density = np.sum(middle_third > 0) / (middle_third.shape[0] * middle_third.shape[1])
        lower_density = np.sum(lower_third > 0) / (lower_third.shape[0] * lower_third.shape[1])
        
        # In right-side-up text, lower third should have more text than upper third
        baseline_score = lower_density / (upper_density + 1e-6)
        
        # 2. Check character ascender patterns (letters like 'b', 'd', 'f', 'h', 'k', 'l', 't')
        # These should point upward in right-side-up text
        ascender_score = self._detect_ascender_patterns(binary)
        
        # 3. Check text line spacing (should be more regular in right-side-up)
        line_spacing_score = 1.0 - self._measure_line_spacing_irregularity(binary)
        
        # Combine scores
        right_side_up_score = 0.4 * baseline_score + 0.3 * ascender_score + 0.3 * line_spacing_score
        return min(right_side_up_score, 1.0)
    
    def _detect_descender_patterns(self, binary_image):
        """Detect patterns that suggest upside-down text (descenders pointing down)"""
        
        height, width = binary_image.shape
        
        # Look for connected components that extend into the lower part of the image
        # These could be descenders like 'g', 'j', 'p', 'q', 'y'
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        if num_labels < 2:
            return 0.5
        
        descender_count = 0
        total_count = 0
        
        for i in range(1, min(num_labels, 50)):  # Limit for speed
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:  # Filter noise
                continue
            
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Check if component extends into lower third of image
            if y + h > 2 * height // 3:
                descender_count += 1
            total_count += 1
        
        if total_count == 0:
            return 0.5
        
        return descender_count / total_count
    
    def _detect_ascender_patterns(self, binary_image):
        """Detect patterns that suggest right-side-up text (ascenders pointing up)"""
        
        height, width = binary_image.shape
        
        # Look for connected components that extend into the upper part of the image
        # These could be ascenders like 'b', 'd', 'f', 'h', 'k', 'l', 't'
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        if num_labels < 2:
            return 0.5
        
        ascender_count = 0
        total_count = 0
        
        for i in range(1, min(num_labels, 50)):  # Limit for speed
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:  # Filter noise
                continue
            
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Check if component extends into upper third of image
            if y < height // 3:
                ascender_count += 1
            total_count += 1
        
        if total_count == 0:
            return 0.5
        
        return ascender_count / total_count
    
    def _measure_line_spacing_irregularity(self, binary_image):
        """Measure irregularity of line spacing (higher = more irregular)"""
        
        # Find horizontal projection profile
        horizontal_projection = np.sum(binary_image, axis=1)
        
        # Find text lines (rows with text)
        threshold = np.mean(horizontal_projection)
        text_rows = np.where(horizontal_projection > threshold)[0]
        
        if len(text_rows) < 3:
            return 0.5
        
        # Calculate gaps between consecutive text rows
        gaps = []
        for i in range(1, min(len(text_rows), 20)):  # Limit for speed
            gap = text_rows[i] - text_rows[i-1]
            if gap > 1:
                gaps.append(gap)
        
        if len(gaps) < 2:
            return 0.5
        
        # Measure irregularity: higher variance = more irregular
        gap_variance = np.var(gaps)
        gap_mean = np.mean(gaps)
        
        if gap_mean == 0:
            return 0.5
        
        # Normalize variance relative to mean gap size
        normalized_variance = gap_variance / (gap_mean ** 2)
        irregularity_score = min(normalized_variance / 2.0, 1.0)  # Cap at 1.0
        
        return irregularity_score
    
    def get_task_info(self):
        """Get information about this task"""
        
        return {
            'task_id': self.task_id,
            'name': self.task_name,
            'description': 'Detect and correct upside-down or sideways pages using ML-based readability analysis',
            'order': 3,
            'dependencies': ['task_2_cropping'],
            'output_format': 'png',
            'supported_inputs': ['image', 'pdf'],
            'status': 'ready'
        }

# Standalone execution for testing
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test the task
    task = OrientationCorrectionTask(logger)
    
    print("🧪 Testing Orientation Correction Task")
    print("=" * 40)
    
    # Test with a sample image
    test_image = "input/test_image.png"
    if os.path.exists(test_image):
        result = task.run(test_image, 'image', 'output')
        if result:
            print(f"✅ Task completed successfully!")
            print(f"   Output: {result['output']}")
            print(f"   Final orientation: {result['final_orientation']}°")
        else:
            print("❌ Task failed!")
    else:
        print(f"⚠️  Test image not found: {test_image}")
        print("   Create an input folder with test images to test")
