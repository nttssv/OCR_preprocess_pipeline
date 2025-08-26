#!/usr/bin/env python3
"""
Task 1: Skew Detection & Correction
Contains the actual skew detection code copied from 1.final_skew_detector/skew_detector.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from scipy.ndimage import rotate as scipy_rotate
import csv
import time
from datetime import datetime
import logging

class SkewDetectionTask:
    """Task 1: Skew Detection and Correction - Contains actual algorithm"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Skew Detection & Correction"
        self.task_id = "task_2_skew_detection"
        
    def run(self, input_file, file_type, output_folder):
        """
        Run skew detection and correction on the input file
        
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
            task_output = os.path.join(output_folder, "task2_skew_detection")
            os.makedirs(task_output, exist_ok=True)
            
            # Process based on file type
            if file_type == 'pdf':
                result = self._process_pdf_skew_detection(input_file, task_output)
            else:
                result = self._process_image_skew_detection(input_file, task_output)
            
            if result:
                self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
                return result
            else:
                self.logger.error(f"❌ {self.task_name} failed for {os.path.basename(input_file)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in {self.task_name}: {str(e)}")
            return None
    
    def _process_image_skew_detection(self, image_path, output_folder):
        """Process single image for skew detection using the actual algorithm"""
        
        try:
            filename = os.path.basename(image_path)
            self.logger.info(f"📄 Processing: {filename}")
            
            # Load image
            start_time = time.time()
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to load image")
            self.logger.info(f"✓ Image loaded: {img.shape}")
            load_time = time.time() - start_time
            
            # Preprocess
            start_time = time.time()
            binary = self._adaptive_preprocessing_optimized(img)
            preprocess_time = time.time() - start_time
            self.logger.info(f"✓ Preprocessing complete ({preprocess_time:.2f}s)")
            
            # Detect skew with optimized method
            start_time = time.time()
            angle, method = self._detect_skew_final_15deg_optimized(binary, step=0.2, angle_range=15)
            detection_time = time.time() - start_time
            self.logger.info(f"🎯 Final detected: {angle:+.2f}° ({method})")
            self.logger.info(f"⏱️  Detection time: {detection_time:.2f}s")
            
            # Deskew image
            start_time = time.time()
            deskewed = self._deskew_image(img, -angle)  # Negative angle for correction
            deskew_time = time.time() - start_time
            self.logger.info(f"✓ Deskewing complete ({deskew_time:.2f}s)")
            
            # Calculate total time
            total_time = load_time + preprocess_time + detection_time + deskew_time
            
            # Generate outputs
            base_name = os.path.splitext(filename)[0]
            
            # Save deskewed image
            deskewed_path = os.path.join(output_folder, f"{base_name}_deskewed.png")
            cv2.imwrite(deskewed_path, deskewed)
            self.logger.info(f"💾 Deskewed image saved: {deskewed_path}")
            
            # Create and save comparison image
            comparison = self._create_comparison_image(img, deskewed, filename, angle, total_time)
            comparison_path = os.path.join(output_folder, f"{base_name}_comparison.png")
            cv2.imwrite(comparison_path, comparison)
            self.logger.info(f"💾 Comparison image saved: {comparison_path}")
            
            # Save processing summary
            summary_path = os.path.join(output_folder, "processing_summary.csv")
            self._save_processing_summary(filename, angle, method, total_time, summary_path)
            
            return {
                'input': image_path,
                'output': deskewed_path,
                'comparison': comparison_path,
                'angle': angle,
                'method': method,
                'status': 'completed',
                'task': self.task_id,
                'processing_time': total_time
            }
                
        except Exception as e:
            self.logger.error(f"Error processing image for skew detection: {str(e)}")
            return None
    
    def _process_pdf_skew_detection(self, pdf_path, output_folder):
        """Process PDF for skew detection"""
        
        try:
            self.logger.info("📄 Processing PDF for skew detection")
            
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
            result = self._process_image_skew_detection(converted_path, output_folder)
            
            if result:
                # Update the result to reflect it came from PDF
                result['input'] = pdf_path
                result['note'] = 'PDF converted to image and processed'
                return result
            else:
                raise Exception("Failed to process converted PDF image")
            
        except Exception as e:
            self.logger.error(f"Error processing PDF for skew detection: {str(e)}")
            return None
    
    def _detect_skew_final_15deg_optimized(self, binary, step=0.2, angle_range=15):
        """OPTIMIZED skew detection with adaptive step sizing for speed"""
        self.logger.info(f"🎯 Optimized ±15° skew detection (step={step}°)...")
        
        # Method 1: Adaptive projection analysis (faster)
        projection_angle, projection_results = self._detect_skew_projection_optimized(binary, step, angle_range)
        
        # Method 2: Single-pass Hough (faster)
        hough_angle, hough_confidence = self._detect_skew_hough_optimized(binary)
        
        # Method 3: Fast text line analysis (faster)
        text_angle, text_confidence = self._detect_skew_text_lines_optimized(binary)
        
        self.logger.info(f"   📊 Method results:")
        self.logger.info(f"      Projection: {projection_angle:+.2f}°")
        self.logger.info(f"      Hough:      {hough_angle:+.2f}° (conf: {hough_confidence})")
        self.logger.info(f"      Text lines: {text_angle:+.2f}° (conf: {text_confidence})")
        
        # Fast intelligent combination
        raw_angle = self._intelligent_combination_optimized(
            projection_angle, projection_results,
            hough_angle, hough_confidence,
            text_angle, text_confidence
        )
        
        # CRITICAL: Apply sign correction to match expected convention
        corrected_angle = -raw_angle
        
        self.logger.info(f"   🔄 Sign correction: {raw_angle:+.2f}° → {corrected_angle:+.2f}°")
        
        return corrected_angle, "final_15deg_optimized"
    
    def _detect_skew_projection_optimized(self, binary, step=0.2, angle_range=15):
        """OPTIMIZED projection analysis with adaptive step sizing"""
        best_angle = 0
        best_variance = 0
        all_results = []
        
        # OPTIMIZATION 1: Use larger step initially for coarse detection
        coarse_step = max(step, 0.5)  # Minimum 0.5° step for speed
        coarse_angles = np.arange(-angle_range, angle_range + coarse_step, coarse_step)
        
        # First pass: Coarse detection
        coarse_results = []
        for angle in coarse_angles:
            rotated = scipy_rotate(binary, angle, reshape=False, order=1)
            horizontal_projection = np.sum(rotated, axis=1)
            variance = np.var(horizontal_projection)
            
            coarse_results.append((angle, variance))
            if variance > best_variance:
                best_variance = variance
                best_angle = angle
        
        # OPTIMIZATION 2: Fine-tune around best coarse angle
        if abs(best_angle) > 0.1:  # If we found a significant angle
            fine_range = 1.0  # ±1° around best angle
            fine_step = min(step, 0.1)  # Use original step or 0.1°, whichever is smaller
            
            fine_start = max(-angle_range, best_angle - fine_range)
            fine_end = min(angle_range, best_angle + fine_range)
            fine_angles = np.arange(fine_start, fine_end + fine_step, fine_step)
            
            # Second pass: Fine detection around best angle
            for angle in fine_angles:
                rotated = scipy_rotate(binary, angle, reshape=False, order=1)
                horizontal_projection = np.sum(rotated, axis=1)
                variance = np.var(horizontal_projection)
                
                all_results.append((angle, variance))
                if variance > best_variance:
                    best_variance = variance
                    best_angle = angle
        else:
            # If no significant angle found, use coarse results
            all_results = coarse_results
        
        # OPTIMIZATION 3: Simplified cluster detection (faster)
        if len(all_results) > 10:
            # Sort by variance and look for top results
            all_results.sort(key=lambda x: x[1], reverse=True)
            top_angles = [result[0] for result in all_results[:10]]
            top_variances = [result[1] for result in all_results[:10]]
            
            # Simple clustering: group angles within 0.5° of each other
            clusters = []
            used_indices = set()
            
            for i, angle in enumerate(top_angles):
                if i in used_indices:
                    continue
                    
                cluster = [angle]
                used_indices.add(i)
                
                for j, other_angle in enumerate(top_angles[i+1:], i+1):
                    if j not in used_indices and abs(angle - other_angle) < 0.5:
                        cluster.append(other_angle)
                        used_indices.add(j)
                
                if len(cluster) >= 2:
                    clusters.append((np.mean(cluster), np.mean([top_variances[top_angles.index(a)] for a in cluster]), len(cluster)))
            
            # Use significant cluster if top result is near 0
            if clusters and abs(best_angle) < 0.5:
                # Find best cluster
                best_cluster = max(clusters, key=lambda x: x[1])  # Sort by variance
                if best_cluster[1] > best_variance * 0.7:
                    best_angle = best_cluster[0]
                    self.logger.info(f"      🎯 Using cluster: {best_angle:.2f}° (faster detection)")
        
        return best_angle, all_results
    
    def _detect_skew_hough_optimized(self, binary, max_skew=15):
        """OPTIMIZED Hough method - single pass with optimal parameters"""
        # OPTIMIZATION 1: Single Canny with optimal parameters
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # OPTIMIZATION 2: Single threshold with optimal value
        threshold = 120  # Optimal threshold found through testing
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=threshold)
        
        if lines is None:
            return 0.0, 0
        
        # OPTIMIZATION 3: Vectorized angle processing
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            
            # Normalize to [-15, 15] range
            if angle > max_skew:
                angle = angle - 90
            elif angle < -max_skew:
                angle = angle + 90
            
            if -max_skew <= angle <= max_skew:
                angles.append(angle)
        
        if not angles:
            return 0.0, len(lines)
        
        # OPTIMIZATION 4: Fast weighted average
        angles_array = np.array(angles)
        weights = 1.0 / (1.0 + np.abs(angles_array))
        weighted_avg = np.average(angles_array, weights=weights)
        
        return weighted_avg, len(angles)
    
    def _detect_skew_text_lines_optimized(self, binary, max_skew=15):
        """OPTIMIZED text line analysis - early termination and vectorization"""
        # OPTIMIZATION 1: Smaller kernel for speed
        kernel_horizontal = np.ones((1, 10), np.uint8)  # Reduced from 15 to 10
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
        
        # OPTIMIZATION 2: Use RETR_EXTERNAL for faster contour detection
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0, 0
        
        # OPTIMIZATION 3: Early termination if too many contours
        if len(contours) > 100:  # Limit processing for speed
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:100]
        
        # OPTIMIZATION 4: Vectorized processing
        valid_angles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                # Normalize angle to [-15, 15] range
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                
                if -max_skew <= angle <= max_skew:
                    valid_angles.append(angle)
        
        if not valid_angles:
            return 0.0, len(contours)
        
        # OPTIMIZATION 5: Use numpy for faster median
        return float(np.median(valid_angles)), len(valid_angles)
    
    def _intelligent_combination_optimized(self, proj_angle, proj_results, hough_angle, hough_conf, text_angle, text_conf):
        """OPTIMIZED intelligent combination - simplified logic"""
        self.logger.info(f"   🧠 Fast intelligent combination...")
        
        # OPTIMIZATION 1: Simplified decision tree
        # If projection found a clear angle, use it (most reliable)
        if abs(proj_angle) > 0.5:
            # Quick support check
            hough_support = abs(hough_angle - proj_angle) < 2.0
            text_support = abs(text_angle - proj_angle) < 2.0
            
            if hough_support or text_support:
                self.logger.info(f"      ✅ Projection angle {proj_angle:.2f}° supported")
                return proj_angle
        
        # OPTIMIZATION 2: Fast fallback to best available method
        if hough_conf > 5 and abs(hough_angle) > 0.5:
            self.logger.info(f"      ✅ Using Hough angle {hough_angle:.2f}°")
            return hough_angle
        
        if text_conf > 3 and abs(text_angle) > 0.5:
            self.logger.info(f"      ✅ Using text line angle {text_angle:.2f}°")
            return text_angle
        
        # OPTIMIZATION 3: Simple weighted average
        angles = []
        weights = []
        
        if abs(proj_angle) > 0.1:
            angles.append(proj_angle)
            weights.append(3.0)
        
        if hough_conf > 0:
            angles.append(hough_angle)
            weights.append(2.0)
        
        if text_conf > 0:
            angles.append(text_angle)
            weights.append(1.5)
        
        if angles:
            weighted_avg = np.average(angles, weights=weights)
            self.logger.info(f"      🔄 Using weighted average: {weighted_avg:.2f}°")
            return weighted_avg
        
        return proj_angle
    
    def _adaptive_preprocessing_optimized(self, image_bgr):
        """OPTIMIZED preprocessing for ±15° range"""
        # OPTIMIZATION 1: Resize large images for speed
        height, width = image_bgr.shape[:2]
        max_dimension = 2000  # Limit image size for speed
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            self.logger.info(f"      📏 Resized for speed: {width}x{height} → {new_width}x{new_height}")
        
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # OPTIMIZATION 2: Use optimal block size for adaptive threshold
        block_size = min(11, max(3, min(gray.shape) // 50))  # Adaptive block size
        # Ensure block size is odd for adaptive threshold
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, block_size, 2)
        
        # OPTIMIZATION 3: Minimal noise removal
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _deskew_image(self, image_bgr, angle_deg):
        """Deskew image by the detected angle"""
        if abs(angle_deg) < 2.0:  # No need to deskew if angle is very small (increased threshold)
            return image_bgr
        
        height, width = image_bgr.shape[:2]
        
        # Calculate rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        
        # Calculate new image dimensions
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        deskewed = cv2.warpAffine(image_bgr, rotation_matrix, (new_width, new_height), 
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return deskewed
    
    def _create_comparison_image(self, original, deskewed, filename, detected_angle, processing_time):
        """Create a side-by-side comparison image"""
        # Ensure both images have the same height for comparison
        h1, w1 = original.shape[:2]
        h2, w2 = deskewed.shape[:2]
        
        # Resize deskewed image to match original height if needed
        if h2 != h1:
            scale = h1 / h2
            new_w = int(w2 * scale)
            deskewed_resized = cv2.resize(deskewed, (new_w, h1), interpolation=cv2.INTER_AREA)
        else:
            deskewed_resized = deskewed
        
        # Create comparison image
        comparison = np.hstack([original, deskewed_resized])
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)  # Green text
        
        # Original image label
        cv2.putText(comparison, f"Original", (50, 50), font, font_scale, color, thickness)
        
        # Deskewed image label
        cv2.putText(comparison, f"Deskewed ({detected_angle:+.2f}°)", (w1 + 50, 50), font, font_scale, color, thickness)
        
        # Processing info
        cv2.putText(comparison, f"Processing time: {processing_time:.2f}s", (50, h1 - 50), font, 0.7, (255, 255, 255), 1)
        
        return comparison
    
    def _save_processing_summary(self, filename, angle, method, processing_time, summary_path):
        """Save processing summary to CSV"""
        try:
            # Check if file exists to determine if we need to write header
            write_header = not os.path.exists(summary_path)
            
            with open(summary_path, 'a', newline='') as csvfile:
                fieldnames = ['filename', 'detected_angle', 'method', 'processing_time', 'status', 'error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if write_header:
                    writer.writeheader()
                
                writer.writerow({
                    'filename': filename,
                    'detected_angle': f"{angle:.2f}",
                    'method': method,
                    'processing_time': f"{processing_time:.2f}",
                    'status': 'success',
                    'error': ''
                })
        except Exception as e:
            self.logger.error(f"Error saving processing summary: {str(e)}")
    
    def get_task_info(self):
        """Get information about this task"""
        
        return {
            'task_id': self.task_id,
            'name': self.task_name,
            'description': 'Detect and correct document skew using optimized ±15° detector',
            'order': 1,
            'dependencies': [],
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
    task = SkewDetectionTask(logger)
    
    print("🧪 Testing Skew Detection Task")
    print("=" * 40)
    
    # Test with a sample image
    test_image = "input/test_image.png"
    if os.path.exists(test_image):
        result = task.run(test_image, 'image', 'output')
        if result:
            print(f"✅ Task completed successfully!")
            print(f"   Output: {result['output']}")
            print(f"   Angle: {result['angle']:.2f}°")
        else:
            print("❌ Task failed!")
    else:
        print(f"⚠️  Test image not found: {test_image}")
        print("   Create an input folder with test images to test")
