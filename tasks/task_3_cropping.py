#!/usr/bin/env python3
"""
Task 2: Document Cropping
Contains the actual extreme-tight text cropping code copied from 2.cropping/extreme_tight_text_cropping.py
"""

import os
import cv2
import numpy as np

import logging

class DocumentCroppingTask:
    """Task 2: Document Cropping - Contains actual extreme-tight cropping algorithm"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Document Cropping"
        self.task_id = "task_3_cropping"
        self.config = {}
        
    def run(self, input_file, file_type, output_folder):
        """
        Run document cropping on the input file
        
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
            task_output = os.path.join(output_folder, "task3_cropping")
            os.makedirs(task_output, exist_ok=True)
            
            # Process based on file type
            if file_type == 'pdf':
                result = self._process_pdf_cropping(input_file, task_output)
            else:
                result = self._process_image_cropping(input_file, task_output)
            
            if result:
                self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
                return result
            else:
                self.logger.error(f"❌ {self.task_name} failed for {os.path.basename(input_file)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in {self.task_name}: {str(e)}")
            return None
    
    def _process_image_cropping(self, image_path, output_folder):
        """Process single image for cropping using the actual algorithm"""
        
        try:
            filename = os.path.basename(image_path)
            self.logger.info(f"📄 Processing: {filename}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Failed to load image")
            
            height, width = image.shape[:2]
            self.logger.info(f"   📏 Original size: {width}x{height}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Step 1: Detect main text block using line-column analysis
            self.logger.info("   📊 Step 1: Line-column text block detection...")
            
            # Convert to binary for analysis
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Analyze rows
            row_has_text = []
            for y in range(height):
                row = binary[y, :]
                text_pixels = np.sum(row > 0)
                density = text_pixels / width
                has_text = density > 0.02  # 2% threshold
                row_has_text.append(has_text)
            
            # Find text rows
            text_rows = [i for i, has_text in enumerate(row_has_text) if has_text]
            if not text_rows:
                # Fallback: use entire image if no text detected
                first_text_row = 0
                last_text_row = height - 1
            else:
                first_text_row = min(text_rows)
                last_text_row = max(text_rows)
            
            # Analyze columns
            col_has_text = []
            for x in range(width):
                col = binary[:, x]
                text_pixels = np.sum(col > 0)
                density = text_pixels / height
                has_text = density > 0.02  # 2% threshold
                col_has_text.append(has_text)
            
            # Find text columns
            text_cols = [i for i, has_text in enumerate(col_has_text) if has_text]
            if not text_cols:
                # Fallback: use entire image if no text detected
                first_text_col = 0
                last_text_col = width - 1
            else:
                first_text_col = min(text_cols)
                last_text_col = max(text_cols)
            
            # Define main text block
            text_block_x = first_text_col
            text_block_y = first_text_row
            text_block_w = last_text_col - first_text_col + 1
            text_block_h = last_text_row - first_text_row + 1
            
            self.logger.info(f"   📍 Main text block: {text_block_w}x{text_block_h} at ({text_block_x}, {text_block_y})")
            self.logger.info(f"   📊 Text area: {(text_block_w * text_block_h) / (width * height) * 100:.1f}% of image")
            
            # Step 2: Find extreme-tight cropping boundaries using character-level analysis
            self.logger.info("   🔍 Step 2: Finding extreme-tight cropping boundaries...")
            
            # Create a very precise text density map with smaller windows
            window_size = 10  # Smaller window for more precision
            density_threshold = 0.001  # Extremely low threshold
            
            # Left boundary: find where text density drops to extremely low levels
            left_boundary = text_block_x
            for x in range(text_block_x - 1, max(0, text_block_x - 150), -1):
                # Check a small window around this column
                start_x = max(0, x - window_size // 2)
                end_x = min(width, x + window_size // 2)
                
                window = binary[:, start_x:end_x]
                text_density = np.sum(window > 0) / (window.shape[0] * window.shape[1])
                
                if text_density < density_threshold:  # Extremely low text density
                    left_boundary = x + 1
                    break
            
            # Right boundary: find where text density drops to extremely low levels
            right_boundary = text_block_x + text_block_w
            for x in range(text_block_x + text_block_w, min(width, text_block_x + text_block_w + 150)):
                # Check a small window around this column
                start_x = max(0, x - window_size // 2)
                end_x = min(width, x + window_size // 2)
                
                window = binary[:, start_x:end_x]
                text_density = np.sum(window > 0) / (window.shape[0] * window.shape[1])
                
                if text_density < density_threshold:  # Extremely low text density
                    right_boundary = x
                    break
            
            # Top boundary: find where text density drops to extremely low levels
            top_boundary = text_block_y
            for y in range(text_block_y - 1, max(0, text_block_y - 100), -1):
                # Check a small window around this row
                start_y = max(0, y - window_size // 2)
                end_y = min(height, y + window_size // 2)
                
                window = binary[start_y:end_y, :]
                text_density = np.sum(window > 0) / (window.shape[0] * window.shape[1])
                
                if text_density < density_threshold:  # Extremely low text density
                    top_boundary = y + 1
                    break
            
            # Bottom boundary: find where text density drops to extremely low levels
            bottom_boundary = text_block_y + text_block_h
            for y in range(text_block_y + text_block_h, min(height, text_block_y + text_block_h + 100)):
                # Check a small window around this row
                start_y = max(0, y - window_size // 2)
                end_y = min(height, y + window_size // 2)
                
                window = binary[start_y:end_y, :]
                text_density = np.sum(window > 0) / (window.shape[0] * window.shape[1])
                
                if text_density < density_threshold:  # Extremely low text density
                    bottom_boundary = y
                    break
            
            # Step 3: Apply extreme-tight cropping
            self.logger.info("   ✂️  Step 3: Applying extreme-tight cropping...")
            
            # Crop the image to extreme-tight boundaries
            cropped_image = image[top_boundary:bottom_boundary, left_boundary:right_boundary]
            
            crop_width = right_boundary - left_boundary
            crop_height = bottom_boundary - top_boundary
            
            self.logger.info(f"   📍 Extreme-tight cropping area: {crop_width}x{crop_height} at ({left_boundary}, {top_boundary})")
            self.logger.info(f"   📊 Cropped area: {(crop_width * crop_height) / (width * height) * 100:.1f}% of image")
            
            # Step 4: Final cleaning of any remaining artifacts (skip in fast mode)
            holes_removed = 0
            edges_removed = 0
            
            if not self.config.get('fast_mode', False) and not self.config.get('skip_artifact_removal', False):
                self.logger.info("   🔧 Step 4: Final cleaning of remaining artifacts...")
                
                # Convert cropped to grayscale
                cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                
                # Remove any remaining punch holes
                _, cropped_dark = cv2.threshold(cropped_gray, 100, 255, cv2.THRESH_BINARY_INV)
                cropped_dark = cv2.morphologyEx(cropped_dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                
                hole_contours_cropped, _ = cv2.findContours(cropped_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in hole_contours_cropped:
                    area = cv2.contourArea(contour)
                    if 15 <= area <= 400:
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.5:
                                mask = np.zeros_like(cropped_gray)
                                cv2.drawContours(mask, [contour], -1, 255, -1)
                                cropped_image = cv2.inpaint(cropped_image, mask, 3, cv2.INPAINT_NS)
                                holes_removed += 1
                
                self.logger.info(f"   ✅ Removed {holes_removed} remaining punch holes")
            else:
                self.logger.info("   ⚡ Skipping artifact removal for fast mode")
                cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            # Remove any remaining scanner edges (skip in fast mode)
            if not self.config.get('fast_mode', False) and not self.config.get('skip_artifact_removal', False):
                cropped_edges = cv2.Canny(cropped_gray, 25, 75)
                horizontal_kernel = np.ones((1, 20), np.uint8)
                vertical_kernel = np.ones((20, 1), np.uint8)
                
                horizontal_lines = cv2.morphologyEx(cropped_edges, cv2.MORPH_OPEN, horizontal_kernel)
                vertical_lines = cv2.morphologyEx(cropped_edges, cv2.MORPH_OPEN, vertical_kernel)
                
                scanner_artifacts = cv2.bitwise_or(horizontal_lines, vertical_lines)
                scanner_artifacts = cv2.dilate(scanner_artifacts, np.ones((2, 2), np.uint8))
                
                mask = scanner_artifacts.astype(np.uint8)
                edges_removed = np.sum(mask > 0)
                
                if edges_removed > 0:
                    blurred = cv2.medianBlur(cropped_image, 3)
                    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    mask_normalized = mask_3d.astype(np.float32) / 255.0
                    
                    cropped_image = cropped_image.astype(np.float32)
                    blurred = blurred.astype(np.float32)
                    
                    cropped_image = cropped_image * (1 - mask_normalized) + blurred * mask_normalized
                    cropped_image = cropped_image.astype(np.uint8)
                
                self.logger.info(f"   ✅ Removed {edges_removed} remaining scanner edges")
            else:
                self.logger.info("   ⚡ Skipping edge removal for fast mode")
            
            # Step 5: Quality check (skip in fast mode)
            if not self.config.get('fast_mode', False) and not self.config.get('skip_quality_check', False):
                self.logger.info("   🔍 Step 5: Quality check...")
                
                # Check final image quality
                final_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                
                # Check for remaining dark areas
                _, final_dark = cv2.threshold(final_gray, 120, 255, cv2.THRESH_BINARY_INV)
                remaining_dark_pixels = np.sum(final_dark > 0)
            else:
                self.logger.info("   ⚡ Skipping quality check for fast mode")
                final_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                remaining_dark_pixels = 0
            
            # Check for remaining edges (skip in fast mode)
            if not self.config.get('fast_mode', False) and not self.config.get('skip_quality_check', False):
                final_edges = cv2.Canny(final_gray, 30, 90)
                remaining_edge_pixels = np.sum(final_edges > 0)
                self.logger.info(f"   📊 Final quality: {remaining_dark_pixels} dark pixels, {remaining_edge_pixels} edge pixels")
            else:
                remaining_edge_pixels = 0
            
            # Generate outputs
            base_name = os.path.splitext(filename)[0]
            
            # Save cropped image
            cropped_path = os.path.join(output_folder, f"{base_name}_cropped.png")
            cv2.imwrite(cropped_path, cropped_image)
            self.logger.info(f"💾 Cropped image saved: {cropped_path}")
            
            # Create and save comparison image
            comparison = self._create_comparison_image(image, cropped_image, filename, 
                                                     text_block_x, text_block_y, text_block_w, text_block_h,
                                                     left_boundary, top_boundary, crop_width, crop_height,
                                                     holes_removed, edges_removed,
                                                     remaining_dark_pixels, remaining_edge_pixels)
            comparison_path = os.path.join(output_folder, f"{base_name}_comparison.png")
            cv2.imwrite(comparison_path, comparison)
            self.logger.info(f"💾 Comparison image saved: {comparison_path}")
            
            return {
                'input': image_path,
                'output': cropped_path,
                'comparison': comparison_path,
                'crop_area': (left_boundary, top_boundary, crop_width, crop_height),
                'holes_removed': holes_removed,
                'edges_removed': edges_removed,
                'status': 'completed',
                'task': self.task_id
            }
                
        except Exception as e:
            self.logger.error(f"Error processing image for cropping: {str(e)}")
            return None
    
    def _process_pdf_cropping(self, pdf_path, output_folder):
        """Process PDF for cropping"""
        
        try:
            self.logger.info("📄 Processing PDF for cropping")
            
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
            result = self._process_image_cropping(converted_path, output_folder)
            
            if result:
                # Update the result to reflect it came from PDF
                result['input'] = pdf_path
                result['note'] = 'PDF converted to image and processed'
                return result
            else:
                raise Exception("Failed to process converted PDF image")
            
        except Exception as e:
            self.logger.error(f"Error processing PDF for cropping: {str(e)}")
            return None
    
    def _create_comparison_image(self, original, result, filename, 
                                text_x, text_y, text_w, text_h,
                                crop_x, crop_y, crop_w, crop_h,
                                holes_removed, edges_removed,
                                remaining_dark, remaining_edges):
        """Create comparison visualization"""
        
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
        
        # Cropped image label
        cv2.putText(comparison, "Cropped", (width + 50, 50), font, font_scale, color, thickness)
        
        # Add analysis info
        info_text = f"Text Block: {text_w}x{text_h} at ({text_x},{text_y}) | Crop: {crop_w}x{crop_h} at ({crop_x},{crop_y})"
        cv2.putText(comparison, info_text, (50, height - 30), font, 0.6, (255, 255, 255), 1)
        
        # Add quality metrics
        quality_text = f"Holes removed: {holes_removed} | Edges removed: {edges_removed} | Dark pixels: {remaining_dark}"
        cv2.putText(comparison, quality_text, (50, height - 10), font, 0.6, (255, 255, 255), 1)
        
        return comparison
    
    def get_task_info(self):
        """Get information about this task"""
        
        return {
            'task_id': self.task_id,
            'name': self.task_name,
            'description': 'Remove blank borders, punch holes, and scanner edges using extreme-tight cropping',
            'order': 2,
            'dependencies': ['task_1_skew_detection'],
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
    task = DocumentCroppingTask(logger)
    
    print("🧪 Testing Document Cropping Task")
    print("=" * 40)
    
    # Test with a sample image
    test_image = "input/test_image.png"
    if os.path.exists(test_image):
        result = task.run(test_image, 'image', 'output')
        if result:
            print(f"✅ Task completed successfully!")
            print(f"   Output: {result['output']}")
            print(f"   Crop area: {result['crop_area']}")
        else:
            print("❌ Task failed!")
    else:
        print(f"⚠️  Test image not found: {test_image}")
        print("   Create an input folder with test images to test")
