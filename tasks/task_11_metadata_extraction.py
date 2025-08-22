#!/usr/bin/env python3
"""
Task 11: Metadata Extraction (Pre-OCR)
Extract comprehensive metadata before OCR processing for orchestration layer.
"""

import cv2
import numpy as np
import os
import logging
import json
import hashlib
from PIL import Image, ExifTags
import shutil
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
import mimetypes

class MetadataExtractionTask:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Metadata Extraction (Pre-OCR)"
        self.task_id = "task_11_metadata_extraction"
        self.config = {
            # Image Analysis Settings
            "analyze_resolution": True,
            "analyze_color_depth": True,
            "analyze_compression": True,
            "calculate_file_hash": True,
            
            # Content Analysis Settings
            "analyze_text_density": True,
            "analyze_image_density": True,
            "analyze_graphics_presence": True,
            "analyze_table_presence": True,
            "analyze_layout_structure": True,
            
            # Quality Analysis Settings
            "analyze_image_quality": True,
            "detect_blur": True,
            "detect_noise": True,
            "detect_low_contrast": True,
            "analyze_brightness": True,
            
            # Geometric Analysis
            "analyze_page_orientation": True,
            "analyze_skew_angle": True,
            "detect_margins": True,
            "analyze_aspect_ratio": True,
            
            # Content Type Detection
            "detect_document_type": True,
            "detect_form_fields": True,
            "detect_signatures": True,
            "detect_stamps": True,
            "detect_barcodes": True,
            
            # Output Settings
            "save_sidecar_json": True,
            "include_thumbnail": True,
            "thumbnail_size": (200, 200),
            "sidecar_suffix": "_metadata",
            "embed_processing_info": True,
            
            # Analysis Thresholds
            "text_density_threshold": 0.1,
            "graphics_threshold": 0.2,
            "blur_threshold": 100,
            "contrast_threshold": 50,
            "noise_threshold": 0.1
        }
    
    def run(self, input_file, file_type, output_folder):
        """
        Main entry point for metadata extraction
        
        Returns:
            dict: Task result with comprehensive metadata
        """
        try:
            self.logger.info(f"🔄 Running {self.task_name} on {os.path.basename(input_file)}")
            
            # Process metadata extraction
            result_path, metadata_results = self._process_metadata_extraction(input_file, output_folder)
            
            self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
            
            return {
                'status': 'completed',
                'output': result_path,
                'task_name': self.task_name,
                'processing_time': None,
                'metadata': {
                    'input_file': input_file,
                    'output_file': result_path,
                    'file_type': file_type,
                    'extracted_metadata': metadata_results
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ {self.task_name} failed for {input_file}: {str(e)}")
            return {
                'status': 'failed',
                'output': input_file,
                'task_name': self.task_name,
                'error': str(e),
                'metadata': {
                    'input_file': input_file,
                    'file_type': file_type
                }
            }
    
    def _process_metadata_extraction(self, image_path, output_folder):
        """Process comprehensive metadata extraction"""
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        self.logger.info(f"   🔍 Step 1: Basic file and image analysis")
        
        # Load image for analysis
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Initialize metadata collection
        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'source_file': image_path,
            'filename': filename
        }
        
        # Step 1: File-level metadata
        file_metadata = self._extract_file_metadata(image_path)
        metadata['file_metadata'] = file_metadata
        self.logger.info(f"   📊 File: {file_metadata['file_size_mb']:.1f}MB, {file_metadata['mime_type']}")
        
        # Step 2: Image properties
        image_metadata = self._extract_image_metadata(image, image_path)
        metadata['image_metadata'] = image_metadata
        self.logger.info(f"   📐 Image: {image_metadata['dimensions']['width']}x{image_metadata['dimensions']['height']}, {image_metadata['color_depth']} bits")
        
        # Step 3: Content analysis
        self.logger.info(f"   🔍 Step 2: Analyzing content structure and density")
        content_metadata = self._analyze_content_structure(image)
        metadata['content_analysis'] = content_metadata
        self.logger.info(f"   📊 Content: {content_metadata['text_density']:.1f}% text, {content_metadata['graphics_density']:.1f}% graphics")
        
        # Step 4: Quality analysis
        self.logger.info(f"   🔍 Step 3: Analyzing image quality metrics")
        quality_metadata = self._analyze_image_quality(image)
        metadata['quality_analysis'] = quality_metadata
        self.logger.info(f"   📊 Quality: blur={quality_metadata['blur_score']:.1f}, contrast={quality_metadata['contrast_score']:.1f}")
        
        # Step 5: Layout and structure analysis
        self.logger.info(f"   🔍 Step 4: Analyzing layout and geometric properties")
        layout_metadata = self._analyze_layout_structure(image)
        metadata['layout_analysis'] = layout_metadata
        
        # Step 6: Document type detection
        self.logger.info(f"   🔍 Step 5: Detecting document type and special elements")
        document_metadata = self._detect_document_features(image)
        metadata['document_analysis'] = document_metadata
        self.logger.info(f"   📄 Document type: {document_metadata['document_type']}")
        
        # Step 7: Generate processing recommendations
        processing_metadata = self._generate_processing_recommendations(metadata)
        metadata['processing_recommendations'] = processing_metadata
        
        # Step 8: Save metadata sidecar
        if self.config['save_sidecar_json']:
            sidecar_path = self._save_metadata_sidecar(metadata, output_folder, filename)
            self.logger.info(f"   💾 Metadata sidecar saved: {os.path.basename(sidecar_path)}")
        
        # Step 9: Generate thumbnail if requested
        if self.config['include_thumbnail']:
            thumbnail_path = self._generate_thumbnail(image, output_folder, filename)
            metadata['thumbnail_path'] = thumbnail_path
            if thumbnail_path:
                self.logger.info(f"   🖼️  Thumbnail saved: {os.path.basename(thumbnail_path)}")
        
        return image_path, metadata
    
    def _extract_file_metadata(self, file_path):
        """Extract file-level metadata"""
        
        try:
            stat = os.stat(file_path)
            
            metadata = {
                'file_size_bytes': stat.st_size,
                'file_size_mb': stat.st_size / (1024 * 1024),
                'creation_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modification_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'mime_type': mimetypes.guess_type(file_path)[0] or 'unknown',
                'file_extension': os.path.splitext(file_path)[1].lower()
            }
            
            # Calculate file hash if enabled
            if self.config['calculate_file_hash']:
                metadata['file_hash'] = self._calculate_file_hash(file_path)
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not extract file metadata: {str(e)}")
            return {'error': str(e)}
    
    def _extract_image_metadata(self, image, image_path):
        """Extract image-specific metadata"""
        
        try:
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            metadata = {
                'dimensions': {
                    'width': int(width),
                    'height': int(height),
                    'channels': int(channels)
                },
                'total_pixels': int(width * height),
                'aspect_ratio': float(width / height),
                'color_depth': int(channels * 8),  # Assuming 8 bits per channel
                'color_space': 'BGR' if channels == 3 else 'Grayscale'
            }
            
            # Try to extract EXIF data
            try:
                pil_image = Image.open(image_path)
                exif_data = {}
                
                if hasattr(pil_image, '_getexif'):
                    exif = pil_image._getexif()
                    if exif:
                        for tag, value in exif.items():
                            tag_name = ExifTags.TAGS.get(tag, tag)
                            exif_data[tag_name] = str(value)
                
                metadata['exif_data'] = exif_data
                
                # Extract DPI information
                if 'XResolution' in exif_data and 'YResolution' in exif_data:
                    try:
                        x_res = float(exif_data['XResolution'])
                        y_res = float(exif_data['YResolution'])
                        metadata['dpi'] = {'x': x_res, 'y': y_res}
                    except:
                        pass
                        
            except Exception:
                metadata['exif_data'] = {}
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not extract image metadata: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_content_structure(self, image):
        """Analyze content structure and density"""
        
        try:
            height, width = image.shape[:2]
            total_area = width * height
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Text density estimation (based on edge detection and contours)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours for text-like regions
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_area = 0
            graphics_area = 0
            small_objects = 0
            medium_objects = 0
            large_objects = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Classify based on size and aspect ratio
                if area < 100:
                    small_objects += 1
                elif area < 5000:
                    medium_objects += 1
                    # Text-like characteristics: moderate size, reasonable aspect ratio
                    if 0.1 < aspect_ratio < 10 and area > 50:
                        text_area += area
                else:
                    large_objects += 1
                    # Graphics-like: large areas
                    graphics_area += area
            
            # Calculate densities
            text_density = (text_area / total_area * 100) if total_area > 0 else 0
            graphics_density = (graphics_area / total_area * 100) if total_area > 0 else 0
            
            # Detect potential tables (regular grid patterns)
            table_score = self._detect_table_patterns(gray)
            
            # Analyze white space
            white_pixels = np.sum(gray > 200)
            white_space_ratio = white_pixels / total_area if total_area > 0 else 0
            
            metadata = {
                'text_density': float(text_density),
                'graphics_density': float(graphics_density),
                'white_space_ratio': float(white_space_ratio * 100),
                'object_counts': {
                    'small_objects': int(small_objects),
                    'medium_objects': int(medium_objects),
                    'large_objects': int(large_objects)
                },
                'table_likelihood_score': float(table_score),
                'content_complexity': self._calculate_content_complexity(small_objects, medium_objects, large_objects),
                'estimated_text_regions': int(medium_objects),
                'estimated_graphic_regions': int(large_objects)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not analyze content structure: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_image_quality(self, image):
        """Analyze image quality metrics"""
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contrast analysis
            contrast_score = gray.std()
            
            # Brightness analysis
            brightness = gray.mean()
            
            # Noise estimation (using local standard deviation)
            noise_score = self._estimate_noise(gray)
            
            # Sharpness estimation
            sharpness_score = self._estimate_sharpness(gray)
            
            # Overall quality assessment
            quality_flags = {
                'is_blurred': blur_score < self.config['blur_threshold'],
                'is_low_contrast': contrast_score < self.config['contrast_threshold'],
                'is_too_dark': brightness < 50,
                'is_too_bright': brightness > 200,
                'is_noisy': noise_score > self.config['noise_threshold']
            }
            
            # Calculate overall quality score
            quality_issues = sum(quality_flags.values())
            overall_quality_score = max(0, 100 - (quality_issues * 20))
            
            metadata = {
                'blur_score': float(blur_score),
                'contrast_score': float(contrast_score),
                'brightness_score': float(brightness),
                'noise_score': float(noise_score),
                'sharpness_score': float(sharpness_score),
                'overall_quality_score': float(overall_quality_score),
                'quality_flags': quality_flags,
                'quality_assessment': self._assess_quality_level(overall_quality_score)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not analyze image quality: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_layout_structure(self, image):
        """Analyze layout and geometric properties"""
        
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect margins by analyzing edge distribution
            margins = self._detect_margins(gray)
            
            # Analyze orientation (portrait vs landscape)
            orientation = 'portrait' if height > width else 'landscape' if width > height else 'square'
            
            # Estimate skew angle (simplified)
            skew_angle = self._estimate_skew_angle(gray)
            
            # Analyze page structure (columns, regions)
            page_structure = self._analyze_page_structure(gray)
            
            metadata = {
                'page_orientation': orientation,
                'estimated_skew_angle': float(skew_angle),
                'margins': margins,
                'page_structure': page_structure,
                'content_regions': self._identify_content_regions(gray),
                'reading_order_estimate': self._estimate_reading_order(gray)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not analyze layout structure: {str(e)}")
            return {'error': str(e)}
    
    def _detect_document_features(self, image):
        """Detect document type and special features"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Document type detection
            document_type = self._classify_document_type(image)
            
            # Form field detection
            form_fields = self._detect_form_fields(gray)
            
            # Signature detection (basic)
            signatures = self._detect_signatures(gray)
            
            # Stamp detection (basic)
            stamps = self._detect_stamps(gray)
            
            # Barcode detection (basic)
            barcodes = self._detect_barcodes(gray)
            
            # Logo/letterhead detection
            logos = self._detect_logos(gray)
            
            metadata = {
                'document_type': document_type,
                'contains_form_fields': len(form_fields) > 0,
                'form_fields_count': len(form_fields),
                'contains_signatures': len(signatures) > 0,
                'signature_regions': len(signatures),
                'contains_stamps': len(stamps) > 0,
                'stamp_regions': len(stamps),
                'contains_barcodes': len(barcodes) > 0,
                'barcode_regions': len(barcodes),
                'contains_logos': len(logos) > 0,
                'logo_regions': len(logos),
                'special_features': self._identify_special_features(image)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not detect document features: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of the file"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    def _detect_table_patterns(self, gray):
        """Detect table-like patterns in the image"""
        try:
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Count significant lines
            h_line_count = np.sum(horizontal_lines > 0)
            v_line_count = np.sum(vertical_lines > 0)
            
            # Calculate table score based on line presence
            total_pixels = gray.shape[0] * gray.shape[1]
            line_density = (h_line_count + v_line_count) / total_pixels
            
            return min(100, line_density * 1000)  # Scale to 0-100
            
        except Exception:
            return 0.0
    
    def _calculate_content_complexity(self, small, medium, large):
        """Calculate content complexity score"""
        total_objects = small + medium + large
        if total_objects == 0:
            return 0
        
        # Weight larger objects more heavily
        complexity = (small * 1 + medium * 2 + large * 3) / total_objects
        return min(100, complexity * 20)  # Scale to 0-100
    
    def _estimate_noise(self, gray):
        """Estimate noise level in the image"""
        try:
            # Use local standard deviation to estimate noise
            kernel = np.ones((5, 5), np.float32) / 25
            mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            
            diff = np.abs(gray.astype(np.float32) - mean_img)
            noise_level = np.mean(diff) / 255.0
            
            return noise_level
            
        except Exception:
            return 0.0
    
    def _estimate_sharpness(self, gray):
        """Estimate image sharpness"""
        try:
            # Use Sobel operator to detect edges
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            sharpness = np.mean(np.sqrt(sobelx**2 + sobely**2))
            return sharpness
            
        except Exception:
            return 0.0
    
    def _assess_quality_level(self, score):
        """Assess quality level based on score"""
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        elif score >= 20:
            return 'poor'
        else:
            return 'very_poor'
    
    def _detect_margins(self, gray):
        """Detect page margins"""
        try:
            height, width = gray.shape
            
            # Analyze edge density to estimate margins
            edge_threshold = 20
            
            # Top margin
            top_margin = 0
            for i in range(min(height//4, 100)):
                if np.mean(gray[i, :]) < 240:  # Found content
                    top_margin = i
                    break
            
            # Bottom margin
            bottom_margin = 0
            for i in range(min(height//4, 100)):
                row = height - 1 - i
                if np.mean(gray[row, :]) < 240:
                    bottom_margin = i
                    break
            
            # Left margin
            left_margin = 0
            for i in range(min(width//4, 100)):
                if np.mean(gray[:, i]) < 240:
                    left_margin = i
                    break
            
            # Right margin
            right_margin = 0
            for i in range(min(width//4, 100)):
                col = width - 1 - i
                if np.mean(gray[:, col]) < 240:
                    right_margin = i
                    break
            
            return {
                'top': int(top_margin),
                'bottom': int(bottom_margin),
                'left': int(left_margin),
                'right': int(right_margin)
            }
            
        except Exception:
            return {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    def _estimate_skew_angle(self, gray):
        """Estimate skew angle of the document"""
        try:
            # Simple skew detection using Hough line transform
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:10]:  # Check first 10 lines
                    angle = theta * 180 / np.pi
                    if angle > 45:
                        angle = angle - 90
                    angles.append(angle)
                
                if angles:
                    return np.median(angles)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_page_structure(self, gray):
        """Analyze page structure for columns and regions"""
        try:
            height, width = gray.shape
            
            # Analyze horizontal projection to find text lines
            horizontal_projection = np.sum(gray < 200, axis=1)
            
            # Analyze vertical projection to find columns
            vertical_projection = np.sum(gray < 200, axis=0)
            
            # Estimate number of columns
            # Find valleys in vertical projection
            valleys = []
            threshold = np.mean(vertical_projection) * 0.3
            
            for i in range(1, len(vertical_projection) - 1):
                if (vertical_projection[i] < threshold and 
                    vertical_projection[i] < vertical_projection[i-1] and 
                    vertical_projection[i] < vertical_projection[i+1]):
                    valleys.append(i)
            
            estimated_columns = len(valleys) + 1 if valleys else 1
            
            # Estimate text lines
            line_threshold = np.mean(horizontal_projection) * 0.2
            text_lines = np.sum(horizontal_projection > line_threshold)
            
            return {
                'estimated_columns': min(estimated_columns, 5),  # Cap at 5 columns
                'estimated_text_lines': int(text_lines),
                'layout_type': 'multi_column' if estimated_columns > 1 else 'single_column'
            }
            
        except Exception:
            return {'estimated_columns': 1, 'estimated_text_lines': 0, 'layout_type': 'single_column'}
    
    def _identify_content_regions(self, gray):
        """Identify major content regions"""
        try:
            # Simplified region detection
            height, width = gray.shape
            
            # Divide into 3x3 grid and analyze each region
            regions = []
            for i in range(3):
                for j in range(3):
                    y1 = i * height // 3
                    y2 = (i + 1) * height // 3
                    x1 = j * width // 3
                    x2 = (j + 1) * width // 3
                    
                    region = gray[y1:y2, x1:x2]
                    content_density = np.sum(region < 200) / (region.shape[0] * region.shape[1])
                    
                    region_type = 'text' if content_density > 0.1 else 'whitespace'
                    if content_density > 0.3:
                        region_type = 'dense_content'
                    
                    regions.append({
                        'position': f'grid_{i}_{j}',
                        'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'type': region_type,
                        'density': float(content_density)
                    })
            
            return regions
            
        except Exception:
            return []
    
    def _estimate_reading_order(self, gray):
        """Estimate reading order based on content layout"""
        try:
            # Simple heuristic based on content distribution
            height, width = gray.shape
            
            # Check if content is distributed left-to-right or top-to-bottom
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            top_half = gray[:height//2, :]
            bottom_half = gray[height//2:, :]
            
            left_content = np.sum(left_half < 200)
            right_content = np.sum(right_half < 200)
            top_content = np.sum(top_half < 200)
            bottom_content = np.sum(bottom_half < 200)
            
            if left_content > right_content * 1.5:
                horizontal_flow = 'left_heavy'
            elif right_content > left_content * 1.5:
                horizontal_flow = 'right_heavy'
            else:
                horizontal_flow = 'balanced'
            
            if top_content > bottom_content * 1.5:
                vertical_flow = 'top_heavy'
            elif bottom_content > top_content * 1.5:
                vertical_flow = 'bottom_heavy'
            else:
                vertical_flow = 'balanced'
            
            # Estimate reading order
            if horizontal_flow == 'balanced' and vertical_flow == 'top_heavy':
                reading_order = 'top_to_bottom'
            elif horizontal_flow == 'left_heavy':
                reading_order = 'left_to_right'
            else:
                reading_order = 'standard'  # Default assumption
            
            return {
                'horizontal_flow': horizontal_flow,
                'vertical_flow': vertical_flow,
                'estimated_reading_order': reading_order
            }
            
        except Exception:
            return {'estimated_reading_order': 'standard'}
    
    def _classify_document_type(self, image):
        """Classify the type of document"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            height, width = gray.shape
            
            # Basic classification based on content analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # Text density estimation
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_like_contours = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # Text-like size
                    text_like_contours += 1
            
            text_density = text_like_contours / max(1, len(contours))
            
            # Classification logic
            if edge_density > 0.2 and text_density < 0.3:
                return 'form_or_invoice'
            elif text_density > 0.7:
                return 'text_document'
            elif edge_density > 0.1:
                return 'mixed_content'
            elif height > width * 1.2:
                return 'letter_or_report'
            else:
                return 'general_document'
                
        except Exception:
            return 'unknown'
    
    def _detect_form_fields(self, gray):
        """Detect form fields (simplified)"""
        try:
            # Look for rectangular regions that might be form fields
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            form_fields = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Reasonable form field size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Form fields are often rectangular with specific aspect ratios
                    if 1.5 < aspect_ratio < 10:
                        form_fields.append({'x': x, 'y': y, 'width': w, 'height': h})
            
            return form_fields
            
        except Exception:
            return []
    
    def _detect_signatures(self, gray):
        """Detect potential signature regions (simplified)"""
        try:
            # Look for irregular, handwritten-like regions
            # This is a very basic implementation
            edges = cv2.Canny(gray, 30, 100)
            
            # Use different morphological operations to detect handwriting
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            signatures = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 20000:  # Signature-like size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Signatures often have specific characteristics
                    if 1.5 < aspect_ratio < 6:
                        signatures.append({'x': x, 'y': y, 'width': w, 'height': h})
            
            return signatures
            
        except Exception:
            return []
    
    def _detect_stamps(self, gray):
        """Detect potential stamp regions (simplified)"""
        try:
            # Look for circular or rectangular regions with high edge density
            edges = cv2.Canny(gray, 50, 150)
            
            # Use circular Hough transform to detect circular stamps
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                     param1=50, param2=30, minRadius=20, maxRadius=100)
            
            stamps = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    stamps.append({'x': x-r, 'y': y-r, 'width': 2*r, 'height': 2*r, 'type': 'circular'})
            
            return stamps
            
        except Exception:
            return []
    
    def _detect_barcodes(self, gray):
        """Detect potential barcode regions (simplified)"""
        try:
            # Look for patterns with regular vertical lines
            # This is a very basic implementation
            
            # Use morphological operations to detect barcode-like patterns
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            processed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            barcodes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Barcodes have high aspect ratios
                    if aspect_ratio > 3:
                        barcodes.append({'x': x, 'y': y, 'width': w, 'height': h})
            
            return barcodes
            
        except Exception:
            return []
    
    def _detect_logos(self, gray):
        """Detect potential logo regions (simplified)"""
        try:
            # Look for dense, compact regions that might be logos
            edges = cv2.Canny(gray, 50, 150)
            
            # Use morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            logos = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Logo-like size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Logos are often roughly square or rectangular
                    if 0.5 < aspect_ratio < 3:
                        logos.append({'x': x, 'y': y, 'width': w, 'height': h})
            
            return logos
            
        except Exception:
            return []
    
    def _identify_special_features(self, image):
        """Identify special features in the document"""
        try:
            features = []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Check for watermarks (very light text)
            # Look for very faint patterns
            enhanced = cv2.equalizeHist(gray)
            diff = cv2.absdiff(gray, enhanced)
            if np.mean(diff) > 10:
                features.append('possible_watermark')
            
            # Check for ruled lines (notebook paper, forms)
            # Look for regular horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            if np.sum(horizontal_lines > 0) > gray.shape[0] * gray.shape[1] * 0.01:
                features.append('ruled_lines')
            
            # Check for very high contrast (might be scanned text)
            if gray.std() > 80:
                features.append('high_contrast_text')
            
            return features
            
        except Exception:
            return []
    
    def _generate_processing_recommendations(self, metadata):
        """Generate processing recommendations based on metadata analysis"""
        
        recommendations = {
            'ocr_settings': [],
            'preprocessing_steps': [],
            'quality_flags': [],
            'special_handling': []
        }
        
        try:
            # Quality-based recommendations
            quality = metadata.get('quality_analysis', {})
            if quality.get('quality_flags', {}).get('is_blurred'):
                recommendations['preprocessing_steps'].append('apply_sharpening_filter')
                recommendations['quality_flags'].append('low_sharpness_detected')
            
            if quality.get('quality_flags', {}).get('is_low_contrast'):
                recommendations['preprocessing_steps'].append('enhance_contrast')
                recommendations['quality_flags'].append('low_contrast_detected')
            
            if quality.get('quality_flags', {}).get('is_noisy'):
                recommendations['preprocessing_steps'].append('noise_reduction')
                recommendations['quality_flags'].append('noise_detected')
            
            # Content-based recommendations
            content = metadata.get('content_analysis', {})
            if content.get('table_likelihood_score', 0) > 50:
                recommendations['ocr_settings'].append('enable_table_detection')
                recommendations['special_handling'].append('structured_data_extraction')
            
            if content.get('text_density', 0) > 70:
                recommendations['ocr_settings'].append('optimize_for_dense_text')
            
            # Document type recommendations
            doc_analysis = metadata.get('document_analysis', {})
            if doc_analysis.get('contains_form_fields'):
                recommendations['special_handling'].append('form_field_extraction')
            
            if doc_analysis.get('contains_signatures'):
                recommendations['special_handling'].append('preserve_signature_regions')
            
            if doc_analysis.get('contains_barcodes'):
                recommendations['special_handling'].append('barcode_extraction')
            
            # Layout-based recommendations
            layout = metadata.get('layout_analysis', {})
            if layout.get('page_structure', {}).get('estimated_columns', 1) > 1:
                recommendations['ocr_settings'].append('multi_column_layout')
            
            if abs(layout.get('estimated_skew_angle', 0)) > 1:
                recommendations['preprocessing_steps'].append('skew_correction_needed')
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not generate recommendations: {str(e)}")
        
        return recommendations
    
    def _save_metadata_sidecar(self, metadata, output_folder, filename):
        """Save metadata as JSON sidecar file"""
        
        sidecar_path = os.path.join(output_folder, f"{filename}{self.config['sidecar_suffix']}.json")
        
        # Add extraction configuration to metadata
        metadata['extraction_config'] = {
            'task_version': '1.0',
            'extraction_timestamp': datetime.now().isoformat(),
            'enabled_analyzers': {
                'file_metadata': self.config['analyze_resolution'],
                'content_analysis': self.config['analyze_text_density'],
                'quality_analysis': self.config['analyze_image_quality'],
                'layout_analysis': self.config['analyze_page_orientation'],
                'document_features': self.config['detect_document_type']
            }
        }
        
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return sidecar_path
    
    def _generate_thumbnail(self, image, output_folder, filename):
        """Generate thumbnail image"""
        
        try:
            thumbnail_path = os.path.join(output_folder, f"{filename}_thumbnail.jpg")
            
            # Resize image to thumbnail size
            height, width = image.shape[:2]
            thumb_w, thumb_h = self.config['thumbnail_size']
            
            # Calculate aspect ratio preserving dimensions
            aspect_ratio = width / height
            if aspect_ratio > 1:  # Landscape
                new_w = thumb_w
                new_h = int(thumb_w / aspect_ratio)
            else:  # Portrait
                new_h = thumb_h
                new_w = int(thumb_h * aspect_ratio)
            
            # Resize and save
            thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(thumbnail_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            return thumbnail_path
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not generate thumbnail: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    task = MetadataExtractionTask()
    print("Metadata Extraction Task initialized")
    print("Configuration:", task.config)
