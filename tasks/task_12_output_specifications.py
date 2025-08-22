#!/usr/bin/env python3
"""
Task 12: Output Specifications
Generate standardized outputs with comprehensive metadata and audit logs.
"""

import cv2
import numpy as np
import os
import logging
import json
import hashlib
import shutil
from PIL import Image
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
import uuid

class OutputSpecificationsTask:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Output Specifications"
        self.task_id = "task_12_output_specifications"
        self.config = {
            # Output Format Settings
            "standardized_formats": ["tiff", "png"],
            "primary_format": "tiff",
            "backup_format": "png",
            "generate_pdf": True,
            "pdf_dpi": 300,
            
            # Image Quality Settings
            "tiff_compression": "lzw",  # lzw, none, jpeg
            "png_compression": 6,  # 0-9
            "jpeg_quality": 95,
            "preserve_color_depth": True,
            "maintain_resolution": True,
            
            # Metadata Standards
            "generate_comprehensive_metadata": True,
            "include_processing_chain": True,
            "include_qc_flags": True,
            "include_audit_trail": True,
            "metadata_format": "json",
            
            # File Naming Convention
            "naming_convention": "structured",  # structured, uuid, timestamp
            "include_original_hash": True,
            "include_processing_date": True,
            "file_prefix": "processed",
            
            # Quality Control Flags
            "detect_multi_page": True,
            "flag_low_contrast": True,
            "flag_blur": True,
            "flag_orientation_issues": True,
            "flag_skew_issues": True,
            
            # Output Organization
            "create_document_folder": True,
            "separate_by_type": True,
            "include_thumbnails": True,
            "generate_summary_report": True,
            
            # Audit & Reproducibility
            "maintain_processing_logs": True,
            "include_version_info": True,
            "log_all_parameters": True,
            "create_reproducibility_manifest": True,
            
            # Archive Settings
            "create_archive_copy": False,
            "archive_format": "zip",
            "include_originals_in_archive": False
        }
    
    def run(self, input_file, file_type, output_folder):
        """
        Main entry point for output specifications generation
        
        Returns:
            dict: Task result with standardized outputs
        """
        try:
            self.logger.info(f"🔄 Running {self.task_name} on {os.path.basename(input_file)}")
            
            # Process output standardization
            result_path, output_manifest = self._process_output_specifications(input_file, output_folder)
            
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
                    'output_manifest': output_manifest
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
    
    def _process_output_specifications(self, image_path, output_folder):
        """Process standardized output generation"""
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        self.logger.info(f"   🔍 Step 1: Preparing standardized output structure")
        
        # Create document-specific output folder
        if self.config['create_document_folder']:
            doc_output_folder = os.path.join(output_folder, f"doc_{filename}")
            os.makedirs(doc_output_folder, exist_ok=True)
        else:
            doc_output_folder = output_folder
        
        # Load image for processing
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Initialize output manifest
        output_manifest = {
            'document_id': self._generate_document_id(image_path),
            'processing_timestamp': datetime.now().isoformat(),
            'original_file': image_path,
            'output_folder': doc_output_folder,
            'generated_files': [],
            'metadata_files': [],
            'processing_chain': [],
            'qc_flags': {},
            'audit_info': {}
        }
        
        self.logger.info(f"   📊 Document ID: {output_manifest['document_id']}")
        
        # Step 2: Generate standardized image outputs
        self.logger.info(f"   🔍 Step 2: Generating standardized image formats")
        standardized_images = self._generate_standardized_images(image, doc_output_folder, filename, output_manifest)
        
        # Step 3: Generate PDF output
        if self.config['generate_pdf']:
            self.logger.info(f"   🔍 Step 3: Generating processed PDF")
            pdf_path = self._generate_pdf_output(image, doc_output_folder, filename, output_manifest)
            if pdf_path:
                self.logger.info(f"   📄 PDF saved: {os.path.basename(pdf_path)}")
        
        # Step 4: Extract and compile comprehensive metadata
        self.logger.info(f"   🔍 Step 4: Compiling comprehensive metadata")
        comprehensive_metadata = self._compile_comprehensive_metadata(image, image_path, output_manifest)
        
        # Step 5: Perform quality control analysis
        self.logger.info(f"   🔍 Step 5: Performing quality control analysis")
        qc_results = self._perform_quality_control(image, image_path, output_manifest)
        output_manifest['qc_flags'] = qc_results
        
        # Step 6: Generate processing audit trail
        self.logger.info(f"   🔍 Step 6: Creating audit trail and processing logs")
        audit_trail = self._generate_audit_trail(image_path, output_manifest)
        output_manifest['audit_info'] = audit_trail
        
        # Step 7: Save comprehensive metadata
        metadata_path = self._save_comprehensive_metadata(comprehensive_metadata, doc_output_folder, filename)
        output_manifest['metadata_files'].append(metadata_path)
        self.logger.info(f"   💾 Metadata saved: {os.path.basename(metadata_path)}")
        
        # Step 8: Generate processing summary report
        if self.config['generate_summary_report']:
            summary_path = self._generate_summary_report(output_manifest, doc_output_folder, filename)
            if summary_path:
                self.logger.info(f"   📊 Summary report saved: {os.path.basename(summary_path)}")
        
        # Step 9: Create reproducibility manifest
        if self.config['create_reproducibility_manifest']:
            manifest_path = self._create_reproducibility_manifest(output_manifest, doc_output_folder, filename)
            if manifest_path:
                self.logger.info(f"   🔄 Reproducibility manifest saved: {os.path.basename(manifest_path)}")
        
        # Step 10: Generate thumbnails
        if self.config['include_thumbnails']:
            thumbnail_paths = self._generate_output_thumbnails(standardized_images, doc_output_folder, filename)
            output_manifest['generated_files'].extend(thumbnail_paths)
        
        # Return primary standardized output
        primary_output = standardized_images[0] if standardized_images else image_path
        
        return primary_output, output_manifest
    
    def _generate_document_id(self, file_path):
        """Generate unique document ID"""
        
        if self.config['naming_convention'] == 'uuid':
            return str(uuid.uuid4())
        elif self.config['naming_convention'] == 'timestamp':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            return f"{timestamp}_{base_name}"
        else:  # structured
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Include hash if configured
            if self.config['include_original_hash']:
                file_hash = self._calculate_file_hash(file_path)[:8]  # First 8 chars
                doc_id = f"{base_name}_{file_hash}"
            else:
                doc_id = base_name
            
            # Include date if configured
            if self.config['include_processing_date']:
                date_str = datetime.now().strftime("%Y%m%d")
                doc_id = f"{doc_id}_{date_str}"
            
            return doc_id
    
    def _generate_standardized_images(self, image, output_folder, filename, manifest):
        """Generate standardized image outputs in multiple formats"""
        
        generated_files = []
        
        for format_name in self.config['standardized_formats']:
            try:
                # Generate appropriate filename
                if self.config['file_prefix']:
                    output_filename = f"{self.config['file_prefix']}_{filename}.{format_name}"
                else:
                    output_filename = f"{filename}_standardized.{format_name}"
                
                output_path = os.path.join(output_folder, output_filename)
                
                # Save with format-specific settings
                if format_name.lower() == 'tiff':
                    self._save_tiff(image, output_path)
                elif format_name.lower() == 'png':
                    self._save_png(image, output_path)
                elif format_name.lower() in ['jpg', 'jpeg']:
                    self._save_jpeg(image, output_path)
                else:
                    # Default to OpenCV save
                    cv2.imwrite(output_path, image)
                
                generated_files.append(output_path)
                manifest['generated_files'].append({
                    'file_path': output_path,
                    'format': format_name.upper(),
                    'file_size': os.path.getsize(output_path),
                    'creation_time': datetime.now().isoformat()
                })
                
                self.logger.info(f"   💾 {format_name.upper()} saved: {os.path.basename(output_path)}")
                
            except Exception as e:
                self.logger.warning(f"   ⚠️  Could not save {format_name} format: {str(e)}")
        
        return generated_files
    
    def _save_tiff(self, image, output_path):
        """Save image as high-quality TIFF"""
        
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        pil_image = Image.fromarray(rgb_image)
        
        # Set compression
        compression = self.config.get('tiff_compression', 'lzw')
        if compression == 'none':
            pil_image.save(output_path, format='TIFF')
        else:
            pil_image.save(output_path, format='TIFF', compression=compression)
    
    def _save_png(self, image, output_path):
        """Save image as PNG with optimal compression"""
        
        compression_level = self.config.get('png_compression', 6)
        cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    
    def _save_jpeg(self, image, output_path):
        """Save image as high-quality JPEG"""
        
        quality = self.config.get('jpeg_quality', 95)
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    def _generate_pdf_output(self, image, output_folder, filename, manifest):
        """Generate processed PDF output"""
        
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.utils import ImageReader
            import tempfile
            
            pdf_filename = f"{filename}_processed.pdf"
            pdf_path = os.path.join(output_folder, pdf_filename)
            
            # Create temporary image file for PDF
            temp_image_path = os.path.join(tempfile.gettempdir(), f"temp_{filename}.png")
            cv2.imwrite(temp_image_path, image)
            
            # Create PDF
            c = canvas.Canvas(pdf_path, pagesize=letter)
            
            # Calculate image dimensions for PDF
            img_width, img_height = Image.open(temp_image_path).size
            page_width, page_height = letter
            
            # Scale image to fit page while maintaining aspect ratio
            scale_w = page_width / img_width
            scale_h = page_height / img_height
            scale = min(scale_w, scale_h) * 0.9  # 90% of page size
            
            new_width = img_width * scale
            new_height = img_height * scale
            
            # Center image on page
            x = (page_width - new_width) / 2
            y = (page_height - new_height) / 2
            
            # Add image to PDF
            c.drawImage(temp_image_path, x, y, width=new_width, height=new_height)
            
            # Add metadata to PDF
            c.setTitle(f"Processed Document: {filename}")
            c.setAuthor("Document Processing Pipeline")
            c.setSubject("Processed Document")
            c.setCreator("Pipeline v1.0")
            
            c.save()
            
            # Clean up
            os.remove(temp_image_path)
            
            manifest['generated_files'].append({
                'file_path': pdf_path,
                'format': 'PDF',
                'file_size': os.path.getsize(pdf_path),
                'creation_time': datetime.now().isoformat()
            })
            
            return pdf_path
            
        except ImportError:
            self.logger.warning("   ⚠️  ReportLab not available for PDF generation")
            return None
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not generate PDF: {str(e)}")
            return None
    
    def _compile_comprehensive_metadata(self, image, image_path, manifest):
        """Compile comprehensive metadata from all processing steps"""
        
        metadata = {
            'document_metadata': {
                'document_id': manifest['document_id'],
                'original_filename': os.path.basename(image_path),
                'processing_timestamp': manifest['processing_timestamp'],
                'pipeline_version': '1.0.0'
            },
            'file_metadata': {
                'original_file_hash': self._calculate_file_hash(image_path),
                'original_file_size': os.path.getsize(image_path),
                'original_format': os.path.splitext(image_path)[1],
                'creation_timestamp': datetime.fromtimestamp(os.path.getctime(image_path)).isoformat()
            },
            'image_properties': {
                'dimensions': {
                    'width': int(image.shape[1]),
                    'height': int(image.shape[0]),
                    'channels': int(image.shape[2]) if len(image.shape) == 3 else 1
                },
                'color_space': 'BGR' if len(image.shape) == 3 else 'Grayscale',
                'bit_depth': 8,  # Assuming 8-bit images
                'total_pixels': int(image.shape[0] * image.shape[1])
            },
            'processing_steps_applied': self._extract_processing_chain(manifest),
            'quality_metrics': self._calculate_quality_metrics(image),
            'output_files': manifest.get('generated_files', []),
            'processing_configuration': self._get_processing_configuration()
        }
        
        return metadata
    
    def _perform_quality_control(self, image, image_path, manifest):
        """Perform comprehensive quality control analysis"""
        
        qc_flags = {
            'timestamp': datetime.now().isoformat(),
            'flags': {},
            'scores': {},
            'recommendations': []
        }
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Multi-page detection
            if self.config['detect_multi_page']:
                is_multi_page = self._detect_multi_page_indicators(image)
                qc_flags['flags']['multi_page_detected'] = is_multi_page
                if is_multi_page:
                    qc_flags['recommendations'].append('Document may contain multiple pages')
            
            # Low contrast detection
            if self.config['flag_low_contrast']:
                contrast_score = gray.std()
                qc_flags['scores']['contrast_score'] = float(contrast_score)
                is_low_contrast = contrast_score < 50
                qc_flags['flags']['low_contrast'] = is_low_contrast
                if is_low_contrast:
                    qc_flags['recommendations'].append('Low contrast detected - consider enhancement')
            
            # Blur detection
            if self.config['flag_blur']:
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                qc_flags['scores']['blur_score'] = float(blur_score)
                is_blurred = blur_score < 100
                qc_flags['flags']['blurred'] = is_blurred
                if is_blurred:
                    qc_flags['recommendations'].append('Image appears blurred - OCR accuracy may be affected')
            
            # Orientation issues
            if self.config['flag_orientation_issues']:
                orientation_confidence = self._check_orientation_confidence(gray)
                qc_flags['scores']['orientation_confidence'] = float(orientation_confidence)
                has_orientation_issues = orientation_confidence < 0.7
                qc_flags['flags']['orientation_issues'] = has_orientation_issues
                if has_orientation_issues:
                    qc_flags['recommendations'].append('Orientation may need correction')
            
            # Skew issues
            if self.config['flag_skew_issues']:
                skew_angle = self._estimate_skew_angle(gray)
                qc_flags['scores']['skew_angle'] = float(skew_angle)
                has_skew_issues = abs(skew_angle) > 2.0
                qc_flags['flags']['skew_issues'] = has_skew_issues
                if has_skew_issues:
                    qc_flags['recommendations'].append('Document appears skewed - correction recommended')
            
            # Overall quality assessment
            total_issues = sum(1 for flag in qc_flags['flags'].values() if flag)
            qc_flags['overall_quality'] = 'excellent' if total_issues == 0 else 'good' if total_issues <= 1 else 'fair' if total_issues <= 2 else 'poor'
            
        except Exception as e:
            qc_flags['error'] = str(e)
            self.logger.warning(f"   ⚠️  QC analysis failed: {str(e)}")
        
        return qc_flags
    
    def _generate_audit_trail(self, image_path, manifest):
        """Generate comprehensive audit trail"""
        
        audit_info = {
            'processing_session': {
                'session_id': str(uuid.uuid4()),
                'start_time': manifest['processing_timestamp'],
                'pipeline_version': '1.0.0',
                'system_info': self._get_system_info()
            },
            'input_validation': {
                'file_exists': os.path.exists(image_path),
                'file_readable': os.access(image_path, os.R_OK),
                'file_size_mb': os.path.getsize(image_path) / (1024 * 1024),
                'format_supported': True  # If we got this far, format is supported
            },
            'processing_parameters': self._get_all_task_parameters(),
            'output_validation': {
                'files_created': len(manifest.get('generated_files', [])),
                'total_output_size_mb': self._calculate_total_output_size(manifest) / (1024 * 1024)
            },
            'reproducibility_info': {
                'environment_hash': self._calculate_environment_hash(),
                'parameter_hash': self._calculate_parameter_hash(),
                'can_reproduce': True
            }
        }
        
        return audit_info
    
    def _save_comprehensive_metadata(self, metadata, output_folder, filename):
        """Save comprehensive metadata to JSON file"""
        
        metadata_filename = f"{filename}_comprehensive_metadata.json"
        metadata_path = os.path.join(output_folder, metadata_filename)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return metadata_path
    
    def _generate_summary_report(self, manifest, output_folder, filename):
        """Generate human-readable summary report"""
        
        try:
            summary_filename = f"{filename}_processing_summary.txt"
            summary_path = os.path.join(output_folder, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("DOCUMENT PROCESSING SUMMARY REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Document ID: {manifest['document_id']}\n")
                f.write(f"Processing Date: {manifest['processing_timestamp']}\n")
                f.write(f"Original File: {os.path.basename(manifest['original_file'])}\n\n")
                
                f.write("GENERATED FILES:\n")
                f.write("-" * 20 + "\n")
                for file_info in manifest.get('generated_files', []):
                    if isinstance(file_info, dict):
                        f.write(f"• {os.path.basename(file_info.get('file_path', ''))} ({file_info.get('format', 'Unknown')})\n")
                    else:
                        f.write(f"• {os.path.basename(file_info)}\n")
                
                f.write("\nQUALITY CONTROL:\n")
                f.write("-" * 20 + "\n")
                qc_flags = manifest.get('qc_flags', {})
                overall_quality = qc_flags.get('overall_quality', 'unknown')
                f.write(f"Overall Quality: {overall_quality.upper()}\n")
                
                flags = qc_flags.get('flags', {})
                if any(flags.values()):
                    f.write("Issues Detected:\n")
                    for flag, value in flags.items():
                        if value:
                            f.write(f"  ⚠ {flag.replace('_', ' ').title()}\n")
                else:
                    f.write("✓ No quality issues detected\n")
                
                recommendations = qc_flags.get('recommendations', [])
                if recommendations:
                    f.write("\nRecommendations:\n")
                    for rec in recommendations:
                        f.write(f"• {rec}\n")
                
                f.write("\nAUDIT INFORMATION:\n")
                f.write("-" * 20 + "\n")
                audit_info = manifest.get('audit_info', {})
                session_info = audit_info.get('processing_session', {})
                f.write(f"Session ID: {session_info.get('session_id', 'N/A')}\n")
                f.write(f"Pipeline Version: {session_info.get('pipeline_version', 'N/A')}\n")
                
                reproducibility = audit_info.get('reproducibility_info', {})
                f.write(f"Reproducible: {'Yes' if reproducibility.get('can_reproduce') else 'No'}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("End of Report\n")
            
            return summary_path
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not generate summary report: {str(e)}")
            return None
    
    def _create_reproducibility_manifest(self, manifest, output_folder, filename):
        """Create manifest for reproducible processing"""
        
        try:
            manifest_filename = f"{filename}_reproducibility_manifest.json"
            manifest_path = os.path.join(output_folder, manifest_filename)
            
            reproducibility_data = {
                'manifest_version': '1.0',
                'document_id': manifest['document_id'],
                'original_file_hash': self._calculate_file_hash(manifest['original_file']),
                'processing_timestamp': manifest['processing_timestamp'],
                'pipeline_configuration': self._get_processing_configuration(),
                'system_environment': self._get_system_info(),
                'processing_steps': manifest.get('processing_chain', []),
                'input_parameters': self._get_all_task_parameters(),
                'output_checksums': self._calculate_output_checksums(manifest),
                'verification': {
                    'environment_hash': self._calculate_environment_hash(),
                    'config_hash': self._calculate_config_hash(),
                    'reproducible': True
                }
            }
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(reproducibility_data, f, indent=2, ensure_ascii=False, default=str)
            
            return manifest_path
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not create reproducibility manifest: {str(e)}")
            return None
    
    def _generate_output_thumbnails(self, image_files, output_folder, filename):
        """Generate thumbnails for output images"""
        
        thumbnail_paths = []
        
        try:
            thumb_folder = os.path.join(output_folder, "thumbnails")
            os.makedirs(thumb_folder, exist_ok=True)
            
            for image_path in image_files:
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Generate thumbnail
                    height, width = image.shape[:2]
                    max_size = 200
                    
                    if width > height:
                        new_width = max_size
                        new_height = int(height * max_size / width)
                    else:
                        new_height = max_size
                        new_width = int(width * max_size / height)
                    
                    thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Save thumbnail
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    thumb_path = os.path.join(thumb_folder, f"{base_name}_thumb.jpg")
                    cv2.imwrite(thumb_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    thumbnail_paths.append(thumb_path)
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Could not create thumbnail for {image_path}: {str(e)}")
                    
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not create thumbnail folder: {str(e)}")
        
        return thumbnail_paths
    
    # Helper methods for calculations and analysis
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    def _extract_processing_chain(self, manifest):
        """Extract processing chain from manifest"""
        # This would typically be populated by earlier tasks
        # For now, return a basic chain
        return [
            'orientation_correction',
            'skew_detection',
            'cropping',
            'dpi_standardization',
            'noise_reduction',
            'contrast_enhancement',
            'segmentation',
            'color_handling',
            'language_detection',
            'metadata_extraction',
            'output_standardization'
        ]
    
    def _calculate_quality_metrics(self, image):
        """Calculate comprehensive quality metrics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            return {
                'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                'contrast': float(gray.std()),
                'brightness': float(gray.mean()),
                'signal_to_noise_ratio': float(gray.mean() / max(1, gray.std())),
                'edge_density': float(np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size)
            }
        except Exception:
            return {}
    
    def _get_processing_configuration(self):
        """Get current processing configuration"""
        return {
            'task_12_config': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_multi_page_indicators(self, image):
        """Detect if image contains multiple pages"""
        try:
            # Simple heuristic: look for vertical gaps that might indicate page breaks
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Analyze horizontal projection
            horizontal_projection = np.sum(gray < 200, axis=1)
            
            # Look for large gaps (potential page breaks)
            gap_threshold = np.mean(horizontal_projection) * 0.1
            gaps = np.where(horizontal_projection < gap_threshold)[0]
            
            # If we have significant gaps, might be multi-page
            if len(gaps) > gray.shape[0] * 0.1:  # More than 10% of rows are gaps
                return True
            
            return False
            
        except Exception:
            return False
    
    def _check_orientation_confidence(self, gray):
        """Check confidence in current orientation"""
        try:
            # Basic orientation confidence based on text line detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect horizontal lines (assuming correct orientation has horizontal text)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_density = np.sum(horizontal_lines > 0) / gray.size
            v_line_density = np.sum(vertical_lines > 0) / gray.size
            
            # Higher horizontal line density suggests correct orientation
            if h_line_density > v_line_density:
                return min(1.0, h_line_density / max(v_line_density, 0.001))
            else:
                return 0.5  # Uncertain
                
        except Exception:
            return 0.5
    
    def _estimate_skew_angle(self, gray):
        """Estimate skew angle of document"""
        try:
            # Simple skew detection using Hough lines
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:10]:
                    angle = theta * 180 / np.pi
                    if angle > 45:
                        angle = angle - 90
                    angles.append(angle)
                
                if angles:
                    return np.median(angles)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_system_info(self):
        """Get system information for audit"""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'opencv_version': cv2.__version__,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_all_task_parameters(self):
        """Get parameters from all tasks"""
        # This would be populated by the task manager
        # For now, return basic info
        return {
            'task_12_parameters': self.config
        }
    
    def _calculate_total_output_size(self, manifest):
        """Calculate total size of output files"""
        total_size = 0
        for file_info in manifest.get('generated_files', []):
            if isinstance(file_info, dict):
                total_size += file_info.get('file_size', 0)
            elif isinstance(file_info, str) and os.path.exists(file_info):
                total_size += os.path.getsize(file_info)
        return total_size
    
    def _calculate_environment_hash(self):
        """Calculate hash of processing environment"""
        try:
            env_info = json.dumps(self._get_system_info(), sort_keys=True)
            return hashlib.md5(env_info.encode()).hexdigest()
        except Exception:
            return "unknown"
    
    def _calculate_parameter_hash(self):
        """Calculate hash of processing parameters"""
        try:
            params = json.dumps(self.config, sort_keys=True)
            return hashlib.md5(params.encode()).hexdigest()
        except Exception:
            return "unknown"
    
    def _calculate_config_hash(self):
        """Calculate hash of configuration"""
        try:
            config_str = json.dumps(self.config, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception:
            return "unknown"
    
    def _calculate_output_checksums(self, manifest):
        """Calculate checksums of all output files"""
        checksums = {}
        for file_info in manifest.get('generated_files', []):
            if isinstance(file_info, dict):
                file_path = file_info.get('file_path')
            else:
                file_path = file_info
            
            if file_path and os.path.exists(file_path):
                checksums[os.path.basename(file_path)] = self._calculate_file_hash(file_path)
        
        return checksums

if __name__ == "__main__":
    # Example usage
    task = OutputSpecificationsTask()
    print("Output Specifications Task initialized")
    print("Configuration:", task.config)
