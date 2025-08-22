#!/usr/bin/env python3
"""
Task 10: Language & Script Detection
Detect dominant script and language for multilingual OCR path optimization.
"""

import cv2
import numpy as np
import os
import logging
import json
import re
from PIL import Image
import shutil
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import pytesseract
from collections import Counter

class LanguageDetectionTask:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Language & Script Detection"
        self.task_id = "task_10_language_detection"
        self.config = {
            # Script Detection Settings
            "enable_script_detection": True,
            "supported_scripts": ["latin", "vietnamese", "numeric", "mrz"],
            "confidence_threshold": 0.7,
            "min_text_area": 100,
            
            # Language Detection Settings
            "enable_language_detection": True,
            "supported_languages": ["eng", "vie", "fra", "spa", "deu"],
            "tesseract_config": "--oem 3 --psm 6",
            "min_word_length": 3,
            
            # Vietnamese Specific
            "vietnamese_patterns": [
                r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
                r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]'
            ],
            
            # MRZ Detection
            "mrz_patterns": [
                r'^[A-Z0-9<]{30,44}$',  # Standard MRZ line
                r'^P<[A-Z]{3}[A-Z<]{39}$',  # Passport MRZ
                r'^[ACIV][A-Z0-9<]{30,43}$'  # ID/Visa MRZ
            ],
            
            # Numeric Detection
            "numeric_threshold": 0.6,  # 60% numeric content
            "date_patterns": [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}'
            ],
            
            # Handwriting Detection
            "enable_handwriting_detection": True,
            "handwriting_threshold": 0.3,
            "preserve_handwritten_zones": True,
            
            # Output Settings
            "save_language_report": True,
            "create_script_overlay": True,
            "lang_suffix": "_language_analysis",
            "overlay_alpha": 0.3
        }
    
    def run(self, input_file, file_type, output_folder):
        """
        Main entry point for language and script detection
        
        Returns:
            dict: Task result with detected languages and scripts
        """
        try:
            self.logger.info(f"🔄 Running {self.task_name} on {os.path.basename(input_file)}")
            
            # Load and analyze image
            image = cv2.imread(input_file)
            if image is None:
                raise ValueError(f"Could not load image: {input_file}")
            
            # Process language and script detection
            result_path, analysis_results = self._process_language_detection(input_file, output_folder)
            
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
                    'language_analysis': analysis_results
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
    
    def _process_language_detection(self, image_path, output_folder):
        """Process language and script detection with comprehensive analysis"""
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        self.logger.info(f"   🔍 Step 1: Loading image and extracting text regions")
        
        # Load image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        self.logger.info(f"   📊 Image: {filename} ({width}x{height})")
        
        # Extract text using OCR for analysis
        extracted_text = self._extract_text_for_analysis(image)
        self.logger.info(f"   📝 Extracted {len(extracted_text)} characters for analysis")
        
        # Step 2: Script Detection
        self.logger.info(f"   🔍 Step 2: Detecting dominant scripts")
        script_analysis = self._detect_scripts(extracted_text, image)
        
        # Step 3: Language Detection
        self.logger.info(f"   🔍 Step 3: Detecting languages")
        language_analysis = self._detect_languages(extracted_text, image)
        
        # Step 4: Special Zone Detection (MRZ, Handwriting)
        self.logger.info(f"   🔍 Step 4: Detecting special zones (MRZ, handwriting)")
        special_zones = self._detect_special_zones(extracted_text, image)
        
        # Combine all analysis results
        analysis_results = {
            'dominant_script': script_analysis['dominant_script'],
            'script_confidence': script_analysis['confidence'],
            'detected_scripts': script_analysis['all_scripts'],
            'dominant_language': language_analysis['dominant_language'],
            'language_confidence': language_analysis['confidence'],
            'detected_languages': language_analysis['all_languages'],
            'special_zones': special_zones,
            'multilingual_flag': len(language_analysis['all_languages']) > 1,
            'handwriting_zones': special_zones.get('handwriting_regions', []),
            'text_content_sample': extracted_text[:200] if extracted_text else "",
            'processing_recommendation': self._get_processing_recommendation(script_analysis, language_analysis, special_zones)
        }
        
        # Print detected language on result (as requested)
        dominant_lang = analysis_results['dominant_language']
        confidence = analysis_results['language_confidence']
        self.logger.info(f"   🌍 DETECTED LANGUAGE: {dominant_lang.upper()} (confidence: {confidence:.1f}%)")
        
        # Step 5: Generate outputs
        self.logger.info(f"   🔍 Step 5: Generating analysis outputs")
        
        # Save language analysis report
        if self.config['save_language_report']:
            report_path = self._generate_language_report(analysis_results, output_folder, filename)
            self.logger.info(f"   📊 Language report saved: {os.path.basename(report_path)}")
        
        # Create script overlay visualization
        if self.config['create_script_overlay']:
            overlay_path = self._create_script_overlay(image, analysis_results, output_folder, filename)
            if overlay_path:
                self.logger.info(f"   🖼️  Script overlay saved: {os.path.basename(overlay_path)}")
        
        return image_path, analysis_results
    
    def _extract_text_for_analysis(self, image):
        """Extract text from image using OCR for language analysis"""
        try:
            # Convert to RGB for pytesseract
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Extract text with confidence scores
            text = pytesseract.image_to_string(pil_image, config=self.config['tesseract_config'])
            
            return text.strip()
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  OCR extraction failed: {str(e)}")
            return ""
    
    def _detect_scripts(self, text, image):
        """Detect dominant scripts in the text"""
        
        if not text:
            return {
                'dominant_script': 'unknown',
                'confidence': 0.0,
                'all_scripts': {}
            }
        
        script_scores = {
            'latin': 0,
            'vietnamese': 0,
            'numeric': 0,
            'mrz': 0
        }
        
        total_chars = len(text)
        
        # Count Latin characters (basic English alphabet)
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        script_scores['latin'] = latin_count / total_chars if total_chars > 0 else 0
        
        # Count Vietnamese characters (with diacritics)
        vietnamese_count = 0
        for pattern in self.config['vietnamese_patterns']:
            vietnamese_count += len(re.findall(pattern, text))
        script_scores['vietnamese'] = vietnamese_count / total_chars if total_chars > 0 else 0
        
        # Count numeric characters
        numeric_count = len(re.findall(r'\d', text))
        script_scores['numeric'] = numeric_count / total_chars if total_chars > 0 else 0
        
        # Check for MRZ patterns
        mrz_detected = False
        for pattern in self.config['mrz_patterns']:
            if re.search(pattern, text, re.MULTILINE):
                mrz_detected = True
                break
        script_scores['mrz'] = 1.0 if mrz_detected else 0.0
        
        # Determine dominant script
        dominant_script = max(script_scores, key=script_scores.get)
        confidence = script_scores[dominant_script] * 100
        
        return {
            'dominant_script': dominant_script,
            'confidence': confidence,
            'all_scripts': {k: v * 100 for k, v in script_scores.items()}
        }
    
    def _detect_languages(self, text, image):
        """Detect languages in the text"""
        
        if not text:
            return {
                'dominant_language': 'unknown',
                'confidence': 0.0,
                'all_languages': {}
            }
        
        language_scores = {}
        
        # Use multiple detection methods
        
        # Method 1: Tesseract language detection
        try:
            for lang_code in self.config['supported_languages']:
                try:
                    # Test OCR with specific language
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    
                    lang_text = pytesseract.image_to_string(
                        pil_image, 
                        lang=lang_code,
                        config=f"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                    )
                    
                    # Calculate confidence based on text quality
                    confidence = self._calculate_language_confidence(lang_text, lang_code)
                    language_scores[lang_code] = confidence
                    
                except Exception:
                    language_scores[lang_code] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"   ⚠️  Tesseract language detection failed: {str(e)}")
        
        # Method 2: Pattern-based detection
        pattern_scores = self._detect_language_patterns(text)
        
        # Combine scores
        for lang, score in pattern_scores.items():
            if lang in language_scores:
                language_scores[lang] = max(language_scores[lang], score)
            else:
                language_scores[lang] = score
        
        # Ensure we have at least some languages
        if not language_scores:
            language_scores = {'eng': 50.0}  # Default to English with moderate confidence
        
        # Determine dominant language
        dominant_language = max(language_scores, key=language_scores.get)
        confidence = language_scores[dominant_language]
        
        return {
            'dominant_language': dominant_language,
            'confidence': confidence,
            'all_languages': language_scores
        }
    
    def _detect_language_patterns(self, text):
        """Detect languages based on character patterns and common words"""
        
        scores = {}
        
        # Vietnamese detection (diacritics)
        vietnamese_chars = 0
        for pattern in self.config['vietnamese_patterns']:
            vietnamese_chars += len(re.findall(pattern, text))
        
        if vietnamese_chars > 0:
            scores['vie'] = min(95.0, vietnamese_chars / len(text) * 200)
        
        # English detection (common English words)
        english_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']
        english_count = 0
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in english_words:
                english_count += 1
        
        if words:
            scores['eng'] = min(90.0, (english_count / len(words)) * 100)
        
        # French detection (common patterns)
        french_patterns = [r'\b(le|la|les|un|une|des|du|de|et|est|avec|pour|dans|sur)\b']
        french_count = 0
        for pattern in french_patterns:
            french_count += len(re.findall(pattern, text.lower()))
        
        if french_count > 0:
            scores['fra'] = min(85.0, (french_count / len(words)) * 100) if words else 0
        
        return scores
    
    def _calculate_language_confidence(self, text, lang_code):
        """Calculate confidence score for detected language"""
        
        if not text.strip():
            return 0.0
        
        # Basic confidence based on text length and character variety
        text_length = len(text.strip())
        unique_chars = len(set(text.lower()))
        
        # Base confidence
        confidence = min(80.0, text_length / 10)  # Up to 80% based on length
        
        # Bonus for character variety
        if unique_chars > 10:
            confidence += 15.0
        elif unique_chars > 5:
            confidence += 10.0
        
        # Language-specific adjustments
        if lang_code == 'vie':
            # Check for Vietnamese diacritics
            vietnamese_chars = 0
            for pattern in self.config['vietnamese_patterns']:
                vietnamese_chars += len(re.findall(pattern, text))
            if vietnamese_chars > 0:
                confidence += 20.0
        
        return min(95.0, confidence)
    
    def _detect_special_zones(self, text, image):
        """Detect special zones like MRZ and handwriting"""
        
        special_zones = {
            'mrz_detected': False,
            'mrz_lines': [],
            'handwriting_detected': False,
            'handwriting_regions': [],
            'numeric_heavy': False,
            'date_patterns': []
        }
        
        # MRZ Detection
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            for pattern in self.config['mrz_patterns']:
                if re.match(pattern, line):
                    special_zones['mrz_detected'] = True
                    special_zones['mrz_lines'].append({
                        'line_number': i,
                        'content': line,
                        'pattern': 'MRZ'
                    })
        
        # Numeric content detection
        if text:
            numeric_ratio = len(re.findall(r'\d', text)) / len(text)
            special_zones['numeric_heavy'] = numeric_ratio > self.config['numeric_threshold']
        
        # Date pattern detection
        for pattern in self.config['date_patterns']:
            dates = re.findall(pattern, text)
            special_zones['date_patterns'].extend(dates)
        
        # Handwriting detection (simplified - based on text extraction quality)
        if self.config['enable_handwriting_detection']:
            # If text extraction is poor but image has content, might be handwriting
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edge_density = np.sum(cv2.Canny(gray, 50, 150) > 0) / (gray.shape[0] * gray.shape[1])
            
            if edge_density > 0.1 and len(text) < 50:  # High edge density but low text extraction
                special_zones['handwriting_detected'] = True
                special_zones['handwriting_regions'] = [{'region': 'full_image', 'confidence': edge_density}]
        
        return special_zones
    
    def _get_processing_recommendation(self, script_analysis, language_analysis, special_zones):
        """Generate processing recommendations based on analysis"""
        
        recommendations = []
        
        # Script-based recommendations
        dominant_script = script_analysis['dominant_script']
        if dominant_script == 'vietnamese':
            recommendations.append("Use Vietnamese OCR models for optimal accuracy")
        elif dominant_script == 'mrz':
            recommendations.append("Use specialized MRZ OCR processing")
        elif dominant_script == 'numeric':
            recommendations.append("Use numeric-optimized OCR settings")
        
        # Language-based recommendations
        dominant_lang = language_analysis['dominant_language']
        if dominant_lang != 'eng':
            recommendations.append(f"Configure OCR for {dominant_lang} language")
        
        if language_analysis.get('multilingual_flag'):
            recommendations.append("Use multilingual OCR pipeline")
        
        # Special zones recommendations
        if special_zones['mrz_detected']:
            recommendations.append("Process MRZ zones with specialized MRZ OCR")
        
        if special_zones['handwriting_detected']:
            recommendations.append("Preserve handwriting zones for handwriting recognition models")
        
        if special_zones['numeric_heavy']:
            recommendations.append("Optimize for numeric content recognition")
        
        return recommendations
    
    def _generate_language_report(self, analysis_results, output_folder, filename):
        """Generate detailed language analysis report"""
        
        report_path = os.path.join(output_folder, f"{filename}{self.config['lang_suffix']}.json")
        
        report = {
            'language_detection_report': {
                'file_name': filename,
                'analysis_timestamp': datetime.now().isoformat(),
                'dominant_script': analysis_results['dominant_script'],
                'script_confidence': analysis_results['script_confidence'],
                'detected_scripts': analysis_results['detected_scripts'],
                'dominant_language': analysis_results['dominant_language'],
                'language_confidence': analysis_results['language_confidence'],
                'detected_languages': analysis_results['detected_languages'],
                'multilingual_document': analysis_results['multilingual_flag'],
                'special_zones': analysis_results['special_zones'],
                'handwriting_zones': analysis_results['handwriting_zones'],
                'processing_recommendations': analysis_results['processing_recommendation'],
                'text_sample': analysis_results['text_content_sample']
            },
            'configuration': {
                'supported_scripts': self.config['supported_scripts'],
                'supported_languages': self.config['supported_languages'],
                'confidence_threshold': self.config['confidence_threshold']
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def _create_script_overlay(self, image, analysis_results, output_folder, filename):
        """Create visualization overlay showing detected scripts and languages"""
        
        try:
            overlay_path = os.path.join(output_folder, f"{filename}_script_overlay.png")
            
            # Create overlay image
            overlay = image.copy()
            height, width = overlay.shape[:2]
            
            # Create info panel
            panel_height = 200
            panel = np.ones((panel_height, width, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Add text information
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Title
            cv2.putText(panel, "Language & Script Analysis", 
                       (10, 30), font, 0.9, (0, 0, 0), thickness)
            
            # Dominant script
            script_text = f"Dominant Script: {analysis_results['dominant_script'].upper()} ({analysis_results['script_confidence']:.1f}%)"
            cv2.putText(panel, script_text, 
                       (10, 60), font, font_scale, (0, 0, 255), thickness)
            
            # Dominant language
            lang_text = f"Dominant Language: {analysis_results['dominant_language'].upper()} ({analysis_results['language_confidence']:.1f}%)"
            cv2.putText(panel, lang_text, 
                       (10, 90), font, font_scale, (255, 0, 0), thickness)
            
            # Special zones
            special_info = []
            if analysis_results['special_zones']['mrz_detected']:
                special_info.append("MRZ Detected")
            if analysis_results['special_zones']['handwriting_detected']:
                special_info.append("Handwriting Detected")
            if analysis_results['special_zones']['numeric_heavy']:
                special_info.append("Numeric Heavy")
            
            if special_info:
                special_text = f"Special Zones: {', '.join(special_info)}"
                cv2.putText(panel, special_text, 
                           (10, 120), font, font_scale, (0, 128, 0), thickness)
            
            # Multilingual flag
            if analysis_results['multilingual_flag']:
                cv2.putText(panel, "MULTILINGUAL DOCUMENT", 
                           (10, 150), font, font_scale, (255, 0, 255), thickness)
            
            # Recommendations
            recommendations = analysis_results['processing_recommendation'][:2]  # First 2 recommendations
            for i, rec in enumerate(recommendations):
                if len(rec) > 60:
                    rec = rec[:57] + "..."
                cv2.putText(panel, f"• {rec}", 
                           (10, 180 + i * 20), font, 0.5, (0, 0, 0), 1)
            
            # Combine original image with panel
            result = np.vstack([overlay, panel])
            
            # Save overlay
            cv2.imwrite(overlay_path, result)
            
            return overlay_path
            
        except Exception as e:
            self.logger.error(f"Error creating script overlay: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    task = LanguageDetectionTask()
    print("Language & Script Detection Task initialized")
    print("Configuration:", task.config)
