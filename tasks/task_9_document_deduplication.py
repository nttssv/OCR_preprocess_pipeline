#!/usr/bin/env python3
"""
Task 9: Document Deduplication
Advanced perceptual hashing for detecting repeated attachments and avoiding redundant OCR.
"""

import cv2
import numpy as np
import os
import logging
import hashlib
import sqlite3
import json
from PIL import Image
import shutil
from typing import List, Tuple, Dict, Optional
from datetime import datetime

class DocumentDeduplicationTask:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Document Deduplication"
        self.task_id = "task_9_document_deduplication"
        self.config = {
            # Perceptual Hashing Settings
            "phash_size": 16,                    # Hash dimension (16x16 = 256-bit hash)
            "phash_threshold": 10,               # Hamming distance threshold for duplicates
            "enable_phash": True,                # Enable perceptual hashing
            "enable_content_hash": True,         # Enable content-based hashing
            "enable_size_check": True,           # Enable size-based pre-filtering
            
            # Duplicate Detection Settings
            "similarity_threshold": 85,          # Percentage similarity for duplicates
            "size_tolerance": 0.1,               # 10% size difference tolerance
            "enable_rotation_invariant": True,   # Detect rotated duplicates
            "enable_scale_invariant": True,      # Detect scaled duplicates
            
            # Processing Options
            "skip_duplicates": True,             # Skip processing duplicates
            "index_duplicates": True,            # Index duplicates in database
            "generate_fingerprints": True,       # Generate audit fingerprints
            "create_comparison": True,           # Create comparison images
            
            # Database Settings
            "db_path": "deduplication/dedup_index.db",
            "enable_audit_trail": True,         # Maintain audit trail
            "max_similar_files": 10,            # Max similar files to track per hash
            
            # Output Settings
            "save_duplicate_report": True,      # Save duplicate detection report
            "highlight_duplicates": True,       # Highlight duplicates in comparisons
            "dedup_suffix": "_dedup_report",    # Suffix for deduplication reports
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the deduplication database"""
        try:
            db_dir = os.path.dirname(self.config['db_path'])
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.config['db_path'])
            cursor = conn.cursor()
            
            # Create deduplication table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_hashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    perceptual_hash TEXT NOT NULL,
                    image_dimensions TEXT NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'new',
                    duplicate_group_id TEXT,
                    similarity_score REAL,
                    audit_fingerprint TEXT,
                    UNIQUE(file_path)
                )
            ''')
            
            # Create duplicate groups table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS duplicate_groups (
                    group_id TEXT PRIMARY KEY,
                    master_file TEXT NOT NULL,
                    total_duplicates INTEGER DEFAULT 0,
                    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON document_hashes(content_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON document_hashes(perceptual_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_size ON document_hashes(file_size)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_duplicate_group ON document_hashes(duplicate_group_id)')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"📊 Deduplication database initialized: {self.config['db_path']}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize deduplication database: {str(e)}")
            raise
    
    def run(self, input_file, file_type, output_folder):
        """
        Main entry point for document deduplication
        
        Returns:
            dict: Task result with status and output path
        """
        try:
            self.logger.info(f"🔄 Running {self.task_name} on {os.path.basename(input_file)}")
            
            # Load and analyze image
            image = cv2.imread(input_file)
            if image is None:
                raise ValueError(f"Could not load image: {input_file}")
            
            # Process deduplication
            result_path = self._process_document_deduplication(input_file, output_folder)
            
            self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
            
            return {
                'status': 'completed',
                'output': result_path,
                'task_name': self.task_name,
                'processing_time': None,  # Could add timing if needed
                'metadata': {
                    'input_file': input_file,
                    'output_file': result_path,
                    'file_type': file_type
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ {self.task_name} failed for {input_file}: {str(e)}")
            return {
                'status': 'failed',
                'output': input_file,  # Return original file path on failure
                'task_name': self.task_name,
                'error': str(e),
                'metadata': {
                    'input_file': input_file,
                    'file_type': file_type
                }
            }
    
    def _process_document_deduplication(self, image_path, output_folder):
        """Process document deduplication with perceptual hashing"""
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        self.logger.info(f"   🔍 Step 1: Analyzing document for deduplication")
        
        # Load image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        file_size = os.path.getsize(image_path)
        
        self.logger.info(f"   📊 Document: {filename} ({width}x{height}, {file_size/1024:.1f}KB)")
        
        # Generate hashes
        content_hash = self._generate_content_hash(image_path)
        perceptual_hash = self._generate_perceptual_hash(image)
        audit_fingerprint = self._generate_audit_fingerprint(image, image_path)
        
        self.logger.info(f"   🔐 Generated hashes: content={content_hash[:12]}..., phash={perceptual_hash[:16]}...")
        
        # Check for duplicates
        duplicate_info = self._check_duplicates(
            image_path, file_size, content_hash, perceptual_hash, 
            f"{width}x{height}", audit_fingerprint
        )
        
        if duplicate_info['is_duplicate']:
            self.logger.info(f"   🔄 Step 2: Duplicate detected - processing duplicate handling")
            return self._handle_duplicate(image_path, duplicate_info, output_folder)
        else:
            self.logger.info(f"   ✨ Step 2: Unique document - proceeding with normal processing")
            return self._handle_unique_document(image_path, content_hash, perceptual_hash, 
                                              f"{width}x{height}", audit_fingerprint, output_folder)
    
    def _generate_content_hash(self, image_path):
        """Generate content-based hash (SHA256) of the file"""
        hasher = hashlib.sha256()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _generate_perceptual_hash(self, image):
        """
        Generate perceptual hash (pHash) for image similarity detection
        
        Args:
            image: OpenCV image array
            
        Returns:
            str: Hexadecimal perceptual hash
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize to hash size for normalization
            hash_size = self.config['phash_size']
            resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply DCT (Discrete Cosine Transform)
            dct = cv2.dct(np.float32(resized))
            
            # Keep only the top-left 8x8 portion (low frequencies)
            dct_reduced = dct[:8, :8]
            
            # Calculate median
            median = np.median(dct_reduced)
            
            # Generate hash: 1 if above median, 0 if below
            hash_bits = dct_reduced > median
            
            # Convert to hex string
            hash_string = ''
            for i in range(0, 64, 4):  # Process 4 bits at a time
                bit_chunk = hash_bits.flatten()[i:i+4]
                if len(bit_chunk) == 4:
                    hex_val = (bit_chunk[0] * 8 + bit_chunk[1] * 4 + 
                              bit_chunk[2] * 2 + bit_chunk[3] * 1)
                    hash_string += format(hex_val, 'x')
            
            return hash_string
            
        except Exception as e:
            self.logger.error(f"Error generating perceptual hash: {str(e)}")
            # Fallback to simple hash
            return hashlib.md5(image.tobytes()).hexdigest()[:16]
    
    def _generate_audit_fingerprint(self, image, image_path):
        """Generate comprehensive audit fingerprint"""
        try:
            # Image characteristics
            height, width = image.shape[:2]
            file_size = os.path.getsize(image_path)
            
            # Statistical features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # Create fingerprint
            fingerprint = {
                'dimensions': f"{width}x{height}",
                'file_size': file_size,
                'mean_brightness': round(mean_brightness, 2),
                'std_brightness': round(std_brightness, 2),
                'edge_density': round(edge_density, 4),
                'timestamp': datetime.now().isoformat(),
                'file_name': os.path.basename(image_path)
            }
            
            return json.dumps(fingerprint, sort_keys=True)
            
        except Exception as e:
            self.logger.error(f"Error generating audit fingerprint: {str(e)}")
            return json.dumps({'error': str(e), 'timestamp': datetime.now().isoformat()})
    
    def _calculate_hamming_distance(self, hash1, hash2):
        """Calculate Hamming distance between two hex hashes"""
        if len(hash1) != len(hash2):
            return float('inf')
        
        distance = 0
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                # Convert hex chars to binary and count different bits
                val1 = int(hash1[i], 16)
                val2 = int(hash2[i], 16)
                xor = val1 ^ val2
                distance += bin(xor).count('1')
        
        return distance
    
    def _check_duplicates(self, file_path, file_size, content_hash, perceptual_hash, dimensions, audit_fingerprint):
        """Check for duplicates in the database"""
        try:
            conn = sqlite3.connect(self.config['db_path'])
            cursor = conn.cursor()
            
            # Check for exact content match first
            cursor.execute('''
                SELECT file_path, file_name, duplicate_group_id, similarity_score
                FROM document_hashes 
                WHERE content_hash = ?
            ''', (content_hash,))
            
            exact_match = cursor.fetchone()
            if exact_match:
                conn.close()
                return {
                    'is_duplicate': True,
                    'type': 'exact_content',
                    'master_file': exact_match[0],
                    'similarity_score': 100.0,
                    'duplicate_group_id': exact_match[2]
                }
            
            # Check for perceptual similarity
            if self.config['enable_phash']:
                cursor.execute('''
                    SELECT file_path, file_name, perceptual_hash, duplicate_group_id
                    FROM document_hashes
                ''')
                
                all_hashes = cursor.fetchall()
                best_match = None
                best_distance = float('inf')
                
                for row in all_hashes:
                    existing_path, existing_name, existing_phash, group_id = row
                    distance = self._calculate_hamming_distance(perceptual_hash, existing_phash)
                    
                    if distance < best_distance and distance <= self.config['phash_threshold']:
                        best_distance = distance
                        similarity_score = max(0, 100 - (distance * 5))  # Convert to percentage
                        
                        if similarity_score >= self.config['similarity_threshold']:
                            best_match = {
                                'file_path': existing_path,
                                'similarity_score': similarity_score,
                                'hamming_distance': distance,
                                'duplicate_group_id': group_id
                            }
                
                conn.close()
                
                if best_match:
                    return {
                        'is_duplicate': True,
                        'type': 'perceptual_similarity',
                        'master_file': best_match['file_path'],
                        'similarity_score': best_match['similarity_score'],
                        'hamming_distance': best_match['hamming_distance'],
                        'duplicate_group_id': best_match['duplicate_group_id']
                    }
            
            conn.close()
            return {'is_duplicate': False}
            
        except Exception as e:
            self.logger.error(f"Error checking duplicates: {str(e)}")
            return {'is_duplicate': False, 'error': str(e)}
    
    def _handle_duplicate(self, image_path, duplicate_info, output_folder):
        """Handle detected duplicate document"""
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        self.logger.info(f"   🔍 Duplicate type: {duplicate_info['type']}")
        self.logger.info(f"   📊 Similarity: {duplicate_info['similarity_score']:.1f}%")
        self.logger.info(f"   📁 Master file: {os.path.basename(duplicate_info['master_file'])}")
        
        # Update database with duplicate info
        self._update_duplicate_in_database(image_path, duplicate_info)
        
        # Generate duplicate report
        if self.config['save_duplicate_report']:
            report_path = self._generate_duplicate_report(image_path, duplicate_info, output_folder)
            self.logger.info(f"   📊 Duplicate report saved: {os.path.basename(report_path)}")
        
        # Create comparison if enabled
        if self.config['create_comparison']:
            comparison_path = self._create_duplicate_comparison(image_path, duplicate_info, output_folder)
            if comparison_path:
                self.logger.info(f"   🖼️  Comparison saved: {os.path.basename(comparison_path)}")
        
        if self.config['skip_duplicates']:
            self.logger.info(f"   ⏭️  Skipping duplicate processing (configured)")
            # Return the master file path to avoid reprocessing
            return duplicate_info['master_file']
        else:
            self.logger.info(f"   🔄 Processing duplicate anyway (configured)")
            return image_path
    
    def _handle_unique_document(self, image_path, content_hash, perceptual_hash, dimensions, audit_fingerprint, output_folder):
        """Handle unique (non-duplicate) document"""
        
        # Store in database
        self._store_document_in_database(image_path, content_hash, perceptual_hash, dimensions, audit_fingerprint)
        
        self.logger.info(f"   💾 Document indexed for future deduplication")
        
        # Continue with normal processing
        return image_path
    
    def _store_document_in_database(self, file_path, content_hash, perceptual_hash, dimensions, audit_fingerprint):
        """Store document information in database"""
        try:
            conn = sqlite3.connect(self.config['db_path'])
            cursor = conn.cursor()
            
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            cursor.execute('''
                INSERT OR REPLACE INTO document_hashes 
                (file_path, file_name, file_size, content_hash, perceptual_hash, 
                 image_dimensions, processing_status, audit_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (file_path, file_name, file_size, content_hash, perceptual_hash, 
                  dimensions, 'processed', audit_fingerprint))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing document in database: {str(e)}")
    
    def _update_duplicate_in_database(self, file_path, duplicate_info):
        """Update database with duplicate information"""
        try:
            conn = sqlite3.connect(self.config['db_path'])
            cursor = conn.cursor()
            
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # Generate or get duplicate group ID
            group_id = duplicate_info.get('duplicate_group_id')
            if not group_id:
                group_id = hashlib.md5(duplicate_info['master_file'].encode()).hexdigest()[:16]
            
            cursor.execute('''
                INSERT OR REPLACE INTO document_hashes 
                (file_path, file_name, file_size, content_hash, perceptual_hash, 
                 image_dimensions, processing_status, duplicate_group_id, similarity_score, audit_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (file_path, file_name, file_size, 'duplicate', 'duplicate', 
                  'duplicate', 'duplicate', group_id, duplicate_info['similarity_score'], 'duplicate'))
            
            # Update duplicate group
            cursor.execute('''
                INSERT OR REPLACE INTO duplicate_groups 
                (group_id, master_file, total_duplicates, updated)
                VALUES (?, ?, 
                    (SELECT COUNT(*) FROM document_hashes WHERE duplicate_group_id = ?),
                    CURRENT_TIMESTAMP)
            ''', (group_id, duplicate_info['master_file'], group_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating duplicate in database: {str(e)}")
    
    def _generate_duplicate_report(self, image_path, duplicate_info, output_folder):
        """Generate detailed duplicate detection report"""
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        report_path = os.path.join(output_folder, f"{filename}{self.config['dedup_suffix']}.json")
        
        report = {
            'duplicate_detection': {
                'file_path': image_path,
                'file_name': os.path.basename(image_path),
                'is_duplicate': True,
                'duplicate_type': duplicate_info['type'],
                'master_file': duplicate_info['master_file'],
                'similarity_score': duplicate_info['similarity_score'],
                'detection_timestamp': datetime.now().isoformat(),
                'hamming_distance': duplicate_info.get('hamming_distance'),
                'duplicate_group_id': duplicate_info.get('duplicate_group_id'),
                'processing_action': 'skipped' if self.config['skip_duplicates'] else 'processed'
            },
            'configuration': {
                'phash_threshold': self.config['phash_threshold'],
                'similarity_threshold': self.config['similarity_threshold'],
                'skip_duplicates': self.config['skip_duplicates']
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    def _create_duplicate_comparison(self, image_path, duplicate_info, output_folder):
        """Create visual comparison between duplicate and master file"""
        
        try:
            filename = os.path.splitext(os.path.basename(image_path))[0]
            comparison_path = os.path.join(output_folder, f"{filename}_duplicate_comparison.png")
            
            # Load images
            duplicate_img = cv2.imread(image_path)
            master_img = cv2.imread(duplicate_info['master_file'])
            
            if duplicate_img is None or master_img is None:
                return None
            
            # Resize images to same height for comparison
            target_height = 600
            
            # Resize duplicate image
            h1, w1 = duplicate_img.shape[:2]
            scale1 = target_height / h1
            new_w1 = int(w1 * scale1)
            duplicate_resized = cv2.resize(duplicate_img, (new_w1, target_height))
            
            # Resize master image
            h2, w2 = master_img.shape[:2]
            scale2 = target_height / h2
            new_w2 = int(w2 * scale2)
            master_resized = cv2.resize(master_img, (new_w2, target_height))
            
            # Create comparison image
            gap = 20
            total_width = new_w1 + new_w2 + gap
            comparison = np.ones((target_height + 100, total_width, 3), dtype=np.uint8) * 255
            
            # Place images
            comparison[50:50+target_height, :new_w1] = duplicate_resized
            comparison[50:50+target_height, new_w1+gap:] = master_resized
            
            # Add labels
            cv2.putText(comparison, f"DUPLICATE: {os.path.basename(image_path)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(comparison, f"MASTER: {os.path.basename(duplicate_info['master_file'])}", 
                       (new_w1 + gap + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)
            
            # Add similarity info
            similarity_text = f"Similarity: {duplicate_info['similarity_score']:.1f}%"
            cv2.putText(comparison, similarity_text, 
                       (total_width // 2 - 100, target_height + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Add type info
            type_text = f"Type: {duplicate_info['type']}"
            cv2.putText(comparison, type_text, 
                       (total_width // 2 - 80, target_height + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Save comparison
            cv2.imwrite(comparison_path, comparison)
            
            return comparison_path
            
        except Exception as e:
            self.logger.error(f"Error creating duplicate comparison: {str(e)}")
            return None
    
    def get_duplicate_statistics(self):
        """Get statistics about duplicates in the database"""
        try:
            conn = sqlite3.connect(self.config['db_path'])
            cursor = conn.cursor()
            
            # Total documents
            cursor.execute('SELECT COUNT(*) FROM document_hashes')
            total_docs = cursor.fetchone()[0]
            
            # Total duplicates
            cursor.execute('SELECT COUNT(*) FROM document_hashes WHERE processing_status = "duplicate"')
            total_duplicates = cursor.fetchone()[0]
            
            # Duplicate groups
            cursor.execute('SELECT COUNT(*) FROM duplicate_groups')
            duplicate_groups = cursor.fetchone()[0]
            
            # Top duplicate groups
            cursor.execute('''
                SELECT master_file, total_duplicates 
                FROM duplicate_groups 
                ORDER BY total_duplicates DESC 
                LIMIT 5
            ''')
            top_groups = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_documents': total_docs,
                'total_duplicates': total_duplicates,
                'duplicate_groups': duplicate_groups,
                'duplicate_percentage': (total_duplicates / total_docs * 100) if total_docs > 0 else 0,
                'top_duplicate_groups': top_groups
            }
            
        except Exception as e:
            self.logger.error(f"Error getting duplicate statistics: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    task = DocumentDeduplicationTask()
    print("Document Deduplication Task initialized")
    print("Configuration:", task.config)
