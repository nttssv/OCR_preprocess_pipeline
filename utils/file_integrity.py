#!/usr/bin/env python3
"""
File Integrity Checker
Prevents file corruption and ensures file integrity during pipeline processing
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class FileIntegrityChecker:
    """Checks and maintains file integrity during processing"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.file_hashes = {}  # Store file hashes for verification
        self.file_sizes = {}   # Store file sizes for verification
        
    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file"""
        try:
            if not os.path.exists(file_path):
                return ""
            
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error computing hash for {file_path}: {e}")
            return ""
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            if os.path.exists(file_path):
                return os.path.getsize(file_path)
            return 0
        except Exception as e:
            self.logger.error(f"Error getting file size for {file_path}: {e}")
            return 0
    
    def register_file(self, file_path: str) -> bool:
        """Register a file for integrity monitoring"""
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"File does not exist: {file_path}")
                return False
            
            file_hash = self.compute_file_hash(file_path)
            file_size = self.get_file_size(file_path)
            
            if file_hash and file_size > 0:
                self.file_hashes[file_path] = file_hash
                self.file_sizes[file_path] = file_size
                self.logger.info(f"Registered file for integrity monitoring: {file_path} ({file_size} bytes)")
                return True
            else:
                self.logger.error(f"Failed to register file: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error registering file {file_path}: {e}")
            return False
    
    def verify_file_integrity(self, file_path: str) -> Tuple[bool, str]:
        """
        Verify that a file hasn't been corrupted
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            if file_path not in self.file_hashes:
                return False, "File not registered for monitoring"
            
            # Check file size
            current_size = self.get_file_size(file_path)
            original_size = self.file_sizes[file_path]
            
            if current_size != original_size:
                return False, f"File size changed: {original_size} -> {current_size} bytes"
            
            # Check file hash
            current_hash = self.compute_file_hash(file_path)
            original_hash = self.file_hashes[file_path]
            
            if current_hash != original_hash:
                return False, f"File hash changed: {original_hash[:8]} -> {current_hash[:8]}"
            
            return True, "File integrity verified"
            
        except Exception as e:
            return False, f"Error verifying file integrity: {e}"
    
    def verify_all_files(self) -> Dict[str, Tuple[bool, str]]:
        """Verify integrity of all registered files"""
        results = {}
        for file_path in self.file_hashes.keys():
            is_valid, message = self.verify_file_integrity(file_path)
            results[file_path] = (is_valid, message)
            
            if not is_valid:
                self.logger.error(f"File integrity check failed: {file_path} - {message}")
            else:
                self.logger.info(f"File integrity check passed: {file_path}")
        
        return results
    
    def cleanup_registration(self, file_path: str) -> None:
        """Remove a file from integrity monitoring"""
        if file_path in self.file_hashes:
            del self.file_hashes[file_path]
        if file_path in self.file_sizes:
            del self.file_sizes[file_path]
        self.logger.info(f"Removed file from integrity monitoring: {file_path}")
    
    def get_corrupted_files(self) -> List[str]:
        """Get list of files that have been corrupted"""
        corrupted = []
        for file_path in self.file_hashes.keys():
            is_valid, _ = self.verify_file_integrity(file_path)
            if not is_valid:
                corrupted.append(file_path)
        return corrupted
    
    def restore_file_from_backup(self, file_path: str, backup_path: str) -> bool:
        """Restore a corrupted file from backup"""
        try:
            if not os.path.exists(backup_path):
                self.logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            import shutil
            shutil.copy2(backup_path, file_path)
            
            # Re-register the restored file
            if self.register_file(file_path):
                self.logger.info(f"Successfully restored file from backup: {file_path}")
                return True
            else:
                self.logger.error(f"Failed to re-register restored file: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error restoring file {file_path} from backup: {e}")
            return False
