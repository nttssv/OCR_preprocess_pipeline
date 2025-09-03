#!/usr/bin/env python3
"""
File Handler Service
Handles file uploads, downloads, validation, and storage
"""

import os
import hashlib
import tempfile
import mimetypes
from typing import Tuple, Dict, Any, Optional
import aiofiles
import httpx
from fastapi import UploadFile, HTTPException

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Service for handling file operations"""
    
    def __init__(self):
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_extensions = settings.ALLOWED_FILE_EXTENSIONS
        self.upload_dir = settings.UPLOAD_DIR
        
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)
        
        logger.info(f"📁 File handler initialized - Upload dir: {self.upload_dir}")
        logger.info(f"📏 Max file size: {self.max_file_size / (1024*1024):.1f}MB")
        logger.info(f"📋 Allowed extensions: {', '.join(self.allowed_extensions)}")
    
    async def validate_file(self, file: UploadFile) -> None:
        """
        Validate uploaded file
        
        Args:
            file: FastAPI UploadFile object
            
        Raises:
            HTTPException: If validation fails
        """
        
        # Check if file is provided
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Supported types: {', '.join(self.allowed_extensions)}"
            )
        
        # Check file size by reading in chunks
        file_size = 0
        chunk_size = 8192  # 8KB chunks
        
        # Reset file position
        await file.seek(0)
        
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            
            if file_size > self.max_file_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {self.max_file_size / (1024*1024):.1f}MB"
                )
        
        # Reset file position for actual use
        await file.seek(0)
        
        logger.info(f"✅ File validation passed: {file.filename} ({file_size / 1024:.1f}KB)")
    
    async def save_uploaded_file(self, file: UploadFile) -> Tuple[str, Dict[str, Any]]:
        """
        Save uploaded file to disk
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Tuple of (file_path, file_info)
        """
        
        try:
            # Generate unique filename
            file_ext = os.path.splitext(file.filename)[1].lower()
            temp_filename = f"upload_{hash(file.filename + str(os.urandom(8)))}{file_ext}"
            file_path = os.path.join(self.upload_dir, temp_filename)
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Get file info
            file_size = len(content)
            file_info = {
                "original_filename": file.filename,
                "saved_filename": temp_filename,
                "size": file_size,
                "type": file_ext[1:] if file_ext else "unknown",
                "mime_type": file.content_type or mimetypes.guess_type(file.filename)[0]
            }
            
            logger.info(f"💾 File saved: {file.filename} -> {temp_filename} ({file_size / 1024:.1f}KB)")
            
            return file_path, file_info
            
        except Exception as e:
            logger.error(f"❌ Error saving file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    async def download_from_url(self, url: str, filename: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Download file from URL
        
        Args:
            url: URL to download from
            filename: Optional custom filename
            
        Returns:
            Tuple of (file_path, file_info)
        """
        
        try:
            logger.info(f"🌐 Downloading file from URL: {url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Send HEAD request first to check file size and type
                head_response = await client.head(url, follow_redirects=True)
                
                if head_response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot access URL: HTTP {head_response.status_code}"
                    )
                
                # Check content length
                content_length = head_response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {self.max_file_size / (1024*1024):.1f}MB"
                    )
                
                # Determine filename
                if not filename:
                    # Try to get filename from Content-Disposition header
                    content_disposition = head_response.headers.get('content-disposition', '')
                    if 'filename=' in content_disposition:
                        filename = content_disposition.split('filename=')[1].strip('"')
                    else:
                        # Use last part of URL
                        filename = os.path.basename(url.split('?')[0])
                
                if not filename:
                    filename = "downloaded_file"
                
                # Validate file extension
                file_ext = os.path.splitext(filename)[1].lower()
                if not file_ext:
                    # Try to guess from content type
                    content_type = head_response.headers.get('content-type', '')
                    if 'pdf' in content_type:
                        file_ext = '.pdf'
                    elif 'image' in content_type:
                        if 'png' in content_type:
                            file_ext = '.png'
                        elif 'jpeg' in content_type or 'jpg' in content_type:
                            file_ext = '.jpg'
                        else:
                            file_ext = '.png'  # default
                    else:
                        file_ext = '.pdf'  # default to PDF
                    
                    filename += file_ext
                
                if file_ext not in self.allowed_extensions:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File type {file_ext} not allowed. Supported types: {', '.join(self.allowed_extensions)}"
                    )
                
                # Download the file
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                content = response.content
                
                # Check size again
                if len(content) > self.max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {self.max_file_size / (1024*1024):.1f}MB"
                    )
                
                # Save file
                temp_filename = f"download_{hash(url + str(os.urandom(8)))}{file_ext}"
                file_path = os.path.join(self.upload_dir, temp_filename)
                
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)
                
                # Get file info
                file_info = {
                    "original_filename": filename,
                    "saved_filename": temp_filename,
                    "size": len(content),
                    "type": file_ext[1:] if file_ext else "unknown",
                    "mime_type": response.headers.get('content-type'),
                    "source_url": url
                }
                
                logger.info(f"✅ File downloaded: {url} -> {temp_filename} ({len(content) / 1024:.1f}KB)")
                
                return file_path, file_info
        
        except httpx.HTTPError as e:
            logger.error(f"❌ HTTP error downloading from {url}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error downloading from {url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")
    
    async def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file for caching and deduplication
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash string
        """
        
        try:
            hasher = hashlib.sha256()
            
            async with aiofiles.open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                chunk_size = 8192
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            logger.debug(f"🔍 File hash calculated: {os.path.basename(file_path)} -> {file_hash[:16]}...")
            
            return file_hash
            
        except Exception as e:
            logger.error(f"❌ Error calculating file hash for {file_path}: {str(e)}")
            return ""
    
    async def cleanup_file(self, file_path: str) -> bool:
        """
        Clean up (delete) a file
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ File cleaned up: {os.path.basename(file_path)}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up file {file_path}: {str(e)}")
            return False
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        
        try:
            if not os.path.exists(file_path):
                return {}
            
            stat = os.stat(file_path)
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            
            return {
                "filename": filename,
                "size": stat.st_size,
                "type": file_ext[1:] if file_ext else "unknown",
                "mime_type": mimetypes.guess_type(file_path)[0],
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting file info for {file_path}: {str(e)}")
            return {}
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old uploaded files
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of files cleaned up
        """
        
        try:
            import time
            
            cleanup_count = 0
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for filename in os.listdir(self.upload_dir):
                file_path = os.path.join(self.upload_dir, filename)
                
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    
                    if file_age > max_age_seconds:
                        if await self.cleanup_file(file_path):
                            cleanup_count += 1
            
            if cleanup_count > 0:
                logger.info(f"🧹 Cleaned up {cleanup_count} old files")
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"❌ Error during file cleanup: {str(e)}")
            return 0