#!/usr/bin/env python3
"""
Performance Optimization Service
Handles caching, streaming, and performance monitoring
"""

import os
import time
import asyncio
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
import psutil

from ..core.config import settings
from ..core.logger import get_logger
from ..core.database import CacheOps

logger = get_logger(__name__)


class PerformanceOptimizer:
    """Service for performance optimization and monitoring"""
    
    def __init__(self):
        self.cache_dir = settings.CACHE_DIR
        self.max_cache_size = settings.MAX_CACHE_SIZE
        self.cache_expiry_hours = settings.CACHE_EXPIRY_HOURS
        
        # Performance monitoring
        self.processing_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0,
            "fallback_usage": 0
        }
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"🚀 Performance optimizer initialized")
        logger.info(f"💾 Cache directory: {self.cache_dir}")
        logger.info(f"📏 Max cache size: {self.max_cache_size / (1024*1024*1024):.1f}GB")
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check available system resources"""
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(self.cache_dir)
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available,
                "disk_usage": disk.percent,
                "disk_free": disk.free,
                "recommendation": self._get_performance_recommendation(
                    cpu_percent, memory.percent, disk.percent
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking system resources: {str(e)}")
            return {"error": str(e)}
    
    def _get_performance_recommendation(
        self, 
        cpu_percent: float, 
        memory_percent: float, 
        disk_percent: float
    ) -> str:
        """Get performance recommendation based on system resources"""
        
        if cpu_percent > 80:
            return "high_cpu_usage"
        elif memory_percent > 85:
            return "high_memory_usage"
        elif disk_percent > 90:
            return "low_disk_space"
        elif cpu_percent > 60 or memory_percent > 70:
            return "moderate_load"
        else:
            return "optimal"
    
    async def optimize_for_file_size(self, file_size: int) -> Dict[str, Any]:
        """Get optimization strategy based on file size"""
        
        # Check system resources first
        resources = await self.check_system_resources()
        
        # Base optimization on file size
        if file_size < 5 * 1024 * 1024:  # < 5MB
            strategy = {
                "transformation_type": "enhanced",
                "enable_caching": True,
                "batch_processing": False,
                "memory_limit": 512 * 1024 * 1024,  # 512MB
                "timeout": 60
            }
        elif file_size < 25 * 1024 * 1024:  # < 25MB
            strategy = {
                "transformation_type": "deskewing",
                "enable_caching": True,
                "batch_processing": False,
                "memory_limit": 1024 * 1024 * 1024,  # 1GB
                "timeout": 120
            }
        elif file_size < 50 * 1024 * 1024:  # < 50MB
            strategy = {
                "transformation_type": "deskewing",
                "enable_caching": True,
                "batch_processing": True,
                "memory_limit": 2048 * 1024 * 1024,  # 2GB
                "timeout": 180
            }
        else:  # >= 50MB
            strategy = {
                "transformation_type": "basic",
                "enable_caching": False,  # Don't cache very large files
                "batch_processing": True,
                "memory_limit": 4096 * 1024 * 1024,  # 4GB
                "timeout": 300
            }
        
        # Adjust based on system resources
        if resources.get("recommendation") == "high_memory_usage":
            strategy["transformation_type"] = "basic"
            strategy["memory_limit"] = min(strategy["memory_limit"], 1024 * 1024 * 1024)
        elif resources.get("recommendation") == "high_cpu_usage":
            strategy["transformation_type"] = "basic"
            strategy["timeout"] = min(strategy["timeout"], 180)
        
        return strategy
    
    async def enable_streaming_processing(
        self, 
        file_path: str, 
        chunk_size: int = 1024 * 1024  # 1MB chunks
    ) -> bool:
        """
        Enable streaming processing for large files
        
        Args:
            file_path: Path to the file
            chunk_size: Size of chunks for streaming
            
        Returns:
            True if streaming is recommended
        """
        
        try:
            file_size = os.path.getsize(file_path)
            
            # Enable streaming for files > 25MB
            if file_size > 25 * 1024 * 1024:
                logger.info(f"📡 Enabling streaming processing for large file: {file_size / (1024*1024):.1f}MB")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking file for streaming: {str(e)}")
            return False
    
    async def manage_cache_size(self) -> Dict[str, Any]:
        """Manage cache size and cleanup old entries"""
        
        try:
            # Calculate current cache size
            cache_size = 0
            cache_files = []
            
            if os.path.exists(self.cache_dir):
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_stat = os.stat(file_path)
                        cache_size += file_stat.st_size
                        cache_files.append({
                            "path": file_path,
                            "size": file_stat.st_size,
                            "mtime": file_stat.st_mtime
                        })
            
            cleanup_stats = {
                "initial_size": cache_size,
                "files_removed": 0,
                "space_freed": 0
            }
            
            # If cache exceeds limit, remove oldest files
            if cache_size > self.max_cache_size:
                logger.info(f"🧹 Cache size ({cache_size / (1024*1024):.1f}MB) exceeds limit, cleaning up...")
                
                # Sort by modification time (oldest first)
                cache_files.sort(key=lambda x: x["mtime"])
                
                space_to_free = cache_size - (self.max_cache_size * 0.8)  # Free to 80% of limit
                
                for file_info in cache_files:
                    if cleanup_stats["space_freed"] >= space_to_free:
                        break
                    
                    try:
                        os.remove(file_info["path"])
                        cleanup_stats["files_removed"] += 1
                        cleanup_stats["space_freed"] += file_info["size"]
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove cache file {file_info['path']}: {e}")
            
            final_size = cache_size - cleanup_stats["space_freed"]
            
            logger.info(
                f"💾 Cache management complete: "
                f"{final_size / (1024*1024):.1f}MB used, "
                f"{cleanup_stats['files_removed']} files removed"
            )
            
            return {
                "cache_size_mb": final_size / (1024*1024),
                "cache_limit_mb": self.max_cache_size / (1024*1024),
                "utilization_percent": (final_size / self.max_cache_size) * 100,
                "cleanup_performed": cleanup_stats["files_removed"] > 0,
                **cleanup_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error managing cache: {str(e)}")
            return {"error": str(e)}
    
    async def create_processing_batch(
        self, 
        file_paths: List[str], 
        max_batch_size: int = 5
    ) -> List[List[str]]:
        """
        Create processing batches for multiple files
        
        Args:
            file_paths: List of file paths to process
            max_batch_size: Maximum files per batch
            
        Returns:
            List of batches (each batch is a list of file paths)
        """
        
        batches = []
        current_batch = []
        current_batch_size = 0
        
        for file_path in file_paths:
            try:
                file_size = os.path.getsize(file_path)
                
                # If adding this file would exceed batch limits, start new batch
                if (len(current_batch) >= max_batch_size or 
                    current_batch_size + file_size > 100 * 1024 * 1024):  # 100MB per batch
                    
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_batch_size = 0
                
                current_batch.append(file_path)
                current_batch_size += file_size
                
            except Exception as e:
                logger.warning(f"⚠️ Could not get size for {file_path}: {e}")
                current_batch.append(file_path)
        
        # Add remaining files as final batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"📦 Created {len(batches)} processing batches from {len(file_paths)} files")
        
        return batches
    
    async def monitor_processing_performance(
        self, 
        document_id: str, 
        start_time: float, 
        file_size: int,
        transformation_type: str,
        cache_hit: bool = False
    ):
        """Monitor and record processing performance"""
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.processing_stats["total_processed"] += 1
        
        if cache_hit:
            self.processing_stats["cache_hits"] += 1
        else:
            self.processing_stats["cache_misses"] += 1
        
        # Update average processing time
        total_proc = self.processing_stats["total_processed"]
        current_avg = self.processing_stats["avg_processing_time"]
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (total_proc - 1) + processing_time) / total_proc
        )
        
        # Log performance metrics
        throughput = file_size / processing_time if processing_time > 0 else 0
        
        logger.info(
            f"📊 Performance: {document_id} - "
            f"{processing_time:.2f}s, "
            f"{file_size / (1024*1024):.1f}MB, "
            f"{throughput / (1024*1024):.1f}MB/s, "
            f"{'CACHE' if cache_hit else transformation_type}"
        )
        
        # Check if performance is below threshold
        if processing_time > 60 and file_size < 10 * 1024 * 1024:  # >60s for <10MB file
            logger.warning(
                f"⚠️ Slow processing detected: {processing_time:.2f}s for {file_size / (1024*1024):.1f}MB file"
            )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        cache_hit_rate = 0
        if self.processing_stats["total_processed"] > 0:
            cache_hit_rate = (
                self.processing_stats["cache_hits"] / 
                self.processing_stats["total_processed"] * 100
            )
        
        # Get system resources
        resources = await self.check_system_resources()
        
        # Get cache info
        cache_info = await self.manage_cache_size()
        
        return {
            "processing_stats": {
                **self.processing_stats,
                "cache_hit_rate_percent": cache_hit_rate
            },
            "system_resources": resources,
            "cache_info": cache_info,
            "recommendations": await self._get_performance_recommendations()
        }
    
    async def _get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        
        recommendations = []
        
        # Check cache hit rate
        total_proc = self.processing_stats["total_processed"]
        if total_proc > 10:
            cache_hit_rate = (self.processing_stats["cache_hits"] / total_proc) * 100
            if cache_hit_rate < 20:
                recommendations.append("Consider increasing cache expiry time")
        
        # Check average processing time
        avg_time = self.processing_stats["avg_processing_time"]
        if avg_time > 30:
            recommendations.append("Consider using faster transformation types for large files")
        
        # Check system resources
        resources = await self.check_system_resources()
        rec = resources.get("recommendation", "optimal")
        
        if rec == "high_cpu_usage":
            recommendations.append("High CPU usage detected - consider reducing concurrent processing")
        elif rec == "high_memory_usage":
            recommendations.append("High memory usage detected - consider basic transformations")
        elif rec == "low_disk_space":
            recommendations.append("Low disk space - consider reducing cache size or cleanup")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations
    
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        
        try:
            cleanup_count = 0
            current_time = time.time()
            expiry_seconds = self.cache_expiry_hours * 3600
            
            if os.path.exists(self.cache_dir):
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        try:
                            file_mtime = os.path.getmtime(file_path)
                            
                            if current_time - file_mtime > expiry_seconds:
                                os.remove(file_path)
                                cleanup_count += 1
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Could not clean up {file_path}: {e}")
            
            if cleanup_count > 0:
                logger.info(f"🧹 Cleaned up {cleanup_count} expired cache files")
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up expired cache: {str(e)}")
            return 0


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()