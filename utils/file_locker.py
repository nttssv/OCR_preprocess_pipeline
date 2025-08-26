#!/usr/bin/env python3
"""
File Locker
Prevents race conditions and concurrent access conflicts during parallel processing
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Set
from contextlib import contextmanager

class FileLocker:
    """Manages file locks to prevent concurrent access conflicts"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._locks: Dict[str, threading.Lock] = {}
        self._lock_holders: Dict[str, int] = {}  # thread_id -> file_path
        self._global_lock = threading.Lock()
        
    def _get_lock(self, file_path: str) -> threading.Lock:
        """Get or create a lock for a specific file"""
        with self._global_lock:
            if file_path not in self._locks:
                self._locks[file_path] = threading.Lock()
            return self._locks[file_path]
    
    def _get_thread_id(self) -> int:
        """Get current thread ID"""
        return threading.get_ident()
    
    def acquire_lock(self, file_path: str, timeout: float = 30.0) -> bool:
        """
        Acquire a lock for a file
        
        Args:
            file_path: Path to the file to lock
            timeout: Maximum time to wait for lock (seconds)
            
        Returns:
            bool: True if lock acquired, False if timeout
        """
        thread_id = self._get_thread_id()
        lock = self._get_lock(file_path)
        
        # Check if current thread already holds this lock
        if self._lock_holders.get(thread_id) == file_path:
            self.logger.debug(f"Thread {thread_id} already holds lock for {file_path}")
            return True
        
        # Try to acquire the lock
        if lock.acquire(timeout=timeout):
            self._lock_holders[thread_id] = file_path
            self.logger.debug(f"Thread {thread_id} acquired lock for {file_path}")
            return True
        else:
            self.logger.warning(f"Thread {thread_id} failed to acquire lock for {file_path} (timeout: {timeout}s)")
            return False
    
    def release_lock(self, file_path: str) -> bool:
        """
        Release a lock for a file
        
        Args:
            file_path: Path to the file to unlock
            
        Returns:
            bool: True if lock released, False if error
        """
        thread_id = self._get_thread_id()
        lock = self._get_lock(file_path)
        
        # Check if current thread holds this lock
        if self._lock_holders.get(thread_id) != file_path:
            self.logger.warning(f"Thread {thread_id} tried to release lock for {file_path} but doesn't hold it")
            return False
        
        try:
            lock.release()
            del self._lock_holders[thread_id]
            self.logger.debug(f"Thread {thread_id} released lock for {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error releasing lock for {file_path}: {e}")
            return False
    
    @contextmanager
    def file_lock(self, file_path: str, timeout: float = 30.0):
        """
        Context manager for file locking
        
        Usage:
            with file_locker.file_lock("path/to/file.png"):
                # File operations here
                pass
        """
        acquired = self.acquire_lock(file_path, timeout)
        if not acquired:
            raise RuntimeError(f"Failed to acquire lock for {file_path}")
        
        try:
            yield
        finally:
            self.release_lock(file_path)
    
    def is_locked(self, file_path: str) -> bool:
        """Check if a file is currently locked"""
        lock = self._get_lock(file_path)
        return lock.locked()
    
    def get_lock_holder(self, file_path: str) -> Optional[int]:
        """Get the thread ID that currently holds the lock for a file"""
        for thread_id, locked_file in self._lock_holders.items():
            if locked_file == file_path:
                return thread_id
        return None
    
    def get_locked_files(self) -> Set[str]:
        """Get set of currently locked files"""
        return set(self._lock_holders.values())
    
    def force_release_all_locks(self) -> None:
        """Force release all locks (use with caution)"""
        with self._global_lock:
            for thread_id in list(self._lock_holders.keys()):
                del self._lock_holders[thread_id]
            self._locks.clear()
        self.logger.warning("All file locks force-released")
    
    def cleanup_unused_locks(self) -> int:
        """Remove locks for files that no longer exist"""
        cleaned = 0
        with self._global_lock:
            for file_path in list(self._locks.keys()):
                if not os.path.exists(file_path):
                    del self._locks[file_path]
                    cleaned += 1
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} unused file locks")
        return cleaned
    
    def get_lock_status(self) -> Dict[str, Dict]:
        """Get detailed status of all locks"""
        status = {}
        with self._global_lock:
            for file_path, lock in self._locks.items():
                holder_thread = self.get_lock_holder(file_path)
                status[file_path] = {
                    'locked': lock.locked(),
                    'holder_thread': holder_thread,
                    'exists': os.path.exists(file_path)
                }
        return status
