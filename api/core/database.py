#!/usr/bin/env python3
"""
Database Models and Operations
SQLAlchemy models for document processing tracking
"""

import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List

from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from .config import settings

# Create SQLAlchemy base
Base = declarative_base()

# Database engine
engine = None
SessionLocal = None


class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransformationType(str, Enum):
    """Available transformation types"""
    DESKEWING = "deskewing"
    BASIC = "basic"
    ENHANCED = "enhanced"
    COMPREHENSIVE = "comprehensive"


class Document(Base):
    """Document processing record"""
    
    __tablename__ = "documents"
    
    # Primary identifier
    id = Column(String, primary_key=True, index=True)
    
    # File information
    filename = Column(String, nullable=False)
    original_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)
    content_hash = Column(String, nullable=True, index=True)
    
    # Processing information
    status = Column(String, default=ProcessingStatus.PENDING, index=True)
    transformation_type = Column(String, default=TransformationType.DESKEWING)
    tasks_completed = Column(JSON, default=list)
    
    # Results
    output_path = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    processing_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Processing details
    worker_id = Column(String, nullable=True)
    processing_node = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate processing duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_expired(self) -> bool:
        """Check if document processing has expired (>24 hours old)"""
        if self.created_at:
            return datetime.utcnow() - self.created_at > timedelta(hours=24)
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "document_id": self.id,
            "filename": self.filename,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "status": self.status,
            "transformation_type": self.transformation_type,
            "tasks_completed": self.tasks_completed or [],
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "processing_metadata": self.processing_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_available": bool(self.output_path and os.path.exists(self.output_path))
        }


class ProcessingCache(Base):
    """Cache for processed documents to avoid reprocessing"""
    
    __tablename__ = "processing_cache"
    
    # Primary key
    id = Column(String, primary_key=True, index=True)
    
    # Cache key (hash of file + transformation type)
    cache_key = Column(String, unique=True, index=True, nullable=False)
    
    # Original document info
    original_filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False, index=True)
    transformation_type = Column(String, nullable=False)
    
    # Cached result
    output_path = Column(String, nullable=False)
    processing_time = Column(Float, nullable=False)
    cache_metadata = Column(JSON, nullable=True)
    
    # Cache management
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        expiry_time = datetime.utcnow() - timedelta(hours=settings.CACHE_EXPIRY_HOURS)
        return self.created_at < expiry_time
    
    @property
    def is_valid(self) -> bool:
        """Check if cached file still exists and cache is not expired"""
        return not self.is_expired and os.path.exists(self.output_path)
    
    def update_access(self):
        """Update access tracking"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


async def init_db():
    """Initialize database"""
    global engine, SessionLocal
    
    try:
        # Create database directory if needed
        db_dir = os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", ""))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Create engine
        engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
        )
        
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        print(f"✅ Database initialized: {settings.DATABASE_URL}")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Document operations
class DocumentOps:
    """Document database operations"""
    
    @staticmethod
    def create_document(
        db,
        filename: str,
        original_path: str,
        file_size: int,
        file_type: str,
        transformation_type: str = "deskewing",
        content_hash: Optional[str] = None
    ) -> Document:
        """Create a new document record"""
        document_id = str(uuid.uuid4())
        
        doc = Document(
            id=document_id,
            filename=filename,
            original_path=original_path,
            file_size=file_size,
            file_type=file_type,
            transformation_type=transformation_type,
            content_hash=content_hash,
            status=ProcessingStatus.PENDING
        )
        
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        return doc
    
    @staticmethod
    def get_document(db, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        return db.query(Document).filter(Document.id == document_id).first()
    
    @staticmethod
    def update_document_status(
        db,
        document_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> Optional[Document]:
        """Update document status"""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = status
            if error_message:
                doc.error_message = error_message
            
            if status == ProcessingStatus.IN_PROGRESS:
                doc.started_at = datetime.utcnow()
            elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                doc.completed_at = datetime.utcnow()
                if doc.started_at:
                    doc.processing_time = (doc.completed_at - doc.started_at).total_seconds()
            
            db.commit()
            db.refresh(doc)
        
        return doc
    
    @staticmethod
    def set_document_output(
        db,
        document_id: str,
        output_path: str,
        processing_metadata: Optional[dict] = None
    ) -> Optional[Document]:
        """Set document output path and metadata"""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.output_path = output_path
            if processing_metadata:
                doc.processing_metadata = processing_metadata
            
            db.commit()
            db.refresh(doc)
        
        return doc
    
    @staticmethod
    def get_documents_by_status(db, status: ProcessingStatus, limit: int = 100) -> List[Document]:
        """Get documents by status"""
        return db.query(Document).filter(Document.status == status).limit(limit).all()
    
    @staticmethod
    def cleanup_expired_documents(db, hours: int = 48) -> int:
        """Clean up old document records"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        expired_docs = db.query(Document).filter(
            Document.created_at < cutoff_time,
            Document.status.in_([ProcessingStatus.COMPLETED, ProcessingStatus.FAILED])
        ).all()
        
        count = 0
        for doc in expired_docs:
            # Remove output files if they exist
            if doc.output_path and os.path.exists(doc.output_path):
                try:
                    os.remove(doc.output_path)
                except OSError:
                    pass
            
            db.delete(doc)
            count += 1
        
        db.commit()
        return count


# Cache operations
class CacheOps:
    """Cache database operations"""
    
    @staticmethod
    def get_cache_key(file_hash: str, transformation_type: str) -> str:
        """Generate cache key"""
        return f"{file_hash}:{transformation_type}"
    
    @staticmethod
    def get_cached_result(db, file_hash: str, transformation_type: str) -> Optional[ProcessingCache]:
        """Get cached processing result"""
        cache_key = CacheOps.get_cache_key(file_hash, transformation_type)
        
        cache_entry = db.query(ProcessingCache).filter(
            ProcessingCache.cache_key == cache_key
        ).first()
        
        if cache_entry and cache_entry.is_valid:
            cache_entry.update_access()
            db.commit()
            return cache_entry
        elif cache_entry:
            # Remove invalid cache entry
            db.delete(cache_entry)
            db.commit()
        
        return None
    
    @staticmethod
    def create_cache_entry(
        db,
        file_hash: str,
        transformation_type: str,
        original_filename: str,
        output_path: str,
        processing_time: float,
        processing_metadata: Optional[dict] = None
    ) -> ProcessingCache:
        """Create cache entry"""
        cache_key = CacheOps.get_cache_key(file_hash, transformation_type)
        cache_id = str(uuid.uuid4())
        
        cache_entry = ProcessingCache(
            id=cache_id,
            cache_key=cache_key,
            file_hash=file_hash,
            transformation_type=transformation_type,
            original_filename=original_filename,
            output_path=output_path,
            processing_time=processing_time,
            cache_metadata=processing_metadata or {}
        )
        
        db.add(cache_entry)
        db.commit()
        db.refresh(cache_entry)
        
        return cache_entry
    
    @staticmethod
    def cleanup_expired_cache(db) -> int:
        """Clean up expired cache entries"""
        expiry_time = datetime.utcnow() - timedelta(hours=settings.CACHE_EXPIRY_HOURS)
        
        expired_entries = db.query(ProcessingCache).filter(
            ProcessingCache.created_at < expiry_time
        ).all()
        
        count = 0
        for entry in expired_entries:
            # Remove cached file
            if os.path.exists(entry.output_path):
                try:
                    os.remove(entry.output_path)
                except OSError:
                    pass
            
            db.delete(entry)
            count += 1
        
        db.commit()
        return count