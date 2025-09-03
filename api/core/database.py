#!/usr/bin/env python3
"""
Database Models and Operations
SQLAlchemy models for document processing tracking
"""

import os
import uuid
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean, Float, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
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
    """Enhanced Document processing record with comprehensive metadata tracking"""
    
    __tablename__ = "documents"
    
    # Primary identifier
    id = Column(String, primary_key=True, index=True)
    
    # File information
    filename = Column(String, nullable=False, index=True)  # Index for fast filename lookup
    original_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String, nullable=False)
    content_hash = Column(String, nullable=True, index=True)  # SHA-256 hash for uniqueness
    
    # Enhanced file metadata
    original_filename = Column(String, nullable=False, index=True)  # Original uploaded filename
    mime_type = Column(String, nullable=True)
    file_extension = Column(String, nullable=True)
    
    # Document structure metadata
    page_count = Column(Integer, nullable=True)  # Total pages in document
    document_dimensions = Column(JSON, nullable=True)  # {"width": int, "height": int, "unit": "px"}
    total_file_size = Column(Integer, nullable=True)  # Total size including all pages
    
    # Processing information
    status = Column(String, default=ProcessingStatus.PENDING, index=True)
    transformation_type = Column(String, default=TransformationType.DESKEWING)
    pipeline_applied = Column(JSON, nullable=True)  # List of processing steps applied
    tasks_completed = Column(JSON, default=list)
    
    # Results and storage paths
    output_path = Column(String, nullable=True)
    storage_paths = Column(JSON, nullable=True)  # {"processed": path, "thumbnails": path, "metadata": path}
    blob_references = Column(JSON, nullable=True)  # For cloud storage references
    processing_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    processing_metadata = Column(JSON, nullable=True)
    
    # Enhanced timestamps with indexes
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Processing details
    worker_id = Column(String, nullable=True)
    processing_node = Column(String, nullable=True)
    
    # Relationship with page-level data
    pages = relationship("DocumentPage", back_populates="document", cascade="all, delete-orphan")
    
    # Create composite indexes for performance
    __table_args__ = (
        Index('idx_hash_filename', content_hash, original_filename),
        Index('idx_status_created', status, created_at),
        Index('idx_type_size', file_type, file_size),
    )
    
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
    
    def update_access_time(self):
        """Update last accessed timestamp"""
        self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "document_id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "content_hash": self.content_hash,
            "page_count": self.page_count,
            "document_dimensions": self.document_dimensions,
            "status": self.status,
            "transformation_type": self.transformation_type,
            "pipeline_applied": self.pipeline_applied or [],
            "tasks_completed": self.tasks_completed or [],
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "processing_metadata": self.processing_metadata or {},
            "storage_paths": self.storage_paths or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "output_available": bool(self.output_path and os.path.exists(self.output_path))
        }
    
    def get_metadata_summary(self) -> dict:
        """Get comprehensive metadata summary"""
        return {
            "document_id": self.id,
            "file_info": {
                "original_filename": self.original_filename,
                "filename": self.filename,
                "file_size": self.file_size,
                "file_type": self.file_type,
                "mime_type": self.mime_type,
                "content_hash": self.content_hash,
            },
            "structure": {
                "page_count": self.page_count,
                "dimensions": self.document_dimensions,
                "total_size": self.total_file_size
            },
            "processing": {
                "status": self.status,
                "transformation_type": self.transformation_type,
                "pipeline_applied": self.pipeline_applied or [],
                "processing_time": self.processing_time,
                "processing_metadata": self.processing_metadata or {}
            },
            "storage": {
                "output_path": self.output_path,
                "storage_paths": self.storage_paths or {},
                "blob_references": self.blob_references or {}
            },
            "timestamps": {
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
            }
        }


class DocumentPage(Base):
    """Page-level storage and metadata for O(1) retrieval"""
    
    __tablename__ = "document_pages"
    
    # Primary identifier
    id = Column(String, primary_key=True, index=True)
    
    # Foreign key to document
    document_id = Column(String, ForeignKey('documents.id'), nullable=False, index=True)
    
    # Page identification
    page_number = Column(Integer, nullable=False, index=True)  # 1-based page numbering
    page_hash = Column(String, nullable=True, index=True)  # Hash of individual page content
    
    # Page metadata
    page_dimensions = Column(JSON, nullable=True)  # {"width": int, "height": int, "dpi": int}
    page_size_bytes = Column(Integer, nullable=True)
    
    # Storage paths for fast retrieval
    original_image_path = Column(String, nullable=True)  # Path to original page image
    processed_image_path = Column(String, nullable=True)  # Path to processed page image
    thumbnail_path = Column(String, nullable=True)  # Path to thumbnail for quick preview
    
    # Processing metadata for this page
    processing_status = Column(String, default="pending")
    processing_time = Column(Float, nullable=True)
    processing_metadata = Column(JSON, nullable=True)  # Page-specific processing details
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)  # Computed quality score (0.0-1.0)
    is_blank = Column(Boolean, default=False)  # Whether page is blank/empty
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship back to document
    document = relationship("Document", back_populates="pages")
    
    # Create composite indexes for performance
    __table_args__ = (
        Index('idx_document_page', document_id, page_number),
        Index('idx_hash_lookup', page_hash),
    )
    
    def __repr__(self):
        return f"<DocumentPage(id={self.id}, document_id={self.document_id}, page_number={self.page_number})>"
    
    def update_access_time(self):
        """Update last accessed timestamp"""
        self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "page_id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "page_hash": self.page_hash,
            "dimensions": self.page_dimensions,
            "size_bytes": self.page_size_bytes,
            "processing_status": self.processing_status,
            "processing_time": self.processing_time,
            "quality_score": self.quality_score,
            "is_blank": self.is_blank,
            "paths": {
                "original": self.original_image_path,
                "processed": self.processed_image_path,
                "thumbnail": self.thumbnail_path
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "available": bool(self.processed_image_path and os.path.exists(self.processed_image_path))
        }


class FileMetadata(Base):
    """Comprehensive file metadata for advanced querying and analytics"""
    
    __tablename__ = "file_metadata"
    
    # Primary identifier
    id = Column(String, primary_key=True, index=True)
    
    # Foreign key to document
    document_id = Column(String, ForeignKey('documents.id'), nullable=False, index=True)
    
    # File characteristics
    file_format = Column(String, nullable=True)  # Detected file format
    compression = Column(String, nullable=True)  # Compression type if any
    color_space = Column(String, nullable=True)  # RGB, CMYK, Grayscale, etc.
    bit_depth = Column(Integer, nullable=True)  # Bits per channel
    
    # Technical metadata
    creator_software = Column(String, nullable=True)  # Software that created the file
    creation_date = Column(DateTime, nullable=True)  # File creation date from metadata
    modification_date = Column(DateTime, nullable=True)  # Last modification date
    
    # Content analysis
    has_text = Column(Boolean, default=False)  # Whether document contains text
    has_images = Column(Boolean, default=False)  # Whether document contains embedded images
    language_detected = Column(String, nullable=True)  # Detected language
    
    # Processing analytics
    average_processing_time = Column(Float, nullable=True)  # Average time per page
    total_processing_operations = Column(Integer, default=0)  # Number of times processed
    
    # Custom metadata (extensible)
    custom_metadata = Column(JSON, nullable=True)  # For additional metadata fields
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<FileMetadata(id={self.id}, document_id={self.document_id})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "metadata_id": self.id,
            "document_id": self.document_id,
            "file_characteristics": {
                "format": self.file_format,
                "compression": self.compression,
                "color_space": self.color_space,
                "bit_depth": self.bit_depth
            },
            "technical_info": {
                "creator_software": self.creator_software,
                "creation_date": self.creation_date.isoformat() if self.creation_date else None,
                "modification_date": self.modification_date.isoformat() if self.modification_date else None
            },
            "content_analysis": {
                "has_text": self.has_text,
                "has_images": self.has_images,
                "language_detected": self.language_detected
            },
            "processing_analytics": {
                "average_processing_time": self.average_processing_time,
                "total_operations": self.total_processing_operations
            },
            "custom_metadata": self.custom_metadata or {},
            "timestamps": {
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None
            }
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


# Enhanced Document Operations for File Management System
class EnhancedDocumentOps(DocumentOps):
    """Enhanced document operations with comprehensive metadata and search capabilities"""
    
    @staticmethod
    def create_document_with_metadata(
        db,
        filename: str,
        original_filename: str,
        original_path: str,
        file_size: int,
        file_type: str,
        content_hash: str,
        transformation_type: str = "deskewing",
        page_count: Optional[int] = None,
        document_dimensions: Optional[Dict[str, Any]] = None,
        mime_type: Optional[str] = None
    ) -> Document:
        """Create document with comprehensive metadata"""
        document_id = str(uuid.uuid4())
        
        doc = Document(
            id=document_id,
            filename=filename,
            original_filename=original_filename,
            original_path=original_path,
            file_size=file_size,
            total_file_size=file_size,
            file_type=file_type,
            mime_type=mime_type,
            file_extension=os.path.splitext(original_filename)[1].lower(),
            content_hash=content_hash,
            page_count=page_count,
            document_dimensions=document_dimensions,
            transformation_type=transformation_type,
            status=ProcessingStatus.PENDING
        )
        
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        return doc
    
    @staticmethod
    def search_by_filename(db, filename: str, exact_match: bool = False) -> List[Document]:
        """Search documents by filename"""
        query = db.query(Document)
        
        if exact_match:
            return query.filter(
                (Document.filename == filename) | 
                (Document.original_filename == filename)
            ).all()
        else:
            search_term = f"%{filename}%"
            return query.filter(
                (Document.filename.like(search_term)) | 
                (Document.original_filename.like(search_term))
            ).all()
    
    @staticmethod
    def search_by_hash(db, content_hash: str) -> List[Document]:
        """Search documents by content hash (SHA-256)"""
        return db.query(Document).filter(Document.content_hash == content_hash).all()
    
    @staticmethod
    def search_documents(
        db,
        filename: Optional[str] = None,
        content_hash: Optional[str] = None,
        file_type: Optional[str] = None,
        status: Optional[str] = None,
        transformation_type: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """Advanced document search with multiple filters"""
        query = db.query(Document)
        
        if filename:
            search_term = f"%{filename}%"
            query = query.filter(
                (Document.filename.like(search_term)) |
                (Document.original_filename.like(search_term))
            )
        
        if content_hash:
            query = query.filter(Document.content_hash == content_hash)
        
        if file_type:
            query = query.filter(Document.file_type == file_type)
        
        if status:
            query = query.filter(Document.status == status)
        
        if transformation_type:
            query = query.filter(Document.transformation_type == transformation_type)
        
        if date_from:
            query = query.filter(Document.created_at >= date_from)
        
        if date_to:
            query = query.filter(Document.created_at <= date_to)
        
        return query.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
    
    @staticmethod
    def get_document_with_pages(db, document_id: str) -> Optional[Document]:
        """Get document with all its pages loaded"""
        from sqlalchemy.orm import joinedload
        
        return db.query(Document).options(
            joinedload(Document.pages)
        ).filter(Document.id == document_id).first()
    
    @staticmethod
    def update_document_storage_paths(
        db,
        document_id: str,
        storage_paths: Dict[str, str],
        blob_references: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """Update document storage paths and blob references"""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.storage_paths = storage_paths
            if blob_references:
                doc.blob_references = blob_references
            db.commit()
            db.refresh(doc)
        return doc


class DocumentPageOps:
    """Page-level database operations for O(1) retrieval"""
    
    @staticmethod
    def create_page(
        db,
        document_id: str,
        page_number: int,
        page_hash: Optional[str] = None,
        page_dimensions: Optional[Dict[str, Any]] = None,
        page_size_bytes: Optional[int] = None
    ) -> DocumentPage:
        """Create a new document page record"""
        page_id = str(uuid.uuid4())
        
        page = DocumentPage(
            id=page_id,
            document_id=document_id,
            page_number=page_number,
            page_hash=page_hash,
            page_dimensions=page_dimensions,
            page_size_bytes=page_size_bytes
        )
        
        db.add(page)
        db.commit()
        db.refresh(page)
        
        return page
    
    @staticmethod
    def get_page(db, document_id: str, page_number: int) -> Optional[DocumentPage]:
        """Get specific page by document ID and page number"""
        page = db.query(DocumentPage).filter(
            DocumentPage.document_id == document_id,
            DocumentPage.page_number == page_number
        ).first()
        
        if page:
            page.update_access_time()
            db.commit()
        
        return page
    
    @staticmethod
    def get_document_pages(db, document_id: str) -> List[DocumentPage]:
        """Get all pages for a document, ordered by page number"""
        return db.query(DocumentPage).filter(
            DocumentPage.document_id == document_id
        ).order_by(DocumentPage.page_number).all()
    
    @staticmethod
    def update_page_paths(
        db,
        page_id: str,
        original_path: Optional[str] = None,
        processed_path: Optional[str] = None,
        thumbnail_path: Optional[str] = None
    ) -> Optional[DocumentPage]:
        """Update page storage paths"""
        page = db.query(DocumentPage).filter(DocumentPage.id == page_id).first()
        if page:
            if original_path:
                page.original_image_path = original_path
            if processed_path:
                page.processed_image_path = processed_path
            if thumbnail_path:
                page.thumbnail_path = thumbnail_path
            
            db.commit()
            db.refresh(page)
        
        return page
    
    @staticmethod
    def update_page_processing_status(
        db,
        page_id: str,
        status: str,
        processing_time: Optional[float] = None,
        quality_score: Optional[float] = None,
        is_blank: Optional[bool] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[DocumentPage]:
        """Update page processing status and metrics"""
        page = db.query(DocumentPage).filter(DocumentPage.id == page_id).first()
        if page:
            page.processing_status = status
            if processing_time is not None:
                page.processing_time = processing_time
            if quality_score is not None:
                page.quality_score = quality_score
            if is_blank is not None:
                page.is_blank = is_blank
            if processing_metadata:
                page.processing_metadata = processing_metadata
            
            if status == "completed":
                page.processed_at = datetime.utcnow()
            
            db.commit()
            db.refresh(page)
        
        return page


class FileMetadataOps:
    """File metadata database operations"""
    
    @staticmethod
    def create_metadata(
        db,
        document_id: str,
        file_format: Optional[str] = None,
        compression: Optional[str] = None,
        color_space: Optional[str] = None,
        bit_depth: Optional[int] = None,
        creator_software: Optional[str] = None,
        creation_date: Optional[datetime] = None,
        has_text: bool = False,
        has_images: bool = False,
        language_detected: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> FileMetadata:
        """Create comprehensive file metadata record"""
        metadata_id = str(uuid.uuid4())
        
        metadata = FileMetadata(
            id=metadata_id,
            document_id=document_id,
            file_format=file_format,
            compression=compression,
            color_space=color_space,
            bit_depth=bit_depth,
            creator_software=creator_software,
            creation_date=creation_date,
            has_text=has_text,
            has_images=has_images,
            language_detected=language_detected,
            custom_metadata=custom_metadata or {}
        )
        
        db.add(metadata)
        db.commit()
        db.refresh(metadata)
        
        return metadata
    
    @staticmethod
    def get_metadata(db, document_id: str) -> Optional[FileMetadata]:
        """Get file metadata by document ID"""
        return db.query(FileMetadata).filter(
            FileMetadata.document_id == document_id
        ).first()
    
    @staticmethod
    def update_processing_analytics(
        db,
        document_id: str,
        processing_time: float
    ) -> Optional[FileMetadata]:
        """Update processing analytics"""
        metadata = db.query(FileMetadata).filter(
            FileMetadata.document_id == document_id
        ).first()
        
        if metadata:
            # Update processing analytics
            if metadata.average_processing_time is None:
                metadata.average_processing_time = processing_time
            else:
                # Calculate running average
                total_ops = metadata.total_processing_operations
                current_total = metadata.average_processing_time * total_ops
                metadata.average_processing_time = (current_total + processing_time) / (total_ops + 1)
            
            metadata.total_processing_operations += 1
            metadata.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(metadata)
        
        return metadata


# Utility functions for file hashing
def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error calculating file hash: {e}")
        return ""


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