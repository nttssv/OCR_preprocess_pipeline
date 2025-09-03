#!/usr/bin/env python3
"""
Document Management API Endpoints
Advanced file management with metadata tracking, search, and fast retrieval
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..core.database import (
    get_db, Document, DocumentPage, FileMetadata,
    EnhancedDocumentOps, DocumentPageOps, FileMetadataOps
)
from ..core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Document Management"])


# Pydantic models for API responses
class DocumentMetadataResponse(BaseModel):
    """Document metadata API response model"""
    document_id: str
    file_info: Dict[str, Any]
    structure: Dict[str, Any]
    processing: Dict[str, Any]
    storage: Dict[str, Any]
    timestamps: Dict[str, Any]

class DocumentSearchResponse(BaseModel):
    """Document search results response model"""
    total_results: int
    documents: List[Dict[str, Any]]
    page: int
    page_size: int

class PageInfoResponse(BaseModel):
    """Page information response model"""
    page_id: str
    document_id: str
    page_number: int
    page_hash: Optional[str]
    dimensions: Optional[Dict[str, Any]]
    size_bytes: Optional[int]
    processing_status: str
    quality_score: Optional[float]
    is_blank: bool
    available: bool
    paths: Dict[str, Optional[str]]
    timestamps: Dict[str, Any]


@router.get("/{document_id}/metadata", response_model=DocumentMetadataResponse)
async def get_document_metadata(
    document_id: str = Path(..., description="Document ID"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive document metadata
    
    Returns stored metadata including:
    - File information (hash, size, type)
    - Document structure (page count, dimensions)
    - Processing history and pipeline applied
    - Storage paths and references
    - Timestamps and access history
    """
    
    try:
        # Get document with enhanced metadata
        document = EnhancedDocumentOps.get_document_with_pages(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update access time
        document.update_access_time()
        db.commit()
        
        # Get file metadata
        file_metadata = FileMetadataOps.get_metadata(db, document_id)
        
        # Build comprehensive metadata response
        metadata = document.get_metadata_summary()
        
        # Add file metadata if available
        if file_metadata:
            metadata["file_info"].update({
                "format": file_metadata.file_format,
                "compression": file_metadata.compression,
                "color_space": file_metadata.color_space,
                "bit_depth": file_metadata.bit_depth,
                "creator_software": file_metadata.creator_software
            })
            
            metadata["content_analysis"] = {
                "has_text": file_metadata.has_text,
                "has_images": file_metadata.has_images,
                "language_detected": file_metadata.language_detected
            }
            
            metadata["processing_analytics"] = {
                "average_processing_time": file_metadata.average_processing_time,
                "total_operations": file_metadata.total_processing_operations
            }
        
        # Add pages information
        if document.pages:
            metadata["pages"] = [
                {
                    "page_number": page.page_number,
                    "processing_status": page.processing_status,
                    "quality_score": page.quality_score,
                    "is_blank": page.is_blank,
                    "available": bool(page.processed_image_path and os.path.exists(page.processed_image_path))
                }
                for page in document.pages
            ]
        
        logger.info(f"📊 Retrieved metadata for document: {document_id}")
        return DocumentMetadataResponse(**metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving document metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metadata: {str(e)}")


@router.get("/search", response_model=DocumentSearchResponse)
async def search_documents(
    name: Optional[str] = Query(None, description="Search by filename (partial match)"),
    hash: Optional[str] = Query(None, description="Search by SHA-256 hash (exact match)"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    transformation_type: Optional[str] = Query(None, description="Filter by transformation type"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db)
):
    """
    Search documents with advanced filtering
    
    **Search Options:**
    - **name**: Partial filename match (searches both original and processed names)
    - **hash**: Exact SHA-256 hash match for uniqueness verification
    - **file_type**: Filter by file type (pdf, png, jpg, etc.)
    - **status**: Filter by processing status (pending, completed, failed, etc.)
    - **transformation_type**: Filter by applied transformation (basic, deskewing, etc.)
    - **date_from/date_to**: Date range filtering
    
    **Performance Features:**
    - Indexed searches on hash and filename for fast lookup
    - Pagination for large result sets
    - Sorted by creation date (newest first)
    """
    
    try:
        # Parse date filters if provided
        parsed_date_from = None
        parsed_date_to = None
        
        if date_from:
            try:
                parsed_date_from = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO format.")
        
        if date_to:
            try:
                parsed_date_to = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO format.")
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Search with enhanced operations
        documents = EnhancedDocumentOps.search_documents(
            db=db,
            filename=name,
            content_hash=hash,
            file_type=file_type,
            status=status,
            transformation_type=transformation_type,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            limit=page_size,
            offset=offset
        )
        
        # Get total count for pagination (simplified for performance)
        total_results = len(documents) if len(documents) < page_size else page_size * page + 1
        
        # Convert to response format
        document_data = []
        for doc in documents:
            doc_dict = doc.to_dict()
            # Add page count if available
            if doc.pages:
                doc_dict["page_count"] = len(doc.pages)
                doc_dict["pages_available"] = sum(
                    1 for page in doc.pages 
                    if page.processed_image_path and os.path.exists(page.processed_image_path)
                )
            
            document_data.append(doc_dict)
        
        search_params = {
            "name": name,
            "hash": hash,
            "file_type": file_type,
            "status": status,
            "transformation_type": transformation_type,
            "date_from": date_from,
            "date_to": date_to
        }
        active_filters = {k: v for k, v in search_params.items() if v is not None}
        
        logger.info(f"🔍 Document search completed: {len(documents)} results, filters: {active_filters}")
        
        return DocumentSearchResponse(
            total_results=total_results,
            documents=document_data,
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/{document_id}/pages/{page_number}", response_class=FileResponse)
async def get_document_page(
    document_id: str = Path(..., description="Document ID"),
    page_number: int = Path(..., ge=1, description="Page number (1-based)"),
    thumbnail: bool = Query(False, description="Return thumbnail instead of full image"),
    db: Session = Depends(get_db)
):
    """
    Retrieve preprocessed page image with O(1) access
    
    **Fast Retrieval Features:**
    - Pre-generated page-level images during ingestion
    - Direct file system access for optimal performance
    - Thumbnail support for quick previews
    - Automatic access tracking for analytics
    
    **Response:**
    - Returns the processed page image directly
    - Content-Type set appropriately for image format
    - Supports both full resolution and thumbnail versions
    """
    
    try:
        # Get page with O(1) lookup using indexes
        page = DocumentPageOps.get_page(db, document_id, page_number)
        if not page:
            raise HTTPException(
                status_code=404, 
                detail=f"Page {page_number} not found for document {document_id}"
            )
        
        # Determine which image to return
        image_path = None
        if thumbnail and page.thumbnail_path:
            image_path = page.thumbnail_path
        elif page.processed_image_path:
            image_path = page.processed_image_path
        elif page.original_image_path:
            image_path = page.original_image_path
        
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(
                status_code=404,
                detail=f"Page image not available for page {page_number}"
            )
        
        # Update access time for analytics (async to avoid blocking)
        page.update_access_time()
        db.commit()
        
        # Determine media type
        file_ext = os.path.splitext(image_path)[1].lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.bmp': 'image/bmp'
        }
        media_type = media_type_map.get(file_ext, 'image/png')
        
        # Generate filename for download
        page_type = "thumbnail" if thumbnail else "processed"
        filename = f"{document_id}_page_{page_number}_{page_type}{file_ext}"
        
        logger.info(f"📄 Serving page {page_number} for document {document_id} ({page_type})")
        
        return FileResponse(
            path=image_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving page: {str(e)}")


@router.get("/{document_id}/pages", response_model=List[PageInfoResponse])
async def get_document_pages_info(
    document_id: str = Path(..., description="Document ID"),
    db: Session = Depends(get_db)
):
    """
    Get information about all pages in a document
    
    **Returns:**
    - List of all pages with metadata
    - Processing status for each page
    - Availability and quality metrics
    - Fast access without loading actual images
    """
    
    try:
        # Get document to verify existence
        document = EnhancedDocumentOps.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get all pages for the document
        pages = DocumentPageOps.get_document_pages(db, document_id)
        
        # Convert to response format
        pages_info = []
        for page in pages:
            page_dict = page.to_dict()
            pages_info.append(PageInfoResponse(
                page_id=page_dict["page_id"],
                document_id=page_dict["document_id"],
                page_number=page_dict["page_number"],
                page_hash=page_dict["page_hash"],
                dimensions=page_dict["dimensions"],
                size_bytes=page_dict["size_bytes"],
                processing_status=page_dict["processing_status"],
                quality_score=page_dict["quality_score"],
                is_blank=page_dict["is_blank"],
                available=page_dict["available"],
                paths=page_dict["paths"],
                timestamps={
                    "created_at": page_dict["created_at"],
                    "processed_at": page_dict["processed_at"],
                    "last_accessed": page_dict["last_accessed"]
                }
            ))
        
        logger.info(f"📋 Retrieved info for {len(pages)} pages of document {document_id}")
        return pages_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving pages info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving pages info: {str(e)}")


@router.get("/{document_id}/analytics")
async def get_document_analytics(
    document_id: str = Path(..., description="Document ID"),
    db: Session = Depends(get_db)
):
    """
    Get document processing analytics and performance metrics
    
    **Analytics Include:**
    - Processing performance metrics
    - Access patterns and frequency
    - Quality scores and blank page detection
    - Storage usage and optimization suggestions
    """
    
    try:
        # Get document with pages
        document = EnhancedDocumentOps.get_document_with_pages(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get file metadata for additional analytics
        file_metadata = FileMetadataOps.get_metadata(db, document_id)
        
        # Calculate analytics
        analytics = {
            "document_id": document_id,
            "processing_performance": {
                "total_processing_time": document.processing_time,
                "average_time_per_page": None,
                "processing_efficiency": None
            },
            "access_analytics": {
                "last_accessed": document.last_accessed.isoformat() if document.last_accessed else None,
                "total_operations": file_metadata.total_processing_operations if file_metadata else 0
            },
            "quality_metrics": {
                "total_pages": len(document.pages) if document.pages else 0,
                "processed_pages": 0,
                "blank_pages": 0,
                "average_quality_score": None,
                "quality_distribution": {}
            },
            "storage_analytics": {
                "original_size": document.file_size,
                "total_storage_used": document.total_file_size or document.file_size,
                "compression_ratio": None,
                "storage_paths": document.storage_paths or {}
            }
        }
        
        # Calculate page-level analytics
        if document.pages:
            processed_pages = sum(1 for page in document.pages if page.processing_status == "completed")
            blank_pages = sum(1 for page in document.pages if page.is_blank)
            quality_scores = [page.quality_score for page in document.pages if page.quality_score is not None]
            
            analytics["quality_metrics"].update({
                "processed_pages": processed_pages,
                "blank_pages": blank_pages,
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else None
            })
            
            # Calculate average processing time per page
            page_times = [page.processing_time for page in document.pages if page.processing_time is not None]
            if page_times:
                analytics["processing_performance"]["average_time_per_page"] = sum(page_times) / len(page_times)
        
        # Calculate processing efficiency
        if document.processing_time and document.page_count:
            expected_time = document.page_count * 30  # 30 seconds baseline per page
            analytics["processing_performance"]["processing_efficiency"] = expected_time / document.processing_time
        
        logger.info(f"📊 Generated analytics for document {document_id}")
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")