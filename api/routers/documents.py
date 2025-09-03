#!/usr/bin/env python3
"""
Document Processing API Routes
REST endpoints for document transformation services
"""

import os
import hashlib
import tempfile
import mimetypes
from typing import Optional, List
import aiofiles
import httpx

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from ..core.config import settings, get_transformation_config, get_fallback_strategy
from ..core.database import get_db, DocumentOps, CacheOps
from ..core.logger import get_logger
from ..schemas.document import (
    DocumentResponse, 
    DocumentStatusResponse, 
    TransformationRequest,
    TransformationResponse,
    AvailableTransformationsResponse
)
from ..services.transformation import TransformationService
from ..services.file_handler import FileHandler

# Initialize router and logger
router = APIRouter()
logger = get_logger(__name__)

# Initialize services
transformation_service = TransformationService()
file_handler = FileHandler()


@router.post("/transform", response_model=TransformationResponse)
async def transform_document(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    transformations: str = Form("deskewing"),
    filename: Optional[str] = Form(None)
):
    """
    Transform a document with specified preprocessing steps
    
    **Upload Methods:**
    - **File Upload**: Upload file directly using the `file` parameter
    - **URL Reference**: Provide a URL to download the file using the `url` parameter
    
    **Transformation Types:**
    - **deskewing** (default): Fast orientation correction, skew detection, and cropping
    - **basic**: Orientation correction and cropping only
    - **enhanced**: Full preprocessing with noise reduction and contrast enhancement  
    - **comprehensive**: Complete pipeline with all features
    
    **Response:**
    Returns a document ID and processing status. Use the status endpoint to monitor progress.
    """
    
    try:
        # Validate input
        if not file and not url:
            raise HTTPException(
                status_code=400,
                detail="Either file upload or URL must be provided"
            )
        
        if file and url:
            raise HTTPException(
                status_code=400, 
                detail="Provide either file upload or URL, not both"
            )
        
        # Validate transformation type
        transformation_config = get_transformation_config(transformations)
        if not transformation_config:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transformation type: {transformations}"
            )
        
        # Handle file input
        if file:
            # Validate file
            await file_handler.validate_file(file)
            
            # Save uploaded file
            file_path, file_info = await file_handler.save_uploaded_file(file)
            original_filename = file.filename
            
        else:  # URL input
            # Validate and download from URL
            file_path, file_info = await file_handler.download_from_url(url, filename)
            original_filename = filename or os.path.basename(url)
        
        # Generate file hash for caching
        file_hash = await file_handler.calculate_file_hash(file_path)
        
        # Check cache first if enabled
        if settings.ENABLE_CACHING:
            cached_result = CacheOps.get_cached_result(db, file_hash, transformations)
            if cached_result:
                logger.info(f"🚀 Cache hit for {original_filename} - {transformations}")
                
                # Create document record pointing to cached result
                document = DocumentOps.create_document(
                    db=db,
                    filename=original_filename,
                    original_path=file_path,
                    file_size=file_info["size"],
                    file_type=file_info["type"],
                    transformation_type=transformations,
                    content_hash=file_hash
                )
                
                # Update to completed status immediately
                DocumentOps.update_document_status(db, document.id, "completed")
                DocumentOps.set_document_output(
                    db, 
                    document.id, 
                    cached_result.output_path,
                    cached_result.cache_metadata
                )
                
                return TransformationResponse(
                    document_id=document.id,
                    status="completed",
                    message=f"Document processed successfully (cached result)",
                    transformation_type=transformations,
                    estimated_time=f"{cached_result.processing_time:.1f} seconds",
                    location=f"/documents/{document.id}/result"
                )
        
        # Create document record
        document = DocumentOps.create_document(
            db=db,
            filename=original_filename,
            original_path=file_path,
            file_size=file_info["size"],
            file_type=file_info["type"],
            transformation_type=transformations,
            content_hash=file_hash
        )
        
        logger.info(f"📄 Created document record: {document.id} for {original_filename}")
        
        # Get fallback strategy based on file size
        fallback_config = get_fallback_strategy(file_info["size"])
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document.id,
            file_path,
            transformations,
            fallback_config
        )
        
        # Estimate processing time based on transformation type and file size
        transformation_info = get_transformation_config(transformations)
        estimated_time = transformation_info.get("expected_time", "5-30 seconds")
        
        return TransformationResponse(
            document_id=document.id,
            status="pending",
            message=f"Document processing started with {transformation_info['name']}",
            transformation_type=transformations,
            estimated_time=estimated_time,
            location=f"/documents/{document.id}/status"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Transform document error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the processing status of a document
    
    **Status Values:**
    - **pending**: Document is queued for processing
    - **in_progress**: Document is currently being processed
    - **completed**: Processing completed successfully
    - **failed**: Processing failed with error
    - **cancelled**: Processing was cancelled
    
    **Response includes:**
    - Current status and progress information
    - Processing time and estimated completion
    - Error details if processing failed
    - Download link when completed
    """
    
    document = DocumentOps.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Build response
    response_data = document.to_dict()
    
    # Add multi-page information if available
    metadata = document.processing_metadata or {}
    if metadata.get('is_multipage'):
        response_data["is_multipage"] = True
        response_data["total_pages"] = metadata.get('total_pages', 0)
        response_data["processed_pages"] = metadata.get('processed_pages', 0)
        all_outputs = metadata.get('all_page_outputs', [])
        if all_outputs:
            response_data["page_files"] = [os.path.basename(path) for path in all_outputs]
    else:
        response_data["is_multipage"] = False
        response_data["total_pages"] = 1
    
    # Add download link if completed
    if document.status == "completed" and document.output_path:
        response_data["download_url"] = f"/documents/{document_id}/result"
    
    # Add progress information
    if document.status == "in_progress":
        # Calculate estimated completion based on tasks completed
        total_tasks = len(get_transformation_config(document.transformation_type)["tasks"])
        completed_tasks = len(document.tasks_completed or [])
        progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        response_data["progress"] = {
            "percentage": round(progress_percentage, 1),
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "current_task": document.tasks_completed[-1] if document.tasks_completed else None
        }
    
    return DocumentStatusResponse(**response_data)


@router.get("/{document_id}/result")
async def get_document_result(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Download the processed document result
    
    **File Formats:**
    - Single documents return the processed file directly
    - Multi-page PDFs return a ZIP archive with all pages
    - Includes processing metadata and comparison images
    
    **Headers:**
    - Content-Disposition: attachment with original filename
    - Content-Type: appropriate MIME type for the file
    """
    
    document = DocumentOps.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Document processing not completed. Status: {document.status}"
        )
    
    if not document.output_path or not os.path.exists(document.output_path):
        raise HTTPException(status_code=404, detail="Processed file not found")
    
    # Check if this is a multi-page document
    metadata = document.processing_metadata or {}
    all_page_outputs = metadata.get('all_page_outputs', [])
    
    if len(all_page_outputs) > 1:
        # Multi-page document - create ZIP archive
        import zipfile
        import tempfile
        from datetime import datetime
        
        # Create temporary ZIP file
        zip_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        zip_temp.close()
        
        try:
            with zipfile.ZipFile(zip_temp.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, page_path in enumerate(all_page_outputs, 1):
                    if os.path.exists(page_path):
                        page_name = os.path.basename(page_path)
                        zipf.write(page_path, page_name)
                        logger.info(f"📦 Added {page_name} to ZIP archive")
                
                # Add processing summary if it exists
                doc_dir = os.path.dirname(document.output_path)
                summary_files = [f for f in os.listdir(doc_dir) if f.endswith('_summary.txt')]
                for summary_file in summary_files:
                    summary_path = os.path.join(doc_dir, summary_file)
                    if os.path.exists(summary_path):
                        zipf.write(summary_path, summary_file)
            
            # Generate filename for ZIP
            base_name = os.path.splitext(document.filename)[0]
            zip_filename = f"{base_name}_all_pages.zip"
            
            return FileResponse(
                zip_temp.name,
                media_type='application/zip',
                filename=zip_filename,
                headers={
                    "Content-Disposition": f"attachment; filename={zip_filename}",
                    "X-Total-Pages": str(len(all_page_outputs)),
                    "X-Processing-Type": "multi-page"
                }
            )
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(zip_temp.name):
                os.unlink(zip_temp.name)
            raise HTTPException(status_code=500, detail=f"Failed to create ZIP archive: {str(e)}")
    
    else:
        # Single page document - return file directly
        filename = f"processed_{document.filename}"
        base_name = os.path.splitext(filename)[0]
        processed_filename = f"{base_name}.png"
        
        media_type = mimetypes.guess_type(document.output_path)[0] or 'image/png'
        
        return FileResponse(
            document.output_path,
            media_type=media_type,
            filename=processed_filename,
            headers={
                "Content-Disposition": f"attachment; filename={processed_filename}",
                "X-Total-Pages": "1",
                "X-Processing-Type": "single-page"
            }
        )


@router.get("/{document_id}/metadata")
async def get_document_metadata(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed metadata about the document processing
    
    **Metadata includes:**
    - Original document information (size, type, hash)
    - Processing configuration and tasks executed
    - Quality metrics and detected issues
    - Timing information and performance stats
    """
    
    document = DocumentOps.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    metadata = {
        "document_info": {
            "document_id": document.id,
            "original_filename": document.filename,
            "file_size": document.file_size,
            "file_type": document.file_type,
            "content_hash": document.content_hash
        },
        "processing_info": {
            "transformation_type": document.transformation_type,
            "status": document.status,
            "tasks_completed": document.tasks_completed or [],
            "processing_time": document.processing_time,
            "error_message": document.error_message
        },
        "timestamps": {
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "started_at": document.started_at.isoformat() if document.started_at else None,
            "completed_at": document.completed_at.isoformat() if document.completed_at else None
        },
        "pipeline_metadata": document.processing_metadata or {}
    }
    
    return metadata


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a document and its processed results
    
    **Actions performed:**
    - Removes document record from database
    - Deletes original uploaded file
    - Deletes processed output files
    - Clears any cached results
    """
    
    document = DocumentOps.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove files
        if document.original_path and os.path.exists(document.original_path):
            os.remove(document.original_path)
        
        if document.output_path and os.path.exists(document.output_path):
            os.remove(document.output_path)
        
        # Remove from database
        db.delete(document)
        db.commit()
        
        logger.info(f"🗑️ Deleted document: {document_id}")
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.get("/transformations", response_model=AvailableTransformationsResponse)
async def get_available_transformations():
    """
    Get information about available transformation types
    
    **Transformation Types:**
    - **deskewing**: Fast preprocessing for OCR (default)
    - **basic**: Minimal processing for speed
    - **enhanced**: Quality-focused processing
    - **comprehensive**: Complete feature set
    
    **Each transformation includes:**
    - Description and use cases
    - Expected processing time
    - Quality level and features
    """
    
    from ..core.config import TRANSFORMATION_CONFIGS
    
    transformations = []
    for transform_id, config in TRANSFORMATION_CONFIGS.items():
        transformations.append({
            "id": transform_id,
            "name": config["name"],
            "description": config["description"],
            "expected_time": config["expected_time"],
            "quality": config["quality"],
            "tasks": len(config["tasks"])
        })
    
    return AvailableTransformationsResponse(transformations=transformations)


# Background processing function
async def process_document_background(
    document_id: str,
    file_path: str,
    transformation_type: str,
    fallback_config: dict
):
    """Background task for document processing"""
    
    from ..core.database import SessionLocal
    
    db = SessionLocal()
    
    try:
        logger.info(f"🔄 Starting background processing for document {document_id}")
        
        # Update status to in_progress
        DocumentOps.update_document_status(db, document_id, "in_progress")
        
        # Process the document
        result = await transformation_service.process_document(
            document_id=document_id,
            file_path=file_path,
            transformation_type=transformation_type,
            fallback_config=fallback_config,
            db=db
        )
        
        if result["success"]:
            # Update document with results
            DocumentOps.update_document_status(db, document_id, "completed")
            DocumentOps.set_document_output(
                db, 
                document_id, 
                result["output_path"],
                result.get("metadata", {})
            )
            
            logger.info(f"✅ Document {document_id} processed successfully")
            
            # Cache the result if enabled
            if settings.ENABLE_CACHING:
                document = DocumentOps.get_document(db, document_id)
                if document and document.content_hash:
                    CacheOps.create_cache_entry(
                        db=db,
                        file_hash=document.content_hash,
                        transformation_type=transformation_type,
                        original_filename=document.filename,
                        output_path=result["output_path"],
                        processing_time=result.get("processing_time", 0),
                        processing_metadata=result.get("metadata", {})
                    )
            
        else:
            # Update document with error
            error_message = result.get("error", "Unknown processing error")
            DocumentOps.update_document_status(db, document_id, "failed", error_message)
            logger.error(f"❌ Document {document_id} processing failed: {error_message}")
    
    except Exception as e:
        logger.error(f"❌ Background processing error for {document_id}: {str(e)}")
        DocumentOps.update_document_status(db, document_id, "failed", str(e))
    
    finally:
        db.close()


# Admin endpoints for monitoring
@router.get("/admin/stats")
async def get_processing_stats(db: Session = Depends(get_db)):
    """Get processing statistics (admin endpoint)"""
    
    from sqlalchemy import func
    from ..core.database import Document
    
    stats = db.query(
        Document.status,
        func.count(Document.id).label('count')
    ).group_by(Document.status).all()
    
    total_docs = db.query(func.count(Document.id)).scalar()
    
    return {
        "total_documents": total_docs,
        "status_breakdown": {stat.status: stat.count for stat in stats},
        "cache_enabled": settings.ENABLE_CACHING,
        "max_file_size": f"{settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
    }