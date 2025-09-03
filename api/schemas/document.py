#!/usr/bin/env python3
"""
Document API Schemas
Pydantic models for request/response validation
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TransformationRequest(BaseModel):
    """Request model for document transformation"""
    transformations: str = Field(
        default="deskewing",
        description="Type of transformation to apply",
        examples=["deskewing", "basic", "enhanced", "comprehensive"]
    )
    filename: Optional[str] = Field(
        default=None,
        description="Custom filename (used when uploading via URL)"
    )


class TransformationResponse(BaseModel):
    """Response model for transformation request"""
    document_id: str = Field(description="Unique document identifier")
    status: str = Field(description="Current processing status")
    message: str = Field(description="Human-readable status message")
    transformation_type: str = Field(description="Type of transformation applied")
    estimated_time: str = Field(description="Estimated processing time")
    location: str = Field(description="URL to check status or download result")


class DocumentResponse(BaseModel):
    """Basic document information response"""
    document_id: str
    filename: str
    file_size: int
    file_type: str
    status: str
    transformation_type: str
    created_at: Optional[datetime]
    processing_time: Optional[float]
    output_available: bool


class ProgressInfo(BaseModel):
    """Processing progress information"""
    percentage: float = Field(description="Completion percentage (0-100)")
    completed_tasks: int = Field(description="Number of completed tasks")
    total_tasks: int = Field(description="Total number of tasks")
    current_task: Optional[str] = Field(description="Currently executing task")


class DocumentStatusResponse(BaseModel):
    """Detailed document status response"""
    document_id: str
    filename: str
    file_size: int
    file_type: str
    status: str
    transformation_type: str
    tasks_completed: List[str] = Field(default_factory=list)
    processing_time: Optional[float]
    error_message: Optional[str]
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    output_available: bool
    download_url: Optional[str] = Field(default=None)
    progress: Optional[ProgressInfo] = Field(default=None)


class TransformationInfo(BaseModel):
    """Information about a transformation type"""
    id: str = Field(description="Transformation identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Detailed description")
    expected_time: str = Field(description="Expected processing time range")
    quality: str = Field(description="Quality level (fast/balanced/high/maximum)")
    tasks: int = Field(description="Number of processing tasks")


class AvailableTransformationsResponse(BaseModel):
    """Response with available transformation types"""
    transformations: List[TransformationInfo]


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    status_code: int = Field(description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class ProcessingStatsResponse(BaseModel):
    """Processing statistics response"""
    total_documents: int
    status_breakdown: Dict[str, int]
    cache_enabled: bool
    max_file_size: str


class FileValidationError(BaseModel):
    """File validation error details"""
    field: str = Field(description="Field that failed validation")
    message: str = Field(description="Validation error message")
    received_value: Optional[Any] = Field(description="Value that failed validation")


class UploadResponse(BaseModel):
    """File upload response"""
    success: bool
    message: str
    file_info: Optional[Dict[str, Any]] = Field(default=None)
    errors: Optional[List[FileValidationError]] = Field(default=None)