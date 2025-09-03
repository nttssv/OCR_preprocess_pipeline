#!/usr/bin/env python3
"""
Document Processing Pipeline API
FastAPI-based REST API for PDF transformation services

Features:
- PDF transformation with default deskewing
- File upload and URL reference support
- Document ID tracking and status monitoring
- Streaming/batch processing for large files
- Benchmark-based fallback strategies
"""

import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our API modules
from api.routers import documents
from api.routers import document_management
from api.core.config import settings
from api.core.database import init_db
from api.core.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    setup_logging()
    await init_db()
    
    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs(settings.CACHE_DIR, exist_ok=True)
    
    print(f"🚀 API Server starting on {settings.HOST}:{settings.PORT}")
    print(f"📁 Upload directory: {settings.UPLOAD_DIR}")
    print(f"📁 Output directory: {settings.OUTPUT_DIR}")
    print(f"🔧 Max file size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB")
    print(f"⚡ Max workers: {settings.MAX_WORKERS}")
    
    yield
    
    # Shutdown
    print("🛑 API Server shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Document Processing Pipeline API",
    description="""
    REST API for intelligent document preprocessing and transformation.
    
    ## Features
    
    * **PDF Transformation**: Upload PDFs and apply preprocessing (default: deskewing)
    * **File Upload**: Support for direct file uploads and URL references
    * **Status Tracking**: Monitor processing status with document IDs
    * **Performance Optimized**: Streaming processing for large files with intelligent caching
    * **Fallback Strategies**: Automatic optimization based on file size and processing time
    
    ## Default Transformation Pipeline
    
    1. **Orientation Correction**: Detect and fix upside-down/rotated pages
    2. **Skew Detection & Correction**: Advanced ±15° skew detection and correction
    3. **Document Cropping**: Remove borders, punch holes, and scanner artifacts
    
    ## Usage Examples
    
    ### Upload a file:
    ```bash
    curl -X POST "http://localhost:8000/documents/transform" \\
         -F "file=@document.pdf" \\
         -F "transformations=deskewing"
    ```
    
    ### Check status:
    ```bash
    curl "http://localhost:8000/documents/{document_id}/status"
    ```
    """,
    version="1.0.0",
    contact={
        "name": "Document Processing Pipeline",
        "url": "https://github.com/your-repo/OCR_preprocess_pipeline",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    documents.router,
    prefix="/documents",
    tags=["documents"]
)

# Include document management router
app.include_router(
    document_management.router,
    tags=["Document Management"]
)

# Serve static files (for processed documents)
if os.path.exists(settings.OUTPUT_DIR):
    app.mount("/files", StaticFiles(directory=settings.OUTPUT_DIR), name="files")

# Health check endpoint
@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Document Processing Pipeline API"
    }

# Root endpoint with API documentation
@app.get("/", response_class=HTMLResponse, tags=["system"])
async def root():
    """Root endpoint with API information"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Processing Pipeline API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #27ae60; font-weight: bold; }
            .path { color: #e74c3c; font-family: monospace; }
            .description { color: #7f8c8d; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h1 class="header">🚀 Document Processing Pipeline API</h1>
        <p>Intelligent document preprocessing and transformation service.</p>
        
        <h2>📚 API Documentation</h2>
        <p><a href="/docs" target="_blank">Interactive API Docs (Swagger UI)</a></p>
        <p><a href="/redoc" target="_blank">Alternative Docs (ReDoc)</a></p>
        
        <h2>🔧 Key Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="path">/documents/transform</span>
            <div class="description">Transform PDF documents with intelligent preprocessing</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/documents/{document_id}/status</span>
            <div class="description">Check processing status of a document</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/documents/{document_id}/result</span>
            <div class="description">Download processed document results</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/documents/{document_id}/metadata</span>
            <div class="description">Get comprehensive document metadata and analytics</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/documents/search</span>
            <div class="description">Search documents by filename, hash, or other filters</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/documents/{document_id}/pages/{pageNumber}</span>
            <div class="description">Retrieve preprocessed page image with O(1) fast access</div>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/health</span>
            <div class="description">API health check</div>
        </div>
        
        <h2>💡 Quick Start</h2>
        <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px;">
# Upload and transform a PDF
curl -X POST "http://localhost:8000/documents/transform" \\
     -F "file=@document.pdf" \\
     -F "transformations=deskewing"

# Check processing status
curl "http://localhost:8000/documents/{document_id}/status"

# Download results
curl "http://localhost:8000/documents/{document_id}/result" -o result.zip
        </pre>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d;">
            <p>Document Processing Pipeline API v1.0.0</p>
        </footer>
    </body>
    </html>
    """)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Resource not found",
        "message": "The requested resource was not found",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later.",
        "status_code": 500
    }


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.MAX_WORKERS,
        access_log=True
    )