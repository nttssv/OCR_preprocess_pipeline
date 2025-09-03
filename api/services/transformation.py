#!/usr/bin/env python3
"""
Transformation Service
Core service for document processing pipeline integration with multi-page PDF support
"""

import os
import sys
import time
import tempfile
import shutil
from typing import Dict, Any, Optional
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from document_processing_pipeline import DocumentProcessingPipeline
from pipeline_config import get_execution_mode
from document_specific_config import apply_document_config
from document_specific_config_fast import apply_fast_document_config
from utils.blank_page_detection import BlankPageDetector

from ..core.config import settings, get_transformation_config
from ..core.logger import get_logger
from ..core.database import DocumentOps

logger = get_logger(__name__)


class TransformationService:
    """Service for handling document transformations using the existing pipeline"""
    
    def __init__(self):
        self.blank_detector = BlankPageDetector()
        logger.info("🔧 Transformation service initialized")
    
    async def process_document(
        self,
        document_id: str,
        file_path: str,
        transformation_type: str = "deskewing",
        fallback_config: Optional[Dict] = None,
        db = None
    ) -> Dict[str, Any]:
        """
        Process a document using the existing pipeline
        
        Args:
            document_id: Unique document identifier
            file_path: Path to the input file
            transformation_type: Type of transformation to apply
            fallback_config: Optional fallback configuration
            db: Database session for progress tracking
            
        Returns:
            Dictionary with processing results
        """
        
        start_time = time.time()
        logger.info(f"🔄 Processing document {document_id}: {os.path.basename(file_path)}")
        
        try:
            # Check if this is a PDF file that needs multi-page processing
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.pdf':
                return await self._process_pdf_multipage(
                    document_id, file_path, transformation_type, fallback_config, db
                )
            else:
                return await self._process_single_image(
                    document_id, file_path, transformation_type, fallback_config, db
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"❌ Document {document_id} processing error: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time,
                "metadata": {
                    "transformation_type": transformation_type,
                    "error_details": str(e)
                }
            }
    
    async def _process_pdf_multipage(
        self,
        document_id: str,
        pdf_path: str,
        transformation_type: str = "deskewing",
        fallback_config: Optional[Dict] = None,
        db = None
    ) -> Dict[str, Any]:
        """
        Process a multi-page PDF by converting all pages to separate PNG files
        
        Args:
            document_id: Unique document identifier
            pdf_path: Path to the PDF file
            transformation_type: Type of transformation to apply
            fallback_config: Optional fallback configuration
            db: Database session for progress tracking
            
        Returns:
            Dictionary with processing results including all page outputs
        """
        
        start_time = time.time()
        logger.info(f"📄 Processing multi-page PDF: {os.path.basename(pdf_path)}")
        
        try:
            # Get transformation configuration
            transform_config = get_transformation_config(transformation_type)
            if not transform_config:
                raise ValueError(f"Invalid transformation type: {transformation_type}")
            
            # Create unique temporary and output directories for this document
            temp_dir = os.path.join(settings.TEMP_DIR, f"doc_{document_id}")
            output_dir = os.path.join(settings.OUTPUT_DIR, f"doc_{document_id}")
            
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert PDF to individual page images
            pdf_temp_dir = os.path.join(temp_dir, "pdf_pages")
            os.makedirs(pdf_temp_dir, exist_ok=True)
            
            # Use the existing PDF conversion utility
            from utils.ingestion import convert_pdf_to_images
            page_images = convert_pdf_to_images(pdf_path, pdf_temp_dir, dpi=250)
            
            if not page_images:
                raise RuntimeError("Failed to convert PDF pages to images")
            
            logger.info(f"📄 Converted PDF to {len(page_images)} page images")
            
            # Process each page individually
            processed_pages = []
            all_page_results = []
            total_processing_time = 0
            
            for i, page_image_path in enumerate(page_images, 1):
                logger.info(f"🔄 Processing page {i}/{len(page_images)}")
                
                # Process this page as a single image
                page_result = await self._process_single_image(
                    f"{document_id}_page_{i}",
                    page_image_path,
                    transformation_type,
                    fallback_config,
                    None  # Don't update DB for individual pages
                )
                
                if page_result["success"]:
                    # Copy the processed page to the main output directory with proper naming
                    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    page_output_name = f"{base_name}_page_{i:03d}.png"
                    page_output_path = os.path.join(output_dir, page_output_name)
                    
                    import shutil
                    shutil.copy2(page_result["output_path"], page_output_path)
                    
                    processed_pages.append(page_output_path)
                    logger.info(f"✅ Page {i} processed: {page_output_name}")
                else:
                    logger.error(f"❌ Page {i} processing failed: {page_result.get('error', 'Unknown error')}")
                
                all_page_results.append(page_result)
                total_processing_time += page_result.get("processing_time", 0)
            
            # Update database with final results
            if db:
                document = DocumentOps.get_document(db, document_id)
                if document:
                    DocumentOps.set_document_output(
                        db, 
                        document_id, 
                        processed_pages[0] if processed_pages else None,  # Primary output
                        {
                            "total_pages": len(page_images),
                            "processed_pages": len(processed_pages),
                            "all_page_outputs": processed_pages,
                            "transformation_type": transformation_type
                        }
                    )
            
            processing_time = time.time() - start_time
            
            if processed_pages:
                logger.info(f"✅ Multi-page PDF processed: {len(processed_pages)} pages in {processing_time:.2f}s")
                
                return {
                    "success": True,
                    "output_path": processed_pages[0],  # Primary output (first page)
                    "all_outputs": processed_pages,     # All page outputs
                    "total_pages": len(page_images),
                    "processed_pages": len(processed_pages),
                    "processing_time": processing_time,
                    "metadata": {
                        "transformation_type": transformation_type,
                        "is_multipage": True,
                        "page_results": all_page_results,
                        "fallback_used": bool(fallback_config)
                    }
                }
            else:
                raise RuntimeError("No pages were successfully processed")
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Multi-page PDF processing failed: {str(e)}"
            logger.error(f"❌ Document {document_id} processing error: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time,
                "metadata": {
                    "transformation_type": transformation_type,
                    "is_multipage": True,
                    "error_details": str(e)
                }
            }
    
    async def _process_single_image(
        self,
        document_id: str,
        file_path: str,
        transformation_type: str = "deskewing",
        fallback_config: Optional[Dict] = None,
        db = None
    ) -> Dict[str, Any]:
        """
        Process a single image file using the existing pipeline
        
        Args:
            document_id: Unique document identifier
            file_path: Path to the input image file
            transformation_type: Type of transformation to apply
            fallback_config: Optional fallback configuration
            db: Database session for progress tracking
            
        Returns:
            Dictionary with processing results
        """
        
        start_time = time.time()
        logger.info(f"🔄 Processing single image {document_id}: {os.path.basename(file_path)}")
        
        try:
            # Get transformation configuration
            transform_config = get_transformation_config(transformation_type)
            if not transform_config:
                raise ValueError(f"Invalid transformation type: {transformation_type}")
            
            # Create unique temporary and output directories for this document
            temp_dir = os.path.join(settings.TEMP_DIR, f"doc_{document_id}")
            output_dir = os.path.join(settings.OUTPUT_DIR, f"doc_{document_id}")
            
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Early blank page detection to save processing time
            blank_result = self.blank_detector.is_blank_page(file_path)
            if blank_result['is_blank']:
                logger.info(f"📄 Document {document_id}: BLANK PAGE DETECTED - Skipping processing")
                
                # Create a simple output for blank pages
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                blank_output_path = os.path.join(output_dir, f"{base_name}_blank.png")
                shutil.copy2(file_path, blank_output_path)
                
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "output_path": blank_output_path,
                    "processing_time": processing_time,
                    "metadata": {
                        "blank_page": True,
                        "blank_reason": blank_result.get('reason', 'Unknown'),
                        "transformation_type": transformation_type,
                        "tasks_completed": []
                    }
                }
            
            # Initialize the document processing pipeline
            pipeline = DocumentProcessingPipeline(
                input_folder=os.path.dirname(file_path),
                output_folder=output_dir,
                temp_folder=temp_dir
            )
            
            # Configure pipeline based on transformation type
            await self._configure_pipeline(pipeline, transformation_type, fallback_config)
            
            # Apply document-specific configuration
            filename = os.path.basename(file_path)
            if transformation_type in ["basic", "deskewing"] or (fallback_config and fallback_config.get("strategy") == "basic"):
                # Use fast configuration for basic/fallback processing
                apply_fast_document_config(pipeline, filename)
            else:
                # Use standard configuration for enhanced processing
                apply_document_config(pipeline, filename)
            
            # Get the task manager and configured tasks
            task_manager = pipeline.get_task_manager()
            if not task_manager:
                raise RuntimeError("Task manager not available")
            
            # Get execution mode configuration
            mode_config = get_execution_mode(self._get_execution_mode(transformation_type))
            enabled_tasks = mode_config['tasks']
            standalone_mode = mode_config.get('standalone_mode', False)
            
            # Update database with tasks to be completed
            if db:
                document = DocumentOps.get_document(db, document_id)
                if document:
                    DocumentOps.set_document_output(
                        db, 
                        document_id, 
                        None,  # No output yet
                        {"total_tasks": len(enabled_tasks), "planned_tasks": enabled_tasks}
                    )
            
            # Determine file type (the pipeline expects 'image' for most cases)
            file_ext = os.path.splitext(file_path)[1].lower()
            file_type = 'image'  # We're always processing images in this method
            
            # Process with progress tracking
            result = await self._process_with_progress_tracking(
                task_manager, enabled_tasks, file_path, file_type, temp_dir, 
                standalone_mode, document_id, db
            )
            
            # Handle the result and create final output
            final_output_path = await self._handle_processing_result(
                result, enabled_tasks, file_path, output_dir, filename, transformation_type
            )
            
            processing_time = time.time() - start_time
            
            if final_output_path and os.path.exists(final_output_path):
                logger.info(f"✅ Document {document_id} processed successfully in {processing_time:.2f}s")
                
                return {
                    "success": True,
                    "output_path": final_output_path,
                    "processing_time": processing_time,
                    "metadata": {
                        "transformation_type": transformation_type,
                        "tasks_completed": enabled_tasks,
                        "pipeline_results": result if isinstance(result, dict) else {},
                        "fallback_used": bool(fallback_config)
                    }
                }
            else:
                raise RuntimeError("Processing completed but no output file was generated")
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"❌ Document {document_id} processing error: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time,
                "metadata": {
                    "transformation_type": transformation_type,
                    "error_details": str(e)
                }
            }
    
    async def _configure_pipeline(
        self, 
        pipeline: DocumentProcessingPipeline, 
        transformation_type: str,
        fallback_config: Optional[Dict] = None
    ):
        """Configure pipeline tasks based on transformation type"""
        
        # Get transformation configuration
        transform_config = get_transformation_config(transformation_type)
        
        # Apply fallback if specified
        if fallback_config and "strategy" in fallback_config:
            fallback_transform_config = get_transformation_config(fallback_config["strategy"])
            if fallback_transform_config:
                logger.info(f"🔄 Applying fallback strategy: {fallback_config['strategy']}")
                transform_config = fallback_transform_config
        
        # Disable all tasks first
        for task_key in pipeline.pipeline_config.keys():
            task_name = pipeline.pipeline_config[task_key]['name']
            pipeline.enable_task(task_name, False)
        
        # Enable only the tasks for this transformation type
        for task_id in transform_config['tasks']:
            if task_id in pipeline.pipeline_config:
                task_name = pipeline.pipeline_config[task_id]['name']
                pipeline.enable_task(task_name, True)
                logger.debug(f"✅ Enabled task: {task_name}")
    
    def _get_execution_mode(self, transformation_type: str) -> str:
        """Map transformation type to execution mode"""
        
        mode_mapping = {
            "deskewing": "orient_skew_crop",
            "basic": "orient_only", 
            "enhanced": "with_enhancement",
            "comprehensive": "comprehensive_pipeline"
        }
        
        return mode_mapping.get(transformation_type, "orient_skew_crop")
    
    async def _process_with_progress_tracking(
        self,
        task_manager,
        enabled_tasks,
        file_path: str,
        file_type: str,
        temp_dir: str,
        standalone_mode: bool,
        document_id: str,
        db
    ):
        """Process document with progress tracking"""
        
        try:
            # Use the existing task manager's run_task_chain method
            result = task_manager.run_task_chain(
                enabled_tasks, 
                file_path, 
                file_type, 
                temp_dir, 
                standalone_mode
            )
            
            # Update progress in database
            if db and result:
                completed_tasks = []
                for task_id in enabled_tasks:
                    if task_id in result and result[task_id] and result[task_id].get('status') == 'completed':
                        completed_tasks.append(task_id)
                
                document = DocumentOps.get_document(db, document_id)
                if document:
                    # Update the document with completed tasks
                    document.tasks_completed = completed_tasks
                    db.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing error: {str(e)}")
            raise
    
    async def _handle_processing_result(
        self,
        result: Dict,
        enabled_tasks: list,
        file_path: str,
        output_dir: str,
        filename: str,
        transformation_type: str
    ) -> Optional[str]:
        """Handle processing result and create final output"""
        
        if not result:
            raise RuntimeError("No processing result returned")
        
        # Find the final output from the last successful task
        final_output = None
        for task_id in reversed(enabled_tasks):
            if task_id in result and result[task_id] and result[task_id].get('output'):
                final_output = result[task_id]['output']
                break
        
        if not final_output or not os.path.exists(final_output):
            raise RuntimeError("No valid output file found from processing tasks")
        
        # Create the final result file
        base_name = os.path.splitext(filename)[0]
        final_result_path = os.path.join(output_dir, f"{base_name}_processed.png")
        
        # Copy the final output to the result location
        shutil.copy2(final_output, final_result_path)
        
        # Create a summary file with processing information
        summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        await self._create_processing_summary(
            summary_path, filename, transformation_type, enabled_tasks, result
        )
        
        return final_result_path
    
    async def _create_processing_summary(
        self,
        summary_path: str,
        filename: str,
        transformation_type: str,
        enabled_tasks: list,
        result: Dict
    ):
        """Create a processing summary file"""
        
        try:
            with open(summary_path, 'w') as f:
                f.write(f"Document Processing Summary\n")
                f.write(f"=" * 40 + "\n\n")
                f.write(f"File: {filename}\n")
                f.write(f"Transformation: {transformation_type}\n")
                f.write(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Tasks executed:\n")
                
                for i, task_id in enumerate(enabled_tasks, 1):
                    task_result = result.get(task_id, {})
                    status = task_result.get('status', 'unknown')
                    f.write(f"  {i}. {task_id}: {status}\n")
                    
                    if status == 'completed' and 'processing_time' in task_result:
                        f.write(f"     Processing time: {task_result['processing_time']:.2f}s\n")
                
                f.write(f"\nProcessing completed successfully!\n")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not create processing summary: {str(e)}")
    
    async def get_supported_transformations(self) -> Dict[str, Any]:
        """Get information about supported transformation types"""
        
        from ..core.config import TRANSFORMATION_CONFIGS
        return TRANSFORMATION_CONFIGS
    
    async def validate_transformation_type(self, transformation_type: str) -> bool:
        """Validate that a transformation type is supported"""
        
        supported_types = await self.get_supported_transformations()
        return transformation_type in supported_types