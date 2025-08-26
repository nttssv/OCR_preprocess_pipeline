#!/usr/bin/env python3
"""
Document Processing Pipeline - Main Entry Point
===============================================
Intelligent parallel processing with automatic quality optimization

Features:
- Parallel processing for faster execution
- Automatic quality fixes for blurry/gray documents  
- Smart document-specific configurations
- Full OCR preprocessing pipeline

Usage:
    python run.py                        # Process all files with auto-detected workers
    python run.py --workers 4            # Use 4 parallel workers
    python run.py --mode skew_only        # Run specific processing mode
    python run.py --input my_docs         # Use custom input folder
"""

import os
import sys
import time
import argparse
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
from functools import partial
import cv2
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processing_pipeline import DocumentProcessingPipeline
from pipeline_config import get_execution_mode, get_pipeline_config
from document_specific_config import apply_document_config
from document_specific_config_fast import apply_fast_document_config
from utils.ingestion import (
    expand_inputs_with_pdfs,
    compute_page_hash,
    record_page,
    record_document,
    compute_document_id,
)
from utils.file_integrity import FileIntegrityChecker
from utils.file_locker import FileLocker

class IntelligentDocumentProcessor:
    """
    Main Document Processing Pipeline with Parallel Execution and Quality Optimization
    """
    
    def __init__(self, input_folder="input", output_folder="output", temp_folder="temp", max_workers=None, pdf_dpi: int = 250, index_db_path: str = "ingestion/index.db", enable_comparisons: bool = True, allow_duplicates: bool = False, production_mode: bool = False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.temp_folder = temp_folder
        self.pdf_dpi = pdf_dpi
        self.index_db_path = index_db_path
        self.enable_comparisons = enable_comparisons and not production_mode
        self.allow_duplicates = allow_duplicates
        self.production_mode = production_mode
        
        # Determine optimal number of workers - REDUCED FOR STABILITY
        if max_workers is None:
            cpu_count = mp.cpu_count()
            self.max_workers = min(cpu_count, 2)  # Cap at 2 to prevent race conditions
        else:
            self.max_workers = max_workers
        
        print(f"🚀 Document Processing Pipeline initialized with {self.max_workers} workers")
        if self.production_mode:
            print(f"🏭 Production mode: Single output file per page")
        else:
            print(f"✨ Quality optimization enabled (auto-fixes blur/gray issues)")
        
        # Create necessary folders
        self._create_folders()
        
        # Initialize file integrity and locking systems
        self.file_integrity_checker = FileIntegrityChecker()
        self.file_locker = FileLocker()
    
    def _create_folders(self):
        """Create necessary folders for the pipeline"""
        folders = [self.input_folder, self.output_folder, self.temp_folder]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
    
    def get_input_files(self):
        """Expand inputs: convert PDFs to page images, dedup using index, and return processing list."""
        if not os.path.exists(self.input_folder):
            print(f"❌ Input folder '{self.input_folder}' not found")
            return []

        # Skip deduplication if allow_duplicates is enabled
        if self.allow_duplicates:
            print("🐛 Debug mode: Bypassing deduplication")
            images_to_process, skipped = expand_inputs_with_pdfs(
                input_folder=self.input_folder,
                temp_folder=self.temp_folder,
                dpi=self.pdf_dpi,
                db_path=None,  # No database = no deduplication
            )
        else:
            images_to_process, skipped = expand_inputs_with_pdfs(
                input_folder=self.input_folder,
                temp_folder=self.temp_folder,
                dpi=self.pdf_dpi,
                db_path=self.index_db_path,
            )

        if skipped:
            print(f"⚠️  Skipped {len(skipped)} duplicate pages (already seen)")

        # Register all input files for integrity monitoring
        print("🔒 Registering files for integrity monitoring...")
        for _, file_path in images_to_process:
            if os.path.exists(file_path):
                self.file_integrity_checker.register_file(file_path)
        
        print(f"📁 Prepared {len(images_to_process)} page images for processing")
        return images_to_process
    
    def generate_comparison(self, original_path, processed_path, output_path, is_blank=False):
        """Generate side-by-side comparison of original vs processed image"""
        try:
            # Read original image
            original = cv2.imread(original_path)
            if original is None:
                print(f"❌ Could not read original image for comparison: {original_path}")
                return False
            
            # Handle blank page case
            if is_blank:
                # Create a blank page indicator image
                height, width = original.shape[:2]
                blank_indicator = np.ones((height, width, 3), dtype=np.uint8) * 255
                
                # Add "BLANK PAGE" text overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3.0
                thickness = 4
                text = "BLANK PAGE"
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Center the text
                text_x = (width - text_width) // 2
                text_y = (height + text_height) // 2
                
                # Add semi-transparent background
                cv2.rectangle(blank_indicator, (text_x - 30, text_y - text_height - 30), 
                             (text_x + text_width + 30, text_y + 30), (200, 200, 200), -1)
                
                # Add text
                cv2.putText(blank_indicator, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)
                
                processed = blank_indicator
            else:
                # Read processed image for normal comparison
                processed = cv2.imread(processed_path)
                if processed is None:
                    print(f"❌ Could not read processed image for comparison: {processed_path}")
                    return False
            
            # Get dimensions
            orig_h, orig_w = original.shape[:2]
            proc_h, proc_w = processed.shape[:2]
            
            # Calculate target height (use the larger one)
            target_height = max(orig_h, proc_h)
            
            # Resize both images to same height, maintaining aspect ratio
            orig_ratio = orig_w / orig_h
            proc_ratio = proc_w / proc_h
            
            new_orig_w = int(target_height * orig_ratio)
            new_proc_w = int(target_height * proc_ratio)
            
            resized_orig = cv2.resize(original, (new_orig_w, target_height))
            resized_proc = cv2.resize(processed, (new_proc_w, target_height))
            
            # Add labels
            label_height = 30
            label_orig = np.ones((label_height, new_orig_w, 3), dtype=np.uint8) * 255
            label_proc = np.ones((label_height, new_proc_w, 3), dtype=np.uint8) * 255
            
            cv2.putText(label_orig, "ORIGINAL", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(label_proc, "PROCESSED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Combine images with labels
            orig_with_label = np.vstack([label_orig, resized_orig])
            proc_with_label = np.vstack([label_proc, resized_proc])
            
            # Create horizontal comparison
            comparison = np.hstack([orig_with_label, proc_with_label])
            
            # Save comparison
            cv2.imwrite(output_path, comparison)
            return True
            
        except Exception as e:
            print(f"❌ Error generating comparison: {str(e)}")
            return False
    
    def process_single_file(self, file_info, mode_name):
        """Process a single file through the pipeline with quality optimization"""
        file_type, file_path = file_info
        filename = os.path.basename(file_path)
        
        try:
            print(f"🔄 Processing {filename} in worker {os.getpid()}")
            
            # Check file integrity before processing
            with self.file_locker.file_lock(file_path):
                is_valid, message = self.file_integrity_checker.verify_file_integrity(file_path)
                if not is_valid:
                    print(f"❌ File integrity check failed for {filename}: {message}")
                    return {
                        'filename': filename,
                        'status': 'failed',
                        'error': f'File integrity check failed: {message}',
                        'duration': 0
                    }
                print(f"✅ File integrity verified for {filename}")
                
                # Create backup before processing
                backup_dir = os.path.join(self.temp_folder, 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f"💾 Created backup: {os.path.basename(backup_path)}")
            
            # Create unique temp folder for this file to avoid conflicts
            file_temp_folder = os.path.join(self.temp_folder, f"worker_{os.getpid()}_{filename}")
            os.makedirs(file_temp_folder, exist_ok=True)
            
            # Create unique output folder for this file
            file_output_folder = os.path.join(self.output_folder, f"processed_{filename}")
            os.makedirs(file_output_folder, exist_ok=True)
            
            # 🔍 EARLY BLANK PAGE DETECTION - Skip processing if page is blank
            from utils.blank_page_detection import BlankPageDetector
            blank_detector = BlankPageDetector()
            blank_result = blank_detector.is_blank_page(file_path)
            
            if blank_result['is_blank']:
                print(f"📄 {filename}: BLANK PAGE DETECTED - Skipping processing")
                # Create blank page result
                blank_output = blank_detector.create_blank_page_result(file_path, file_output_folder)
                
                # Generate comparison for blank page
                if self.enable_comparisons:
                    comparison_path = os.path.join(file_output_folder, f"{filename}_comparison.png")
                    if self.generate_comparison(file_path, None, comparison_path, is_blank=True):
                        print(f"📊 Generated blank page comparison: {os.path.basename(comparison_path)}")
                
                return {
                    'filename': filename,
                    'status': 'blank_page',
                    'result': blank_output,
                    'duration': 0,
                    'output_folder': file_output_folder,
                    'blank_detection': blank_result
                }
            
            # Initialize pipeline for this file
            pipeline = DocumentProcessingPipeline(
                input_folder=self.input_folder,
                output_folder=file_output_folder,
                temp_folder=file_temp_folder
            )
            
            # Configure pipeline for the specified mode
            self._configure_pipeline_for_mode(pipeline, mode_name)
            
            # 🔧 APPLY CONFIGURATION - Quality fixes or fast processing
            if mode_name == 'production_mode':
                apply_fast_document_config(pipeline, filename)
            else:
                apply_document_config(pipeline, filename)
            
            # Process only this file
            start_time = time.time()
            
            # Get task manager and run the configured tasks
            task_manager = pipeline.get_task_manager()
            if not task_manager:
                return {
                    'filename': filename,
                    'status': 'failed',
                    'error': 'Task manager not available',
                    'duration': 0
                }
            
            # Get enabled tasks for this mode
            mode_config = get_execution_mode(mode_name)
            enabled_tasks = mode_config['tasks']
            standalone_mode = mode_config.get('standalone_mode', False)
            
            # For production mode, don't force standalone mode since we need proper task chain
            # if mode_name == 'production_mode':
            #     standalone_mode = True
            
            # Run the task chain (PDFs are pre-converted to images, so file_type is 'image')
            result = task_manager.run_task_chain(enabled_tasks, file_path, 'image', file_temp_folder, standalone_mode)
            
            processing_time = time.time() - start_time
            
            # Copy final result(s) to output folder
            output_result_path = None
            output_result_paths = []
            if result:
                # Check if we have multi-page results from task_7_multipage_segmentation
                multipage_result = result.get('task_7_multipage_segmentation')
                if multipage_result and multipage_result.get('pages_detected', 0) > 1:
                    # Handle multi-page documents
                    all_outputs = multipage_result.get('all_outputs', [])
                    if all_outputs:
                        base_filename = os.path.splitext(filename)[0]
                        import shutil
                        
                        for i, page_output in enumerate(all_outputs):
                            if os.path.exists(page_output):
                                # Create result file with format {filename}_result1.png, {filename}_result2.png, etc.
                                result_filename = f"{base_filename}_result{i+1}.png"
                                result_path = os.path.join(file_output_folder, result_filename)
                                shutil.copy2(page_output, result_path)
                                output_result_paths.append(result_path)
                                print(f"📄 Generated multi-page result: {result_filename}")
                        
                        # Set primary output to first page
                        output_result_path = output_result_paths[0] if output_result_paths else None
                        
                        # Generate comparison image for multi-page (use first page) if enabled
                        if self.enable_comparisons and output_result_path:
                            comparison_path = os.path.join(file_output_folder, f"{base_filename}_comparison.png")
                            if self.generate_comparison(file_path, output_result_path, comparison_path):
                                print(f"📊 Generated comparison: {os.path.basename(comparison_path)}")
                    else:
                        # Fallback to single page handling
                        self._handle_single_page_result(result, enabled_tasks, file_path, file_output_folder, filename)
                else:
                    # Handle single-page documents
                    output_result_path = self._handle_single_page_result(result, enabled_tasks, file_path, file_output_folder, filename)
                    if output_result_path:
                        output_result_paths = [output_result_path]

            # Record page hash in index to support dedup across runs
            try:
                p_hash = compute_page_hash(file_path)
                # Derive a document id if original file exists (best-effort)
                # If the page comes from a converted PDF, its parent directory includes the PDF base name
                parent = Path(file_path).parent
                possible_pdf = None
                for ext in ('.pdf', '.PDF'):
                    cand = Path(self.input_folder) / f"{parent.name}{ext}"
                    if cand.exists():
                        possible_pdf = str(cand)
                        break
                document_id = compute_document_id(possible_pdf) if possible_pdf else compute_document_id(file_path)
                record_document(self.index_db_path, document_id, possible_pdf or file_path)
                record_page(self.index_db_path, p_hash, document_id, file_path, output_result_path)
            except Exception:
                pass
            
            # Check file integrity after processing
            with self.file_locker.file_lock(file_path):
                is_valid, message = self.file_integrity_checker.verify_file_integrity(file_path)
                if not is_valid:
                    print(f"⚠️  File integrity compromised after processing {filename}: {message}")
                    # Continue with result but flag the issue
                    result['integrity_warning'] = message
            
            return {
                'filename': filename,
                'status': 'completed',
                'result': result,
                'duration': processing_time,
                'output_folder': file_output_folder
            }
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            
            # Check if file was corrupted during processing
            try:
                with self.file_locker.file_lock(file_path):
                    is_valid, message = self.file_integrity_checker.verify_file_integrity(file_path)
                    if not is_valid:
                        print(f"⚠️  File {filename} was corrupted during processing: {message}")
                        # Try to restore from backup if available
                        backup_path = os.path.join(self.temp_folder, 'backups', os.path.basename(file_path))
                        if os.path.exists(backup_path):
                            print(f"🔄 Attempting to restore {filename} from backup...")
                            if self.file_integrity_checker.restore_file_from_backup(file_path, backup_path):
                                print(f"✅ Successfully restored {filename} from backup")
                            else:
                                print(f"❌ Failed to restore {filename} from backup")
            except Exception as recovery_error:
                print(f"⚠️  Error during file recovery: {recovery_error}")
            
            return {
                'filename': filename,
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _configure_pipeline_for_mode(self, pipeline, mode_name):
        """Configure the pipeline for a specific execution mode"""
        
        mode_config = get_execution_mode(mode_name)
        enabled_tasks = mode_config['tasks']
        
        # Disable all tasks first
        for task_key in pipeline.pipeline_config.keys():
            pipeline.enable_task(pipeline.pipeline_config[task_key]['name'], False)
        
        # Enable only the tasks for this mode
        for task_id in enabled_tasks:
            if task_id in pipeline.pipeline_config:
                task_name = pipeline.pipeline_config[task_id]['name']
                pipeline.enable_task(task_name, True)
    
    def run(self, mode_name='full_pipeline'):
        """Run the pipeline with intelligent parallel processing and quality optimization"""
        
        print(f"🚀 Starting Document Processing Pipeline")
        print(f"📋 Mode: {mode_name}")
        print(f"👥 Workers: {self.max_workers}")
        print(f"🔧 Quality fixes: ENABLED (auto-detects blur/gray issues)")
        print("=" * 60)
        
        # Get input files
        input_files = self.get_input_files()
        if not input_files:
            print("❌ No input files found")
            return False
        
        print(f"📄 Processing {len(input_files)} files in parallel...")
        
        start_time = time.time()
        results = []
        
        # Process files in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create partial function with mode_name
            process_func = partial(self.process_single_file, mode_name=mode_name)
            
            # Submit all files for processing
            future_to_file = {executor.submit(process_func, file_info): file_info 
                             for file_info in input_files}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'completed':
                        print(f"✅ {result['filename']} completed in {result['duration']:.2f}s")
                    else:
                        print(f"❌ {result['filename']} failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    filename = os.path.basename(file_info[1])
                    print(f"❌ {filename} exception: {str(e)}")
                    results.append({
                        'filename': filename,
                        'status': 'failed',
                        'error': str(e),
                        'duration': 0
                    })
        
        total_time = time.time() - start_time
        
        # Final file integrity check
        print("\n🔒 Performing final file integrity check...")
        integrity_results = self.file_integrity_checker.verify_all_files()
        corrupted_files = self.file_integrity_checker.get_corrupted_files()
        
        if corrupted_files:
            print(f"⚠️  Found {len(corrupted_files)} corrupted files:")
            for file_path in corrupted_files:
                print(f"   - {os.path.basename(file_path)}")
        else:
            print("✅ All files maintain integrity")
        
        # Generate summary
        self._generate_summary(results, total_time, mode_name)
        
        return True
    
    def _handle_single_page_result(self, result, enabled_tasks, file_path, file_output_folder, filename):
        """Handle result generation for single-page documents"""
        
        # Find the final output from the last successful task
        final_output = None
        for task_id in reversed(enabled_tasks):
            if task_id in result and result[task_id].get('output'):
                final_output = result[task_id]['output']
                break
        
        if final_output and os.path.exists(final_output):
            import shutil
            
            if self.production_mode:
                # Production mode: Generate only 1 clean result file
                base_name = os.path.splitext(filename)[0]
                final_result_path = os.path.join(file_output_folder, f"{base_name}.png")
                shutil.copy2(final_output, final_result_path)
                print(f"📄 Production result: {base_name}.png")
                return final_result_path
            else:
                # Standard mode: Generate result with suffix
                final_result_path = os.path.join(file_output_folder, f"{os.path.splitext(filename)[0]}_result.png")
                shutil.copy2(final_output, final_result_path)
                
                # Generate comparison image (original vs final result) if enabled
                if self.enable_comparisons:
                    comparison_path = os.path.join(file_output_folder, f"{os.path.splitext(filename)[0]}_comparison.png")
                    if self.generate_comparison(file_path, final_result_path, comparison_path):
                        print(f"📊 Generated comparison: {os.path.basename(comparison_path)}")
                
                return final_result_path
        
        return None
    
    def _generate_summary(self, results, total_time, mode_name):
        """Generate processing summary"""
        
        successful = [r for r in results if r['status'] == 'completed']
        failed = [r for r in results if r['status'] == 'failed']
        blank_pages = [r for r in results if r['status'] == 'blank_page']
        
        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        print(f"🚀 Mode: {mode_name}")
        print(f"👥 Workers: {self.max_workers}")
        print(f"📄 Total files: {len(results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"📄 Blank pages: {len(blank_pages)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        
        if successful:
            avg_time = sum(r['duration'] for r in successful) / len(successful)
            print(f"⏱️  Average time per file: {avg_time:.2f} seconds")
            
            # Calculate speedup compared to sequential processing
            total_processing_time = sum(r['duration'] for r in successful)
            speedup = total_processing_time / total_time if total_time > 0 else 1
            print(f"🚀 Speedup factor: {speedup:.2f}x")
        
        if blank_pages:
            print(f"\n📄 Blank pages detected:")
            for result in blank_pages:
                blank_info = result.get('blank_detection', {})
                reason = blank_info.get('reason', 'Unknown')
                print(f"   - {result['filename']}: {reason}")
        
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed:
                print(f"   - {result['filename']}: {result.get('error', 'Unknown error')}")
        
        print(f"\n📁 Results saved in: {self.output_folder}")
        print("🎉 Processing completed!")
        print("✨ Quality fixes automatically applied where needed!")
        if blank_pages:
            print("📄 Blank pages skipped to save processing time!")

def main():
    """Main function to run the document processing pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Document Processing Pipeline with Quality Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Process all files with auto workers
  python run.py --workers 4               # Use 4 parallel workers
  python run.py --mode skew_only          # Run only skew detection
  python run.py --input my_docs           # Use custom input folder
  
Quality Features:
  • Automatic blur prevention for problematic documents
  • Gray background fixes for better OCR
  • Intelligent document-specific processing
  • Parallel execution for faster processing
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full_pipeline', 'skew_only', 'crop_only', 'orient_only', 'skew_and_crop', 'crop_and_orient', 'with_dpi_standardization', 'noise_only', 'with_denoising', 'contrast_only', 'with_enhancement', 'segmentation_only', 'with_segmentation', 'color_handling_only', 'with_color_handling', 'deduplication_only', 'with_deduplication', 'language_detection_only', 'metadata_extraction_only', 'output_standardization_only', 'comprehensive_pipeline', 'production_mode'],
        default='full_pipeline',
        help='Pipeline execution mode (default: full_pipeline)'
    )
    
    parser.add_argument(
        '--workers', 
        type=int,
        default=2,  # Default to 2 workers for stability
        help='Number of parallel workers (default: 2 for stability, max: 4)'
    )
    
    parser.add_argument(
        '--input', 
        default='input',
        help='Input folder path (default: input)'
    )
    
    parser.add_argument(
        '--output', 
        default='output',
        help='Output folder path (default: output)'
    )
    
    parser.add_argument(
        '--temp', 
        default='temp',
        help='Temporary folder path (default: temp)'
    )
    
    parser.add_argument(
        '--pdf-dpi',
        type=int,
        default=250,
        help='Conversion DPI for PDF pages (200–300 recommended; default: 250)'
    )

    parser.add_argument(
        '--index-db',
        default='ingestion/index.db',
        help='Path to ingestion index database for deduplication (default: ingestion/index.db)'
    )
    
    parser.add_argument(
        '--no-comparisons',
        action='store_true',
        help='Disable automatic comparison generation (enabled by default)'
    )
    
    parser.add_argument(
        '--allow-duplicates',
        action='store_true',
        help='Allow processing duplicate files (bypass deduplication for debugging)'
    )

    args = parser.parse_args()
    
    # Validate workers argument
    if args.workers is not None:
        if args.workers < 1 or args.workers > 4:
            print(f"❌ Invalid number of workers: {args.workers}")
            print(f"   Workers must be between 1 and 4 for stability")
            return
        print(f"🔒 Using {args.workers} workers for stability")
    
    # Validate input folder
    if not os.path.exists(args.input):
        print(f"❌ Input folder '{args.input}' not found")
        print(f"   Creating input folder...")
        os.makedirs(args.input, exist_ok=True)
        print(f"   ✅ Input folder created")
        print(f"   💡 Place your documents in the '{args.input}' folder and run again")
        return
    
    # Check if input folder has files
    input_files = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    if not input_files:
        print(f"❌ No files found in input folder '{args.input}'")
        print(f"   💡 Place your documents in the '{args.input}' folder and run again")
        return
    
    print("🎯 Document Processing Pipeline")
    print("=" * 60)
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"🔄 Mode: {args.mode}")
    print(f"👥 Workers: {args.workers or 'auto-detect'}")
    print(f"📄 Files: {len(input_files)}")
    print(f"📄 PDF DPI: {args.pdf_dpi}")
    print(f"🗂️  Index DB: {args.index_db}")
    print(f"🖼️  Comparisons: {'enabled' if not args.no_comparisons else 'disabled'}")
    print(f"🐛 Allow duplicates: {'enabled' if args.allow_duplicates else 'disabled'}")
    
    # Initialize and run processor
    try:
        processor = IntelligentDocumentProcessor(
            input_folder=args.input,
            output_folder=args.output,
            temp_folder=args.temp,
            max_workers=args.workers,
            pdf_dpi=args.pdf_dpi,
            index_db_path=args.index_db,
            enable_comparisons=not args.no_comparisons,
            allow_duplicates=args.allow_duplicates,
            production_mode=(args.mode == 'production_mode')
        )
        
        success = processor.run(args.mode)
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
