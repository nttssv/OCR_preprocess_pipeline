#!/usr/bin/env python3
"""
End-to-End Document Processing Pipeline
Runs all document processing tasks in sequence: Skew Detection → Cropping → Orientation Correction
"""

import os
import sys
import cv2
import time
import shutil
from pathlib import Path
import logging
from datetime import datetime

# Add current directory to path to import task modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import task manager and individual tasks
try:
    from tasks.task_manager import TaskManager
    from tasks import TASK_REGISTRY
    TASKS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Task modules not available: {e}")
    print("   Make sure the tasks folder is properly set up")
    TASKS_AVAILABLE = False

class DocumentProcessingPipeline:
    """End-to-End Document Processing Pipeline"""
    
    def __init__(self, input_folder="input", output_folder="output", temp_folder="temp"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.temp_folder = temp_folder
        self.pipeline_results = {}
        self.start_time = None
        
        # Create necessary folders
        self._create_folders()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize task manager
        if TASKS_AVAILABLE:
            self.task_manager = TaskManager(self.logger)
            self.logger.info("✅ Task manager initialized successfully")
        else:
            self.task_manager = None
            self.logger.warning("⚠️  Task manager not available")
        
        # Pipeline configuration
        self.pipeline_config = {
            "task_1_skew_detection": {
                "name": "Skew Detection & Correction",
                "enabled": True,
                "description": "Detect and correct document skew angles"
            },
            "task_2_cropping": {
                "name": "Document Cropping",
                "enabled": True,
                "description": "Remove blank borders, punch holes, and scanner edges"
            },
            "task_3_size_dpi_standardization": {
                "name": "Size & DPI Standardization",
                "enabled": True,
                "description": "Standardize image dimensions and improve DPI to 300 for optimal OCR processing"
            },
            "task_4_orientation_correction": {
                "name": "Orientation Correction",
                "enabled": True,
                "description": "Detect and correct upside-down or sideways pages"
            }
        }
    
    def _create_folders(self):
        """Create necessary folders for the pipeline"""
        folders = [self.input_folder, self.output_folder, self.temp_folder]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"📁 Created/verified folder: {folder}")
    
    def _setup_logging(self):
        """Setup logging for the pipeline"""
        log_file = os.path.join(self.output_folder, f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Document Processing Pipeline Started")
    
    def get_input_files(self):
        """Get list of input files to process"""
        if not os.path.exists(self.input_folder):
            self.logger.error(f"Input folder '{self.input_folder}' not found")
            return []
        
        # Supported file extensions
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        pdf_extensions = ('.pdf',)
        
        input_files = []
        
        for file in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file)
            if os.path.isfile(file_path):
                file_ext = file.lower()
                if file_ext.endswith(image_extensions):
                    input_files.append(('image', file_path))
                elif file_ext.endswith(pdf_extensions):
                    input_files.append(('pdf', file_path))
        
        self.logger.info(f"📁 Found {len(input_files)} input files")
        return input_files
    
    def run_task_1_skew_detection(self, input_file, file_type):
        """Run Task 1: Skew Detection and Correction"""
        
        if not self.task_manager:
            self.logger.warning("⚠️  Task manager not available - skipping")
            return None
        
        try:
            return self.task_manager.run_task("task_1_skew_detection", input_file, file_type, self.temp_folder)
        except Exception as e:
            self.logger.error(f"❌ Error in Task 1: {str(e)}")
            return None
    
    def run_task_2_cropping(self, input_file, file_type):
        """Run Task 2: Document Cropping"""
        
        if not self.task_manager:
            self.logger.warning("⚠️  Task manager not available - skipping")
            return None
        
        try:
            return self.task_manager.run_task("task_2_cropping", input_file, file_type, self.temp_folder)
        except Exception as e:
            self.logger.error(f"❌ Error in Task 2: {str(e)}")
            return None
    
    def run_task_3_size_dpi_standardization(self, input_file, file_type):
        """Run Task 3: Size & DPI Standardization"""
        
        if not self.task_manager:
            self.logger.warning("⚠️  Task manager not available - skipping")
            return None
        
        try:
            return self.task_manager.run_task("task_3_size_dpi_standardization", input_file, file_type, self.temp_folder)
        except Exception as e:
            self.logger.error(f"❌ Error in Task 3: {str(e)}")
            return None
    
    def run_task_4_orientation_correction(self, input_file, file_type):
        """Run Task 4: Orientation Correction"""
        
        if not self.task_manager:
            self.logger.warning("⚠️  Task manager not available - skipping")
            return None
        
        try:
            return self.task_manager.run_task("task_4_orientation_correction", input_file, file_type, self.temp_folder)
        except Exception as e:
            self.logger.error(f"❌ Error in Task 4: {str(e)}")
            return None
    
    def run_pipeline(self):
        """Run the complete end-to-end pipeline"""
        
        self.start_time = time.time()
        self.logger.info("🚀 Starting End-to-End Document Processing Pipeline")
        
        # Get input files
        input_files = self.get_input_files()
        if not input_files:
            self.logger.error("❌ No input files found - pipeline cannot continue")
            return False
        
        # Process each file through the pipeline
        for file_type, file_path in input_files:
            filename = os.path.basename(file_path)
            file_start_time = time.time()
            
            self.logger.info(f"\n📄 Processing file: {filename} ({file_type.upper()})")
            self.logger.info(f"⏱️  File processing started at: {datetime.now().strftime('%H:%M:%S')}")
            
            file_results = {
                'input_file': file_path,
                'file_type': file_type,
                'tasks': {},
                'timing': {}
            }
            
            # Task 1: Skew Detection
            if self.pipeline_config["task_1_skew_detection"]["enabled"]:
                task1_start = time.time()
                task1_result = self.run_task_1_skew_detection(file_path, file_type)
                task1_time = time.time() - task1_start
                
                file_results['tasks']['skew_detection'] = task1_result
                file_results['timing']['skew_detection'] = task1_time
                
                if task1_result:
                    # Use output from task 1 as input for task 2
                    next_input = task1_result['output']
                    # If Task 1 processed a PDF, update file_type to 'image' for subsequent tasks
                    if file_type == 'pdf' and task1_result.get('note', '').startswith('PDF converted'):
                        file_type = 'image'
                        self.logger.info(f"🔄 Updated file_type from 'pdf' to 'image' for subsequent tasks")
                    self.logger.info(f"✅ Task 1 completed in {task1_time:.2f}s")
                else:
                    # If task 1 failed, use original input
                    next_input = file_path
                    self.logger.warning(f"⚠️  Task 1 failed after {task1_time:.2f}s, using original input")
            else:
                next_input = file_path
            
            # Task 2: Cropping
            if self.pipeline_config["task_2_cropping"]["enabled"]:
                task2_start = time.time()
                task2_result = self.run_task_2_cropping(next_input, file_type)
                task2_time = time.time() - task2_start
                
                file_results['tasks']['cropping'] = task2_result
                file_results['timing']['cropping'] = task2_time
                
                if task2_result:
                    # Use output from task 2 as input for task 3
                    next_input = task2_result['output']
                    self.logger.info(f"✅ Task 2 completed in {task2_time:.2f}s")
                else:
                    self.logger.warning(f"⚠️  Task 2 failed after {task2_time:.2f}s, continuing with previous input")
                # If task 2 failed, continue with previous input
            
            # Task 3: Size & DPI Standardization
            if self.pipeline_config["task_3_size_dpi_standardization"]["enabled"]:
                task3_start = time.time()
                task3_result = self.run_task_3_size_dpi_standardization(next_input, file_type)
                task3_time = time.time() - task3_start
                
                file_results['tasks']['size_dpi_standardization'] = task3_result
                file_results['timing']['size_dpi_standardization'] = task3_time
                
                if task3_result:
                    # Use output from task 3 as input for task 4
                    next_input = task3_result['output']
                    self.logger.info(f"✅ Task 3 completed in {task3_time:.2f}s")
                else:
                    self.logger.warning(f"⚠️  Task 3 failed after {task3_time:.2f}s")
            
            # Task 4: Orientation Correction
            if self.pipeline_config["task_4_orientation_correction"]["enabled"]:
                task4_start = time.time()
                task4_result = self.run_task_4_orientation_correction(next_input, file_type)
                task4_time = time.time() - task4_start
                
                file_results['tasks']['orientation_correction'] = task4_result
                file_results['timing']['orientation_correction'] = task4_time
                
                if task4_result:
                    self.logger.info(f"✅ Task 4 completed in {task4_time:.2f}s")
                else:
                    self.logger.warning(f"⚠️  Task 4 failed after {task4_time:.2f}s")
            
            # Calculate total file processing time
            file_total_time = time.time() - file_start_time
            file_results['timing']['total'] = file_total_time
            
            self.logger.info(f"⏱️  File {filename} completed in {file_total_time:.2f}s")
            
            # Store results for this file
            self.pipeline_results[filename] = file_results
        
        # Generate final output
        self._generate_final_output()
        
        # Calculate and log pipeline statistics
        self._log_pipeline_statistics()
        
        return True
    
    def run_task_chain(self, task_ids, input_file, file_type):
        """Run a specific chain of tasks"""
        
        if not self.task_manager:
            self.logger.error("❌ Task manager not available")
            return None
        
        try:
            return self.task_manager.run_task_chain(task_ids, input_file, file_type, self.temp_folder)
        except Exception as e:
            self.logger.error(f"❌ Error running task chain: {str(e)}")
            return None
    
    def _generate_final_output(self):
        """Generate final output and organize results"""
        
        self.logger.info("📊 Generating final output...")
        
        # Create final output folder
        final_output = os.path.join(self.output_folder, "pipeline_final_results")
        os.makedirs(final_output, exist_ok=True)
        
        # Process each file to create final result and comparison
        for filename, results in self.pipeline_results.items():
            base_name = os.path.splitext(filename)[0]
            
            # Get the final processed image (from the last successful task)
            final_processed_image = None
            final_comparison_image = None
            
            # Find the final processed image from the last successful task
            if results['tasks'].get('orientation_correction') and results['tasks']['orientation_correction'].get('output'):
                final_processed_image = results['tasks']['orientation_correction']['output']
                final_comparison_image = results['tasks']['orientation_correction'].get('comparison')
            elif results['tasks'].get('size_dpi_standardization') and results['tasks']['size_dpi_standardization'].get('output'):
                final_processed_image = results['tasks']['size_dpi_standardization']['output']
                final_comparison_image = results['tasks']['size_dpi_standardization'].get('comparison')
            elif results['tasks'].get('cropping') and results['tasks']['cropping'].get('output'):
                final_processed_image = results['tasks']['cropping']['output']
                final_comparison_image = results['tasks']['cropping'].get('comparison')
            elif results['tasks'].get('skew_detection') and results['tasks']['skew_detection'].get('output'):
                final_processed_image = results['tasks']['skew_detection']['output']
                final_comparison_image = results['tasks']['skew_detection'].get('comparison')
            
            if final_processed_image and os.path.exists(final_processed_image):
                # Create the final result file: {no}_xxx_result.png
                final_result_path = os.path.join(final_output, f"{base_name}_result.png")
                shutil.copy2(final_processed_image, final_result_path)
                self.logger.info(f"✅ Created final result: {final_result_path}")
            
            # Create a comprehensive final comparison that includes skewness information
            if final_processed_image and os.path.exists(final_processed_image):
                # Get skewness information from task 1
                skew_angle = None
                if results['tasks'].get('skew_detection') and results['tasks']['skew_detection'].get('angle'):
                    skew_angle = results['tasks']['skew_detection']['angle']
                
                # Create comprehensive comparison
                comprehensive_comparison = self._create_comprehensive_comparison(
                    results['input_file'], 
                    final_processed_image, 
                    base_name,
                    skew_angle,
                    results['tasks']
                )
                
                if comprehensive_comparison is not None:
                    final_comparison_path = os.path.join(final_output, f"{base_name}_comparison.png")
                    cv2.imwrite(final_comparison_path, comprehensive_comparison)
                    self.logger.info(f"✅ Created comprehensive comparison: {final_comparison_path}")
            
            # Also create the detailed folder structure for debugging
            file_output_dir = os.path.join(final_output, f"{base_name}_detailed")
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Copy original input
            shutil.copy2(results['input_file'], file_output_dir)
            
            # Copy results from each task
            for task_name, task_result in results['tasks'].items():
                if task_result and 'output' in task_result:
                    task_output_file = os.path.join(file_output_dir, f"{task_name}_{os.path.basename(task_result['output'])}")
                    shutil.copy2(task_result['output'], task_output_file)
        
        self.logger.info(f"✅ Final output generated in: {final_output}")
    
    def _create_comprehensive_comparison(self, original_path, processed_path, base_name, skew_angle, task_results):
        """Create a comprehensive comparison image with skewness and processing information"""
        
        try:
            import cv2
            import numpy as np
            
            # Load images
            original = cv2.imread(original_path)
            processed = cv2.imread(processed_path)
            
            if original is None or processed is None:
                self.logger.warning(f"⚠️  Could not load images for comprehensive comparison")
                return None
            
            # Ensure both images have the same height for comparison
            height, width = original.shape[:2]
            if processed.shape[0] != height:
                scale = height / processed.shape[0]
                new_w = int(processed.shape[1] * scale)
                processed_resized = cv2.resize(processed, (new_w, height), interpolation=cv2.INTER_AREA)
            else:
                processed_resized = processed
            
            # Create side-by-side comparison
            comparison = np.hstack([original, processed_resized])
            
            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (0, 255, 0)  # Green text
            
            # Original image label
            cv2.putText(comparison, "Original", (50, 50), font, font_scale, color, thickness)
            
            # Processed image label
            cv2.putText(comparison, "Final Processed", (width + 50, 50), font, font_scale, color, thickness)
            
            # Add skewness information with better visibility
            if skew_angle is not None:
                skew_text = f"Detected Skew: {skew_angle:+.2f}° | Applied Correction: {-skew_angle:+.2f}°"
                # Add black outline for better visibility
                cv2.putText(comparison, skew_text, (50, height - 60), font, 0.7, (0, 0, 0), 3)  # Black outline
                cv2.putText(comparison, skew_text, (50, height - 60), font, 0.7, (255, 255, 0), 2)  # Yellow text
            
            # Add processing summary with better visibility
            summary_text = f"Processing: Skew Detection → Cropping → Orientation Correction"
            # Add black outline for better visibility
            cv2.putText(comparison, summary_text, (50, height - 30), font, 0.6, (0, 0, 0), 3)  # Black outline
            cv2.putText(comparison, summary_text, (50, height - 30), font, 0.6, (0, 255, 255), 1)  # Cyan text
            
            # Add file information with better visibility
            file_text = f"File: {base_name} | Pipeline: End-to-End Document Processing"
            # Add black outline for better visibility
            cv2.putText(comparison, file_text, (50, height - 10), font, 0.6, (0, 0, 0), 3)  # Black outline
            cv2.putText(comparison, file_text, (50, height - 10), font, 0.6, (255, 0, 255), 1)  # Magenta text
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"❌ Error creating comprehensive comparison: {str(e)}")
            return None
    
    def _log_pipeline_statistics(self):
        """Log pipeline statistics and summary"""
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*60)
        
        # Count successful tasks and collect timing data
        total_files = len(self.pipeline_results)
        successful_tasks = 0
        failed_tasks = 0
        total_task_times = {
            'skew_detection': 0,
            'cropping': 0,
            'orientation_correction': 0
        }
        
        for filename, results in self.pipeline_results.items():
            self.logger.info(f"\n📄 File: {filename}")
            
            # Log timing information
            if 'timing' in results:
                timing = results['timing']
                self.logger.info(f"   ⏱️  Total processing time: {timing.get('total', 0):.2f}s")
                
                for task_name, task_result in results['tasks'].items():
                    task_time = timing.get(task_name, 0)
                    if task_result and task_result.get('status') == 'completed':
                        self.logger.info(f"   ✅ {task_name}: SUCCESS ({task_time:.2f}s)")
                        successful_tasks += 1
                        total_task_times[task_name] += task_time
                    else:
                        self.logger.info(f"   ❌ {task_name}: FAILED ({task_time:.2f}s)")
                        failed_tasks += 1
            else:
                # Fallback for older results without timing
                for task_name, task_result in results['tasks'].items():
                    if task_result and task_result.get('status') == 'completed':
                        self.logger.info(f"   ✅ {task_name}: SUCCESS")
                        successful_tasks += 1
                    else:
                        self.logger.info(f"   ❌ {task_name}: FAILED")
                        failed_tasks += 1
        
        self.logger.info(f"\n📈 STATISTICS:")
        self.logger.info(f"   Total files processed: {total_files}")
        self.logger.info(f"   Successful tasks: {successful_tasks}")
        self.logger.info(f"   Failed tasks: {failed_tasks}")
        self.logger.info(f"   Total execution time: {total_time:.2f} seconds")
        self.logger.info(f"   Average time per file: {total_time/total_files:.2f} seconds")
        
        # Show task-specific timing statistics
        if total_files > 0:
            self.logger.info(f"\n⏱️  TASK TIMING BREAKDOWN:")
            for task_name, total_time in total_task_times.items():
                if successful_tasks > 0:
                    avg_time = total_time / total_files
                    self.logger.info(f"   {task_name}: {avg_time:.2f}s average per file")
        
        self.logger.info("\n🎉 Pipeline execution completed!")
    
    def add_new_task(self, task_id, task_name, task_description, task_function):
        """Add a new task to the pipeline (for future extensibility)"""
        
        self.pipeline_config[f"task_{task_id}_{task_name.lower().replace(' ', '_')}"] = {
            "name": task_name,
            "enabled": True,
            "description": task_description,
            "function": task_function
        }
        
        self.logger.info(f"✅ Added new task: {task_name} (ID: {task_id})")
    
    def enable_task(self, task_name, enabled=True):
        """Enable or disable a specific task"""
        
        for task_key, task_config in self.pipeline_config.items():
            if task_config['name'] == task_name:
                task_config['enabled'] = enabled
                status = "enabled" if enabled else "disabled"
                self.logger.info(f"✅ Task '{task_name}' {status}")
                return True
        
        self.logger.warning(f"⚠️  Task '{task_name}' not found")
        return False
    
    def get_task_manager(self):
        """Get the task manager instance"""
        return self.task_manager
    
    def get_available_tasks(self):
        """Get information about all available tasks"""
        if self.task_manager:
            return self.task_manager.get_all_task_info()
        return {}

def main():
    """Main function to run the pipeline"""
    
    print("🎯 End-to-End Document Processing Pipeline")
    print("=" * 60)
    
    # Check if task modules are available
    if not TASKS_AVAILABLE:
        print("❌ Task modules are not available")
        print("   Make sure the tasks folder is properly set up:")
        print("   - tasks/task_1_skew_detection.py")
        print("   - tasks/task_2_cropping.py")
        print("   - tasks/task_3_orientation_correction.py")
        print("   - tasks/task_manager.py")
        return False
    
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline()
    
    # Run the pipeline
    success = pipeline.run_pipeline()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Check the output folder for results")
    else:
        print("\n❌ Pipeline failed - check logs for details")
    
    return success

if __name__ == "__main__":
    main()
