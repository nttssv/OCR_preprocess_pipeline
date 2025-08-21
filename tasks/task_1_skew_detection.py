#!/usr/bin/env python3
"""
Task 1: Skew Detection & Correction
Runs the exact same script from final_skew_detector/skew_detector.py
"""

import os
import sys
import cv2
import shutil
import logging
import subprocess
from pathlib import Path

# Add parent directory to path to import task modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class SkewDetectionTask:
    """Task 1: Skew Detection and Correction - Running exact same script"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Skew Detection & Correction"
        self.task_id = "task_1_skew_detection"
        
    def run(self, input_file, file_type, output_folder):
        """
        Run skew detection and correction on the input file
        
        Args:
            input_file (str): Path to input file
            file_type (str): Type of file ('image' or 'pdf')
            output_folder (str): Folder to save results
            
        Returns:
            dict: Task result with status and output path
        """
        
        try:
            self.logger.info(f"🔄 Running {self.task_name} on {os.path.basename(input_file)}")
            
            # Create task-specific output folder
            task_output = os.path.join(output_folder, "task1_skew_detection")
            os.makedirs(task_output, exist_ok=True)
            
            # Process based on file type
            if file_type == 'pdf':
                result = self._process_pdf_skew_detection(input_file, task_output)
            else:
                result = self._process_image_skew_detection(input_file, task_output)
            
            if result:
                self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
                return result
            else:
                self.logger.error(f"❌ {self.task_name} failed for {os.path.basename(input_file)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in {self.task_name}: {str(e)}")
            return None
    
    def _process_image_skew_detection(self, image_path, output_folder):
        """Process single image for skew detection by running the original script"""
        
        try:
            # Get the path to the original skew detector script
            script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "1. final_skew_detector")
            script_path = os.path.join(script_dir, "skew_detector.py")
            
            if not os.path.exists(script_path):
                self.logger.error(f"Original skew detector script not found: {script_path}")
                return None
            
            self.logger.info("✅ Running exact same skew detection script from final_skew_detector")
            
            # Create a temporary input folder for the script
            temp_input_dir = os.path.join(script_dir, "input")
            os.makedirs(temp_input_dir, exist_ok=True)
            
            # Copy the image to the script's input folder
            temp_input_path = os.path.join(temp_input_dir, os.path.basename(image_path))
            shutil.copy2(image_path, temp_input_path)
            
            # Change to the script directory and run it
            original_cwd = os.getcwd()
            os.chdir(script_dir)
            
            try:
                # Run the original script
                result = subprocess.run([sys.executable, "skew_detector.py"], 
                                     capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    self.logger.error(f"Script failed: {result.stderr}")
                    return None
                
                self.logger.info("✅ Original script executed successfully")
                
                # Find the output file
                output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_deskewed.png"
                script_output_path = os.path.join(script_dir, "output", "deskewed", output_filename)
                
                if not os.path.exists(script_output_path):
                    self.logger.error(f"Script output not found: {script_output_path}")
                    # List what's actually in the output folder
                    output_dir = os.path.join(script_dir, "output", "deskewed")
                    if os.path.exists(output_dir):
                        actual_files = os.listdir(output_dir)
                        self.logger.error(f"Files in output/deskewed: {actual_files}")
                    return None
                
                # Copy the result to our task output folder
                task_output_filename = f"skew_corrected_{os.path.basename(image_path)}"
                task_output_path = os.path.join(output_folder, task_output_filename)
                shutil.copy2(script_output_path, task_output_path)
                
                # Clean up temporary files
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                
                return {
                    'input': image_path,
                    'output': task_output_path,
                    'status': 'completed',
                    'task': self.task_id,
                    'method': 'exact_script_execution'
                }
                
            finally:
                # Always restore original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            self.logger.error(f"Error running original skew detection script: {str(e)}")
            return None
    
    def _process_pdf_skew_detection(self, pdf_path, output_folder):
        """Process PDF for skew detection"""
        
        try:
            self.logger.info("📄 PDF skew detection not yet implemented")
            self.logger.info("💡 For now, returning placeholder result")
            
            # Placeholder result for PDF processing
            return {
                'input': pdf_path,
                'output': pdf_path,  # Placeholder
                'status': 'completed',
                'task': self.task_id,
                'note': 'PDF processing placeholder - needs implementation'
            }
            
        except Exception as e:
            self.logger.error(f"Error processing PDF for skew detection: {str(e)}")
            return None
    
    def get_task_info(self):
        """Get information about this task"""
        
        return {
            'task_id': self.task_id,
            'name': self.task_name,
            'description': 'Detect and correct document skew angles by running exact same script',
            'order': 1,
            'dependencies': [],
            'output_format': 'png',
            'supported_inputs': ['image', 'pdf'],
            'status': 'ready'
        }

# Standalone execution for testing
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test the task
    task = SkewDetectionTask(logger)
    
    print("🧪 Testing Skew Detection Task")
    print("=" * 40)
    
    # Show task info
    info = task.get_task_info()
    print(f"Task ID: {info['task_id']}")
    print(f"Name: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Order: {info['order']}")
    print(f"Dependencies: {info['dependencies']}")
    print(f"Supported Inputs: {info['supported_inputs']}")
    
    print("\n📝 Note: This task file is designed to integrate with the pipeline.")
    print("   For standalone testing, ensure the required modules are available.")
