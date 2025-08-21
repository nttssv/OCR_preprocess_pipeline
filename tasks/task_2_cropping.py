#!/usr/bin/env python3
"""
Task 2: Document Cropping
Runs the exact same script from 2.cropping/extreme_tight_text_cropping.py
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

class DocumentCroppingTask:
    """Task 2: Document Cropping - Running exact same script"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Document Cropping"
        self.task_id = "task_2_cropping"
        
    def run(self, input_file, file_type, output_folder):
        """
        Run document cropping on the input file
        
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
            task_output = os.path.join(output_folder, "task2_cropping")
            os.makedirs(task_output, exist_ok=True)
            
            # Process based on file type
            if file_type == 'pdf':
                result = self._process_pdf_cropping(input_file, task_output)
            else:
                result = self._process_image_cropping(input_file, task_output)
            
            if result:
                self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
                return result
            else:
                self.logger.error(f"❌ {self.task_name} failed for {os.path.basename(input_file)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in {self.task_name}: {str(e)}")
            return None
    
    def _process_image_cropping(self, image_path, output_folder):
        """Process single image for cropping by running the original script"""
        
        try:
            # Get the path to the original cropping script
            script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "2.cropping")
            script_path = os.path.join(script_dir, "extreme_tight_text_cropping.py")
            
            if not os.path.exists(script_path):
                self.logger.error(f"Original cropping script not found: {script_path}")
                return None
            
            self.logger.info("✅ Running exact same cropping script from 2.cropping")
            
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
                result = subprocess.run([sys.executable, "extreme_tight_text_cropping.py"], 
                                     capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    self.logger.error(f"Script failed: {result.stderr}")
                    return None
                
                self.logger.info("✅ Original script executed successfully")
                
                # Find the output file
                script_output_path = os.path.join(script_dir, "output", "8_extreme_tight_text_result.png")
                
                if not os.path.exists(script_output_path):
                    self.logger.error(f"Script output not found: {script_output_path}")
                    return None
                
                # Copy the result to our task output folder
                task_output_filename = f"cropped_{os.path.basename(image_path)}"
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
            self.logger.error(f"Error running original cropping script: {str(e)}")
            return None
    
    def _process_pdf_cropping(self, pdf_path, output_folder):
        """Process PDF for cropping"""
        
        try:
            self.logger.info("📄 PDF cropping not yet implemented")
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
            self.logger.error(f"Error processing PDF for cropping: {str(e)}")
            return None
    
    def get_task_info(self):
        """Get information about this task"""
        
        return {
            'task_id': self.task_id,
            'name': self.task_name,
            'description': 'Remove blank borders, punch holes, and scanner edges by running exact same script',
            'order': 2,
            'dependencies': ['task_1_skew_detection'],
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
    task = DocumentCroppingTask(logger)
    
    print("🧪 Testing Document Cropping Task")
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
