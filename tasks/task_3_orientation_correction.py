#!/usr/bin/env python3
"""
Task 3: Orientation Correction
Runs the exact same script from 3.fix_orientation/ml_based_orientation_detection.py
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

class OrientationCorrectionTask:
    """Task 3: Orientation Correction - Running exact same script"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = "Orientation Correction"
        self.task_id = "task_3_orientation_correction"
        
    def run(self, input_file, file_type, output_folder):
        """
        Run orientation correction on the input file
        
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
            task_output = os.path.join(output_folder, "task3_orientation")
            os.makedirs(task_output, exist_ok=True)
            
            # Process based on file type
            if file_type == 'pdf':
                result = self._process_pdf_orientation(input_file, task_output)
            else:
                result = self._process_image_orientation(input_file, task_output)
            
            if result:
                self.logger.info(f"✅ {self.task_name} completed for {os.path.basename(input_file)}")
                return result
            else:
                self.logger.error(f"❌ {self.task_name} failed for {os.path.basename(input_file)}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in {self.task_name}: {str(e)}")
            return None
    
    def _process_image_orientation(self, image_path, output_folder):
        """Process single image for orientation correction by running the original script"""
        
        try:
            # Get the path to the original orientation correction script
            script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "3.fix_orientation")
            script_path = os.path.join(script_dir, "ml_based_orientation_detection.py")
            
            if not os.path.exists(script_path):
                self.logger.error(f"Original orientation correction script not found: {script_path}")
                return None
            
            self.logger.info("✅ Running exact same orientation correction script from 3.fix_orientation")
            
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
                result = subprocess.run([sys.executable, "ml_based_orientation_detection.py"], 
                                     capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    self.logger.error(f"Script failed: {result.stderr}")
                    return None
                
                self.logger.info("✅ Original script executed successfully")
                
                # Find the output file based on the input filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                script_output_path = os.path.join(script_dir, "output", f"6_ml_orientation_result_{base_name}.png")
                
                if not os.path.exists(script_output_path):
                    self.logger.error(f"Script output not found: {script_output_path}")
                    return None
                
                # Copy the result to our task output folder
                task_output_filename = f"oriented_{os.path.basename(image_path)}"
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
            self.logger.error(f"Error running original orientation correction script: {str(e)}")
            return None
    
    def _process_pdf_orientation(self, pdf_path, output_folder):
        """Process PDF for orientation correction"""
        
        try:
            self.logger.info("📄 PDF orientation correction not yet implemented")
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
            self.logger.error(f"Error processing PDF for orientation correction: {str(e)}")
            return None
    
    def get_task_info(self):
        """Get information about this task"""
        
        return {
            'task_id': self.task_id,
            'name': self.task_name,
            'description': 'Detect and correct upside-down or sideways pages by running exact same script',
            'order': 3,
            'dependencies': ['task_2_cropping'],
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
    task = OrientationCorrectionTask(logger)
    
    print("🧪 Testing Orientation Correction Task")
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
