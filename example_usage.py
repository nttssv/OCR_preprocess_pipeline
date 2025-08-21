#!/usr/bin/env python3
"""
Example Usage of the End-to-End Document Processing Pipeline
Demonstrates how to use individual tasks and the complete pipeline
"""

import os
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def example_1_individual_tasks():
    """Example 1: Run individual tasks"""
    
    print("\n" + "="*60)
    print("🧪 Example 1: Running Individual Tasks")
    print("="*60)
    
    try:
        from tasks.task_manager import TaskManager
        
        # Setup logging
        logger = setup_logging()
        
        # Initialize task manager
        task_manager = TaskManager(logger)
        
        # Show available tasks
        print("📋 Available Tasks:")
        for task_id, task in task_manager.get_all_tasks().items():
            info = task.get_task_info()
            print(f"   {info['order']}. {info['name']} ({task_id})")
            print(f"      Description: {info['description']}")
            print(f"      Dependencies: {info['dependencies']}")
        
        # Show execution order
        print(f"\n🚀 Execution Order: {' → '.join(task_manager.get_execution_order())}")
        
        print("\n✅ Task manager initialized successfully!")
        print("💡 You can now run individual tasks or the complete pipeline")
        
    except ImportError as e:
        print(f"❌ Error importing task manager: {e}")
        print("   Make sure all task files are in the tasks/ folder")

def example_2_task_chain():
    """Example 2: Run a specific chain of tasks"""
    
    print("\n" + "="*60)
    print("🧪 Example 2: Running Task Chains")
    print("="*60)
    
    try:
        from tasks.task_manager import TaskManager
        
        # Setup logging
        logger = setup_logging()
        
        # Initialize task manager
        task_manager = TaskManager(logger)
        
        # Example: Run only skew detection and cropping
        task_chain = ["task_1_skew_detection", "task_2_cropping"]
        
        print(f"🔗 Task Chain: {' → '.join(task_chain)}")
        
        # Validate the chain
        is_valid, message = task_manager.validate_task_chain(task_chain)
        if is_valid:
            print(f"✅ Task chain is valid: {message}")
        else:
            print(f"❌ Task chain is invalid: {message}")
        
        print("\n💡 You can create custom task chains like:")
        print("   - ['task_1_skew_detection']  # Only skew detection")
        print("   - ['task_2_cropping', 'task_3_orientation_correction']  # Skip skew detection")
        print("   - ['task_1_skew_detection', 'task_3_orientation_correction']  # Skip cropping")
        
    except ImportError as e:
        print(f"❌ Error importing task manager: {e}")

def example_3_complete_pipeline():
    """Example 3: Run the complete pipeline"""
    
    print("\n" + "="*60)
    print("🧪 Example 3: Running Complete Pipeline")
    print("="*60)
    
    try:
        from document_processing_pipeline import DocumentProcessingPipeline
        
        print("🚀 Complete Pipeline Structure:")
        print("   1. Skew Detection & Correction")
        print("   2. Document Cropping")
        print("   3. Orientation Correction")
        
        print("\n💡 To run the complete pipeline:")
        print("   python run_pipeline.py")
        print("   python run_pipeline.py --mode full_pipeline")
        
        print("\n💡 To run specific modes:")
        print("   python run_pipeline.py --mode skew_only")
        print("   python run_pipeline.py --mode crop_only")
        print("   python run_pipeline.py --mode orient_only")
        print("   python run_pipeline.py --mode skew_and_crop")
        print("   python run_pipeline.py --mode crop_and_orient")
        
    except ImportError as e:
        print(f"❌ Error importing pipeline: {e}")

def example_4_adding_new_tasks():
    """Example 4: How to add new tasks"""
    
    print("\n" + "="*60)
    print("🧪 Example 4: Adding New Tasks")
    print("="*60)
    
    print("🔌 To add new tasks (4, 5, 6, etc.):")
    print("\n1. Create a new task file:")
    print("   tasks/task_4_ocr.py")
    print("   tasks/task_5_classification.py")
    print("   tasks/task_6_quality_assessment.py")
    
    print("\n2. Follow the task template:")
    print("   class OCRTask:")
    print("       def __init__(self, logger=None):")
    print("           self.task_name = 'OCR Text Extraction'")
    print("           self.task_id = 'task_4_ocr'")
    print("       def run(self, input_file, file_type, output_folder):")
    print("           # Implementation here")
    print("           pass")
    print("       def get_task_info(self):")
    print("           return {...}")
    
    print("\n3. Update task_manager.py:")
    print("   from .task_4_ocr import OCRTask")
    print("   self.tasks['task_4_ocr'] = OCRTask(logger)")
    print("   self.dependencies['task_4_ocr'] = ['task_3_orientation_correction']")
    
    print("\n4. Update pipeline_config.py:")
    print("   'task_4_ocr': {")
    print("       'name': 'OCR Text Extraction',")
    print("       'enabled': True,")
    print("       'order': 4,")
    print("       'dependencies': ['task_3_orientation_correction']")
    print("   }")
    
    print("\n✅ New tasks will automatically integrate with the pipeline!")

def main():
    """Main function to run all examples"""
    
    print("🎯 End-to-End Document Processing Pipeline - Examples")
    print("=" * 60)
    
    # Run all examples
    example_1_individual_tasks()
    example_2_task_chain()
    example_3_complete_pipeline()
    example_4_adding_new_tasks()
    
    print("\n" + "="*60)
    print("🎉 Examples completed!")
    print("="*60)
    
    print("\n📝 Next Steps:")
    print("1. Place your documents in the input/ folder")
    print("2. Run the pipeline: python run_pipeline.py")
    print("3. Check results in the output/ folder")
    print("4. Customize tasks as needed")
    
    print("\n🔗 For more information:")
    print("- README.md: Complete documentation")
    print("- run_pipeline.py --help: Command line options")
    print("- run_pipeline.py --config: Show configuration")
    print("- run_pipeline.py --list-modes: Show execution modes")

if __name__ == "__main__":
    main()
