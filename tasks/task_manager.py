#!/usr/bin/env python3
"""
Task Manager
Manages all individual tasks and their execution
"""

import os
import sys
import logging
from pathlib import Path

# Import individual task classes
from .task_1_skew_detection import SkewDetectionTask
from .task_2_cropping import DocumentCroppingTask
from .task_3_size_dpi_standardization import SizeDPIStandardizationTask
from .task_4_orientation_correction import OrientationCorrectionTask

class TaskManager:
    """Manages all pipeline tasks"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize all tasks
        self.tasks = {
            "task_1_skew_detection": SkewDetectionTask(logger),
            "task_2_cropping": DocumentCroppingTask(logger),
            "task_3_size_dpi_standardization": SizeDPIStandardizationTask(logger),
            "task_4_orientation_correction": OrientationCorrectionTask(logger)
        }
        
        # Task dependencies
        self.dependencies = {
            "task_1_skew_detection": [],
            "task_2_cropping": ["task_1_skew_detection"],
            "task_3_size_dpi_standardization": ["task_2_cropping"],
            "task_4_orientation_correction": ["task_3_size_dpi_standardization"]
        }
        
        # Task execution order
        self.execution_order = [
            "task_1_skew_detection",
            "task_2_cropping", 
            "task_3_size_dpi_standardization",
            "task_4_orientation_correction"
        ]
    
    def get_task(self, task_id):
        """Get a specific task by ID"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self):
        """Get all available tasks"""
        return self.tasks
    
    def get_task_info(self, task_id):
        """Get information about a specific task"""
        task = self.get_task(task_id)
        if task:
            return task.get_task_info()
        return None
    
    def get_all_task_info(self):
        """Get information about all tasks"""
        return {task_id: task.get_task_info() for task_id, task in self.tasks.items()}
    
    def get_dependencies(self, task_id):
        """Get dependencies for a specific task"""
        return self.dependencies.get(task_id, [])
    
    def get_execution_order(self):
        """Get the recommended execution order for tasks"""
        return self.execution_order.copy()
    
    def validate_task_chain(self, task_ids):
        """Validate that a sequence of tasks can be executed"""
        
        executed_tasks = set()
        
        for task_id in task_ids:
            if task_id not in self.tasks:
                return False, f"Task {task_id} not found"
            
            # Check dependencies
            deps = self.dependencies.get(task_id, [])
            for dep in deps:
                if dep not in executed_tasks:
                    return False, f"Task {task_id} depends on {dep} which hasn't been executed"
            
            executed_tasks.add(task_id)
        
        return True, "Task chain is valid"
    
    def run_task(self, task_id, input_file, file_type, output_folder):
        """Run a specific task"""
        
        task = self.get_task(task_id)
        if not task:
            self.logger.error(f"Task {task_id} not found")
            return None
        
        self.logger.info(f"🚀 Executing task: {task_id}")
        return task.run(input_file, file_type, output_folder)
    
    def run_task_chain(self, task_ids, input_file, file_type, output_folder):
        """Run a sequence of tasks in order"""
        
        # Validate the task chain
        is_valid, message = self.validate_task_chain(task_ids)
        if not is_valid:
            self.logger.error(f"Invalid task chain: {message}")
            return None
        
        results = {}
        current_input = input_file
        
        for task_id in task_ids:
            self.logger.info(f"\n🔄 Running task chain: {task_id}")
            
            # Run the task
            result = self.run_task(task_id, current_input, file_type, output_folder)
            
            if result and result.get('status') == 'completed':
                results[task_id] = result
                # Use output from this task as input for next task
                current_input = result['output']
                self.logger.info(f"✅ Task {task_id} completed, output: {os.path.basename(current_input)}")
            else:
                self.logger.error(f"❌ Task {task_id} failed")
                results[task_id] = result
                # Continue with previous input if task failed
                self.logger.warning(f"⚠️  Continuing with previous input: {os.path.basename(current_input)}")
        
        return results
    
    def run_full_pipeline(self, input_file, file_type, output_folder):
        """Run the complete pipeline (all tasks in order)"""
        
        self.logger.info("🚀 Running full pipeline")
        return self.run_task_chain(self.execution_order, input_file, file_type, output_folder)
    
    def get_task_status(self):
        """Get status of all tasks"""
        
        status = {}
        for task_id, task in self.tasks.items():
            info = task.get_task_info()
            status[task_id] = {
                'name': info['name'],
                'description': info['description'],
                'order': info['order'],
                'dependencies': info['dependencies'],
                'status': info['status'],
                'supported_inputs': info['supported_inputs']
            }
        
        return status
    
    def add_new_task(self, task_id, task_class, dependencies=None):
        """Add a new task to the manager"""
        
        if task_id in self.tasks:
            self.logger.warning(f"Task {task_id} already exists, overwriting")
        
        self.tasks[task_id] = task_class(self.logger)
        self.dependencies[task_id] = dependencies or []
        
        # Update execution order
        if task_id not in self.execution_order:
            self.execution_order.append(task_id)
        
        self.logger.info(f"✅ Added new task: {task_id}")
    
    def remove_task(self, task_id):
        """Remove a task from the manager"""
        
        if task_id in self.tasks:
            del self.tasks[task_id]
            del self.dependencies[task_id]
            
            # Remove from execution order
            if task_id in self.execution_order:
                self.execution_order.remove(task_id)
            
            self.logger.info(f"✅ Removed task: {task_id}")
        else:
            self.logger.warning(f"Task {task_id} not found")
    
    def enable_task(self, task_id, enabled=True):
        """Enable or disable a task"""
        
        task = self.get_task(task_id)
        if task:
            # This would require adding an enabled flag to task classes
            # For now, we'll just log the action
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"✅ Task {task_id} {status}")
            return True
        else:
            self.logger.warning(f"Task {task_id} not found")
            return False

# Standalone execution for testing
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test the task manager
    manager = TaskManager(logger)
    
    print("🧪 Testing Task Manager")
    print("=" * 40)
    
    # Show all tasks
    print("📋 Available Tasks:")
    for task_id, task in manager.get_all_tasks().items():
        info = task.get_task_info()
        print(f"   {info['order']}. {info['name']} ({task_id})")
        print(f"      Description: {info['description']}")
        print(f"      Dependencies: {info['dependencies']}")
        print(f"      Status: {info['status']}")
        print()
    
    # Show execution order
    print("🚀 Execution Order:")
    for i, task_id in enumerate(manager.get_execution_order(), 1):
        print(f"   {i}. {task_id}")
    
    # Show dependencies
    print("\n🔗 Dependencies:")
    for task_id, deps in manager.dependencies.items():
        if deps:
            print(f"   {task_id} depends on: {', '.join(deps)}")
        else:
            print(f"   {task_id} has no dependencies")
    
    print("\n📝 Note: This task manager is designed to integrate with the pipeline.")
    print("   For standalone testing, ensure all task modules are available.")
