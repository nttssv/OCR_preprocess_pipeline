"""
Tasks Package for End-to-End Document Processing Pipeline

This package contains all individual task implementations that can be
executed by the pipeline. Each task is a separate module that can be
independently tested, maintained, and updated.
"""

from .task_1_orientation_correction import OrientationCorrectionTask
from .task_2_skew_detection import SkewDetectionTask
from .task_3_cropping import DocumentCroppingTask
from .task_4_size_dpi_standardization import SizeDPIStandardizationTask
from .task_5_noise_reduction import NoiseReductionTask
from .task_6_contrast_enhancement import ContrastEnhancementTask
from .task_manager import TaskManager

__all__ = [
    'OrientationCorrectionTask',
    'SkewDetectionTask',
    'DocumentCroppingTask',
    'SizeDPIStandardizationTask',
    'NoiseReductionTask',
    'ContrastEnhancementTask',
    'TaskManager'
]

# Version information
__version__ = "1.0.0"
__author__ = "Document Processing Pipeline Team"
__description__ = "Individual task implementations for document processing pipeline"

# Task registry for easy access
TASK_REGISTRY = {
    "task_1_orientation_correction": OrientationCorrectionTask,
    "task_2_skew_detection": SkewDetectionTask,
    "task_3_cropping": DocumentCroppingTask,
    "task_4_size_dpi_standardization": SizeDPIStandardizationTask,
    "task_5_noise_reduction": NoiseReductionTask,
    "task_6_contrast_enhancement": ContrastEnhancementTask
}

def get_task_class(task_id):
    """Get a task class by ID"""
    return TASK_REGISTRY.get(task_id)

def get_all_task_classes():
    """Get all available task classes"""
    return TASK_REGISTRY.copy()

def list_available_tasks():
    """List all available task IDs"""
    return list(TASK_REGISTRY.keys())

def create_task_instance(task_id, logger=None):
    """Create a task instance by ID"""
    task_class = get_task_class(task_id)
    if task_class:
        return task_class(logger)
    return None
