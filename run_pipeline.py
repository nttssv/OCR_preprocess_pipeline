#!/usr/bin/env python3
"""
Simple Pipeline Runner
Quick and easy way to run the document processing pipeline
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processing_pipeline import DocumentProcessingPipeline
from pipeline_config import get_execution_mode, get_pipeline_config

def main():
    """Main function to run the pipeline with command line options"""
    
    parser = argparse.ArgumentParser(
        description="End-to-End Document Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --mode skew_only   # Run only skew detection
  python run_pipeline.py --input my_docs    # Use custom input folder
  python run_pipeline.py --config           # Show configuration
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full_pipeline', 'skew_only', 'crop_only', 'orient_only', 'skew_and_crop', 'crop_and_orient', 'with_dpi_standardization', 'noise_only', 'with_denoising', 'contrast_only', 'with_enhancement'],
        default='full_pipeline',
        help='Pipeline execution mode (default: full_pipeline)'
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
        '--config', 
        action='store_true',
        help='Show pipeline configuration and exit'
    )
    
    parser.add_argument(
        '--list-modes', 
        action='store_true',
        help='List all available execution modes and exit'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--auto-run', 
        action='store_true',
        help='Run pipeline automatically without confirmation (default: True)'
    )
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.config:
        show_configuration()
        return
    
    # List modes if requested
    if args.list_modes:
        list_execution_modes()
        return
    
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
    
    print("🎯 End-to-End Document Processing Pipeline")
    print("=" * 60)
    print(f"📁 Input folder: {args.input}")
    print(f"📁 Output folder: {args.output}")
    print(f"🔄 Execution mode: {args.mode}")
    print(f"📄 Files to process: {len(input_files)}")
    
    # Show execution mode details
    mode_config = get_execution_mode(args.mode)
    print(f"📋 Mode description: {mode_config['description']}")
    print(f"🔧 Tasks to run: {', '.join(mode_config['tasks'])}")
    
    # Auto-run by default (no confirmation needed)
    if args.auto_run or True:  # Always auto-run
        print("\n" + "="*60)
        print("🚀 Starting pipeline execution automatically...")
        
        # Initialize and run pipeline
        try:
            pipeline = DocumentProcessingPipeline(
                input_folder=args.input,
                output_folder=args.output,
                temp_folder=args.temp
            )
            
            # Configure pipeline based on execution mode
            configure_pipeline_for_mode(pipeline, args.mode)
            
            # Run the pipeline with timing
            start_time = time.time()
            success = pipeline.run_pipeline()
            total_time = time.time() - start_time
            
            if success:
                print("\n🎉 Pipeline completed successfully!")
                print(f"📁 Check the output folder '{args.output}' for results")
                print(f"⏱️  Total execution time: {total_time:.2f} seconds")
                print(f"📊 Average time per file: {total_time/len(input_files):.2f} seconds")
            else:
                print("\n❌ Pipeline failed - check logs for details")
                sys.exit(1)
                
        except Exception as e:
            print(f"\n💥 Pipeline execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Confirm execution (legacy option)
        print("\n" + "="*60)
        response = input("🚀 Start pipeline execution? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Pipeline execution cancelled")
            return
        
        # Initialize and run pipeline
        try:
            pipeline = DocumentProcessingPipeline(
                input_folder=args.input,
                output_folder=args.output,
                temp_folder=args.temp
            )
            
            # Configure pipeline based on execution mode
            configure_pipeline_for_mode(pipeline, args.mode)
            
            # Run the pipeline
            success = pipeline.run_pipeline()
            
            if success:
                print("\n🎉 Pipeline completed successfully!")
                print(f"📁 Check the output folder '{args.output}' for results")
            else:
                print("\n❌ Pipeline failed - check logs for details")
                sys.exit(1)
                
        except Exception as e:
            print(f"\n💥 Pipeline execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def show_configuration():
    """Show the current pipeline configuration"""
    
    print("🔧 Pipeline Configuration")
    print("=" * 60)
    
    config = get_pipeline_config()
    print(f"Pipeline: {config['pipeline_name']}")
    print(f"Version: {config['version']}")
    print(f"Description: {config['description']}")
    
    print("\n📋 Available Tasks:")
    for task_id, task_config in config['tasks'].items():
        status = "✅ Enabled" if task_config['enabled'] else "❌ Disabled"
        order = task_config['order']
        print(f"   {order}. {task_config['name']} - {status}")
        print(f"      Description: {task_config['description']}")
        if task_config['dependencies']:
            print(f"      Dependencies: {', '.join(task_config['dependencies'])}")
    
    print("\n📁 Folder Structure:")
    folders = config['folders']
    for folder_name, folder_path in folders.items():
        print(f"   {folder_name}: {folder_path}")
    
    print("\n⚙️  Quality Settings:")
    quality = config['quality']
    print(f"   Output DPI: {quality['output_dpi']}")
    print(f"   Compression Quality: {quality['compression_quality']}%")
    print(f"   Preserve Metadata: {quality['preserve_metadata']}")
    
    print("\n🚀 Performance Settings:")
    performance = config['performance']
    print(f"   Memory Limit: {performance['memory_limit'] / (1024**3):.1f} GB")
    print(f"   Timeout per Task: {performance['timeout_per_task']} seconds")
    print(f"   Max Retries: {performance['max_retries']}")

def list_execution_modes():
    """List all available execution modes"""
    
    print("🚀 Available Execution Modes")
    print("=" * 60)
    
    from pipeline_config import EXECUTION_MODES
    
    for mode_id, mode_config in EXECUTION_MODES.items():
        print(f"\n📋 {mode_id.upper()}")
        print(f"   Name: {mode_config['name']}")
        print(f"   Description: {mode_config['description']}")
        print(f"   Tasks: {', '.join(mode_config['tasks'])}")

def configure_pipeline_for_mode(pipeline, mode_name):
    """Configure the pipeline for a specific execution mode"""
    
    mode_config = get_execution_mode(mode_name)
    enabled_tasks = mode_config['tasks']
    
    print(f"\n⚙️  Configuring pipeline for mode: {mode_name}")
    
    # Disable all tasks first
    for task_key in pipeline.pipeline_config.keys():
        pipeline.enable_task(pipeline.pipeline_config[task_key]['name'], False)
    
    # Enable only the tasks for this mode
    for task_id in enabled_tasks:
        if task_id in pipeline.pipeline_config:
            task_name = pipeline.pipeline_config[task_id]['name']
            pipeline.enable_task(task_name, True)
            print(f"   ✅ Enabled: {task_name}")
        else:
            print(f"   ⚠️  Task not found: {task_id}")
    
    print(f"   🎯 Pipeline configured for {len(enabled_tasks)} tasks")

if __name__ == "__main__":
    main()
