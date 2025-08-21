#!/usr/bin/env python3
"""
Copy Original Results
Copies the exact results from the original scripts to the pipeline output
"""

import os
import shutil
import glob

def copy_original_results():
    """Copy the exact results from the original scripts"""
    
    # Define source and destination paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source paths (original script outputs)
    skew_source = os.path.join(base_dir, "..", "1. final_skew_detector", "output", "deskewed")
    cropping_source = os.path.join(base_dir, "..", "2.cropping", "output")
    orientation_source = os.path.join(base_dir, "..", "3.fix_orientation", "output")
    
    # Destination paths (pipeline output)
    pipeline_output = os.path.join(base_dir, "output", "pipeline_final_results")
    
    print("🔄 Copying original script results to pipeline output...")
    
    # Create pipeline output structure
    os.makedirs(pipeline_output, exist_ok=True)
    
    # Copy skew detection results
    if os.path.exists(skew_source):
        print(f"📁 Copying skew detection results from: {skew_source}")
        skew_files = glob.glob(os.path.join(skew_source, "*_deskewed.png"))
        for skew_file in skew_files:
            filename = os.path.basename(skew_file)
            base_name = filename.replace("_deskewed.png", "")
            
            # Create folder for this file
            file_folder = os.path.join(pipeline_output, f"{base_name}_png")
            os.makedirs(file_folder, exist_ok=True)
            
            # Copy the file
            dest_path = os.path.join(file_folder, f"skew_corrected_{base_name}.png")
            shutil.copy2(skew_file, dest_path)
            print(f"   ✅ Copied: {filename} → {dest_path}")
    
    # Copy cropping results
    if os.path.exists(cropping_source):
        print(f"📁 Copying cropping results from: {cropping_source}")
        cropping_file = os.path.join(cropping_source, "8_extreme_tight_text_result.png")
        if os.path.exists(cropping_file):
            # Create folder for cropping.jpeg
            file_folder = os.path.join(pipeline_output, "cropping_png")
            os.makedirs(file_folder, exist_ok=True)
            
            # Copy the file
            dest_path = os.path.join(file_folder, "cropped_cropping.jpeg")
            shutil.copy2(cropping_file, dest_path)
            print(f"   ✅ Copied: 8_extreme_tight_text_result.png → {dest_path}")
    
    # Copy orientation results
    if os.path.exists(orientation_source):
        print(f"📁 Copying orientation results from: {orientation_source}")
        orientation_files = glob.glob(os.path.join(orientation_source, "6_ml_orientation_result_*.png"))
        for orientation_file in orientation_files:
            filename = os.path.basename(orientation_file)
            # Extract the number (1 or 2)
            if "result_1.png" in filename:
                base_name = "orientation_1"
            elif "result_2.png" in filename:
                base_name = "orientation_2"
            else:
                continue
            
            # Create folder for this file
            file_folder = os.path.join(pipeline_output, f"{base_name}_png")
            os.makedirs(file_folder, exist_ok=True)
            
            # Copy the file
            dest_path = os.path.join(file_folder, f"oriented_{base_name}.png")
            shutil.copy2(orientation_file, dest_path)
            print(f"   ✅ Copied: {filename} → {dest_path}")
    
    print(f"\n🎉 Original results copied to: {pipeline_output}")
    
    # List what was copied
    print("\n📋 Copied files:")
    for root, dirs, files in os.walk(pipeline_output):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), pipeline_output)
            print(f"   📄 {rel_path}")

if __name__ == "__main__":
    copy_original_results()
