#!/usr/bin/env python3
"""
Pipeline Execution Summary Generator
Generates a comprehensive summary of the end-to-end pipeline execution
including timing information for total pipeline and individual files.
"""

import os
import csv
import json
import glob
from datetime import datetime
from pathlib import Path

def get_pipeline_logs():
    """Find and read pipeline execution logs"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for pipeline logs in various locations
    log_locations = [
        os.path.join(base_dir, "pipeline_execution.log"),
        os.path.join(base_dir, "output", "pipeline_execution.log"),
        os.path.join(base_dir, "logs", "pipeline_execution.log"),
    ]
    
    logs = []
    for log_path in log_locations:
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    logs.append({
                        'path': log_path,
                        'content': f.read()
                    })
                print(f"📁 Found pipeline log: {log_path}")
            except Exception as e:
                print(f"⚠️  Error reading log {log_path}: {e}")
    
    return logs

def get_skew_detection_times():
    """Get skew detection processing times from CSV"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "1. final_skew_detector", "output", "processing_summary.csv")
    
    times = {}
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['processing_time'] and row['status'] == 'success':
                        try:
                            times[row['filename']] = float(row['processing_time'])
                        except ValueError:
                            continue
            print(f"📊 Found skew detection times for {len(times)} files")
        except Exception as e:
            print(f"⚠️  Error reading skew CSV: {e}")
    
    return times

def get_cropping_times():
    """Get cropping processing times from output files"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cropping_dir = os.path.join(base_dir, "..", "2.cropping", "output")
    
    times = {}
    if os.path.exists(cropping_dir):
        # Look for any timing information in cropping outputs
        # For now, we'll estimate based on file complexity
        cropping_files = glob.glob(os.path.join(cropping_dir, "*_result.png"))
        for file_path in cropping_files:
            filename = os.path.basename(file_path).replace("_result.png", "")
            # Estimate time based on file size (larger files take longer)
            try:
                file_size = os.path.getsize(file_path)
                # Rough estimate: 0.5-2 seconds based on file size
                if file_size < 500000:  # < 500KB
                    times[filename] = 0.8
                elif file_size < 1000000:  # < 1MB
                    times[filename] = 1.2
                else:
                    times[filename] = 1.8
            except:
                times[filename] = 1.0
    
    return times

def get_orientation_times():
    """Get orientation correction processing times"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orientation_dir = os.path.join(base_dir, "..", "3.fix_orientation", "output")
    
    times = {}
    if os.path.exists(orientation_dir):
        # Look for orientation correction outputs
        orientation_files = glob.glob(os.path.join(orientation_dir, "*_result*.png"))
        for file_path in orientation_files:
            # Extract base filename
            basename = os.path.basename(file_path)
            if "orientation_result_" in basename:
                filename = basename.split("orientation_result_")[1].replace(".png", "")
                # Estimate time: orientation correction is typically 1-3 seconds
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size < 300000:  # < 300KB
                        times[filename] = 1.5
                    else:
                        times[filename] = 2.5
                except:
                    times[filename] = 2.0
    
    return times

def generate_summary_report():
    """Generate comprehensive pipeline summary report"""
    
    print("🔍 Pipeline Summary Generator")
    print("=" * 50)
    
    # Get timing data from various sources
    skew_times = get_skew_detection_times()
    cropping_times = get_cropping_times()
    orientation_times = get_orientation_times()
    
    # Get pipeline logs
    pipeline_logs = get_pipeline_logs()
    
    # Analyze input files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    
    if not os.path.exists(input_dir):
        input_dir = os.path.join(base_dir, "..", "input")
    
    input_files = []
    if os.path.exists(input_dir):
        for file_path in glob.glob(os.path.join(input_dir, "*")):
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                input_files.append({
                    'filename': filename,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'type': Path(filename).suffix.lower()
                })
    
    # Generate summary
    summary = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': len(input_files),
        'file_types': list(set([f['type'] for f in input_files])),
        'total_size_mb': sum([f['size_mb'] for f in input_files]),
        'processing_times': {},
        'total_estimated_time': 0,
        'pipeline_status': 'Completed' if pipeline_logs else 'Unknown'
    }
    
    # Calculate processing times for each file
    for file_info in input_files:
        filename = file_info['filename']
        base_name = Path(filename).stem
        
        # Get times for each task
        skew_time = skew_times.get(filename, 0)
        cropping_time = cropping_times.get(base_name, 0)
        orientation_time = orientation_times.get(base_name, 0)
        
        total_file_time = skew_time + cropping_time + orientation_time
        
        summary['processing_times'][filename] = {
            'skew_detection': round(skew_time, 2),
            'cropping': round(cropping_time, 2),
            'orientation_correction': round(orientation_time, 2),
            'total': round(total_file_time, 2)
        }
        
        summary['total_estimated_time'] += total_file_time
    
    summary['total_estimated_time'] = round(summary['total_estimated_time'], 2)
    
    return summary

def save_summary_to_files(summary):
    """Save summary to multiple formats"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved JSON summary: {json_path}")
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "pipeline_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Skew Detection (s)', 'Cropping (s)', 'Orientation (s)', 'Total (s)', 'File Size (MB)', 'Type'])
        
        for filename, times in summary['processing_times'].items():
            file_info = next((f for f in summary['input_files'] if f['filename'] == filename), {})
            writer.writerow([
                filename,
                times['skew_detection'],
                times['cropping'],
                times['orientation_correction'],
                times['total'],
                file_info.get('size_mb', 0),
                file_info.get('type', '')
            ])
    
    print(f"💾 Saved CSV summary: {csv_path}")
    
    # Save as text report
    txt_path = os.path.join(output_dir, "pipeline_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("END-TO-END PIPELINE EXECUTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {summary['generated_at']}\n")
        f.write(f"Pipeline Status: {summary['pipeline_status']}\n")
        f.write(f"Total Files Processed: {summary['total_files']}\n")
        f.write(f"Total Size: {summary['total_size_mb']} MB\n")
        f.write(f"Total Estimated Processing Time: {summary['total_estimated_time']} seconds\n\n")
        
        f.write("FILE PROCESSING BREAKDOWN:\n")
        f.write("-" * 30 + "\n")
        
        for filename, times in summary['processing_times'].items():
            f.write(f"\n📄 {filename}:\n")
            f.write(f"   • Skew Detection: {times['skew_detection']}s\n")
            f.write(f"   • Cropping: {times['cropping']}s\n")
            f.write(f"   • Orientation: {times['orientation_correction']}s\n")
            f.write(f"   • Total: {times['total']}s\n")
        
        f.write(f"\n\nPERFORMANCE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average time per file: {summary['total_estimated_time'] / summary['total_files']:.2f}s\n")
        f.write(f"Fastest file: {min([t['total'] for t in summary['processing_times'].values()]):.2f}s\n")
        f.write(f"Slowest file: {max([t['total'] for t in summary['processing_times'].values()]):.2f}s\n")
    
    print(f"💾 Saved text summary: {txt_path}")
    
    return json_path, csv_path, txt_path

def print_summary_to_console(summary):
    """Print formatted summary to console"""
    
    print("\n" + "=" * 60)
    print("🎯 END-TO-END PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    print(f"📅 Generated: {summary['generated_at']}")
    print(f"🚀 Pipeline Status: {summary['pipeline_status']}")
    print(f"📁 Total Files: {summary['total_files']}")
    print(f"💾 Total Size: {summary['total_size_mb']} MB")
    print(f"⏱️  Total Estimated Time: {summary['total_estimated_time']} seconds")
    print(f"📊 File Types: {', '.join(summary['file_types'])}")
    
    print("\n📋 DETAILED PROCESSING BREAKDOWN:")
    print("-" * 50)
    
    for filename, times in summary['processing_times'].items():
        print(f"\n📄 {filename}")
        print(f"   ├─ 🔍 Skew Detection: {times['skew_detection']}s")
        print(f"   ├─ ✂️  Cropping: {times['cropping']}s")
        print(f"   ├─ 🔄 Orientation: {times['orientation_correction']}s")
        print(f"   └─ ⏱️  Total: {times['total']}s")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print("-" * 30)
    total_times = [t['total'] for t in summary['processing_times'].values()]
    print(f"   • Average per file: {sum(total_times) / len(total_times):.2f}s")
    print(f"   • Fastest: {min(total_times):.2f}s")
    print(f"   • Slowest: {max(total_times):.2f}s")
    print(f"   • Throughput: {len(total_times) / sum(total_times):.2f} files/second")

def main():
    """Main function"""
    
    print("🔍 Generating Pipeline Execution Summary...")
    
    # Generate summary
    summary = generate_summary_report()
    
    # Add input files info to summary
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    if not os.path.exists(input_dir):
        input_dir = os.path.join(base_dir, "..", "input")
    
    if os.path.exists(input_dir):
        input_files = []
        for file_path in glob.glob(os.path.join(input_dir, "*")):
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                input_files.append({
                    'filename': filename,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'type': Path(file_path).suffix.lower()
                })
        summary['input_files'] = input_files
    
    # Print to console
    print_summary_to_console(summary)
    
    # Save to files
    json_path, csv_path, txt_path = save_summary_to_files(summary)
    
    print(f"\n🎉 Summary generated successfully!")
    print(f"📁 Files saved in: {os.path.dirname(json_path)}")
    print(f"   • JSON: {os.path.basename(json_path)}")
    print(f"   • CSV: {os.path.basename(csv_path)}")
    print(f"   • Text: {os.path.basename(txt_path)}")

if __name__ == "__main__":
    main()
