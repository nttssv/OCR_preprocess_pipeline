#!/usr/bin/env python3
"""
Example Usage Scripts for Document Processing API
Demonstrates how to use the API for various document processing tasks
"""

import os
import sys
import time
import asyncio
import json
import requests
from typing import Optional, Dict, Any
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 300  # 5 minutes


class DocumentProcessingClient:
    """Client for interacting with the Document Processing API"""
    
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Test connection
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print(f"✅ Connected to API at {self.base_url}")
            else:
                print(f"⚠️ API connection warning: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to API at {self.base_url}: {str(e)}")
            print("   Make sure the API server is running with: python api_main.py")
    
    def get_available_transformations(self) -> Dict[str, Any]:
        """Get available transformation types"""
        try:
            response = self.session.get(f"{self.base_url}/documents/transformations")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting transformations: {str(e)}")
            return {}
    
    def upload_file(
        self, 
        file_path: str, 
        transformation_type: str = "deskewing"
    ) -> Optional[str]:
        """
        Upload a file for processing
        
        Args:
            file_path: Path to the file to upload
            transformation_type: Type of transformation to apply
            
        Returns:
            Document ID if successful, None otherwise
        """
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        try:
            filename = os.path.basename(file_path)
            print(f"📤 Uploading {filename} for {transformation_type} processing...")
            
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, self._get_mime_type(file_path))}
                data = {'transformations': transformation_type}
                
                response = self.session.post(
                    f"{self.base_url}/documents/transform",
                    files=files,
                    data=data,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                document_id = result['document_id']
                
                print(f"✅ Upload successful!")
                print(f"   Document ID: {document_id}")
                print(f"   Status: {result['status']}")
                print(f"   Estimated time: {result['estimated_time']}")
                
                return document_id
                
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            return None
    
    def upload_from_url(
        self, 
        url: str, 
        transformation_type: str = "deskewing",
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload a file from URL for processing
        
        Args:
            url: URL to download the file from
            transformation_type: Type of transformation to apply
            filename: Optional custom filename
            
        Returns:
            Document ID if successful, None otherwise
        """
        
        try:
            print(f"🌐 Uploading from URL: {url}")
            
            data = {
                'url': url,
                'transformations': transformation_type
            }
            
            if filename:
                data['filename'] = filename
            
            response = self.session.post(
                f"{self.base_url}/documents/transform",
                data=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            document_id = result['document_id']
            
            print(f"✅ URL upload successful!")
            print(f"   Document ID: {document_id}")
            print(f"   Status: {result['status']}")
            
            return document_id
            
        except Exception as e:
            print(f"❌ URL upload failed: {str(e)}")
            return None
    
    def wait_for_completion(
        self, 
        document_id: str, 
        check_interval: int = 2,
        max_wait_time: int = 300
    ) -> Dict[str, Any]:
        """
        Wait for document processing to complete
        
        Args:
            document_id: Document ID to monitor
            check_interval: Seconds between status checks
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            Final status information
        """
        
        print(f"⏳ Waiting for processing to complete...")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                status = self.get_status(document_id)
                
                if status['status'] == 'completed':
                    processing_time = status.get('processing_time', 0)
                    print(f"✅ Processing completed in {processing_time:.1f}s!")
                    return status
                elif status['status'] == 'failed':
                    print(f"❌ Processing failed: {status.get('error_message', 'Unknown error')}")
                    return status
                elif status['status'] == 'in_progress':
                    progress = status.get('progress', {})
                    if progress:
                        print(f"🔄 Progress: {progress.get('percentage', 0):.1f}% "
                              f"({progress.get('completed_tasks', 0)}/{progress.get('total_tasks', 0)} tasks)")
                    else:
                        print(f"🔄 Status: {status['status']}")
                else:
                    print(f"⏳ Status: {status['status']}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"❌ Error checking status: {str(e)}")
                time.sleep(check_interval)
        
        print(f"⏰ Timeout waiting for completion after {max_wait_time}s")
        return self.get_status(document_id)
    
    def get_status(self, document_id: str) -> Dict[str, Any]:
        """Get document processing status"""
        try:
            response = self.session.get(f"{self.base_url}/documents/{document_id}/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting status: {str(e)}")
            return {}
    
    def download_result(self, document_id: str, output_path: str) -> bool:
        """
        Download processed document result
        
        Args:
            document_id: Document ID
            output_path: Path to save the result
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            print(f"📥 Downloading result to {output_path}...")
            
            response = self.session.get(
                f"{self.base_url}/documents/{document_id}/result",
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(output_path)
            print(f"✅ Download complete: {file_size / 1024:.1f}KB saved to {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            return False
    
    def get_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get document processing metadata"""
        try:
            response = self.session.get(f"{self.base_url}/documents/{document_id}/metadata")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting metadata: {str(e)}")
            return {}
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its results"""
        try:
            response = self.session.delete(f"{self.base_url}/documents/{document_id}")
            response.raise_for_status()
            print(f"🗑️ Document {document_id} deleted successfully")
            return True
        except Exception as e:
            print(f"❌ Error deleting document: {str(e)}")
            return False
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type for file"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'


def example_1_basic_file_upload():
    """Example 1: Basic file upload and processing"""
    
    print("\n" + "="*60)
    print("📄 Example 1: Basic File Upload")
    print("="*60)
    
    client = DocumentProcessingClient()
    
    # Check if we have test files
    test_file = "input/sample.pdf"  # Adjust path as needed
    
    if not os.path.exists(test_file):
        print(f"⚠️ Test file not found: {test_file}")
        print("   Place a PDF or image file in the 'input' folder to test")
        return
    
    # Upload file
    document_id = client.upload_file(test_file, "deskewing")
    
    if document_id:
        # Wait for completion
        final_status = client.wait_for_completion(document_id)
        
        if final_status.get('status') == 'completed':
            # Download result
            output_file = f"example_output/result_{document_id}.png"
            success = client.download_result(document_id, output_file)
            
            if success:
                print(f"🎉 Processing complete! Result saved to: {output_file}")


def example_2_multiple_transformation_types():
    """Example 2: Test different transformation types"""
    
    print("\n" + "="*60)
    print("🔧 Example 2: Multiple Transformation Types")
    print("="*60)
    
    client = DocumentProcessingClient()
    
    # Get available transformations
    transformations = client.get_available_transformations()
    
    if not transformations:
        print("❌ Could not get transformation types")
        return
    
    print("📋 Available transformations:")
    for transform in transformations.get('transformations', []):
        print(f"   • {transform['id']}: {transform['name']}")
        print(f"     {transform['description']}")
        print(f"     Expected time: {transform['expected_time']}")
        print()
    
    # Test with a sample file
    test_file = "input/sample.pdf"
    if os.path.exists(test_file):
        transformation_types = ["basic", "deskewing", "enhanced"]
        
        for transform_type in transformation_types:
            print(f"\n🔄 Testing {transform_type} transformation...")
            
            document_id = client.upload_file(test_file, transform_type)
            
            if document_id:
                status = client.wait_for_completion(document_id, max_wait_time=120)
                
                if status.get('status') == 'completed':
                    output_file = f"example_output/{transform_type}_result.png"
                    client.download_result(document_id, output_file)
                    print(f"✅ {transform_type} result saved to: {output_file}")


def example_3_url_upload():
    """Example 3: Upload from URL"""
    
    print("\n" + "="*60)
    print("🌐 Example 3: URL Upload")
    print("="*60)
    
    client = DocumentProcessingClient()
    
    # Example with a publicly accessible PDF (replace with actual URL)
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    print(f"📡 Attempting to process document from URL...")
    print(f"   URL: {test_url}")
    
    document_id = client.upload_from_url(test_url, "basic", "url_test.pdf")
    
    if document_id:
        final_status = client.wait_for_completion(document_id)
        
        if final_status.get('status') == 'completed':
            output_file = f"example_output/url_result.png"
            client.download_result(document_id, output_file)
            print(f"🎉 URL processing complete! Result: {output_file}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple files"""
    
    print("\n" + "="*60)
    print("📦 Example 4: Batch Processing")
    print("="*60)
    
    client = DocumentProcessingClient()
    
    # Find files in input directory
    input_dir = "input"
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Get all supported files
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    input_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            input_files.append(os.path.join(input_dir, file))
    
    if not input_files:
        print(f"⚠️ No supported files found in {input_dir}")
        return
    
    print(f"📄 Found {len(input_files)} files to process:")
    for file in input_files:
        print(f"   • {os.path.basename(file)}")
    
    # Process files
    document_ids = []
    
    for file_path in input_files:
        print(f"\n🔄 Processing {os.path.basename(file_path)}...")
        document_id = client.upload_file(file_path, "deskewing")
        
        if document_id:
            document_ids.append((document_id, os.path.basename(file_path)))
    
    # Wait for all to complete
    print(f"\n⏳ Waiting for {len(document_ids)} documents to complete...")
    
    for document_id, filename in document_ids:
        print(f"\n📄 Checking {filename}...")
        status = client.wait_for_completion(document_id, max_wait_time=180)
        
        if status.get('status') == 'completed':
            output_file = f"example_output/batch_{filename}.png"
            client.download_result(document_id, output_file)
            print(f"✅ {filename} complete: {output_file}")


def example_5_monitoring_and_metadata():
    """Example 5: Detailed monitoring and metadata"""
    
    print("\n" + "="*60)
    print("📊 Example 5: Monitoring and Metadata")
    print("="*60)
    
    client = DocumentProcessingClient()
    
    test_file = "input/sample.pdf"
    if not os.path.exists(test_file):
        print(f"⚠️ Test file not found: {test_file}")
        return
    
    # Upload with detailed monitoring
    document_id = client.upload_file(test_file, "enhanced")
    
    if document_id:
        print(f"\n📊 Monitoring document {document_id}...")
        
        # Monitor with detailed progress
        start_time = time.time()
        
        while True:
            status = client.get_status(document_id)
            current_time = time.time()
            elapsed = current_time - start_time
            
            print(f"\n⏱️ Elapsed: {elapsed:.1f}s")
            print(f"📋 Status: {status.get('status', 'unknown')}")
            
            if 'progress' in status:
                progress = status['progress']
                print(f"📈 Progress: {progress.get('percentage', 0):.1f}%")
                print(f"📝 Tasks: {progress.get('completed_tasks', 0)}/{progress.get('total_tasks', 0)}")
                if progress.get('current_task'):
                    print(f"🔄 Current: {progress['current_task']}")
            
            if status.get('status') in ['completed', 'failed']:
                break
            
            time.sleep(3)
        
        # Get detailed metadata
        if status.get('status') == 'completed':
            print(f"\n📋 Getting detailed metadata...")
            metadata = client.get_metadata(document_id)
            
            if metadata:
                print(f"\n📊 Processing Summary:")
                doc_info = metadata.get('document_info', {})
                proc_info = metadata.get('processing_info', {})
                
                print(f"   📄 File: {doc_info.get('original_filename', 'unknown')}")
                print(f"   📏 Size: {doc_info.get('file_size', 0) / 1024:.1f} KB")
                print(f"   🔧 Transform: {proc_info.get('transformation_type', 'unknown')}")
                print(f"   ⏱️ Time: {proc_info.get('processing_time', 0):.1f}s")
                print(f"   ✅ Tasks: {len(proc_info.get('tasks_completed', []))}")
                
                # Save metadata to file
                metadata_file = f"example_output/metadata_{document_id}.json"
                os.makedirs("example_output", exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                print(f"   💾 Metadata saved: {metadata_file}")


def main():
    """Run all examples"""
    
    print("🚀 Document Processing API - Usage Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use the Document Processing API")
    print()
    
    # Create output directory
    os.makedirs("example_output", exist_ok=True)
    
    try:
        # Run examples
        example_1_basic_file_upload()
        example_2_multiple_transformation_types()
        example_3_url_upload()
        example_4_batch_processing()
        example_5_monitoring_and_metadata()
        
        print("\n" + "="*60)
        print("🎉 All examples completed!")
        print("📁 Check the 'example_output' folder for results")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")


if __name__ == "__main__":
    main()