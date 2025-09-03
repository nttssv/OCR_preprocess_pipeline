#!/usr/bin/env python3
"""
Simple API Client Script
Quick and easy command-line interface for the Document Processing API
"""

import os
import sys
import argparse
import time
import requests
from pathlib import Path


def main():
    """Main command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="Document Processing API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python api_client.py upload document.pdf
  python api_client.py upload --transform enhanced document.pdf
  python api_client.py status abc123-document-id
  python api_client.py download abc123-document-id result.png
  python api_client.py url https://example.com/doc.pdf
        """
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload a file for processing')
    upload_parser.add_argument('file_path', help='Path to the file to upload')
    upload_parser.add_argument(
        '--transform', 
        choices=['basic', 'deskewing', 'enhanced', 'comprehensive'],
        default='deskewing',
        help='Transformation type (default: deskewing)'
    )
    upload_parser.add_argument('--wait', action='store_true', help='Wait for processing to complete')
    upload_parser.add_argument('--output', help='Output file path (if --wait is used)')
    
    # URL upload command
    url_parser = subparsers.add_parser('url', help='Upload from URL')
    url_parser.add_argument('url', help='URL to download and process')
    url_parser.add_argument(
        '--transform',
        choices=['basic', 'deskewing', 'enhanced', 'comprehensive'],
        default='deskewing',
        help='Transformation type (default: deskewing)'
    )
    url_parser.add_argument('--filename', help='Custom filename')
    url_parser.add_argument('--wait', action='store_true', help='Wait for processing to complete')
    url_parser.add_argument('--output', help='Output file path (if --wait is used)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check document status')
    status_parser.add_argument('document_id', help='Document ID to check')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download processed result')
    download_parser.add_argument('document_id', help='Document ID')
    download_parser.add_argument('output_path', help='Output file path')
    
    # List transformations command
    subparsers.add_parser('transformations', help='List available transformation types')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a document')
    delete_parser.add_argument('document_id', help='Document ID to delete')
    
    # Health check command
    subparsers.add_parser('health', help='Check API health')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize API client
    api_url = args.api_url.rstrip('/')
    
    try:
        # Execute command
        if args.command == 'upload':
            cmd_upload(api_url, args)
        elif args.command == 'url':
            cmd_url_upload(api_url, args)
        elif args.command == 'status':
            cmd_status(api_url, args)
        elif args.command == 'download':
            cmd_download(api_url, args)
        elif args.command == 'transformations':
            cmd_transformations(api_url)
        elif args.command == 'delete':
            cmd_delete(api_url, args)
        elif args.command == 'health':
            cmd_health(api_url)
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def cmd_upload(api_url: str, args):
    """Handle file upload command"""
    
    if not os.path.exists(args.file_path):
        print(f"❌ File not found: {args.file_path}")
        return
    
    print(f"📤 Uploading {args.file_path} with {args.transform} transformation...")
    
    try:
        with open(args.file_path, 'rb') as f:
            files = {'file': (os.path.basename(args.file_path), f)}
            data = {'transformations': args.transform}
            
            response = requests.post(
                f"{api_url}/documents/transform",
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
            
            if args.wait:
                wait_and_download(api_url, document_id, args.output)
            else:
                print(f"\n💡 To check status: python {sys.argv[0]} --api-url {api_url} status {document_id}")
                print(f"💡 To download result: python {sys.argv[0]} --api-url {api_url} download {document_id} output.png")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload failed: {str(e)}")


def cmd_url_upload(api_url: str, args):
    """Handle URL upload command"""
    
    print(f"🌐 Uploading from URL: {args.url}")
    
    try:
        data = {
            'url': args.url,
            'transformations': args.transform
        }
        
        if args.filename:
            data['filename'] = args.filename
        
        response = requests.post(
            f"{api_url}/documents/transform",
            data=data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        document_id = result['document_id']
        
        print(f"✅ URL upload successful!")
        print(f"   Document ID: {document_id}")
        print(f"   Status: {result['status']}")
        
        if args.wait:
            wait_and_download(api_url, document_id, args.output)
        else:
            print(f"\n💡 To check status: python {sys.argv[0]} --api-url {api_url} status {document_id}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ URL upload failed: {str(e)}")


def cmd_status(api_url: str, args):
    """Handle status check command"""
    
    print(f"📋 Checking status for document {args.document_id}...")
    
    try:
        response = requests.get(f"{api_url}/documents/{args.document_id}/status")
        response.raise_for_status()
        
        status = response.json()
        
        print(f"📄 Document: {status['filename']}")
        print(f"📊 Status: {status['status']}")
        print(f"🔧 Transform: {status['transformation_type']}")
        print(f"📏 File size: {status['file_size'] / 1024:.1f} KB")
        
        if status.get('progress'):
            progress = status['progress']
            print(f"📈 Progress: {progress['percentage']:.1f}% ({progress['completed_tasks']}/{progress['total_tasks']} tasks)")
        
        if status.get('processing_time'):
            print(f"⏱️ Processing time: {status['processing_time']:.1f}s")
        
        if status.get('error_message'):
            print(f"❌ Error: {status['error_message']}")
        
        if status.get('download_url'):
            print(f"📥 Download available: {api_url}{status['download_url']}")
            print(f"💡 Use: python {sys.argv[0]} --api-url {api_url} download {args.document_id} output.png")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Status check failed: {str(e)}")


def cmd_download(api_url: str, args):
    """Handle download command"""
    
    print(f"📥 Downloading result for document {args.document_id}...")
    
    try:
        response = requests.get(
            f"{api_url}/documents/{args.document_id}/result",
            stream=True,
            timeout=300
        )
        response.raise_for_status()
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        
        with open(args.output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(args.output_path)
        print(f"✅ Download complete: {file_size / 1024:.1f}KB saved to {args.output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {str(e)}")


def cmd_transformations(api_url: str):
    """Handle transformations list command"""
    
    print("📋 Available transformation types:")
    
    try:
        response = requests.get(f"{api_url}/documents/transformations")
        response.raise_for_status()
        
        result = response.json()
        
        for transform in result['transformations']:
            print(f"\n🔧 {transform['id']}")
            print(f"   Name: {transform['name']}")
            print(f"   Description: {transform['description']}")
            print(f"   Expected time: {transform['expected_time']}")
            print(f"   Quality: {transform['quality']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get transformations: {str(e)}")


def cmd_delete(api_url: str, args):
    """Handle delete command"""
    
    print(f"🗑️ Deleting document {args.document_id}...")
    
    try:
        response = requests.delete(f"{api_url}/documents/{args.document_id}")
        response.raise_for_status()
        
        print(f"✅ Document deleted successfully")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Delete failed: {str(e)}")


def cmd_health(api_url: str):
    """Handle health check command"""
    
    print(f"🏥 Checking API health at {api_url}...")
    
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()
        
        health = response.json()
        
        print(f"✅ API is healthy!")
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Service: {health['service']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API health check failed: {str(e)}")
        print(f"   Make sure the API server is running with: python api_main.py")


def wait_and_download(api_url: str, document_id: str, output_path: str = None):
    """Wait for processing to complete and download result"""
    
    print(f"\n⏳ Waiting for processing to complete...")
    
    if not output_path:
        output_path = f"result_{document_id}.png"
    
    max_wait = 300  # 5 minutes
    check_interval = 3
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{api_url}/documents/{document_id}/status")
            response.raise_for_status()
            
            status = response.json()
            
            if status['status'] == 'completed':
                print(f"✅ Processing completed!")
                
                # Download result
                print(f"📥 Downloading to {output_path}...")
                download_response = requests.get(
                    f"{api_url}/documents/{document_id}/result",
                    stream=True
                )
                download_response.raise_for_status()
                
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(output_path)
                print(f"🎉 Success! Result saved to {output_path} ({file_size / 1024:.1f}KB)")
                return
                
            elif status['status'] == 'failed':
                print(f"❌ Processing failed: {status.get('error_message', 'Unknown error')}")
                return
            else:
                # Show progress if available
                if 'progress' in status:
                    progress = status['progress']
                    print(f"🔄 Progress: {progress['percentage']:.1f}% ({progress['completed_tasks']}/{progress['total_tasks']} tasks)")
                else:
                    print(f"🔄 Status: {status['status']}")
            
            time.sleep(check_interval)
            
        except Exception as e:
            print(f"⚠️ Error checking status: {str(e)}")
            time.sleep(check_interval)
    
    print(f"⏰ Timeout waiting for completion after {max_wait}s")


if __name__ == "__main__":
    main()