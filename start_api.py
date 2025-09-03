#!/usr/bin/env python3
"""
API Server Startup Script
Easy startup with automatic configuration and health checks
"""

import os
import sys
import time
import argparse
import subprocess
import requests
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('sqlalchemy', 'sqlalchemy'),
        ('pydantic', 'pydantic'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pillow', 'PIL'),
        ('httpx', 'httpx')
    ]
    
    missing = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True


def setup_directories():
    """Create necessary directories"""
    
    print("📁 Setting up directories...")
    
    directories = [
        'api_uploads',
        'processed_documents',  # Dedicated folder for processed files
        'api_temp',
        'api_cache',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ {directory}")
    
    print("✅ Directories ready")


def check_port_available(port):
    """Check if port is available"""
    
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False


def wait_for_api(host, port, timeout=30):
    """Wait for API to become available"""
    
    url = f"http://{host}:{port}/health"
    print(f"⏳ Waiting for API to start at {url}...")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ API is running and healthy!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\n❌ API did not start within {timeout} seconds")
    return False


def main():
    """Main startup function"""
    
    parser = argparse.ArgumentParser(
        description="Document Processing API Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_api.py                    # Start with defaults
  python start_api.py --port 8080        # Custom port
  python start_api.py --host 0.0.0.0     # Listen on all interfaces
  python start_api.py --workers 4        # Custom worker count
  python start_api.py --dev              # Development mode
        """
    )
    
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--dev', action='store_true', help='Enable development mode (auto-reload)')
    parser.add_argument('--no-deps-check', action='store_true', help='Skip dependency check')
    parser.add_argument('--no-setup', action='store_true', help='Skip directory setup')
    
    args = parser.parse_args()
    
    print("🚀 Document Processing API Startup")
    print("=" * 50)
    
    # Check dependencies
    if not args.no_deps_check:
        if not check_dependencies():
            print("\n💡 Run 'pip install -r requirements.txt' to install missing packages")
            return 1
    
    # Setup directories
    if not args.no_setup:
        setup_directories()
    
    # Check port availability
    if not check_port_available(args.port):
        print(f"❌ Port {args.port} is already in use")
        print(f"💡 Try a different port with --port {args.port + 1}")
        return 1
    
    # Prepare startup command
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "api_main:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.dev:
        cmd.extend(["--reload", "--log-level", "debug"])
        print("🔧 Development mode enabled (auto-reload)")
    else:
        cmd.extend(["--workers", str(args.workers)])
        print(f"👥 Starting with {args.workers} worker(s)")
    
    print(f"🌐 Starting API server at http://{args.host}:{args.port}")
    print("📚 API documentation will be available at:")
    print(f"   • Interactive docs: http://{args.host}:{args.port}/docs")
    print(f"   • Alternative docs: http://{args.host}:{args.port}/redoc")
    print()
    
    try:
        # Start the server
        process = subprocess.Popen(cmd)
        
        # Wait for startup
        if wait_for_api(args.host, args.port):
            print("\n🎉 API server started successfully!")
            print("\n💡 Quick start:")
            print(f"   curl {args.host}:{args.port}/health")
            print(f"   python examples/api_client.py --api-url http://{args.host}:{args.port} health")
            print("\n⏹️  Press Ctrl+C to stop the server")
            
            # Wait for process
            process.wait()
        else:
            print("❌ Failed to start API server")
            process.terminate()
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping API server...")
        process.terminate()
        print("✅ Server stopped")
        return 0
    
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())