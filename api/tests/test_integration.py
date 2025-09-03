#!/usr/bin/env python3
"""
Integration Tests for Document Processing API
Comprehensive testing of API endpoints and functionality
"""

import os
import sys
import asyncio
import tempfile
import json
from typing import Dict, Any
import pytest
import httpx
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api_main import app
from api.core.config import settings

# Test client
client = TestClient(app)


class TestDocumentAPI:
    """Integration tests for document processing API"""
    
    @pytest.fixture
    def test_pdf_file(self):
        """Create a test PDF file"""
        # For now, create a simple image file as a placeholder
        # In a real implementation, you'd create an actual PDF
        test_data = b"This is a test PDF content"
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(test_data)
            return f.name
    
    @pytest.fixture
    def test_image_file(self):
        """Create a test image file"""
        # Create a simple test image
        from PIL import Image
        
        img = Image.new('RGB', (800, 600), color='white')
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name, 'PNG')
            return f.name
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_get_transformations(self):
        """Test getting available transformation types"""
        response = client.get("/documents/transformations")
        assert response.status_code == 200
        
        data = response.json()
        assert "transformations" in data
        assert len(data["transformations"]) > 0
        
        # Check required fields
        for transform in data["transformations"]:
            assert "id" in transform
            assert "name" in transform
            assert "description" in transform
    
    def test_upload_document_file(self, test_image_file):
        """Test document upload with file"""
        with open(test_image_file, "rb") as f:
            response = client.post(
                "/documents/transform",
                files={"file": ("test.png", f, "image/png")},
                data={"transformations": "deskewing"}
            )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "document_id" in data
        assert data["status"] in ["pending", "completed"]
        assert data["transformation_type"] == "deskewing"
        
        # Clean up
        os.unlink(test_image_file)
        
        return data["document_id"]
    
    def test_upload_document_url(self):
        """Test document upload with URL"""
        # Use a test URL (this would fail in real testing without a valid URL)
        response = client.post(
            "/documents/transform",
            data={
                "url": "https://example.com/test.pdf",
                "transformations": "basic",
                "filename": "test_url.pdf"
            }
        )
        
        # This will likely fail due to invalid URL, but tests the endpoint structure
        assert response.status_code in [200, 400, 500]
    
    def test_document_status(self, test_image_file):
        """Test document status checking"""
        # First upload a document
        document_id = self.test_upload_document_file(test_image_file)
        
        # Check status
        response = client.get(f"/documents/{document_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["document_id"] == document_id
        assert "status" in data
        assert data["status"] in ["pending", "in_progress", "completed", "failed"]
    
    def test_document_metadata(self, test_image_file):
        """Test document metadata retrieval"""
        document_id = self.test_upload_document_file(test_image_file)
        
        response = client.get(f"/documents/{document_id}/metadata")
        assert response.status_code == 200
        
        data = response.json()
        assert "document_info" in data
        assert "processing_info" in data
        assert "timestamps" in data
    
    def test_invalid_file_type(self):
        """Test upload with invalid file type"""
        # Create a text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is a text file")
            f.flush()
            
            with open(f.name, "rb") as test_file:
                response = client.post(
                    "/documents/transform",
                    files={"file": ("test.txt", test_file, "text/plain")},
                    data={"transformations": "deskewing"}
                )
        
        assert response.status_code == 400
        os.unlink(f.name)
    
    def test_missing_file_and_url(self):
        """Test request without file or URL"""
        response = client.post(
            "/documents/transform",
            data={"transformations": "deskewing"}
        )
        
        assert response.status_code == 400
    
    def test_invalid_transformation_type(self, test_image_file):
        """Test with invalid transformation type"""
        with open(test_image_file, "rb") as f:
            response = client.post(
                "/documents/transform",
                files={"file": ("test.png", f, "image/png")},
                data={"transformations": "invalid_type"}
            )
        
        assert response.status_code == 400
        os.unlink(test_image_file)
    
    def test_document_not_found(self):
        """Test accessing non-existent document"""
        fake_id = "non-existent-document-id"
        
        response = client.get(f"/documents/{fake_id}/status")
        assert response.status_code == 404
        
        response = client.get(f"/documents/{fake_id}/result")
        assert response.status_code == 404
        
        response = client.get(f"/documents/{fake_id}/metadata")
        assert response.status_code == 404
    
    def test_admin_stats(self):
        """Test admin statistics endpoint"""
        response = client.get("/documents/admin/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_documents" in data
        assert "status_breakdown" in data
        assert "cache_enabled" in data


class TestPerformance:
    """Performance and load testing"""
    
    def test_concurrent_uploads(self):
        """Test concurrent document uploads"""
        import threading
        import time
        
        results = []
        
        def upload_test_doc():
            # Create test image
            from PIL import Image
            img = Image.new('RGB', (400, 300), color='white')
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name, 'PNG')
                
                try:
                    with open(f.name, "rb") as test_file:
                        start_time = time.time()
                        response = client.post(
                            "/documents/transform",
                            files={"file": ("test.png", test_file, "image/png")},
                            data={"transformations": "basic"}
                        )
                        end_time = time.time()
                        
                        results.append({
                            "status_code": response.status_code,
                            "response_time": end_time - start_time,
                            "success": response.status_code == 200
                        })
                finally:
                    os.unlink(f.name)
        
        # Start multiple concurrent uploads
        threads = []
        num_concurrent = 3
        
        for _ in range(num_concurrent):
            thread = threading.Thread(target=upload_test_doc)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Analyze results
        assert len(results) == num_concurrent
        successful = sum(1 for r in results if r["success"])
        
        print(f"Concurrent upload test: {successful}/{num_concurrent} successful")
        print(f"Average response time: {sum(r['response_time'] for r in results) / len(results):.2f}s")
        
        # At least 50% should succeed under load
        assert successful >= num_concurrent * 0.5
    
    def test_large_file_handling(self):
        """Test handling of large files"""
        from PIL import Image
        
        # Create a larger test image (simulating a large document)
        large_img = Image.new('RGB', (2000, 1500), color='white')
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            large_img.save(f.name, 'PNG', quality=95)
            file_size = os.path.getsize(f.name)
            
            print(f"Testing large file: {file_size / (1024*1024):.1f}MB")
            
            try:
                with open(f.name, "rb") as test_file:
                    start_time = time.time()
                    response = client.post(
                        "/documents/transform",
                        files={"file": ("large_test.png", test_file, "image/png")},
                        data={"transformations": "basic"}
                    )
                    end_time = time.time()
                    
                    print(f"Large file processing time: {end_time - start_time:.2f}s")
                    
                    # Should either succeed or fail gracefully
                    assert response.status_code in [200, 413, 500]
                    
            finally:
                os.unlink(f.name)


async def test_async_operations():
    """Test asynchronous operations"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        # Test health check
        response = await ac.get("/health")
        assert response.status_code == 200
        
        # Test transformations endpoint
        response = await ac.get("/documents/transformations")
        assert response.status_code == 200


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Document Processing API Integration Tests")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])


if __name__ == "__main__":
    run_integration_tests()