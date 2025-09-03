#!/usr/bin/env python3
\"\"\"
Document Management System Tests
Comprehensive testing for the enhanced file management system
\"\"\"

import os
import tempfile
import asyncio
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import the API and database components
from api_main import app
from api.core.database import Base, get_db, init_db
from api.core.database import (
    EnhancedDocumentOps, DocumentPageOps, FileMetadataOps,
    calculate_file_hash
)
from api.services.enhanced_file_service import enhanced_file_service
from api.services.cache_service import cache_service


class TestDocumentManagementSystem:
    \"\"\"Test suite for the Document Management System\"\"\"
    
    def setup_method(self):
        \"\"\"Set up test database and client\"\"\"
        # Create test database
        self.test_db_url = \"sqlite:///./test_documents.db\"
        self.engine = create_engine(self.test_db_url, connect_args={\"check_same_thread\": False})
        self.TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        # Override dependency
        def override_get_db():
            db = self.TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()
        
        app.dependency_overrides[get_db] = override_get_db
        
        # Create test client
        self.client = TestClient(app)
        
        # Create test files
        self.test_files = self._create_test_files()
    
    def teardown_method(self):
        \"\"\"Clean up test database and files\"\"\"
        # Remove test database
        if os.path.exists(\"test_documents.db\"):
            os.remove(\"test_documents.db\")
        
        # Clean up test files
        for file_path in self.test_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clear app overrides
        app.dependency_overrides.clear()
    
    def _create_test_files(self) -> dict:
        \"\"\"Create test files for testing\"\"\"
        test_files = {}
        
        # Create a simple test PDF (placeholder)
        test_pdf_content = b\"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000125 00000 n \ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n179\n%%EOF\"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=\".pdf\") as f:
            f.write(test_pdf_content)
            test_files['test_pdf'] = f.name
        
        # Create a simple test image
        try:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='white')
            with tempfile.NamedTemporaryFile(delete=False, suffix=\".png\") as f:
                img.save(f.name, \"PNG\")
                test_files['test_image'] = f.name
        except ImportError:
            # If PIL not available, create a dummy file
            with tempfile.NamedTemporaryFile(delete=False, suffix=\".png\") as f:
                f.write(b\"dummy image data\")
                test_files['test_image'] = f.name
        
        return test_files
    
    def test_database_schema(self):
        \"\"\"Test that all database tables are created correctly\"\"\"
        db = self.TestingSessionLocal()
        
        # Test document creation with enhanced metadata
        document = EnhancedDocumentOps.create_document_with_metadata(
            db=db,
            filename=\"test_document.pdf\",
            original_filename=\"original_test.pdf\",
            original_path=\"/tmp/test.pdf\",
            file_size=1024,
            file_type=\"pdf\",
            content_hash=\"abc123hash\",
            page_count=3,
            document_dimensions={\"width\": 612, \"height\": 792, \"unit\": \"points\"}
        )
        
        assert document is not None
        assert document.filename == \"test_document.pdf\"
        assert document.original_filename == \"original_test.pdf\"
        assert document.page_count == 3
        assert document.content_hash == \"abc123hash\"
        
        # Test page creation
        page = DocumentPageOps.create_page(
            db=db,
            document_id=document.id,
            page_number=1,
            page_hash=\"page1hash\",
            page_dimensions={\"width\": 612, \"height\": 792, \"unit\": \"points\"}
        )
        
        assert page is not None
        assert page.document_id == document.id
        assert page.page_number == 1
        assert page.page_hash == \"page1hash\"
        
        # Test metadata creation
        metadata = FileMetadataOps.create_metadata(
            db=db,
            document_id=document.id,
            file_format=\"PDF\",
            has_text=True,
            language_detected=\"en\"
        )
        
        assert metadata is not None
        assert metadata.document_id == document.id
        assert metadata.file_format == \"PDF\"
        assert metadata.has_text is True
        
        db.close()
    
    def test_search_functionality(self):
        \"\"\"Test document search capabilities\"\"\"
        db = self.TestingSessionLocal()
        
        # Create test documents
        doc1 = EnhancedDocumentOps.create_document_with_metadata(
            db=db,
            filename=\"invoice_2023.pdf\",
            original_filename=\"Invoice_January_2023.pdf\",
            original_path=\"/tmp/invoice.pdf\",
            file_size=2048,
            file_type=\"pdf\",
            content_hash=\"hash1\"
        )
        
        doc2 = EnhancedDocumentOps.create_document_with_metadata(
            db=db,
            filename=\"receipt_2023.pdf\",
            original_filename=\"Receipt_January_2023.pdf\",
            original_path=\"/tmp/receipt.pdf\",
            file_size=1024,
            file_type=\"pdf\",
            content_hash=\"hash2\"
        )
        
        # Test filename search
        results = EnhancedDocumentOps.search_by_filename(db, \"invoice\")
        assert len(results) == 1
        assert results[0].id == doc1.id
        
        # Test hash search
        results = EnhancedDocumentOps.search_by_hash(db, \"hash2\")
        assert len(results) == 1
        assert results[0].id == doc2.id
        
        # Test advanced search
        results = EnhancedDocumentOps.search_documents(
            db=db,
            filename=\"2023\",
            file_type=\"pdf\",
            limit=10
        )
        assert len(results) == 2
        
        db.close()
    
    def test_api_endpoints(self):
        \"\"\"Test API endpoints functionality\"\"\"
        db = self.TestingSessionLocal()
        
        # Create a test document
        document = EnhancedDocumentOps.create_document_with_metadata(
            db=db,
            filename=\"api_test.pdf\",
            original_filename=\"API_Test_Document.pdf\",
            original_path=\"/tmp/api_test.pdf\",
            file_size=4096,
            file_type=\"pdf\",
            content_hash=\"api_hash_123\",
            page_count=2
        )
        
        # Create pages
        page1 = DocumentPageOps.create_page(
            db=db, document_id=document.id, page_number=1
        )
        page2 = DocumentPageOps.create_page(
            db=db, document_id=document.id, page_number=2
        )
        
        # Update document status to completed
        EnhancedDocumentOps.update_document_status(db, document.id, \"completed\")
        
        db.close()
        
        # Test metadata endpoint
        response = self.client.get(f\"/documents/{document.id}/metadata\")
        assert response.status_code == 200
        
        metadata = response.json()
        assert metadata[\"document_id\"] == document.id
        assert metadata[\"file_info\"][\"original_filename\"] == \"API_Test_Document.pdf\"
        assert metadata[\"structure\"][\"page_count\"] == 2
        
        # Test search endpoint
        response = self.client.get(\"/documents/search?name=API_Test\")
        assert response.status_code == 200
        
        search_results = response.json()
        assert search_results[\"total_results\"] >= 1
        assert len(search_results[\"documents\"]) >= 1
        
        # Test search by hash
        response = self.client.get(f\"/documents/search?hash=api_hash_123\")
        assert response.status_code == 200
        
        search_results = response.json()
        assert search_results[\"total_results\"] >= 1
        
        # Test pages info endpoint
        response = self.client.get(f\"/documents/{document.id}/pages\")
        assert response.status_code == 200
        
        pages_info = response.json()
        assert len(pages_info) == 2
        assert pages_info[0][\"page_number\"] == 1
        assert pages_info[1][\"page_number\"] == 2
    
    def test_cache_functionality(self):
        \"\"\"Test caching system\"\"\"
        # Test memory cache
        test_data = {\"test\": \"data\", \"number\": 123}
        
        # Set cache entry
        success = cache_service.memory_cache.set(\"test_key\", test_data, ttl=60)
        assert success is True
        
        # Get cache entry
        cached_data = cache_service.memory_cache.get(\"test_key\")
        assert cached_data == test_data
        
        # Test cache statistics
        stats = cache_service.get_cache_statistics()
        assert \"memory_cache\" in stats
        assert \"file_cache\" in stats
        assert stats[\"memory_cache\"][\"size\"] >= 1
    
    def test_file_hash_calculation(self):
        \"\"\"Test file hash calculation\"\"\"
        # Test with existing test files
        if self.test_files['test_pdf']:
            hash1 = calculate_file_hash(self.test_files['test_pdf'])
            assert hash1 is not None
            assert len(hash1) == 64  # SHA-256 hex length
            
            # Calculate again to ensure consistency
            hash2 = calculate_file_hash(self.test_files['test_pdf'])
            assert hash1 == hash2
    
    def test_performance_indexes(self):
        \"\"\"Test that database indexes work correctly for performance\"\"\"
        db = self.TestingSessionLocal()
        
        # Create multiple documents
        for i in range(10):
            EnhancedDocumentOps.create_document_with_metadata(
                db=db,
                filename=f\"perf_test_{i}.pdf\",
                original_filename=f\"Performance_Test_{i}.pdf\",
                original_path=f\"/tmp/perf_{i}.pdf\",
                file_size=1024 * i,
                file_type=\"pdf\",
                content_hash=f\"perf_hash_{i}\"
            )
        
        # Test that searches are fast (this is a basic test)
        import time
        
        start_time = time.time()
        results = EnhancedDocumentOps.search_by_filename(db, \"perf_test\")
        end_time = time.time()
        
        # Should find all 10 documents quickly
        assert len(results) == 10
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second
        
        # Test hash search performance
        start_time = time.time()
        results = EnhancedDocumentOps.search_by_hash(db, \"perf_hash_5\")
        end_time = time.time()
        
        assert len(results) == 1
        assert (end_time - start_time) < 0.1  # Should be very fast with index
        
        db.close()


def run_tests():
    \"\"\"Run all tests\"\"\"
    test_suite = TestDocumentManagementSystem()
    
    print(\"🧪 Running Document Management System Tests...\")
    
    tests = [
        (\"Database Schema\", test_suite.test_database_schema),
        (\"Search Functionality\", test_suite.test_search_functionality),
        (\"API Endpoints\", test_suite.test_api_endpoints),
        (\"Cache Functionality\", test_suite.test_cache_functionality),
        (\"File Hash Calculation\", test_suite.test_file_hash_calculation),
        (\"Performance Indexes\", test_suite.test_performance_indexes)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f\"\n🔧 Testing {test_name}...\")
            test_suite.setup_method()
            test_func()
            test_suite.teardown_method()
            print(f\"✅ {test_name}: PASSED\")
            passed += 1
        except Exception as e:
            print(f\"❌ {test_name}: FAILED - {str(e)}\")
            failed += 1
            try:
                test_suite.teardown_method()
            except:
                pass
    
    print(f\"\n📊 Test Results: {passed} passed, {failed} failed\")
    
    if failed == 0:
        print(\"🎉 All tests passed! Document Management System is ready.\")
    else:
        print(f\"⚠️ {failed} tests failed. Please review and fix issues.\")
    
    return failed == 0


if __name__ == \"__main__\":
    # Run tests if script is executed directly
    success = run_tests()
    exit(0 if success else 1)