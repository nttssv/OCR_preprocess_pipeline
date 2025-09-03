# Document Processing Pipeline API

A powerful REST API for intelligent document preprocessing and transformation, built on top of the existing OCR preprocessing pipeline.

## 🚀 Features

- **PDF Transformation**: Upload PDFs and apply intelligent preprocessing
- **Multiple Input Methods**: Support for direct file uploads and URL references  
- **Default Deskewing**: Fast orientation correction, skew detection, and cropping
- **Modular Pipeline**: Extensible transformation chain for various processing needs
- **Performance Optimized**: Streaming processing, intelligent caching, and fallback strategies
- **Status Tracking**: Real-time processing status with document ID management
- **Benchmark Tested**: Performance-tested with automatic threshold optimization

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Transformation │    │  Core Pipeline  │
│   (FastAPI)     │───▶│    Service      │───▶│   (Existing)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Handler  │    │  Performance    │    │   Task Manager  │
│   (Upload/URL)  │    │   Optimizer     │    │   (12 Tasks)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Database     │    │      Cache      │    │   File System   │
│   (SQLAlchemy)  │    │   Management    │    │    Storage      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📚 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd OCR_preprocess_pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional)
cp .env.example .env
# Edit .env with your configuration
```

### 2. Start the API Server

```bash
# Start the API server
python api_main.py

# Or with custom configuration
python api_main.py --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8000` with:
- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

### 3. Basic Usage

#### Upload a File
```bash
curl -X POST "http://localhost:8000/documents/transform" \
     -F "file=@document.pdf" \
     -F "transformations=deskewing"
```

#### Check Status
```bash
curl "http://localhost:8000/documents/{document_id}/status"
```

#### Download Result
```bash
curl "http://localhost:8000/documents/{document_id}/result" -o result.png
```

## 🔧 Transformation Types

| Type | Description | Use Case | Speed | Quality |
|------|-------------|----------|-------|---------|
| **basic** | Orientation correction + cropping | Quick processing | ⚡⚡⚡ | ⭐⭐ |
| **deskewing** | Orientation + skew detection + cropping | Default OCR prep | ⚡⚡ | ⭐⭐⭐ |
| **enhanced** | Full preprocessing + noise reduction | High quality | ⚡ | ⭐⭐⭐⭐ |
| **comprehensive** | Complete pipeline with all features | Maximum quality | 🐌 | ⭐⭐⭐⭐⭐ |

## 📡 API Endpoints

### Document Transformation

#### `POST /documents/transform`

Transform a document with specified preprocessing steps.

**Parameters:**
- `file` (file): Document file to upload
- `url` (string): Alternative URL to download document
- `transformations` (string): Transformation type (default: "deskewing")
- `filename` (string): Custom filename for URL uploads

**Response:**
```json
{
  "document_id": "uuid-string",
  "status": "pending",
  "message": "Document processing started",
  "transformation_type": "deskewing", 
  "estimated_time": "5-15 seconds",
  "location": "/documents/{id}/status"
}
```

#### `GET /documents/{document_id}/status`

Get processing status and progress information.

**Response:**
```json
{
  "document_id": "uuid-string",
  "filename": "document.pdf",
  "status": "completed",
  "transformation_type": "deskewing",
  "processing_time": 12.5,
  "progress": {
    "percentage": 100.0,
    "completed_tasks": 3,
    "total_tasks": 3
  },
  "download_url": "/documents/{id}/result"
}
```

#### `GET /documents/{document_id}/result`

Download the processed document result.

**Response:** Binary file content with appropriate headers.

### Utility Endpoints

#### `GET /documents/transformations`

Get available transformation types and their descriptions.

#### `GET /documents/{document_id}/metadata`

Get detailed processing metadata and quality metrics.

#### `DELETE /documents/{document_id}`

Delete a document and its processed results.

#### `GET /health`

API health check endpoint.

## 💻 Command Line Client

Use the included command-line client for easy API interaction:

```bash
# Upload a file
python examples/api_client.py upload document.pdf --wait --output result.png

# Upload from URL
python examples/api_client.py url https://example.com/doc.pdf --transform enhanced

# Check status
python examples/api_client.py status abc123-document-id

# Download result
python examples/api_client.py download abc123-document-id result.png

# List transformations
python examples/api_client.py transformations

# Health check
python examples/api_client.py health
```

## 🐍 Python Client Examples

```python
from examples.api_usage_examples import DocumentProcessingClient

# Initialize client
client = DocumentProcessingClient()

# Upload and process
document_id = client.upload_file("document.pdf", "deskewing")

# Wait for completion
status = client.wait_for_completion(document_id)

# Download result
client.download_result(document_id, "result.png")
```

## ⚡ Performance Features

### Automatic Optimization

The API automatically optimizes processing based on:

- **File Size**: Smaller files get enhanced processing, larger files use fast modes
- **System Resources**: CPU and memory usage influence transformation selection
- **Processing History**: Learning from previous processing times

### Intelligent Caching

- **Content-based**: Files with identical content reuse cached results
- **Transformation-aware**: Cache considers transformation type
- **Automatic cleanup**: Expired cache entries are removed automatically

### Fallback Strategies

| File Size | Default Strategy | Timeout | Fallback |
|-----------|------------------|---------|----------|
| < 10MB | Enhanced | 60s | Deskewing |
| < 50MB | Deskewing | 120s | Basic |
| ≥ 50MB | Basic | 180s | Error |

### Streaming Processing

Large files (>25MB) automatically use streaming processing to reduce memory usage.

## 🧪 Testing & Benchmarking

### Run Benchmark Tests

```bash
python api/tests/benchmark.py
```

This generates a comprehensive performance report including:
- Transformation speed analysis
- Size threshold recommendations
- Concurrent processing performance
- System resource optimization

### Run Integration Tests

```bash
python api/tests/test_integration.py
```

Tests all API endpoints with various file types and scenarios.

### Example Benchmark Results

```
Transformation Performance:
• basic: 3.2s average, 2.1 MB/s throughput
• deskewing: 8.5s average, 1.4 MB/s throughput  
• enhanced: 24.1s average, 0.8 MB/s throughput

Size Recommendations:
• Small files (<5MB): Use enhanced transformation
• Medium files (5-25MB): Use deskewing transformation
• Large files (>25MB): Use basic transformation
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
# Server
HOST=127.0.0.1
PORT=8000
DEBUG=False

# Processing
MAX_FILE_SIZE=104857600  # 100MB
MAX_WORKERS=4
TASK_TIMEOUT=300

# Performance  
ENABLE_CACHING=True
CACHE_EXPIRY_HOURS=24
LARGE_FILE_THRESHOLD=52428800  # 50MB

# Storage
UPLOAD_DIR=./api_uploads
OUTPUT_DIR=./api_output
TEMP_DIR=./api_temp
```

### Processing Pipeline Configuration

The API inherits all configuration options from the original pipeline:

- **Task Selection**: Choose which processing tasks to run
- **Quality Settings**: Control processing quality vs speed
- **Output Formats**: Customize output file formats
- **Parallel Processing**: Control worker count and concurrency

## 📊 Monitoring

### Performance Statistics

Get processing statistics:

```bash
curl "http://localhost:8000/documents/admin/stats"
```

Response includes:
- Total documents processed
- Cache hit/miss rates
- Average processing times
- System resource usage

### Database Management

The API uses SQLite by default with automatic management:

- **Document tracking**: All uploads and processing status
- **Cache management**: Automatic cleanup of expired entries
- **Performance metrics**: Historical processing data

## 🛠️ Development

### Adding New Transformations

1. **Define in config**:
```python
# In api/core/config.py
TRANSFORMATION_CONFIGS["custom"] = {
    "name": "Custom Processing",
    "description": "Custom transformation pipeline",
    "tasks": ["task_1_orientation_correction", "custom_task"],
    "expected_time": "10-20 seconds",
    "quality": "high"
}
```

2. **Update service**:
```python
# In api/services/transformation.py
def _get_execution_mode(self, transformation_type: str) -> str:
    mode_mapping = {
        # ... existing mappings
        "custom": "custom_mode"
    }
    return mode_mapping.get(transformation_type, "orient_skew_crop")
```

### Custom Task Implementation

Follow the existing task structure in `tasks/` directory:

```python
class CustomTask:
    def __init__(self, logger=None):
        self.logger = logger
        self.task_name = "Custom Processing"
        self.task_id = "custom_task"
    
    def run(self, input_file, file_type, output_folder):
        # Implementation here
        return {
            'input': input_file,
            'output': output_path,
            'status': 'completed',
            'task': self.task_id
        }
```

## 🐛 Troubleshooting

### Common Issues

**API won't start:**
```bash
# Check if port is in use
lsof -i :8000

# Try different port
python api_main.py --port 8080
```

**File upload fails:**
```bash
# Check file size limit
curl "http://localhost:8000/documents/transformations"

# Verify file type support
file document.pdf
```

**Processing timeouts:**
```bash
# Check system resources
python -c "from api.services.performance import performance_optimizer; import asyncio; print(asyncio.run(performance_optimizer.check_system_resources()))"

# Use faster transformation
curl -F "transformations=basic" ...
```

**Cache issues:**
```bash
# Clear cache
rm -rf api_cache/*

# Disable caching
echo "ENABLE_CACHING=False" >> .env
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
DEBUG=True python api_main.py
```

This provides:
- Detailed request/response logging
- Processing step information
- Performance timing data
- Error stack traces

## 📈 Performance Tuning

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB free space

**Recommended:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: SSD with 50GB+ free space

### Optimization Tips

1. **Adjust worker count** based on CPU cores:
   ```bash
   MAX_WORKERS=8  # For 8-core systems
   ```

2. **Tune cache settings** for your storage:
   ```bash
   MAX_CACHE_SIZE=2147483648  # 2GB cache
   CACHE_EXPIRY_HOURS=48      # Keep cache longer
   ```

3. **Use appropriate transformations**:
   - Batch processing: Use "basic" transformation
   - Real-time processing: Use "deskewing"
   - Archival quality: Use "enhanced"

4. **Monitor system resources**:
   ```bash
   python api/tests/benchmark.py
   ```

## 📝 License

This API is built on the Document Processing Pipeline and inherits its license terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Run the benchmark tests
3. Check the interactive API docs at `/docs`
4. Review the example scripts in `examples/`

---

**Built with ❤️ using FastAPI and the Document Processing Pipeline**