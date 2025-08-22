#!/usr/bin/env python3
"""
Ingestion utilities
- Fast PDF → image conversion (uses PyMuPDF if available, falls back to pdf2image)
- Stable document IDs (SHA256 of file contents)
- Page image hashing for deduplication
- Lightweight SQLite index to skip duplicates across runs
"""

import os
import io
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

DEFAULT_DPI = 250  # Good balance of speed/quality


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    sha = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


def compute_document_id(file_path: str) -> str:
    """Stable document ID based on file contents (SHA256)."""
    return compute_file_sha256(file_path)


def normalize_image_for_hash(image_path: str) -> bytes:
    """Normalize an image and return PNG bytes for stable hashing.

    Steps:
    - Load with Pillow
    - Convert to L (grayscale)
    - Resize to max dimension 1024 (keeps aspect) to stabilize hashing
    - Save to PNG bytes buffer
    """
    img = Image.open(image_path)
    img = img.convert('L')
    max_dim = 1024
    w, h = img.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def compute_page_hash(image_path: str) -> str:
    """SHA256 of normalized PNG bytes."""
    data = normalize_image_for_hash(image_path)
    return hashlib.sha256(data).hexdigest()


def init_index(db_path: str) -> None:
    ensure_dir(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                page_hash TEXT PRIMARY KEY,
                document_id TEXT,
                source_path TEXT,
                output_path TEXT,
                created_at REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_path TEXT,
                created_at REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def page_exists(db_path: str, page_hash: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pages WHERE page_hash=?", (page_hash,))
        return cur.fetchone() is not None
    finally:
        conn.close()


def record_document(db_path: str, document_id: str, source_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO documents(document_id, source_path, created_at) VALUES(?, ?, ?)",
            (document_id, source_path, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def record_page(db_path: str, page_hash: str, document_id: str, source_path: str, output_path: Optional[str]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO pages(page_hash, document_id, source_path, output_path, created_at) VALUES(?, ?, ?, ?, ?)",
            (page_hash, document_id, source_path, output_path or '', time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def convert_pdf_to_images(pdf_path: str, out_dir: str, dpi: int = DEFAULT_DPI) -> List[str]:
    """Convert all pages of a PDF to images at the given DPI.

    Uses PyMuPDF (fitz) if available for speed; falls back to pdf2image.
    Returns list of image paths (PNG), one per page.
    """
    ensure_dir(out_dir)
    images: List[str] = []

    # Try fast path: PyMuPDF
    try:
        import fitz  # type: ignore
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        doc = fitz.open(pdf_path)
        try:
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_path = os.path.join(out_dir, f"page_{page_index + 1:04d}.png")
                pix.save(img_path)
                images.append(img_path)
        finally:
            doc.close()
        return images
    except Exception:
        # Fallback to pdf2image
        try:
            from pdf2image import convert_from_path  # type: ignore
            pages = convert_from_path(pdf_path, dpi=dpi)
            for i, pil_img in enumerate(pages, start=1):
                img_path = os.path.join(out_dir, f"page_{i:04d}.png")
                pil_img.save(img_path, format='PNG')
                images.append(img_path)
            return images
        except Exception as e:
            raise RuntimeError(f"PDF conversion failed for {os.path.basename(pdf_path)}: {e}")


def expand_inputs_with_pdfs(input_folder: str, temp_folder: str, dpi: int = DEFAULT_DPI, db_path: Optional[str] = None) -> Tuple[list, list]:
    """Expand inputs: convert PDFs to page images and build processing list.

    Returns (images_to_process, skipped_pages)
    where images_to_process is list of tuples: (file_type, image_path)
    """
    supported_images = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    supported_pdfs = ('.pdf',)

    images_to_process = []
    skipped_pages = []

    # Initialize index if provided
    if db_path:
        init_index(db_path)

    for name in os.listdir(input_folder):
        src = os.path.join(input_folder, name)
        if not os.path.isfile(src):
            continue
        lower = name.lower()
        if lower.endswith(supported_images):
            if db_path:
                # Compute page hash for dedup
                try:
                    p_hash = compute_page_hash(src)
                    if page_exists(db_path, p_hash):
                        skipped_pages.append(src)
                        continue
                except Exception:
                    # Non-fatal; proceed
                    pass
            images_to_process.append(('image', src))
        elif lower.endswith(supported_pdfs):
            # Convert to page images
            doc_id = compute_document_id(src)
            if db_path:
                record_document(db_path, doc_id, src)
            pdf_out_dir = os.path.join(temp_folder, 'pdf_pages', os.path.splitext(name)[0])
            page_imgs = convert_pdf_to_images(src, pdf_out_dir, dpi=dpi)
            for img_path in page_imgs:
                if db_path:
                    try:
                        p_hash = compute_page_hash(img_path)
                        if page_exists(db_path, p_hash):
                            skipped_pages.append(img_path)
                            continue
                    except Exception:
                        pass
                images_to_process.append(('image', img_path))

    return images_to_process, skipped_pages


