#!/usr/bin/env python
"""
Dynamic document loader for production
Supports: PDF, DOCX, PPTX, MD, HTML, TXT, CSV, JSON, XML, YAML, and more
"""

import os
import json
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import magic  # python-magic for MIME type detection
import mimetypes
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
)
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    """File information"""
    path: Path
    extension: str
    mime_type: str
    file_size: int
    format: str

class DynamicDocumentLoader:
    """Dynamic document loader for any file format"""
    
    def __init__(self, supported_formats: List[str] = None):
        # Default supported formats
        self.supported_formats = supported_formats or [
            '.pdf', '.docx', '.pptx', '.md', '.html', '.htm',
            '.txt', '.csv', '.json', '.xml', '.yaml', '.yml',
            '.xlsx', '.xls', '.jpg', '.jpeg', '.png',
        ]
        
        # Format-specific loaders
        self.loaders = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.pptx': self._load_pptx,
            '.md': self._load_markdown,
            '.html': self._load_html,
            '.htm': self._load_html,
            '.txt': self._load_text,
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.xml': self._load_xml,
            '.yaml': self._load_yaml,
            '.yml': self._load_yaml,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.jpg': self._load_image,
            '.jpeg': self._load_image,
            '.png': self._load_image,
        }
        
        # Register custom loaders
        self._register_custom_loaders()
    
    def _register_custom_loaders(self):
        """Register custom loaders for specific formats"""
        # Add any custom loaders here
        pass
    
    def detect_format(self, file_path: Path) -> FileInfo:
        """Detect file format and return FileInfo"""
        extension = file_path.suffix.lower()
        
        # Get MIME type
        mime_type = magic.from_file(str(file_path), mime=True)
        if not mime_type:
            mime_type = mimetypes.guess_type(str(file_path))[0] or 'unknown'
        
        # Determine format
        if extension in self.loaders:
            format = extension[1:]  # Remove dot
        elif mime_type.startswith('image/'):
            format = 'image'
        else:
            format = 'unknown'
        
        return FileInfo(
            path=file_path,
            extension=extension,
            mime_type=mime_type,
            file_size=file_path.stat().st_size,
            format=format
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document dynamically based on its format"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect format
        file_info = self.detect_format(path)
        
        # Select loader
        if file_info.extension in self.loaders:
            loader_func = self.loaders[file_info.extension]
            docs = loader_func(path)
            logger.info(f"✅ Loaded {len(docs)} chunks from {path.name} ({file_info.format})")
            return docs
        else:
            logger.warning(f"⚠️ Unsupported format: {file_info.extension}")
            return []
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """Load all documents from a directory"""
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_docs = []
        total_files = 0
        
        # Walk directory
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                
                # Check extension
                ext = file_path.suffix.lower()
                if ext not in self.supported_formats:
                    continue
                
                try:
                    docs = self.load_document(str(file_path))
                    all_docs.extend(docs)
                    total_files += 1
                except Exception as e:
                    logger.error(f"❌ Error loading {file}: {e}")
            
            if not recursive:
                break
        
        logger.info(f"📊 Loaded {len(all_docs)} chunks from {total_files} files")
        return all_docs
    
    # ================================================================
    # FORMAT-SPECIFIC LOADERS
    # ================================================================
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF document"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    
    def _load_docx(self, file_path: Path) -> List[Document]:
        """Load DOCX document"""
        loader = Docx2txtLoader(str(file_path))
        return loader.load()
    
    def _load_pptx(self, file_path: Path) -> List[Document]:
        """Load PPTX document"""
        loader = UnstructuredPowerPointLoader(str(file_path))
        return loader.load()
    
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """Load Markdown document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(page_content=content, metadata={'source': str(file_path)})]
    
    def _load_html(self, file_path: Path) -> List[Document]:
        """Load HTML document"""
        loader = UnstructuredHTMLLoader(str(file_path))
        return loader.load()
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """Load text document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(page_content=content, metadata={'source': str(file_path)})]
    
    def _load_csv(self, file_path: Path) -> List[Document]:
        """Load CSV document"""
        loader = CSVLoader(str(file_path))
        return loader.load()
    
    def _load_json(self, file_path: Path) -> List[Document]:
        """Load JSON document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        content = json.dumps(data, indent=2)
        return [Document(page_content=content, metadata={'source': str(file_path)})]
    
    def _load_xml(self, file_path: Path) -> List[Document]:
        """Load XML document"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        content = ET.tostring(root, encoding='unicode')
        return [Document(page_content=content, metadata={'source': str(file_path)})]
    
    def _load_yaml(self, file_path: Path) -> List[Document]:
        """Load YAML document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        content = yaml.dump(data, default_flow_style=False)
        return [Document(page_content=content, metadata={'source': str(file_path)})]
    
    def _load_excel(self, file_path: Path) -> List[Document]:
        """Load Excel document"""
        loader = UnstructuredExcelLoader(str(file_path))
        return loader.load()
    
    def _load_image(self, file_path: Path) -> List[Document]:
        """Load image document (requires OCR)"""
        try:
            loader = UnstructuredImageLoader(str(file_path))
            return loader.load()
        except Exception as e:
            logger.warning(f"⚠️ Image OCR failed for {file_path.name}: {e}")
            return []


# ================================================================
# WATCHER FOR AUTOMATIC INGESTION
# ================================================================

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

class DocumentWatcher(FileSystemEventHandler):
    """Watch for new documents and auto-ingest"""
    
    def __init__(self, loader: DynamicDocumentLoader, ingest_func):
        self.loader = loader
        self.ingest_func = ingest_func
        self.processed = set()
    
    def on_created(self, event):
        """Handle new file creation"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            ext = file_path.suffix.lower()
            
            if ext in self.loader.supported_formats:
                print(f"📄 New document detected: {file_path.name}")
                try:
                    docs = self.loader.load_document(str(file_path))
                    if docs:
                        self.ingest_func(docs)
                        print(f"✅ Ingested {len(docs)} chunks from {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to ingest {file_path.name}: {e}")


# ================================================================
# PRODUCTION USAGE
# ================================================================

def setup_dynamic_ingestion():
    """Setup dynamic ingestion for production"""
    
    # 1. Initialize loader
    loader = DynamicDocumentLoader()
    
    # 2. Load all existing documents
    docs = loader.load_directory("./documents", recursive=True)
    print(f"✅ Loaded {len(docs)} documents")
    
    # 3. Set up watcher for new documents
    event_handler = DocumentWatcher(loader, ingest_documents)
    observer = Observer()
    observer.schedule(event_handler, "./documents", recursive=True)
    observer.start()
    
    print("👁️ Watching for new documents...")
    
    return loader, observer

def ingest_documents(docs):
    """Function to ingest documents into the system"""
    # Your existing ingestion logic
    pass

if __name__ == "__main__":
    setup_dynamic_ingestion()